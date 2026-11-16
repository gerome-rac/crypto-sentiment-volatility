"""Confirmatory family with the two-sided SQ4 test (D8 Deviation 7), hand-made inputs only."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from model.confirmatory_sq4 import holm, one_run


def test_holm_matches_statsmodels():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.uniform(0, 0.3, 4)
        assert np.allclose(holm(p), multipletests(p, method="holm")[1], atol=1e-12)


def _write_run(tmp_path, p, theta2, se2, qlike_against=(False, None, False), h2_rest=(True, True)):
    d = tmp_path / "run"
    d.mkdir()
    ph = holm(np.asarray(p))
    pd.DataFrame(dict(hypothesis=["H1", "H2", "H3", "H4"], p=p, p_holm=ph,
                      qlike_points_against=[qlike_against[0], np.nan, qlike_against[2], np.nan],
                      verdict=["x"] * 4,
                      R1_in_MCS10_of_R1R2_both_losses=[np.nan, h2_rest[0], np.nan, np.nan],
                      DM_no_improvement_of_R2_both=[np.nan, h2_rest[1], np.nan, np.nan])).to_csv(d / "confirmatory_holm.csv", index=False)
    pd.DataFrame(dict(condition=["theta1 = beta_fear > 0", "theta2 = beta_fear - beta_anger > 0"],
                      estimate=[5.0, theta2], boot_p=[0.1, 0.9], dk_se=[2.0, se2], dk_p=[0.1, 0.9],
                      units=["raw", "raw"], block_length=[3.0, 3.0], reps=[1999, 1999])).to_csv(d / "h4_bootstrap.csv", index=False)
    return str(d)


def test_two_sided_sq4_and_verdicts(tmp_path):
    run = _write_run(tmp_path, p=[0.0004, 0.011, 0.87, 0.97], theta2=-30.0, se2=16.0)
    out = one_run(run).set_index("subquestion")
    z = -30.0 / 16.0
    assert np.isclose(out.loc["SQ4", "p"], 2 * norm.sf(abs(z)))
    assert np.isclose(out.loc["SQ4", "cum_anger"], 5.0 + 30.0)
    assert np.allclose(out.p_holm.to_numpy(), holm(out.p.to_numpy()))
    assert out.loc["SQ1", "verdict"] == "supported"
    assert out.loc["SQ2", "verdict"] == "not supported"        # CW significant after Holm
    assert out.loc["SQ3", "verdict"] == "not supported"
    assert out.loc["SQ4", "verdict"] == "no significant difference"


def test_sq2_needs_all_three_conditions(tmp_path):
    run = _write_run(tmp_path, p=[0.0004, 0.30, 0.40, 0.90], theta2=-1.0, se2=16.0, h2_rest=(True, False))
    out = one_run(run).set_index("subquestion")
    assert out.loc["SQ2", "verdict"] == "not supported"        # DM condition fails
