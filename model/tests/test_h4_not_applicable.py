"""Added 28 Oct 2026 (for arm A60): H4 is 'not applicable' when R3 holds no
fear/anger terms; the Holm family never shrinks silently; the confirmatory A00 run must test H4. Synthetic data only."""
import json

import numpy as np
import pandas as pd
import pytest

from model import panel, tests_eval as te
from model import run as runmod
from model.inference import H4_TERMS, h4_applicability, h4_not_applicable_table
from model.config import EngineConfig

SIX = [t for v in ("emo_fear", "emo_anger") for t in H4_TERMS[v]]


def test_h4_applicability_rules():
    other = ["c_h1", "emo_happy_h1", "emo_certainty_h7"]
    assert h4_applicability(other + SIX) == (True, "")
    ok, why = h4_applicability(other)
    assert not ok and "emo_fear" in why and "emo_anger" in why
    ok, why = h4_applicability(other + list(H4_TERMS["emo_fear"]))
    assert not ok and "emo_anger" in why and "emo_fear_h1" not in why
    with pytest.raises(ValueError, match="only some"):
        h4_applicability(other + ["emo_fear_h1", "emo_fear_h7"] + list(H4_TERMS["emo_anger"]))
    t = h4_not_applicable_table(why, EngineConfig())
    assert len(t) == 1 and t.condition.iloc[0].startswith("H4 not applicable") and np.isnan(t.boot_p.iloc[0])
    assert list(t.columns) == ["condition", "estimate", "boot_p", "dk_se", "dk_p", "units", "block_length", "reps"]


def _nested():
    return pd.DataFrame([dict(test=t, cw_p=p, dm_mse=-1.0, dm_mse_p2=0.5, dm_mse_p_big_better=0.6, dm_qlike=-1.0,
                              dm_qlike_p2=0.5, dm_qlike_p_big_better=0.6) for t, p in (("H1", 0.001), ("H2", 0.40),
                                                                                       ("H3", 0.30))])


def _mcs():
    return pd.DataFrame(dict(loss=["MSE_log", "QLIKE"], model=["R1", "R1"], in_mcs=[True, True], alpha=[0.10, 0.10]))


def test_confirmatory_family_with_and_without_h4():
    c4 = te.confirmatory(_nested(), _mcs(), 0.02, 0.05, _mcs()).set_index("hypothesis")
    assert (c4.holm_family_size == 4).all()                                         # A00: four tests
    na = te.confirmatory(_nested(), _mcs(), None, 0.05, _mcs(), h4_status="not applicable",
                         h4_reason="R3 holds no emo_fear_h1/h7/h30 terms").set_index("hypothesis")
    h3 = te.holm([0.001, 0.40, 0.30], 0.05)["p_holm"]
    assert (na.holm_family_size == 3).all() and np.allclose(na.loc[["H1", "H2", "H3"], "p_holm"], h3)
    assert na.loc["H4", "verdict"].startswith("not applicable (R3 holds no emo_fear") and np.isnan(na.loc["H4", "p"])
    assert na.loc["H4", "supported_mse"] is None or pd.isna(na.loc["H4", "supported_mse"])
    nc = te.confirmatory(_nested(), _mcs(), None, 0.05, _mcs()).set_index("hypothesis")   # no in-sample stage
    assert nc.loc["H4", "verdict"].startswith("not computed") and (nc.holm_family_size == 3).all()


def test_tested_h4_without_p_value_is_refused():
    for p in (None, np.nan):
        with pytest.raises(ValueError, match="shrink"):
            te.confirmatory(_nested(), _mcs(), p, 0.05, _mcs(), h4_status="tested")
    with pytest.raises(ValueError, match="shrink"):
        te.confirmatory(_nested(), _mcs(), np.nan, 0.05, _mcs())


def test_run_without_fear_anger_writes_not_applicable(tmp_path, tables):
    pdir = tmp_path / "panel"
    pdir.mkdir()
    for k, t in tables.items():
        t.to_parquet(pdir / f"{k}.parquet", index=False)
    spec = pd.read_csv(panel.SPEC_PATH, dtype=str, keep_default_na=False)
    spec = spec[~spec.name.isin(["emo_fear", "emo_anger", "emo_fear_sd", "emo_anger_sd"])]
    sp = tmp_path / "spec_noH4.csv"
    spec.to_csv(sp, index=False)
    # machinery test: OLS (with pure-noise regressors a LASSO keeps nothing, the rungs coincide and arch's MCS
    # divides by a zero variance), no bootstrap Wald, MCS block length 1 for the 3-day loss series
    cj = tmp_path / "cfg.json"
    cj.write_text(json.dumps({"mcs_block_size": 1, "wald_bootstrap_reps": 0}))
    runs = tmp_path / "runs"
    base = ["--spec", "A00", "--panel", str(pdir), "--variable-spec", str(sp), "--max-days", "3", "--mcs-reps", "200",
            "--estimator", "ols", "--lasso-boot-reps", "0", "--n-jobs", "1", "--config-json", str(cj),
            "--out-root", str(runs)]
    assert runmod.main(base + ["--rung", "all", "--label", "noH4"]) == 0
    (out,) = list(runs.glob("A00_noH4_*"))
    h4 = pd.read_csv(out / "h4_bootstrap.csv")
    assert len(h4) == 1 and h4.condition.iloc[0].startswith("H4 not applicable")
    assert not (out / "h4_coef_ci.csv").exists()
    cf = pd.read_csv(out / "confirmatory_holm.csv").set_index("hypothesis")
    assert cf.loc["H4", "verdict"].startswith("not applicable") and (cf.holm_family_size == 3).all()
    assert np.isfinite(cf.loc[["H1", "H2", "H3"], "p_holm"]).all()
    with pytest.raises(ValueError, match="confirmatory arm"):            # the unlabelled A00 run must test H4
        runmod.main(base + ["--rung", "R2,R3"])
