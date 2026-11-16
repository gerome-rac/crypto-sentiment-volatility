"""Added 7 Oct 2026 (lead): H2 rule as amended by Deviation 4 of the D8 note."""
import numpy as np
import pandas as pd
from model.tests_eval import confirmatory


def _nested(dm_mse_p2, dm_mse_big, dm_q_p2, dm_q_big, cw_h2=0.6):
    rows = []
    for t, p in (("H1", 0.001), ("H2", cw_h2), ("H3", 0.2)):
        rows.append(dict(test=t, cw_p=p, dm_mse=-1.0, dm_mse_p2=dm_mse_p2, dm_mse_p_big_better=dm_mse_big,
                         dm_qlike=-1.0, dm_qlike_p2=dm_q_p2, dm_qlike_p_big_better=dm_q_big))
    return pd.DataFrame(rows)


def _mcs(models_in, models):
    rows = []
    for lo in ("MSE_log", "QLIKE"):
        for m in models:
            rows.append(dict(loss=lo, model=m, mcs_pvalue=0.5 if m in models_in else 0.01, in_mcs=m in models_in, alpha=0.10))
    return pd.DataFrame(rows)


def test_h2_supported_when_r3_dominates_full_set():
    full = _mcs({"R3"}, ["R0", "R1", "R2", "R3", "HARRV"])
    pair = _mcs({"R1", "R2"}, ["R1", "R2"])
    out = confirmatory(_nested(0.4, 0.8, 0.4, 0.8), full, 0.5, 0.05, pair).set_index("hypothesis")
    assert out.loc["H2", "verdict"] == "supported" and not out.loc["H2", "R1_in_MCS10_full_set_both_losses"]


def test_h2_not_blocked_when_r2_significantly_worse():
    pair = _mcs({"R1"}, ["R1", "R2"])                       # R2 dropped, R1 kept
    out = confirmatory(_nested(0.01, 0.995, 0.02, 0.99), pair, 0.5, 0.05, pair).set_index("hypothesis")
    assert out.loc["H2", "verdict"] == "supported" and not out.loc["H2", "DM_two_sided_insignificant_both"]


def test_h2_rejected_when_r2_better():
    pair = _mcs({"R2"}, ["R1", "R2"])
    out = confirmatory(_nested(0.01, 0.005, 0.03, 0.015, cw_h2=0.001), pair, 0.5, 0.05, pair).set_index("hypothesis")
    assert out.loc["H2", "verdict"] == "not supported"
