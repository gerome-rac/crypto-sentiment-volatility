#!/usr/bin/env python3
"""tether_check.py -- robustness check of 14 Nov 2026 on the placement of the two Tether news terms.

Question (Gerome, 14 Nov 2026): Tether news volume is a headline count and could be read as attention. In the main
specification it is a control and enters R0. Does that placement drive the attention step (R0 -> R1)?

Arms, both on the main panel and settings (spec A00, rolling 365 days, adaptive LASSO), out-of-sample part only:
  A85  the two Tether terms (log level, daily change) are in no rung;
  A86  the two Tether terms enter with the attention block at R1 and are not in R0.

Inputs:  <runs_dir> with the main run and the two arm runs (names on the command line).
Outputs: <out_dir>/tether_check_results.csv          one row per run and step
         <out_dir>/cum_loss_diff_<arm>.csv           input of fig_cum_loss_diff_20261110.R (one figure per arm)
         <out_dir>/cum_loss_R1_compare.csv           cumulative loss difference R0 - R1 of the three runs, both losses

Checks (asserted):
  - every run has 885 forecast days and 4,425 forecasts per rung, on the same coin-days and targets;
  - last cumulative value / 885 = full-sample loss difference of the run (as in oos_appendix_tables_20261108.py);
  - A85 and A86 have the same R0 (no Tether term in either): forecasts equal;
  - R1, R2 and R3 of A86 hold the same regressors as in the main run: forecasts agree up to solver tolerance (printed).

  - Holm: the arms were run without the in-sample part, so their stored Holm family has three tests. For the
    attention step the adjusted value over the four tests of the thesis is 4 x p, because its p-value is the smallest
    of the family in every run (asserted); it is written as cw_p_holm_4tests.

Also written: Diebold-Mariano test (two-sided, Bartlett, 6 lags, as in the main run) of R0 with the Tether terms
against R0 without them, on both losses (tether_check_r0.csv).

Usage (from the code/ folder): python <this file> <runs_dir> <main_run> <A85_run> <A86_run> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from model.tests_eval import day_mean, dm_hln, loss_frame  # noqa: E402

runs_dir, main_run, a85_run, a86_run, out_dir = sys.argv[1:6]
runs_dir, out_dir = Path(runs_dir), Path(out_dir)
RUNS = {"main": main_run, "A85": a85_run, "A86": a86_run}
LABEL = {"main": "Main specification (Tether terms in R0)",
         "A85": "Tether terms in no rung",
         "A86": "Tether terms enter with attention at R1"}
STEPS = {"H1": "R0 -> R1 (attention)", "H2": "R1 -> R2 (polarity)", "H3": "R2 -> R3 (emotions)"}


def write(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    back = pd.read_csv(tmp)
    assert len(back) == len(df) and list(back.columns) == list(df.columns), path
    tmp.replace(path)


fc, rows, cum = {}, [], {}
for key, run in RUNS.items():
    d = runs_dir / run
    f = pd.read_parquet(d / "forecasts.parquet")
    f = f[f.rung.isin(["R0", "R1", "R2", "R3"])].sort_values(["rung", "coin", "target"]).reset_index(drop=True)
    assert (f.groupby("rung").size() == 4425).all() and f.target.nunique() == 885, key
    fc[key] = f
    ls = pd.read_csv(d / "loss_summary.csv").set_index("model")
    nt = pd.read_csv(d / "nested_tests.csv").set_index("test")
    ch = pd.read_csv(d / "confirmatory_holm.csv").set_index("hypothesis")
    cl = pd.read_parquet(d / "cum_loss_diff.parquet")
    assert len(cl) == 885, key
    for r in ("R1", "R2", "R3"):
        for col, lo in (("dse", "MSE_log"), ("dqlike", "QLIKE")):
            got = cl[f"cum_{col}_R0_minus_{r}"].iloc[-1] / 885
            assert abs(got - (ls.loc["R0", lo] - ls.loc[r, lo])) < 1e-10, (key, r, col)
    cum[key] = cl
    assert nt.loc["H1", "cw_p"] == nt.loc[["H1", "H2", "H3"], "cw_p"].min(), key
    for h, step in STEPS.items():
        small, big = nt.loc[h, "small"], nt.loc[h, "big"]
        rows.append({
            "run": key, "specification": LABEL[key], "step": step,
            "mse_small": ls.loc[small, "MSE_log"], "mse_big": ls.loc[big, "MSE_log"],
            "qlike_small": ls.loc[small, "QLIKE"], "qlike_big": ls.loc[big, "QLIKE"],
            "r2_oos_pct": 100 * nt.loc[h, "r2_oos_big_vs_small"],
            "qlike_gain_pct": 100 * (1 - ls.loc[big, "QLIKE"] / ls.loc[small, "QLIKE"]),
            "cw_stat": nt.loc[h, "cw_stat"], "cw_p": nt.loc[h, "cw_p"],
            "cw_p_holm_run": ch.loc[h, "p_holm"], "holm_family_run": int(ch.loc[h, "holm_family_size"]),
            "cw_p_holm_4tests": min(1.0, 4 * nt.loc[h, "cw_p"]) if h == "H1" else np.nan,
            "dm_mse_p_big_better": nt.loc[h, "dm_mse_p_big_better"],
            "dm_qlike_p_big_better": nt.loc[h, "dm_qlike_p_big_better"],
            "verdict": ch.loc[h, "verdict"],
        })

# Same coin-days and targets in every run.
for key in ("A85", "A86"):
    assert fc[key][["rung", "coin", "target"]].equals(fc["main"][["rung", "coin", "target"]]), key
    assert np.allclose(fc[key].y, fc["main"].y, atol=0, rtol=0), key


def maxdiff(a: str, b: str, rung: str) -> float:
    x = fc[a].loc[fc[a].rung == rung, "f_log"].to_numpy()
    y = fc[b].loc[fc[b].rung == rung, "f_log"].to_numpy()
    return float(np.abs(x - y).max())


assert maxdiff("A85", "A86", "R0") == 0.0, maxdiff("A85", "A86", "R0")
same = {r: maxdiff("A86", "main", r) for r in ("R1", "R2", "R3")}
print("A85 and A86 share R0: forecasts identical")
print("A86 against the main run, largest absolute difference of the log forecast:", {k: f"{v:.2e}" for k, v in same.items()})
print("R0 of the arms against R0 of the main run, largest absolute difference:", f"{maxdiff('A85', 'main', 'R0'):.3f}")

# R0 with the Tether terms (main) against R0 without them (the arms), on the daily cross-coin mean losses.
r0 = []
la = loss_frame(fc["main"][fc["main"].rung == "R0"]); lb = loss_frame(fc["A85"][fc["A85"].rung == "R0"])
for col, name in (("se", "MSE_log"), ("qlike", "QLIKE")):
    a, b = day_mean(la[col], la["target"]), day_mean(lb[col], lb["target"])
    assert len(a) == 885 == len(b)
    t = dm_hln(a, b, h=1, lags=6, variance="hac")
    r0.append({"loss": name, "with_tether": a.mean(), "without_tether": b.mean(), "dm": t["dm_hln"],
               "p_two_sided": t["p_two_sided"]})
r0 = pd.DataFrame(r0)

res = pd.DataFrame(rows)
out_dir.mkdir(parents=True, exist_ok=True)
write(out_dir / "tether_check_results.csv", res)
write(out_dir / "tether_check_r0.csv", r0)

cols = ["target"] + [f"cum_{a}_R0_minus_{r}" for a in ("dse", "dqlike") for r in ("R1", "R2", "R3")]
for key in ("A85", "A86"):
    write(out_dir / f"cum_loss_diff_{key}.csv", cum[key][cols])

cmp_ = pd.DataFrame({"target": cum["main"].target})
for key in RUNS:
    assert (cum[key].target.to_numpy() == cmp_.target.to_numpy()).all()
    cmp_[f"dse_{key}"] = cum[key]["cum_dse_R0_minus_R1"].to_numpy()
    cmp_[f"dqlike_{key}"] = cum[key]["cum_dqlike_R0_minus_R1"].to_numpy()
write(out_dir / "cum_loss_R1_compare.csv", cmp_)

pd.set_option("display.width", 250)
show = res[["run", "step", "mse_small", "mse_big", "r2_oos_pct", "qlike_gain_pct", "cw_p", "cw_p_holm_4tests",
            "dm_mse_p_big_better", "dm_qlike_p_big_better", "verdict"]]
print(show.round(4).to_string(index=False))
print("R0 with against without the Tether terms:"); print(r0.round(5).to_string(index=False))
print("final cumulative difference R0 - R1 (squared error):",
      {k: round(float(cmp_[f"dse_{k}"].iloc[-1]), 3) for k in RUNS})
