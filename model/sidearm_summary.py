"""Summary of the side-arm runs of 12 Nov 2026 (aggregate emotion dispersion; valenced emotions).

Reads only stored run outputs. Per arm:
  * the run's own step R2 -> R3 (loss_summary.csv, nested_tests.csv): change of the MSE of log RV and of QLIKE, Clark-West p;
  * for the arms that add terms to the primary's emotion block: the direct nested comparison primary R3 -> arm R3
    (Clark-West with the engine's function, day-averaged adjusted differentials, the run's HAC lags);
  * the selection frequency of each new variable across the estimation windows (selection_freq_variables_R3.csv);
  * the in-sample OLS coefficients of the new terms with Driscoll-Kraay p-values (insample_coef.csv), where estimated.
Asserted: every run ended with exit 0 and 4,425 forecasts per rung on the primary's coin-days; the engine settings
equal the primary's; the forecasts of R0, R1 and R2 equal the primary's (only the emotion block changed); the
Clark-West function called here reproduces the run's stored R2 -> R3 statistic.

    cd code && python -m model.sidearm_summary --queue data/pipeline/model_runs/_queue_sidearms_20261112.csv \
        --primary A00_final_A50_20261028T134243Z --out data/pipeline/model_runs/_sidearm_results_20261112.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from .tests_eval import clark_west, forecasts_val, loss_frame

RUNS = os.path.join(os.path.dirname(__file__), "..", "data", "pipeline", "model_runs")
SKIP_CFG = {"label", "n_jobs"}


def load_fc(run: str, bt: str) -> dict:
    f = pd.read_parquet(os.path.join(RUNS, run, "forecasts.parquet"))
    return {r: loss_frame(g, bt).sort_values(["target", "coin"]).reset_index(drop=True) for r, g in f.groupby("rung")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    q = pd.read_csv(a.queue, keep_default_na=False)
    cfg0 = json.load(open(os.path.join(RUNS, a.primary, "config.json")))
    bt = cfg0["qlike_backtransform"]
    P = load_fc(a.primary, bt)
    spec0 = set(pd.read_csv(os.path.join(os.path.dirname(__file__), "variable_spec_A50.csv")).name)
    ls0 = pd.read_csv(os.path.join(RUNS, a.primary, "loss_summary.csv")).set_index("model")
    rows, coefs = [], []
    for r in q.itertuples():
        dirs = sorted(glob.glob(os.path.join(RUNS, f"A00_{r.label}_2*")))
        assert len(dirs) == 1, f"{r.run_id}: {len(dirs)} run directories"
        run = os.path.basename(dirs[0])
        log = open(os.path.join(RUNS, "_logs_sidearms_20261112", f"{r.run_id}.log")).read()
        assert log.rstrip().endswith("exit=0"), f"{r.run_id}: run did not end with exit 0"
        cfg = json.load(open(os.path.join(dirs[0], "config.json")))
        diff = {k for k in set(cfg) | set(cfg0) if k not in SKIP_CFG and cfg.get(k) != cfg0.get(k)}
        assert not diff, f"{r.run_id}: engine settings differ from the primary: {diff}"
        F = load_fc(run, bt)
        dmax = 0.0
        for rung in ("R0", "R1", "R2", "R3"):
            assert len(F[rung]) == 4425 and F[rung][["coin", "target"]].equals(P[rung][["coin", "target"]]), (r.run_id, rung)
            if rung != "R3":
                dmax = max(dmax, float(np.abs(F[rung].f_log - P[rung].f_log).max()))
        assert dmax < 1e-10, f"{r.run_id}: R0-R2 forecasts differ from the primary (max {dmax:.2e})"
        ls = pd.read_csv(os.path.join(dirs[0], "loss_summary.csv")).set_index("model")
        nt = pd.read_csv(os.path.join(dirs[0], "nested_tests.csv")).set_index("test")
        tgt = F["R3"]["target"]
        own = clark_west(F["R3"].y, forecasts_val(F["R2"]), forecasts_val(F["R3"]), groups=tgt)
        assert abs(own["cw_stat"] - nt.loc["H3", "cw_stat"]) < 1e-9 and own["lags"] == nt.loc["H3", "hac_lags"], r.run_id
        sp = pd.read_csv(os.path.join(os.path.dirname(__file__), f"variable_spec_{r.run_id}.csv"), keep_default_na=False)
        new = [n for n in sp.name if n not in spec0]
        nested = not r.run_id.endswith("_alone")
        row = {"arm": r.run_id, "run_dir": run, "new_variables": " ".join(new), "emotion_terms": int(r.emotion_terms),
               "placement": "added to the primary's emotion block" if nested else "alone on R2",
               "mse_R2": ls.loc["R2", "MSE_log"], "mse_R3": ls.loc["R3", "MSE_log"],
               "d_mse_R2_R3_pct": 100 * (ls.loc["R3", "MSE_log"] / ls.loc["R2", "MSE_log"] - 1),
               "d_qlike_R2_R3_pct": 100 * (ls.loc["R3", "QLIKE"] / ls.loc["R2", "QLIKE"] - 1),
               "cw_stat_R2_R3": nt.loc["H3", "cw_stat"], "cw_p_R2_R3": nt.loc["H3", "cw_p"],
               "d_mse_vs_primary_R3_pct": 100 * (ls.loc["R3", "MSE_log"] / ls0.loc["R3", "MSE_log"] - 1),
               "max_abs_diff_R0_R2_vs_primary": dmax}
        if nested:
            d = clark_west(F["R3"].y, forecasts_val(P["R3"]), forecasts_val(F["R3"]), groups=tgt)
            row.update(cw_stat_primaryR3_armR3=d["cw_stat"], cw_p_primaryR3_armR3=d["p_one_sided"])
        sf = pd.read_csv(os.path.join(dirs[0], "selection_freq_variables_R3.csv")).set_index("variable")["freq"]
        row["selection_freq_new"] = " ".join(f"{n}={sf[n]:.3f}" for n in new)
        row["selection_freq_new_max"] = float(max(sf[n] for n in new))
        ic = os.path.join(dirs[0], "insample_coef.csv")
        if os.path.exists(ic):
            c = pd.read_csv(ic)
            c = c[(c.rung == "R3") & c.variable.str.rsplit("_", n=1).str[0].isin(new)]
            for x in c.itertuples():
                coefs.append({"arm": r.run_id, "term": x.variable, "beta_z": x.beta_z, "se_dk": x.se_dk, "t_dk": x.t_dk, "p_dk": x.p_dk})
        rows.append(row)
    out = pd.DataFrame(rows)
    tmp = a.out + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, a.out)
    cf = a.out.replace(".csv", "_insample_coefs.csv")
    pd.DataFrame(coefs).to_csv(cf + ".tmp", index=False)
    os.replace(cf + ".tmp", cf)
    pr = {"primary": a.primary, "primary_d_mse_R2_R3_pct": 100 * (ls0.loc["R3", "MSE_log"] / ls0.loc["R2", "MSE_log"] - 1)}
    print(json.dumps(pr))
    show = ["arm", "emotion_terms", "d_mse_R2_R3_pct", "cw_p_R2_R3", "d_mse_vs_primary_R3_pct", "cw_p_primaryR3_armR3", "selection_freq_new_max"]
    print(out[[c for c in show if c in out.columns]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
