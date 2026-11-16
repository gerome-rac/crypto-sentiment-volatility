#!/usr/bin/env python3
"""emotion_robustness.py -- the emotion step (R2 -> R3) across the 45 robustness runs (15 Nov 2026, descriptive).

emotion_step_runs.csv   one row per run: unadjusted Clark-West p of the step, the run's own Holm p, MSE of R1, R2
                        and R3, change of the MSE by the step, R2 and R3 against R2 of the main run, gain of R2 over
                        R1 and of R3 over R2 on 7 September 2021, Clark-West p without that day, number of windows
                        with an emotion term.
Checks (asserted): 45 runs; three unadjusted p-values below 0.05; the main run reproduces p = 0.873.
Usage (from the code/ folder): python <this file> <runs_dir> <arm_results.csv> <main_run> <out_dir>
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from model.tests_eval import clark_west  # noqa: E402

runs_dir, arm_csv, main_run, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
t = pd.read_csv(arm_csv); assert len(t) == 45
day = pd.Timestamp("2021-09-07")


def one(run):
    f = pd.read_parquet(runs_dir / run / "forecasts.parquet"); f["target"] = pd.to_datetime(f["target"])
    w = f[f.rung.isin(["R1", "R2", "R3"])].pivot_table(index=["target", "coin"], columns="rung", values="f_log")
    w["y"] = f[f.rung == "R2"].set_index(["target", "coin"]).y
    tg = w.index.get_level_values("target")
    g32 = ((w.y - w.R2) ** 2 - (w.y - w.R3) ** 2).groupby("target").mean()
    g21 = ((w.y - w.R1) ** 2 - (w.y - w.R2) ** 2).groupby("target").mean()
    cw = lambda d: clark_west(d.y.to_numpy(), d.R2.to_numpy(), d.R3.to_numpy(), groups=d.index.get_level_values("target").to_numpy(), lags=6)["p_one_sided"]
    c3 = pd.read_parquet(runs_dir / run / "coefs_R3.parquet")
    emo = [x for x in c3.columns if x.startswith("emo_")]
    mse = {r: float(((w.y - w[r]) ** 2).mean()) for r in ("R1", "R2", "R3")}
    return dict(n_days=int(tg.nunique()), mse_R1=mse["R1"], mse_R2=mse["R2"], mse_R3=mse["R3"], cw_p=cw(w), cw_p_without_7sep=cw(w[tg != day]),
                gain_R2_over_R1_7sep=float(g21.get(day, np.nan)), gain_R3_over_R2_7sep=float(g32.get(day, np.nan)),
                windows_with_emotion=int((c3[emo] != 0).any(axis=1).sum()) if emo else 0)


m = one(main_run); assert abs(m["cw_p"] - 0.873) < 5e-4
rows = [dict(run_id="main", arm_id="main", run_dir=main_run, H3_p_holm_own=np.nan, **m)]
for _, r in t.iterrows():
    if not os.path.exists(runs_dir / r.run_dir / "coefs_R3.parquet"): continue
    o = one(r.run_dir); assert abs(o["cw_p"] - r.H3_cw_p) < 1e-6 or o["n_days"] != 885, r.run_dir
    rows.append(dict(run_id=r.run_id, arm_id=r.arm_id, run_dir=r.run_dir, H3_p_holm_own=r.H3_p_holm_own, **o))
x = pd.DataFrame(rows)
x["step_mse_change_pct"] = 100 * (x.mse_R3 / x.mse_R2 - 1)
x["R2_vs_main_R2_pct"] = 100 * (x.mse_R2 / m["mse_R2"] - 1); x["R3_vs_main_R2_pct"] = 100 * (x.mse_R3 / m["mse_R2"] - 1)
a = x[x.run_id != "main"]; assert len(a) == 45 and int((a.cw_p < 0.05).sum()) == 3
out.mkdir(parents=True, exist_ok=True); x.sort_values("cw_p").to_csv(out / "emotion_step_runs.csv", index=False)
pd.set_option("display.width", 260)
print(x.sort_values("cw_p").head(9)[["run_id", "run_dir", "cw_p", "H3_p_holm_own", "step_mse_change_pct", "mse_R2", "mse_R3", "R2_vs_main_R2_pct", "R3_vs_main_R2_pct", "gain_R2_over_R1_7sep", "cw_p_without_7sep", "windows_with_emotion"]].round(4).to_string(index=False))
print("runs: p > 0.5:", int((a.cw_p > 0.5).sum()), "| p < 0.10:", int((a.cw_p < 0.10).sum()), "| p < 0.05:", int((a.cw_p < 0.05).sum()), "| median p:", round(a.cw_p.median(), 3),
      "| runs with R3 below the main R2:", int((a.mse_R3 < m["mse_R2"]).sum()), "| without 7 Sep p < 0.05:", int((a.cw_p_without_7sep < 0.05).sum()))
