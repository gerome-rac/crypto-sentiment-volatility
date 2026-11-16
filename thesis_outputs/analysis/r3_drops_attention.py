#!/usr/bin/env python3
"""r3_drops_attention.py -- in how many of the 885 estimation windows R2 keeps at least one attention term while R3
keeps none, for the main run (adaptive LASSO) and the elastic-net check, with the Clark-West p-value of the emotion
step of each run. Written 14 Nov 2026 for the Discussion (emotions paragraph).
Usage (from the code/ folder): python <this file> <runs_dir> <main_run> <enet_run> <out_csv>
"""
import sys
from pathlib import Path

import pandas as pd

runs, main_run, enet_run, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
rows = []
for name, run in (("adaptive LASSO (main)", main_run), ("adaptive elastic net", enet_run)):
    c2 = pd.read_parquet(runs / run / "coefs_R2.parquet").sort_values("origin")
    c3 = pd.read_parquet(runs / run / "coefs_R3.parquet").sort_values("origin")
    assert len(c2) == 885 == len(c3) and (c2.origin.to_numpy() == c3.origin.to_numpy()).all()
    att = [x for x in c3.columns if x.startswith(("att_", "mkt_"))]
    assert len(att) == 45 and set(att) <= set(c2.columns)
    a2 = (c2[att] != 0).any(axis=1).to_numpy(); a3 = (c3[att] != 0).any(axis=1).to_numpy()
    emo = [x for x in c3.columns if x.startswith("emo_")]
    nt = pd.read_csv(runs / run / "nested_tests.csv").set_index("test")
    rows.append({"run": name, "run_dir": run, "windows": len(c2), "R2_keeps_attention": int(a2.sum()),
                 "R3_keeps_attention": int(a3.sum()), "R2_keeps_R3_drops": int((a2 & ~a3).sum()),
                 "R3_no_emotion_term": int((c3[emo] == 0).all(axis=1).sum()), "cw_p_emotion_step": nt.loc["H3", "cw_p"]})
df = pd.DataFrame(rows); df.to_csv(out, index=False); print(df.round(3).to_string(index=False))
