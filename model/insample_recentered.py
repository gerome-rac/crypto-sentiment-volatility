"""In-sample adaptive LASSO with recentered-bootstrap p-values (Audrino & Camponovo 2018), as in
Audrino, Sigrist & Ballinari (2020, pp. 345-346). Post-result addition of 7 Nov 2026 (D8 deviation log).

    cd code && python -m model.insample_recentered --spec A00 --panel <panel parquet> \
        --variable-spec model/variable_spec_A00.csv --rungs R0,R1,R2,R3 --reps 1999 --n-jobs 6 \
        [--compare-run <run dir with insample_lasso_<rung>.csv>]

Builds the panel exactly as model.run does, re-estimates each rung's in-sample adaptive LASSO (the original
fit must reproduce the compared run's coefficients and lambda), and writes to
code/data/pipeline/model_runs/<spec>_insample_recentered_<UTC timestamp>/:
  config.json, insample_lasso_recentered_<rung>.csv, insample_lasso_recentered_summary.csv, runtime.json, MD5SUMS.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from .config import EngineConfig
from .har import FeatureBuilder
from .inference import insample_lasso_recentered
from .panel import build_grid, load_tables, load_variable_spec, placebo_shift
from .run import DEFAULT_OUT, _write, md5, specs_for_rungs


def summary(df: pd.DataFrame) -> pd.DataFrame:
    pen = df[df.penalised]
    g = pen.groupby(["rung", "block"])
    return pd.DataFrame(dict(
        n_variables=g.size(), n_selected=g.selected.sum(),
        n_p_lt_01=g.p_value.apply(lambda s: int((s < 0.01).sum())),
        n_p_lt_05=g.p_value.apply(lambda s: int((s < 0.05).sum())),
        n_p_lt_10=g.p_value.apply(lambda s: int((s < 0.10).sum())),
        n_holm_lt_05=g.p_holm_block.apply(lambda s: int((s < 0.05).sum())))).reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m model.insample_recentered")
    ap.add_argument("--spec", default="A00")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--variable-spec", required=True)
    ap.add_argument("--rungs", default="R0,R1,R2,R3")
    ap.add_argument("--reps", type=int, default=1999)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--block", type=float, default=None, help="mean block length in days; default automatic")
    ap.add_argument("--label", default="insample_recentered")
    ap.add_argument("--compare-run", default=None)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args(argv)

    cfg = EngineConfig.for_spec(a.spec, window=None, n_jobs=a.n_jobs, placebo_shift_days=0, label=a.label,
                                estimator=None, qlike_backtransform=None, mcs_reps=None, boot_reps=None)
    rungs = a.rungs.split(",")
    specs = specs_for_rungs(load_variable_spec(a.variable_spec), rungs, cfg)
    grid = build_grid(load_tables(a.panel), specs)
    grid = placebo_shift(grid, specs, 0, cfg.placebo_mode, cfg.placebo_blocks)
    builder = FeatureBuilder(grid, specs, cfg)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(a.out_root or DEFAULT_OUT, f"{cfg.spec}_{a.label}_{stamp}")
    os.makedirs(out, exist_ok=False)
    conf = json.loads(cfg.to_json())
    conf.update(method="recentered block bootstrap, Audrino & Camponovo (2018) eq. 4 and Remark 4",
                reps=a.reps, seed=(cfg.seed + 2 if a.seed is None else a.seed), rungs=rungs,
                block=("automatic (Politis & White 2004 on day-mean residuals)" if a.block is None else a.block),
                panel_md5=md5(a.panel) if os.path.isfile(a.panel) else None, variable_spec_md5=md5(a.variable_spec),
                compare_run=os.path.basename(os.path.normpath(a.compare_run)) if a.compare_run else None)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(conf, fh, indent=1, default=str)

    runtime, frames = {}, []
    for r in rungs:
        t0 = time.perf_counter()
        df = insample_lasso_recentered(builder, r, cfg, reps=a.reps, seed=a.seed, n_jobs=a.n_jobs, block=a.block)
        runtime[r] = time.perf_counter() - t0
        if a.compare_run:
            ref_path = os.path.join(a.compare_run, f"insample_lasso_{r}.csv")
            if os.path.exists(ref_path):
                ref = pd.read_csv(ref_path)
                assert list(ref.variable) == list(df.variable), f"{r}: column order differs from {ref_path}"
                assert np.allclose(ref.beta_z.to_numpy(), df.beta_z.to_numpy(), atol=1e-8, rtol=0), f"{r}: beta_z differs"
                assert np.isclose(ref["lambda"].iloc[0], df["lambda"].iloc[0], rtol=1e-10, atol=0), f"{r}: lambda differs"
                runtime[f"{r}_reproduces_compare_run"] = True
        _write(df, os.path.join(out, f"insample_lasso_recentered_{r}.csv"))
        frames.append(df)
        print(f"{r}: {runtime[r]:.1f}s, block {df.boot_block_length.iloc[0]:.2f} days", flush=True)
    _write(summary(pd.concat(frames, ignore_index=True)), os.path.join(out, "insample_lasso_recentered_summary.csv"))
    with open(os.path.join(out, "runtime.json"), "w") as fh:
        json.dump(runtime, fh, indent=1)
    with open(os.path.join(out, "MD5SUMS"), "w") as fh:
        for f in sorted(os.listdir(out)):
            if f != "MD5SUMS":
                fh.write(f"{md5(os.path.join(out, f))}  {f}\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
