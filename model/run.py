"""Entry point.

    cd code && python -m model.run --spec A00 --rung all --panel data/pipeline/panel \
        [--placebo-shift-days 365] [--n-jobs 6] [--label placebo]

Engine arms (model/config.py SPEC_ARMS): A00, A01 (per coin; needs --no-insample), A06, A55, A71, and from
28 Oct 2026 A61 (Duan QLIKE), A62 (--eval-target <column> [--extra-table <parquet>]; evaluation target only) and
A53 (drop rule, --drop-days-flag abd_drop_flag). Every new option defaults to A00's behaviour.

Writes to code/data/pipeline/model_runs/<spec>[_<label>]_<UTC timestamp>/:
  config.json, forecasts.parquet, coefs_<rung>.parquet, window_meta_<rung>.parquet,
  loss_summary.csv, nested_tests.csv, mcs.csv, cum_loss_diff.parquet, per_coin.csv,
  confirmatory_holm.csv, insample_coef_<rung>.csv, insample_wald.csv, insample_fit.csv,
  h4_bootstrap.csv, h4_coef_ci.csv, insample_lasso_<rung>.csv, selection_freq_*.csv,
  runtime.json, MD5SUMS.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from .config import LADDER, NESTED_COMPARISONS, RUNG_BLOCKS, EngineConfig, rung_blocks_for
from .har import FeatureBuilder
from .inference import (h4_applicability, h4_not_applicable_table, h4_test, insample_lasso, ols_dk_nw,
                        selection_frequencies)
from .oos import run_oos
from .panel import build_grid, load_tables, load_variable_spec, placebo_shift
from .tests_eval import confirmatory, evaluate

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pipeline", "model_runs")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    if path.endswith(".parquet"):
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def specs_for_rungs(specs, rungs, cfg=None):
    rb = rung_blocks_for(cfg) if cfg is not None else RUNG_BLOCKS
    blocks = set(b for r in rungs for b in rb[r]) | {"har_rv"}
    return [v for v in specs if v.block in blocks]


def run(args) -> str:
    cfg = EngineConfig.for_spec(args.spec, window=args.window, n_jobs=args.n_jobs,
                                placebo_shift_days=args.placebo_shift_days, label=args.label,
                                estimator=args.estimator, qlike_backtransform=args.qlike_backtransform,
                                mcs_reps=args.mcs_reps, boot_reps=args.boot_reps,
                                drop_days_flag=getattr(args, "drop_days_flag", None),
                                eval_target=getattr(args, "eval_target", None))
    if args.config_json:
        over = json.load(open(args.config_json))
        for k, v in over.items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown config field {k}")
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    rungs = list(LADDER) if args.rung == "all" else args.rung.split(",")
    if args.with_harrv or args.rung == "all":
        rungs = rungs + ["HARRV"]
    if args.rung == "all" and tuple(rung_blocks_for(cfg)["R0"]) != tuple(RUNG_BLOCKS["HARCJ"]):
        rungs = rungs + ["HARCJ"]          # pure HAR-CJ reference (Deviation 3), outside the MCS
    if cfg.unit == "per_coin" and not args.no_insample:
        # 28 Oct 2026: the in-sample stage (Stage 3) is the pooled panel regression for every unit; under a per-coin arm
        # (A01) it would write pooled tables under the arm's name. Per-coin in-sample estimation is not defined.
        raise SystemExit("unit 'per_coin' (arm A01): the in-sample stage is pooled only; run with --no-insample")
    specs = specs_for_rungs(load_variable_spec(args.variable_spec), rungs, cfg)
    tables = load_tables(args.panel, extra=getattr(args, "extra_table", None) or ())
    extra_cols = (cfg.drop_days_flag,) if cfg.drop_days_flag else ()
    value_cols = (cfg.eval_target,) if cfg.eval_target != "rv" else ()
    grid = build_grid(tables, specs, extra_cols=extra_cols, value_cols=value_cols)
    grid = placebo_shift(grid, specs, cfg.placebo_shift_days, cfg.placebo_mode, cfg.placebo_blocks)
    builder = FeatureBuilder(grid, specs, cfg)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{cfg.spec}{'_' + cfg.label if cfg.label else ''}_{stamp}"
    out = os.path.join(args.out_root or DEFAULT_OUT, name)
    os.makedirs(out, exist_ok=False)
    with open(os.path.join(out, "config.json"), "w") as fh:
        fh.write(cfg.to_json())
    runtime = {}
    fcs = {}
    for r in rungs:
        fc, coefs, meta, wall, blocks = run_oos(builder, cfg, r, max_days=args.max_days)
        fcs[r] = fc
        runtime[f"oos_{r}_seconds"] = wall
        _write(coefs, os.path.join(out, f"coefs_{r}.parquet"))
        _write(meta, os.path.join(out, f"window_meta_{r}.parquet"))
        if r in ("R1", "R2", "R3"):
            names = [c for c in coefs.columns if c not in ("origin", "group")]
            base = [_base_of(n, builder.specs) for n in names]
            sf = selection_frequencies(coefs, names, blocks, base)
            for k, t in sf.items():
                t.insert(0, "rung", r)
                _write(t, os.path.join(out, f"selection_freq_{k}_{r}.csv"))
        print(f"[{r}] {len(fc)} forecasts, {wall:.1f} s", flush=True)
    _write(pd.concat(fcs.values(), ignore_index=True), os.path.join(out, "forecasts.parquet"))
    ev = evaluate(fcs, cfg, NESTED_COMPARISONS, bench="R0" if "R0" in fcs else rungs[0])
    for k, t in ev.items():
        _write(t, os.path.join(out, f"{k}.{'parquet' if k == 'cum_loss_diff' else 'csv'}"))
    p_h4 = None
    h4_status, h4_reason = "not computed", ""
    if not args.no_insample:
        t0 = time.perf_counter()
        coefs_is, walds, fits = [], [], []
        for r in [x for x in rungs if x in LADDER]:
            ins = ols_dk_nw(builder, r, cfg)
            coefs_is.append(ins["coef"]); walds.append(ins["wald"]); fits.append(dict(rung=r, **ins["fit"]))
            if cfg.wald_bootstrap_reps and r != "R0":
                from .inference import wald_bootstrap
                walds.append(wald_bootstrap(ins, builder, cfg, cfg.wald_bootstrap_reps))
            if r == "R3":
                # 28 Oct 2026: H4 needs emo_fear and emo_anger at 1/7/30 in R3; an arm without them (e.g. A60)
                # gets a 'not applicable' row instead of stopping, the confirmatory A00 run never does.
                h4_ok, h4_reason = h4_applicability(ins["names"])
                if h4_ok:
                    h4 = h4_test(ins, builder, cfg)
                    _write(h4["table"], os.path.join(out, "h4_bootstrap.csv"))
                    _write(h4["coef_ci"], os.path.join(out, "h4_coef_ci.csv"))
                    p_h4 = h4["p_h4"]
                    h4_status = "tested"
                else:
                    if cfg.spec == "A00" and not cfg.label:
                        raise ValueError(f"A00 is the confirmatory arm and must test H4, but {h4_reason}")
                    _write(h4_not_applicable_table(h4_reason, cfg), os.path.join(out, "h4_bootstrap.csv"))
                    h4_status = "not applicable"
            if r != "R0" and args.insample_lasso:
                _write(insample_lasso(builder, r, cfg, args.lasso_boot_reps), os.path.join(out, f"insample_lasso_{r}.csv"))
        _write(pd.concat(coefs_is, ignore_index=True), os.path.join(out, "insample_coef.csv"))
        _write(pd.concat(walds, ignore_index=True) if any(len(w) for w in walds) else pd.DataFrame(), os.path.join(out, "insample_wald.csv"))
        _write(pd.DataFrame(fits), os.path.join(out, "insample_fit.csv"))
        runtime["insample_seconds"] = time.perf_counter() - t0
    if "nested_tests" in ev and len(ev["nested_tests"]):
        _write(confirmatory(ev["nested_tests"], ev["mcs"], p_h4, cfg.holm_alpha, ev.get("mcs_h2"),
                            h4_status=h4_status, h4_reason=h4_reason), os.path.join(out, "confirmatory_holm.csv"))
    with open(os.path.join(out, "runtime.json"), "w") as fh:
        json.dump(runtime, fh, indent=2)
    files = sorted(f for f in os.listdir(out) if f != "MD5SUMS")
    with open(os.path.join(out, "MD5SUMS"), "w") as fh:
        for f in files:
            fh.write(f"{md5(os.path.join(out, f))}  {f}\n")
    return out


def _base_of(col: str, specs) -> str:
    for suf in ("_sq_h1", "_sq_h7", "_sq_h30", "_h1", "_h7", "_h30", "_dlog"):
        if col.endswith(suf) and col[: -len(suf)] in specs:
            return col[: -len(suf)]
    return col


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m model.run")
    ap.add_argument("--spec", default="A00")
    ap.add_argument("--rung", default="all", help="all | R0,R1,... | HARRV")
    ap.add_argument("--panel", required=True, help="directory with the four *_daily.parquet tables, or one merged parquet")
    ap.add_argument("--variable-spec", default=None)
    ap.add_argument("--placebo-shift-days", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--window", default=None, help="override: integer days or 'expanding'")
    ap.add_argument("--estimator", default=None, choices=[None, "adaptive_lasso", "enet", "ols"])
    ap.add_argument("--qlike-backtransform", default=None, choices=[None, "lognormal", "duan"])
    ap.add_argument("--drop-days-flag", default=None,
                    help="boolean panel column; flagged coin-days are dropped (arm A53: abd_drop_flag)")
    ap.add_argument("--eval-target", default=None,
                    help="column whose next-day value is the evaluation target (arm A62); default rv")
    ap.add_argument("--extra-table", action="append", default=None,
                    help="further coin-day parquet adding columns (e.g. the A62 subsampled RV); repeatable")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--mcs-reps", type=int, default=None)
    ap.add_argument("--boot-reps", type=int, default=None)
    ap.add_argument("--with-harrv", action="store_true")
    ap.add_argument("--no-insample", action="store_true")
    ap.add_argument("--insample-lasso", action="store_true", default=True)
    ap.add_argument("--lasso-boot-reps", type=int, default=199)
    ap.add_argument("--config-json", default=None, help="JSON file with EngineConfig overrides")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--max-days", type=int, default=None, help="development only: first N forecast days")
    a = ap.parse_args(argv)
    if a.window not in (None, "expanding"):
        a.window = int(a.window)
    out = run(a)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
