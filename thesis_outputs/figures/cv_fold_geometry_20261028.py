#!/usr/bin/env python
"""cv_fold_geometry_20261028.py - window schedule and CV fold geometry of the thesis engine, from its own code.

Runs the verbatim functions `schedule` (model/oos.py) and `day_folds` (model/lasso.py) and the constants and
EngineConfig defaults of model/config.py on a synthetic calendar (no data are read, the engine is not imported):
the calendar is every UTC day from SAMPLE_FIRST_DAY to SAMPLE_LAST_DAY, and each window's rows are its origin
days x 5 coins in day-major order, as FeatureBuilder.rows builds them when no coin-day is missing (A00).

Outputs (in <outdir>):
  cv_windows_20261028.csv        one row per rolling window: number, first/last training origin, forecast origin, target
  cv_folds_window365_20261028.csv  one row per fold and window day: role (Training / Validation / Removed (buffer))
Usage: python cv_fold_geometry_20261028.py <thesis code dir (contains model/)> <outdir>
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

code_dir, outdir = Path(sys.argv[1]), Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)


def grab(path: Path, names):
    """Source of the named top-level functions / assignments / classes of a module, in file order."""
    src = path.read_text()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        nm = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
        if nm in names:
            keep.append(ast.get_source_segment(src, node))
    return "\n\n".join(keep)


ns = {"np": np, "pd": pd}
const = ["COINS", "SAMPLE_FIRST_DAY", "SAMPLE_LAST_DAY", "N_DAYS", "FIRST_ORIGIN", "LAST_ORIGIN", "FIRST_TARGET",
         "LAST_TARGET", "N_FORECAST_DAYS", "HAR_LAGS"]
exec(grab(code_dir / "model" / "config.py", const), ns)

# EngineConfig defaults that the CV and the window use (read from the class body, not re-typed)
cfg_src = (code_dir / "model" / "config.py").read_text()
cls = next(n for n in ast.parse(cfg_src).body if isinstance(n, ast.ClassDef) and n.name == "EngineConfig")
defaults = {}
for st in cls.body:
    if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
        if st.target.id in ("window", "first_target", "last_target", "cv_scheme", "cv_folds", "cv_gap_days", "cv_rule"):
            defaults[st.target.id] = ast.literal_eval(st.value)
cfg = type("Cfg", (), defaults)()

FUT = "from __future__ import annotations\n"          # as in the modules: annotations are not evaluated
exec(FUT + grab(code_dir / "model" / "lasso.py", ["day_folds"]), ns)
exec(FUT + grab(code_dir / "model" / "oos.py", ["schedule"]), ns)

days = pd.date_range(ns["SAMPLE_FIRST_DAY"], ns["SAMPLE_LAST_DAY"], freq="D")
assert len(days) == ns["N_DAYS"], (len(days), ns["N_DAYS"])


class Builder:                                    # stub: only the calendar lookup that schedule() uses
    def day_index(self, day):
        return int(days.get_loc(pd.Timestamp(day)))


sched = ns["schedule"](Builder(), cfg)
assert len(sched) == ns["N_FORECAST_DAYS"], len(sched)
C = len(ns["COINS"])

win = pd.DataFrame([{"window": i + 1, "first_origin": days[tr[0]].date(), "last_origin": days[tr[-1]].date(),
                     "forecast_origin": days[te].date(), "target": days[te + 1].date(), "n_days": len(tr)}
                    for i, (te, tr) in enumerate(sched)])
assert (win.n_days == cfg.window).all()
assert str(win.target.iloc[0]) == str(ns["FIRST_TARGET"].date()) and str(win.target.iloc[-1]) == str(ns["LAST_TARGET"].date())
win.to_csv(outdir / "cv_windows_20261028.csv", index=False)

# fold roles: day_folds on the window's day-major rows; identical geometry in every window (checked)
ref = None
for i, (te, tr) in enumerate(sched):
    day_idx = np.repeat(tr, C)
    folds = ns["day_folds"](day_idx, cfg.cv_scheme, cfg.cv_folds, cfg.cv_gap_days)
    geo = []
    for k, (tr_rows, va_rows) in enumerate(folds):
        trd, vad = np.unique(day_idx[tr_rows]), np.unique(day_idx[va_rows])
        assert len(tr_rows) == C * len(trd) and len(va_rows) == C * len(vad)   # whole days, all coins together
        assert not set(trd) & set(vad)
        role = np.where(np.isin(tr, vad), "Validation", np.where(np.isin(tr, trd), "Training", "Removed (buffer)"))
        geo.append(role)
    geo = np.array(geo)
    if ref is None:
        ref = geo
        first_dates = days[tr]
    else:
        assert (geo == ref).all(), f"window {i + 1} differs"

rows = []
for k in range(cfg.cv_folds):
    for j in range(cfg.window):
        rows.append({"fold": k + 1, "day_in_window": j + 1, "date_window1": first_dates[j].date(), "role": ref[k, j]})
fd = pd.DataFrame(rows)
fd.to_csv(outdir / "cv_folds_window365_20261028.csv", index=False)

summ = fd.groupby(["fold", "role"]).size().unstack(fill_value=0)
print("config:", defaults)
print("windows:", len(win), "| first:", win.iloc[0].to_dict(), "| last:", win.iloc[-1].to_dict())
print("days per fold and role (identical in all", len(sched), "windows):")
print(summ.to_string())
for k in range(cfg.cv_folds):
    v = fd[(fd.fold == k + 1) & (fd.role == "Validation")].day_in_window
    print(f"fold {k + 1}: validation days {v.min()}-{v.max()}")
