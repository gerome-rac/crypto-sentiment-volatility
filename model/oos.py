"""Out-of-sample loop: rolling 365-day pooled window (1,825 coin-day rows), re-estimated every
day; first target 30 Aug 2020, last 31 Jan 2023 (885 days x 5 coins = 4,425 forecasts per rung).
Also rolling 250 days (A06), expanding (A55) and per-coin estimation (A01).
Parallel over forecast days with joblib (loky), BLAS limited to one thread per worker.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import FIRST_ORIGIN, LAST_ORIGIN, EngineConfig
from .har import FeatureBuilder
from .lasso import fit_window


def schedule(builder: FeatureBuilder, cfg: EngineConfig):
    """List of (forecast-origin day index, training origin day indices)."""
    o0, o1 = builder.day_index(FIRST_ORIGIN), builder.day_index(LAST_ORIGIN)
    t0 = builder.day_index(pd.Timestamp(cfg.first_target) - pd.Timedelta(days=1))
    t1 = builder.day_index(pd.Timestamp(cfg.last_target) - pd.Timedelta(days=1))
    out = []
    for te in range(t0, t1 + 1):
        if cfg.window == "expanding":
            tr = np.arange(o0, te)
        else:
            w = int(cfg.window)
            tr = np.arange(te - w, te)
            if tr[0] < o0:
                raise ValueError(f"window {w} needs origins before {FIRST_ORIGIN.date()} for target day {te + 1}")
        assert tr[-1] == te - 1 and te <= o1
        out.append((te, tr))
    return out


def forecast_coin_days(builder: FeatureBuilder, cfg: EngineConfig) -> pd.DataFrame:
    """28 Oct 2026: the (coin, origin, target) rows a full run forecasts, without fitting (schedule plus the drop
    rule), for checks of an arm's evaluation set."""
    rows = []
    for te, _ in schedule(builder, cfg):
        for c in range(builder.C):
            ok = np.isfinite(builder.y[c, te]) and (builder.row_ok is None or bool(builder.row_ok[c, te]))
            if ok:
                rows.append((builder.coins[c], builder.days[te], builder.days[te + 1]))
    return pd.DataFrame(rows, columns=["coin", "origin", "target"])


def _one(builder: FeatureBuilder, cfg: EngineConfig, rung: str, te: int, tr: np.ndarray,
         var_subset=None) -> Dict:
    t0 = time.perf_counter()
    C = builder.C
    res = dict(te=te, f=np.full(C, np.nan), y=np.full(C, np.nan), rv_next=np.full(C, np.nan),
               sigma2=np.full(C, np.nan), duan=np.full(C, np.nan), ok=np.ones(C, bool))
    groups = [None] if cfg.unit == "pooled" else [np.array([c]) for c in range(C)]
    coefs, extras = [], []
    for g in groups:
        d = builder.design(rung, tr, np.array([te]), coins_idx=g, var_subset=var_subset)
        cidx_tr = d.coin_tr if g is None else np.zeros(len(d.coin_tr), int)
        cidx_te = d.coin_te if g is None else np.zeros(len(d.coin_te), int)
        ncoin = C if g is None else 1
        # 28 Oct 2026: a coin without a training row in the window has no fixed effect; never forecast it silently
        present = np.bincount(cidx_tr, minlength=ncoin) > 0
        if not present.all():
            lost = [builder.coins[i] for i in (np.flatnonzero(~present) if g is None else g)]
            raise ValueError(f"forecast origin {builder.days[te].date()}: no training row for {lost} in the window, so "
                             "the coin's fixed effect is not identified (open question A52 in engine_arms_20261028.md)")
        fit = fit_window(d.X_tr, d.y_tr, d.penalised, cidx_tr, d.day_tr, ncoin, cfg, sw=d.w_tr)
        f = fit.predict(d.X_te, cidx_te)
        sl = d.coin_te
        res["f"][sl] = f
        res["y"][sl] = d.y_te
        res["rv_next"][sl] = builder.rv_next[d.coin_te, d.day_te]
        if d.te_ok is not None:                  # drop rule: no forecast from or for a flagged coin-day
            res["ok"][sl] = d.te_ok
        res["sigma2"][sl] = fit.sigma2
        res["duan"][sl] = fit.duan
        coefs.append(fit.coef_z)
        extras.append(dict(lam=fit.lam, ridge_alpha_rel=fit.ridge_alpha_rel,
                           at_edge=fit.cv.get("at_edge", False), i_lambda=fit.cv.get("i_sel", -1),
                           n_selected=int(np.count_nonzero(fit.coef_z[d.penalised])), n_rows=len(d.y_tr),
                           n_nan_train=d.info["n_nan_filled_train"], n_nan_test=d.info["n_nan_filled_test"],
                           estimator=fit.estimator))
        names, blocks = d.names, d.blocks
    res["coef_z"] = np.vstack(coefs)          # (1 or C) x p
    res["names"], res["blocks"] = names, blocks
    res["extra"] = extras
    res["seconds"] = time.perf_counter() - t0
    return res


def _chunk(builder, cfg, rung, items, var_subset):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return [_one(builder, cfg, rung, te, tr, var_subset) for te, tr in items]


def run_oos(builder: FeatureBuilder, cfg: EngineConfig, rung: str, n_jobs: Optional[int] = None,
            max_days: Optional[int] = None, var_subset=None):
    """Returns (forecasts DataFrame, coefficient DataFrame, window-meta DataFrame, wall seconds)."""
    sched = schedule(builder, cfg)
    if max_days:
        sched = sched[:max_days]
    nj = n_jobs or cfg.n_jobs
    t0 = time.perf_counter()
    if nj == 1:
        out = _chunk(builder, cfg, rung, sched, var_subset)
    else:
        chunks = [sched[i::nj * 4] for i in range(nj * 4)]
        parts = Parallel(n_jobs=nj, backend="loky")(delayed(_chunk)(builder, cfg, rung, c, var_subset) for c in chunks if c)
        out = sorted([r for p in parts for r in p], key=lambda r: r["te"])
    wall = time.perf_counter() - t0
    days = builder.days
    fr = []
    for r in out:
        for c in range(builder.C):
            fr.append(dict(rung=rung, coin=builder.coins[c], origin=days[r["te"]], target=days[r["te"] + 1],
                           y=r["y"][c], f_log=r["f"][c], rv_next=r["rv_next"][c],
                           sigma2=r["sigma2"][c], duan=r["duan"][c]))
    fc = pd.DataFrame(fr)
    fc["F_lognormal"] = np.exp(fc.f_log + fc.sigma2 / 2)
    fc["F_duan"] = np.exp(fc.f_log) * fc.duan
    if builder.y_eval is not None:               # A62: evaluation target replaces y and rv_next; fit target kept
        te_idx = np.repeat([r["te"] for r in out], builder.C)
        c_idx = np.tile(np.arange(builder.C), len(out))
        fc["y_fit"], fc["rv_next_fit"] = fc["y"], fc["rv_next"]
        fc["y"] = builder.y_eval[c_idx, te_idx]
        fc["rv_next"] = builder.rv_next_eval[c_idx, te_idx]
    ok = np.concatenate([r["ok"] for r in out])
    if not ok.all():                             # drop rule: forecasts from or for flagged coin-days do not exist
        fc = fc[ok].reset_index(drop=True)
        out = [r for r in out if r["ok"].any()]
    names = out[0]["names"]
    rows = []
    for r in out:
        for g, cz in enumerate(r["coef_z"]):
            rows.append([days[r["te"]], g] + list(cz))
    coefs = pd.DataFrame(rows, columns=["origin", "group"] + names)
    meta = pd.DataFrame([dict(origin=days[r["te"]], group=g, seconds=r["seconds"], **e)
                         for r in out for g, e in enumerate(r["extra"])])
    assert fc.groupby("coin").size().nunique() == 1
    assert np.isfinite(fc.f_log).all(), "non-finite forecast"
    assert np.isfinite(fc.y).all() and np.isfinite(fc.rv_next).all(), "non-finite target in a forecast row"
    return fc, coefs, meta, wall, out[0]["blocks"]
