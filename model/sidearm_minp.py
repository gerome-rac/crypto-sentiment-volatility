"""Multiplicity adjustment of the in-sample joint tests of the side arms that accounts for their dependence.

The side arms of 12 Nov 2026 are variants of two questions (does an aggregate measure of emotion dispersion matter;
does the tone of an emotion matter), and several of them are near-duplicates (the within-post Gini and entropy
measures correlate at 1.00). Holm's adjustment is valid under any dependence but treats near-duplicates as separate
tests. Here the joint Driscoll-Kraay Wald statistic of each arm's new terms is bootstrapped on COMMON resampled days
(stationary block bootstrap over days, one block length for all arms: the optimal length for the primary R3
residuals), and the adjusted p-values are the step-down min-p values of Westfall and Young (1993, Algorithm 2.8; as
in Romano and Wolf 2005): for the arm with the k-th smallest raw p, the share of draws in which the smallest
bootstrap p among the arms not yet stepped past is at most that raw p, made monotone.

    cd code && python -m model.sidearm_minp --queue data/pipeline/model_runs/_queue_sidearms_20261112.csv \
        --out data/pipeline/model_runs/_sidearm_insample_20261112_minp.csv [--reps 1999]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import EngineConfig
from .inference import dk_cov_within, stationary_block_bootstrap_days
from .sidearm_insample import HERE, fit_r3

FAMILIES = {
    "dispersion, next to the emotion terms": ["A75", "A76", "A77", "A78", "A79", "A80"],
    "dispersion, alone on R2": ["A75_alone", "A76_alone", "A77_alone", "A78_alone", "A79_alone", "A80_alone"],
    "tone of emotions": ["A81", "A82", "A83", "A84"],
    "all ten added-term arms": ["A75", "A76", "A77", "A78", "A79", "A80", "A81", "A82", "A83", "A84"],
    "all sixteen arms": None,
}


def one_arm(row, draws, bl_days):
    cfg = EngineConfig.for_spec("A00")
    spec0 = set(pd.read_csv(os.path.join(HERE, "variable_spec_A50.csv")).name)
    builder, i3, _ = fit_r3(row["panel"], row["variable_spec"], cfg)
    sp = pd.read_csv(os.path.join(HERE, f"variable_spec_{row['run_id']}.csv"), keep_default_na=False)
    new = [n for n in sp.name if n not in spec0]
    sel = np.array([any(n == f"{v}_{k}" for v in new for k in ("h1", "h7", "h30")) for n in i3["names"]])
    d, Z = i3["design"], i3["Z"]
    ud, inv = np.unique(d.day_tr, return_inverse=True)
    assert len(ud) == bl_days, row["run_id"]
    bw = float(np.floor(4 * (len(ud) / 100) ** (2 / 9)))
    b, V = dk_cov_within(Z, d.y_tr, d.coin_tr, inv, bw)
    W = float(b[sel] @ np.linalg.solve(V[np.ix_(sel, sel)], b[sel]))
    rows_by_day = [np.flatnonzero(inv == i) for i in range(len(ud))]
    Ws = np.zeros(len(draws))
    for r, didx in enumerate(draws):
        rows = np.concatenate([rows_by_day[i] for i in didx])
        pos = np.repeat(np.arange(len(didx)), [len(rows_by_day[i]) for i in didx])
        bb, VV = dk_cov_within(Z[rows], d.y_tr[rows], d.coin_tr[rows], pos, bw)
        db = (bb - b)[sel]
        Ws[r] = db @ np.linalg.solve(VV[np.ix_(sel, sel)], db)
    return row["run_id"], W, Ws, int(sel.sum())


def stepdown_minp(p_raw: np.ndarray, P: np.ndarray) -> np.ndarray:
    """p_raw: (m,) raw bootstrap p-values; P: (B, m) bootstrap p-value of each draw's statistic within its own arm."""
    order = np.argsort(p_raw)
    adj = np.zeros(len(p_raw))
    prev = 0.0
    for k, j in enumerate(order):
        mn = P[:, order[k:]].min(axis=1)
        a = (1 + np.sum(mn <= p_raw[j])) / (P.shape[0] + 1)
        prev = max(prev, a)
        adj[j] = prev
    return adj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=1999)
    ap.add_argument("--n-jobs", type=int, default=5)
    a = ap.parse_args(argv)
    q = pd.read_csv(a.queue, keep_default_na=False)
    cfg = EngineConfig.for_spec("A00")
    # common block length: optimal length for the day-mean residuals of the primary R3
    from arch.bootstrap import optimal_block_length
    _, p3, _ = fit_r3(q.panel.iloc[0], os.path.join("model", "variable_spec_A50.csv"), cfg)
    ud, inv = np.unique(p3["design"].day_tr, return_inverse=True)
    bl = max(1.0, float(optimal_block_length(np.bincount(inv, weights=p3["resid"]) / np.bincount(inv))["stationary"].iloc[0]))
    draws = list(stationary_block_bootstrap_days(len(ud), bl, a.reps, cfg.seed + 7))
    res = Parallel(n_jobs=a.n_jobs)(delayed(one_arm)(r._asdict(), draws, len(ud)) for r in q.itertuples(index=False))
    arms = [x[0] for x in res]
    W = np.array([x[1] for x in res])
    WS = np.column_stack([x[2] for x in res])                       # (B, m)
    B = WS.shape[0]
    p_raw = (1 + (WS >= W[None, :]).sum(axis=0)) / (B + 1)
    # bootstrap p-value of each draw within its own arm: share of draws with a statistic at least as large
    ranks = np.argsort(np.argsort(-WS, axis=0), axis=0) + 1         # 1 = largest
    P = ranks / B
    out = pd.DataFrame({"arm": arms, "q": [x[3] for x in res], "wald_dk": W, "p_boot": p_raw})
    for fam, members in FAMILIES.items():
        idx = np.array([arms.index(m) for m in (members or arms)])
        adj = stepdown_minp(p_raw[idx], P[:, idx])
        hs = np.sort(p_raw[idx])
        ho = np.maximum.accumulate(np.minimum(1.0, hs * (len(idx) - np.arange(len(idx)))))
        holm = np.empty(len(idx)); holm[np.argsort(p_raw[idx])] = ho
        col = pd.Series(np.nan, index=range(len(arms)))
        col.iloc[idx] = adj
        out[f"minp | {fam}"] = col.to_numpy()
        colh = pd.Series(np.nan, index=range(len(arms)))
        colh.iloc[idx] = holm
        out[f"holm | {fam}"] = colh.to_numpy()
    out.attrs = {}
    out.to_csv(a.out + ".tmp", index=False)
    os.replace(a.out + ".tmp", a.out)
    pd.set_option("display.width", 260)
    print(f"block length {bl:.2f} days, {B} draws")
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
