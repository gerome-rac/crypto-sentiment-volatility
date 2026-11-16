"""In-sample results of the side arms of 12 Nov 2026 (R3, OLS on all days, coin fixed effects, Driscoll-Kraay).

For each arm of the queue (also the `_alone` arms, which were run without the engine's in-sample stage):
  * fit: number of slope regressors, within R2 and MSE of R3, against R2 and the primary R3;
  * joint test of the new terms: Driscoll-Kraay Wald statistic with its chi-square p-value and with the stationary
    block bootstrap over days of the engine (model.inference.wald_bootstrap, here on the new terms only; 499 draws);
  * cumulative coefficient of each new variable (sum of its 1-, 7- and 30-day coefficients in raw units, as the
    engine's fear and anger test) with a two-sided Driscoll-Kraay z-test, and, for the valenced emotions, the
    difference positive minus negative of the cumulative coefficients;
  * from the stored run outputs, where the in-sample stage ran: the fear-minus-anger difference (h4_bootstrap.csv,
    two-sided Driscoll-Kraay p) and the in-sample adaptive LASSO's selection of the new terms (insample_lasso_R3.csv).
Asserted: the OLS coefficients computed here equal the run's stored insample_coef.csv, and the emotion-block Wald
statistic equals the stored insample_wald.csv, for every arm with a stored in-sample stage.

    cd code && python -m model.sidearm_insample --queue data/pipeline/model_runs/_queue_sidearms_20261112.csv \
        --primary A00_final_A50_20261028T134243Z --out data/pipeline/model_runs/_sidearm_insample_20261112
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from .config import LADDER, EngineConfig
from .har import FeatureBuilder
from .inference import dk_cov_within, ols_dk_nw, stationary_block_bootstrap_days
from .panel import build_grid, load_tables, load_variable_spec
from .run import specs_for_rungs

HERE = os.path.dirname(__file__)
RUNS = os.path.join(HERE, "..", "data", "pipeline", "model_runs")
EMO6 = ["happy", "sad", "anger", "disgust", "surprise", "fear"]


def fit_r3(panel: str, spec: str, cfg: EngineConfig):
    specs = specs_for_rungs(load_variable_spec(spec), list(LADDER), cfg)
    grid = build_grid(load_tables(panel), specs)
    builder = FeatureBuilder(grid, specs, cfg)
    return builder, ols_dk_nw(builder, "R3", cfg), ols_dk_nw(builder, "R2", cfg)


def wald_subset(ins, sel: np.ndarray, cfg: EngineConfig, reps: int):
    """Engine's bootstrap Wald test (inference.wald_bootstrap) on the columns in `sel`."""
    from arch.bootstrap import optimal_block_length
    d, Z = ins["design"], ins["Z"]
    ud, inv = np.unique(d.day_tr, return_inverse=True)
    bw = cfg.dk_bandwidth if cfg.dk_bandwidth is not None else float(np.floor(4 * (len(ud) / 100) ** (2 / 9)))
    b, V = dk_cov_within(Z, d.y_tr, d.coin_tr, inv, bw)
    bs = b[sel]
    W = float(bs @ np.linalg.solve(V[np.ix_(sel, sel)], bs))
    bl = cfg.boot_block or max(1.0, float(optimal_block_length(np.bincount(inv, weights=ins["resid"]) / np.bincount(inv))["stationary"].iloc[0]))
    rows_by_day = [np.flatnonzero(inv == i) for i in range(len(ud))]
    Ws = np.zeros(reps)
    for r, didx in enumerate(stationary_block_bootstrap_days(len(ud), bl, reps, cfg.seed + 7)):
        rows = np.concatenate([rows_by_day[i] for i in didx])
        pos = np.repeat(np.arange(len(didx)), [len(rows_by_day[i]) for i in didx])
        bb, VV = dk_cov_within(Z[rows], d.y_tr[rows], d.coin_tr[rows], pos, bw)
        db = (bb - b)[sel]
        Ws[r] = db @ np.linalg.solve(VV[np.ix_(sel, sel)], db)
    q = int(sel.sum())
    return dict(q=q, wald_dk=W, p_chi2=float(1 - stats.chi2.cdf(W, q)), p_boot=float((1 + np.sum(Ws >= W)) / (reps + 1)),
                boot_reps=reps, block_length=bl)


def cum(ins, var: str):
    names, sd = ins["names"], ins["sd"]
    a = np.zeros(len(names))
    for k in ("h1", "h7", "h30"):
        j = names.index(f"{var}_{k}")
        a[j] = 1.0 / sd[j]
    return a


def ztest(ins, a: np.ndarray):
    b = ins["coef"].beta_z.to_numpy()
    est = float(a @ b)
    se = float(np.sqrt(a @ ins["V_dk"] @ a))
    return est, se, float(2 * (1 - stats.norm.cdf(abs(est / se))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--out", required=True, help="prefix of the three output tables")
    ap.add_argument("--reps", type=int, default=499)
    a = ap.parse_args(argv)
    q = pd.read_csv(a.queue, keep_default_na=False)
    cfg = EngineConfig.for_spec("A00")
    spec0 = set(pd.read_csv(os.path.join(HERE, "variable_spec_A50.csv")).name)
    fit0 = pd.read_csv(os.path.join(RUNS, a.primary, "insample_fit.csv")).set_index("rung")
    arms, cums, lasso = [], [], []
    for r in q.itertuples():
        builder, i3, i2 = fit_r3(r.panel, r.variable_spec, cfg)
        assert abs(i2["fit"]["R2_within"] - fit0.loc["R2", "R2_within"]) < 1e-10, r.run_id      # R2 is the primary's R2
        sp = pd.read_csv(os.path.join(HERE, f"variable_spec_{r.run_id}.csv"), keep_default_na=False)
        new = [n for n in sp.name if n not in spec0]
        names = i3["names"]
        sel = np.array([any(n == f"{v}_{k}" for v in new for k in ("h1", "h7", "h30")) for n in names])
        assert int(sel.sum()) == 3 * len(new), r.run_id
        run = os.path.basename(sorted(glob.glob(os.path.join(RUNS, f"A00_{r.label}_2*")))[0])
        stored = os.path.join(RUNS, run, "insample_coef.csv")
        row = {"arm": r.run_id, "run_dir": run, "k_R3": i3["fit"]["k"], "new_terms": int(sel.sum()),
               "R2_within_R2": i2["fit"]["R2_within"], "R2_within_R3": i3["fit"]["R2_within"],
               "R2_within_primary_R3": fit0.loc["R3", "R2_within"], "MSE_log_R3": i3["fit"]["MSE_log"]}
        if os.path.exists(stored):
            c = pd.read_csv(stored)
            c = c[c.rung == "R3"].set_index("variable").reindex(names)
            assert float(np.abs(c.beta_z.to_numpy() - i3["coef"].beta_z.to_numpy()).max()) < 1e-8, r.run_id
            w = pd.read_csv(os.path.join(RUNS, run, "insample_wald.csv"))
            w = w[(w.rung == "R3") & (w["cov"] == "DK")].iloc[0]
            blk = np.array(i3["coef"].block == "emotion")
            b = i3["coef"].beta_z.to_numpy()
            W = float(b[blk] @ np.linalg.solve(i3["V_dk"][np.ix_(blk, blk)], b[blk]))
            assert abs(W - w.wald) < 1e-6 * max(1.0, w.wald), r.run_id
            h4 = pd.read_csv(os.path.join(RUNS, run, "h4_bootstrap.csv")).iloc[1]
            row.update(fear_minus_anger=h4.estimate, fear_minus_anger_p_two_sided=2 * min(h4.dk_p, 1 - h4.dk_p))
            il = pd.read_csv(os.path.join(RUNS, run, "insample_lasso_R3.csv")).set_index("variable")
            for n in np.array(names)[sel]:
                lasso.append({"arm": r.run_id, "term": n, "selected": bool(il.loc[n, "selected"]), "beta_z": il.loc[n, "beta_z"],
                              "boot_selection_freq": il.loc[n, "boot_selection_freq"]})
            row["lasso_new_terms_selected"] = int(sum(x["selected"] for x in lasso if x["arm"] == r.run_id))
        row.update(wald_subset(i3, sel, cfg, a.reps))
        arms.append(row)
        for v in new:
            est, se, p = ztest(i3, cum(i3, v))
            cums.append({"arm": r.run_id, "quantity": f"Cum({v})", "estimate": est, "se_dk": se, "z": est / se, "p_two_sided": p})
        pairs = [k for k in EMO6 if f"emo_{k}_pos" in new and f"emo_{k}_neg" in new]
        for k in pairs:
            est, se, p = ztest(i3, cum(i3, f"emo_{k}_pos") - cum(i3, f"emo_{k}_neg"))
            cums.append({"arm": r.run_id, "quantity": f"Cum(emo_{k}_pos) - Cum(emo_{k}_neg)", "estimate": est, "se_dk": se,
                         "z": est / se, "p_two_sided": p})
        print(r.run_id, "done", flush=True)
    for name, tab in (("arms", arms), ("cumulative", cums), ("lasso", lasso)):
        f = f"{a.out}_{name}.csv"
        pd.DataFrame(tab).to_csv(f + ".tmp", index=False)
        os.replace(f + ".tmp", f)
    pd.set_option("display.width", 250)
    A = pd.DataFrame(arms)
    print(A[["arm", "k_R3", "new_terms", "R2_within_R3", "wald_dk", "q", "p_chi2", "p_boot", "fear_minus_anger",
             "fear_minus_anger_p_two_sided", "lasso_new_terms_selected"]].round(4).to_string(index=False))
    print(pd.DataFrame(cums).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
