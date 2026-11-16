#!/usr/bin/env python3
"""regime_blocks.py -- the text blocks by market phase, in sample (15 Nov 2026; post-result, descriptive).

Extends Part 2 of arm A07 (model.arm_regime_split): pooled OLS with every regressor and the coin effects interacted
with the phase of the target day, Driscoll-Kraay covariance, so each phase has its own coefficients. A07 reported
the polarity terms and fear minus anger. This script adds, for the block that enters in each rung (attention in R1,
polarity in R2, emotions in R3) and for both dating rules (lt20, ma200):

  blocks_wald.csv     joint Wald tests: block zero in bull, zero in bear, equal across phases; chi-square p and a
                      stationary-block-bootstrap p (days resampled, W* = (R(b*-b))'(R V* R')^-1 (R(b*-b)), as
                      model.inference.wald_bootstrap), because chi-square critical values reject far too often for
                      blocks of 45 and 54 terms (thesis, Appendix C).
  blocks_cum.csv      cumulative effect (1 + 7 + 30 days) of each variable and form by phase and bull minus bear, raw
                      units, Driscoll-Kraay z-test (one restriction each).
  blocks_terms.csv    every coefficient of the three blocks by phase.

Checks (asserted): the R2 and R3 coefficients equal the stored part2_coef_by_regime.csv of the A07 run, and the
chi-square p-values of the polarity block equal its part2_wald_by_regime.csv.

Usage (from the code/ folder):
  python <this file> <main_run_dir> <panel.parquet> <variable_spec.csv> <A07_run_dir> <out_dir> [reps]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path.cwd()))
from arch.bootstrap import optimal_block_length  # noqa: E402
from model.arm_regime_split import REGIMES, build_builder, load_run  # noqa: E402
from model.inference import _zdesign, dk_cov_within, insample_design, stationary_block_bootstrap_days  # noqa: E402

run_dir, panel, spec, a07, out = sys.argv[1:6]
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 499
a07, out = Path(a07), Path(out)
cfg, _, _ = load_run(run_dir)
builder = build_builder(panel, spec, cfg)
dating = pd.read_csv(a07 / "regime_dating.csv", parse_dates=["day"]).set_index("day")
ENTRY = {"R1": "attention", "R2": "polarity", "R3": "emotion"}
stored = pd.read_csv(a07 / "part2_coef_by_regime.csv")
stored_w = pd.read_csv(a07 / "part2_wald_by_regime.csv")


def design(rung, rule):
    d = insample_design(builder, rung)
    Z, names, mu, sd, keep = _zdesign(d)
    blocks = [bk for bk, k in zip(d.blocks, keep) if k]
    reg = dating[f"{rule}_regime"].reindex(builder.days[d.day_tr + 1]).to_numpy(dtype=object)
    use = np.array([x in REGIMES for x in reg])
    origin = builder.days[d.day_tr][use]
    assert len(np.unique(origin)) == len(pd.date_range(origin.min(), origin.max(), freq="D"))
    Zu, y = Z[use], d.y_tr[use]
    coins = np.array(builder.coins)[d.coin_tr][use]
    bull = reg[use] == "bull"
    X = np.hstack([Zu * bull[:, None], Zu * (~bull)[:, None]])
    grp = pd.factorize(np.array([f"{c}|{'bull' if b else 'bear'}" for c, b in zip(coins, bull)]))[0]
    ud, pos = np.unique(origin, return_inverse=True)
    bw = cfg.dk_bandwidth if cfg.dk_bandwidth is not None else float(np.floor(4 * (len(ud) / 100) ** (2 / 9)))
    return dict(X=X, y=y, grp=grp, pos=pos, bw=bw, names=names, blocks=blocks, sd=np.r_[sd, sd], p=Zu.shape[1], n_days=len(ud))


def sel(names, terms, p, regime):
    a = np.zeros(2 * p); off = 0 if regime == "bull" else p
    for t in terms: a[off + names.index(t)] = 1.0
    return a


wald, cum, terms_out = [], [], []
for rule in ("lt20", "ma200"):
    for rung, block in ENTRY.items():
        D = design(rung, rule)
        X, y, grp, pos, bw, names, p, sd2 = D["X"], D["y"], D["grp"], D["pos"], D["bw"], D["names"], D["p"], D["sd"]
        b, V = dk_cov_within(X, y, grp, pos, bw)
        braw, Vraw = b / sd2, V / np.outer(sd2, sd2)
        se = np.sqrt(np.diag(Vraw))
        if rung in ("R2", "R3"):                       # reproduction of the stored A07 coefficients
            s = stored[(stored.rule == rule) & (stored.rung == rung)]
            got = pd.DataFrame(dict(variable=names * 2, regime=["bull"] * p + ["bear"] * p, coef=braw, se=se))
            m = s.merge(got, on=["variable", "regime"]); assert len(m) == 2 * p
            assert np.allclose(m.coef_raw, m.coef, rtol=1e-7, atol=1e-9) and np.allclose(m.se_dk_raw, m.se, rtol=1e-6), (rule, rung)
        bt = [n for n, bk in zip(names, D["blocks"]) if bk == block]
        R = {"zero in bull": np.vstack([sel(names, [t], p, "bull") for t in bt]),
             "zero in bear": np.vstack([sel(names, [t], p, "bear") for t in bt]),
             "equal across phases": np.vstack([sel(names, [t], p, "bull") - sel(names, [t], p, "bear") for t in bt])}
        W = {k: float((r @ b) @ np.linalg.solve(r @ V @ r.T, r @ b)) for k, r in R.items()}
        # stationary block bootstrap over days, centred at the estimate
        C = int(grp.max()) + 1; cnt = np.bincount(grp, minlength=C)
        Xd = X - (np.vstack([np.bincount(grp, weights=X[:, j], minlength=C) for j in range(X.shape[1])]).T / cnt[:, None])[grp]
        resid = (y - (np.bincount(grp, weights=y, minlength=C) / cnt)[grp]) - Xd @ b
        dm = np.bincount(pos, weights=resid) / np.bincount(pos)
        bl = cfg.boot_block or max(1.0, float(optimal_block_length(dm)["stationary"].iloc[0]))
        rows_by_day = [np.flatnonzero(pos == i) for i in range(D["n_days"])]
        Ws = {k: np.full(reps, np.nan) for k in R}
        for r_i, didx in enumerate(stationary_block_bootstrap_days(D["n_days"], bl, reps, cfg.seed + 11)):
            rows = np.concatenate([rows_by_day[i] for i in didx])
            newpos = np.repeat(np.arange(len(didx)), [len(rows_by_day[i]) for i in didx])
            try:
                bb, VV = dk_cov_within(X[rows], y[rows], grp[rows], newpos, bw)
                for k, r in R.items():
                    dd = r @ (bb - b); Ws[k][r_i] = dd @ np.linalg.solve(r @ VV @ r.T, dd)
            except np.linalg.LinAlgError:
                pass
        for k, r in R.items():
            ok = np.isfinite(Ws[k]); q = r.shape[0]
            wald.append(dict(rule=rule, rung=rung, block=block, test=k, q=q, wald=W[k], p_chi2=float(1 - stats.chi2.cdf(W[k], q)),
                             p_boot=float((1 + np.sum(Ws[k][ok] >= W[k])) / (ok.sum() + 1)), boot_reps=int(ok.sum()), block_length=bl,
                             n_rows=len(y), n_days=D["n_days"]))
        if block == "polarity":
            for k, lab in (("equal across phases", "polarity (9 terms) equal across regimes"), ("zero in bull", "polarity (9 terms) jointly zero in bull"),
                           ("zero in bear", "polarity (9 terms) jointly zero in bear")):
                sp = float(stored_w[(stored_w.rule == rule) & (stored_w.rung == "R2") & (stored_w.block == lab)].p.iloc[0])
                assert abs(sp - (1 - stats.chi2.cdf(W[k], len(bt)))) < 1e-8, (rule, k)
        # cumulative effects per variable and form
        base = sorted({t.rsplit("_h", 1)[0] for t in bt if t.rsplit("_h", 1)[-1] in ("1", "7", "30")})
        for v in base:
            ts = [f"{v}_h{k}" for k in (1, 7, 30)]
            if not all(t in names for t in ts): continue
            for regime, a in (("bull", sel(names, ts, p, "bull")), ("bear", sel(names, ts, p, "bear")),
                              ("bull minus bear", sel(names, ts, p, "bull") - sel(names, ts, p, "bear"))):
                est, s_ = float(a @ braw), float(np.sqrt(a @ Vraw @ a))
                cum.append(dict(rule=rule, rung=rung, block=block, variable=v, regime=regime, cumulative=est, se_dk=s_,
                                p=float(2 * (1 - stats.norm.cdf(abs(est / s_))))))
        for j, n in enumerate(names):
            if D["blocks"][j] == block:
                for regime, off in (("bull", 0), ("bear", p)):
                    terms_out.append(dict(rule=rule, rung=rung, block=block, variable=n, regime=regime, coef_raw=braw[off + j], se_dk=se[off + j],
                                          p=float(2 * (1 - stats.norm.cdf(abs(braw[off + j] / se[off + j]))))))
        print(rule, rung, block, "done; block length", round(bl, 2), flush=True)

out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(wald).to_csv(out / "blocks_wald.csv", index=False)
pd.DataFrame(cum).to_csv(out / "blocks_cum.csv", index=False)
pd.DataFrame(terms_out).to_csv(out / "blocks_terms.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(wald)[["rule", "rung", "block", "test", "q", "wald", "p_chi2", "p_boot", "boot_reps"]].round(4).to_string(index=False))
