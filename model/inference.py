"""In-sample estimation and inference (test plan Stage 3, tests 3.3-3.14).

Each rung is estimated once on all coin-days (origins 30 Aug 2019 - 30 Jan 2023, 6,250 rows),
with the in-window transforms computed on that whole sample (the in-sample 'window').
  * OLS with coin FE; Driscoll-Kraay SEs (linearmodels PanelOLS cov_type='kernel', primary) and
    Newey-West within coin (statsmodels 'hac-panel', comparison);
  * joint Wald tests per block (3.6 attention in R1, 3.7 polarity in R2, 3.8 emotions in R3);
  * H4 (3.9): theta1 = sum_k beta_fear,k (k = 1, 7, 30, linear terms at the centre) > 0 and
    theta2 = theta1 - sum_k beta_anger,k > 0; stationary block bootstrap over days (Politis &
    Romano 1994; automatic block length, Politis & White 2004 via arch), p_H4 = max(p1, p2);
  * adaptive LASSO in-sample (3.10) with block-bootstrap intervals at fixed lambda (3.5, 3.14);
  * selection frequencies across the rolling windows (3.13).
No t-statistics are produced for LASSO coefficients.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import FIRST_ORIGIN, LAST_ORIGIN, EngineConfig, hac_default_lags
from .har import FeatureBuilder
from .lasso import fe_dummies, fit_window

WALD_BLOCK = {"R1": "attention", "R2": "polarity", "R3": "emotion"}
WALD_TEST_ID = {"R1": "3.6", "R2": "3.7", "R3": "3.8"}


def insample_design(builder: FeatureBuilder, rung: str):
    o0, o1 = builder.day_index(FIRST_ORIGIN), builder.day_index(LAST_ORIGIN)
    tr = np.arange(o0, o1 + 1)
    return builder.design(rung, tr, np.array([o1]))


def _zdesign(d):
    mu, sd = d.X_tr.mean(0), d.X_tr.std(0)
    keep = sd > 0
    Z = (d.X_tr[:, keep] - mu[keep]) / sd[keep]
    names = [n for n, k in zip(d.names, keep) if k]
    return Z, names, mu[keep], sd[keep], keep


def ols_dk_nw(builder: FeatureBuilder, rung: str, cfg: EngineConfig) -> Dict:
    from linearmodels.panel import PanelOLS
    import statsmodels.api as sm
    d = insample_design(builder, rung)
    Z, names, mu, sd, keep = _zdesign(d)
    blocks = [b for b, k in zip(d.blocks, keep) if k]
    coins = np.array(builder.coins)[d.coin_tr]
    days = builder.days[d.day_tr]
    idx = pd.MultiIndex.from_arrays([coins, days], names=["coin", "day"])
    Xdf = pd.DataFrame(Z, index=idx, columns=names)
    y = pd.Series(d.y_tr, index=idx, name="y")
    mod = PanelOLS(y, Xdf, entity_effects=True)
    kw = dict(cov_type="kernel", kernel="bartlett")
    if cfg.dk_bandwidth is not None:
        kw["bandwidth"] = cfg.dk_bandwidth
    dk = mod.fit(**kw)
    Tdays = len(np.unique(d.day_tr))
    L = cfg.nw_lags if cfg.nw_lags is not None else hac_default_lags(Tdays)
    order = np.lexsort((d.day_tr, d.coin_tr))          # coin-major, time-ordered within coin
    Xn = np.column_stack([Z, fe_dummies(d.coin_tr, builder.C)])[order]
    nw = sm.OLS(d.y_tr[order], Xn).fit(cov_type="hac-panel",
                                        cov_kwds=dict(groups=d.coin_tr[order], maxlags=L))
    b = dk.params.to_numpy()
    assert np.allclose(b, nw.params[: len(b)], atol=1e-8), "PanelOLS and dummy OLS disagree"
    resid = d.y_tr - (Z @ b + nw.params[len(b):][d.coin_tr])
    coef = pd.DataFrame(dict(rung=rung, variable=names, block=blocks, beta_z=b, beta_raw=b / sd,
                             se_dk=dk.std_errors.to_numpy(), t_dk=dk.tstats.to_numpy(), p_dk=dk.pvalues.to_numpy(),
                             se_nw=nw.bse[: len(b)], t_nw=nw.tvalues[: len(b)], p_nw=nw.pvalues[: len(b)]))
    V_dk = dk.cov.to_numpy()
    V_nw = np.asarray(nw.cov_params())[: len(b), : len(b)]
    # linearmodels' default Bartlett bandwidth is floor(4 (T/100)^(2/9)) over the T time periods (7 at T = 1,250)
    bw = cfg.dk_bandwidth if cfg.dk_bandwidth is not None else float(np.floor(4 * (Tdays / 100) ** (2 / 9)))
    fit = dict(n=len(y), k=len(b), R2_within=float(dk.rsquared_within), MSE_log=float(np.mean(resid ** 2)),
               dk_bandwidth=bw, nw_lags=L)
    wald = []
    if rung in WALD_BLOCK:
        blk = WALD_BLOCK[rung]
        sel = np.array([bb == blk for bb in blocks])
        for lab, V in (("DK", V_dk), ("NW", V_nw)):
            R = np.eye(len(b))[sel]
            rb = R @ b
            W = float(rb @ np.linalg.solve(R @ V @ R.T, rb))
            from scipy import stats
            q = int(sel.sum())
            wald.append(dict(test=WALD_TEST_ID[rung], rung=rung, block=blk, cov=lab, q=q, wald=W,
                             p=float(1 - stats.chi2.cdf(W, q))))
    return dict(coef=coef, fit=fit, wald=pd.DataFrame(wald), V_dk=V_dk, design=d, Z=Z, names=names,
                sd=sd, keep=keep, resid=resid)


def dk_cov_within(Z: np.ndarray, y: np.ndarray, coin_idx: np.ndarray, day_pos: np.ndarray, bw: float):
    """Driscoll-Kraay covariance of the within (coin-demeaned) OLS slopes, Bartlett kernel, with
    linearmodels' scale n/(n - k - C). `day_pos` gives the time order (0..T-1) of each row.
    Returns (beta, V)."""
    C = int(coin_idx.max()) + 1
    cnt = np.bincount(coin_idx, minlength=C)
    Zd = Z - (np.vstack([np.bincount(coin_idx, weights=Z[:, j], minlength=C) for j in range(Z.shape[1])]).T / cnt[:, None])[coin_idx]
    yd = y - (np.bincount(coin_idx, weights=y, minlength=C) / cnt)[coin_idx]
    A = Zd.T @ Zd
    b = np.linalg.solve(A, Zd.T @ yd)
    e = yd - Zd @ b
    T = int(day_pos.max()) + 1
    H = np.zeros((T, Z.shape[1]))
    np.add.at(H, day_pos, Zd * e[:, None])
    S = H.T @ H
    L = int(bw)
    for l in range(1, L + 1):
        G = H[l:].T @ H[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    Ai = np.linalg.inv(A)
    n, k = Z.shape
    V = Ai @ S @ Ai * n / (n - k - C)
    return b, (V + V.T) / 2


def wald_bootstrap(ins: Dict, builder: FeatureBuilder, cfg: EngineConfig, reps: int = 999) -> pd.DataFrame:
    """Option (OP-WALD): stationary block bootstrap over days of the DK Wald statistic,
    W* = (b* - b)' V*_DK^-1 (b* - b) on the block, p = P*(W* >= W)."""
    from arch.bootstrap import optimal_block_length
    from scipy import stats
    d, Z = ins["design"], ins["Z"]
    rung = d_rung = ins["coef"].rung.iloc[0]
    if rung not in WALD_BLOCK:
        return pd.DataFrame()
    blocks = ins["coef"].block.to_numpy()
    sel = blocks == WALD_BLOCK[rung]
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
    return pd.DataFrame([dict(test=WALD_TEST_ID[rung], rung=rung, block=WALD_BLOCK[rung], cov="DK_bootstrap", q=q,
                              wald=W, p=float((1 + np.sum(Ws >= W)) / (reps + 1)), p_chi2=float(1 - stats.chi2.cdf(W, q)),
                              boot_reps=reps, block_length=bl, boot_W_median=float(np.median(Ws)))])


H4_TERMS = {v: tuple(f"{v}_{k}" for k in ("h1", "h7", "h30")) for v in ("emo_fear", "emo_anger")}


def h4_applicability(names) -> tuple:
    """28 Oct 2026: H4 (D8 section 4: beta_fear > 0 and beta_fear - beta_anger > 0, beta = sum of the 1/7/30 linear
    terms) needs the six regressors emo_fear_h1/h7/h30 and emo_anger_h1/h7/h30 in R3. Returns (True, "") when all six
    are present, (False, reason) when fear's or anger's terms are absent altogether (an arm whose R3 holds other emotion
    measures, e.g. A60), and raises when only some of a variable's terms are present (an inconsistent design)."""
    have = {v: [t for t in terms if t in names] for v, terms in H4_TERMS.items()}
    partial = sorted(v for v, h in have.items() if 0 < len(h) < len(H4_TERMS[v]))
    if partial:
        raise ValueError(f"H4: R3 holds only some of the 1/7/30 terms of {partial}; expected all or none")
    missing = sorted(v for v, h in have.items() if not h)
    if missing:
        return False, "R3 holds no " + " and no ".join(f"{v}_h1/h7/h30" for v in missing) + " terms"
    return True, ""


def h4_not_applicable_table(reason: str, cfg: EngineConfig) -> pd.DataFrame:
    """h4_bootstrap.csv for an arm without the H4 regressors: one 'not applicable' row, same columns as h4_test."""
    return pd.DataFrame([dict(condition=f"H4 not applicable: {reason}", estimate=np.nan, boot_p=np.nan, dk_se=np.nan,
                              dk_p=np.nan, units=cfg.h4_units, block_length=np.nan, reps=0)])


def _h4_weights(names, sd, units: str):
    def w(var):
        a = np.zeros(len(names))
        for k in ("h1", "h7", "h30"):
            nm = f"{var}_{k}"
            if nm in names:
                j = names.index(nm)
                a[j] = 1.0 / sd[j] if units == "raw" else 1.0
        return a
    a1 = w("emo_fear")
    a2 = a1 - w("emo_anger")
    assert a1.any() and np.count_nonzero(a1) == 3, "H4 needs emo_fear_h1/h7/h30 in R3"
    return a1, a2


def stationary_block_bootstrap_days(n_days: int, block: float, reps: int, seed: int):
    from arch.bootstrap import StationaryBootstrap
    bs = StationaryBootstrap(block, np.arange(n_days), seed=seed)
    for data, _ in bs.bootstrap(reps):
        yield data[0]


def h4_test(ins: Dict, builder: FeatureBuilder, cfg: EngineConfig) -> Dict:
    """Pre-registered H4 statistic (MASTERPLAN 6 Oct (11); D8 note section 4)."""
    from arch.bootstrap import optimal_block_length
    from scipy import stats
    d, Z, names, sd = ins["design"], ins["Z"], ins["names"], ins["sd"]
    a1, a2 = _h4_weights(names, sd, cfg.h4_units)
    X = np.column_stack([Z, fe_dummies(d.coin_tr, builder.C)])
    p = Z.shape[1]
    b, *_ = np.linalg.lstsq(X, d.y_tr, rcond=None)
    th = np.array([a1 @ b[:p], a2 @ b[:p]])
    # DK comparison (one-sided t)
    V = ins["V_dk"]
    se = np.sqrt([a1 @ V @ a1, a2 @ V @ a2])
    p_dk = 1 - stats.norm.cdf(th / se)
    # block length from the day-level influence series of theta1, theta2
    ud, inv = np.unique(d.day_tr, return_inverse=True)
    XtX_inv = np.linalg.pinv(X.T @ X)
    e = d.y_tr - X @ b
    infl = []
    for a in (a1, a2):
        aa = np.r_[a, np.zeros(builder.C)]
        s = (X @ (XtX_inv @ aa)) * e
        infl.append(np.bincount(inv, weights=s))
    if cfg.boot_block is None:
        bl = max(1.0, max(float(optimal_block_length(x)["stationary"].iloc[0]) for x in infl))
    else:
        bl = float(cfg.boot_block)
    rows_by_day = [np.flatnonzero(inv == i) for i in range(len(ud))]
    draws = np.zeros((cfg.boot_reps, 2))
    coefs = np.zeros((cfg.boot_reps, p))
    for r, didx in enumerate(stationary_block_bootstrap_days(len(ud), bl, cfg.boot_reps, cfg.seed)):
        rows = np.concatenate([rows_by_day[i] for i in didx])
        bb, *_ = np.linalg.lstsq(X[rows], d.y_tr[rows], rcond=None)
        coefs[r] = bb[:p]
        draws[r] = [a1 @ bb[:p], a2 @ bb[:p]]
    if cfg.h4_pvalue == "centred":
        pb = [(np.sum(draws[:, i] - th[i] >= th[i]) + 1) / (cfg.boot_reps + 1) for i in range(2)]
    else:
        pb = [(np.sum(draws[:, i] <= 0) + 1) / (cfg.boot_reps + 1) for i in range(2)]
    out = pd.DataFrame([
        dict(condition="theta1 = beta_fear > 0", estimate=th[0], boot_p=pb[0], dk_se=se[0], dk_p=p_dk[0]),
        dict(condition="theta2 = beta_fear - beta_anger > 0", estimate=th[1], boot_p=pb[1], dk_se=se[1], dk_p=p_dk[1]),
    ])
    out["units"] = cfg.h4_units
    out["block_length"] = bl
    out["reps"] = cfg.boot_reps
    anger = (a1 - a2) @ b[:p]
    ci = pd.DataFrame(dict(variable=names, beta_raw=b[:p] / sd, ci_lo_raw=np.percentile(coefs, 2.5, 0) / sd,
                           ci_hi_raw=np.percentile(coefs, 97.5, 0) / sd))
    return dict(table=out, p_h4=float(max(pb)), p_h4_dk=float(max(p_dk)), anger=float(anger), coef_ci=ci,
                block_length=bl)


def insample_lasso(builder: FeatureBuilder, rung: str, cfg: EngineConfig, boot_reps: int = 199) -> pd.DataFrame:
    """Adaptive LASSO on all coin-days (test 3.10) with stationary block-bootstrap intervals at the
    CV-chosen lambda (3.5/3.14: descriptive, no t-statistics)."""
    d = insample_design(builder, rung)
    fit = fit_window(d.X_tr, d.y_tr, d.penalised, d.coin_tr, d.day_tr, builder.C, cfg, sw=d.w_tr)
    out = pd.DataFrame(dict(rung=rung, variable=d.names, block=d.blocks, penalised=d.penalised,
                            beta_z=fit.coef_z, selected=fit.coef_z != 0))
    out["lambda"] = fit.lam
    if boot_reps and d.penalised.any():
        from arch.bootstrap import optimal_block_length
        ud, inv = np.unique(d.day_tr, return_inverse=True)
        bl = max(1.0, float(optimal_block_length(np.bincount(inv, weights=fit.resid) / np.bincount(inv))["stationary"].iloc[0]))
        rows_by_day = [np.flatnonzero(inv == i) for i in range(len(ud))]
        B = np.zeros((boot_reps, len(d.names)))
        for r, didx in enumerate(stationary_block_bootstrap_days(len(ud), bl, boot_reps, cfg.seed + 1)):
            rows = np.concatenate([rows_by_day[i] for i in didx])
            # re-index days so the CV folds stay contiguous blocks of the resampled series
            f = fit_window(d.X_tr[rows], d.y_tr[rows], d.penalised, d.coin_tr[rows],
                           np.repeat(np.arange(len(didx)), [len(rows_by_day[i]) for i in didx]),
                           builder.C, cfg, fixed_lambda=fit.lam, fixed_ridge_rel=fit.ridge_alpha_rel)
            B[r] = f.coef_z
        out["boot_ci_lo_z"] = np.percentile(B, 2.5, 0)
        out["boot_ci_hi_z"] = np.percentile(B, 97.5, 0)
        out["boot_selection_freq"] = (B != 0).mean(0)
        out["boot_block_length"] = bl
        out["boot_reps"] = boot_reps
    return out


def selection_frequencies(coefs: pd.DataFrame, names, blocks, base) -> Dict[str, pd.DataFrame]:
    """Test 3.13: share of rolling windows with a non-zero coefficient, per column, per variable
    (any of its columns) and per block."""
    M = coefs[list(names)].to_numpy() != 0
    col = pd.DataFrame(dict(column=names, block=blocks, variable=base, freq=M.mean(0)))
    var = (pd.DataFrame(M, columns=names).T.assign(variable=base).groupby("variable").any().T.mean()
           .rename("freq").reset_index())
    var["block"] = var.variable.map(dict(zip(base, blocks)))
    blk = (pd.DataFrame(M, columns=names).T.assign(block=blocks).groupby("block").any().T.mean()
           .rename("freq").reset_index())
    blk["n_windows"] = M.shape[0]
    return dict(columns=col, variables=var, blocks=blk)


# ----------------------------------------------------------------------------- recentered bootstrap
# 7 Nov 2026 (Gerome; post-result addition, logged in the D8 deviation log): p-values for the in-sample
# adaptive LASSO coefficients with the recentered block bootstrap of Audrino & Camponovo (2018, J. Time Ser.
# Anal. 39, pp. 115-117), as used by Audrino, Sigrist & Ballinari (2020, IJF 36, pp. 345-346).
#   * bootstrap samples: stationary block bootstrap over whole days (all coins of a day together); A&C Remark 3
#     allows the stationary bootstrap, whose circular resampling keeps E*[mean gradient at the OLS estimate] = 0;
#   * eq. (4): in every bootstrap sample the adaptive LASSO (same lambda, same relative ridge penalty, ridge
#     weights re-estimated on the sample) penalises |theta_j - c_j|, with c_j = the full-sample OLS estimate of
#     column j if the original adaptive LASSO set it to zero, else 0;
#   * Remark 4: the law of (theta_AL - theta_0) is approximated by that of (theta_tilde* - theta_OLS); basic
#     bootstrap intervals [theta_AL - q(1-a/2), theta_AL - q(a/2)] and the p-value of H0: theta_j = 0 is the
#     smallest level at which 0 leaves that interval.
# Coefficients are compared in raw units (each bootstrap sample is standardised on itself, as in fit_window).


def _recentered_fit(X, y, pen, coin_idx, n_coins, cfg: EngineConfig, lam: float, a_rel: float,
                    c_raw: np.ndarray, sw=None) -> np.ndarray:
    """A&C (2018) eq. (4) on one sample at fixed lambda and fixed relative ridge penalty; returns raw-unit
    coefficients of all columns of X (unpenalised ones recovered as in lasso._finish)."""
    from .lasso import _Prep, _path
    pr = _Prep(X, y, pen, coin_idx, n_coins, np.arange(len(y)), None, sw)
    p_all = X.shape[1]
    beta = np.zeros(len(pr.pen_cols))
    if len(pr.pen_cols):
        g = np.abs(pr.ridge_coefs([a_rel * pr.n])[0]) ** cfg.gamma
        c_z = c_raw[pr.pen_cols] * pr.sd[pr.pen_cols]          # recentering point in this sample's z-units
        ry = pr.ry - pr.rP @ c_z                                # theta = c + delta: penalise |delta|
        Xt = pr.rP * g
        lmax = np.max(np.abs(Xt.T @ ry)) / pr.n
        grid = lmax * np.logspace(0, np.log10(cfg.lambda_eps), cfg.n_lambda)
        W = _path(Xt, ry, np.r_[grid[grid > lam], lam], cfg, "adaptive_lasso")
        beta = c_z + W[:, -1] * g
    b_u = pr.Bu_y - (pr.Bu_P @ beta if len(beta) else 0.0)
    coef_z = np.zeros(p_all)
    coef_z[pr.unp_cols] = b_u[: len(pr.unp_cols)]
    coef_z[pr.pen_cols] = beta
    return np.where(pr.sd > 0, coef_z / np.where(pr.sd > 0, pr.sd, 1.0), 0.0)


def recentered_bootstrap(X, y, pen, coin_idx, day_idx, n_coins, cfg: EngineConfig, reps: int, seed: int,
                         sw=None, block: Optional[float] = None, n_jobs: int = 1) -> Dict:
    """Original adaptive LASSO fit (lambda and ridge penalty by the engine's CV), full-sample OLS, and the
    recentered bootstrap of A&C (2018). Returns raw-unit estimates, bootstrap deviations, p-values, intervals."""
    from arch.bootstrap import optimal_block_length
    al = fit_window(X, y, pen, coin_idx, day_idx, n_coins, cfg, sw=sw)
    ols = fit_window(X, y, pen, coin_idx, day_idx, n_coins, cfg, estimator="ols", sw=sw)
    th_al, th_ols = al.coef_raw, ols.coef_raw
    c_raw = np.where(pen & (th_al == 0), th_ols, 0.0)
    ud, inv = np.unique(day_idx, return_inverse=True)
    if block is None:
        dm = np.bincount(inv, weights=al.resid) / np.bincount(inv)
        block = max(1.0, float(optimal_block_length(dm)["stationary"].iloc[0]))
    rows_by_day = [np.flatnonzero(inv == i) for i in range(len(ud))]
    draws = [np.concatenate([rows_by_day[i] for i in didx])
             for didx in stationary_block_bootstrap_days(len(ud), block, reps, seed)]

    def one(rows):
        return _recentered_fit(X[rows], y[rows], pen, coin_idx[rows], n_coins, cfg, al.lam, al.ridge_alpha_rel,
                               c_raw, sw=None if sw is None else sw[rows])
    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        TH = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(r) for r in draws))
    else:
        TH = np.array([one(r) for r in draws])
    D = TH - th_ols                                             # approximates the law of theta_AL - theta_0
    lo = (1 + (D <= th_al).sum(0)) / (reps + 1)
    hi = (1 + (D >= th_al).sum(0)) / (reps + 1)
    p = np.minimum(1.0, 2 * np.minimum(lo, hi))
    q_lo, q_hi = np.percentile(D, 2.5, axis=0), np.percentile(D, 97.5, axis=0)
    return dict(theta_al=th_al, theta_ols=th_ols, sd=al.sd, D=D, p=p, ci_lo=th_al - q_hi, ci_hi=th_al - q_lo,
                lam=al.lam, ridge_alpha_rel=al.ridge_alpha_rel, block=block, reps=reps, seed=seed,
                al_coef_z=al.coef_z)


def insample_lasso_recentered(builder: FeatureBuilder, rung: str, cfg: EngineConfig, reps: int = 1999,
                              seed: Optional[int] = None, n_jobs: int = 1, block: Optional[float] = None) -> pd.DataFrame:
    """In-sample adaptive LASSO of one rung with recentered-bootstrap p-values (A&C 2018), Holm within block."""
    from statsmodels.stats.multitest import multipletests
    d = insample_design(builder, rung)
    sd_seed = cfg.seed + 2 if seed is None else seed
    r = recentered_bootstrap(d.X_tr, d.y_tr, d.penalised, d.coin_tr, d.day_tr, builder.C, cfg, reps, sd_seed,
                             sw=d.w_tr, block=block, n_jobs=n_jobs)
    sd = r["sd"]
    out = pd.DataFrame(dict(rung=rung, variable=d.names, block=d.blocks, penalised=d.penalised,
                            beta_z=r["al_coef_z"], beta_raw=r["theta_al"], beta_ols_raw=r["theta_ols"],
                            selected=r["theta_al"] != 0, p_value=r["p"],
                            ci_lo_raw=r["ci_lo"], ci_hi_raw=r["ci_hi"], ci_lo_z=r["ci_lo"] * sd, ci_hi_z=r["ci_hi"] * sd))
    out["p_holm_block"] = np.nan
    for blk, idx in out[out.penalised].groupby("block").groups.items():
        out.loc[idx, "p_holm_block"] = multipletests(out.loc[idx, "p_value"].to_numpy(), method="holm")[1]
    out["lambda"] = r["lam"]
    out["ridge_alpha_rel"] = r["ridge_alpha_rel"]
    out["boot_block_length"] = r["block"]
    out["boot_reps"] = reps
    out["boot_seed"] = sd_seed
    return out
