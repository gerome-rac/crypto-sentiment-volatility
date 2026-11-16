"""Forecast evaluation (test plan Stage 4): losses, Clark-West, Diebold-Mariano with the
Harvey-Leybourne-Newbold correction, Model Confidence Set (arch), OOS R^2, cumulative loss
differences, Giacomini-White, Holm, and per-coin evaluation of the pooled forecasts.

Panel aggregation (cfg.panel_agg = "daymean"): coin-day losses are averaged across the five
coins per target day before CW/DM/MCS/GW, giving T = 885 daily observations whose HAC variance
absorbs the cross-coin correlation (as in the 6 Oct descriptive comparison).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import MCS_EXCLUDE, EngineConfig, hac_default_lags


# ----------------------------------------------------------------------------- losses
def loss_frame(fc: pd.DataFrame, backtransform: str = "lognormal") -> pd.DataFrame:
    F = fc["F_lognormal"] if backtransform == "lognormal" else fc["F_duan"]
    out = fc[["coin", "target", "y", "f_log", "rv_next"]].copy()
    out["se"] = (fc["y"] - fc["f_log"]) ** 2
    r = fc["rv_next"] / F
    out["qlike"] = r - np.log(r) - 1.0
    return out


def qlike(rv, F):
    r = np.asarray(rv) / np.asarray(F)
    return r - np.log(r) - 1.0


def day_mean(s: pd.Series, target: pd.Series) -> np.ndarray:
    return s.groupby(target.values).mean().sort_index().to_numpy()


# ----------------------------------------------------------------------------- HAC
def hac_var(e: np.ndarray, lags: int, divisor: str = "T") -> float:
    """Bartlett long-run variance of a (demeaned inside) series. divisor "T" (Newey-West) or
    "T-l" (mean of the cross products, as in the 6 Oct descriptive script)."""
    e = np.asarray(e, float) - np.mean(e)
    T = len(e)
    v = np.dot(e, e) / T
    for l in range(1, lags + 1):
        cp = np.dot(e[l:], e[:-l])
        v += 2.0 * (1.0 - l / (lags + 1.0)) * (cp / T if divisor == "T" else cp / (T - l))
    return float(v)


def _lags(T, cfg: Optional[EngineConfig], lags=None):
    if lags is not None:
        return int(lags)
    if cfg is not None and cfg.hac_lags is not None:
        return int(cfg.hac_lags)
    return hac_default_lags(T)


# ----------------------------------------------------------------------------- tests
def clark_west(y, f_small, f_big, groups=None, lags=None, cfg=None) -> Dict:
    """Clark & West (2007) MSPE-adjusted test, H0 equal MSPE, H1 the bigger model is better
    (one-sided). Coin-day adjusted differentials are averaged per day when `groups` is given."""
    y, f1, f2 = map(lambda a: np.asarray(a, float), (y, f_small, f_big))
    adj = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    if groups is not None:
        adj = pd.Series(adj).groupby(np.asarray(groups)).mean().sort_index().to_numpy()
    T = len(adj)
    L = _lags(T, cfg, lags)
    m = adj.mean()
    v = hac_var(adj, L)
    if v <= 0 or not np.isfinite(v):
        return dict(cw_stat=np.nan, p_one_sided=1.0, mean_adj=m, T=T, lags=L, degenerate=True)
    t = m / np.sqrt(v / T)
    return dict(cw_stat=float(t), p_one_sided=float(1 - stats.norm.cdf(t)), mean_adj=float(m),
                T=T, lags=L, degenerate=False)


def dm_hln(loss_a, loss_b, h: int = 1, lags=None, cfg=None, variance: Optional[str] = None,
           divisor: str = "T") -> Dict:
    """Diebold-Mariano on d = loss_a - loss_b (positive mean: b is better) with the
    Harvey-Leybourne-Newbold (1997) factor sqrt((T+1-2h+h(h-1)/T)/T) and t(T-1) reference.
    variance "hac": Bartlett with `lags`; "hln": autocovariances up to h-1 only."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    T = len(d)
    var_kind = variance or (cfg.dm_variance if cfg is not None else "hac")
    L = _lags(T, cfg, lags) if var_kind == "hac" else h - 1
    v = hac_var(d, L, divisor)
    if not v > 0:
        return dict(dm=np.nan, dm_hln=np.nan, p_two_sided=1.0, p_b_better=1.0, mean_d=float(d.mean()), T=T, lags=L,
                    degenerate=True)
    dm = d.mean() / np.sqrt(v / T)
    k = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    s = dm * k
    return dict(dm=float(dm), dm_hln=float(s), p_two_sided=float(2 * (1 - stats.t.cdf(abs(s), T - 1))),
                p_b_better=float(1 - stats.t.cdf(s, T - 1)), mean_d=float(d.mean()), T=T, lags=L)


GW_MIN_NONZERO = 30      # fewer non-zero loss differences: no asymptotic test (degenerate)
GW_MAX_COND = 1e12       # condition number above which Omega counts as singular


def giacomini_white(loss_a, loss_b, conditional: bool = True) -> Dict:
    """Giacomini & White (2006) test of equal (conditional) predictive ability, h = 1,
    instruments h_t = (1, d_t) for d_{t+1}; statistic n Zbar' Omega^-1 Zbar ~ chi2(q).
    Valid for rolling (finite) windows only."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    if conditional:
        H = np.column_stack([np.ones(len(d) - 1), d[:-1]])
        Z = H * d[1:, None]
    else:
        Z = d[:, None]
    n, q = Z.shape
    n_nonzero = int(np.sum(d != 0))
    if n_nonzero == 0:
        return dict(gw_stat=np.nan, df=q, p=1.0, mean_d=0.0, degenerate=True, n_nonzero=0)
    zbar = Z.mean(0)
    Om = Z.T @ Z / n
    # 7 Oct 2026 (lead, reviewer finding e4af3f9a): no pseudo-inverse. A rank-deficient or
    # ill-conditioned Omega, or loss differences that are non-zero on too few days, is reported
    # as degenerate (statistic and p-value NaN) instead of being silently regularised.
    if n_nonzero < GW_MIN_NONZERO or np.linalg.matrix_rank(Om) < q or np.linalg.cond(Om) > GW_MAX_COND:
        return dict(gw_stat=np.nan, df=q, p=np.nan, mean_d=float(d.mean()), degenerate=True, n_nonzero=n_nonzero)
    stat = float(n * zbar @ np.linalg.solve(Om, zbar))
    return dict(gw_stat=stat, df=q, p=float(1 - stats.chi2.cdf(stat, q)), mean_d=float(d.mean()), degenerate=False,
                n_nonzero=n_nonzero)


def holm(pvals: Sequence[float], alpha: float = 0.05) -> Dict:
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(pvals, float)
    rej, padj, _, _ = multipletests(p, alpha=alpha, method="holm")
    return dict(p=p.tolist(), p_holm=padj.tolist(), reject=rej.tolist())


def mcs(losses: pd.DataFrame, alpha: float, reps: int = 10000, block_size=None,
        method: str = "R", seed: int = 0) -> pd.DataFrame:
    """Hansen-Lunde-Nason Model Confidence Set via arch.bootstrap.MCS (stationary bootstrap)."""
    from arch.bootstrap import MCS, optimal_block_length
    if block_size is None:
        base = losses.iloc[:, 0].to_numpy()
        diffs = losses.to_numpy() - base[:, None]
        cands = [optimal_block_length(diffs[:, j])["stationary"].iloc[0]
                 for j in range(1, diffs.shape[1]) if np.std(diffs[:, j]) > 0]
        block_size = int(max(1, np.ceil(np.nanmax(cands)))) if cands else 1
    m = MCS(losses, size=alpha, reps=reps, block_size=block_size, method=method,
            bootstrap="stationary", seed=seed)
    m.compute()
    pv = m.pvalues.iloc[:, 0]
    out = pd.DataFrame({"model": pv.index, "mcs_pvalue": pv.values})
    out["in_mcs"] = out.model.isin(m.included)
    out["alpha"] = alpha
    out["block_size"] = block_size
    out["mean_loss"] = out.model.map(losses.mean())
    return out


def oos_r2(se_model, se_bench) -> float:
    return float(1 - np.sum(se_model) / np.sum(se_bench))


# ----------------------------------------------------------------------------- ladder
def evaluate(forecasts: Dict[str, pd.DataFrame], cfg: EngineConfig, comparisons, bench: str = "R0",
             with_gw: bool = True) -> Dict[str, pd.DataFrame]:
    """All Stage-4 tables for a set of rungs with identical coin-days."""
    L = {k: loss_frame(v, cfg.qlike_backtransform).sort_values(["target", "coin"]).reset_index(drop=True)
         for k, v in forecasts.items()}
    keys = list(L)
    ref = L[keys[0]][["coin", "target"]]
    for k in keys:
        assert L[k][["coin", "target"]].equals(ref), f"{k}: coin-days differ from {keys[0]}"
    tgt = ref["target"]
    D = {k: {lo: day_mean(L[k][lo], tgt) for lo in ("se", "qlike")} for k in keys}
    T = len(next(iter(D.values()))["se"])
    out: Dict[str, pd.DataFrame] = {}
    rows = []
    for k in keys:
        r = dict(model=k, n=len(L[k]), T_days=T, MSE_log=L[k].se.mean(), QLIKE=L[k].qlike.mean())
        if bench in L:
            r["R2_oos_vs_" + bench] = oos_r2(L[k].se, L[bench].se)
        rows.append(r)
    out["loss_summary"] = pd.DataFrame(rows)
    rows = []
    for small, big, lab in comparisons:
        if small not in L or big not in L:
            continue
        cw = clark_west(L[big].y, forecasts_val(L[small]), forecasts_val(L[big]), groups=tgt, cfg=cfg)
        dms = dm_hln(D[small]["se"], D[big]["se"], cfg=cfg)
        dmq = dm_hln(D[small]["qlike"], D[big]["qlike"], cfg=cfg)
        r = dict(test=lab, small=small, big=big, cw_stat=cw["cw_stat"], cw_p=cw["p_one_sided"],
                 cw_degenerate=cw["degenerate"], hac_lags=cw["lags"],
                 dm_mse=dms["dm_hln"], dm_mse_p2=dms["p_two_sided"], dm_mse_p_big_better=dms["p_b_better"],
                 dm_qlike=dmq["dm_hln"], dm_qlike_p2=dmq["p_two_sided"], dm_qlike_p_big_better=dmq["p_b_better"],
                 r2_oos_big_vs_small=oos_r2(L[big].se, L[small].se))
        if with_gw and cfg.window != "expanding":
            g = giacomini_white(D[small]["se"], D[big]["se"], cfg.gw_conditional)
            r.update(gw_mse_stat=g["gw_stat"], gw_mse_p=g["p"], gw_mse_degenerate=g["degenerate"])
            g = giacomini_white(D[small]["qlike"], D[big]["qlike"], cfg.gw_conditional)
            r.update(gw_qlike_stat=g["gw_stat"], gw_qlike_p=g["p"], gw_qlike_degenerate=g["degenerate"])
        rows.append(r)
    out["nested_tests"] = pd.DataFrame(rows)
    mc = []
    for lo in ("se", "qlike"):
        Lm = pd.DataFrame({k: D[k][lo] for k in keys if k not in MCS_EXCLUDE})
        for a in cfg.mcs_alphas:
            t = mcs(Lm, a, cfg.mcs_reps, cfg.mcs_block_size, cfg.mcs_method, cfg.seed)
            t.insert(0, "loss", "MSE_log" if lo == "se" else "QLIKE")
            mc.append(t)
    out["mcs"] = pd.concat(mc, ignore_index=True)
    # Deviation 4 of the D8 note (7 Oct 2026): the H2 condition uses the 10% MCS of {R1, R2} only
    if "R1" in keys and "R2" in keys:
        mh = []
        for lo in ("se", "qlike"):
            Lm2 = pd.DataFrame({k: D[k][lo] for k in ("R1", "R2")})
            t = mcs(Lm2, 0.10, cfg.mcs_reps, cfg.mcs_block_size, cfg.mcs_method, cfg.seed)
            t.insert(0, "loss", "MSE_log" if lo == "se" else "QLIKE")
            mh.append(t)
        out["mcs_h2"] = pd.concat(mh, ignore_index=True)
    if bench in L:
        days = np.sort(tgt.unique())
        cum = pd.DataFrame({"target": days})
        for k in keys:
            if k == bench:
                continue
            cum[f"cum_dse_{bench}_minus_{k}"] = np.cumsum(D[bench]["se"] - D[k]["se"])
            cum[f"cum_dqlike_{bench}_minus_{k}"] = np.cumsum(D[bench]["qlike"] - D[k]["qlike"])
        out["cum_loss_diff"] = cum
    # per-coin evaluation of the pooled forecasts (test 4.13)
    pc = []
    for coin in ref.coin.unique():
        m = (ref.coin == coin).to_numpy()
        for small, big, lab in comparisons:
            if small not in L or big not in L:
                continue
            a, b = L[small][m], L[big][m]
            cw = clark_west(b.y, a.f_log, b.f_log, cfg=cfg)
            dms = dm_hln(a.se, b.se, cfg=cfg)
            dmq = dm_hln(a.qlike, b.qlike, cfg=cfg)
            pc.append(dict(coin=coin, test=lab, small=small, big=big, MSE_small=a.se.mean(), MSE_big=b.se.mean(),
                           QLIKE_small=a.qlike.mean(), QLIKE_big=b.qlike.mean(),
                           r2_oos=oos_r2(b.se, a.se), cw_stat=cw["cw_stat"], cw_p=cw["p_one_sided"],
                           dm_mse=dms["dm_hln"], dm_mse_p2=dms["p_two_sided"],
                           dm_qlike=dmq["dm_hln"], dm_qlike_p2=dmq["p_two_sided"]))
    if pc:
        t = pd.DataFrame(pc)
        t["cw_p_holm_within_test"] = t.groupby("test").cw_p.transform(lambda s: holm(s.values, cfg.holm_alpha)["p_holm"])
        out["per_coin"] = t
    return out


def forecasts_val(lf: pd.DataFrame) -> np.ndarray:
    return lf["f_log"].to_numpy()


def confirmatory(nested: pd.DataFrame, mcs_tab: pd.DataFrame, p_h4: Optional[float], alpha: float = 0.05,
                 mcs_h2: Optional[pd.DataFrame] = None, h4_status: Optional[str] = None,
                 h4_reason: str = "") -> pd.DataFrame:
    """D8 note section 4: Holm over the four confirmatory p-values and the decision rules.
    H2 as amended by Deviation 4 (7 Oct 2026): R1 in the 10% MCS of {R1, R2} on both losses, and no
    significant one-sided Diebold-Mariano improvement of R2 over R1 (5%) on either loss.
    h4_status (28 Oct 2026): "tested" (default when p_h4 is given; p_h4 must then be finite, so a missing H4 never
    shrinks the family), "not applicable" (the arm's R3 has no fear/anger terms, e.g. A60) or "not computed" (no
    in-sample stage; default when p_h4 is None). In the last two cases the H4 row says so and Holm runs over H1-H3."""
    if h4_status is None:
        h4_status = "tested" if p_h4 is not None else "not computed"
    if h4_status not in ("tested", "not applicable", "not computed"):
        raise ValueError(h4_status)
    if h4_status == "tested" and not (p_h4 is not None and np.isfinite(p_h4)):
        raise ValueError("H4 was tested but has no finite p-value; the Holm family would shrink silently")
    if h4_status != "tested":
        p_h4 = None
    pm = {r.test: r.cw_p for r in nested.itertuples()}
    tests = ["H1", "H2", "H3", "H4"]
    p = [pm.get("H1", np.nan), pm.get("H2", np.nan), pm.get("H3", np.nan), p_h4 if p_h4 is not None else np.nan]
    avail = [i for i, x in enumerate(p) if np.isfinite(x)]
    padj = [np.nan] * 4
    if avail:
        h = holm([p[i] for i in avail], alpha)
        for j, i in enumerate(avail):
            padj[i] = h["p_holm"][j]
    rows = []
    for i, t in enumerate(tests):
        r = dict(hypothesis=t, p=p[i], p_holm=padj[i], holm_family_size=len(avail))
        nr = nested[nested.test == t]
        if t in ("H1", "H3"):
            r["supported_mse"] = bool(np.isfinite(padj[i]) and padj[i] < alpha)
        if t == "H2":
            def _in10(tab):
                sel = tab[(tab.alpha == 0.10) & (tab.model == "R1")] if tab is not None and len(tab) else None
                return bool(sel is not None and len(sel) == 2 and sel.in_mcs.all())
            in10 = _in10(mcs_h2)
            dm_ins = bool(len(nr) and nr.dm_mse_p_big_better.iloc[0] >= alpha and nr.dm_qlike_p_big_better.iloc[0] >= alpha)
            r["supported_mse"] = bool(np.isfinite(padj[i]) and padj[i] >= alpha and in10 and dm_ins)
            r["R1_in_MCS10_of_R1R2_both_losses"] = in10
            r["DM_no_improvement_of_R2_both"] = dm_ins
            r["R1_in_MCS10_full_set_both_losses"] = _in10(mcs_tab)          # reported, not deciding
            r["DM_two_sided_insignificant_both"] = bool(len(nr) and nr.dm_mse_p2.iloc[0] >= alpha and nr.dm_qlike_p2.iloc[0] >= alpha)
        if t == "H4":
            r["supported_mse"] = bool(np.isfinite(padj[i]) and padj[i] < alpha)
        if t in ("H1", "H3") and len(nr):
            # QLIKE robustness rule: DM on QLIKE significantly the other way (small model better)
            q_against = bool(nr.dm_qlike.iloc[0] < 0 and nr.dm_qlike_p2.iloc[0] < alpha)
            r["qlike_points_against"] = q_against
            r["verdict"] = ("supported on MSE, not robust to the loss" if r["supported_mse"] and q_against
                            else ("supported" if r["supported_mse"] else "not supported"))
        else:
            r["verdict"] = "supported" if r.get("supported_mse") else "not supported"
        if t == "H4" and h4_status != "tested":
            fam = "H1-H3" if avail == [0, 1, 2] else (", ".join(tests[k] for k in avail) or "empty")
            r["supported_mse"] = None
            r["verdict"] = (f"not applicable ({h4_reason}); Holm family {fam}" if h4_status == "not applicable"
                            else f"not computed (no in-sample stage); Holm family {fam}")
        rows.append(r)
    return pd.DataFrame(rows)
