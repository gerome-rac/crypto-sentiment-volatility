"""Pooled OLS / adaptive LASSO / adaptive elastic net with unpenalised HAR terms and coin FE.

Objective (sklearn scaling): (1/2n)||y - U b - P beta||^2 + lam * sum_j |beta_j| / g_j,
with U = [z-scored HAR terms, coin dummies] unpenalised, P = z-scored penalised columns,
g_j = |beta_ridge_j|^gamma (Zou 2006; ridge first stage). The unpenalised part is partialled
out exactly (Frisch-Waugh): b = U^+ (y - P beta), so the LASSO runs on M_U y and M_U P.
lambda (and the first-stage ridge penalty) are chosen by time-series CV over whole days:
forward-chaining (default) or contiguous blocked folds; never random K-fold (test plan 3.12).
Standardisation, partialling-out and the adaptive weights are recomputed inside every fold.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import enet_path, lasso_path

from .config import EngineConfig


@dataclass
class Fit:
    coef_z: np.ndarray            # per column of X, z-units (0 for unselected / constant columns)
    fe: np.ndarray                # coin effects in the z-scored parametrisation
    mu: np.ndarray
    sd: np.ndarray
    resid: np.ndarray
    sigma2: float                 # in-window residual variance (ddof 0)
    duan: float                   # in-window mean of exp(resid) (Duan 1983 smearing factor)
    lam: float = np.nan
    ridge_alpha_rel: float = np.nan
    cv: Dict = field(default_factory=dict)
    estimator: str = "ols"

    @property
    def coef_raw(self) -> np.ndarray:
        return np.where(self.sd > 0, self.coef_z / np.where(self.sd > 0, self.sd, 1.0), 0.0)

    def predict(self, X: np.ndarray, coin_idx: np.ndarray) -> np.ndarray:
        Z = _apply_std(X, self.mu, self.sd)
        return Z @ self.coef_z + self.fe[coin_idx]


def _std(X: np.ndarray):
    mu = X.mean(axis=0) if X.shape[1] else np.zeros(0)
    sd = X.std(axis=0) if X.shape[1] else np.zeros(0)
    sd = np.where(sd > 1e-12 * np.maximum(1.0, np.abs(mu)), sd, 0.0)
    return mu, sd


def _apply_std(X, mu, sd):
    return np.where(sd > 0, (X - mu) / np.where(sd > 0, sd, 1.0), 0.0)


def fe_dummies(coin_idx: np.ndarray, n_coins: int) -> np.ndarray:
    D = np.zeros((len(coin_idx), n_coins))
    D[np.arange(len(coin_idx)), coin_idx] = 1.0
    return D


def day_folds(day_idx: np.ndarray, scheme: str, K: int, gap: int = 0):
    """Row-index folds over whole days. forward: K+1 contiguous blocks, fold k trains on blocks
    < k and validates on block k. blocked: K contiguous blocks, each held out once, `gap` days
    removed on both sides of the validation block."""
    ud = np.unique(day_idx)
    if scheme == "forward":
        parts = np.array_split(ud, K + 1)
        out = []
        for k in range(1, K + 1):
            tr_days = np.concatenate(parts[:k])
            if gap:
                tr_days = tr_days[tr_days < parts[k][0] - gap]
            out.append((np.flatnonzero(np.isin(day_idx, tr_days)), np.flatnonzero(np.isin(day_idx, parts[k]))))
        return out
    if scheme == "blocked":
        parts = np.array_split(ud, K)
        out = []
        for k in range(K):
            v = parts[k]
            tr_days = ud[(ud < v[0] - gap) | (ud > v[-1] + gap)]
            out.append((np.flatnonzero(np.isin(day_idx, tr_days)), np.flatnonzero(np.isin(day_idx, v))))
        return out
    raise ValueError(f"cv_scheme {scheme!r}: only 'forward' or 'blocked' (random K-fold is not allowed)")


class _Prep:
    """Standardise on the training rows, partial out U, and carry validation rows along."""

    def __init__(self, X, y, pen, coin_idx, n_coins, tr, va=None, sw=None):
        mu, sd = _std(X[tr])
        Z = _apply_std(X, mu, sd)
        D = fe_dummies(coin_idx, n_coins)
        keepc = sd > 0
        self.mu, self.sd = mu, sd
        self.pen_cols = np.flatnonzero(pen & keepc)
        self.unp_cols = np.flatnonzero(~pen & keepc)
        U = np.column_stack([Z[:, self.unp_cols], D])
        P = Z[:, self.pen_cols]
        r = np.sqrt(sw) if sw is not None else np.ones(len(y))
        Ut, Pt, yt = U[tr] * r[tr, None], P[tr] * r[tr, None], y[tr] * r[tr]
        B, *_ = np.linalg.lstsq(Ut, np.column_stack([yt, Pt]), rcond=None)
        self.Bu_y, self.Bu_P = B[:, 0], B[:, 1:]
        self.ry = yt - Ut @ self.Bu_y
        self.rP = Pt - Ut @ self.Bu_P
        self.n = len(tr)
        if va is not None:
            self.base_v = U[va] @ self.Bu_y
            self.rPv = P[va] - U[va] @ self.Bu_P
            self.yv = y[va]
        self.U, self.P, self.y, self.tr = U, P, y, tr

    def ridge_coefs(self, alpha_abs_list):
        Us, S, Vt = np.linalg.svd(self.rP, full_matrices=False)
        uty = Us.T @ self.ry
        return [Vt.T @ (S / (S ** 2 + a) * uty) for a in alpha_abs_list]


def _path(Xt, y, alphas, cfg: EngineConfig, estimator: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if estimator == "enet":
            _, W, _ = enet_path(Xt, y, l1_ratio=cfg.enet_l1_ratio, alphas=alphas, max_iter=5000)
        else:
            _, W, _ = lasso_path(Xt, y, alphas=alphas, max_iter=5000)
    return W


def fit_window(X: np.ndarray, y: np.ndarray, pen: np.ndarray, coin_idx: np.ndarray,
               day_idx: np.ndarray, n_coins: int, cfg: EngineConfig,
               estimator: Optional[str] = None, sw: Optional[np.ndarray] = None,
               fixed_lambda: Optional[float] = None, fixed_ridge_rel: Optional[float] = None) -> Fit:
    est = estimator or cfg.estimator
    allrows = np.arange(len(y))
    full = _Prep(X, y, pen, coin_idx, n_coins, allrows, None, sw)
    p_all = X.shape[1]
    if est == "ols" or len(full.pen_cols) == 0:
        beta = np.zeros(0)
        if len(full.pen_cols):
            beta, *_ = np.linalg.lstsq(full.rP, full.ry, rcond=None)
        return _finish(full, beta, p_all, n_coins, coin_idx, X, y, est if len(full.pen_cols) else "ols", {})

    folds = day_folds(day_idx, cfg.cv_scheme, cfg.cv_folds, cfg.cv_gap_days)
    preps = [_Prep(X, y, pen, coin_idx, n_coins, tr, va, sw) for tr, va in folds]
    cv: Dict = {}
    # first stage: ridge penalty (relative to n) by the same CV
    if fixed_ridge_rel is not None:
        a_rel = fixed_ridge_rel
    elif cfg.ridge_alpha_cv:
        grid = np.array(cfg.ridge_grid_rel)
        sse = np.zeros(len(grid)); nv = 0
        for pr in preps:
            for i, b in enumerate(pr.ridge_coefs(grid * pr.n)):
                sse[i] += np.sum((pr.yv - pr.base_v - pr.rPv @ b) ** 2)
            nv += len(pr.yv)
        a_rel = float(grid[int(np.argmin(sse))])
        cv["ridge_cv_mse"] = (sse / nv).tolist()
    else:
        a_rel = cfg.ridge_alpha_fixed_rel
    g_full = np.abs(full.ridge_coefs([a_rel * full.n])[0]) ** cfg.gamma
    Xt = full.rP * g_full
    lmax = np.max(np.abs(Xt.T @ full.ry)) / full.n
    if est == "enet":
        lmax /= cfg.enet_l1_ratio
    lam_grid = lmax * np.logspace(0, np.log10(cfg.lambda_eps), cfg.n_lambda)
    if fixed_lambda is None:
        mse_f = np.zeros((len(preps), len(lam_grid))); nvs = np.zeros(len(preps))
        for f, pr in enumerate(preps):
            g = np.abs(pr.ridge_coefs([a_rel * pr.n])[0]) ** cfg.gamma
            # fold-level columns may differ (constant columns dropped); align on the full prep's columns
            W = _path(pr.rP * g, pr.ry, lam_grid, cfg, est) * g[:, None]
            pred = pr.base_v[:, None] + pr.rPv @ W
            mse_f[f] = np.mean((pr.yv[:, None] - pred) ** 2, axis=0)
            nvs[f] = len(pr.yv)
        mse = (mse_f * nvs[:, None]).sum(0) / nvs.sum()
        i_min = int(np.argmin(mse))
        if cfg.cv_rule == "1se":
            se = mse_f.std(axis=0, ddof=1) / np.sqrt(len(preps))
            i_sel = int(np.flatnonzero(mse <= mse[i_min] + se[i_min])[0])
        else:
            i_sel = i_min
        lam = float(lam_grid[i_sel])
        cv.update(lam_grid=lam_grid.tolist(), cv_mse=mse.tolist(), i_sel=i_sel, at_edge=bool(i_sel in (0, len(lam_grid) - 1)))
        W = _path(Xt, full.ry, lam_grid[: i_sel + 1], cfg, est)
        beta = W[:, -1] * g_full
    else:
        lam = float(fixed_lambda)
        grid_to = lam_grid[lam_grid > lam]
        W = _path(Xt, full.ry, np.r_[grid_to, lam], cfg, est)
        beta = W[:, -1] * g_full
    fit = _finish(full, beta, p_all, n_coins, coin_idx, X, y, est, cv)
    fit.lam, fit.ridge_alpha_rel = lam, a_rel
    return fit


def _finish(pr: _Prep, beta, p_all, n_coins, coin_idx, X, y, est, cv) -> Fit:
    b_u = pr.Bu_y - (pr.Bu_P @ beta if len(beta) else 0.0)
    coef_z = np.zeros(p_all)
    coef_z[pr.unp_cols] = b_u[: len(pr.unp_cols)]
    coef_z[pr.pen_cols] = beta
    fe = b_u[len(pr.unp_cols):]
    fit = Fit(coef_z, fe, pr.mu, pr.sd, np.zeros(0), np.nan, np.nan, cv=cv, estimator=est)
    resid = y - fit.predict(X, coin_idx)
    fit.resid = resid
    fit.sigma2 = float(np.mean(resid ** 2))
    fit.duan = float(np.mean(np.exp(resid)))
    return fit
