"""HAR design builder with every data-driven transform computed inside the estimation window.

Per window (training origin days `tr`, forecast origin days `te`):
  1. impute flagged days (Twitter zero-capture days, flag column `impute_flag`) with the
     window average of the series (per coin; market series are equal across coins);
  2. optional A10 composition level adjustment (hook, off by default);
  3. HAR form: averages of the daily values over the trailing 1/7/30 days over the available
     days (no minimum count; MASTERPLAN 6 Oct (8)), then the transform (log, log1p or level);
     transforms 'log_daily' / 'log1p_daily' instead take the log of each day first and average the
     daily logs (attention: 'log(1 + posts) per coin-day ... in HAR form', MASTERPLAN 6 Oct (4));
  4. day-t log change for the attention variables flagged `dlog`;
  5. centred squares: (x_t - m_w)^2 with m_w the window mean of x, then the 1/7/30 averages;
  6. controls: day-t value only.
Z-scoring happens in the fitting functions (model.lasso), on the rows they estimate on.
Rows are ordered day-major (all coins of a day together).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import COINS, HAR_LAGS, RUNG_BLOCKS, UNPENALISED_BLOCKS, EngineConfig, rung_blocks_for
from .panel import VarSpec


def rolling_nanmean(x: np.ndarray, k: int) -> np.ndarray:
    """Trailing k-day mean over the non-NaN days (truncated at the sample start); NaN if none.
    x has shape (n_coins, n_days)."""
    ok = ~np.isnan(x)
    x0 = np.where(ok, x, 0.0)
    cs = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x0, axis=1)], axis=1)
    cn = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(ok, axis=1)], axis=1)
    T = x.shape[1]
    hi = np.arange(1, T + 1)
    lo = np.maximum(0, hi - k)
    s = cs[:, hi] - cs[:, lo]
    n = cn[:, hi] - cn[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _tf(x: np.ndarray, transform: str) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        if transform == "log":
            return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)
        if transform == "log1p":
            return np.where(x > -1, np.log1p(np.where(x > -1, x, 0.0)), np.nan)
        return x


@dataclass
class Design:
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    names: List[str]
    blocks: List[str]
    base_var: List[str]
    kind: List[str]               # "h1","h7","h30","dlog","sq_h1",...,"day"
    penalised: np.ndarray         # bool per column of X
    coin_tr: np.ndarray           # coin index per training row
    day_tr: np.ndarray            # day index per training row
    coin_te: np.ndarray
    day_te: np.ndarray
    w_tr: Optional[np.ndarray] = None
    info: Dict = field(default_factory=dict)
    te_ok: Optional[np.ndarray] = None   # 28 Oct 2026: test rows that may be forecast (None = all; drop rule only)


class FeatureBuilder:
    def __init__(self, grid: pd.DataFrame, specs: List[VarSpec], cfg: EngineConfig,
                 coins: Sequence[str] = COINS):
        self.cfg = cfg
        self.coins = list(coins)
        self.days = grid.index.get_level_values("day").unique().sort_values()
        self.C, self.T = len(self.coins), len(self.days)
        self.specs = {v.name: v for v in specs}
        self.raw: Dict[str, np.ndarray] = {}
        self.flags: Dict[str, np.ndarray] = {}
        for v in specs:
            self.raw[v.name] = self._arr(grid[v.name]).astype(float)
            if v.impute_flag:
                self.flags[v.impute_flag] = self._arr(grid[v.impute_flag]).astype(bool)
        for extra in (cfg.composition_flag, cfg.composition_full_flag):
            if extra in grid.columns and extra not in self.flags:
                self.flags[extra] = self._arr(grid[extra]).astype(bool)
        self.raw2p = {n[:-3]: self._arr(grid[n]).astype(float) for n in grid.columns if n.endswith("_2p")}
        self.weights = self._arr(grid[cfg.weight_col]).astype(float) if cfg.weight_col else None
        rv = self.raw.get("rv")
        if rv is None:
            raise KeyError("volatility column 'rv' is required (target log rv_{t+1})")
        # 28 Oct 2026, drop rule (arm A53; off for A00): flagged coin-days leave the sample. The daily values of the
        # drop_blocks variables (price-based volatility measures) and the target are missing on them, so the HAR
        # 7/30-day averages run over the observed days; row_ok marks origin days that may form a row.
        self.drop = None
        self.row_ok = None
        if getattr(cfg, "drop_days_flag", None):
            flag = cfg.drop_days_flag
            if flag not in grid.columns:
                raise KeyError(f"drop_days_flag {flag!r} is not a panel column")
            self.drop = self._arr(grid[flag]).astype(bool)
            for v in specs:
                if v.block in tuple(cfg.drop_blocks):
                    self.raw[v.name] = np.where(self.drop, np.nan, self.raw[v.name])
            rv = np.where(self.drop, np.nan, rv)
            ok = ~np.isnan(rv)
            assert np.all(rv[ok] > 0) and np.array_equal(~ok, self.drop), "rv must be positive on every kept coin-day"
            self.row_ok = ~self.drop
        else:
            assert np.all(rv > 0), "rv must be positive"
        self.y = np.full_like(rv, np.nan)
        self.y[:, :-1] = np.log(rv[:, 1:])         # target: next-day log RV
        self.rv_next = np.full_like(rv, np.nan)
        self.rv_next[:, :-1] = rv[:, 1:]
        # 28 Oct 2026, evaluation target (arm A62; "rv" = A00): the losses use the next-day value of this column,
        # the fit keeps log rv_{t+1}.
        self.y_eval = self.rv_next_eval = None
        ev_col = getattr(cfg, "eval_target", "rv")
        if ev_col != "rv":
            if ev_col not in grid.columns:
                raise KeyError(f"eval_target {ev_col!r} is not a panel column (pass its table with --extra-table)")
            ev = self._arr(grid[ev_col]).astype(float)
            if self.drop is not None:
                ev = np.where(self.drop, np.nan, ev)
            okv = ~np.isnan(ev) if self.drop is not None else np.ones(ev.shape, bool)
            assert np.all(np.isfinite(ev[okv]) & (ev[okv] > 0)), f"eval_target {ev_col} must be finite and positive"
            self.rv_next_eval = np.full_like(ev, np.nan)
            self.rv_next_eval[:, :-1] = ev[:, 1:]
            self.y_eval = np.full_like(ev, np.nan)
            self.y_eval[:, :-1] = np.log(ev[:, 1:])

    def _arr(self, s: pd.Series) -> np.ndarray:
        return s.unstack("coin").reindex(columns=self.coins).reindex(self.days).to_numpy().T

    def day_index(self, day) -> int:
        return int(self.days.get_loc(pd.Timestamp(day)))

    # ------------------------------------------------------------------ per-window features
    def columns_for(self, rung: str, var_subset: Optional[Sequence[str]] = None) -> List[VarSpec]:
        blocks = rung_blocks_for(self.cfg)[rung]
        vs = [v for v in self.specs.values() if v.block in blocks]
        if var_subset is not None:
            vs = [v for v in vs if v.name in var_subset or v.block in UNPENALISED_BLOCKS]
        return vs

    def _window_series(self, v: VarSpec, tr_days: np.ndarray, info: Dict) -> np.ndarray:
        x = self.raw[v.name].copy()
        if v.impute_flag:
            a = self.flags[v.impute_flag]
            xa = np.where(a, np.nan, x)[:, tr_days]
            if self.cfg.impute_stat == "count_mean":
                val = np.nanmean(xa, axis=1)
            elif self.cfg.impute_stat == "log1p_mean":
                val = np.expm1(np.nanmean(np.log1p(xa), axis=1))
            else:
                raise ValueError(self.cfg.impute_stat)
            x = np.where(a, val[:, None], x)
            info.setdefault("impute_value", {})[v.name] = val.tolist()
            info.setdefault("n_imputed_days_in_window", {})[v.name] = int(a[:, tr_days].sum())
        if self.cfg.composition_adjustment == "level_shift" and v.name in self.raw2p:
            # A10 hook: on flagged (Twitter-absent) days shift the 2-platform index by the in-window mean
            # 3- minus 2-platform difference on fully captured days. Literal reading; OP-A10.
            full = self.flags[self.cfg.composition_full_flag][:, tr_days]
            diff = (self.raw[v.name] - self.raw2p[v.name])[:, tr_days]
            shift = float(np.nanmean(np.where(full, diff, np.nan)))
            x = np.where(self.flags[self.cfg.composition_flag], x + shift, x)
            info.setdefault("composition_shift", {})[v.name] = shift
        elif self.cfg.composition_adjustment not in (None, "level_shift"):
            raise ValueError(self.cfg.composition_adjustment)
        return x

    def features(self, rung: str, tr_days: np.ndarray, info: Dict,
                 var_subset=None) -> tuple:
        """Return (F list of (C,T) arrays, names, blocks, base, kinds) for one window."""
        F, names, blocks, base, kinds = [], [], [], [], []
        for v in self.columns_for(rung, var_subset):
            x = self._window_series(v, tr_days, info)
            tf = v.transform
            daily_log = tf in ("log_daily", "log1p_daily")
            if daily_log:                      # log each day first, then average the daily logs
                x = _tf(x, tf[: -len("_daily")])
                tf = "level"
            if v.har:
                for k in HAR_LAGS:
                    F.append(_tf(rolling_nanmean(x, k), tf)); kinds.append(f"h{k}")
                    names.append(f"{v.name}_h{k}"); blocks.append(v.block); base.append(v.name)
            else:
                F.append(_tf(x, tf)); kinds.append("day")
                names.append(v.name); blocks.append(v.block); base.append(v.name)
            if v.dlog:
                if daily_log:
                    lx = x
                else:
                    lx = _tf(x, "log" if v.transform == "log" else "log1p") if v.transform in ("log", "log1p") else x
                d = np.full_like(lx, np.nan)
                d[:, 1:] = lx[:, 1:] - lx[:, :-1]
                F.append(d); kinds.append("dlog")
                names.append(f"{v.name}_dlog"); blocks.append(v.block); base.append(v.name)
            if v.csq:
                xt = x[:, tr_days]
                if self.cfg.centre_scope == "pooled":
                    m = np.nanmean(xt)
                elif self.cfg.centre_scope == "per_coin":
                    m = np.nanmean(xt, axis=1)[:, None]
                else:
                    raise ValueError(self.cfg.centre_scope)
                info.setdefault("centre", {})[v.name] = np.atleast_1d(m).ravel().tolist()
                sq = (x - m) ** 2
                for k in HAR_LAGS:
                    F.append(rolling_nanmean(sq, k)); kinds.append(f"sq_h{k}")
                    names.append(f"{v.name}_sq_h{k}"); blocks.append(v.block); base.append(v.name)
        return F, names, blocks, base, kinds

    def rows(self, days_idx: np.ndarray, coins_idx: Optional[np.ndarray] = None):
        """Day-major row index (coin, day) for the given origin days."""
        ci = np.arange(self.C) if coins_idx is None else np.asarray(coins_idx)
        dd = np.repeat(np.asarray(days_idx), len(ci))
        cc = np.tile(ci, len(days_idx))
        return cc, dd

    def design(self, rung: str, tr_days: np.ndarray, te_days: np.ndarray,
               coins_idx: Optional[np.ndarray] = None, var_subset=None) -> Design:
        info: Dict = {}
        F, names, blocks, base, kinds = self.features(rung, tr_days, info, var_subset)
        ctr, dtr = self.rows(tr_days, coins_idx)
        cte, dte = self.rows(te_days, coins_idx)
        X_tr = np.column_stack([f[ctr, dtr] for f in F]) if F else np.zeros((len(ctr), 0))
        X_te = np.column_stack([f[cte, dte] for f in F]) if F else np.zeros((len(cte), 0))
        y_tr, y_te = self.y[ctr, dtr], self.y[cte, dte]
        keep = ~np.isnan(y_tr)
        te_ok = None
        if self.row_ok is not None:              # drop rule (A53): no row with a flagged origin or target day
            keep &= self.row_ok[ctr, dtr]
            info["n_rows_dropped_flag"] = int((~keep).sum())
            te_ok = self.row_ok[cte, dte] & ~np.isnan(y_te)
        # NaN regressors (e.g. a within-day SD on a day with < 2 posts): OP-NAN
        nan_tr = np.isnan(X_tr)
        if te_ok is None:
            info["n_nan_filled_train"] = int(nan_tr.sum())
            info["n_nan_filled_test"] = int(np.isnan(X_te).sum())
        else:
            info["n_nan_filled_train"] = int(nan_tr[keep].sum())
            info["n_nan_filled_test"] = int(np.isnan(X_te[te_ok]).sum())
        if self.cfg.nan_policy == "drop":
            keep &= ~nan_tr.any(axis=1)
        elif self.cfg.nan_policy != "window_mean":
            raise ValueError(self.cfg.nan_policy)
        X_tr, y_tr, ctr, dtr = X_tr[keep], y_tr[keep], ctr[keep], dtr[keep]
        if X_tr.shape[1]:
            mu = np.nanmean(X_tr, axis=0)
            mu = np.where(np.isnan(mu), 0.0, mu)
            X_tr = np.where(np.isnan(X_tr), mu, X_tr)
            X_te = np.where(np.isnan(X_te), mu, X_te)
        assert np.isfinite(X_tr).all() and np.isfinite(X_te).all(), "non-finite regressor after the NaN policy"
        pen = np.array([b not in UNPENALISED_BLOCKS for b in blocks], dtype=bool)
        w = self.weights[ctr, dtr] if self.weights is not None else None
        return Design(X_tr, y_tr, X_te, y_te, names, blocks, base, kinds, pen,
                      ctr, dtr, cte, dte, w, info, te_ok)
