"""Panel loading: the four daily tables (or one merged har_panel.parquet) -> one complete
coin x UTC-day grid of RAW daily ingredients. Nothing data-driven is computed here; every
in-window step (centring, imputation, z-scoring, A10) happens in model.har per window.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import COINS, N_DAYS, SAMPLE_FIRST_DAY, SAMPLE_LAST_DAY

TABLES = ("volatility_daily", "text_daily", "attention_daily", "controls_daily")
SPEC_PATH = os.path.join(os.path.dirname(__file__), "variable_spec_default.csv")


@dataclass
class VarSpec:
    name: str
    table: str
    block: str
    scope: str          # "coin" | "market"
    har: bool           # 1/7/30-day averages (True) or day-t value only (False)
    transform: str      # after averaging: "log" | "log1p" | "level" | "none"; before averaging: "log_daily" | "log1p_daily"
    dlog: bool          # add the day-t log change
    csq: bool           # add the in-window centred square (x - window mean)^2 in HAR form
    impute_flag: str    # name of a boolean flag column; flagged days are imputed with the window mean


def load_variable_spec(path: Optional[str] = None) -> List[VarSpec]:
    df = pd.read_csv(path or SPEC_PATH, dtype=str, keep_default_na=False)
    out = []
    for r in df.itertuples(index=False):
        out.append(VarSpec(r.name, r.table, r.block, r.scope, r.har == "1", r.transform,
                           r.dlog == "1", r.csq == "1", r.impute_flag.strip()))
    names = [v.name for v in out]
    assert len(set(names)) == len(names), "duplicate variable names in the spec"
    for v in out:
        assert v.transform in ("log", "log1p", "level", "none", "log_daily", "log1p_daily"), v
        assert v.scope in ("coin", "market"), v
    return out


def utc_days(s: pd.Series) -> pd.Series:
    """D21/PF1: days are UTC calendar days. tz-aware input must be UTC; naive input is taken as a
    UTC calendar date and must be at midnight."""
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        assert str(s.dt.tz) in ("UTC", "utc", "+00:00"), f"day column is in time zone {s.dt.tz}, not UTC"
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    assert (s == s.dt.normalize()).all(), "day column carries a time of day; expected UTC calendar days"
    return s.dt.normalize().astype("datetime64[ns]")


def _read(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "day" not in df.columns:
        raise ValueError(f"{path}: no 'day' column")
    df = df.copy()
    df["day"] = utc_days(df["day"])
    return df


def load_tables(panel: str, extra: Sequence[str] = ()) -> Dict[str, pd.DataFrame]:
    """`panel` is a directory holding any of the four tables, or one merged parquet file.
    `extra` (28 Oct 2026, optional): further coin-day parquet files (columns coin, day, ...), e.g. the subsampled
    RV evaluation target of arm A62; they may only ADD columns (an overlap with the panel raises)."""
    if os.path.isdir(panel):
        out = {}
        for t in TABLES:
            p = os.path.join(panel, f"{t}.parquet")
            if os.path.exists(p):
                out[t] = _read(p)
        if "volatility_daily" not in out:
            raise FileNotFoundError(f"{panel}: volatility_daily.parquet is required")
    else:
        out = {"merged": _read(panel)}
    for i, path in enumerate(extra or ()):
        t = _read(path)
        if "coin" not in t.columns:
            raise ValueError(f"{path}: an extra table needs a 'coin' column")
        assert not t.duplicated(["coin", "day"]).any(), f"{path}: duplicate coin-days"
        have = set().union(*(set(x.columns) for x in out.values())) - {"coin", "day"}
        overlap = sorted(have & (set(t.columns) - {"coin", "day"}))
        if overlap:
            raise ValueError(f"{path}: columns already in the panel: {overlap} (extra tables may only add columns)")
        out[f"extra_{i}"] = t
    return out


def build_grid(tables: Dict[str, pd.DataFrame], specs: List[VarSpec],
               extra_cols: tuple = (), value_cols: tuple = ()) -> pd.DataFrame:
    """Complete coin x day grid (5 x 1,280) of the raw columns the spec needs, market-level
    columns broadcast to every coin. Missing spec columns raise (the message lists them).
    `extra_cols` are boolean flags (missing -> False); `value_cols` (28 Oct 2026, optional) are further
    coin-level numeric columns taken as they are (e.g. the A62 evaluation target)."""
    days = pd.date_range(SAMPLE_FIRST_DAY, SAMPLE_LAST_DAY, freq="D")
    assert len(days) == N_DAYS
    grid = pd.MultiIndex.from_product([list(COINS), days], names=["coin", "day"])
    out = pd.DataFrame(index=grid)
    needed = {v.name: v for v in specs}
    flags = {v.impute_flag for v in specs if v.impute_flag} | set(extra_cols)
    values = [c for c in value_cols if c not in needed and c not in flags]
    missing = []
    srcs = list(tables.values())
    for col in list(needed) + sorted(flags) + values:
        v = needed.get(col)
        found = False
        for t in srcs:
            if col not in t.columns:
                continue
            found = True
            if "coin" in t.columns:
                if v is not None and v.scope == "market" and (t["coin"] == "MARKET").any():
                    s = t[t["coin"] == "MARKET"].set_index("day")[col]
                    assert s.index.is_unique, f"{col}: duplicate MARKET days"
                    out[col] = np.tile(s.reindex(days).to_numpy(), len(COINS))
                else:
                    s = t[t["coin"].isin(COINS)].set_index(["coin", "day"])[col]
                    assert s.index.is_unique, f"{col}: duplicate coin-days"
                    out[col] = s.reindex(grid).to_numpy()
                    if v is not None and v.scope == "market":
                        chk = out[col].unstack("coin")
                        assert (chk.nunique(axis=1, dropna=False) <= 1).all(), f"{col}: market variable differs across coins"
            else:
                s = t.set_index("day")[col]
                assert s.index.is_unique, f"{col}: duplicate days"
                out[col] = np.tile(s.reindex(days).to_numpy(), len(COINS))
            break
        if not found:
            missing.append(col)
    if missing:
        raise KeyError(f"panel lacks columns required by the variable spec: {missing}")
    for f in flags:
        if f in out:
            out[f] = out[f].fillna(False).astype(bool)
    return out


def placebo_shift(grid: pd.DataFrame, specs: List[VarSpec], shift_days: int,
                  mode: str = "circular", blocks=("polarity", "emotion")) -> pd.DataFrame:
    """D8 note section 5: each coin's text series (polarity and emotion variables) shifted by
    `shift_days` calendar days. mode="circular" wraps within the coin's 1,280 days, so no day is
    lost; mode="lag" uses x(t - shift) and leaves the first `shift_days` days NaN."""
    if not shift_days:
        return grid
    g = grid.copy()
    cols = [v.name for v in specs if v.block in blocks and v.name in g.columns]
    for coin in COINS:
        sl = g.index.get_level_values("coin") == coin
        for c in cols:
            x = g.loc[sl, c].to_numpy()
            if mode == "circular":
                g.loc[sl, c] = np.roll(x, shift_days)
            elif mode == "lag":
                y = np.full_like(x, np.nan, dtype=float)
                y[shift_days:] = x[:-shift_days]
                g.loc[sl, c] = y
            else:
                raise ValueError(mode)
    return g


def volatility_from_cj_artifact(path: str, approach: str = "A1") -> pd.DataFrame:
    """Validation helper: volatility_daily-format table (coin, day, rv, c, j) from the C/J
    artifact cj_focused_empirical_daily_20261005.parquet (grid 5, overnight-style outage split)."""
    d = pd.read_parquet(path)
    e = d[(d.grid == 5) & (d.outage == "split") & (d.approach == approach)].copy()
    e["day"] = utc_days(e["day"])
    out = e[["coin", "day", "rv", "c", "j", "jump", "gap_day"]].sort_values(["coin", "day"]).reset_index(drop=True)
    assert len(out) == len(COINS) * N_DAYS, len(out)
    assert np.allclose(out.c + out.j, out.rv)
    return out
