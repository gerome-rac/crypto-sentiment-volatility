"""
cj_methods_20261005.py
======================

One library for the continuous/jump (C/J) split of daily realized variance from 1-minute klines.
It implements, behind one interface, the approaches selected by the three literature tracks of
5 Oct 2026 (subsampling / all-data, robust estimators and tests, crypto practice), so that a Monte
Carlo study and the application to the Binance data (BTC, ETH, BNB, XRP, DOGE; UTC days,
1 Aug 2019 - 31 Jan 2023) can run every approach the same way.

Nothing in this file is a decision. The decided parts of the construction are defaults: UTC days,
previous-tick prices (a 1-minute kline's close is the last trade of that minute; a minute without
trades carries the previous close), 288 five-minute log returns per day in percent, the first one
from the previous day's 23:59 close. Everything else is a parameter. Own choices are labelled OWN
in the registry and in comments. The Kolokolov-Reno (2024) arm R06 is built from the published article
(JBES 42(2), 516-532; read in full 5 Oct 2026); its online supplement (Appendices B and C) was not available.

Interface
---------
DayData(close, prev_close, open=None, high=None, low=None, n_trades=None, outage=None,
        prev_outage_last=None, days=None, pair="")
    close/open/high/low/n_trades/outage: arrays of shape (n_days, 1440), one row per UTC day,
    column k = the kline opening k minutes after 00:00 UTC; prev_close: last close of the previous
    day; outage: exchange-wide missing minutes (see load_binance_1m for the 5 Oct script rule).
run(approach_id, data, days=None, **params) -> pandas.DataFrame (one row per selected day)
    standard columns: pair, day, approach, rv (the RV used as target), c, j (j >= 0, c = rv - j),
    jump, stat, pval, crit, n_used (returns used), j_raw (jump magnitude before the decision),
    drop_day (ABD 2007 fn 9 flag), drop_maxrun, zero_share, outage_min; then diagnostics.
daydata_from_minutes(close, open, high, low, n_trades, ...) -> DayData from flat or (n_days, 1440) arrays
    (NaN minutes carried forward); daydata_from_simulator(klines DataFrame of binance_dgp_simulator_20261005).
run_many(ids, data, days=None, params=None), run_panels({pair: DayData}, ids, n_jobs=3 (at most 4)).
REGISTRY, registry_table(), NOT_IMPLEMENTED.
benchmark(), smoke_table(), null_rejection_table(), calibrate_offset_tests(), simulate_range_lambdas(): the runs
    reported in cj_methods_spec_20261005.md (sections 5-8).

Common parameters: grid_step (minutes; tested 1, 5, 10, 15, 30; default 5), alpha (daily confidence
level, default 0.999), outage ('none' carry-forward | 'split' overnight-style split as in the 5 Oct
script | 'drop' ABD 2007 fn 9 rule: outputs set to NaN on days with more than 20 consecutive empty
5-minute intervals), edge_rule for shifted grids ('E1' drop edge pieces and rescale by counts, MFE
toolbox | 'E2' include the two short edge pieces in RV (products as E1) | 'E3' scale each grid to the
full day, ADS 2012); open choice O3 of the all-data report.

Units: returns are 100 * diff(log price); RV, BV, C, J are in %^2 per UTC day.
Source tags: FT full text read (page/eq.), FT-v working-paper/AAM/arXiv version, FT-code source code
read, SEC formula taken from another paper read in full, ABS abstract only, MEM from memory
(unverified), OWN own choice or derivation (flagged; not a construction rule until Gerome decides).
Sources are those verified in the 5 Oct 2026 formula sheets (artifacts approaches_all_data_20261005,
approaches_robust_20261005, crypto_cj_practice_20261005) unless a comment says otherwise.
"""
from __future__ import annotations

import math
import os
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.special import gamma as _G
from scipy.special import gammaincc as _GINC
from scipy.stats import norm

__version__ = "2026-10-05"
MIN_PER_DAY = 1440
TESTED_GRID_STEPS = (1, 5, 10, 15, 30)

# =====================================================================================================
# 1. Constants (source next to each; unit-tested in test_cj_methods_20261005.py)
# =====================================================================================================
MU1 = math.sqrt(2.0 / math.pi)                          # E|Z| = 0.797885; ABD 2007 p. 703 [FT]
MU43 = 2.0 ** (2.0 / 3.0) * _G(7.0 / 6.0) / _G(0.5)     # E|Z|^(4/3) = 0.830861; ABD 2007 p. 709 [FT]
THETA_BV = math.pi ** 2 / 4.0 + math.pi - 5.0           # 0.608994; ABD 2007 eq. 18; BNS 2006 WP eq. 6 [FT]
MEDRV_C = math.pi / (6.0 - 4.0 * math.sqrt(3.0) + math.pi)            # 1.41936; ADS 2009 WP eq. 4, p. 7 [FT-v]
MEDRQ_C = 3.0 * math.pi / (9.0 * math.pi + 72.0 - 52.0 * math.sqrt(3.0))  # 0.923302 (times N); ADS WP p. 9
MINRV_C = math.pi / (math.pi - 2.0)                     # 2.75194; ADS WP p. 7 [FT-v]
MINRQ_C = math.pi / (3.0 * math.pi - 8.0)               # 2.20497 (times N); ADS WP p. 9 [FT-v]
V_MED = 0.96   # Var(RV - MedRV)/IQ = 2 + 2.96 - 2*2; ADS WP Prop. 3, p. 9; ratio test DU 2012 eq. 15 [FT-v]
V_MIN = 1.81   # Var(RV - MinRV)/IQ = 2 + 3.81 - 2*2; ADS WP Prop. 3; DU 2012 eq. 14 [FT-v]
SHORTH_C = 0.741   # BCL 2011 WP eq. 2.10, p. 9 [FT-v] (= 1/(2 Phi^-1(0.75)) = 0.7413)
WSD_C = 1.081      # BCL 2011 WP eq. 2.12, p. 10 [FT-v] (= 1/E(Z^2 | Z^2 <= 6.635) = 1.0814)
WSD_CUT = 6.635    # chi2(1) 99% quantile used for the WSD weights; BCL eq. 2.12 [FT-v]
RANGE_LAMBDA = {2.0 / 3.0: 1.3528702658, 1.0: math.sqrt(8.0 / math.pi), 2.0: 4.0 * math.log(2.0),
                4.0: 9.0 * 1.2020569031595942}
# E(R^r) for the range R of a standard Brownian motion on [0, 1]: lambda_1, lambda_2, lambda_4 from CP 2007
# fn 7 [FT-v]; lambda_{2/3} = 1.35287 from Feller's range-moment formula (MEM), checked by simulation (OWN).
RK_CSTAR = (12.0 ** 2 / 0.269) ** 0.2   # 3.5134; Parzen kernel, BNHLS 2009 eq. 3 [FT-v]


def cpr_zfactor(gam: float, c_theta: float = 3.0) -> float:
    """Correction factor of Corsi, Pirino & Reno (2010) eq. 3.3 [FT-v AAM p. 7].

    Z_gam(x, y) = cpr_zfactor(gam, c) * y**(gam/2) when x^2 > y (threshold y = c^2 V), i.e.
    [2 Phi(-c) sqrt(pi)]^-1 (2/c^2)^(gam/2) Gamma((gam+1)/2, c^2/2) with the upper incomplete gamma.
    c = 3: 1.09437 (gam = 1), 1.12936 (gam = 4/3); CPR print 1.094 and 1.129.
    """
    upper = _GINC((gam + 1.0) / 2.0, c_theta ** 2 / 2.0) * _G((gam + 1.0) / 2.0)
    return (1.0 / (2.0 * norm.cdf(-c_theta) * math.sqrt(math.pi))) * (2.0 / c_theta ** 2) ** (gam / 2.0) * upper


def mss_k(M: int, r: float) -> float:
    """k_{M,r} = E[chi_M^r] = 2^(r/2) Gamma((M+r)/2)/Gamma(M/2); MSS 2012 eq. 16 [FT-v]."""
    return 2.0 ** (r / 2.0) * _G((M + r) / 2.0) / _G(M / 2.0)


def mss_mtheta(M: int) -> float:
    """M theta_M = M [M^2/k_{M,1}^4 - 1 + 2 (M/k_{M,1}^2 - 1)]; MSS 2012 eq. 9 [FT-v]. M = 5: 2.1439."""
    k1 = mss_k(M, 1.0)
    return M * (M ** 2 / k1 ** 4 - 1.0 + 2.0 * (M / k1 ** 2 - 1.0))


def gumbel_constants(n: int, c: float = 1.0):
    """Lee & Mykland (2008) eqs. 12-13 [FT]: C_n = sqrt(2 log n)/c - (log pi + log log n)/(2 c sqrt(2 log n)),
    S_n = 1/(c sqrt(2 log n)). With c = 1 these are BCL's a_n, b_n (BCL WP pp. 21-22 [FT-v])."""
    ln = math.log(n)
    s = math.sqrt(2.0 * ln)
    return s / c - (math.log(math.pi) + math.log(ln)) / (2.0 * c * s), 1.0 / (c * s)


def gumbel_threshold(size: float, n: int, c: float = 1.0) -> float:
    """Critical value C_n + S_n beta*, beta* = -log(-log(1 - size)) (LM 2008 p. 2543 [FT])."""
    a, b = gumbel_constants(n, c)
    return a + b * (-math.log(-math.log(1.0 - size)))


def gumbel_pvalue(stat, n: int, c: float = 1.0):
    a, b = gumbel_constants(n, c)
    with np.errstate(over="ignore", invalid="ignore"):
        return -np.expm1(-np.exp(-(np.asarray(stat, float) - a) / b))


def sidak_threshold(size: float, n: int) -> float:
    """ABD-Dobrev (2007) eq. 19 [FT]: beta = 1 - (1 - size)^(1/n), critical value Phi^-1(1 - beta/2)."""
    beta = -math.expm1(math.log1p(-size) / n)
    return float(norm.isf(beta / 2.0))


def sidak_pvalue(stat, n: int):
    with np.errstate(invalid="ignore", divide="ignore"):
        return -np.expm1(n * np.log1p(-2.0 * norm.sf(np.asarray(stat, float))))


def ut_threshold(n_tests: int) -> float:
    """Universal threshold sqrt(2 log N); Bajgrowicz, Scaillet & Treccani 2016, WP pp. 8-11 [FT-v]."""
    return math.sqrt(2.0 * math.log(n_tests))


def ut_alpha(n_tests: int) -> float:
    """One-sided level equivalent of the universal threshold (BST p. 9: half the two-sided level)."""
    return float(norm.cdf(ut_threshold(n_tests)))


def lm_window(n_per_day: int, year_days: int = 252) -> int:
    """Lee-Mykland K = smallest integer >= sqrt(year_days * n_obs) (LM 2008 p. 2542 [FT]); 270 for 5-min data."""
    return int(math.ceil(math.sqrt(year_days * n_per_day)))


# =====================================================================================================
# 2. Data container, Binance loader and a simple test data-generating process
# =====================================================================================================
@dataclass
class DayData:
    """1-minute klines of one pair, one row per UTC day (see module docstring)."""
    close: np.ndarray
    prev_close: np.ndarray
    open: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    low: Optional[np.ndarray] = None
    n_trades: Optional[np.ndarray] = None
    outage: Optional[np.ndarray] = None
    prev_outage_last: Optional[np.ndarray] = None
    days: Optional[np.ndarray] = None
    pair: str = ""
    contiguous: Optional[bool] = None
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        c = np.asarray(self.close, dtype=float)
        if c.ndim != 2 or c.shape[1] != MIN_PER_DAY:
            raise ValueError(f"close must have shape (n_days, {MIN_PER_DAY}), got {c.shape}")
        n = c.shape[0]
        self.close = c
        self.prev_close = np.asarray(self.prev_close, dtype=float).reshape(n)
        for name in ("open", "high", "low"):
            v = getattr(self, name)
            if v is not None:
                v = np.asarray(v, dtype=float)
                if v.shape != c.shape:
                    raise ValueError(f"{name} has shape {v.shape}, expected {c.shape}")
                setattr(self, name, v)
        if self.n_trades is not None:
            v = np.asarray(self.n_trades)
            if v.shape != c.shape:
                raise ValueError(f"n_trades has shape {v.shape}, expected {c.shape}")
            self.n_trades = v.astype(np.int64)
        self.outage = (np.zeros(c.shape, bool) if self.outage is None else np.asarray(self.outage, bool))
        if self.outage.shape != c.shape:
            raise ValueError("outage mask has the wrong shape")
        self.prev_outage_last = (np.zeros(n, bool) if self.prev_outage_last is None
                                 else np.asarray(self.prev_outage_last, bool).reshape(n))
        if self.days is None:
            self.days = np.arange(n)
        else:
            d = np.asarray(self.days)
            if d.dtype.kind in "OUS" or d.dtype.kind == "M":
                try:
                    d = d.astype("datetime64[D]")
                except (TypeError, ValueError):
                    pass
            self.days = d
        if self.days.shape[0] != n:
            raise ValueError("days has the wrong length")
        if self.contiguous is None:
            if self.days.dtype.kind == "M" and n > 1:
                self.contiguous = bool(np.all(np.diff(self.days).astype(np.int64) == 1))
            elif self.days.dtype.kind in "iu" and n > 1:
                self.contiguous = bool(np.all(np.diff(self.days) == 1))
            else:
                self.contiguous = True
        if not (np.all(np.isfinite(c)) and np.all(c > 0)):
            raise ValueError("close must be finite and positive (carry the previous close over empty minutes)")
        if not (np.all(np.isfinite(self.prev_close)) and np.all(self.prev_close > 0)):
            raise ValueError("prev_close must be finite and positive")

    @property
    def n_days(self) -> int:
        return self.close.shape[0]

    def day_labels(self) -> np.ndarray:
        if self.days.dtype.kind == "M":
            return np.datetime_as_string(self.days, unit="D")
        return self.days.astype(str)

    def day_keys(self) -> np.ndarray:
        """Integer key per day (days since 1970 or the row label) used to seed randomised tests per day."""
        if self.days.dtype.kind == "M":
            return self.days.astype("datetime64[D]").astype(np.int64)
        return np.asarray(self.days, dtype=np.int64)

    def take(self, idx) -> "DayData":
        idx = np.asarray(idx)
        g = (lambda a: None if a is None else a[idx])
        return DayData(close=self.close[idx], prev_close=self.prev_close[idx], open=g(self.open),
                       high=g(self.high), low=g(self.low), n_trades=g(self.n_trades), outage=self.outage[idx],
                       prev_outage_last=self.prev_outage_last[idx], days=self.days[idx], pair=self.pair,
                       notes=dict(self.notes))


def daydata_from_minutes(close, open=None, high=None, low=None, n_trades=None, outage=None, prev_close0=None,
                         start=None, pair: str = "", missing_as_outage: bool = False) -> DayData:
    """Build a DayData from 1-minute arrays, e.g. the klines of a simulator (open/high/low/close/n_trades).
    Arrays may be flat (n_days * 1440) or (n_days, 1440); minute k of day d is the kline opening k minutes after
    00:00 UTC. Missing closes (NaN) are carried forward from the previous close (previous tick) and open/high/low
    of such minutes set to that close, n_trades to 0; missing_as_outage=True also marks them in the outage mask.
    prev_close of day d is the last close of day d-1; for the first day prev_close0, else the first open, else the
    first close (5 Oct script convention). start: first UTC day (e.g. '2019-08-01') to label the days."""
    def _2d(a, dtype=float):
        if a is None:
            return None
        a = np.array(a, dtype=dtype, copy=True)          # own writable copy (inputs may be read-only views)
        if a.ndim == 1:
            if a.size % MIN_PER_DAY:
                raise ValueError(f"flat arrays need a multiple of {MIN_PER_DAY} minutes, got {a.size}")
            a = a.reshape(-1, MIN_PER_DAY)
        return a
    c = _2d(close)
    flat = c.ravel()
    miss = ~np.isfinite(flat)
    if miss.all():
        raise ValueError("no finite close")
    if miss.any():
        idx = np.where(~miss, np.arange(flat.size), 0)
        np.maximum.accumulate(idx, out=idx)
        first = np.argmax(~miss)
        flat = flat[idx]
        flat[:first] = flat[first]
        c = flat.reshape(c.shape)
    miss2 = miss.reshape(c.shape)
    o, h, l = (_2d(x) for x in (open, high, low))
    for a in (o, h, l):
        if a is not None:
            bad = ~np.isfinite(a) | miss2
            a[bad] = c[bad]
    nt = None if n_trades is None else np.where(miss2, 0, np.nan_to_num(_2d(n_trades), nan=0.0)).astype(np.int64)
    out = None if outage is None else _2d(outage, bool)
    if missing_as_outage:
        out = miss2 if out is None else (out | miss2)
    if prev_close0 is not None:
        p0 = float(prev_close0)
    elif o is not None and np.isfinite(o[0, 0]):
        p0 = float(o[0, 0])
    else:
        p0 = float(c[0, 0])
    prev = np.concatenate([[p0], c[:-1, -1]])
    n = c.shape[0]
    days = (np.arange(n) if start is None else np.datetime64(start, "D") + np.arange(n))
    prev_out = np.zeros(n, bool) if out is None else np.concatenate([[False], out[:-1, -1]])
    return DayData(close=c, prev_close=prev, open=o, high=h, low=l, n_trades=nt, outage=out,
                   prev_outage_last=prev_out, days=days, pair=pair)


def daydata_from_simulator(kl, pair: str = "", missing_as_outage: bool = True, start=None) -> DayData:
    """DayData from the klines DataFrame of binance_dgp_simulator_20261005.simulate_days (columns day, minute,
    open/high/low/close or lopen/lhigh/llow/lclose in 100*log units, n_trades (-1 when the kline is missing),
    present; attrs['prev_close_log'] = 100*log of the previous close), following its docstring (format checked
    5 Oct 2026, simulator not run here). Missing klines are carried forward and, by default, marked as outage
    minutes (as the pair's own missing minutes in the 5 Oct script rule)."""
    kl = kl.sort_values(["day", "minute"])
    if "lclose" in kl:
        conv = (lambda s: np.exp(kl[s].to_numpy(float) / 100.0))
        o, h, l, c = conv("lopen"), conv("lhigh"), conv("llow"), conv("lclose")
    else:
        o, h, l, c = (kl[s].to_numpy(float) for s in ("open", "high", "low", "close"))
    pres = kl["present"].to_numpy(bool) if "present" in kl else np.isfinite(c)
    c = np.where(pres, c, np.nan)
    nt = np.where(pres, kl["n_trades"].to_numpy(float), 0.0) if "n_trades" in kl else None
    p0 = kl.attrs.get("prev_close_log")
    p0 = None if p0 is None else float(np.exp(float(p0) / 100.0))
    return daydata_from_minutes(c, o, h, l, nt, prev_close0=p0, start=start, pair=pair,
                                missing_as_outage=missing_as_outage)


def _runs_ge(b: np.ndarray, minlen: int) -> np.ndarray:
    """Mark runs of True of length >= minlen (1-D); as runs_ge in the 5 Oct script."""
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    out = np.zeros(b.shape, bool)
    for a, z in zip(s, e):
        if z - a >= minlen:
            out[a:z] = True
    return out


def load_binance_1m(raw_dir: str, pairs=("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"),
                    start: str = "2019-08-01", end: str = "2023-01-31", outage_rule: str = "script",
                    min_run: int = 5) -> dict:
    """Read Binance spot 1-minute kline zips (<raw_dir>/<PAIR>/<PAIR>-1m-YYYY-MM.zip; no header; open_time in
    ms, open, high, low, close, volume, close_time, quote_volume, n_trades, ...) in memory, read-only.

    Missing minutes are filled by previous tick (open = high = low = close = previous close, n_trades = 0).
    Outage mask (outage_rule='script', as `gap` in cj_diagnostics_20261005.py): the pair's own missing
    minutes OR runs of >= min_run minutes in which every loaded pair is missing or has no trade (exchange-wide
    inactivity, catches halts that show as zero-trade bars). outage_rule='missing': missing minutes only.
    prev_close of the first day = open of its first minute (no earlier data loaded; 5 Oct script convention).
    """
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC")
    n_days = int((t1 - t0).days) + 1
    nmin = n_days * MIN_PER_DAY
    t0_ms = int(t0.timestamp() * 1000)
    months = pd.period_range(t0.tz_localize(None), t1.tz_localize(None), freq="M")
    raw = {}
    for p in pairs:
        cl = np.full(nmin, np.nan); op = np.full(nmin, np.nan); hi = np.full(nmin, np.nan)
        lo = np.full(nmin, np.nan); nt = np.zeros(nmin, np.int64)
        for mo in months:
            f = os.path.join(raw_dir, p, f"{p}-1m-{mo.strftime('%Y-%m')}.zip")
            if not os.path.exists(f):
                raise FileNotFoundError(f)
            with zipfile.ZipFile(f) as z:
                names = [nm for nm in z.namelist() if nm.endswith(".csv")]
                if len(names) != 1:
                    raise ValueError(f"{f}: expected one csv")
                with z.open(names[0]) as fh:
                    df = pd.read_csv(fh, header=None, usecols=[0, 1, 2, 3, 4, 8],
                                     names=["ot", "o", "h", "l", "c", "n"],
                                     dtype={"ot": "int64", "o": "float64", "h": "float64", "l": "float64",
                                            "c": "float64", "n": "int64"})
            ot = df["ot"].to_numpy()
            if not np.all(ot % 60000 == 0):
                raise ValueError(f"{f}: open_time not in whole minutes (ms expected)")
            idx = (ot - t0_ms) // 60000
            ok = (idx >= 0) & (idx < nmin)
            ii = idx[ok]
            cl[ii] = df["c"].to_numpy()[ok]; op[ii] = df["o"].to_numpy()[ok]
            hi[ii] = df["h"].to_numpy()[ok]; lo[ii] = df["l"].to_numpy()[ok]; nt[ii] = df["n"].to_numpy()[ok]
        missing = ~np.isfinite(cl)
        if missing[0]:
            raise ValueError(f"{p}: first minute missing; cannot set the first previous tick")
        cff = pd.Series(cl).ffill().to_numpy()
        for arr in (op, hi, lo):
            arr[missing] = cff[missing]
        nt[missing] = 0
        raw[p] = dict(close=cff, open=op, high=hi, low=lo, n=nt, missing=missing)
    if outage_rule == "script":
        inactive = np.logical_and.reduce([raw[p]["missing"] | (raw[p]["n"] == 0) for p in pairs])
        ex_gap = _runs_ge(inactive, min_run)
    elif outage_rule == "missing":
        ex_gap = np.zeros(nmin, bool)
    else:
        raise ValueError("outage_rule must be 'script' or 'missing'")
    days = np.array(pd.date_range(t0.tz_localize(None), periods=n_days, freq="D").values, dtype="datetime64[D]")
    out = {}
    for p in pairs:
        r = raw[p]
        gap = ex_gap | r["missing"]
        sh = (n_days, MIN_PER_DAY)
        prev_close = np.empty(n_days)
        prev_close[0] = r["open"][0]
        prev_close[1:] = r["close"].reshape(sh)[:-1, -1]
        prev_out = np.zeros(n_days, bool)
        prev_out[1:] = gap.reshape(sh)[:-1, -1]
        out[p] = DayData(close=r["close"].reshape(sh), prev_close=prev_close, open=r["open"].reshape(sh),
                         high=r["high"].reshape(sh), low=r["low"].reshape(sh), n_trades=r["n"].reshape(sh),
                         outage=gap.reshape(sh), prev_outage_last=prev_out, days=days, pair=p,
                         notes={"prev_close_day0": "open of the first minute", "outage_rule": outage_rule,
                                "missing_minutes": int(r["missing"].sum())})
    return out


def simulate_test_days(n_days: int = 100, seed: int = 0, iv: float = 1.0, pattern="flat", jumps=None,
                       tick: Optional[float] = None, stale_prob: float = 0.0, outage=None, sub: int = 1,
                       price0: float = 100.0, start: str = "2020-01-01", stale_block: int = 1):
    """Simple test data-generating process for unit tests and benchmarks (NOT the calibrated simulator).

    Efficient log price (percent units) = Brownian motion with daily integrated variance `iv` (%^2), spread over
    the day by a deterministic intraday pattern ('flat', 'ushape' or an array of 1440 weights); `sub` sub-steps per
    minute (all observed as trades, so highs/lows exist when sub > 1). Optional: jumps=dict(per_day=k,
    size=m) adds k jumps of m local 1-minute standard deviations at random minutes (random sign);
    tick rounds observed prices to a tick (zero returns appear); stale_prob = probability that a minute has no
    trade (close = previous close, n_trades = 0), drawn per block of stale_block minutes (stale_block = 5: Bernoulli
    staleness of whole 5-minute intervals, the design of the robust report's own check); outage=dict(days=[...],
    start=minute, length=minutes) blocks of
    missing minutes (carried prices, outage mask True). Returns (DayData, truth) with truth['iv'] (IV per day,
    continuous part), truth['iq'] (integrated quarticity), truth['j2'] (sum of squared jumps), truth['n_jumps'].
    """
    rng = np.random.default_rng(seed)
    if isinstance(pattern, str):
        u = (np.arange(MIN_PER_DAY) + 0.5) / MIN_PER_DAY
        if pattern == "flat":
            w = np.ones(MIN_PER_DAY)
        elif pattern == "ushape":
            w = 1.0 + 2.0 * (2.0 * u - 1.0) ** 2   # max/min variance ratio 3 (toy U-shape)
        else:
            raise ValueError("pattern must be 'flat', 'ushape' or an array")
    else:
        w = np.asarray(pattern, float)
    w = w / w.mean()
    var_min = iv * w / MIN_PER_DAY                      # variance per minute (%^2)
    inc = rng.standard_normal((n_days, MIN_PER_DAY, sub)) * np.sqrt(var_min / sub)[None, :, None]
    j2 = np.zeros(n_days)
    nj = np.zeros(n_days, int)
    if jumps:
        k = int(jumps.get("per_day", 1))
        size = float(jumps.get("size", 10.0))
        for d in range(n_days):
            mins = rng.choice(MIN_PER_DAY, size=k, replace=False)
            for m in mins:
                jmp = size * math.sqrt(var_min[m]) * (1 if rng.random() < 0.5 else -1)
                inc[d, m, -1] += jmp
                j2[d] += jmp ** 2
                nj[d] += 1
    p0 = 100.0 * math.log(price0)
    path = np.concatenate([[p0], p0 + np.cumsum(inc.ravel())])        # efficient log price, continuous over days
    px = np.exp(path / 100.0)
    if tick:
        px = np.round(px / tick) * tick
    sub_px = px[1:].reshape(n_days, MIN_PER_DAY, sub)
    close = sub_px[:, :, -1].copy()
    high = sub_px.max(axis=2)
    low = sub_px.min(axis=2)
    open_ = sub_px[:, :, 0].copy()
    ntr = np.full((n_days, MIN_PER_DAY), sub, dtype=np.int64)
    out_mask = np.zeros((n_days, MIN_PER_DAY), bool)
    if outage:
        for d in outage.get("days", []):
            a = int(outage.get("start", 120)); L = int(outage.get("length", 480))
            out_mask[d, a:a + L] = True
    if stale_block > 1:
        nb = MIN_PER_DAY // stale_block
        no_trade = np.repeat(rng.random((n_days, nb)) < stale_prob, stale_block, axis=1) | out_mask
    else:
        no_trade = (rng.random((n_days, MIN_PER_DAY)) < stale_prob) | out_mask
    flat_close = close.ravel()
    nt_flat = no_trade.ravel()
    first = px[0]
    for arr in (open_, high, low):
        arr_flat = arr.ravel()
        arr_flat[nt_flat] = np.nan
    flat_close[nt_flat] = np.nan
    cff = pd.Series(np.concatenate([[first], flat_close])).ffill().to_numpy()
    prev_flat = cff[:-1]
    close = cff[1:].reshape(n_days, MIN_PER_DAY)
    prevm = prev_flat.reshape(n_days, MIN_PER_DAY)
    for arr in (open_, high, low):
        np.copyto(arr, prevm, where=no_trade)
    # high/low of a traded minute also cover the minute's open (the first trade)
    ntr[no_trade] = 0
    prev_close = np.concatenate([[first], close[:-1, -1]])
    prev_out = np.concatenate([[False], out_mask[:-1, -1]])
    days = np.arange(np.datetime64(start, "D"), np.datetime64(start, "D") + n_days)
    iq = (iv ** 2) * np.mean(w ** 2)
    data = DayData(close=close, prev_close=prev_close, open=open_, high=high, low=low, n_trades=ntr,
                   outage=out_mask, prev_outage_last=prev_out, days=days, pair="SIM")
    truth = dict(iv=np.full(n_days, iv), iq=np.full(n_days, iq), j2=j2, n_jumps=nj)
    return data, truth


# =====================================================================================================
# 3. Numerical primitives (NaN marks a return that is not used; sums run over usable terms only)
# =====================================================================================================
def _div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _sc(x: np.ndarray):
    """Row sums over finite entries and their counts."""
    f = np.isfinite(x)
    return np.where(f, x, 0.0).sum(axis=1), f.sum(axis=1)


def _lagprod(a: np.ndarray, lags) -> np.ndarray:
    """Element-wise product prod_l a[:, j - l] for all j with j - max(lags) >= 0."""
    L = max(lags)
    N = a.shape[1]
    if N <= L:
        return np.empty((a.shape[0], 0))
    out = a[:, L - lags[0]:N - lags[0]].copy()
    for l in lags[1:]:
        out *= a[:, L - l:N - l]
    return out


def _nobs(r: np.ndarray) -> np.ndarray:
    return np.isfinite(r).sum(axis=1)


def realized_variance(r: np.ndarray) -> np.ndarray:
    return np.nansum(r * r, axis=1)


def bipower(r: np.ndarray, skip: int = 1, n=None) -> np.ndarray:
    """mu1^-2 * n * mean(|r_j| |r_{j-1-skip}|) over usable products.
    skip = 1: ABD 2007 eq. 22 (factor N/(N-2)); skip = 0: BNS 2004 / HT 2005 adjacent BV (factor N/(N-1))."""
    n = _nobs(r) if n is None else n
    s, k = _sc(_lagprod(np.abs(r), (0, skip + 1)))
    return MU1 ** -2 * n * _div(s, k)


def tripower_quarticity(r: np.ndarray, skip: int = 1, n=None) -> np.ndarray:
    """mu43^-3 * n^2 * mean(prod |r|^(4/3)) at lags 0, d, 2d (d = skip + 1).
    skip = 1: ABD 2007 eq. 23 (N mu43^-3 N/(N-4) sum); skip = 0: HT 2005 eq. 3 / ABD eq. 15 (N/(N-2))."""
    n = _nobs(r) if n is None else n
    d = skip + 1
    s, k = _sc(_lagprod(np.abs(r) ** (4.0 / 3.0), (0, d, 2 * d)))
    return MU43 ** -3 * n * n * _div(s, k)


def _med3(a, b, c):
    return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))


def medrv(r: np.ndarray, n=None):
    """MedRV and MedRQ of ADS 2009 WP eq. 4 and p. 9 [FT-v] (factors N/(N-2)); returns (MedRV, MedRQ)."""
    n = _nobs(r) if n is None else n
    a = np.abs(r)
    m = _med3(a[:, :-2], a[:, 1:-1], a[:, 2:])
    s2, k = _sc(m * m)
    s4, _ = _sc(m ** 4)
    return MEDRV_C * n * _div(s2, k), MEDRQ_C * n * n * _div(s4, k)


def minrv(r: np.ndarray, n=None):
    """MinRV and MinRQ of ADS 2009 WP p. 7, p. 9 [FT-v] (factors N/(N-1)); returns (MinRV, MinRQ)."""
    n = _nobs(r) if n is None else n
    a = np.abs(r)
    m = np.minimum(a[:, :-1], a[:, 1:])
    s2, k = _sc(m * m)
    s4, _ = _sc(m ** 4)
    return MINRV_C * n * _div(s2, k), MINRQ_C * n * n * _div(s4, k)


def ratio_stat(n, rv, x, q, v):
    """Max-adjusted ratio statistic sqrt(n) (1 - x/rv) / sqrt(v max(1, q/x^2)) (HT 2005 eq. 9; ABD 2007 eq. 18)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(n) * ((rv - x) / rv) / np.sqrt(v * np.maximum(1.0, q / x ** 2))


def _out(rv, j_raw, jump, stat, pval, crit, n_used, **diag):
    jump = np.asarray(jump, dtype=bool)
    j_raw = np.asarray(j_raw, dtype=float)
    j = np.where(jump, j_raw, 0.0)
    rv = np.asarray(rv, dtype=float)
    return dict(rv=rv, c=rv - j, j=j, jump=jump, stat=np.asarray(stat, float), pval=np.asarray(pval, float),
                crit=np.broadcast_to(np.asarray(crit, float), rv.shape).copy(), n_used=np.asarray(n_used),
                j_raw=j_raw, **diag)


def _rv_only(rv, n_used, **diag):
    nan = np.full(np.shape(rv), np.nan)
    return dict(rv=np.asarray(rv, float), c=nan, j=nan, jump=None, stat=nan, pval=nan, crit=nan,
                n_used=np.asarray(n_used), j_raw=nan, **diag)


def _max_run(b: np.ndarray) -> np.ndarray:
    best = np.zeros(b.shape[0], np.int64)
    cur = np.zeros(b.shape[0], np.int64)
    for k in range(b.shape[1]):
        cur = (cur + 1) * b[:, k]
        np.maximum(best, cur, out=best)
    return best


def _require_contiguous(data: DayData, what: str):
    if not data.contiguous:
        raise ValueError(f"{what} needs consecutive UTC days in DayData (cross-day window); pass the full panel "
                         f"and select output days with days=...")


# =====================================================================================================
# 4. Per-call preparation: prices, grid returns, outage masks (cached)
# =====================================================================================================
class _Prep:
    """Prices, returns and masks for one DayData; `rows` are the output rows. Cross-day estimators read
    the full panel (full=True) for their windows; outputs are computed for `rows` only."""

    def __init__(self, data: DayData, rows):
        self.data = data
        self.rows = np.asarray(rows, dtype=np.int64)
        self.n = int(self.rows.size)
        self.all_rows = bool(self.n == data.n_days and np.array_equal(self.rows, np.arange(data.n_days)))
        self._c = {}

    def cached(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def P(self, full: bool = False) -> np.ndarray:
        """Natural log prices, shape (n, 1441): column 0 = previous day's last close, column k = close of the
        kline opening k-1 minutes after 00:00 UTC (the price at minute k). Returns are 100 * diff(log price),
        in this operation order, as in the 5 Oct script."""
        if full or self.all_rows:
            return self.cached("P_full", lambda: np.log(
                np.concatenate([self.data.prev_close[:, None], self.data.close], axis=1)))
        return self.cached("P_rows", lambda: self.P(True)[self.rows])

    def r(self, g: int, s: int = 0, full: bool = False) -> np.ndarray:
        """Returns (percent) on the grid of step g minutes starting at minute s (s >= 1: full intervals only)."""
        full = full or self.all_rows
        return self.cached(("r", g, s, full), lambda: 100.0 * np.diff(self.P(full)[:, s::g], axis=1))

    def edges(self, g: int, s: int, full: bool = False):
        P = self.P(full)
        return 100.0 * (P[:, s] - P[:, 0]), 100.0 * (P[:, MIN_PER_DAY] - P[:, MIN_PER_DAY - g + s])

    def gapext(self, full: bool = False) -> np.ndarray:
        full = full or self.all_rows

        def mk():
            d = self.data
            G = np.concatenate([d.prev_outage_last[:, None], d.outage], axis=1)
            return G if full else G[self.rows]
        return self.cached(("gapext", full), mk)

    def gapcum(self, full: bool = False) -> np.ndarray:
        full = full or self.all_rows
        return self.cached(("gapcum", full), lambda: np.cumsum(self.gapext(full), axis=1, dtype=np.int32))

    def affected(self, t, full: bool = False) -> np.ndarray:
        """Intervals (t_{j-1}, t_j] touched by an outage minute in [t_{j-1}, t_j] (start point included), as in
        the 5 Oct script: an interval is affected if it contains a gap minute or starts at a stale grid price."""
        G = self.gapext(full)
        C = self.gapcum(full)
        t = np.asarray(t)
        return (C[:, t[1:]] - C[:, t[:-1]] + G[:, t[:-1]]) > 0

    def obs(self, g: int, s: int, mode: str, full: bool = False) -> np.ndarray:
        full = full or self.all_rows
        n = self.data.n_days if full else self.n
        t = np.arange(s, MIN_PER_DAY + 1, g)
        if mode != "split":
            return np.ones((n, t.size - 1), bool)
        return self.cached(("obs", g, s, full), lambda: ~self.affected(t, full))

    def obs_edges(self, g: int, s: int, mode: str, full: bool = False):
        full = full or self.all_rows
        n = self.data.n_days if full else self.n
        if mode != "split":
            return np.ones(n, bool), np.ones(n, bool)
        head = ~self.affected(np.array([0, s]), full)[:, 0]
        tail = ~self.affected(np.array([MIN_PER_DAY - g + s, MIN_PER_DAY]), full)[:, 0]
        return head, tail

    def rmask(self, g: int, s: int, mode: str, full: bool = False) -> np.ndarray:
        full = full or self.all_rows
        if mode != "split":
            return self.r(g, s, full)
        return self.cached(("rm", g, s, full), lambda: np.where(self.obs(g, s, mode, full), self.r(g, s, full),
                                                                np.nan))

    def rows_of(self, full_array: np.ndarray) -> np.ndarray:
        return full_array if self.all_rows else full_array[self.rows]

    # --- common diagnostics
    def zero_share(self, g: int, mode: str) -> np.ndarray:
        r = self.rmask(g, 0, mode)
        return _div(((r == 0) & np.isfinite(r)).sum(1), np.isfinite(r).sum(1))

    def outage_min(self) -> np.ndarray:
        return self.rows_of(self.data.outage).sum(axis=1)

    def drop_flag(self, basis: str = "trades", run: int = 20):
        """ABD 2007 fn 9 (p. 703) [FT]: a day is dropped if it has more than `run` (20) consecutive 5-minute
        intervals without new prices. basis='trades': an interval is empty if none of its minutes has a trade
        (missing minutes count as no trade); basis='outage': if all its minutes are outage minutes."""
        def mk():
            d = self.data
            if basis == "trades" and d.n_trades is not None:
                empty_min = self.rows_of(d.n_trades) == 0
            else:
                empty_min = self.rows_of(d.outage)
            empty = empty_min.reshape(empty_min.shape[0], MIN_PER_DAY // 5, 5).all(axis=2)
            mr = _max_run(empty)
            return mr > run, mr
        return self.cached(("drop", basis, run), mk)


# =====================================================================================================
# 5. Periodicity (BCL 2011 WSD), time-of-day factors (BTL 2013) and cross-day windows
# =====================================================================================================
def _bcl_scale(rf: np.ndarray) -> np.ndarray:
    """BCL 2011 WP eq. 2.7 with lambda = one UTC day: s^2 = (pi/2) mean(|r_l||r_{l-1}|) = adjacent BV / N."""
    s, k = _sc(_lagprod(np.abs(rf), (0, 1)))
    return np.sqrt((math.pi / 2.0) * _div(s, k))


def _shorth_wsd(S: np.ndarray):
    """Weighted standard deviation periodicity factors of BCL 2011 (eqs. 2.10-2.13, WP pp. 9-10 [FT-v]).

    S: (B, N, K) standardized returns of N periodicity cells over K days (NaN = missing). Returns (f, n_guard):
    f (B, N) normalised so that mean(f^2) = 1 over the N cells; cells without a usable scale (fewer than 2 values,
    or a shortest half of zero, e.g. mostly stale zeros) are set to 1 = no filtering (guard, OWN).
    """
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fin = np.isfinite(S)
        Ss = np.sort(S, axis=2)
        nv = fin.sum(axis=2)
        h = nv // 2 + 1
        K = S.shape[2]
        m = np.arange(K)[None, None, :]
        hi = np.minimum(m + (h - 1)[..., None], K - 1)
        D = np.take_along_axis(Ss, hi, axis=2) - Ss
        ok = (m <= (nv - h)[..., None]) & (nv[..., None] >= 2)
        D = np.where(ok, D, np.inf)
        sh = SHORTH_C * D.min(axis=2)
        good = np.isfinite(sh) & (sh > 0) & (nv >= 2)
        sh = np.where(good, sh, np.nan)
        fS = sh / np.sqrt(np.nanmean(sh * sh, axis=1))[:, None]
        w = fin & good[..., None] & ((S / fS[..., None]) ** 2 <= WSD_CUT)
        num = np.where(w, S * S, 0.0).sum(axis=2)
        den = w.sum(axis=2)
        wsd = np.sqrt(WSD_C * _div(num, den))
        goodw = np.isfinite(wsd) & (wsd > 0)
        wsd = np.where(goodw, wsd, np.nan)
        f = wsd / np.sqrt(np.nanmean(wsd * wsd, axis=1))[:, None]
        f = np.where(goodw, f, 1.0)
    return f, (~goodw).sum(axis=1)


def _periodicity(prep: _Prep, g: int, mode: str, p: dict) -> dict:
    """WSD periodicity factors for the output rows, estimated from standardized returns r/s_day of a window of
    days. pf_window='trailing' (default): the pf_days days before day t (no look-ahead; MSS 2012 use the
    previous 100 days for their diurnal pattern, OWN transfer to WSD); 'full': all days of the panel (BCL, DHI;
    look-ahead). Days with less than pf_min_nonzero non-zero returns are not used (Aysan, Caporin & Cepni 2024,
    sec. 3: 25%). With fewer than pf_min_days usable days (default: the full 100) the factors are 1, i.e. the test
    runs unfiltered (OWN). Measured cost of the short window (5 Oct, no jumps, flat volatility, nominal 5%): R18
    rejects 10.0% with 100 trailing days, 7.1% with 250, 6.2% with 365 and 4.9% with the full sample."""
    key = ("pf", g, mode, p["pf_window"], p["pf_days"], p["pf_min_days"], p["pf_cells"], p["pf_min_nonzero"])

    def mk():
        rf = prep.rmask(g, 0, mode, full=True)
        s = _bcl_scale(rf)
        with np.errstate(invalid="ignore", divide="ignore"):
            rbar = rf / s[:, None]
        nz = _div(((rf != 0) & np.isfinite(rf)).sum(1), np.isfinite(rf).sum(1))
        elig = np.isfinite(s) & (s > 0) & (nz >= p["pf_min_nonzero"])
        rbar[~elig] = np.nan
        D, N = rbar.shape
        rows = prep.rows
        if p["pf_window"] == "full":
            if p["pf_cells"] != "tod":
                raise ValueError("pf_cells='tod_dow' is implemented for the trailing window only")
            f1, ng = _shorth_wsd(rbar.T[None, :, :])
            ok = elig.sum() >= p["pf_min_days"]
            f = np.repeat(f1 if ok else np.ones((1, N)), rows.size, axis=0)
            return dict(f=f, avail=np.full(rows.size, ok), ndays=np.full(rows.size, int(elig.sum())),
                        nguard=np.repeat(ng, rows.size), s=s[rows])
        if p["pf_window"] != "trailing":
            raise ValueError("pf_window must be 'trailing' or 'full'")
        _require_contiguous(prep.data, "periodicity window")
        k = int(p["pf_days"])
        lags = np.arange(1, k + 1) if p["pf_cells"] == "tod" else 7 * np.arange(1, k // 7 + 1)
        pad = np.vstack([np.full((1, N), np.nan), rbar])
        f = np.ones((rows.size, N))
        avail = np.zeros(rows.size, bool)
        nd = np.zeros(rows.size, np.int64)
        ngd = np.zeros(rows.size, np.int64)
        chunk = max(1, int(3e6 // (N * lags.size)))
        for a in range(0, rows.size, chunk):
            t = rows[a:a + chunk]
            idx = t[:, None] - lags[None, :]
            idx = np.where(idx >= 0, idx + 1, 0)
            Sx = pad[idx]                                   # (B, K, N)
            ndays = np.isfinite(Sx).any(axis=2).sum(axis=1)
            fb, ng = _shorth_wsd(np.transpose(Sx, (0, 2, 1)))
            ok = ndays >= p["pf_min_days"]
            f[a:a + chunk] = np.where(ok[:, None], fb, 1.0)
            avail[a:a + chunk] = ok
            nd[a:a + chunk] = ndays
            ngd[a:a + chunk] = np.where(ok, ng, 0)
        return dict(f=f, avail=avail, ndays=nd, nguard=ngd, s=s[rows])
    return prep.cached(key, mk)


def _trailing_sums(X: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    """Sum of X over the k days before each output row (rows of X = panel days)."""
    cs = np.vstack([np.zeros((1,) + X.shape[1:]), np.cumsum(X, axis=0)])
    lo = np.maximum(rows - k, 0)
    return cs[rows] - cs[lo]


def _ext_returns(prep: _Prep, g: int, mode: str, nback: int) -> np.ndarray:
    """Returns of days t-nback..t concatenated for each output row t (NaN for days outside the panel)."""
    _require_contiguous(prep.data, "backward rolling window")
    rf = prep.rmask(g, 0, mode, full=True)
    D, N = rf.shape
    pad = np.vstack([np.full((1, N), np.nan), rf])
    idx = prep.rows[:, None] - np.arange(nback, -1, -1)[None, :]
    idx = np.where(idx >= 0, idx + 1, 0)
    return pad[idx].reshape(prep.n, (nback + 1) * N)


def _r24(prep: _Prep, g: int, flags: np.ndarray) -> dict:
    """R24, 1-minute concentration of flagged g-minute intervals (OWN diagnostic, motivated by COP 2014 and COR
    2022; reporting only): share of the largest squared 1-minute return in the interval and sign agreement
    |sum sign| / #non-zero of its 1-minute returns, averaged over the day's flagged intervals."""
    if g == 1:
        nan = np.full(prep.n, np.nan)
        return dict(r24_n_flagged=flags.sum(1), r24_maxshare_mean=nan, r24_maxshare_min=nan, r24_sign_agree_mean=nan)
    x = prep.r(1, 0).reshape(prep.n, MIN_PER_DAY // g, g)
    x2 = x * x
    with np.errstate(invalid="ignore", divide="ignore"):
        share = x2.max(axis=2) / x2.sum(axis=2)
        nzc = (x != 0).sum(axis=2)
        agree = np.abs(np.sign(x).sum(axis=2)) / nzc
    fl = flags & np.isfinite(share)
    cnt = fl.sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return dict(r24_n_flagged=flags.sum(1),
                    r24_maxshare_mean=_div(np.where(fl, share, 0).sum(1), cnt),
                    r24_maxshare_min=np.where(cnt > 0, np.nanmin(np.where(fl, share, np.nan), axis=1), np.nan),
                    r24_sign_agree_mean=_div(np.where(fl & np.isfinite(agree), agree, 0).sum(1),
                                             (fl & np.isfinite(agree)).sum(1)))


# =====================================================================================================
# 6. Approaches
# =====================================================================================================
def _crit_normal(p):
    return float(norm.ppf(p["alpha"]))


# --- RV targets ---------------------------------------------------------------------------------------
def _ap_rv01(prep, p):
    g = p["grid_step"]
    r = prep.r(g, 0)
    return _rv_only(np.sum(r * r, axis=1), np.full(prep.n, r.shape[1]))


def _offsets(p, g):
    o = p.get("offsets", "all")
    offs = list(range(g)) if (o is None or (isinstance(o, str) and o == "all")) else sorted({int(x) for x in o})
    if 0 not in offs or any(not 0 <= s < g for s in offs):
        raise ValueError(f"offsets must be minutes 0..{g - 1} and include 0, got {offs}")
    return offs


def _ss(prep, p) -> dict:
    """Subsampled RV, staggered BV and TQ over grid offsets (C02/RV02), per edge rule, plus per-offset
    ABD statistics Z_s (T02). E1: pooled sums rescaled by counts (MFE realized_* code [FT-code]); E2: RV with
    the two short edge pieces (Patton-Sheppard 2015 p. 686 analogue; products as E1); E3: each grid scaled to
    the full day (ADS 2012 NBER WP p. 12 [FT-v]). In split mode N_ref = observed returns on the base grid."""
    g, mode, edge = p["grid_step"], p["outage"], p["edge_rule"]
    offs = _offsets(p, g)
    key = ("ss", g, mode, edge, tuple(offs))

    def mk():
        N0 = MIN_PER_DAY // g
        per = []
        for s in offs:
            r_all = prep.r(g, s)
            r = prep.rmask(g, s, mode)
            n_s = _nobs(r)
            rva = np.sum(r_all * r_all, axis=1)
            rvo = realized_variance(r) if mode == "split" else rva
            a = np.abs(r)
            bvs, bvc = _sc(_lagprod(a, (0, 2)))
            tqs, tqc = _sc(_lagprod(a ** (4.0 / 3.0), (0, 2, 4)))
            if edge == "E2" and s > 0:
                h, e = prep.edges(g, s)
                oh, oe = prep.obs_edges(g, s, mode)
                rva2 = rva + h * h + e * e
                rvo2 = rvo + np.where(oh, h * h, 0.0) + np.where(oe, e * e, 0.0)
            else:
                rva2, rvo2 = rva, rvo
            bv_s = MU1 ** -2 * n_s * _div(bvs, bvc)
            tq_s = MU43 ** -3 * n_s * n_s * _div(tqs, tqc)
            per.append(dict(n=n_s, Nall=r_all.shape[1], rva=rva, rvo=rvo, rva2=rva2, rvo2=rvo2, bvs=bvs, bvc=bvc,
                            tqs=tqs, tqc=tqc, bv=bv_s, tq=tq_s, z=ratio_stat(n_s, rvo, bv_s, tq_s, THETA_BV)))
        b0 = per[0]
        nref = b0["n"].astype(float)
        st = (lambda k: np.stack([d[k] for d in per]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if edge in ("E1", "E2"):
                bv = MU1 ** -2 * nref * _div(st("bvs").sum(0), st("bvc").sum(0))
                tq = MU43 ** -3 * nref ** 2 * _div(st("tqs").sum(0), st("tqc").sum(0))
                if edge == "E1":
                    rv_all = N0 * st("rva").sum(0) / sum(d["Nall"] for d in per)
                    rv_obs = nref * _div(st("rvo").sum(0), st("n").sum(0))
                else:
                    rv_all = st("rva2").mean(0)
                    rv_obs = st("rvo2").mean(0)
            elif edge == "E3":
                rv_all = np.mean([N0 * d["rva"] / d["Nall"] for d in per], axis=0)
                rv_obs = np.nanmean([nref * _div(d["rvo"], d["n"]) for d in per], axis=0)
                bv = np.nanmean([MU1 ** -2 * nref * _div(d["bvs"], d["bvc"]) for d in per], axis=0)
                tq = np.nanmean([MU43 ** -3 * nref ** 2 * _div(d["tqs"], d["tqc"]) for d in per], axis=0)
            else:
                raise ValueError("edge_rule must be E1, E2 or E3")
        if mode != "split":
            rv_obs = rv_all
        return dict(rv_all=rv_all, rv_obs=rv_obs, bv=bv, tq=tq, z=st("z"), nref=nref, n_total=st("n").sum(0),
                    offsets=offs, b0=b0, per=per)
    return prep.cached(key, mk)


def _ap_rv02(prep, p):
    ss = _ss(prep, p)
    return _rv_only(ss["rv_all"], ss["n_total"], rv0=ss["b0"]["rva"])


def _ap_rv03(prep, p):
    """Two-scale RV (ZMA 2005 eqs. 32-33, 41 [FT-v]; MFE rvtsD [FT-code]): slow scale K (default = grid_step)
    in 1-minute returns, [Y,Y]^avg = mean over the K offset grids of the full-interval RVs (no edge rescaling,
    as ZMA), nbar = (n - K + 1)/K, TSRV_adj = ([Y,Y]^avg - (nbar/n) RV1) / (1 - nbar/n)."""
    x = prep.r(1, 0)
    n = x.shape[1]
    rv1 = np.sum(x * x, axis=1)
    if p["K_rule"] == "fixed":
        K = np.full(prep.n, int(p["K"] or p["grid_step"]))
    elif p["K_rule"] == "mfe":
        # MFE K* = ceil(c* n^(2/3)), c* = (12 w^4 / IQ)^(1/3), w^2 = RV_dense/(2 #nonzero), IQ = RV_sparse^2 [FT-code];
        # RV_sparse = RV on grid_step (OWN mapping); K clamped to [2, Kmax] (OWN)
        g = p["grid_step"]
        rvs = np.sum(prep.r(g, 0) ** 2, axis=1)
        w2 = _div(rv1, 2.0 * (x != 0).sum(1))
        cst = (12.0 * w2 ** 2 / rvs ** 2) ** (1.0 / 3.0)
        K = np.clip(np.ceil(cst * n ** (2.0 / 3.0)), 2, p["K_max"]).astype(int)
    else:
        raise ValueError("K_rule must be 'fixed' or 'mfe'")
    if np.any(K < 2):
        raise ValueError("TSRV needs a slow scale K >= 2 (grid_step 1 is not meaningful here)")
    tsrv = np.full(prep.n, np.nan)
    avg_out = np.full(prep.n, np.nan)
    for k in np.unique(K):
        sel = K == k
        P = prep.P()[sel]
        rvk = [np.sum((100.0 * np.diff(P[:, s::k], axis=1)) ** 2, axis=1) for s in range(k)]
        avg = np.mean(rvk, axis=0)
        nbar = (n - k + 1) / k
        tsrv[sel] = (avg - (nbar / n) * rv1[sel]) / (1.0 - nbar / n)
        avg_out[sel] = avg
    return _rv_only(tsrv, np.full(prep.n, n), rv1=rv1, rv_avg=avg_out, K=K)


def _parzen(x):
    x = np.abs(x)
    return np.where(x <= 0.5, 1 - 6 * x ** 2 + 6 * x ** 3, np.where(x <= 1.0, 2 * (1 - x) ** 3, 0.0))


def _ap_rv05(prep, p):
    """Realized kernel, Parzen, non-flat-top, jittered end points (BNHLS 2009 eq. 2-3, PDF pp. 5-7 [FT-v]).
    K = gamma_0 + 2 sum_{h=1}^H k(h/(H+1)) gamma_h, gamma_h = sum_j x_j x_{j-h}; H = ceil(c* xi^(4/5) n^(3/5)),
    xi^2 = omega^2 / IV_sparse, omega^2 = mean_i RV_dense^(i) / (2 n_nz^(i)) with every q-th price ~2 minutes
    apart, IV_sparse = 20-minute RV subsampled over 1-minute offsets (OWN mapping for klines); jitter: the
    first and last m prices are averaged (indexing MEM)."""
    b = int(p["grid_step"])
    m = int(p["jitter_m"])
    Pb = 100.0 * prep.P()[:, ::b]              # log prices (percent) every b minutes, 1440/b + 1 points
    X = np.concatenate([Pb[:, :m].mean(1, keepdims=True), Pb[:, m:-m], Pb[:, -m:].mean(1, keepdims=True)], axis=1)
    x = np.diff(X, axis=1)
    n = x.shape[1]
    q = max(1, int(round(2.0 / b)))
    om = []
    for i in range(q):
        rr = np.diff(Pb[:, i::q], axis=1)
        om.append(_div(np.sum(rr * rr, axis=1), 2.0 * (rr != 0).sum(1)))
    omega2 = np.mean(om, axis=0)
    Pm = 100.0 * prep.P()
    iv_sp = np.mean([np.sum(np.diff(Pm[:, s::20], axis=1) ** 2, axis=1) * (72.0 / (len(range(s, 1441, 20)) - 1))
                     for s in range(20)], axis=0)
    xi2 = _div(omega2, iv_sp)
    H = np.ceil(RK_CSTAR * np.sqrt(xi2) ** 0.8 * n ** 0.6)
    H = np.where(np.isfinite(H), H, 1).astype(int)
    H = np.clip(H, 1, n // 2)
    Hmax = int(H.max()) if H.size else 1
    rk = np.sum(x * x, axis=1)
    for h in range(1, Hmax + 1):
        gh = np.sum(x[:, h:] * x[:, :-h], axis=1)
        wgt = np.where(h <= H, _parzen(h / (H + 1.0)), 0.0)
        rk = rk + 2.0 * wgt * gh
    return _rv_only(rk, np.full(prep.n, n), H=H, omega2=omega2)


# --- range-based (RV09, C08, X06) ------------------------------------------------------------------------
# lambda_{r,m} = E[s^r], s = range of m equally spaced observations of a standard BM on [0, 1] (CP 2007 eq. 1.1;
# finite-m values by simulation as in MFE realized_range_simulation.m [FT-code]). Table m = 2..64 and a fit
# lambda_r (1 + a/sqrt(m-1) + b/(m-1)) beyond (OWN simulation, see simulate_range_lambdas). Filled below.
_LAM_TABLE = {}
_LAM_FIT = {}


def range_lambda(r: float, m) -> np.ndarray:
    """lambda_{r,m} for arrays of m (number of observed prices in the interval); m <= 1 returns 1 (empty range)."""
    m = np.asarray(m, dtype=float)
    key = min(_LAM_TABLE, key=lambda k: abs(k - r)) if _LAM_TABLE else None
    if key is None or abs(key - r) > 1e-9:
        raise KeyError(f"no lambda table for r = {r}")
    tab = np.asarray(_LAM_TABLE[key])            # index 0 -> m = 2
    a, b = _LAM_FIT[key]
    lam_inf = RANGE_LAMBDA[min(RANGE_LAMBDA, key=lambda k: abs(k - r))]
    mm = np.maximum(m, 2.0)
    fit = lam_inf * (1.0 + a / np.sqrt(mm - 1.0) + b / (mm - 1.0))
    idx = np.clip(mm.astype(np.int64) - 2, 0, tab.size - 1)
    out = np.where(mm <= tab.size + 1, tab[idx], fit)
    return np.where(m <= 1, 1.0, out)


def _ranges(prep, g):
    """Per g-minute interval: log range s (percent) = max(start price, highs) - min(start price, lows), from the
    1-minute klines (OWN mapping), and m = 1 + number of trades (observed prices incl. the start price)."""
    d = prep.data
    if d.high is None or d.low is None or d.n_trades is None:
        raise ValueError("range-based approaches need high, low and n_trades")
    N = MIN_PER_DAY // g
    lh = 100.0 * np.log(prep.rows_of(d.high)).reshape(prep.n, N, g).max(axis=2)
    ll = 100.0 * np.log(prep.rows_of(d.low)).reshape(prep.n, N, g).min(axis=2)
    start = 100.0 * prep.P()[:, 0:MIN_PER_DAY:g]
    s = np.maximum(lh, start) - np.minimum(ll, start)
    m = 1 + prep.rows_of(d.n_trades).reshape(prep.n, N, g).sum(axis=2)
    return s, m


def _ap_rv09(prep, p):
    """Realized range RRV = sum s_i^2 / lambda_{2,m_i} (CP 2007 eq. 1.1 [FT-v]); with mvd=True the Martens-van
    Dijk scaling RR_S,t = [sum_{l=1..q} RR_{t-l} / sum_{l=1..q} RR^D_{t-l}] RR^D_t with RR the squared daily
    range and lambda = 4 ln 2 (MvD 2007 eq. 11, q = 66 [FT-v]); fewer than mvd_min_days previous days: NaN."""
    g = p["grid_step"]
    s, m = _ranges(prep, g)
    if not p["mvd"]:
        rrv = np.sum(s * s / range_lambda(2.0, m), axis=1)
        return _rv_only(rrv, np.full(prep.n, s.shape[1]), m_median=np.median(m, axis=1))
    _require_contiguous(prep.data, "Martens-van Dijk scaling")
    full = _Prep(prep.data, np.arange(prep.data.n_days))
    sf, _ = _ranges(full, g)
    rr_intra = np.sum(sf * sf, axis=1) / RANGE_LAMBDA[2.0]
    d = prep.data
    lh = 100.0 * np.log(np.maximum(d.high.max(axis=1), d.prev_close))
    ll = 100.0 * np.log(np.minimum(d.low.min(axis=1), d.prev_close))
    rr_day = (lh - ll) ** 2 / RANGE_LAMBDA[2.0]
    q = int(p["mvd_q"])
    num = _trailing_sums(rr_day[:, None], prep.rows, q)[:, 0]
    den = _trailing_sums(rr_intra[:, None], prep.rows, q)[:, 0]
    navail = np.minimum(prep.rows, q)
    scale = np.where(navail >= p["mvd_min_days"], _div(num, den), np.nan)
    rr_t = rr_intra[prep.rows]
    return _rv_only(scale * rr_t, np.full(prep.n, s.shape[1]), mvd_scale=scale, rr_unscaled=rr_t)


def _ap_x06(prep, p):
    """Range C/J (CP 2012 eqs. 23-24, PDF p. 27 [FT-v]): RBV = n mean(s_i s_{i+1} / (lambda_1 lambda_1)),
    RTV = n mean(prod_{j=0..2} s_{i+j}^(2/3) / lambda_{2/3}); J = max(RRV - RTV, 0) without a test (CP 2012)."""
    g, mode = p["grid_step"], p["outage"]
    s, m = _ranges(prep, g)
    obs = prep.obs(g, 0, mode)
    rrv_all = np.sum(s * s / range_lambda(2.0, m), axis=1)
    so = np.where(obs, s, np.nan)
    n = obs.sum(1)
    rrv_o = np.nansum(so * so / range_lambda(2.0, m), axis=1) if mode == "split" else rrv_all
    u1 = so / range_lambda(1.0, m)
    s1, k1 = _sc(_lagprod(u1, (0, 1)))
    rbv = n * _div(s1, k1)
    u23 = so ** (2.0 / 3.0) / range_lambda(2.0 / 3.0, m)
    s3, k3 = _sc(_lagprod(u23, (0, 1, 2)))
    rtv = n * _div(s3, k3)
    jr = np.maximum(rrv_o - rtv, 0.0)
    nan = np.full(prep.n, np.nan)
    return _out(rrv_all, jr, jr > 0, nan, nan, np.nan, n, rbv=rbv, rtv=rtv)


# --- ABD block and adjacent HT (X01 = C01 + T01; A1) -------------------------------------------------------
def _abd_base(prep, p, skip=1, g=None, mode=None):
    g = p["grid_step"] if g is None else g
    mode = p["outage"] if mode is None else mode

    def mk():
        r_all = prep.r(g, 0)
        r = prep.rmask(g, 0, mode)
        n = _nobs(r)
        rv_t = np.sum(r_all * r_all, axis=1)
        rv_o = realized_variance(r) if mode == "split" else rv_t
        bv = bipower(r, skip, n)
        tq = tripower_quarticity(r, skip, n)
        return dict(rv_t=rv_t, rv_o=rv_o, bv=bv, tq=tq, z=ratio_stat(n, rv_o, bv, tq, THETA_BV), n=n)
    return prep.cached(("abd", g, mode, skip), mk)


def _ap_abd(prep, p, skip=1):
    b = _abd_base(prep, p, skip)
    crit = _crit_normal(p)
    return _out(b["rv_t"], np.maximum(b["rv_o"] - b["bv"], 0.0), b["z"] > crit, b["z"], norm.sf(b["z"]), crit,
                b["n"], bv=b["bv"], tq=b["tq"], rv_obs=b["rv_o"])


def _ap_x01(prep, p):
    return _ap_abd(prep, p, 1)


def _ap_a1(prep, p):
    return _ap_abd(prep, p, 0)


# --- subsampling: T02, X02, X03 (T03, T04), C03, C05 ---------------------------------------------------
# theta_ss = lim N0 Var(RV_ss - BV_ss)/IQ and the null quantiles of Zbar = mean_s Z_s: own simulation
# (calibrate_offset_tests; constant volatility, Gaussian 1-minute returns, n_sim days, seed); filled below.
THETA_SS_TABLE = {}
ZBAR_PROBS = ()
ZBAR_NULL = {}
CALIBRATION_INFO = {}


def theta_ss(g: int, edge: str) -> float:
    if g == 1:
        return THETA_BV
    if (g, edge) not in THETA_SS_TABLE:
        raise KeyError(f"no theta_ss for grid_step={g}, edge_rule={edge}: run calibrate_offset_tests and pass "
                       f"theta_ss=...")
    return THETA_SS_TABLE[(g, edge)]


def _zbar_null(g: int):
    if g not in ZBAR_NULL:
        raise KeyError(f"no Zbar null quantiles for grid_step={g}: run calibrate_offset_tests(grid_step={g}) and add "
                       f"its quantiles to ZBAR_NULL")
    return np.asarray(ZBAR_NULL[g])


def zbar_crit(alpha: float, g: int) -> float:
    if g == 1:
        return float(norm.ppf(alpha))
    probs = np.asarray(ZBAR_PROBS)
    q = _zbar_null(g)
    x = norm.ppf(probs)
    xa = norm.ppf(alpha)
    if xa > x[-1]:
        return float(q[-1] + (xa - x[-1]) * (q[-1] - q[-2]) / (x[-1] - x[-2]))
    return float(np.interp(xa, x, q))


def zbar_pvalue(z, g: int):
    z = np.asarray(z, float)
    if g == 1:
        return norm.sf(z)
    probs = np.asarray(ZBAR_PROBS)
    q = _zbar_null(g)
    x = norm.ppf(probs)
    xz = np.interp(z, q, x)
    hi = z > q[-1]
    lo = z < q[0]
    xz = np.where(hi, x[-1] + (z - q[-1]) * (x[-1] - x[-2]) / (q[-1] - q[-2]), xz)
    xz = np.where(lo, x[0] + (z - q[0]) * (x[1] - x[0]) / (q[1] - q[0]), xz)
    return norm.sf(xz)


def _ap_t02(prep, p):
    """T02: per-offset ABD statistics Z_0..Z_{g-1} and the agreement count A = #{s: Z_s > z_alpha} (diagnostic,
    OWN). rv, c, j, jump, stat are those of X01 (base grid)."""
    ss = _ss(prep, p)
    b = _abd_base(prep, p, 1)
    crit = _crit_normal(p)
    Z = ss["z"]
    diag = {f"z_s{s}": Z[i] for i, s in enumerate(ss["offsets"])}
    agree = (Z > crit).sum(0)
    return _out(b["rv_t"], np.maximum(b["rv_o"] - b["bv"], 0.0), b["z"] > crit, b["z"], norm.sf(b["z"]), crit,
                b["n"], agree=agree, n_offsets=np.full(prep.n, len(ss["offsets"])),
                jump_any=agree >= 1, jump_all=agree == len(ss["offsets"]), **diag)


def _ap_x02(prep, p):
    """X02 hybrid (OWN combination of sourced pieces): subsampled RV/BV/TQ, jump days from the base-grid ABD
    Z_0; J = 1{Z_0 > z} max(RV_ss - BV_ss, 0), C = RV_ss - J."""
    ss = _ss(prep, p)
    crit = _crit_normal(p)
    z0 = ss["b0"]["z"]
    return _out(ss["rv_all"], np.maximum(ss["rv_obs"] - ss["bv"], 0.0), z0 > crit, z0, norm.sf(z0), crit,
                ss["n_total"], bv_ss=ss["bv"], tq_ss=ss["tq"], rv0=ss["b0"]["rva"])


def _ap_x03(prep, p):
    """X03: offset-robust decision with subsampled magnitudes (OWN). decision='zss' (T03a): Z_ss with simulated
    theta_ss; 'zbar' (T03b): mean of Z_s against its simulated null; 'pmean' (T04): 2 x mean p-value
    (Rueschendorf 1982 / Meng 1993 / Vovk & Wang 2020 [ABS]); 'cauchy' (T04): Liu & Xie 2020 eqs. 1 and 3
    [FT-v NIH AAM sec. 2] on one-sided p-values (their Theorem 1 is stated for two-sided p-values: OWN use)."""
    g = p["grid_step"]
    ss = _ss(prep, p)
    dec = p["decision"]
    alpha = p["alpha"]
    Z = ss["z"]
    jr = np.maximum(ss["rv_obs"] - ss["bv"], 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if dec == "zss":
            th = p["theta_ss"] if p["theta_ss"] is not None else theta_ss(g, p["edge_rule"])
            stat = ratio_stat(ss["nref"], ss["rv_obs"], ss["bv"], ss["tq"], th)
            crit = float(norm.ppf(alpha))
            pval = norm.sf(stat)
            extra = dict(theta_ss=np.full(prep.n, th))
        elif dec == "zbar":
            stat = np.nanmean(Z, axis=0)
            crit = zbar_crit(alpha, g)
            pval = zbar_pvalue(stat, g)
            extra = {}
        elif dec == "pmean":
            ps = norm.sf(Z)
            pm = np.minimum(1.0, 2.0 * np.nanmean(ps, axis=0))
            stat = norm.isf(np.clip(pm, 1e-300, 1.0 - 1e-12))      # finite statistic when pm = 1
            crit = float(norm.ppf(alpha))
            pval = pm
            extra = {}
        elif dec == "cauchy":
            ps = norm.sf(Z)
            T = np.nanmean(np.tan((0.5 - ps) * np.pi), axis=0)
            stat = T
            pval = 0.5 - np.arctan(T) / np.pi
            crit = float(np.tan((alpha - 0.5) * np.pi))
            extra = {}
        else:
            raise ValueError("decision must be zss, zbar, pmean or cauchy")
    return _out(ss["rv_all"], jr, stat > crit, stat, pval, crit, ss["n_total"], bv_ss=ss["bv"], tq_ss=ss["tq"],
                z0=ss["b0"]["z"], **extra)


def _skipavg_bv(prep, p):
    """C03: average of skip-0..skip-(Q-1) BV per grid, then over offsets (PS 2015 eq. 11, p. 686 [FT]). debias
    False: PS literal, mu1^-2 sum (no finite-sample factor), simple mean over offsets; True: MFE-style factor
    (n mean form) combined by the edge rule (OWN addition)."""
    g, mode, edge = p["grid_step"], p["outage"], p["edge_rule"]
    offs = _offsets(p, g)
    Q = int(p["n_skips"])
    vals = []
    sums = np.zeros((Q, prep.n)); cnts = np.zeros((Q, prep.n))
    nref = _nobs(prep.rmask(g, 0, mode)).astype(float)
    for s in offs:
        r = prep.rmask(g, s, mode)
        n_s = _nobs(r)
        a = np.abs(r)
        per_q = []
        for q in range(Q):
            sq, kq = _sc(_lagprod(a, (0, q + 1)))
            if p["debias"]:
                sums[q] += sq; cnts[q] += kq
                per_q.append(MU1 ** -2 * nref * _div(sq, kq))
            else:
                per_q.append(MU1 ** -2 * (n_s - 1 - q) * _div(sq, kq))
        vals.append(np.mean(per_q, axis=0))
    if p["debias"] and edge in ("E1", "E2"):
        return np.mean([MU1 ** -2 * nref * _div(sums[q], cnts[q]) for q in range(Q)], axis=0)
    return np.mean(vals, axis=0)


def _ap_c03(prep, p):
    """C03 with the X02 decision (OWN combination): J = 1{Z_0 > z} max(RV_ss - BV_PS, 0)."""
    ss = _ss(prep, p)
    bvps = _skipavg_bv(prep, p)
    crit = _crit_normal(p)
    z0 = ss["b0"]["z"]
    return _out(ss["rv_all"], np.maximum(ss["rv_obs"] - bvps, 0.0), z0 > crit, z0, norm.sf(z0), crit,
                ss["n_total"], bv_ps=bvps)


def _ap_c05(prep, p):
    """C05: subsampled MedRV or MinRV (ADS 2012 NBER WP p. 12: average over offsets, scaled to the full day
    [FT-v]) combined by the edge rule; jump days from the base-grid MedRV/MinRV ratio test (X02 analogue, OWN)."""
    g, mode, edge = p["grid_step"], p["outage"], p["edge_rule"]
    kind = p["kind"]
    offs = _offsets(p, g)
    ss = _ss(prep, p)
    nref = ss["nref"]
    s2s, cs, s4s = [], [], []
    for s in offs:
        r = prep.rmask(g, s, mode)
        a = np.abs(r)
        m = _med3(a[:, :-2], a[:, 1:-1], a[:, 2:]) if kind == "med" else np.minimum(a[:, :-1], a[:, 1:])
        s2, k = _sc(m * m)
        s4, _ = _sc(m ** 4)
        s2s.append(s2); cs.append(k); s4s.append(s4)
    C2, CQ, V = (MEDRV_C, MEDRQ_C, V_MED) if kind == "med" else (MINRV_C, MINRQ_C, V_MIN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if edge == "E3":
            xss = np.nanmean([C2 * nref * _div(a, b) for a, b in zip(s2s, cs)], axis=0)
        else:
            xss = C2 * nref * _div(np.sum(s2s, 0), np.sum(cs, 0))
    r0 = prep.rmask(g, 0, mode)
    n0 = _nobs(r0)
    x0, q0 = (medrv(r0, n0) if kind == "med" else minrv(r0, n0))
    rv0 = ss["b0"]["rvo"]
    z0 = ratio_stat(n0, rv0, x0, q0, V)
    crit = _crit_normal(p)
    return _out(ss["rv_all"], np.maximum(ss["rv_obs"] - xss, 0.0), z0 > crit, z0, norm.sf(z0), crit,
                ss["n_total"], x_ss=xss, x0=x0)


# --- blocked bipower (X04 = C04 + T06) -------------------------------------------------------------------
def _ap_x04(prep, p):
    """Blocked bipower of Mykland, Shephard & Sheppard (2012, Oxford DP 593 [FT-v]): blocks of M base returns
    (base step = grid_step / M minutes; grid_step 5, M 5: 1-minute returns, 288 blocks), RV_i = block sums of
    squares (eq. 10), BV_M = n/((n_M - 1) k_{M,1}^2) sum (RV_{i-1} RV_i)^(1/2) (eq. 12), IQ by blocked tripower
    MV^(4,3)_M (eq. 17), J_M = sqrt(n)(RV - BV_M)/sqrt((M theta_M - 2) IQ) (eq. 15, form='linear'; 'ratio' is OWN).
    target: 'rv_base' literal MSS (RV of the base returns), 'rv_grid' RV on grid_step (task variant),
    'rv_ss' subsampled RV (catalogue variant, OWN). diurnal=True: the test uses returns divided by trailing WSD
    periodicity factors (MSS diurnal standardisation with BCL's estimator, OWN combination)."""
    g, mode, M = p["grid_step"], p["outage"], int(p["M"])
    if g % M:
        raise ValueError(f"grid_step {g} is not a multiple of M = {M}")
    b = g // M
    x = prep.r(b, 0)
    n_base = x.shape[1]
    nM = n_base // M
    blk = (x * x).reshape(prep.n, nM, M).sum(axis=2)
    obsb = prep.obs(g, 0, mode)
    blk_o = np.where(obsb, blk, np.nan)
    n_obs = M * obsb.sum(axis=1)
    rv_base = blk.sum(axis=1)
    rv_obs = np.nansum(blk_o, axis=1)
    k1, k43, mth = mss_k(M, 1.0), mss_k(M, 4.0 / 3.0), mss_mtheta(M)

    def bvm_mv(bk):
        s, k = _sc(np.sqrt(_lagprod(bk, (0, 1))))
        s3, k3 = _sc(_lagprod(bk ** (2.0 / 3.0), (0, 1, 2)))
        return n_obs * _div(s, k) / k1 ** 2, n_obs ** 2 * _div(s3, k3) / k43 ** 3
    bvm, mv43 = bvm_mv(blk_o)
    if p["diurnal"]:
        pf = _periodicity(prep, b, mode, p)
        xs = np.where(prep.obs(b, 0, mode), x, np.nan) / pf["f"]
        bs = (xs * xs).reshape(prep.n, nM, M).sum(axis=2)
        bs = np.where(obsb, bs, np.nan)
        rv_t = np.nansum(bs, axis=1)
        bv_t, mv_t = bvm_mv(bs)
        extra = dict(pf_available=pf["avail"])
    else:
        rv_t, bv_t, mv_t = rv_obs, bvm, mv43
        extra = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        if p["form"] == "linear":
            stat = np.sqrt(n_obs) * (rv_t - bv_t) / np.sqrt((mth - 2.0) * mv_t)
        elif p["form"] == "ratio":
            stat = ratio_stat(n_obs, rv_t, bv_t, mv_t, mth - 2.0)
        else:
            raise ValueError("form must be 'linear' or 'ratio'")
    tg = p["target"]
    if tg == "rv_base":
        rv, jr = rv_base, np.maximum(rv_obs - bvm, 0.0)
    elif tg == "rv_grid":
        rg_all = prep.r(g, 0)
        rg = prep.rmask(g, 0, mode)
        rv, jr = np.sum(rg_all * rg_all, axis=1), np.maximum(realized_variance(rg) - bvm, 0.0)
    elif tg == "rv_ss":
        ss = _ss(prep, p)
        rv, jr = ss["rv_all"], np.maximum(ss["rv_obs"] - bvm, 0.0)
    else:
        raise ValueError("target must be rv_base, rv_grid or rv_ss")
    crit = _crit_normal(p)
    zb = _div(((blk_o == 0)).sum(1), np.isfinite(blk_o).sum(1))
    return _out(rv, jr, stat > crit, stat, norm.sf(stat), crit, n_obs, bv_m=bvm, iq_m=mv43, rv_base=rv_base,
                zero_block_share=zb, mtheta=np.full(prep.n, mth), **extra)


# --- CPR threshold bipower (R02, R03) --------------------------------------------------------------------
def cpr_local_variance(r2: np.ndarray, L: int = 25, cV: float = 3.0, n_iter: Optional[int] = None,
                       max_iter: int = 100):
    """Iterated local-variance filter of Corsi, Pirino & Reno (2010) App. B, eq. B.1 (AAM p. 29 [FT-v]):
    Gaussian kernel K(i/L), L = 25, neighbours i = -L..L except -1, 0, 1, a neighbour enters only if
    r^2_{j+i} <= cV^2 V_{j+i} (previous pass, at the neighbour's own position), V^(0) = +inf.
    n_iter=None: iterate until the set of removed returns no longer changes (CPR's stop rule; at most max_iter);
    n_iter=k: exactly k passes (the 5 Oct script used 3). NaN entries of r2 are never used.
    Returns (V, passes per row, converged per row)."""
    n, N = r2.shape
    offs = [i for i in range(-L, L + 1) if i not in (-1, 0, 1)]
    w = [math.exp(-0.5 * (i / L) ** 2) for i in offs]
    fin = np.isfinite(r2)
    r2z = np.where(fin, r2, 0.0)
    V = np.full((n, N), np.inf)
    iters = np.zeros(n, np.int64)
    conv = np.zeros(n, bool)
    active = np.arange(n)
    incl_prev = None
    passes = n_iter if n_iter is not None else max_iter
    for it in range(passes):
        if active.size == 0:
            break
        Va = V[active]
        ra = r2z[active]
        with np.errstate(invalid="ignore"):
            incl = fin[active] & (ra <= cV ** 2 * Va)
        if n_iter is None and it > 0:
            same = np.all(incl == incl_prev, axis=1)
            conv[active[same]] = True
            keep = ~same
            active, incl, ra = active[keep], incl[keep], ra[keep]
            if active.size == 0:
                break
        x = np.where(incl, ra, 0.0)
        xi = incl.astype(float)
        num = np.zeros_like(x)
        den = np.zeros_like(x)
        for i, wi in zip(offs, w):
            if i > 0:
                num[:, :-i] += wi * x[:, i:]
                den[:, :-i] += wi * xi[:, i:]
            else:
                num[:, -i:] += wi * x[:, :i]
                den[:, -i:] += wi * xi[:, :i]
        V[active] = _div(num, den)
        iters[active] += 1
        incl_prev = incl
    if n_iter is not None:
        conv[:] = False
    return V, iters, conv


def _cpr_core(prep, p, r, r_prev=None):
    """Threshold, corrected multipowers and C-Tz (CPR 2010 eqs. 2.14-2.15, 3.2-3.5, A.5; Table 1 note for the
    finite-sample factors N/(N - (M-1) - k) [FT-v])."""
    L, cV, ct = int(p["L"]), float(p["c_V"]), float(p["c_theta"])
    if r_prev is not None:
        R = np.concatenate([r_prev, r], axis=1)
        V, iters, conv = cpr_local_variance(R * R, L, cV, p["n_iter"], p["max_iter"])
        V = V[:, r_prev.shape[1]:]
    else:
        V, iters, conv = cpr_local_variance(r * r, L, cV, p["n_iter"], p["max_iter"])
    n_guard = np.zeros(r.shape[0], np.int64)
    if p["guard"]:
        # guard (OWN): V <= 0 (stale neighbourhood) or 0/0 is replaced by the day's mean of the positive V
        bad = ~(V > 0) & np.isfinite(r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fb = np.nanmean(np.where(V > 0, V, np.nan), axis=1)
        V = np.where(bad, fb[:, None], V)
        n_guard = bad.sum(1)
    th = ct ** 2 * V
    a = np.abs(r)
    with np.errstate(invalid="ignore"):
        exc = np.isfinite(r) & (r * r > th)
    z1 = np.where(exc, cpr_zfactor(1.0, ct) * np.sqrt(th), a)
    z43 = np.where(exc, cpr_zfactor(4.0 / 3.0, ct) * th ** (2.0 / 3.0), a ** (4.0 / 3.0))
    n = _nobs(r)
    s, k = _sc(_lagprod(z1, (0, 1)))
    ctbpv = MU1 ** -2 * n * _div(s, k)
    s3, k3 = _sc(_lagprod(z43, (0, 1, 2)))
    cttpv = MU43 ** -3 * n * n * _div(s3, k3)
    s0, k0 = _sc(_lagprod(np.where(exc, np.nan, a), (0, 1)))
    tbpv = MU1 ** -2 * n * _div(s0, k0)
    rv_o = realized_variance(r)
    return dict(ctbpv=ctbpv, cttpv=cttpv, tbpv=tbpv, z=ratio_stat(n, rv_o, ctbpv, cttpv, THETA_BV), n=n,
                rv_o=rv_o, n_exceed=exc.sum(1), passes=iters, converged=conv, n_v_guard=n_guard)


def _compress_nonzero(r: np.ndarray):
    """Move the non-zero entries of each row to the left (order kept); exact zeros are removed, NaN kept."""
    zero = r == 0
    order = np.argsort(zero, axis=1, kind="stable")
    rc = np.take_along_axis(r, order, axis=1)
    nkeep = (~zero).sum(axis=1)
    rc[np.arange(r.shape[1])[None, :] >= nkeep[:, None]] = np.nan
    return rc, order, nkeep


def _ap_cpr(prep, p):
    """R02 (variant='published'): C-Tz test, TJ = 1{C-Tz > z} (RV - TBPV)^+ (CPR eq. 5.7 [FT-v]), filter to
    convergence, V guard. variant='script': the 5 Oct script (3 passes, no guard, J from C-TBPV).
    R03 (nonzero=True): CPR sec. 5.4 no-trade variant, all quantities on the non-zero returns (V recomputed
    after deleting zeros: OWN reading)."""
    g, mode = p["grid_step"], p["outage"]
    r_all = prep.r(g, 0)
    r = prep.rmask(g, 0, mode)
    q = dict(p)
    if p["variant"] == "script":
        q.update(n_iter=3, guard=False)
    r_prev = None
    if p["edge_window"] == "back":
        _require_contiguous(prep.data, "CPR back window")
        rf = prep.rmask(g, 0, mode, full=True)
        pad = np.vstack([np.full((1, rf.shape[1]), np.nan), rf])
        r_prev = pad[np.where(prep.rows >= 1, prep.rows, 0)][:, -int(p["L"]):]     # pad[t] = day t-1
    elif p["edge_window"] != "day":
        raise ValueError("edge_window must be 'day' or 'back'")
    if p["nonzero"]:
        r, _, _ = _compress_nonzero(r)
        if r_prev is not None:
            r_prev, _, _ = _compress_nonzero(r_prev[:, ::-1])
            r_prev = r_prev[:, ::-1]
    cp = _cpr_core(prep, q, r, r_prev)
    rv_t = np.sum(r_all * r_all, axis=1)
    rv_o = cp["rv_o"] if mode == "split" else rv_t
    crit = _crit_normal(p)
    xj = cp["ctbpv"] if p["variant"] == "script" else cp["tbpv"]
    return _out(rv_t, np.maximum(rv_o - xj, 0.0), cp["z"] > crit, cp["z"], norm.sf(cp["z"]), crit, cp["n"],
                ctbpv=cp["ctbpv"], cttpv=cp["cttpv"], tbpv=cp["tbpv"], n_exceed=cp["n_exceed"],
                filter_passes=cp["passes"], filter_converged=cp["converged"], n_v_guard=cp["n_v_guard"])


# --- MedRV / MinRV (R04, R05) ------------------------------------------------------------------------------
def _ap_nn(prep, p):
    g, mode = p["grid_step"], p["outage"]
    r_all = prep.r(g, 0)
    r = prep.rmask(g, 0, mode)
    n = _nobs(r)
    rv_t = np.sum(r_all * r_all, axis=1)
    rv_o = realized_variance(r) if mode == "split" else rv_t
    if p["kind"] == "med":
        x, qq = medrv(r, n)
        v = V_MED
    else:
        x, qq = minrv(r, n)
        v = V_MIN
    z = ratio_stat(n, rv_o, x, qq, v)
    crit = _crit_normal(p)
    return _out(rv_t, np.maximum(rv_o - x, 0.0), z > crit, z, norm.sf(z), crit, n, x_nn=x, q_nn=qq)


# --- Kolokolov-Reno staleness-robust multipowers (R06) -----------------------------------------------------
def kr_multipowers(r: np.ndarray, N: int) -> dict:
    """Staleness-robust multipower variation of Kolokolov & Reno (2024, JBES 42(2), 516-532) [FT]:
    stochastic grid of the non-zero returns tau_0 < tau_1 < ... < tau_{N_n} (p. 518; tau_0 = start of the day,
    whose price is the previous day's 23:59 close), durations Delta(n,i) = tau_i - tau_{i-1}, and (eq. 2.10, p. 519)
        MV^c(X; r) = Dbar_n sum_{i=1}^{N_n-m+1} |Delta(n,i)^(-1/2) Delta_{tau_i} X|^{r_1} ...
                     |Delta(n,i+m-1)^(-1/2) Delta_{tau_{i+m-1}} X|^{r_m},
    with Dbar_n the sample mean of the durations (p. 519). In grid units (D_i intervals, u_i = r_i/sqrt(D_i)):
    RV_c = MV^c([2]) = Dbar sum u_i^2; BV_c = mu1^-2 MV^c([1,1]) = mu1^-2 Dbar sum |u_i||u_{i+1}|;
    TQ_c = mu43^-3 MV^c([4/3,4/3,4/3]) = mu43^-3 N Dbar sum prod |u|^(4/3). No finite-sample factors (as eq.
    2.10). Without zeros all three equal the standard multipowers. NaN entries (outage='split') break products;
    sums are then rescaled to N_nz - m + 1 terms (OWN)."""
    nd, Nn = r.shape
    rc, order, nkeep = _compress_nonzero(r)
    pos = np.where(np.arange(Nn)[None, :] < nkeep[:, None], (order + 1).astype(float), np.nan)
    prev = np.concatenate([np.zeros((nd, 1)), pos[:, :-1]], axis=1)
    dur = pos - prev
    with np.errstate(invalid="ignore", divide="ignore"):
        u = rc / np.sqrt(dur)
    fin = np.isfinite(u)
    NM = fin.sum(axis=1)
    dbar = _div(np.where(fin, dur, 0.0).sum(axis=1), NM)
    s2, k2 = _sc(u * u)
    rvc = dbar * NM * _div(s2, k2)
    s, k = _sc(_lagprod(np.abs(u), (0, 1)))
    bvc = MU1 ** -2 * dbar * (NM - 1) * _div(s, k)
    s3, k3 = _sc(_lagprod(np.abs(u) ** (4.0 / 3.0), (0, 1, 2)))
    tqc = MU43 ** -3 * N * dbar * (NM - 2) * _div(s3, k3)
    return dict(rv_c=rvc, bv_c=bvc, tq_c=tqc, n_nonzero=NM, mean_duration=dbar)


def rounding_impact_ratio(prep, g: int, back_days: int = 5, d_fixed=None) -> np.ndarray:
    """Rounding Impact Ratio of KR 2024 eq. 6.1, p. 527 [FT]: RIR = d / sqrt(Delta_n sum_i (P_i - P_{i-1})^2) with
    price levels P on the grid (Delta_n = 1/n) and d the most frequent non-zero absolute 1-minute price difference
    of the day and the previous 5 days, rounded to 6 significant digits (p. 527). Earlier days that are not in the
    panel are skipped (OWN). d_fixed (scalar or one value per selected day, price units): use the known tick
    instead (OWN option). The mode rule recovers the tick only when most 1-minute price changes are a few ticks;
    for BTCUSDT on 13 Aug 2021 the 8,361 non-zero changes took 5,013 distinct values and the mode (10.01 USDT,
    9 times) is arbitrary (5 Oct check), so use d_fixed for high-priced coins."""
    d = prep.data
    rows = prep.rows
    lo = np.maximum(rows - (back_days if d.contiguous else 0), 0)
    full_close = d.close
    out = np.full(prep.n, np.nan)
    dfix = None if d_fixed is None else np.broadcast_to(np.asarray(d_fixed, float), (prep.n,))
    for i_, (a, t) in enumerate(zip(lo, rows)):
        if dfix is not None:
            dhat = dfix[i_]
        else:
            seg = np.concatenate([[d.prev_close[a]], full_close[a:t + 1].ravel()])
            dif = np.abs(np.diff(seg))
            dif = dif[dif > 0]
            if dif.size == 0:
                continue
            e = np.floor(np.log10(dif))
            sc = 10.0 ** (5 - e)
            rd = np.round(dif * sc) / sc
            vals, cnt = np.unique(rd, return_counts=True)
            dhat = vals[np.argmax(cnt)]
        P = np.concatenate([[d.prev_close[t]], full_close[t, g - 1::g]])
        n = P.size - 1
        den = math.sqrt(np.sum(np.diff(P) ** 2) / n)
        out[i_] = dhat / den if den > 0 else np.nan
    return out


def _ap_r06(prep, p):
    """R06: staleness-corrected BNS test of Kolokolov & Reno (2024) [FT]: eq. 3.5 (p. 520) with the
    staleness-robust multipowers in place of the standard ones (p. 520), i.e.
        Z = (1 - BV_c/RV_c) / sqrt((theta/n) max(1, TQ_c/BV_c^2)),  N(0,1) under no jumps,
    n = number of grid returns (the BNS asymptotics 'need not to be changed', p. 519). KR print theta ~ 2.61 next
    to unnormalised multipowers, which would not centre the statistic; implemented in the Huang-Tauchen form they
    cite (normalised multipowers, theta = 0.609; their Fig. 2 is centred at 0 with 5-95% quantiles near +-1.6).
    Jump measure JV = RV_c - BV_c with both quantities corrected (p. 526 and Fig. 6E: correcting only BPV is
    worse; p. 529). target='rv_c' (default): C = RV_c - J; target='rv': C = RV - J with J capped at RV (OWN).
    norm='Nnz' uses sqrt(N_nz) (OWN variant). kr_day_ok: KR compute only on days with more than 20 returns and
    less than 80% zeros (p. 526); day_filter=True applies it (no test on failing days). rir: eq. 6.1 diagnostic;
    KR recommend eq. 2.10 when the RIR is low (p. 526); their rounding correction is in the online supplement
    (Appendix C), which is not available, so it is not implemented."""
    g, mode = p["grid_step"], p["outage"]
    r_all = prep.r(g, 0)
    r = prep.rmask(g, 0, mode)
    N = r.shape[1]
    kr = kr_multipowers(r, N)
    rvc, bvc, tqc, NM = kr["rv_c"], kr["bv_c"], kr["tq_c"], kr["n_nonzero"]
    n_obs = _nobs(r)
    neff = NM if p["norm"] == "Nnz" else n_obs
    z = ratio_stat(neff, rvc, bvc, tqc, THETA_BV)
    zs = prep.zero_share(g, mode)
    ok = (n_obs > 20) & (zs < 0.8)
    if p["day_filter"]:
        z = np.where(ok, z, np.nan)
    crit = _crit_normal(p)
    rv_t = np.sum(r_all * r_all, axis=1)
    rv_o = realized_variance(r) if mode == "split" else rv_t
    if p["j_from"] == "corrected":
        jr = np.maximum(rvc - bvc, 0.0)
    elif p["j_from"] == "mixed":
        jr = np.maximum(rv_o - bvc, 0.0)
    else:
        raise ValueError("j_from must be 'corrected' or 'mixed'")
    if p["target"] == "rv_c":
        rv = rvc
    elif p["target"] == "rv":
        rv = rv_t
        jr = np.minimum(jr, rv_t)
    else:
        raise ValueError("target must be 'rv_c' or 'rv'")
    diag = dict(rv_c=rvc, bv_c=bvc, tq_c=tqc, rv_raw=rv_t, mean_duration=kr["mean_duration"], kr_day_ok=ok)
    if p["rir"]:
        diag["rir"] = rounding_impact_ratio(prep, g, d_fixed=p["rir_d"])
    return _out(rv, jr, z > crit, z, norm.sf(z), crit, NM, **diag)


# --- periodicity-filtered tests (R07 daily, R18 return level) -----------------------------------------------
def _ap_r07(prep, p):
    """R07: periodicity-filtered returns r/f_WSD (DHI 2025 eq. 6, App. B [FT]); RV^f, adjacent BV^f with N/(N-1),
    adjacent TQ^f, BNS/HT ratio test with 0.61; J^f = 1{Z^f > z}(RV^f - BV^f)^+. rv = RV^f (DHI's filtered
    measure); rv_raw = unfiltered RV, DHI's forecast target. Factors from the trailing window (OWN, no look-ahead)."""
    g, mode = p["grid_step"], p["outage"]
    r = prep.rmask(g, 0, mode)
    pf = _periodicity(prep, g, mode, p)
    rf = r / pf["f"]
    n = _nobs(rf)
    rvf = realized_variance(rf)
    bvf = bipower(rf, 0, n)
    tqf = tripower_quarticity(rf, 0, n)
    z = ratio_stat(n, rvf, bvf, tqf, THETA_BV)
    crit = _crit_normal(p)
    r_all = prep.r(g, 0)
    return _out(rvf, np.maximum(rvf - bvf, 0.0), z > crit, z, norm.sf(z), crit, n, bv_f=bvf,
                rv_raw=np.sum(r_all * r_all, axis=1), pf_available=pf["avail"], pf_days=pf["ndays"])


def _return_level_out(prep, p, g, r_all, flags, stat, pval, crit, n_tested, **diag):
    rv = np.sum(r_all * r_all, axis=1)
    jr = np.sum(np.where(flags, r_all * r_all, 0.0), axis=1)
    d = _r24(prep, g, flags)
    d.update(diag)
    d["n_flags"] = flags.sum(axis=1)
    return _out(rv, jr, flags.any(axis=1), stat, pval, crit, n_tested, **d)


def _nanmax_rows(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(np.where(np.isfinite(x), x, -np.inf), axis=1).astype(float)


def _ap_r18(prep, p):
    """R18: BCL 2011 periodicity-filtered return-level statistic FJ_i = |r_i|/(f_WSD,i s_day) (eq. 3.4, WP p. 22
    [FT-v]) against the LM-type Gumbel threshold (n = returns per day) or the Sidak threshold at the daily size
    1 - alpha; daily J = sum of squared flagged returns, C = RV - J (OWN, analogue of BTL JV = RV - CV).
    max_zero_share < 1: days with more zero returns are not tested (Aysan et al. 2024 fn 21: 75%)."""
    g, mode = p["grid_step"], p["outage"]
    r_all = prep.r(g, 0)
    r = prep.rmask(g, 0, mode)
    N = r.shape[1]
    pf = _periodicity(prep, g, mode, p)
    with np.errstate(invalid="ignore", divide="ignore"):
        fj = np.abs(r) / (pf["f"] * pf["s"][:, None])
    size = 1.0 - p["alpha"]
    stat = _nanmax_rows(fj)
    stat[~np.isfinite(stat)] = np.nan
    if p["threshold"] == "lm":
        crit, pval = gumbel_threshold(size, N, 1.0), gumbel_pvalue(stat, N, 1.0)
    elif p["threshold"] == "sidak":
        crit, pval = sidak_threshold(size, N), sidak_pvalue(stat, N)
    else:
        raise ValueError("threshold must be 'lm' or 'sidak'")
    with np.errstate(invalid="ignore"):
        flags = np.isfinite(fj) & (fj > crit)
    zs = prep.zero_share(g, mode)
    untested = zs > p["max_zero_share"]
    flags[untested] = False
    stat[untested] = np.nan
    pval = np.where(untested, np.nan, pval)
    return _return_level_out(prep, p, g, r_all, flags, stat, pval, crit, _nobs(r), pf_available=pf["avail"],
                             pf_days=pf["ndays"], pf_guard_cells=pf["nguard"], untested_day=untested)


# --- return-level tests without periodicity filter (R16, R17, R19) ----------------------------------------
def _ap_r16(prep, p):
    """R16: Lee & Mykland (2008) L_i = r_i/sigma_i, sigma_i^2 = mean of |r_j||r_{j-1}| over the K-1 returns before
    i (Def. 1, eqs. 7-8, p. 2540 [FT]; window reaches back into earlier days, never forward), K = ceil(sqrt(252 n))
    (p. 2542); daily family: (max|L| - C_n)/S_n against beta* (eqs. 12-13, c = sqrt(2/pi)) with n = returns per
    day; J = sum of squared flagged returns (OWN)."""
    g, mode = p["grid_step"], p["outage"]
    N = MIN_PER_DAY // g
    K = int(p["K"]) if p["K"] else lm_window(N, int(p["K_year"]))
    nb = int(math.ceil((K - 1) / N))
    E = _ext_returns(prep, g, mode, nb)
    a = np.abs(E)
    pi = np.concatenate([np.full((prep.n, 1), np.nan), a[:, 1:] * a[:, :-1]], axis=1)
    fin = np.isfinite(pi)
    cs = np.concatenate([np.zeros((prep.n, 1)), np.cumsum(np.where(fin, pi, 0.0), axis=1)], axis=1)
    cc = np.concatenate([np.zeros((prep.n, 1)), np.cumsum(fin, axis=1)], axis=1)
    i = nb * N + np.arange(N)
    lo = np.maximum(i - K + 2, 0)
    ssum = cs[:, i] - cs[:, lo]
    scnt = cc[:, i] - cc[:, lo]
    sig2 = _div(ssum, scnt)
    sig2[scnt < max(2, int(p["min_frac"] * (K - 2)))] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        Lst = np.abs(E[:, i]) / np.sqrt(sig2)
    size = 1.0 - p["alpha"]
    crit = gumbel_threshold(size, N, MU1)
    stat = _nanmax_rows(Lst)
    stat[~np.isfinite(stat)] = np.nan
    with np.errstate(invalid="ignore"):
        flags = np.isfinite(Lst) & (Lst > crit)
    return _return_level_out(prep, p, g, prep.r(g, 0), flags, stat, gumbel_pvalue(stat, N, MU1), crit,
                             np.isfinite(Lst).sum(1), lm_K=np.full(prep.n, K))


def _ap_r17(prep, p):
    """R17: Andersen, Bollerslev & Dobrev (2007) eq. 19, p. 134 [FT]: flag r_s if |r_s| > Phi^-1(1 - beta/2)
    sqrt(Delta BV_t), beta = 1 - (1 - size)^Delta (Sidak); BV_t adjacent with N/(N-1); J = sum of flagged r^2 (OWN)."""
    g, mode = p["grid_step"], p["outage"]
    r = prep.rmask(g, 0, mode)
    N = r.shape[1]
    bv = bipower(r, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        st = np.abs(r) / np.sqrt(bv / N)[:, None]
    size = 1.0 - p["alpha"]
    crit = sidak_threshold(size, N)
    stat = _nanmax_rows(st)
    stat[~np.isfinite(stat)] = np.nan
    with np.errstate(invalid="ignore"):
        flags = np.isfinite(st) & (st > crit)
    return _return_level_out(prep, p, g, prep.r(g, 0), flags, stat, sidak_pvalue(stat, N), crit, _nobs(r), bv=bv)


def _ap_r19(prep, p):
    """R19: drift-robust LM test of Laurent & Shi (2020, AMSE WP 2018-43 [FT-v]): m_i = median(r_{i-K+1..i}),
    r*_j = r_j - m_i, BV*_i = c1^-2 sum_{j=i-K+2..i} |r*_j||r*_{j-1}|, sigma*_i = sqrt(BV*_i/(K-1)) (p. 17, eqs.
    26-27); J*_i = (r_i - m_i)/sigma*_i ~ N(0,1), so the LM Gumbel constants are used with c = 1 (OWN reading of
    'critical values as in LM'); window back into earlier days; J = sum of flagged r^2 (OWN)."""
    from numpy.lib.stride_tricks import sliding_window_view
    g, mode = p["grid_step"], p["outage"]
    N = MIN_PER_DAY // g
    K = int(p["K"]) if p["K"] else lm_window(N, int(p["K_year"]))
    nb = int(math.ceil((K - 1) / N))
    E = _ext_returns(prep, g, mode, nb)
    i = nb * N + np.arange(N)
    W = sliding_window_view(E, K, axis=1)            # window q covers q..q+K-1
    Jst = np.full((prep.n, N), np.nan)
    chunk = max(1, int(4e6 // (N * K)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a0 in range(0, prep.n, chunk):
            Wt = W[a0:a0 + chunk][:, i - K + 1, :]
            fin = np.isfinite(Wt)
            m = np.median(Wt, axis=2) if fin.all() else np.nanmedian(Wt, axis=2)
            Cc = np.abs(Wt - m[..., None])
            pr = Cc[..., 1:] * Cc[..., :-1]
            okp = np.isfinite(pr)
            cnt = okp.sum(axis=2)
            mp = _div(np.where(okp, pr, 0.0).sum(axis=2), cnt)
            sig = np.sqrt(MU1 ** -2 * mp)
            sig[cnt < max(2, int(p["min_frac"] * (K - 1)))] = np.nan
            Jst[a0:a0 + chunk] = np.abs(E[a0:a0 + chunk][:, i] - m) / sig
    size = 1.0 - p["alpha"]
    crit = gumbel_threshold(size, N, 1.0)
    stat = _nanmax_rows(Jst)
    stat[~np.isfinite(stat)] = np.nan
    with np.errstate(invalid="ignore"):
        flags = np.isfinite(Jst) & (Jst > crit)
    return _return_level_out(prep, p, g, prep.r(g, 0), flags, stat, gumbel_pvalue(stat, N, 1.0), crit,
                             np.isfinite(Jst).sum(1), lm_K=np.full(prep.n, K))


# --- truncation with time-of-day factor (R08) -------------------------------------------------------------
def _ap_r08(prep, p):
    """R08: Bollerslev, Todorov & Li (2013) eqs. 3.3-3.6, p. 311 [FT]: TOD_i from returns truncated at
    tau sqrt(BV ^ RV) n^-varpi, CV_t = sum r^2 1{|r| <= tau sqrt((BV_t ^ RV_t) TOD_i) n^-varpi}, JV = RV - CV
    (no test: every truncated return makes a jump day). tau = 2.5 (BTL) or 3 (BLT 2016 fn 17), varpi = 0.49.
    TOD over the trailing pf_days days (no look-ahead, OWN) or all days ('full', BTL literal)."""
    g, mode = p["grid_step"], p["outage"]
    tau, vp = float(p["tau"]), float(p["varpi"])
    rf = prep.rmask(g, 0, mode, full=True)
    D, N = rf.shape
    bmf = np.minimum(bipower(rf, 0), realized_variance(rf))
    u0 = tau * np.sqrt(bmf) * N ** (-vp)
    with np.errstate(invalid="ignore"):
        tr = np.where(np.isfinite(rf) & (np.abs(rf) <= u0[:, None]), rf * rf, 0.0)
    elig = np.isfinite(bmf) & (bmf > 0)
    tr[~elig] = 0.0
    if p["pf_window"] == "full":
        num = np.repeat(tr.sum(0)[None, :], prep.n, 0)
        nd = np.full(prep.n, int(elig.sum()))
    else:
        _require_contiguous(prep.data, "time-of-day window")
        k = int(p["pf_days"])
        num = _trailing_sums(tr, prep.rows, k)
        nd = _trailing_sums(elig[:, None].astype(float), prep.rows, k)[:, 0].astype(int)
    tod = N * _div(num, num.sum(1)[:, None])
    ok = nd >= p["pf_min_days"]
    tod[~ok] = 1.0
    r = prep.rmask(g, 0, mode)
    bm = bmf[prep.rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.sqrt(bm[:, None] * tod) * N ** (-vp)
        st = np.abs(r) / unit
    keep = np.isfinite(st) & (st <= tau)
    cv = np.sum(np.where(keep, r * r, 0.0), axis=1)
    r_all = prep.r(g, 0)
    rv_t = np.sum(r_all * r_all, axis=1)
    rv_o = realized_variance(r) if mode == "split" else rv_t
    jv = np.maximum(rv_o - cv, 0.0)
    stat = _nanmax_rows(st)
    stat[~np.isfinite(stat)] = np.nan
    return _out(rv_t, jv, jv > 0, stat, np.full(prep.n, np.nan), tau, _nobs(r), cv=cv, tod_available=ok)


# --- Podolskij-Ziggel (R14) -------------------------------------------------------------------------------
def _ap_r14(prep, p):
    """R14: Podolskij & Ziggel (2010) truncated power-variation test as given by Dumitru & Urga (2012, eqs. 18-19)
    and Theodosiou & Zikes (2011, pp. 6-7) [SEC]: T = n^((m-1)/2) sum |r|^m (1 - eta 1{|r| <= c n^-w}),
    statistic T / sqrt(tau^2 n^(m-1) sum |r|^2m 1{.}) ~ N(0,1), eta = 1 +/- tau with probability 1/2 (seeded per
    day), c = c_mult sqrt(BV_t), w = 0.4. J = 1{reject} (RV - TRV)^+ with TRV the truncated RV (OWN pairing)."""
    g, mode = p["grid_step"], p["outage"]
    r = prep.rmask(g, 0, mode)
    N = r.shape[1]
    m, tau, w = float(p["m"]), float(p["tau"]), float(p["w"])
    bv = bipower(r, 0)
    cut2 = (p["c_mult"] * np.sqrt(bv) * N ** (-w))[:, None] ** 2
    keys = prep.data.day_keys()[prep.rows]
    eta = np.empty((prep.n, N))
    for i_, kk in enumerate(keys):
        rng = np.random.default_rng([int(p["seed"]), int(kk) & 0x7FFFFFFF])
        eta[i_] = 1.0 + tau * (2.0 * rng.integers(0, 2, N) - 1.0)
    a = np.abs(r)
    with np.errstate(invalid="ignore"):
        ind = np.isfinite(r) & (a * a <= cut2)
    T = N ** ((m - 1) / 2) * np.nansum(a ** m * (1.0 - eta * ind), axis=1)
    Vv = tau ** 2 * N ** (m - 1) * np.sum(np.where(ind, a ** (2 * m), 0.0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = T / np.sqrt(Vv)
    trv = np.sum(np.where(ind, r * r, 0.0), axis=1)
    r_all = prep.r(g, 0)
    rv_t = np.sum(r_all * r_all, axis=1)
    rv_o = realized_variance(r) if mode == "split" else rv_t
    crit = _crit_normal(p)
    return _out(rv_t, np.maximum(rv_o - trv, 0.0), stat > crit, stat, norm.sf(stat), crit, _nobs(r), trv=trv)


# --- local Gaussian bootstrap (T05) -----------------------------------------------------------------------
def _ap_t05(prep, p):
    """T05: Dovonon, Goncalves, Hounyo & Meddahi (2019) local Gaussian bootstrap of the BNS test (AAM eqs. 4-7,
    pp. 6, 12 [FT-v]): T = sqrt(n)(RV - BV)/sqrt(tau IQ), BV = mu1^-2 sum |r_{i-1}||r_i|, IQ = n mu43^-3 sum |r_i
    r_{i-1} r_{i-2}|^(4/3), tau = 0.609; r*_i = sqrt(v_i) eta_i with v_i the truncated local RV of blocks of
    k_n = floor(sqrt(n)) returns, truncation 2.3 sqrt(BV) n^-0.4; reject if T exceeds the alpha quantile of T*
    (bias_correction=True: Tbar* of eq. 7). Slow: n_boot replications per day. J = 1{reject}(RV - BV)^+ (OWN)."""
    g, mode = p["grid_step"], p["outage"]
    if mode == "split":
        raise ValueError("T05 supports outage='none' or 'drop' only")
    r = prep.r(g, 0)
    n = r.shape[1]
    B = int(p["n_boot"])
    a = np.abs(r)
    rv = np.sum(r * r, axis=1)
    bv = MU1 ** -2 * np.sum(a[:, 1:] * a[:, :-1], axis=1)
    iq = n * MU43 ** -3 * np.sum((a[:, 2:] * a[:, 1:-1] * a[:, :-2]) ** (4.0 / 3.0), axis=1)
    tau = THETA_BV
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.sqrt(n) * (rv - bv) / np.sqrt(tau * iq)
    kn = int(math.floor(math.sqrt(n)))
    u = 2.3 * np.sqrt(bv) * n ** -0.4
    tr = np.where(a <= u[:, None], r * r, 0.0)
    v = np.empty_like(r)
    for st in range(0, n, kn):
        e = min(n, st + kn)
        v[:, st:e] = tr[:, st:e].sum(axis=1, keepdims=True) / (e - st)
    sv = np.sqrt(v)
    Estar = v.sum(1) - np.sum(np.sqrt(v[:, 1:] * v[:, :-1]), axis=1)
    keys = prep.data.day_keys()[prep.rows]
    crit = np.full(prep.n, np.nan)
    pval = np.full(prep.n, np.nan)
    for d in range(prep.n):
        rng = np.random.default_rng([int(p["seed"]), int(keys[d]) & 0x7FFFFFFF])
        tstar = np.empty(B)
        for b0 in range(0, B, 2000):
            bb = min(B, b0 + 2000) - b0
            rs = rng.standard_normal((bb, n)) * sv[d][None, :]
            ab = np.abs(rs)
            rvs = np.sum(rs * rs, axis=1)
            bvs = MU1 ** -2 * np.sum(ab[:, 1:] * ab[:, :-1], axis=1)
            iqs = n * MU43 ** -3 * np.sum((ab[:, 2:] * ab[:, 1:-1] * ab[:, :-2]) ** (4.0 / 3.0), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                ts = np.sqrt(n) * (rvs - bvs - Estar[d]) / np.sqrt(tau * iqs)
                if p["bias_correction"]:
                    ts = ts + 0.5 * np.sqrt(n) * (v[d, 0] + v[d, -1]) / np.sqrt(tau * iqs)
            tstar[b0:b0 + bb] = ts
        crit[d] = np.quantile(tstar, p["alpha"])
        pval[d] = (1.0 + np.sum(tstar >= T[d])) / (B + 1.0)
    return _out(rv, np.maximum(rv - bv, 0.0), T > crit, T, pval, crit, np.full(prep.n, n), bv_dghm=bv,
                n_boot=np.full(prep.n, B))


# --- overlays (R20, R21) and the R24 diagnostic ------------------------------------------------------------
def _base_params(p):
    bp = dict(p.get("base_params") or {})
    for k in GLOBAL_DEFAULTS:
        if k not in bp and k in p and k != "alpha":
            bp[k] = p[k]
    return bp


def _ap_r20(prep, p):
    """R20: universal threshold sqrt(2 log N) of BST 2016 (WP pp. 8-11 [FT-v]) as a one-sided level
    Phi(sqrt(2 log N)) applied to the base approach (daily statistics compared with sqrt(2 log N); return-level
    tests re-run at that daily size). N = ut_n (default 1,280 days per coin; pooled 6,400 is the alternative)."""
    base = p["base"]
    bp = _base_params(p)
    bp["alpha"] = ut_alpha(int(p["ut_n"]))
    out = _call(REGISTRY[base], prep, bp)
    out["ut_threshold"] = np.full(prep.n, ut_threshold(int(p["ut_n"])))
    out["base"] = np.full(prep.n, base, dtype=object)
    return out


def _ap_r21(prep, p):
    """R21: Benjamini-Hochberg step-up FDR at q (default 10%; BST 2016 p. 10, STT 2020 [FT-v]; BH 1995 [MEM])
    over the days of the call: reject the k smallest p-values, k = max{i: p_(i) <= q i/N}; daily tests are
    re-decided, return-level tests re-run at the implied daily size."""
    base = p["base"]
    bp = _base_params(p)
    bp["alpha"] = p.get("base_alpha", GLOBAL_DEFAULTS["alpha"])
    ap = REGISTRY[base]
    out = _call(ap, prep, bp)
    pv = np.asarray(out["pval"], float)
    ok = np.isfinite(pv)
    Nt = int(ok.sum())
    srt = np.sort(pv[ok])
    qq = float(p["q"])
    kmax = 0
    if Nt:
        below = np.nonzero(srt <= qq * np.arange(1, Nt + 1) / Nt)[0]
        kmax = int(below[-1] + 1) if below.size else 0
    pstar = srt[kmax - 1] if kmax else -1.0
    if ap.redecide:
        jump = ok & (pv <= pstar)
        o2 = _out(out["rv"], out["j_raw"], jump, out["stat"], pv, np.nan, out["n_used"],
                  **{k: v for k, v in out.items() if k not in STD_COLS})
    elif kmax:
        bp["alpha"] = 1.0 - float(pstar) * (1 + 1e-12)
        o2 = _call(ap, prep, bp)
    else:
        o2 = dict(out)
        o2["jump"] = np.zeros(prep.n, bool)
        o2["j"] = np.zeros(prep.n)
        o2["c"] = np.asarray(out["rv"], float).copy()
    o2["fdr_pstar"] = np.full(prep.n, pstar if kmax else np.nan)
    o2["fdr_n_tests"] = np.full(prep.n, Nt)
    o2["base"] = np.full(prep.n, base, dtype=object)
    return o2


def _ap_r24(prep, p):
    """R24: the base return-level approach (default R18) with its 1-minute concentration diagnostics (OWN)."""
    bp = _base_params(p)
    bp["alpha"] = p["alpha"]
    out = _call(REGISTRY[p["base"]], prep, bp)
    out["base"] = np.full(prep.n, p["base"], dtype=object)
    return out


# =====================================================================================================
# 7. Registry
# =====================================================================================================
GLOBAL_DEFAULTS = dict(grid_step=5, alpha=0.999, outage="none", edge_rule="E1", drop_basis="trades", drop_run=20)
STD_COLS = ("rv", "c", "j", "jump", "stat", "pval", "crit", "n_used", "j_raw")
_PF = dict(pf_window="trailing", pf_days=100, pf_min_days=100, pf_cells="tod", pf_min_nonzero=0.25)
_CPR = dict(L=25, c_V=3.0, c_theta=3.0, n_iter=None, max_iter=100, guard=True, edge_window="day",
            variant="published", nonzero=False)


@dataclass
class Approach:
    id: str
    name: str
    family: str          # rv_target | combined | diagnostic | overlay
    inputs: str
    formula: str
    source: str
    func: Callable
    defaults: dict = field(default_factory=dict)
    limits: str = ""
    provisional: str = ""
    priority: str = "A"
    verified: str = "FT"
    redecide: bool = True     # daily statistic: overlays can re-decide from pval and j_raw
    split: str = "yes"
    slow: bool = False


def _A(**kw):
    return Approach(**kw)


REGISTRY = {}


def _reg(a: Approach):
    REGISTRY[a.id] = a


_reg(_A(id="RV01", name="5-min RV, single whole-day grid (ABD 2007)", family="rv_target", inputs="close",
        formula="RV = sum_j r_j^2, r_j = p_{gj} - p_{g(j-1)}, j = 1..1440/g, p_0 = previous 23:59 close",
        source="ABD 2007 eqs. 1-2 and Sec. V [FT]", func=_ap_rv01,
        limits="Contains the noise bias of DOGE 2019-20 (1-min RV 43% above 5-min RV).", verified="FT"))
_reg(_A(id="RV02", name="Subsampled 5-min RV over the 1-minute offsets (RV5_ss)", family="rv_target",
        inputs="close", formula="E1: N0 sum_s sum_j r_sj^2 / sum_s N_s; E2: mean_s(h_s^2 + sum_j r_sj^2 + e_s^2); "
        "E3: mean_s(RV_s N0/N_s)", source="ZMA 2005 eq. 13 [FT-v]; LPS 2015 p. 294 [FT]; MFE realized_variance.m "
        "[FT-code]; edge rules E1 MFE [FT-code], E2 PS 2015 p. 686 analogue, E3 ADS 2012 WP p. 12 [FT-v]",
        func=_ap_rv02, defaults=dict(offsets="all"),
        limits="Same noise bias as one sparse RV (ZMA eq. 33); edge rule is open choice O3 (OWN).",
        provisional="edge rule choice (O3)"))
_reg(_A(id="RV03", name="Two-scale RV (TSRV), slow scale K = grid_step", family="rv_target", inputs="close",
        formula="TSRV_adj = ([Y,Y]^avg_K - (nbar/n) RV_1min)/(1 - nbar/n), nbar = (n - K + 1)/K",
        source="ZMA 2005 eqs. 32-33, 41 [FT-v, poor extraction]; MFE rvtsD and K* rule [FT-code]", func=_ap_rv03,
        defaults=dict(K=None, K_rule="fixed", K_max=60), priority="B",
        limits="Removes i.i.d. noise only; dependent noise (DOGE) needs AMZ 2011 (not read). K_rule='mfe' maps "
        "RV_sparse to the grid_step RV and clamps K to [2, K_max] (OWN). outage='split' leaves the target unchanged.",
        split="no effect"))
_reg(_A(id="RV05", name="Realized kernel, Parzen, jittered end points (BNHLS 2009 design)", family="rv_target",
        inputs="close", formula="gamma_0 + 2 sum_{h<=H} k(h/(H+1)) gamma_h; H = ceil(3.5134 xi^0.8 n^0.6)",
        source="BNHLS 2009 eqs. 2-3, PDF pp. 5-7 [FT-v]; BNHLS 2011 on subsampling [FT-v]", func=_ap_rv05,
        defaults=dict(grid_step=1, jitter_m=2), priority="B",
        limits="Default grid_step 1 (all 1-minute closes). q, IV_sparse (20-min subsampled RV) and the jitter "
        "indexing are OWN mappings to klines (jitter indexing MEM). Not jump-robust. Do not subsample.",
        split="no effect"))
_reg(_A(id="RV09", name="Realized range with finite-m lambda (CP 2007)", family="rv_target",
        inputs="high, low, n_trades", formula="RRV = sum_i s_i^2 / lambda_{2,m_i}, m_i = 1 + trades in the bin",
        source="Christensen & Podolskij 2007 eq. 1.1, fn 7 [FT-v]; lambda_{2,m} as MFE realized_range [FT-code]",
        func=_ap_rv09, defaults=dict(mvd=False, mvd_q=66, mvd_min_days=20), priority="B",
        limits="Bin range from kline highs/lows and m from trade counts are OWN mappings; noise biases the range "
        "upward, sparse trading downward (MvD 2007 WP p. 3); LPS: subsampling RRV hurts.", split="no effect"))
_reg(_A(id="RV09-MvD", name="Realized range with Martens-van Dijk scaling", family="rv_target",
        inputs="high, low, n_trades", formula="RR_S,t = [sum_{l=1..q} RR_{t-l} / sum_{l=1..q} RR^D_{t-l}] RR^D_t, q = 66",
        source="Martens & van Dijk 2007 eq. 11, WP pp. 9, 18 [FT-v]", func=_ap_rv09,
        defaults=dict(mvd=True, mvd_q=66, mvd_min_days=20), priority="B",
        limits="Uses the previous q days (no look-ahead); NaN for the first mvd_min_days days (OWN); q = 66 is MvD's "
               "value for equities.", split="no effect"))
_reg(_A(id="X01", name="ABD 2007 HAR-CJ block: staggered BV/TQ, max-adjusted ratio test (C01 + T01)",
        family="combined", inputs="close",
        formula="BV = mu1^-2 N/(N-2) sum|r_j||r_{j-2}|; TQ = N mu43^-3 N/(N-4) sum|r_j r_{j-2} r_{j-4}|^(4/3); "
        "Z = sqrt(N)(1 - BV/RV)/sqrt(0.609 max(1, TQ/BV^2)); J = 1{Z > z_alpha}(RV - BV)",
        source="ABD 2007 eqs. 18-23, pp. 710-711, alpha fn 32 [FT]; HT 2005 eq. 9 [FT]", func=_ap_x01,
        limits="Stale prices (zeros) bias BV down and over-reject; outage catch-up returns over-reject.",
        verified="FT"))
_reg(_A(id="A1", name="Adjacent BV + Huang-Tauchen ratio test with adjacent TQ (crypto default)",
        family="combined", inputs="close",
        formula="BV = mu1^-2 N/(N-1) sum|r_j||r_{j-1}|; TQ = N mu43^-3 N/(N-2) sum|r_j r_{j-1} r_{j-2}|^(4/3); Z as X01",
        source="HT 2005 eqs. 3, 9 [FT]; BNS 2004 eq. 3 [FT]; factors CPR 2010 Table 1 note [FT-v]; crypto: Shen, "
        "Urquhart & Wang 2020 (alpha 0.999), Catania & Sandholdt 2019 eqs. 5, 7, 9 [FT]", func=_ap_a1,
        limits="Lag-1 noise autocorrelation biases adjacent BV upward (ABD pp. 710-711).", verified="FT"))
_reg(_A(id="T02", name="Per-offset ABD tests Z_0..Z_{g-1}: agreement diagnostic", family="diagnostic",
        inputs="close", formula="Z_s = X01 statistic on offset grid s (full intervals); A = #{s: Z_s > z_alpha}",
        source="OWN diagnostic (all-data report 3.6)", func=_ap_t02, defaults=dict(offsets="all"),
        limits="Diagnostic only; rv/c/j/jump/stat equal X01.", verified="OWN"))
_reg(_A(id="X02", name="Hybrid: subsampled RV/BV/TQ, jump days from base-grid Z_0", family="combined",
        inputs="close", formula="J = 1{Z_0 > z} max(RV_ss - BV_ss, 0), C = RV_ss - J",
        source="pieces: ABD 2007 [FT], MFE bvSSDebiased [FT-code]; combination OWN", func=_ap_x02,
        defaults=dict(offsets="all"), limits="Combination not in the literature (OWN); edge rule O3.",
        verified="OWN", provisional="combination needs Gerome's approval"))
_reg(_A(id="X03-Zss", name="Offset-robust ratio test Z_ss with simulated theta_ss (T03a)", family="combined",
        inputs="close", formula="Z_ss = sqrt(N0)(1 - BV_ss/RV_ss)/sqrt(theta_ss max(1, TQ_ss/BV_ss^2)); J as X02",
        source="ratio form HT 2005 [FT]; theta_ss by own simulation (variance not in closed form, CP 2012 p. 10 "
        "[FT-v])", func=_ap_x03, defaults=dict(offsets="all", decision="zss", theta_ss=None), verified="OWN",
        limits="theta_ss from constant-volatility simulation (THETA_SS_TABLE); recalibrate for other designs.",
        provisional="own test; diagnostic until approved"))
_reg(_A(id="X03-Zbar", name="Mean of per-offset Z_s with simulated critical value (T03b)", family="combined",
        inputs="close", formula="Zbar = mean_s Z_s, jump if Zbar > simulated null quantile; J as X02",
        source="OWN (all-data report 3.6)", func=_ap_x03, defaults=dict(offsets="all", decision="zbar",
        theta_ss=None), verified="OWN", limits="Null quantiles from constant-volatility simulation (ZBAR_NULL).",
        provisional="own test; diagnostic until approved"))
_reg(_A(id="X03-Pmean", name="2 x mean p-value merging across offsets (T04)", family="combined", inputs="close",
        formula="p = min(1, 2 mean_s p_s), p_s = 1 - Phi(Z_s); jump if p < 1 - alpha",
        source="Rueschendorf 1982; Meng 1993; Vovk & Wang 2020 abstract [ABS]; application to offsets OWN",
        func=_ap_x03, defaults=dict(offsets="all", decision="pmean", theta_ss=None), priority="B", verified="ABS",
        limits="Valid under any dependence but conservative."))
_reg(_A(id="X03-Cauchy", name="Cauchy combination of per-offset p-values (T04)", family="combined",
        inputs="close", formula="T = mean_s tan((0.5 - p_s) pi), p = 1/2 - arctan(T)/pi",
        source="Liu & Xie 2020 eqs. 1, 3 [FT-v, NIH AAM PMC7531765 sec. 2]; application to offsets OWN",
        func=_ap_x03, defaults=dict(offsets="all", decision="cauchy", theta_ss=None), priority="B",
        verified="FT", limits="Theorem 1 covers two-sided p-values of bivariate-normal z-scores; one-sided use OWN."))
_reg(_A(id="C03", name="Average of skip-0..4 BV (Patton-Sheppard), base-grid decision", family="combined",
        inputs="close", formula="BV_PS = mean_q mu1^-2 sum_j |r_j||r_{j-1-q}| (q = 0..4), mean over offsets; "
        "J = 1{Z_0 > z} max(RV_ss - BV_PS, 0)", source="PS 2015 eq. 11, p. 686, fn 10 [FT]; decision OWN",
        func=_ap_c03, defaults=dict(offsets="all", n_skips=5, debias=False), priority="B",
        limits="PS literal has no finite-sample factor (bias about -1%); debias=True adds MFE factor (OWN).",
        provisional="combination with Z_0 is OWN"))
_reg(_A(id="C05-Med", name="Subsampled MedRV, base-grid MedRV decision", family="combined", inputs="close",
        formula="MedRV_ss by edge rule; J = 1{Z_med,0 > z} max(RV_ss - MedRV_ss, 0)",
        source="ADS 2012 NBER WP p. 12 [FT-v]; constants ADS WP eq. 4 [FT-v]; decision OWN", func=_ap_c05,
        defaults=dict(offsets="all", kind="med"), priority="B", provisional="combination with Z_0 is OWN"))
_reg(_A(id="C05-Min", name="Subsampled MinRV, base-grid MinRV decision", family="combined", inputs="close",
        formula="MinRV_ss by edge rule; J = 1{Z_min,0 > z} max(RV_ss - MinRV_ss, 0)",
        source="ADS 2012 NBER WP p. 12 [FT-v]; decision OWN", func=_ap_c05, defaults=dict(offsets="all", kind="min"),
        priority="B", limits="MinRV is not robust to zeros (min with a zero is zero).",
        provisional="combination with Z_0 is OWN"))
_X04D = dict(M=5, form="linear", target="rv_base", diurnal=False, offsets="all", **_PF)
_reg(_A(id="X04", name="Blocked bipower BV_M (M = 5) from 1-minute miniature RVs and J_M test (C04 + T06)",
        family="combined", inputs="close",
        formula="BV_M = n/((n_M-1) k_{M,1}^2) sum (RV_{i-1}RV_i)^(1/2); J_M = sqrt(n)(RV - BV_M)/sqrt((M theta_M - 2) "
        "MV43); J = 1{J_M > z} max(RV_1m - BV_M, 0)", source="Mykland, Shephard & Sheppard 2012, Oxford DP 593, "
        "eqs. 9-17, Table 2 [FT-v; working paper]", func=_ap_x04, defaults=dict(_X04D),
        limits="Inherits 1-minute noise (DOGE 2019-20); own MC: oversized at n = 1,440 (0.22-0.34% at 0.1%); "
        "grid_step/M sets the base step (OWN mapping).", verified="FT"))
_reg(_A(id="X04-RV5", name="Blocked C/J with the grid RV (5-min RV) as target", family="combined", inputs="close",
        formula="J = 1{J_M > z} max(RV5 - BV_M, 0), C = RV5 - J", source="MSS 2012 [FT-v]; target swap OWN",
        func=_ap_x04, defaults=dict(_X04D, target="rv_grid"), provisional="target swap is OWN"))
_reg(_A(id="X04-RVss", name="Blocked C/J with subsampled RV as target", family="combined", inputs="close",
        formula="J = 1{J_M > z} max(RV_ss - BV_M, 0)", source="MSS 2012 [FT-v]; catalogue X04 variant OWN",
        func=_ap_x04, defaults=dict(_X04D, target="rv_ss"), priority="A", provisional="target swap is OWN"))
_reg(_A(id="X04-D", name="Blocked C/J, test on diurnally standardised returns", family="combined", inputs="close",
        formula="J_M computed on x/f_WSD (trailing 100 days); magnitudes from raw returns",
        source="MSS 2012 PDF p. 23 (diurnal pattern from previous 100 days) [FT-v]; BCL WSD [FT-v]; combination OWN",
        func=_ap_x04, defaults=dict(_X04D, diurnal=True), provisional="combination OWN"))
_reg(_A(id="R02", name="CPR corrected threshold bipower, C-Tz test, J from TBPV (as published)",
        family="combined", inputs="close",
        formula="V: iterated Gaussian-kernel filter (L = 25, c_V = 3, to convergence); theta = 9V; C-TBPV, C-TTriPV "
        "with Z_1 = 1.094, Z_4/3 = 1.129; C-Tz ratio test; TJ = 1{C-Tz > z}(RV - TBPV)^+",
        source="Corsi, Pirino & Reno 2010 eqs. 2.14-2.15, 3.1-3.5, 5.6-5.9, App. B, Table 1 note [FT-v AAM]",
        func=_ap_cpr, defaults=dict(_CPR),
        limits="Window within the UTC day (OWN; edge_window='back' reaches into day t-1); V guard (OWN); "
        "over-rejects with zeros (TZ 2011) and bursts (MMF 2020)."))
_reg(_A(id="R02-script", name="CPR as in the 5 Oct script (3 passes, no guard, J from C-TBPV)", family="combined",
        inputs="close", formula="as R02 but V after 3 passes and J = 1{C-Tz > z} max(RV - C-TBPV, 0)",
        source="cj_diagnostics_20261005.py (for the documented comparison)", func=_ap_cpr,
        defaults=dict(_CPR, variant="script"), limits="Not CPR as published (J understated).", verified="OWN"))
_reg(_A(id="R03", name="CPR on non-zero returns only (no-trade variant)", family="combined", inputs="close",
        formula="R02 with zero returns deleted, sqrt(N_nz)", source="CPR 2010 sec. 5.4, pp. 22-23 [FT-v]",
        func=_ap_cpr, defaults=dict(_CPR, nonzero=True), priority="B",
        limits="Biased toward the null (CPR); V after deleting zeros is an OWN reading."))
_reg(_A(id="R04", name="MedRV + MedRQ ratio test (v = 0.96)", family="combined", inputs="close",
        formula="MedRV = 1.41936 N/(N-2) sum med(|r_{i-1}|,|r_i|,|r_{i+1}|)^2; MedRQ = 0.923302 N N/(N-2) sum med^4; "
        "Z = sqrt(N)(1 - MedRV/RV)/sqrt(0.96 max(1, MedRQ/MedRV^2))",
        source="ADS 2009 NBER WP 15533 eq. 4, p. 7, p. 9, Prop. 3 [FT-v]; Dumitru & Urga 2012 eq. 15 [FT-v]",
        func=_ap_nn, defaults=dict(kind="med"), limits="Partly robust to zeros (ADS conclusion)."))
_reg(_A(id="R05", name="MinRV + MinRQ ratio test (v = 1.81)", family="combined", inputs="close",
        formula="MinRV = 2.75194 N/(N-1) sum min(|r_i|,|r_{i+1}|)^2; MinRQ = 2.20497 N N/(N-1) sum min^4",
        source="ADS 2009 WP p. 7, p. 9, Prop. 3 [FT-v]; DU 2012 eq. 14 [FT-v]", func=_ap_nn,
        defaults=dict(kind="min"), priority="B", limits="Not robust to zeros."))
_KR = dict(norm="n", target="rv_c", j_from="corrected", day_filter=False, rir=True, rir_d=None)
_reg(_A(id="R06", name="Kolokolov-Reno staleness-robust multipowers and corrected BNS test", family="combined",
        inputs="close", formula="MV^c(r) = Dbar sum prod |Delta(n,i)^(-1/2) Delta_tau_i X|^(r_k) over non-zero returns; "
        "Z = (1 - BV_c/RV_c)/sqrt((0.609/n) max(1, TQ_c/BV_c^2)); J = 1{Z > z}(RV_c - BV_c)^+, C = RV_c - J",
        source="Kolokolov & Reno 2024, JBES 42(2): grid p. 518, eqs. 2.8-2.10 p. 519, eq. 3.5 p. 520, JV pp. 526, "
        "529, day filter p. 526, RIR eq. 6.1 p. 527 [FT]", func=_ap_r06, defaults=dict(_KR), verified="FT",
        limits="Built for staleness; under pure rounding it overstates (5 Oct own run of the revised estimator, "
        "1,000 days, tick 0.06 at price 100, 38% zero returns: RV_c 1.49 and BV_c 1.94 x IV; a 300-day run gave 1.60 "
        "and 2.12); the rounding correction is in the online Appendix C (not available). Outage catch-ups "
        "are treated as stale stretches. RIR mode rule (p. 527) fails when 1-minute price changes span many ticks "
        "(BTC): pass rir_d = tick.", provisional="theta: KR print 2.61 with unnormalised multipowers (not "
        "centred); implemented 0.609 with normalised ones (HT form KR cite); first-return duration counted from "
        "00:00 UTC (OWN); target 'rv' cap OWN"))
_reg(_A(id="R06-Nnz", name="Kolokolov-Reno corrected BNS test with sqrt(N_nz) (own variant)", family="combined",
        inputs="close", formula="as R06 with sqrt(N_nz) in place of sqrt(n)", source="as R06; normalisation OWN",
        func=_ap_r06, defaults=dict(_KR, norm="Nnz"), verified="OWN"))
_reg(_A(id="R07", name="Periodicity-filtered BV and ratio test (HARP inputs)", family="combined", inputs="close",
        formula="r^f = r/f_WSD; Z on RV^f, adjacent BV^f, TQ^f; J^f = 1{Z^f > z}(RV^f - BV^f)^+; rv = RV^f",
        source="Dumitru, Hizmeri & Izzeldin 2025 eq. 6, App. B [FT]; BCL 2011 WSD [FT-v]", func=_ap_r07,
        defaults=dict(_PF), priority="B",
        limits="Factors from the trailing 100 days (OWN; DHI pool all days). rv is the filtered RV; rv_raw is the "
        "raw target."))
_reg(_A(id="R08", name="Truncated variation with time-of-day threshold (BTL 2013, tau = 2.5)", family="combined",
        inputs="close", formula="CV = sum r^2 1{|r| <= tau sqrt((BV ^ RV) TOD_i) n^-0.49}; JV = RV - CV",
        source="Bollerslev, Todorov & Li 2013 eqs. 3.3-3.6, p. 311 [FT]", func=_ap_r08,
        defaults=dict(tau=2.5, varpi=0.49, **_PF), priority="B", redecide=False,
        limits="No test: every truncated return makes a jump day. TOD from trailing days (OWN; BTL use all days)."))
_reg(_A(id="R08-BLT", name="Truncated variation with time-of-day threshold (BLT 2016, tau = 3)",
        family="combined", inputs="close", formula="as R08 with tau = 3", source="Bollerslev, Li & Todorov 2016 fn 17, "
        "p. 469 [FT]", func=_ap_r08, defaults=dict(tau=3.0, varpi=0.49, **_PF), priority="B", redecide=False))
_reg(_A(id="R14", name="Podolskij-Ziggel truncated power-variation test (c = 2.3, m = 2)", family="combined",
        inputs="close", formula="T/sqrt(tau^2 n^(m-1) sum |r|^2m 1{|r| <= c n^-0.4}), eta = 1 +/- 0.1",
        source="PZ 2010 [ABS]; formula via Dumitru & Urga 2012 eqs. 18-19 and Theodosiou & Zikes 2011 pp. 6-7 [SEC]",
        func=_ap_r14, defaults=dict(m=2, tau=0.1, w=0.4, c_mult=2.3, seed=20261005),
        priority="B", verified="SEC",
        limits="Randomised (seeded per day); TZ: c = 2.3 oversized, c = 3 better with zeros; J pairing with "
        "truncated RV OWN. Very sensitive to the truncation level: KR 2024 p. 529 [FT] find 6.81%, 2.26%, 0.58% jump "
        "days at 99.9% for c_theta = 5, 6, 7 (their CPR-based threshold, fn 3 p. 521).",
        provisional="square root in the standardisation to confirm against PZ 2010"))
_reg(_A(id="R14-TZ", name="Podolskij-Ziggel test with c = 3 (TZ 2011)", family="combined", inputs="close",
        formula="as R14 with c = 3 sqrt(BV)", source="TZ 2011 pp. 18-19 [FT-v]; formula [SEC]", func=_ap_r14,
        defaults=dict(m=2, tau=0.1, w=0.4, c_mult=3.0, seed=20261005), priority="B",
        verified="SEC"))
_reg(_A(id="R16", name="Lee-Mykland (2008) return-level test aggregated to a daily J", family="combined",
        inputs="close", formula="L_i = r_i/sigma_i (K - 1 previous returns); flag (|L| - C_n)/S_n > beta*; "
        "J = sum r^2 over flagged returns", source="LM 2008 Def. 1, eqs. 7-8, 12-13, pp. 2540-2543 [FT]",
        func=_ap_r16, defaults=dict(K=None, K_year=252, min_frac=0.5), priority="B", redecide=False,
        limits="Window reaches into earlier days (needs contiguous days); no periodicity filter; daily J OWN."))
_reg(_A(id="R17", name="ABD-Dobrev intraday test with Sidak level, aggregated to a daily J", family="combined",
        inputs="close", formula="flag |r_s| > Phi^-1(1 - beta/2) sqrt(Delta BV_t), beta = 1 - (1 - size)^Delta",
        source="Andersen, Bollerslev & Dobrev 2007 eq. 19, p. 134 [FT]", func=_ap_r17, priority="B", redecide=False,
        limits="ABDo use alpha = 1e-5 because of intraday volatility variation; default here is the common 0.999 "
        "(OWN harmonisation)."))
_reg(_A(id="R18", name="BCL periodicity-filtered (WSD) return-level test aggregated to a daily J",
        family="combined", inputs="close",
        formula="FJ_i = |r_i|/(f_WSD,i s_day); flag FJ > LM-type Gumbel threshold (n = 288); J = sum r^2 flagged",
        source="Boudt, Croux & Laurent 2011 eqs. 2.7-2.13, 3.4, WP pp. 8-10, 21-22 [FT-v]; day filter Aysan, "
        "Caporin & Cepni 2024 sec. 3 [FT]", func=_ap_r18,
        defaults=dict(threshold="lm", max_zero_share=1.0, **_PF), redecide=False,
        limits="Factors from the trailing 100 days (OWN; BCL pool all days); outage catch-up returns are tested "
        "unless outage='split'; daily J is OWN (BTL analogue)."))
_reg(_A(id="R19", name="Drift-robust Lee-Mykland test (Laurent & Shi 2020)", family="combined", inputs="close",
        formula="J*_i = (r_i - median_K)/sqrt(BV*_i/(K-1)); flag as LM with c = 1; J = sum r^2 flagged",
        source="Laurent & Shi 2020, AMSE WP 2018-43 pp. 14-22, eqs. 26-27 [FT-v]", func=_ap_r19,
        defaults=dict(K=None, K_year=252, min_frac=0.5), priority="B", redecide=False,
        limits="Window K taken from LM (published-version K not checked); critical values on the N(0,1) scale "
        "(OWN reading)."))
_reg(_A(id="R20", name="Universal-threshold overlay sqrt(2 log N)", family="overlay", inputs="daily statistic",
        formula="decide at the one-sided level Phi(sqrt(2 log N)); N = 1,280 per coin by default",
        source="Bajgrowicz, Scaillet & Treccani 2016, WP pp. 8-11 [FT-v]", func=_ap_r20,
        defaults=dict(base="X01", ut_n=1280, base_params=None),
        limits="N is open choice 3 of the robust report (per coin 1,280 or pooled 6,400)."))
_reg(_A(id="R21", name="Benjamini-Hochberg FDR overlay (q = 10%)", family="overlay", inputs="daily p-values",
        formula="reject the k smallest p-values, k = max{i: p_(i) <= q i/N} over the days of the call",
        source="BST 2016 p. 10 [FT-v]; STT 2020 pp. 12-13 [FT-v]; BH 1995 step-up [MEM]", func=_ap_r21,
        defaults=dict(base="X01", q=0.10, base_params=None), priority="B", verified="MEM",
        limits="Liberal when rejections are dense (BST p. 11); N = days passed in the call."))
_reg(_A(id="R24", name="1-minute concentration of flagged 5-minute intervals (diagnostic)", family="diagnostic",
        inputs="close", formula="share of the largest squared 1-min return; sign agreement of the 1-min returns",
        source="OWN, motivated by Christensen, Oomen & Podolskij 2014 and Christensen, Oomen & Reno 2022",
        func=_ap_r24, defaults=dict(base="R18", base_params=None), priority="B", verified="OWN",
        limits="Reporting only; columns r24_* are also attached to R16-R19."))
_reg(_A(id="T05", name="Local Gaussian bootstrap BNS test (DGHM 2019)", family="combined", inputs="close",
        formula="reject if T = sqrt(n)(RV - BV)/sqrt(0.609 IQ) exceeds the alpha quantile of T* (B draws)",
        source="Dovonon, Goncalves, Hounyo & Meddahi 2019, AAM eqs. 4-7, pp. 6, 12 [FT-v]", func=_ap_t05,
        defaults=dict(n_boot=999, bias_correction=False, seed=20261005), priority="B", slow=True,
        split="no (raises)", limits="Slow; alpha = 0.999 needs about 10^4 draws per day. Bootstrap returns carry "
        "no zeros or noise; J pairing OWN. Last block uses its own length (OWN)."))
_reg(_A(id="X06", name="Range-based C/J: J = max(RRV - RTV, 0) (C08)", family="combined",
        inputs="high, low, n_trades", formula="RBV = n mean(s_i s_{i+1}/lambda_1^2); RTV = n mean(prod s^(2/3)/lambda_2/3)",
        source="Christensen & Podolskij 2012 eqs. 23-24, PDF p. 27 [FT-v]", func=_ap_x06, priority="B",
        limits="No test (every day with RRV > RTV is a jump day); lambda_{1,m}, lambda_{2/3,m} by own simulation."))

NOT_IMPLEMENTED = [
    ("RV04", "Multi-scale RV", "priority C (all-data catalogue)"),
    ("RV06", "Subsampled realized kernels", "priority C; harmful for smooth kernels (BNHLS 2011)"),
    ("RV07", "RV_AC1 (Zhou 1996), subsampled", "priority C; source ABS only"),
    ("RV08 / C06 / X05", "Pre-averaged RV, BV and C/J", "not in this brief; at n = 1,440 a pre-averaging window spans "
     "15-38 minutes and LPS find it harms below 1 minute (all-data report 3.11)"),
    ("RV10", "Optimal sampling frequency (Bandi-Russell)", "priority C; source ABS only"),
    ("C07", "Modulated bipower", "not meaningful at n = 1,440 (at most 38/(c1 c2) blocks per day)"),
    ("C09 / T13", "Wavelet jump removal (JWTSRV) and its bootstrap test", "priority C; formula and test not verified"),
    ("T07", "Local-block subsampling variance (CPTV 2017)", "n >> L p^2 hard to meet at n = 288 or 1,440"),
    ("T08", "Ait-Sahalia-Jacod two-frequency ratio", "priority C; least powerful in MMF 2020"),
    ("T09", "Pre-averaged AJL 2012 test", "needs tick data (MMF: reliable only at 5-second sampling)"),
    ("T10", "Lee-Mykland 2012 noise-robust test", "needs tick data; block constant table not read"),
    ("T11", "CTV 2025 pre-averaged BNS-type test", "needs tick data"),
    ("T12", "Range-based ratio test", "would be an own derivation from the CP 2012 CLT"),
    ("X07 / X08", "HAR on a jump-robust measure; signed jump variation", "priority C (model forms, not C/J splits)"),
    ("R09 / R10", "Truncated RV with local or fixed-rate threshold", "priority C; dominated by TBPV (CPR Table 1)"),
    ("R11", "Quantile RV", "priority C; tuning-heavy, no J test"),
    ("R12", "Drift-robust BV* as a C estimator", "priority C (its test is R19)"),
    ("R13", "Differenced-return volatility (DV)", "formula not read (full text needed); built for tick data"),
    ("R15", "Swap-variance test", "priority C; not a J measure"),
    ("R22", "Drift-burst statistic", "needs tick data (5 observations per drift bandwidth on 1-minute klines)"),
    ("R23", "Excess idle time (EXIT)", "staleness diagnostic; needs tick-data noise variance; C.3 incomplete"),
    ("R25", "Diurnal-sufficiency test (CHP 2018)", "needs pre-averaged tick data"),
    ("R26", "Combinations of tests", "priority C; can be formed from the tables (e.g. X01 and R04 both reject)"),
    ("A7 / A11 / A12", "LM 2012 pre-averaged test; multi-exchange prices; drift-burst view", "tick data or other "
     "exchanges needed"),
]


# Verification level of the core formula. The source strings carry the tag of every piece; entries whose core
# formula was read in a working-paper, accepted-manuscript or arXiv version are FT-v.
for _id in ("RV03", "RV05", "RV09", "RV09-MvD", "X03-Cauchy", "C05-Med", "C05-Min", "X04", "X04-RV5", "X04-RVss",
            "X04-D", "R02", "R03", "R04", "R05", "R18", "R19", "R20", "T05", "X06"):
    REGISTRY[_id].verified = "FT-v"


def registry_table() -> pd.DataFrame:
    rows = []
    for a in REGISTRY.values():
        prm = dict(GLOBAL_DEFAULTS)
        prm.update(a.defaults)
        rows.append(dict(id=a.id, name=a.name, family=a.family, priority=a.priority, inputs=a.inputs,
                         formula=a.formula, source=a.source, verified=a.verified, split=a.split, slow=a.slow,
                         parameters="; ".join(f"{k}={v}" for k, v in prm.items()), limits=a.limits,
                         provisional=a.provisional))
    return pd.DataFrame(rows)


# =====================================================================================================
# 8. Runner
# =====================================================================================================
def _resolve(ap: Approach, user: dict) -> dict:
    allowed = set(GLOBAL_DEFAULTS) | set(ap.defaults)
    bad = set(user) - allowed
    if bad:
        raise ValueError(f"{ap.id}: unknown parameter(s) {sorted(bad)}; allowed {sorted(allowed)}")
    p = dict(GLOBAL_DEFAULTS)
    p.update(ap.defaults)
    p.update(user)
    g = p["grid_step"]
    if not isinstance(g, (int, np.integer)) or g < 1 or MIN_PER_DAY % g:
        raise ValueError(f"grid_step must be a divisor of 1440 minutes, got {g}")
    p["grid_step"] = int(g)
    if not 0.5 < float(p["alpha"]) < 1.0:
        raise ValueError("alpha must be in (0.5, 1)")
    if p["outage"] not in ("none", "split", "drop"):
        raise ValueError("outage must be 'none', 'split' or 'drop'")
    if p["edge_rule"] not in ("E1", "E2", "E3"):
        raise ValueError("edge_rule must be E1, E2 or E3")
    return p


def _call(ap: Approach, prep: _Prep, user: dict) -> dict:
    p = _resolve(ap, user)
    q = dict(p)
    if q["outage"] == "drop":
        q["outage"] = "none"
    return ap.func(prep, q)


def _rows_from_days(data: DayData, days) -> np.ndarray:
    if days is None:
        return np.arange(data.n_days)
    d = np.asarray(days)
    if d.dtype == bool:
        return np.nonzero(d)[0]
    if d.dtype.kind in "iu":
        return np.sort(d.astype(np.int64))
    want = np.asarray(d).astype("datetime64[D]")
    if data.days.dtype.kind != "M":
        raise ValueError("DayData has no date labels; select rows by index")
    pos = np.searchsorted(data.days, want)
    if np.any(pos >= data.n_days) or np.any(data.days[np.minimum(pos, data.n_days - 1)] != want):
        raise ValueError("some requested days are not in the data")
    return np.sort(pos)


def _assemble(prep: _Prep, ap: Approach, p: dict, out: dict) -> pd.DataFrame:
    n = prep.n
    df = pd.DataFrame({"pair": np.full(n, prep.data.pair, dtype=object),
                       "day": prep.data.day_labels()[prep.rows], "approach": np.full(n, ap.id, dtype=object)})
    for k in STD_COLS:
        v = out.get(k)
        if k == "jump":
            if v is None or ap.family == "rv_target":
                df[k] = pd.array([pd.NA] * n, dtype="boolean")
            else:
                df[k] = pd.array(np.asarray(v, bool), dtype="boolean")
        else:
            df[k] = np.asarray(v, dtype=float) if k != "n_used" else np.asarray(v, dtype=float)
    dropf, mr = prep.drop_flag(p["drop_basis"], int(p["drop_run"]))
    df["drop_day"] = dropf
    df["drop_maxrun"] = mr
    df["zero_share"] = prep.zero_share(p["grid_step"], "split" if p["outage"] == "split" else "none")
    df["outage_min"] = prep.outage_min()
    for k, v in out.items():
        if k in STD_COLS:
            continue
        df[k] = v
    if p["outage"] == "drop":
        for k in ("rv", "c", "j", "stat", "pval", "j_raw"):
            df.loc[dropf, k] = np.nan
        df.loc[dropf, "jump"] = pd.NA
    fin = np.isfinite(df["j"].to_numpy(float))
    if np.any(df["j"].to_numpy(float)[fin] < 0):
        raise AssertionError(f"{ap.id}: negative J")
    cc = df["c"].to_numpy(float)
    rv = df["rv"].to_numpy(float)
    jj = df["j"].to_numpy(float)
    ok = np.isfinite(cc) & np.isfinite(rv) & np.isfinite(jj)
    if not np.allclose(cc[ok], rv[ok] - jj[ok], rtol=0, atol=1e-12 * max(1.0, float(np.nanmax(np.abs(rv))) if ok.any() else 1.0)):
        raise AssertionError(f"{ap.id}: C != RV - J")
    return df


def run(approach_id: str, data: DayData, days=None, **params) -> pd.DataFrame:
    """Run one approach on one pair; returns one row per selected day (see module docstring)."""
    ap = REGISTRY[approach_id]
    rows = _rows_from_days(data, days)
    prep = _Prep(data, rows)
    p = _resolve(ap, params)
    q = dict(p)
    if q["outage"] == "drop":
        q["outage"] = "none"
    out = ap.func(prep, q)
    return _assemble(prep, ap, p, out)


def run_many(approach_ids, data: DayData, days=None, params: Optional[dict] = None, common: Optional[dict] = None,
             skip_slow: bool = True, return_timings: bool = False):
    """Run several approaches on one pair with shared caches; params = {id: {...}}, common = params for all.
    return_timings=True also returns {id: seconds} (shared caches are charged to the first approach using them)."""
    rows = _rows_from_days(data, days)
    prep = _Prep(data, rows)
    frames = []
    secs = {}
    for aid in approach_ids:
        ap = REGISTRY[aid]
        if ap.slow and skip_slow:
            continue
        t0 = time.time()
        user = dict(common or {})
        user = {k: v for k, v in user.items() if k in set(GLOBAL_DEFAULTS) | set(ap.defaults)}
        user.update((params or {}).get(aid, {}))
        p = _resolve(ap, user)
        q = dict(p)
        if q["outage"] == "drop":
            q["outage"] = "none"
        frames.append(_assemble(prep, ap, p, ap.func(prep, q)))
        secs[aid] = time.time() - t0
    df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return (df, secs) if return_timings else df


def _run_panel_job(args):
    pair, data, ids, days, params, common, skip_slow = args
    t = time.time()
    df = run_many(ids, data, days=days, params=params, common=common, skip_slow=skip_slow)
    return pair, df, time.time() - t


def run_panels(panels: dict, approach_ids, days=None, params=None, common=None, n_jobs: int = 3,
               skip_slow: bool = True):
    """Run approaches on several pairs, one worker process per pair (default 3 workers, so that workers plus the
    parent are 4 processes; at most 4 workers). Peak memory measured 5 Oct 2026 on 1280-day panels: 0.65-0.80 GB
    per worker, 0.77 GB parent. Returns (DataFrame, seconds per pair)."""
    n_jobs = max(1, min(int(n_jobs), 4, len(panels)))
    jobs = [(k, v, list(approach_ids), days, params, common, skip_slow) for k, v in panels.items()]
    if n_jobs == 1:
        res = [_run_panel_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            res = list(ex.map(_run_panel_job, jobs))
    return pd.concat([r[1] for r in res], ignore_index=True, sort=False), {r[0]: r[2] for r in res}


# =====================================================================================================
# 9. Calibration by simulation (theta_ss, Zbar null; range lambdas)
# =====================================================================================================
ZBAR_PROBS_DEFAULT = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975,
                      0.99, 0.995, 0.999, 0.9995, 0.9999)


def calibrate_offset_tests(grid_step: int = 5, n_sim: int = 200_000, seed: int = 20261005, chunk: int = 10_000,
                           edge_rules=("E1", "E2", "E3"), pattern="flat", probs=ZBAR_PROBS_DEFAULT) -> dict:
    """Own Monte Carlo (OWN) for the offset-robust tests: Gaussian 1-minute returns without jumps or noise, IV = 1
    per day (constant volatility unless pattern='ushape'); theta_ss[edge] = N0 Var(RV_ss - BV_ss)/IQ with the
    library's own subsampling code, and the null quantiles of Zbar = mean_s Z_s at `probs`."""
    diffs = {e: [] for e in edge_rules}
    zbar = []
    z0 = []
    iq = None
    for c0 in range(0, n_sim, chunk):
        m = min(chunk, n_sim - c0)
        data, truth = simulate_test_days(m, seed=seed + c0, iv=1.0, pattern=pattern)
        iq = truth["iq"][0]
        prep = _Prep(data, np.arange(m))
        for e in edge_rules:
            p = _resolve(REGISTRY["X02"], dict(grid_step=grid_step, edge_rule=e))
            ss = _ss(prep, p)
            diffs[e].append(ss["rv_all"] - ss["bv"])
            if e == edge_rules[0]:
                zbar.append(np.mean(ss["z"], axis=0))
                z0.append(ss["z"][0])
    N0 = MIN_PER_DAY // grid_step
    out = dict(grid_step=grid_step, n_sim=n_sim, seed=seed, pattern=pattern, iq=iq,
               theta_ss={e: float(N0 * np.var(np.concatenate(diffs[e])) / iq) for e in edge_rules})
    zb = np.concatenate(zbar)
    out["zbar_probs"] = tuple(probs)
    out["zbar_quantiles"] = tuple(float(x) for x in np.quantile(zb, probs))
    out["zbar_sd"] = float(np.std(zb))
    out["z0_size_999"] = float(np.mean(np.concatenate(z0) > norm.ppf(0.999)))
    return out


def simulate_range_lambdas(m_values, r_values=(2.0 / 3.0, 1.0, 2.0), n_rep: int = 200_000, seed: int = 7) -> dict:
    """lambda_{r,m} = E[range^r] of m equally spaced observations of a standard BM on [0, 1] (OWN simulation, as
    MFE realized_range_simulation.m does for lambda_{2,m})."""
    rng = np.random.default_rng(seed)
    res = {r: [] for r in r_values}
    for m in m_values:
        acc = {r: 0.0 for r in r_values}
        done = 0
        step = max(1, int(2e7 // max(m, 1)))
        while done < n_rep:
            b = min(step, n_rep - done)
            inc = rng.standard_normal((b, m - 1)) / math.sqrt(m - 1)
            path = np.concatenate([np.zeros((b, 1)), np.cumsum(inc, axis=1)], axis=1)
            s = path.max(1) - path.min(1)
            for r in r_values:
                acc[r] += float(np.sum(s ** r))
            done += b
        for r in r_values:
            res[r].append(acc[r] / n_rep)
    return res


# =====================================================================================================
# 10. Simulated constants (filled from runs of calibrate_offset_tests / simulate_range_lambdas; see spec)
# =====================================================================================================
# theta_ss and Zbar null quantiles: calibrate_offset_tests(grid_step=g, n_sim, seed=20261005 + g, chunk=5000),
# constant volatility, Gaussian 1-minute returns, IV = 1, no jumps or noise (OWN Monte Carlo, run 5 Oct 2026).
THETA_SS_TABLE.update({
    (5, "E1"): 0.23782, (5, "E2"): 0.24048, (5, "E3"): 0.23782,
    (10, "E1"): 0.22659, (10, "E2"): 0.23271, (10, "E3"): 0.22659,
    (15, "E1"): 0.22851, (15, "E2"): 0.23801, (15, "E3"): 0.22851,
    (30, "E1"): 0.24641, (30, "E2"): 0.26682, (30, "E3"): 0.24642,
})
ZBAR_PROBS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999)
ZBAR_NULL.update({
    5: (-1.7802, -1.5000, -1.3656, -1.1625, -0.9838, -0.7743, -0.5156, -0.3270, -0.1639, -0.0086, 0.1469, 0.3151, 0.5141, 0.7942, 1.0280, 1.2328, 1.4745, 1.6400, 1.9803, 2.1072, 2.4068),
    10: (-1.6719, -1.4236, -1.2983, -1.1071, -0.9400, -0.7424, -0.4976, -0.3173, -0.1608, -0.0119, 0.1387, 0.3028, 0.4953, 0.7695, 1.0007, 1.2051, 1.4414, 1.5994, 1.9415, 2.0682, 2.3748),
    15: (-1.6467, -1.3980, -1.2771, -1.0903, -0.9287, -0.7356, -0.4940, -0.3170, -0.1621, -0.0147, 0.1355, 0.2978, 0.4914, 0.7662, 0.9969, 1.2033, 1.4463, 1.6116, 1.9656, 2.1119, 2.3742),
    30: (-1.6259, -1.3886, -1.2663, -1.0904, -0.9308, -0.7421, -0.5033, -0.3253, -0.1691, -0.0222, 0.1308, 0.2964, 0.4943, 0.7777, 1.0145, 1.2267, 1.4755, 1.6502, 2.0167, 2.1352, 2.4064),
})
CALIBRATION_INFO.update(dict(design='constant volatility, Gaussian 1-minute returns, IV = 1', date='2026-10-05',
    runs={5: {'n_sim': 1000000, 'seed': 20261010, 'zbar_sd': 0.6115, 'z0_size_999': 0.001492}, 10: {'n_sim': 400000, 'seed': 20261015, 'zbar_sd': 0.5898, 'z0_size_999': 0.001715}, 15: {'n_sim': 400000, 'seed': 20261020, 'zbar_sd': 0.5859, 'z0_size_999': 0.0019325}, 30: {'n_sim': 400000, 'seed': 20261035, 'zbar_sd': 0.5923, 'z0_size_999': 0.0021675}},
    ushape_5=dict(n_sim=200000, seed=777, theta_ss={'E1': 0.24061, 'E2': 0.24809, 'E3': 0.24061}, zbar_q999=2.2203, zbar_sd=0.6319)))

# lambda_{r,m}: simulate_range_lambdas(m = 2..64, n_rep = 400,000, seed 11) and m = 80..4096 (n_rep = 100,000, seed 12);
# m = 2 replaced by the exact E|Z|^r; fit lambda_r (1 + a/sqrt(m-1) + b/(m-1)) on m >= 16 (OWN, 5 Oct 2026; max error
# 0.09% (r = 2/3), 0.14% (r = 1), 0.36% (r = 2) against the simulated points).
_LAM_TABLE[2.0 / 3.0] = (0.80238, 0.94021, 1.00695, 1.04805, 1.07719, 1.09982, 1.11631, 1.13055, 1.14215, 1.15167, 1.16165, 1.16931, 1.17592, 1.18142, 1.18774, 1.19223, 1.19645, 1.20086, 1.20411, 1.20874, 1.21140, 1.21375, 1.21819, 1.22000, 1.22284, 1.22519, 1.22726, 1.22966, 1.23157, 1.23370, 1.23529, 1.23769, 1.23930, 1.24083, 1.24197, 1.24373, 1.24456, 1.24635, 1.24750, 1.24892, 1.25077, 1.25165, 1.25277, 1.25372, 1.25540, 1.25569, 1.25674, 1.25810, 1.25855, 1.25882, 1.26123, 1.26160, 1.26230, 1.26338, 1.26389, 1.26563, 1.26575, 1.26646, 1.26633, 1.26716, 1.26795, 1.26926, 1.26971)
_LAM_FIT[2.0 / 3.0] = (-0.50251, 0.10926)
_LAM_TABLE[1.0] = (0.79788, 0.96354, 1.05298, 1.11093, 1.15328, 1.18673, 1.21161, 1.23330, 1.25107, 1.26558, 1.28128, 1.29332, 1.30362, 1.31231, 1.32253, 1.32949, 1.33610, 1.34329, 1.34839, 1.35588, 1.36004, 1.36385, 1.37102, 1.37400, 1.37857, 1.38229, 1.38568, 1.38953, 1.39262, 1.39612, 1.39874, 1.40283, 1.40524, 1.40782, 1.40968, 1.41256, 1.41386, 1.41666, 1.41861, 1.42102, 1.42414, 1.42554, 1.42738, 1.42883, 1.43166, 1.43219, 1.43376, 1.43623, 1.43689, 1.43723, 1.44130, 1.44193, 1.44299, 1.44469, 1.44563, 1.44867, 1.44871, 1.44993, 1.44959, 1.45103, 1.45236, 1.45465, 1.45529)
_LAM_FIT[1.0] = (-0.73268, 0.25976)
_LAM_TABLE[2.0] = (1.00000, 1.22755, 1.38361, 1.49701, 1.58534, 1.65724, 1.71418, 1.76506, 1.80670, 1.83955, 1.87957, 1.91002, 1.93520, 1.95727, 1.98556, 2.00200, 2.01812, 2.03841, 2.05103, 2.07105, 2.08111, 2.09170, 2.11085, 2.11978, 2.13191, 2.14114, 2.15065, 2.16062, 2.16916, 2.17908, 2.18669, 2.19963, 2.20426, 2.21188, 2.21747, 2.22499, 2.22877, 2.23488, 2.24074, 2.24874, 2.25831, 2.26174, 2.26676, 2.26973, 2.27840, 2.28048, 2.28345, 2.29239, 2.29369, 2.29359, 2.30607, 2.30813, 2.31026, 2.31438, 2.31787, 2.32837, 2.32660, 2.33074, 2.32851, 2.33301, 2.33747, 2.34516, 2.34555)
_LAM_FIT[2.0] = (-1.33676, 0.90115)

# =====================================================================================================
# 11. Smoke run on Binance data, benchmark, command line
# =====================================================================================================
SMOKE_DAYS = ("2019-08-15", "2019-09-13", "2019-12-09", "2020-03-12", "2021-01-11", "2021-05-19", "2021-08-13",
              "2022-05-12", "2022-11-09", "2023-01-15")


def default_ids(include_slow: bool = False):
    return [a for a in REGISTRY if include_slow or not REGISTRY[a].slow]


def _peak_rss_mb() -> float:
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / 2 ** 20 if os.uname().sysname == "Darwin" else v / 2 ** 10
    except Exception:
        return float("nan")


def _run_panel_job_mem(args):
    pair, data, ids, days, params, common, skip_slow = args
    t = time.time()
    df, per = run_many(ids, data, days=days, params=params, common=common, skip_slow=skip_slow, return_timings=True)
    return pair, df, time.time() - t, _peak_rss_mb(), per


def smoke_table(raw_dir: str, days=SMOKE_DAYS, approach_ids=None, n_jobs: int = 3, params=None):
    """Smoke run: load all five pairs for the whole sample (cross-day estimators need their windows), then compute
    every approach, including the bootstrap T05, for the selected days only. Returns (table, seconds per pair)."""
    panels = load_binance_1m(raw_dir)
    ids = approach_ids or default_ids(include_slow=True)
    return run_panels(panels, ids, days=np.array(days, dtype="datetime64[D]"), params=params, n_jobs=n_jobs,
                      skip_slow=False)


def benchmark(n_days: int = 1280, n_pairs: int = 5, n_jobs: int = 3, seed: int = 0, approach_ids=None,
              grid_step: int = 5):
    """Runtime and peak memory per process for all non-bootstrap approaches on simulated panels of the real size
    (n_pairs x n_days x 1440 minutes). Returns (info dict: wall seconds, seconds and peak RSS per pair process,
    parent peak RSS, seconds by approach summed over pairs; result table)."""
    panels = {}
    for i in range(n_pairs):
        d, _ = simulate_test_days(n_days, seed=seed + i, iv=4.0, pattern="ushape", jumps=dict(per_day=1, size=10),
                                  stale_prob=0.05, sub=2, outage=dict(days=[100, 700], start=120, length=300))
        d.pair = f"SIM{i}"
        panels[d.pair] = d
    ids = approach_ids or default_ids()
    jobs = [(k, v, ids, None, None, dict(grid_step=grid_step), True) for k, v in panels.items()]
    t = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(n_jobs, 4))) as ex:
        res = list(ex.map(_run_panel_job_mem, jobs))
    wall = time.time() - t
    per_app = pd.DataFrame([r[4] for r in res], index=[r[0] for r in res]).T
    info = dict(wall_seconds=wall, per_pair={r[0]: dict(seconds=round(r[2], 1), peak_rss_mb=round(r[3], 1)) for r in res},
                parent_peak_rss_mb=round(_peak_rss_mb(), 1), seconds_by_approach=per_app.sum(axis=1).round(2).to_dict())
    return info, pd.concat([r[1] for r in res], ignore_index=True, sort=False)


NULL_DESIGNS = {"flat": dict(pattern="flat"),
                "ushape": dict(pattern="ushape"),
                "ushape_stale5_tick": dict(pattern="ushape", stale_prob=0.05, tick=0.01, price0=100.0)}


def null_rejection_table(designs=None, n_chunk: int = 10, chunk: int = 1000, warm: int = 100, seed0: int = 900000,
                         approach_ids=None, redo=("R20", "R21", "T05")):
    """Null rejection rates (no jumps, IV = 1) at alpha = 0.999 and 0.99, as in section 6.2 of the 5 Oct spec (OWN
    Monte Carlo): n_chunk blocks of `chunk` simulated days per design (seed = seed0 + 1000 * design index + block),
    the first `warm` days of each block dropped (trailing periodicity window). The 0.99 decision is pval < 0.01,
    except for the entries in `redo` (overlays and bootstrap), which are rerun at alpha = 0.99. Returns
    (aggregate table with rate999/rate99, per-block table)."""
    designs = designs or NULL_DESIGNS
    ids = approach_ids or default_ids(include_slow=True)
    rows = []
    for di, (name, kw) in enumerate(designs.items()):
        for c in range(n_chunk):
            d, _ = simulate_test_days(chunk, seed=seed0 + 1000 * di + c, iv=1.0, **kw)
            keep = np.arange(warm, chunk)
            t999 = run_many(ids, d, days=keep, skip_slow=False)
            rd = [a for a in redo if a in ids]
            t99 = run_many(rd, d, days=keep, common=dict(alpha=0.99), skip_slow=False) if rd else None
            for aid, g in t999.groupby("approach"):
                jump = g["jump"].astype("boolean").fillna(False).to_numpy(bool)
                pv = g["pval"].to_numpy(float)
                if aid in rd:
                    j99 = t99.loc[t99.approach == aid, "jump"].astype("boolean").fillna(False).to_numpy(bool)
                elif np.isfinite(pv).any():
                    j99 = pv < 0.01
                else:
                    j99 = jump
                rows.append(dict(dgp=name, chunk=c, approach=aid, n=len(g), rej999=int(jump.sum()),
                                 rej99=int(j99.sum()), tested=int(np.isfinite(g["stat"].to_numpy(float)).sum())))
    per = pd.DataFrame(rows)
    agg = per.groupby(["dgp", "approach"])[["n", "rej999", "rej99", "tested"]].sum().reset_index()
    agg["rate999"] = agg.rej999 / agg.n
    agg["rate99"] = agg.rej99 / agg.n
    return agg, per


if __name__ == "__main__":
    import argparse
    ap_ = argparse.ArgumentParser(description="C/J methods library (5 Oct 2026)")
    sub = ap_.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("smoke", help="smoke table on the Binance klines")
    s1.add_argument("--raw", required=True, help="folder with <PAIR>/<PAIR>-1m-YYYY-MM.zip")
    s1.add_argument("--out", required=True)
    s2 = sub.add_parser("benchmark", help="runtime on simulated panels of the real size")
    s2.add_argument("--n-days", type=int, default=1280)
    s3 = sub.add_parser("registry", help="print the registry table")
    a_ = ap_.parse_args()
    if a_.cmd == "smoke":
        tab, secs = smoke_table(a_.raw)
        tab.to_csv(a_.out, index=False)
        print(secs)
    elif a_.cmd == "benchmark":
        info, _ = benchmark(n_days=a_.n_days)
        print(info)
    else:
        print(registry_table().to_string())
