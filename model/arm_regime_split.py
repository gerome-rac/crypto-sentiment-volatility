"""Arm A07: bull/bear split (register rows A07-R0..R3; test plan 5.3 and the bull/bear part of test 3.16).

POST-RESULT ANALYSIS. The pre-analysis note (D8) was frozen on 7 Oct 2026. The regime rules below were fixed on
3 Nov 2026, after the final A00 results of 28 Oct 2026 had been seen. Nothing here is confirmatory and nothing here
changes a D8 section 4 verdict.

Regime rules: one market-wide regime per UTC day, common to all coins. BTC close of a UTC day = close of the last
1-minute BTCUSDT kline of that day.
  lt20   Lunde & Timmermann (2004, eqs. 1-4) filter with lambda1 = lambda2 = 0.20, the thresholds of the crypto application
         of Zhang, Chan, Chu & Sulieman (2020). A bull state switches to bear on the first day the close falls more than
         lambda2 below the running peak of the bull state; the bear state is dated from the day after that peak. A bear
         state switches to bull on the first day the close rises more than lambda1 above the running trough; the bull
         state is dated from the day after that trough. Initial state: the first threshold crossing from the first day
         decides it (the filter needs an initial state; see lt_filter). Ex post dating: a turning point is dated with
         later prices. Days after the last peak or trough are kept in the current state and flagged.
  ma200  1-200 moving-average rule of Brock, Lakonishok & LeBaron (1992) without a band: bull on a day whose close is
         above its 200-day simple moving average (that close and the 199 before it), bear otherwise; undefined before
         the 200th close. Uses no later information.
Every forecast row (coin, origin t, target t+1) and every in-sample row takes the regime of its TARGET day t+1.

Part 1 (test 5.3). The stored forecasts of a finished A00 run are split by regime and evaluated with the engine's own
model.tests_eval.evaluate and model.tests_eval.confirmatory under the run's config.json: one evaluation, with its own
MCS, per rule x regime. The script first re-evaluates all forecast days and stops unless the stored nested_tests,
mcs, mcs_h2, loss_summary, per_coin and confirmatory_holm tables are reproduced byte for byte. On a regime subset the
engine's HAC treats the regime's days as one series in time order (the Equal Spacing estimator of Datta & Du 2012);
the Amplitude Modulated version (calendar lags, zero on the other regime's days; Datta & Du 2012) of the Clark-West
and DM-MSE statistics is reported as a check.

Part 2 (test 3.16, bull/bear part). The in-sample designs of model.inference.insample_design (all origins; in-window
transforms over the whole sample; z-scored as model.inference.ols_dk_nw) for R2 and R3; pooled OLS with every
regressor and the coin effects interacted with the regime, so the point estimates equal separate fits on each
regime's rows; Driscoll-Kraay covariance (linearmodels PanelOLS, cov_type 'kernel', Bartlett kernel, automatic
bandwidth, as the engine) on the contiguous sample; Wald (Chow-type) tests of equal coefficients across regimes.
Coefficients are reported in raw (probability) units, beta_z / sd.

Usage, from code/ (paths from the command line only):
  python -m model.arm_regime_split run --run-dir <A00 run folder> --panel <panel parquet>
      --variable-spec <spec csv> --klines-dir <folder with BTCUSDT-1m-YYYY-MM.zip>
      --arm-results <merged arm results csv> --out-root <folder for the new run folder> [--stamp <UTC stamp>]
      [--sq4 iut|dk2]   (9 Nov 2026: dk2 uses the two-sided Driscoll-Kraay p of D8 Deviation 7 as the H4 p in the
                         four-test Holm of each regime set; the reproduction check still uses the run's stored p)
  python -m model.arm_regime_split diagnostics --out-dir <A07 run folder> --run-dir <A00 run folder>
  python -m model.arm_regime_split results --out-dir <A07 run folder> --run-dir <A00 run folder> [--name <csv name>]
  python -m model.arm_regime_split figure --out-dir <A07 run folder> [--png <file name>]
  python -m model.arm_regime_split md5sums --out-dir <A07 run folder>
The diagnostics step (regime-difference HAC tests, per-phase contributions, lt20 x ma200 cross-tab) goes beyond the
tables of the brief and is labelled as such in the report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import sys
import zipfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import LADDER, NESTED_COMPARISONS, EngineConfig, hac_default_lags
from .tests_eval import confirmatory, evaluate, hac_var, holm

RULES = {
    "lt20": dict(kind="lunde_timmermann", lam_bull=0.20, lam_bear=0.20,
                 source="Lunde & Timmermann (2004) eqs. 1-4; thresholds 20%/20% as Zhang, Chan, Chu & Sulieman (2020)"),
    "ma200": dict(kind="moving_average", window=200,
                  source="Brock, Lakonishok & LeBaron (1992) 1-200 variable-length moving-average rule, no band"),
}
REGIMES = ("bull", "bear")
REPRO_TABLES = ("nested_tests", "mcs", "mcs_h2", "loss_summary", "per_coin", "confirmatory_holm")
POL_TERMS = [f"pol_{k}" for k in ("h1", "h7", "h30")] + [f"pol_sq_{k}" for k in ("h1", "h7", "h30")] + \
            [f"pol_sd_{k}" for k in ("h1", "h7", "h30")]
POL_FORMS = {"level": ["pol_h1", "pol_h7", "pol_h30"], "centred_square": ["pol_sq_h1", "pol_sq_h7", "pol_sq_h30"],
             "within_day_sd": ["pol_sd_h1", "pol_sd_h7", "pol_sd_h30"]}
FEAR = ["emo_fear_h1", "emo_fear_h7", "emo_fear_h30"]
ANGER = ["emo_anger_h1", "emo_anger_h7", "emo_anger_h30"]
HOLM_FAMILY = "sample/window"


# ----------------------------------------------------------------------------- helpers
def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- BTC daily close
def btc_daily_close(klines_dir: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """UTC-day close = close of the last 1-minute kline of each UTC day (Binance spot monthly kline zips)."""
    files = sorted(glob.glob(os.path.join(klines_dir, f"{symbol}-1m-*.zip")))
    if not files:
        raise FileNotFoundError(f"no {symbol}-1m-*.zip files in the klines folder")
    parts = []
    for f in files:
        with zipfile.ZipFile(f) as z:
            names = [n for n in z.namelist() if n.endswith(".csv")]
            assert len(names) == 1, f"{os.path.basename(f)}: expected one csv"
            with z.open(names[0]) as fh:
                raw = pd.read_csv(fh, header=None, usecols=[0, 4], names=["open_time", "close"], dtype=str)
        ot = pd.to_numeric(raw["open_time"], errors="coerce")
        n_header = int(ot.isna().sum())
        assert n_header <= 1, f"{os.path.basename(f)}: {n_header} non-numeric rows"
        raw = raw[ot.notna()]
        ot = ot[ot.notna()].astype("int64")
        unit = "us" if ot.max() > 10 ** 14 else "ms"          # Binance moved spot klines to microseconds in 2025
        parts.append(pd.DataFrame({"minute": pd.to_datetime(ot.to_numpy(), unit=unit, utc=True),
                                   "close": raw["close"].astype(float).to_numpy()}))
    k = pd.concat(parts, ignore_index=True).sort_values("minute")
    assert str(k["minute"].dt.tz) == "UTC", "kline timestamps are not UTC"
    assert k["minute"].is_unique, "duplicate kline open times"
    assert np.isfinite(k["close"]).all() and (k["close"] > 0).all()
    k["day"] = k["minute"].dt.floor("D").dt.tz_localize(None)
    d = k.groupby("day").agg(btc_close=("close", "last"), last_bar_utc=("minute", "last"),
                             n_minutes=("close", "size")).reset_index()
    full = pd.date_range(d["day"].min(), d["day"].max(), freq="D")
    assert len(full) == len(d) and (d["day"].to_numpy() == full.to_numpy()).all(), "missing UTC days in the klines"
    d["last_bar_utc"] = d["last_bar_utc"].dt.strftime("%H:%M")
    return d


# ----------------------------------------------------------------------------- regime rules
def lt_filter(p, lam_bull: float, lam_bear: float):
    """Lunde & Timmermann (2004) eqs. (1)-(4). Returns (state, unresolved, info); state 1 = bull, 0 = bear.

    In a bull state with running peak P_max (index e): a close >= P_max is a new peak and the days since the old peak
    are bull; a close < (1 - lam_bear) P_max switches to bear, dated from the day after the peak through the crossing
    day, and the crossing close starts the running trough. A bear state mirrors this with lam_bull.
    Initial state (the filter needs one; LT 2004 leave the choice open): from the first day, track the running
    maximum and minimum; if the close first falls more than lam_bear below the running maximum, the days up to that
    maximum are bull and a bear state starts the day after it; if it first rises more than lam_bull above the running
    minimum, the days up to that minimum are bear and a bull state starts the day after it.
    Days after the last peak or trough carry the current state and are flagged `unresolved`."""
    p = np.asarray(p, float)
    n = len(p)
    assert n >= 2 and np.isfinite(p).all() and (p > 0).all()
    state = np.full(n, -1, dtype=int)
    hi = lo = 0
    s = None
    t = 1
    while t < n:
        if p[t] >= p[hi]:
            hi = t
        if p[t] <= p[lo]:
            lo = t
        if p[t] < (1.0 - lam_bear) * p[hi]:
            state[: hi + 1] = 1
            state[hi + 1: t + 1] = 0
            s, init = 0, "bull until the first running maximum that was followed by a fall below the bear threshold"
            break
        if p[t] > (1.0 + lam_bull) * p[lo]:
            state[: lo + 1] = 0
            state[lo + 1: t + 1] = 1
            s, init = 1, "bear until the first running minimum that was followed by a rise above the bull threshold"
            break
        t += 1
    if s is None:
        raise ValueError("no threshold crossing in the sample; the filter dates no state")
    first_cross = t
    e = t
    for u in range(t + 1, n):
        if s == 1:
            if p[u] >= p[e]:
                state[e + 1: u + 1] = 1
                e = u
            elif p[u] < (1.0 - lam_bear) * p[e]:
                state[e + 1: u + 1] = 0
                s, e = 0, u
        else:
            if p[u] <= p[e]:
                state[e + 1: u + 1] = 0
                e = u
            elif p[u] > (1.0 + lam_bull) * p[e]:
                state[e + 1: u + 1] = 1
                s, e = 1, u
    unresolved = np.zeros(n, dtype=bool)
    unresolved[e + 1:] = True
    state[e + 1:] = s
    assert (state >= 0).all()
    return state, unresolved, dict(initial_rule=init, first_crossing_index=int(first_cross), last_extreme_index=int(e))


def phases(days: pd.Series, regime: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Runs of equal regime (days without a regime are skipped and break a run)."""
    rows, cur, start = [], None, None
    vals = regime.to_numpy(dtype=object)
    for i in range(len(vals) + 1):
        v = vals[i] if i < len(vals) else None
        v = v if v in REGIMES else None
        if v != cur:
            if cur is not None:
                rows.append(dict(regime=cur, start=days.iloc[start], end=days.iloc[i - 1], n_days=i - start,
                                 close_start=float(close.iloc[start]), close_end=float(close.iloc[i - 1])))
            cur, start = v, i
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(0, "phase_id", np.arange(1, len(out) + 1))
        out["log_change"] = np.log(out["close_end"] / out["close_start"])
    return out


def regime_dating(close_df: pd.DataFrame) -> pd.DataFrame:
    d = close_df.copy()
    r = RULES["lt20"]
    st, unres, info = lt_filter(d["btc_close"].to_numpy(), r["lam_bull"], r["lam_bear"])
    d["lt20_regime"] = np.where(st == 1, "bull", "bear")
    d["lt20_unresolved_tail"] = unres
    w = RULES["ma200"]["window"]
    ma = d["btc_close"].rolling(w, min_periods=w).mean()
    d["ma200"] = ma
    d["ma200_regime"] = np.where(ma.isna(), None, np.where(d["btc_close"] > ma, "bull", "bear"))
    info["first_day"] = str(d["day"].iloc[0].date())
    info["first_crossing_day"] = str(d["day"].iloc[info["first_crossing_index"]].date())
    info["last_extreme_day"] = str(d["day"].iloc[info["last_extreme_index"]].date())
    d.attrs["lt20_info"] = info
    return d


# ----------------------------------------------------------------------------- part 1
def load_run(run_dir: str):
    cfgd = json.load(open(os.path.join(run_dir, "config.json")))
    cfg = EngineConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfgd.items()})
    assert json.loads(cfg.to_json()) == cfgd, "config.json does not round-trip through EngineConfig"
    fc = pd.read_parquet(os.path.join(run_dir, "forecasts.parquet"))
    order = list(dict.fromkeys(fc["rung"]))                 # the run's own rung order (it fixes the MCS base column)
    fcs = {r: fc[fc["rung"] == r].reset_index(drop=True) for r in order}
    h4 = pd.read_csv(os.path.join(run_dir, "h4_bootstrap.csv"))
    p_h4 = float(h4["boot_p"].max())                        # run.py: p_h4 = max of the two bootstrap p-values
    assert np.isfinite(p_h4)
    return cfg, fcs, p_h4


def sq4_dk2_p(run_dir: str) -> float:
    """D8 Deviation 7 (8 Nov 2026): two-sided Driscoll-Kraay p of Cum(fear) - Cum(anger) = 0 of the run's full-sample
    R3 fit, computed as in model.arm_summary --sq4 dk2."""
    hb = pd.read_csv(os.path.join(run_dir, "h4_bootstrap.csv"))
    t2 = hb[hb["condition"].str.startswith("theta2")].iloc[0]
    p = float(2.0 * stats.norm.sf(abs(t2["estimate"] / t2["dk_se"])))
    assert np.isfinite(p)
    return p


def evaluate_set(fcs: Dict[str, pd.DataFrame], cfg: EngineConfig, p_h4: float, sq4: str = "iut") -> Dict[str, pd.DataFrame]:
    ev = evaluate(fcs, cfg, NESTED_COMPARISONS, bench="R0")
    ev["confirmatory_holm"] = confirmatory(ev["nested_tests"], ev["mcs"], p_h4, cfg.holm_alpha, ev.get("mcs_h2"),
                                           h4_status="tested")
    if sq4 == "dk2":                                        # a two-sided test of a difference has no "supported" direction
        h = ev["confirmatory_holm"]
        i = h.index[h["hypothesis"] == "H4"]
        h.loc[i, "verdict"] = ["fear and anger differ" if x < cfg.holm_alpha else "no significant difference"
                               for x in h.loc[i, "p_holm"]]
    return ev


def reproduction_check(run_dir: str, ev: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each recomputed table is written exactly as model.run writes it (to_csv, index=False) into a system temporary
    folder and compared byte for byte with the stored file."""
    import tempfile
    rows = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name in REPRO_TABLES:
            stored = os.path.join(run_dir, f"{name}.csv")
            tmp = os.path.join(tmp_dir, f"{name}.csv")
            ev[name].to_csv(tmp, index=False)
            byte_equal = open(tmp, "rb").read() == open(stored, "rb").read()
            old, new = pd.read_csv(stored), pd.read_csv(tmp)
            num = [c for c in old.columns if pd.api.types.is_numeric_dtype(old[c]) and not pd.api.types.is_bool_dtype(old[c])]
            diffs = [float(np.nanmax(np.abs(old[c].to_numpy(float) - new[c].to_numpy(float)))) for c in num if len(old)]
            rows.append(dict(table=f"{name}.csv", rows=len(old), same_columns=list(old.columns) == list(new.columns),
                             max_abs_diff=max(diffs) if diffs else 0.0, bytes_equal=byte_equal))
    return pd.DataFrame(rows)


def subset(fcs: Dict[str, pd.DataFrame], days: set) -> Dict[str, pd.DataFrame]:
    return {r: f[f["target"].isin(days)].reset_index(drop=True) for r, f in fcs.items()}


def lrv_amplitude_modulated(x_full: np.ndarray, obs: np.ndarray, lags: int) -> tuple:
    """Datta & Du (2012) Amplitude Modulated Bartlett long-run variance: the series stays on the calendar, the
    unobserved days are set to zero after demeaning over the observed days, and every autocovariance is divided by
    the number of observed days. Returns (mean over observed days, long-run variance, n observed)."""
    n = int(obs.sum())
    m = float(np.mean(x_full[obs]))
    z = np.where(obs, x_full - m, 0.0)
    v = float(z @ z) / n
    for l in range(1, lags + 1):
        v += 2.0 * (1.0 - l / (lags + 1.0)) * float(z[l:] @ z[:-l]) / n
    return m, v, n


def am_check(fcs: Dict[str, pd.DataFrame], days: set, cfg: EngineConfig) -> pd.DataFrame:
    """Clark-West and DM (MSE) with the Equal Spacing HAC (the engine's path on the subset) and the Amplitude
    Modulated HAC, same Bartlett lag rule (engine: floor(4 (T/100)^(2/9)) with T = days in the subset)."""
    L = {k: v.sort_values(["target", "coin"]).reset_index(drop=True) for k, v in fcs.items()}
    tgt = L["R0"]["target"]
    all_days = np.sort(tgt.unique())
    assert len(all_days) == (pd.Timestamp(all_days[-1]) - pd.Timestamp(all_days[0])).days + 1, "forecast days not contiguous"
    obs = np.isin(all_days, np.array(sorted(days), dtype="datetime64[ns]"))
    rows = []
    for small, big, lab in NESTED_COMPARISONS:
        a, b = L[small], L[big]
        y, f1, f2 = b["y"].to_numpy(), a["f_log"].to_numpy(), b["f_log"].to_numpy()
        adj = pd.Series((y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)).groupby(tgt.values).mean().sort_index().to_numpy()
        dse = pd.Series((y - f1) ** 2 - (y - f2) ** 2).groupby(tgt.values).mean().sort_index().to_numpy()
        T = int(obs.sum())
        lags = int(cfg.hac_lags) if cfg.hac_lags is not None else hac_default_lags(T)
        out = dict(test=lab, small=small, big=big, T_days=T, hac_lags=lags)
        for nm, series in (("cw", adj), ("dm_mse", dse)):
            m_es = float(series[obs].mean())
            v_es = hac_var(series[obs], lags)
            m_am, v_am, _ = lrv_amplitude_modulated(series, obs, lags)
            t_es, t_am = m_es / np.sqrt(v_es / T), m_am / np.sqrt(v_am / T)
            if nm == "cw":
                out.update(cw_stat_es=t_es, cw_p_es=float(1 - stats.norm.cdf(t_es)),
                           cw_stat_am=t_am, cw_p_am=float(1 - stats.norm.cdf(t_am)))
            else:
                k = np.sqrt((T + 1 - 2) / T)                 # HLN factor at h = 1, as model.tests_eval.dm_hln
                out.update(dm_mse_p_big_better_es=float(1 - stats.t.cdf(t_es * k, T - 1)),
                           dm_mse_p_big_better_am=float(1 - stats.t.cdf(t_am * k, T - 1)))
        rows.append(out)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- part 2
def build_builder(panel: str, variable_spec: str, cfg: EngineConfig):
    """The FeatureBuilder exactly as model.run builds it for '--rung all'."""
    from .har import FeatureBuilder
    from .panel import build_grid, load_tables, load_variable_spec, placebo_shift
    from .run import specs_for_rungs
    rungs = list(LADDER) + ["HARRV", "HARCJ"]
    specs = specs_for_rungs(load_variable_spec(variable_spec), rungs, cfg)
    grid = build_grid(load_tables(panel), specs)
    grid = placebo_shift(grid, specs, cfg.placebo_shift_days, cfg.placebo_mode, cfg.placebo_blocks)
    return FeatureBuilder(grid, specs, cfg)


def insample_check(builder, cfg: EngineConfig, run_dir: str) -> pd.DataFrame:
    """The engine's ols_dk_nw on this builder must reproduce the stored insample_coef.csv rows of R2 and R3."""
    from .inference import ols_dk_nw
    isc = pd.read_csv(os.path.join(run_dir, "insample_coef.csv"))
    rows = []
    for r in ("R2", "R3"):
        new = ols_dk_nw(builder, r, cfg)["coef"].reset_index(drop=True)
        old = isc[isc["rung"] == r].reset_index(drop=True)
        assert list(new["variable"]) == list(old["variable"]), f"{r}: variable list differs from insample_coef.csv"
        md = {c: float(np.max(np.abs(new[c].to_numpy(float) - old[c].to_numpy(float)))) for c in ("beta_raw", "se_dk", "p_dk")}
        rows.append(dict(rung=r, n_vars=len(new), **{f"max_abs_diff_{c}": v for c, v in md.items()}))
    out = pd.DataFrame(rows)
    assert (out[["max_abs_diff_beta_raw", "max_abs_diff_se_dk", "max_abs_diff_p_dk"]].to_numpy() < 1e-9).all(), \
        "the in-sample design does not reproduce the stored insample_coef.csv"
    return out


def _wald(b, V, R):
    rb = R @ b
    W = float(rb @ np.linalg.solve(R @ V @ R.T, rb))
    q = R.shape[0]
    return W, q, float(1 - stats.chi2.cdf(W, q))


def interacted_fit(builder, rung: str, cfg: EngineConfig, regime_by_day: pd.Series, rule: str):
    """Pooled OLS with every regressor and the coin effects interacted with the regime of the target day t+1;
    Driscoll-Kraay covariance on the contiguous sample of rows with a defined regime."""
    from linearmodels.panel import PanelOLS
    from .inference import _zdesign, dk_cov_within, insample_design
    d = insample_design(builder, rung)
    Z, names, mu, sd, keep = _zdesign(d)
    blocks = [bk for bk, k in zip(d.blocks, keep) if k]
    tgt_days = builder.days[d.day_tr + 1]
    reg = regime_by_day.reindex(tgt_days).to_numpy(dtype=object)
    use = np.array([x in REGIMES for x in reg])
    origin_days = builder.days[d.day_tr][use]
    span = pd.date_range(origin_days.min(), origin_days.max(), freq="D")
    assert len(np.unique(origin_days)) == len(span), "rows with a defined regime are not contiguous in time"
    Zu, yu = Z[use], d.y_tr[use]
    coins = np.array(builder.coins)[d.coin_tr][use]
    bull = reg[use] == "bull"
    for nm, msk in (("bull", bull), ("bear", ~bull)):
        assert (Zu[msk].std(0) > 0).all(), f"{rung}/{rule}: a regressor is constant within the {nm} rows"
    p = Zu.shape[1]
    X = np.hstack([Zu * bull[:, None], Zu * (~bull)[:, None]])
    cols = [f"{n}|bull" for n in names] + [f"{n}|bear" for n in names]
    ent = np.array([f"{c}|{'bull' if b else 'bear'}" for c, b in zip(coins, bull)])
    idx = pd.MultiIndex.from_arrays([ent, origin_days], names=["entity", "day"])
    kw = dict(cov_type="kernel", kernel="bartlett")
    if cfg.dk_bandwidth is not None:
        kw["bandwidth"] = cfg.dk_bandwidth
    fit = PanelOLS(pd.Series(yu, index=idx, name="y"), pd.DataFrame(X, index=idx, columns=cols),
                   entity_effects=True).fit(**kw)
    b, V = fit.params.to_numpy(), fit.cov.to_numpy()
    # check 1: point estimates = separate split-sample fits (entity = coin) on each regime's rows
    chk = {}
    for nm, msk, off in (("bull", bull, 0), ("bear", ~bull, p)):
        ix = pd.MultiIndex.from_arrays([coins[msk], origin_days[msk]], names=["coin", "day"])
        f1 = PanelOLS(pd.Series(yu[msk], index=ix), pd.DataFrame(Zu[msk], index=ix, columns=names),
                      entity_effects=True).fit()
        chk[f"max_abs_diff_split_fit_{nm}"] = float(np.max(np.abs(f1.params.to_numpy() - b[off: off + p])))
    assert max(chk.values()) < 1e-8, f"{rung}/{rule}: interacted estimates differ from the split-sample fits"
    # check 2: the engine's own DK formula (model.inference.dk_cov_within) with coin x regime groups
    grp = pd.factorize(ent)[0]
    ud, day_pos = np.unique(origin_days, return_inverse=True)
    bw = cfg.dk_bandwidth if cfg.dk_bandwidth is not None else float(np.floor(4 * (len(ud) / 100) ** (2 / 9)))
    b2, V2 = dk_cov_within(X, yu, grp, day_pos, bw)
    chk["max_abs_diff_beta_vs_engine_dk"] = float(np.max(np.abs(b2 - b)))
    chk["max_rel_diff_se_vs_engine_dk"] = float(np.max(np.abs(np.sqrt(np.diag(V2)) / np.sqrt(np.diag(V)) - 1)))
    sd2 = np.r_[sd, sd]
    braw, Vraw = b / sd2, V / np.outer(sd2, sd2)
    se = np.sqrt(np.diag(Vraw))
    coef = pd.DataFrame(dict(rule=rule, rung=rung, variable=names * 2, block=blocks * 2,
                             regime=["bull"] * p + ["bear"] * p, coef_raw=braw, se_dk_raw=se,
                             t_dk=braw / se, p_dk=2 * (1 - stats.norm.cdf(np.abs(braw / se)))))
    info = dict(rule=rule, rung=rung, n_rows=int(len(yu)), n_rows_bull=int(bull.sum()), n_rows_bear=int((~bull).sum()),
                n_days=int(len(ud)), first_origin=str(pd.Timestamp(origin_days.min()).date()),
                last_origin=str(pd.Timestamp(origin_days.max()).date()), k_per_regime=p, k_total=2 * p,
                dk_bandwidth=bw, r2_within=float(fit.rsquared_within), **chk)
    return dict(b=braw, V=Vraw, names=names, blocks=blocks, coef=coef, info=info, p=p)


def _sel(names: List[str], terms: List[str], p: int, regime: str) -> np.ndarray:
    a = np.zeros(2 * p)
    off = 0 if regime == "bull" else p
    for t in terms:
        a[off + names.index(t)] = 1.0
    return a


def part2_tables(fit: Dict) -> Dict[str, pd.DataFrame]:
    b, V, names, p = fit["b"], fit["V"], fit["names"], fit["p"]
    rule, rung = fit["info"]["rule"], fit["info"]["rung"]
    lin, wald = [], []

    def comb(label, a, kind, one_sided=False):
        est, se = float(a @ b), float(np.sqrt(a @ V @ a))
        z = est / se
        return dict(rule=rule, rung=rung, quantity=label, kind=kind, estimate=est, se_dk=se, z=z,
                    p_two_sided=float(2 * (1 - stats.norm.cdf(abs(z)))),
                    p_one_sided_positive=float(1 - stats.norm.cdf(z)) if one_sided else np.nan)

    for form, terms in POL_FORMS.items():
        for reg in REGIMES:
            lin.append(dict(comb(f"cumulative {form} (1+7+30)", _sel(names, terms, p, reg), "polarity_cumsum"), regime=reg))
        lin.append(dict(comb(f"cumulative {form} (1+7+30)", _sel(names, terms, p, "bull") - _sel(names, terms, p, "bear"),
                             "polarity_cumsum_difference"), regime="bull minus bear"))
    R_eq = np.vstack([(_sel(names, [t], p, "bull") - _sel(names, [t], p, "bear")) for t in POL_TERMS])
    W, q, pv = _wald(b, V, R_eq)
    wald.append(dict(rule=rule, rung=rung, block="polarity (9 terms) equal across regimes", stat=W, df=q, p=pv))
    for reg in REGIMES:
        R0 = np.vstack([_sel(names, [t], p, reg) for t in POL_TERMS])
        W, q, pv = _wald(b, V, R0)
        wald.append(dict(rule=rule, rung=rung, block=f"polarity (9 terms) jointly zero in {reg}", stat=W, df=q, p=pv))
    R_all = np.hstack([np.eye(p), -np.eye(p)])
    W, q, pv = _wald(b, V, R_all)
    wald.append(dict(rule=rule, rung=rung, block=f"all {p} slopes equal across regimes (Chow-type)", stat=W, df=q, p=pv))
    if all(t in names for t in FEAR + ANGER):
        a1 = {reg: _sel(names, FEAR, p, reg) for reg in REGIMES}
        an = {reg: _sel(names, ANGER, p, reg) for reg in REGIMES}
        for reg in REGIMES:
            lin.append(dict(comb("H4 theta1 = cumulative fear", a1[reg], "h4", True), regime=reg))
            lin.append(dict(comb("H4 theta2 = cumulative fear - cumulative anger", a1[reg] - an[reg], "h4", True), regime=reg))
            lin.append(dict(comb("cumulative anger (shown only)", an[reg], "h4_shown"), regime=reg))
        d1 = a1["bull"] - a1["bear"]
        d2 = (a1["bull"] - an["bull"]) - (a1["bear"] - an["bear"])
        lin.append(dict(comb("H4 theta1 = cumulative fear", d1, "h4_difference"), regime="bull minus bear"))
        lin.append(dict(comb("H4 theta2 = cumulative fear - cumulative anger", d2, "h4_difference"), regime="bull minus bear"))
        lin.append(dict(comb("cumulative anger (shown only)", an["bull"] - an["bear"], "h4_difference"), regime="bull minus bear"))
        for lab, R in (("H4 theta1 equal across regimes", d1[None, :]), ("H4 theta2 equal across regimes", d2[None, :]),
                       ("H4 theta1 and theta2 jointly equal across regimes", np.vstack([d1, d2]))):
            W, q, pv = _wald(b, V, R)
            wald.append(dict(rule=rule, rung=rung, block=lab, stat=W, df=q, p=pv))
        for reg in REGIMES:
            p1 = 1 - stats.norm.cdf(float(a1[reg] @ b) / np.sqrt(a1[reg] @ V @ a1[reg]))
            a2 = a1[reg] - an[reg]
            p2 = 1 - stats.norm.cdf(float(a2 @ b) / np.sqrt(a2 @ V @ a2))
            lin.append(dict(rule=rule, rung=rung, quantity="H4 intersection-union p (DK, max of the two one-sided p)",
                            kind="h4_iu_dk", estimate=np.nan, se_dk=np.nan, z=np.nan, p_two_sided=np.nan,
                            p_one_sided_positive=float(max(p1, p2)), regime=reg))
    return dict(lincomb=pd.DataFrame(lin), wald=pd.DataFrame(wald))


def fullsample_reference(builder, cfg: EngineConfig) -> Dict[str, pd.DataFrame]:
    """The engine's non-interacted fits (ols_dk_nw) for the same quantities, as the all-days reference."""
    from .inference import ols_dk_nw
    out_lin = []
    for rung in ("R2", "R3"):
        ins = ols_dk_nw(builder, rung, cfg)
        names, sd, V = ins["names"], ins["sd"], ins["V_dk"]
        b = ins["coef"]["beta_raw"].to_numpy()
        Vr = V / np.outer(sd, sd)

        def comb(label, plus, kind, minus=(), one_sided=False):
            a = np.zeros(len(names))
            for t in plus:
                a[names.index(t)] += 1.0
            for t in minus:
                a[names.index(t)] -= 1.0
            est, se = float(a @ b), float(np.sqrt(a @ Vr @ a))
            z = est / se
            return dict(rule="all days", rung=rung, quantity=label, kind=kind, estimate=est, se_dk=se, z=z,
                        p_two_sided=float(2 * (1 - stats.norm.cdf(abs(z)))),
                        p_one_sided_positive=float(1 - stats.norm.cdf(z)) if one_sided else np.nan, regime="all")
        for form, terms in POL_FORMS.items():
            out_lin.append(comb(f"cumulative {form} (1+7+30)", terms, "polarity_cumsum"))
        if rung == "R3":
            out_lin.append(comb("H4 theta1 = cumulative fear", FEAR, "h4", one_sided=True))
            out_lin.append(comb("H4 theta2 = cumulative fear - cumulative anger", FEAR, "h4", minus=ANGER, one_sided=True))
            out_lin.append(comb("cumulative anger (shown only)", ANGER, "h4_shown"))
        c = ins["coef"]
        c = c[c["variable"].isin(POL_TERMS + FEAR + ANGER)]
        for r in c.itertuples():
            out_lin.append(dict(rule="all days", rung=rung, quantity=r.variable, kind="coefficient", estimate=r.beta_raw,
                                se_dk=r.se_dk / sd[names.index(r.variable)], z=r.t_dk,
                                p_two_sided=r.p_dk, p_one_sided_positive=np.nan, regime="all"))
    return dict(lincomb=pd.DataFrame(out_lin))


# ----------------------------------------------------------------------------- Holm (28 Oct convention)
def holm_family_check(arm_results: str, a07: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """28 Oct 2026 convention (_arm_results_summary_20261028.md): Holm within each family and hypothesis across its
    runs, one Clark-West p per run. A07 belongs to the sample/window family. Variant 'primary': the two lt20 regime
    evaluations join the family as two members; variant 'primary+ma200': all four evaluations join."""
    m = pd.read_csv(arm_results)
    fam = m[m["family"] == HOLM_FAMILY].reset_index(drop=True)
    rows = []
    for H in ("H1", "H2", "H3"):
        base = fam[["run_id", f"{H}_cw_p", f"{H}_cw_p_holm_family"]].rename(
            columns={f"{H}_cw_p": "p", f"{H}_cw_p_holm_family": "p_holm_stored"})
        rep = holm(base["p"].to_numpy(), alpha)["p_holm"]
        assert np.allclose(rep, base["p_holm_stored"].to_numpy(), atol=1e-12), f"{H}: stored family Holm not reproduced"
        a = a07[a07["test"] == H]
        variants = {"28 Oct family (as stored)": a.iloc[0:0],
                    "primary (lt20 bull + bear)": a[a["rule"] == "lt20"],
                    "primary+ma200 (4 evaluations)": a}
        for vname, extra in variants.items():
            members = pd.concat([base[["run_id", "p"]].assign(member="28 Oct run"),
                                 pd.DataFrame(dict(run_id="A07_" + extra["rule"] + "_" + extra["regime"], p=extra["cw_p"],
                                                   member="A07"))], ignore_index=True)
            adj = holm(members["p"].to_numpy(), alpha)
            for i, r in members.iterrows():
                stored = base.loc[base["run_id"] == r["run_id"], "p_holm_stored"]
                st = float(stored.iloc[0]) if len(stored) else np.nan
                rows.append(dict(hypothesis=H, variant=vname, family=HOLM_FAMILY, family_size=len(members),
                                 run_id=r["run_id"], member=r["member"], p_raw=r["p"], p_holm=adj["p_holm"][i],
                                 reject_5pct=bool(adj["reject"][i]), p_holm_stored_28oct=st,
                                 verdict_changed_vs_28oct=(bool((st < alpha) != (adj["p_holm"][i] < alpha))
                                                           if np.isfinite(st) else None)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main run
def run(args) -> str:
    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(args.out_root, f"A07_{args.label}_{stamp}")
    os.makedirs(out, exist_ok=False)
    inputs = {"run_dir/forecasts.parquet": os.path.join(args.run_dir, "forecasts.parquet"),
              "run_dir/config.json": os.path.join(args.run_dir, "config.json"),
              "run_dir/h4_bootstrap.csv": os.path.join(args.run_dir, "h4_bootstrap.csv"),
              "run_dir/insample_coef.csv": os.path.join(args.run_dir, "insample_coef.csv"),
              "panel": args.panel, "variable_spec": args.variable_spec, "arm_results": args.arm_results}
    inputs.update({f"run_dir/{t}.csv": os.path.join(args.run_dir, f"{t}.csv") for t in REPRO_TABLES})
    in_md5 = {k: md5(v) for k, v in inputs.items()}
    kl = sorted(glob.glob(os.path.join(args.klines_dir, "BTCUSDT-1m-*.zip")))
    in_md5.update({f"klines/{os.path.basename(f)}": md5(f) for f in kl})
    if args.expect_forecasts_md5:
        assert in_md5["run_dir/forecasts.parquet"] == args.expect_forecasts_md5, "forecasts.parquet md5 differs"
    if args.expect_panel_md5:
        assert in_md5["panel"] == args.expect_panel_md5, "panel md5 differs"

    # regime dating
    close = btc_daily_close(args.klines_dir)
    reg = regime_dating(close)
    write_csv(reg, os.path.join(out, "regime_dating.csv"))
    ph = {}
    for rule in RULES:
        ph[rule] = phases(reg["day"], reg[f"{rule}_regime"], reg["btc_close"])
        write_csv(ph[rule], os.path.join(out, f"regime_phases_{rule}.csv"))

    # part 1: reproduction first, then the regime evaluations
    cfg, fcs, p_h4 = load_run(args.run_dir)
    ev_all = evaluate_set(fcs, cfg, p_h4)
    rep = reproduction_check(args.run_dir, ev_all)
    write_csv(rep, os.path.join(out, "reproduction_check.csv"))
    if not rep["bytes_equal"].all():
        raise SystemExit("the evaluation path does not reproduce the stored A00 tables; no regime evaluation was run")
    p_h4_regime = p_h4 if args.sq4 == "iut" else sq4_dk2_p(args.run_dir)   # the stored tables above keep the run's own H4 p
    tdays = np.sort(fcs["R0"]["target"].unique())
    regime_of = {rule: reg.set_index("day")[f"{rule}_regime"] for rule in RULES}
    counts, nested_all, mcs_all, mcsh2_all, conf_all, am_all, loss_all = [], [], [], [], [], [], []
    for rule in RULES:
        rt = regime_of[rule].reindex(pd.DatetimeIndex(tdays)).to_numpy(dtype=object)
        ro = regime_of[rule].reindex(pd.DatetimeIndex(tdays) - pd.Timedelta(days=1)).to_numpy(dtype=object)
        fc0 = fcs["R0"]
        o_reg = regime_of[rule].reindex(pd.DatetimeIndex(fc0["origin"])).to_numpy(dtype=object)
        t_reg = regime_of[rule].reindex(pd.DatetimeIndex(fc0["target"])).to_numpy(dtype=object)
        assert all(x in REGIMES for x in rt), f"{rule}: a forecast target day has no regime"
        phw = ph[rule]
        in_win = phw[(phw["end"] >= pd.Timestamp(tdays[0])) & (phw["start"] <= pd.Timestamp(tdays[-1]))]
        for regime in REGIMES:
            days = set(pd.DatetimeIndex(tdays)[rt == regime])
            sub = subset(fcs, days)
            ev = evaluate_set(sub, cfg, p_h4_regime, args.sq4)
            tag = f"{rule}_{regime}"
            for name in REPRO_TABLES:
                write_csv(ev[name], os.path.join(out, f"part1_{name}_{tag}.csv"))
            am = am_check(fcs, days, cfg)
            for lst, t in ((nested_all, ev["nested_tests"]), (mcs_all, ev["mcs"]), (mcsh2_all, ev["mcs_h2"]),
                           (conf_all, ev["confirmatory_holm"]), (loss_all, ev["loss_summary"]), (am_all, am)):
                t = t.copy()
                t.insert(0, "regime", regime)
                t.insert(0, "rule", rule)
                lst.append(t)
            counts.append(dict(rule=rule, regime=regime, n_target_days=len(days), share_of_target_days=len(days) / len(tdays),
                               n_forecast_rows=len(sub["R0"]),
                               n_phases_touching_window=int((in_win["regime"] == regime).sum()),
                               n_phases_total=int((phw["regime"] == regime).sum())))
        counts.append(dict(rule=rule, regime="boundary (origin and target regime differ)",
                           n_target_days=int(np.sum(rt != ro)), share_of_target_days=float(np.mean(rt != ro)),
                           n_forecast_rows=int(np.sum(o_reg != t_reg)),
                           n_phases_touching_window=len(in_win), n_phases_total=len(phw)))
    write_csv(pd.DataFrame(counts), os.path.join(out, "regime_counts.csv"))
    nested_all = pd.concat(nested_all, ignore_index=True)
    write_csv(nested_all, os.path.join(out, "part1_nested_tests_all.csv"))
    write_csv(pd.concat(mcs_all, ignore_index=True), os.path.join(out, "part1_mcs_all.csv"))
    write_csv(pd.concat(mcsh2_all, ignore_index=True), os.path.join(out, "part1_mcs_h2_all.csv"))
    write_csv(pd.concat(conf_all, ignore_index=True), os.path.join(out, "part1_confirmatory_holm_all.csv"))
    write_csv(pd.concat(loss_all, ignore_index=True), os.path.join(out, "part1_loss_summary_all.csv"))
    write_csv(pd.concat(am_all, ignore_index=True), os.path.join(out, "part1_hac_es_am_check_all.csv"))

    # Holm under the 28 Oct family convention
    hf = holm_family_check(args.arm_results, nested_all[["rule", "regime", "test", "cw_p"]], cfg.holm_alpha)
    write_csv(hf, os.path.join(out, "holm_family_check.csv"))

    # part 2
    builder = build_builder(args.panel, args.variable_spec, cfg)
    isc = insample_check(builder, cfg, args.run_dir)
    write_csv(isc, os.path.join(out, "part2_insample_reproduction_check.csv"))
    infos, coefs, lins, walds = [], [], [], []
    for rule in RULES:
        for rung in ("R2", "R3"):
            f = interacted_fit(builder, rung, cfg, regime_of[rule], rule)
            t = part2_tables(f)
            infos.append(f["info"])
            coefs.append(f["coef"])
            lins.append(t["lincomb"])
            walds.append(t["wald"])
    ref = fullsample_reference(builder, cfg)
    write_csv(pd.DataFrame(infos), os.path.join(out, "part2_fit_info.csv"))
    write_csv(pd.concat(coefs, ignore_index=True), os.path.join(out, "part2_coef_by_regime.csv"))
    write_csv(pd.concat(lins + [ref["lincomb"]], ignore_index=True), os.path.join(out, "part2_lincomb_by_regime.csv"))
    write_csv(pd.concat(walds, ignore_index=True), os.path.join(out, "part2_wald_by_regime.csv"))

    # run config and command
    arm_cfg = dict(arm="A07", label=args.label, stamp_utc=stamp, post_result=True,
                   note="post-result analysis: the regime rules were fixed on 3 Nov 2026, after the 28 Oct 2026 results",
                   rules=RULES, primary_rule="lt20", sensitivity_rule="ma200", regime_day="target day t+1",
                   lt20_initial_state=reg.attrs["lt20_info"], holm_family=HOLM_FAMILY,
                   engine_config_of_evaluated_run=json.loads(cfg.to_json()), p_h4_used_in_confirmatory=p_h4_regime,
                   **({"sq4": args.sq4, "p_h4_stored_bootstrap_max": p_h4} if args.sq4 != "iut" else {}),
                   inputs_md5=in_md5)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(arm_cfg, fh, indent=2, default=str)
    with open(os.path.join(out, "commands.txt"), "w") as fh:
        fh.write("python -m model.arm_regime_split " + " ".join(sys.argv[1:]) + "\n")
    write_md5sums(out, in_md5)
    return out


def write_md5sums(out: str, in_md5: Optional[Dict[str, str]] = None) -> None:
    if in_md5 is None:
        cfgp = os.path.join(out, "config.json")
        in_md5 = json.load(open(cfgp)).get("inputs_md5", {}) if os.path.exists(cfgp) else {}
    files = sorted(f for f in os.listdir(out) if f != "MD5SUMS" and not f.endswith(".tmp"))
    tmp = os.path.join(out, "MD5SUMS.tmp")
    with open(tmp, "w") as fh:
        fh.write("# outputs\n")
        for f in files:
            fh.write(f"{md5(os.path.join(out, f))}  {f}\n")
        fh.write("# inputs\n")
        for k, v in in_md5.items():
            fh.write(f"{v}  {k}\n")
    os.replace(tmp, os.path.join(out, "MD5SUMS"))


# ----------------------------------------------------------------------------- diagnostics (beyond the brief's tables)
def daily_differentials(fcs: Dict[str, pd.DataFrame]) -> Dict[str, tuple]:
    """Per target day: the Clark-West adjusted differential and the MSE loss differential (small minus big), each
    the cross-coin mean, exactly as model.tests_eval.clark_west / evaluate aggregate them (panel_agg 'daymean')."""
    L = {k: v.sort_values(["target", "coin"]).reset_index(drop=True) for k, v in fcs.items()}
    tgt = L["R0"]["target"]
    out = {}
    for small, big, lab in NESTED_COMPARISONS:
        a, b = L[small], L[big]
        y, f1, f2 = b["y"].to_numpy(), a["f_log"].to_numpy(), b["f_log"].to_numpy()
        adj = pd.Series((y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)).groupby(tgt.values).mean().sort_index()
        dse = pd.Series((y - f1) ** 2 - (y - f2) ** 2).groupby(tgt.values).mean().sort_index()
        out[lab] = (small, big, adj, dse)
    return out


def regime_difference_tests(fcs, regime_by_day: pd.Series, rule: str, cfg: EngineConfig) -> pd.DataFrame:
    """Descriptive test that the mean daily differential is the same on bull and bear target days: OLS of the
    full contiguous daily series on (1, bear dummy) with a Newey-West (Bartlett) covariance and the engine's lag rule
    (floor(4 (T/100)^(2/9)) = 6 at T = 885). Not a Giacomini-White test: the lt20 regime is dated ex post."""
    rows = []
    for lab, (small, big, adj, dse) in daily_differentials(fcs).items():
        days = adj.index
        bear = (regime_by_day.reindex(days).to_numpy(dtype=object) == "bear").astype(float)
        X = np.column_stack([np.ones(len(days)), bear])
        T = len(days)
        lags = int(cfg.hac_lags) if cfg.hac_lags is not None else hac_default_lags(T)
        XtXi = np.linalg.inv(X.T @ X)
        for nm, s in (("cw_adjusted_differential", adj.to_numpy()), ("mse_loss_differential", dse.to_numpy())):
            beta = XtXi @ X.T @ s
            g = X * (s - X @ beta)[:, None]
            S = g.T @ g
            for l in range(1, lags + 1):
                G = g[l:].T @ g[:-l]
                S += (1.0 - l / (lags + 1.0)) * (G + G.T)
            V = XtXi @ S @ XtXi
            z = beta[1] / np.sqrt(V[1, 1])
            rows.append(dict(rule=rule, test=lab, small=small, big=big, series=nm, T_days=T, hac_lags=lags,
                             n_bull=int((bear == 0).sum()), n_bear=int(bear.sum()), mean_bull=float(beta[0]),
                             mean_bear=float(beta[0] + beta[1]), diff_bear_minus_bull=float(beta[1]),
                             se_hac=float(np.sqrt(V[1, 1])), z=float(z), p_two_sided=float(2 * (1 - stats.norm.cdf(abs(z))))))
    return pd.DataFrame(rows)


def phase_contributions(fcs, phases_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Share of each regime phase (clipped to the forecast target days) in its regime's sum of daily Clark-West
    adjusted differentials, per comparison."""
    dd = daily_differentials(fcs)
    days = next(iter(dd.values()))[2].index
    rows = []
    for ph in phases_df.itertuples():
        msk = (days >= pd.Timestamp(ph.start)) & (days <= pd.Timestamp(ph.end))
        if not msk.any():
            continue
        for lab, (small, big, adj, dse) in dd.items():
            rows.append(dict(rule=rule, phase_id=ph.phase_id, regime=ph.regime, start=str(days[msk].min().date()),
                             end=str(days[msk].max().date()), n_days=int(msk.sum()), test=lab,
                             sum_cw_adjusted=float(adj[msk].sum()), mean_cw_adjusted=float(adj[msk].mean()),
                             sum_mse_loss_differential=float(dse[msk].sum())))
    out = pd.DataFrame(rows)
    tot = out.groupby(["test", "regime"])["sum_cw_adjusted"].transform("sum")
    out["share_of_regime_cw_sum"] = out["sum_cw_adjusted"] / tot
    return out


def diagnostics(out_dir: str, run_dir: str) -> None:
    cfg, fcs, _ = load_run(run_dir)
    reg = pd.read_csv(os.path.join(out_dir, "regime_dating.csv"), parse_dates=["day"]).set_index("day")
    diffs, contrib = [], []
    for rule in RULES:
        diffs.append(regime_difference_tests(fcs, reg[f"{rule}_regime"], rule, cfg))
        ph = pd.read_csv(os.path.join(out_dir, f"regime_phases_{rule}.csv"), parse_dates=["start", "end"])
        contrib.append(phase_contributions(fcs, ph, rule))
    write_csv(pd.concat(diffs, ignore_index=True), os.path.join(out_dir, "part1_regime_difference_tests.csv"))
    write_csv(pd.concat(contrib, ignore_index=True), os.path.join(out_dir, "part1_phase_contributions.csv"))
    tdays = pd.DatetimeIndex(np.sort(fcs["R0"]["target"].unique()))
    ct = pd.crosstab(reg["lt20_regime"].reindex(tdays).rename("lt20"), reg["ma200_regime"].reindex(tdays).rename("ma200"))
    write_csv(ct.reset_index(), os.path.join(out_dir, "regime_crosstab_target_days.csv"))
    # dating only (no tests): the four filters of Lunde & Timmermann (2004, p. 5 of the 2002 working paper) next to 20/20
    rows = []
    for lb, lr in ((0.20, 0.20), (0.20, 0.15), (0.20, 0.10), (0.15, 0.15), (0.15, 0.10)):
        st, _, _ = lt_filter(reg["btc_close"].to_numpy(), lb, lr)
        lab = pd.Series(np.where(st == 1, "bull", "bear"), index=reg.index)
        ph = phases(pd.Series(reg.index), lab.reset_index(drop=True), reg["btc_close"].reset_index(drop=True))
        tw = lab.reindex(tdays)
        inw = ph[(ph["end"] >= tdays[0]) & (ph["start"] <= tdays[-1])]
        rows.append(dict(lambda1_bull=lb, lambda2_bear=lr, phases_total=len(ph), phases_touching_window=len(inw),
                         bull_target_days=int((tw == "bull").sum()), bear_target_days=int((tw == "bear").sum()),
                         share_bull_target_days=float((tw == "bull").mean()),
                         agreement_with_lt20_target_days=float((tw.to_numpy() == reg["lt20_regime"].reindex(tdays).to_numpy()).mean())))
    write_csv(pd.DataFrame(rows), os.path.join(out_dir, "regime_lt_threshold_dating.csv"))


# ----------------------------------------------------------------------------- consolidated results table
def _mcs_note(mcs: pd.DataFrame, rule: str, regime: str) -> str:
    parts = []
    for lo in ("MSE_log", "QLIKE"):
        for a in (0.10, 0.25):
            t = mcs[(mcs["rule"] == rule) & (mcs["regime"] == regime) & (mcs["loss"] == lo) & np.isclose(mcs["alpha"], a)]
            keep = [m for m in ("R0", "R1", "R2", "R3", "HARRV") if m in set(t.loc[t["in_mcs"], "model"])]
            parts.append(f"{lo} {int(round((1 - a) * 100))}% MCS: {{{', '.join(keep)}}}")
    return "; ".join(parts)


def results_table(out_dir: str, run_dir: str, name: str) -> str:
    """One row per rule x regime x comparison (part 1) or coefficient/combination/test (part 2), plus the Holm
    family rows and the regime counts."""
    nt = pd.read_csv(os.path.join(out_dir, "part1_nested_tests_all.csv"))
    am = pd.read_csv(os.path.join(out_dir, "part1_hac_es_am_check_all.csv"))
    mc = pd.read_csv(os.path.join(out_dir, "part1_mcs_all.csv"))
    mh = pd.read_csv(os.path.join(out_dir, "part1_mcs_h2_all.csv"))
    cf = pd.read_csv(os.path.join(out_dir, "part1_confirmatory_holm_all.csv"))
    hf = pd.read_csv(os.path.join(out_dir, "holm_family_check.csv"))
    rd = pd.read_csv(os.path.join(out_dir, "part1_regime_difference_tests.csv"))
    rc = pd.read_csv(os.path.join(out_dir, "regime_counts.csv"))
    cb = pd.read_csv(os.path.join(out_dir, "part2_coef_by_regime.csv"))
    lc = pd.read_csv(os.path.join(out_dir, "part2_lincomb_by_regime.csv"))
    wd = pd.read_csv(os.path.join(out_dir, "part2_wald_by_regime.csv"))
    fi = pd.read_csv(os.path.join(out_dir, "part2_fit_info.csv"))
    full = pd.read_csv(os.path.join(run_dir, "nested_tests.csv"))
    label = {"H1": "H1: R0 -> R1", "H2": "H2: R1 -> R2", "H3": "H3: R2 -> R3", "R1vsR3": "R1 -> R3"}
    rows = []
    for r in full.itertuples():
        rows.append(dict(part="part1_oos", rule="none (all 885 days, stored A00)", regime="all", n_days=885,
                         item=label[r.test], cw_stat=r.cw_stat, cw_p=r.cw_p, dm_mse_p_big_better=r.dm_mse_p_big_better,
                         dm_qlike_p_big_better=r.dm_qlike_p_big_better, dm_mse_p2=r.dm_mse_p2, dm_qlike_p2=r.dm_qlike_p2,
                         r2_oos=r.r2_oos_big_vs_small, gw_mse_p=r.gw_mse_p, gw_qlike_p=r.gw_qlike_p))
    for r in nt.itertuples():
        a = am[(am["rule"] == r.rule) & (am["regime"] == r.regime) & (am["test"] == r.test)].iloc[0]
        n_days = int(a["T_days"])
        c = cf[(cf["rule"] == r.rule) & (cf["regime"] == r.regime) & (cf["hypothesis"] == r.test)]
        fam = {}
        for v, key in (("primary (lt20 bull + bear)", "p_holm_family_primary"), ("primary+ma200 (4 evaluations)", "p_holm_family_with_ma200")):
            h = hf[(hf["variant"] == v) & (hf["hypothesis"] == r.test) & (hf["run_id"] == f"A07_{r.rule}_{r.regime}")]
            fam[key] = float(h["p_holm"].iloc[0]) if len(h) else np.nan
        rows.append(dict(part="part1_oos", rule=r.rule, regime=r.regime, n_days=n_days, item=label[r.test],
                         cw_stat=r.cw_stat, cw_p=r.cw_p, cw_p_amplitude_modulated=a["cw_p_am"], hac_lags=r.hac_lags,
                         dm_mse_p_big_better=r.dm_mse_p_big_better, dm_qlike_p_big_better=r.dm_qlike_p_big_better,
                         dm_mse_p2=r.dm_mse_p2, dm_qlike_p2=r.dm_qlike_p2, r2_oos=r.r2_oos_big_vs_small,
                         gw_mse_p=r.gw_mse_p, gw_qlike_p=r.gw_qlike_p, mcs_note=_mcs_note(mc, r.rule, r.regime),
                         p_holm_own=float(c["p_holm"].iloc[0]) if len(c) else np.nan,
                         verdict_own_holm=c["verdict"].iloc[0] if len(c) else "", **fam))
    for rule in RULES:
        for regime in REGIMES:
            c = cf[(cf["rule"] == rule) & (cf["regime"] == regime) & (cf["hypothesis"] == "H2")].iloc[0]
            m = mh[(mh["rule"] == rule) & (mh["regime"] == regime)]
            n_days = int(am[(am["rule"] == rule) & (am["regime"] == regime)]["T_days"].iloc[0])
            note = (f"CW p_holm(own) = {c['p_holm']:.4f}; R1 in 10% MCS of {{R1,R2}} on both losses: "
                    f"{c['R1_in_MCS10_of_R1R2_both_losses']} (MCS p of R1: MSE {m[(m['loss'] == 'MSE_log') & (m['model'] == 'R1')]['mcs_pvalue'].iloc[0]:.4f}, "
                    f"QLIKE {m[(m['loss'] == 'QLIKE') & (m['model'] == 'R1')]['mcs_pvalue'].iloc[0]:.4f}); "
                    f"no one-sided DM gain of R2 at 5% on either loss: {c['DM_no_improvement_of_R2_both']}")
            rows.append(dict(part="part1_h2_rule", rule=rule, regime=regime, n_days=n_days, item="H2 rule (Deviation 4)",
                             p=c["p"], p_holm_own=c["p_holm"], verdict_own_holm=c["verdict"], note=note))
    for r in rd.itertuples():
        rows.append(dict(part="part1_regime_difference", rule=r.rule, regime="bear minus bull", n_days=r.T_days,
                         item=f"{label[r.test]} | {r.series}", estimate=r.diff_bear_minus_bull, se=r.se_hac, stat=r.z,
                         p=r.p_two_sided, hac_lags=r.hac_lags,
                         note=f"mean bull {r.mean_bull:.6f} (n={r.n_bull}), mean bear {r.mean_bear:.6f} (n={r.n_bear})"))
    keep = POL_TERMS + FEAR + ANGER
    nrows = {(x.rule, x.rung): x for x in fi.itertuples()}
    for r in cb[cb["variable"].isin(keep)].itertuples():
        x = nrows[(r.rule, r.rung)]
        rows.append(dict(part="part2_coef", rule=r.rule, regime=r.regime, rung=r.rung, item=r.variable,
                         n_rows=x.n_rows_bull if r.regime == "bull" else x.n_rows_bear,
                         estimate=r.coef_raw, se=r.se_dk_raw, stat=r.t_dk, p=r.p_dk))
    for r in lc.itertuples():
        rows.append(dict(part="part2_coef" if r.kind == "coefficient" else "part2_lincomb", rule=r.rule, regime=r.regime,
                         rung=r.rung, item=r.quantity, estimate=r.estimate, se=r.se_dk, stat=r.z, p=r.p_two_sided,
                         p_one_sided=r.p_one_sided_positive))
    for r in wd.itertuples():
        rows.append(dict(part="part2_wald", rule=r.rule, regime="bull vs bear", rung=r.rung, item=r.block,
                         stat=r.stat, df=r.df, p=r.p))
    for r in hf.itertuples():
        rows.append(dict(part="holm_family", rule=r.variant, regime=r.member, item=f"{r.hypothesis} | {r.run_id}",
                         p=r.p_raw, p_holm_family=r.p_holm, note=f"family {r.family}, size {r.family_size}; stored 28 Oct "
                         f"p_holm {r.p_holm_stored_28oct}; verdict changed vs 28 Oct: {r.verdict_changed_vs_28oct}"))
    for r in rc.itertuples():
        rows.append(dict(part="regime_counts", rule=r.rule, regime=r.regime, n_days=r.n_target_days, n_rows=r.n_forecast_rows,
                         estimate=r.share_of_target_days, note=f"phases touching the forecast window: {r.n_phases_touching_window}; "
                         f"phases in the full sample: {r.n_phases_total}"))
    out = pd.DataFrame(rows)
    cols = ["part", "rule", "regime", "rung", "item", "n_days", "n_rows", "cw_stat", "cw_p", "cw_p_amplitude_modulated",
            "hac_lags", "dm_mse_p_big_better", "dm_qlike_p_big_better", "dm_mse_p2", "dm_qlike_p2", "r2_oos",
            "gw_mse_p", "gw_qlike_p", "mcs_note", "p_holm_own", "verdict_own_holm", "p_holm_family_primary",
            "p_holm_family_with_ma200", "estimate", "se", "stat", "df", "p", "p_one_sided", "p_holm_family", "note"]
    out = out.reindex(columns=cols)
    path = os.path.join(out_dir, name)
    write_csv(out, path)
    return path


# ----------------------------------------------------------------------------- figure
def figure(out_dir: str, png: str = "fig_A07_btc_regimes.png", oos_start: str = "2020-08-30"):
    """BTC UTC-day close (log scale): (a) bear phases of the lt20 rule shaded; (b) close against its 200-day moving
    average with the days below it shaded. Font sizes and other style come from the caller's matplotlib rcParams."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    reg = pd.read_csv(os.path.join(out_dir, "regime_dating.csv"), parse_dates=["day"])
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True)

    def shade(ax, col):
        r = reg[col].to_numpy(dtype=object)
        i = 0
        while i < len(r):
            if r[i] == "bear":
                j = i
                while j + 1 < len(r) and r[j + 1] == "bear":
                    j += 1
                ax.axvspan(reg["day"].iloc[i] - pd.Timedelta(hours=12), reg["day"].iloc[j] + pd.Timedelta(hours=12),
                           color="0.84", lw=0, zorder=0)
                i = j + 1
            else:
                i += 1

    small = plt.rcParams["legend.fontsize"]
    for ax in axes:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([5000, 10000, 20000, 40000, 60000]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        ax.axvline(pd.Timestamp(oos_start), color="0.35", lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_ylim(3500, 80000)
        ax.set_xlim(reg["day"].iloc[0] - pd.Timedelta(days=5), reg["day"].iloc[-1] + pd.Timedelta(days=5))
        ax.set_ylabel("BTC close (USDT)", labelpad=6)
    a, b = axes
    shade(a, "lt20_regime")
    a.plot(reg["day"], reg["btc_close"], color="#1f3b73", lw=0.9, zorder=2)
    a.set_title("Bear phases dated ex post by the 20%/20% filter (primary rule)", loc="left")
    # legends and the window note sit in the unshaded space of late 2020, below and above the price line
    a.legend(handles=[Patch(color="0.84", label="bear phase")], loc="lower left", bbox_to_anchor=(0.225, 0.0),
             frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)   # opaque, no edge: masks the window line
    a.text(pd.Timestamp(oos_start) - pd.Timedelta(days=8), 72000, "forecast targets\nfrom 30 Aug 2020",
           ha="right", va="top", color="0.35", fontsize=small)
    shade(b, "ma200_regime")
    l1, = b.plot(reg["day"], reg["btc_close"], color="#1f3b73", lw=0.9, zorder=2, label="close")
    l2, = b.plot(reg["day"], reg["ma200"], color="#c47a00", lw=1.1, zorder=2, label="200-day moving average")
    b.set_title("Close below its 200-day moving average (sensitivity rule)", loc="left")
    b.legend(handles=[l1, l2, Patch(color="0.84", label="below the average (bear)")], loc="lower left",
             bbox_to_anchor=(0.225, 0.0), frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
             labelspacing=0.3, borderpad=0.3)
    b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.075, 1.04, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=plt.rcParams["axes.titlesize"] + 1, ha="right", va="bottom")
    fig.tight_layout()
    path = os.path.join(out_dir, png)
    fig.savefig(path, dpi=300)
    return fig, path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m model.arm_regime_split")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--panel", required=True)
    r.add_argument("--variable-spec", required=True)
    r.add_argument("--klines-dir", required=True)
    r.add_argument("--arm-results", required=True)
    r.add_argument("--out-root", required=True)
    r.add_argument("--label", default="bullbear")
    r.add_argument("--stamp", default=None)
    r.add_argument("--expect-forecasts-md5", default=None)
    r.add_argument("--expect-panel-md5", default=None)
    r.add_argument("--sq4", choices=["iut", "dk2"], default="iut",
                   help="H4 p in the four-test Holm of each regime set: iut = the run's stored bootstrap p (3 Nov 2026); "
                        "dk2 = the two-sided Driscoll-Kraay p of D8 Deviation 7 (8 Nov 2026)")
    f = sub.add_parser("figure")
    f.add_argument("--out-dir", required=True)
    f.add_argument("--png", default="fig_A07_btc_regimes.png")
    m = sub.add_parser("md5sums")
    m.add_argument("--out-dir", required=True)
    g = sub.add_parser("diagnostics")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--run-dir", required=True)
    t = sub.add_parser("results")
    t.add_argument("--out-dir", required=True)
    t.add_argument("--run-dir", required=True)
    t.add_argument("--name", default="arm_A07_bullbear_results.csv")
    a = ap.parse_args(argv)
    if a.cmd != "run":                      # every later step is logged in the run folder before it runs
        with open(os.path.join(a.out_dir, "commands.txt"), "a") as fh:
            fh.write("python -m model.arm_regime_split " + " ".join(sys.argv[1:] if argv is None else argv) + "\n")
    if a.cmd == "run":
        print(run(a))
    elif a.cmd == "figure":
        print(figure(a.out_dir, a.png)[1])
    elif a.cmd == "diagnostics":
        diagnostics(a.out_dir, a.run_dir)
    elif a.cmd == "results":
        print(results_table(a.out_dir, a.run_dir, a.name))
    else:
        write_md5sums(a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
