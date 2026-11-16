#!/usr/bin/env python3
"""
09_volatility.py -- step 9: daily realized-volatility block per coin and UTC day.

Input    data/raw/binance_spot_1m/<PAIR>/<PAIR>-1m-YYYY-MM.zip (Binance spot 1-minute klines,
         BTC/ETH/BNB/XRP/DOGE against USDT, 1 Aug 2019 - 31 Jan 2023), read-only.
Output   data/pipeline/panel/volatility_daily.parquet: one row per coin and UTC day
         (5 x 1,280 = 6,400 rows) with its .stats.json.

Construction (frozen: MASTERPLAN_NOV15.md section 7, 'Decisions of 5-6 Oct 2026' (1)-(6);
pre_analysis_note_D8.md section 2, R0). All estimators come from pipeline/cj_methods.py, an
unchanged copy of the verified C/J method library cj_methods_20261005.py (md5
9abc2fb6048dc712f9c6f06edd223c80); entry A1 with outage = 'split'.
  - Prices: previous tick. A 1-minute kline's close is the last trade of that minute; a missing
    minute carries the previous close. Days are UTC calendar days (D21/PF1): the minute index is
    taken from open_time (ms since the epoch, UTC) and checked against a tz-aware UTC timestamp.
  - Returns: 288 five-minute log returns per UTC day, 100 * diff(log price), one base grid
    (minutes 0, 5, ..., 1440); the first return runs from the previous day's 23:59 close, on
    1 Aug 2019 from the first bar's open. No subsampling.
  - RV  = sum of the 288 squared returns (the full-day RV; the forecast target).
  - BV  = adjacent bipower variation, mu1^-2 N/(N-1) sum |r_j||r_{j-1}| (HT 2005 eq. 3; BNS 2004).
  - TQ  = adjacent tripower quarticity, N mu43^-3 N/(N-2) sum |r_j r_{j-1} r_{j-2}|^(4/3).
  - z_ht = Huang-Tauchen (2005, eq. 9) max-adjusted ratio statistic
           sqrt(N) (1 - BV/RV_obs) / sqrt(theta max(1, TQ/BV^2)), theta = pi^2/4 + pi - 5.
  - jump_flag = z_ht > Phi^-1(0.999) = 3.0902.
  - J = jump_flag * max(RV_base - BV, 0), C = RV - J  (MASTERPLAN 5-6 Oct (3)).
  - Outage days (overnight-style split, 5-6 Oct (6)): BV, TQ and the ratio test use only the
    5-minute intervals untouched by an outage minute; RV stays the full-day RV. rv_obs is the
    RV over the untouched intervals (equal to rv on days without outage). RV_base in J is set by
    the named parameter J_OUTAGE_BASIS:
        'rv_obs'  (main; the library's A1 and the 5 Oct comparison run: J = max(RV_obs - BV, 0);
                   decided by Gerome on 7 Oct 2026, Deviation 1 of planning/pre_analysis_note_D8.md)
        'rv_full' (literal wording of 5-6 Oct (6), "... counts as continuous unless the day is a jump
                   day": J = max(RV - BV, 0); always stored for comparison as c_rvfull, j_rvfull).
    The two versions differ only on outage jump days (13 coin-days).
    Outage minutes: the pair's own missing minutes, or runs of at least 5 minutes in which all five
    pairs are missing or have no trade (library loader, outage_rule = 'script').
  - Robustness ingredients, clearly named (register arms):
        abd_drop_flag   A53: ABD 2007 fn 9, more than 20 consecutive 5-minute intervals without a
                        trade (library drop rule, basis 'trades').
        medrv           A54 (HAR-MedRV benchmark): MedRV of ADS (2012) on all 288 returns
                        (carry-forward; as the 5 Oct diagnostics). PROVISIONAL: the constants must be
                        checked against ADS 2012 before use (MASTERPLAN 5-6 Oct (3)).
        medrv_split     the same on the untouched intervals only (outage handling of MedRV open).
        doge_2019_20    the DOGE coin-days 1 Aug 2019 - 31 Dec 2020 (robustness 'without DOGE 2019-20').
    Not built: A50 (MedRV C/J; MedRQ constants unchecked), A51 (Kolokolov-Reno; provisional),
    A62 (subsampled RV; edge rule open).

No averages, logs or window statistics are formed here: the panel assembly forms the HAR terms
(averages of daily values, logged afterwards) and the estimation engine every in-window quantity.

Columns (units: returns in percent, RV/BV/C/J/MedRV in %^2 per UTC day, TQ in %^4)
    coin, pair, day_utc, rv, rv_obs, bv, tq, z_ht, crit_ht, jump_flag, j_raw, j, c, j_raw_rvfull, j_rvfull, c_rvfull,
    outage_flag, outage_min, missing_min, n_returns, n_used, zero_return_share,
    zero_return_share_all, abd_drop_flag, abd_drop_maxrun, medrv, medrv_split, doge_2019_20

Subcommands (from code/)
------------------------
    python pipeline/09_volatility.py build
    python pipeline/09_volatility.py check --ref-daily <cj_focused_empirical_daily_20261005.parquet>
           --ref-diag <cj_daily_diagnostics_20261005.parquet> [--ref-summary <..._summary_20261005.csv>]
           --out <checks.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import cj_methods as cjm  # noqa: E402
from pipeline import config, runner  # noqa: E402
from pipeline.runner import check, log  # noqa: E402

STEP = "09_volatility"
CODE_SHA = runner.code_fingerprint(__file__, cjm.__file__)
RULES_SHA = "cj_methods-" + runner.file_md5(cjm.__file__)[:16]

# ---- construction parameters (frozen values; one name each) -------------------------------------
GRID_STEP = 5                 # minutes; 288 returns per UTC day
ALPHA = 0.999                 # daily confidence level of the ratio test
APPROACH = "A1"               # library entry: adjacent BV + Huang-Tauchen ratio test
OUTAGE_MODE = "split"         # overnight-style split
OUTAGE_RULE = "script"        # outage-minute mask of the library loader
OUTAGE_MIN_RUN = 5            # minutes of exchange-wide inactivity that make an outage run
DROP_RUN = 20                 # ABD 2007 fn 9 (arm A53)
J_OUTAGE_BASIS = "rv_obs"     # decided 7 Oct 2026 (Gerome), Deviation 1: library A1; 'rv_full' = literal wording
PAIRS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT", "XRP": "XRPUSDT", "DOGE": "DOGEUSDT"}
RAW = config.RAW_DIR / "binance_spot_1m"
FINAL = config.OUT_DIR / "panel" / "volatility_daily.parquet"
N_RET = 1440 // GRID_STEP


def utc_minute_audit(pair: str, t0: pd.Timestamp, n_days: int) -> np.ndarray:
    """Present minutes per UTC day from open_time, with an explicit UTC assertion (D21/PF1)."""
    months = pd.period_range(t0.tz_localize(None), (t0 + pd.Timedelta(days=n_days - 1)).tz_localize(None), freq="M")
    present = np.zeros(n_days, np.int64)
    t0_ms = int(t0.timestamp() * 1000)
    for mo in months:
        f = RAW / pair / f"{pair}-1m-{mo.strftime('%Y-%m')}.zip"
        with zipfile.ZipFile(f) as z:
            nm = [x for x in z.namelist() if x.endswith(".csv")]
            with z.open(nm[0]) as fh:
                ot = pd.read_csv(fh, header=None, usecols=[0], names=["ot"], dtype={"ot": "int64"})["ot"].to_numpy()
        ts = pd.to_datetime(ot, unit="ms", utc=True)
        check(str(ts.tz) == "UTC", f"{pair} {mo}: timestamps are tz-aware UTC")
        check(bool(np.all((ts.year == mo.year) & (ts.month == mo.month))), f"{pair} {mo}: every bar in its UTC month")
        check(len(np.unique(ot)) == len(ot), f"{pair} {mo}: no duplicate open_time")
        day_idx = (ot - t0_ms) // 86_400_000
        day_ts = ((ts.normalize() - t0) // pd.Timedelta(days=1)).to_numpy()
        check(bool(np.array_equal(day_idx, day_ts)), f"{pair} {mo}: minute-index day equals UTC calendar day")
        ok = (day_idx >= 0) & (day_idx < n_days)
        present += np.bincount(day_idx[ok], minlength=n_days)[:n_days]
    return present


def build() -> None:
    t_start = time.time()
    checks: list = []
    t0 = pd.Timestamp(config.WINDOW_START, tz="UTC")
    n_days = config.N_DAYS
    check(str(t0.tz) == "UTC", "window start is a UTC timestamp", checks)
    check(GRID_STEP == 5 and N_RET == 288 and ALPHA == 0.999, "frozen grid (5 min, 288 returns) and alpha 0.999", checks)
    crit = float(cjm.norm.ppf(ALPHA))
    check(abs(crit - 3.0902) < 1e-4, f"critical value {crit:.6f} = 3.0902", checks)
    check(abs(cjm.THETA_BV - (np.pi ** 2 / 4 + np.pi - 5)) < 1e-15, f"theta = pi^2/4 + pi - 5 = {cjm.THETA_BV:.6f}", checks)
    check(J_OUTAGE_BASIS in ("rv_full", "rv_obs"), "J_OUTAGE_BASIS is rv_full or rv_obs", checks)

    log("loading 1-minute klines (library loader, read-only)")
    panels = cjm.load_binance_1m(str(RAW), pairs=tuple(PAIRS.values()), start=config.WINDOW_START,
                                 end=config.WINDOW_END, outage_rule=OUTAGE_RULE, min_run=OUTAGE_MIN_RUN)
    zips = sorted(str(p) for pr in PAIRS.values() for p in (RAW / pr).glob(f"{pr}-1m-*.zip"))
    check(len(zips) == 5 * 42, f"210 monthly zip files (got {len(zips)})", checks)
    inputs = runner.describe_inputs(zips)

    frames = []
    for coin, pair in PAIRS.items():
        d = panels[pair]
        check(d.n_days == n_days and d.contiguous, f"{coin}: {n_days} consecutive UTC days", checks)
        present = utc_minute_audit(pair, t0, n_days)
        missing = 1440 - present
        check(int(missing.sum()) == int(d.notes["missing_minutes"]), f"{coin}: missing minutes {int(missing.sum())} "
              f"equal the loader's count", checks)
        out = cjm.run(APPROACH, d, grid_step=GRID_STEP, alpha=ALPHA, outage=OUTAGE_MODE, drop_run=DROP_RUN)
        prep = cjm._Prep(d, np.arange(d.n_days))
        r_all = prep.r(GRID_STEP, 0)
        r_sp = prep.rmask(GRID_STEP, 0, OUTAGE_MODE)
        check(r_all.shape == (n_days, N_RET), f"{coin}: 288 returns on every day", checks)
        medrv_cf, _ = cjm.medrv(r_all)
        medrv_sp, _ = cjm.medrv(r_sp, cjm._nobs(r_sp))
        zero_all = (r_all == 0).mean(axis=1)
        jump = out["jump"].to_numpy(bool)
        rv = out["rv"].to_numpy(float)
        rv_obs = out["rv_obs"].to_numpy(float)
        bv = out["bv"].to_numpy(float)
        j_lib = out["j"].to_numpy(float)                        # library A1: jump * max(rv_obs - bv, 0)
        base = rv if J_OUTAGE_BASIS == "rv_full" else rv_obs
        j_raw = np.maximum(base - bv, 0.0)
        j = np.where(jump, j_raw, 0.0)
        j_raw_rvfull = np.maximum(rv - bv, 0.0)                 # literal wording of 5-6 Oct (6), for comparison
        j_rvfull = np.where(jump, j_raw_rvfull, 0.0)
        if J_OUTAGE_BASIS == "rv_obs":
            check(bool(np.allclose(j, j_lib, rtol=1e-12, atol=1e-12)), f"{coin}: j equals the library's A1 J", checks)
        days = pd.to_datetime(out["day"]).dt.date
        frames.append(pd.DataFrame({
            "coin": coin, "pair": pair, "day_utc": days,
            "rv": rv, "rv_obs": rv_obs, "bv": bv, "tq": out["tq"].to_numpy(float),
            "z_ht": out["stat"].to_numpy(float), "crit_ht": out["crit"].to_numpy(float), "jump_flag": jump,
            "j_raw": j_raw, "j": j, "c": rv - j, "j_raw_rvfull": j_raw_rvfull, "j_rvfull": j_rvfull, "c_rvfull": rv - j_rvfull,
            "outage_flag": out["outage_min"].to_numpy() > 0, "outage_min": out["outage_min"].to_numpy(np.int64),
            "missing_min": missing.astype(np.int64), "n_returns": np.full(n_days, N_RET, np.int64),
            "n_used": out["n_used"].to_numpy(float).astype(np.int64),
            "zero_return_share": out["zero_share"].to_numpy(float), "zero_return_share_all": zero_all,
            "abd_drop_flag": out["drop_day"].to_numpy(bool), "abd_drop_maxrun": out["drop_maxrun"].to_numpy(np.int64),
            "medrv": medrv_cf, "medrv_split": medrv_sp,
        }))
    df = pd.concat(frames, ignore_index=True)
    df["doge_2019_20"] = (df["coin"] == "DOGE") & (pd.to_datetime(df["day_utc"]) <= pd.Timestamp("2020-12-31"))

    # ---- invariants ------------------------------------------------------------------------------
    cal = pd.date_range(config.WINDOW_START, config.WINDOW_END, freq="D").date
    check(len(df) == 5 * n_days, f"{len(df)} rows = 6,400", checks)
    check(all((g["day_utc"].to_numpy() == cal).all() for _, g in df.groupby("coin")), "complete UTC calendar per coin", checks)
    check(not df.duplicated(["coin", "day_utc"]).any(), "one row per coin and day", checks)
    num = ["rv", "rv_obs", "bv", "tq", "z_ht", "j", "c", "j_rvfull", "c_rvfull", "medrv", "medrv_split"]
    check(bool(np.isfinite(df[num].to_numpy()).all()), "all numeric columns finite", checks)
    check(bool((df["rv"] > 0).all() and (df["bv"] > 0).all() and (df["c"] > 0).all()), "rv, bv, c > 0", checks)
    check(bool((df["j"] >= 0).all() and (df["j_rvfull"] >= 0).all()), "j >= 0", checks)
    check(bool(np.allclose(df["c"] + df["j"], df["rv"], rtol=0, atol=1e-9)), "c + j = rv", checks)
    check(bool(np.allclose(df["c_rvfull"] + df["j_rvfull"], df["rv"], rtol=0, atol=1e-9)), "c_rvfull + j_rvfull = rv", checks)
    check(bool((df["jump_flag"] == (df["z_ht"] > crit)).all()), "jump_flag = z_ht > crit", checks)
    check(bool((df.loc[~df["jump_flag"], "j"] == 0).all()), "j = 0 on non-jump days", checks)
    check(bool(np.allclose(df.loc[df["jump_flag"] & ~df["outage_flag"], "c"],
                           df.loc[df["jump_flag"] & ~df["outage_flag"], "bv"], rtol=1e-12, atol=1e-9)),
          "c = bv on jump days without outage", checks)
    noo = ~df["outage_flag"]
    check(bool((df.loc[noo, "rv_obs"] == df.loc[noo, "rv"]).all() and (df.loc[noo, "n_used"] == N_RET).all()),
          "rv_obs = rv and n_used = 288 on days without outage", checks)
    check(bool((df.loc[noo, "j"] == df.loc[noo, "j_rvfull"]).all()), "j = j_rvfull on days without outage", checks)
    ndiff = df["c"] != df["c_rvfull"]
    check(bool((~ndiff | (df["outage_flag"] & df["jump_flag"])).all()) and int(ndiff.sum()) == int((df["outage_flag"] & df["jump_flag"]).sum()),
          f"c differs from c_rvfull on {int(ndiff.sum())} coin-days, exactly the outage jump days", checks)
    od = df.loc[df["outage_flag"]]
    check(len(od) == 90 and od["day_utc"].nunique() == 18, f"outage coin-days {len(od)} = 90 on "
          f"{od['day_utc'].nunique()} = 18 UTC days", checks)
    check(all(set(g["day_utc"]) == set(od["day_utc"]) for _, g in od.groupby("coin")), "outage days identical across coins", checks)
    check(int(df["abd_drop_flag"].sum()) == 65 and bool(df.loc[df["abd_drop_flag"], "outage_flag"].all()),
          f"A53 drop flag on {int(df['abd_drop_flag'].sum())} = 65 coin-days, all outage days", checks)
    check(int(df["doge_2019_20"].sum()) == 519, f"doge_2019_20 = {int(df['doge_2019_20'].sum())} = 519 coin-days", checks)
    check(bool((df["missing_min"] <= df["outage_min"]).all()), "missing minutes are outage minutes", checks)

    tmp = runner.tmp_path(FINAL)
    df.to_parquet(tmp, index=False)
    back = pd.read_parquet(tmp)
    check(back.shape == df.shape and bool(np.array_equal(back["rv"].to_numpy(), df["rv"].to_numpy())), "re-read equals the table", checks)
    runner.commit(tmp, FINAL)
    counts = {"rows": int(len(df)), "jump_days": {c: int(g["jump_flag"].sum()) for c, g in df.groupby("coin")},
              "outage_coin_days": int(df["outage_flag"].sum()), "outage_jump_days": int((df["outage_flag"] & df["jump_flag"]).sum()),
              "abd_drop_flag": int(df["abd_drop_flag"].sum()), "missing_minutes": {c: int(g["missing_min"].sum()) for c, g in df.groupby("coin")}}
    params = dict(GRID_STEP=GRID_STEP, ALPHA=ALPHA, APPROACH=APPROACH, OUTAGE_MODE=OUTAGE_MODE, OUTAGE_RULE=OUTAGE_RULE,
                  OUTAGE_MIN_RUN=OUTAGE_MIN_RUN, DROP_RUN=DROP_RUN, J_OUTAGE_BASIS=J_OUTAGE_BASIS, crit=crit,
                  theta=cjm.THETA_BV, library_version=cjm.__version__, library_md5=runner.file_md5(cjm.__file__))
    runner.write_stats(FINAL, step=STEP, inputs=inputs, code_sha=CODE_SHA, rules_sha=RULES_SHA, con=None,
                       counts=counts, checks=checks, extra={"parameters": params})
    log(f"wrote {FINAL} ({len(df)} rows, md5 {runner.file_md5(FINAL)}) in {time.time() - t_start:.1f} s")


def _cmp(a: pd.Series, b: pd.Series) -> dict:
    x, y = a.to_numpy(float), b.to_numpy(float)
    d = np.abs(x - y)
    rel = d / np.maximum(np.abs(y), 1e-12)
    return {"max_abs": float(np.nanmax(d)), "max_rel": float(np.nanmax(rel)), "n_diff_gt_1e-9_rel": int((rel > 1e-9).sum()),
            "nan_mismatch": int((np.isnan(x) != np.isnan(y)).sum())}


def check_against(ref_daily: str, ref_diag: str, ref_summary: str | None, out: str) -> None:
    v = pd.read_parquet(FINAL)
    v["day"] = pd.to_datetime(v["day_utc"])
    res: dict = {"volatility_daily_md5": runner.file_md5(FINAL), "ref_daily_md5": runner.file_md5(ref_daily),
                 "ref_diag_md5": runner.file_md5(ref_diag)}
    a = pd.read_parquet(ref_daily)
    a = a[(a["approach"] == APPROACH) & (a["outage"] == OUTAGE_MODE) & (a["grid"] == GRID_STEP)].copy()
    a["day"] = pd.to_datetime(a["day"])
    a = a.rename(columns={k: k + "_ref" for k in a.columns if k not in ("pair", "day")})
    m = v.merge(a, on=["pair", "day"], how="outer", indicator=True)
    res["daily_rows"] = {"ref": int(len(a)), "both": int((m["_merge"] == "both").sum()), "only_one_side": int((m["_merge"] != "both").sum())}
    res["daily_vs_A1_split"] = {
        "rv": _cmp(m["rv"], m["rv_ref"]), "z_ht": _cmp(m["z_ht"], m["stat_ref"]),
        "c_rvfull": _cmp(m["c_rvfull"], m["c_ref"]), "j_rvfull": _cmp(m["j_rvfull"], m["j_ref"]),
        "c": _cmp(m["c"], m["c_ref"]), "j": _cmp(m["j"], m["j_ref"]),
        "n_used": _cmp(m["n_used"], m["n_used_ref"]), "zero_return_share": _cmp(m["zero_return_share"], m["zero_share_ref"]),
        "outage_min": _cmp(m["outage_min"], m["outage_min_ref"]),
        "jump_flag_mismatch": int((m["jump_flag"].astype(bool) != m["jump_ref"].astype(bool)).sum()),
        "outage_flag_mismatch": int((m["outage_flag"] != m["gap_day_ref"].astype(bool)).sum()),
        "abd_drop_flag_mismatch": int((m["abd_drop_flag"] != m["drop_day_ref"].astype(bool)).sum()),
    }
    dj = m.loc[np.abs(m["j"] - m["j_ref"]) > 1e-9 * np.maximum(1, m["j_ref"].abs()), ["coin", "day_utc", "j", "j_ref", "c", "c_ref"]]
    res["c_j_differences_main_vs_ref"] = [{k: (str(r[k]) if k in ("coin", "day_utc") else float(r[k])) for k in dj.columns}
                                          for _, r in dj.iterrows()]
    dk = m.loc[np.abs(m["j_rvfull"] - m["j_ref"]) > 1e-9 * np.maximum(1, m["j_ref"].abs()), ["coin", "day_utc", "j_rvfull", "j_ref", "c_rvfull", "c_ref"]]
    res["c_j_differences_rvfull_vs_ref"] = [{k: (str(r[k]) if k in ("coin", "day_utc") else float(r[k])) for k in dk.columns}
                                            for _, r in dk.iterrows()]
    res["c_j_rvfull_differences_all_on_outage_jump_days"] = bool(m.loc[dk.index, "outage_flag"].all() and m.loc[dk.index, "jump_flag"].all())
    b = pd.read_parquet(ref_diag)
    b["day"] = pd.to_datetime(b["day"])
    mb = v.merge(b, on=["pair", "day"], how="outer", indicator=True)
    res["diag_rows"] = {"ref": int(len(b)), "both": int((mb["_merge"] == "both").sum())}
    res["daily_vs_diagnostics"] = {"rv": _cmp(mb["rv"], mb["RV"]), "medrv": _cmp(mb["medrv"], mb["MedRV"]),
                                   "outage_min": _cmp(mb["outage_min"], mb["gap_minutes"]),
                                   "n_used": _cmp(mb["n_used"], mb["n_obs_intervals"]),
                                   "zero_return_share": _cmp(mb["zero_return_share"], mb["zero_share_5m"]),
                                   "outage_flag_mismatch": int((mb["outage_flag"] != mb["outage_day"].astype(bool)).sum())}
    # jump rates on non-outage days: this table (A1) and the diagnostics' ABD (staggered) test
    nm = mb[~mb["outage_flag"]]
    res["jump_rate_non_outage_A1"] = {c: round(float(g["jump_flag"].mean()), 4) for c, g in nm.groupby("coin")}
    res["jump_rate_non_outage_ABD_staggered_diag"] = {c: round(float(g["jump_abd"].astype(bool).mean()), 4) for c, g in nm.groupby("coin")}
    res["jump_rate_non_outage_ABD_split_diag"] = {c: round(float(g["jump_split"].astype(bool).mean()), 4) for c, g in nm.groupby("coin")}
    res["n_non_outage_days"] = {c: int(len(g)) for c, g in nm.groupby("coin")}
    res["jump_days_A1"] = {c: int(g["jump_flag"].sum()) for c, g in v.groupby("coin")}
    res["outage_days"] = sorted(str(x) for x in v.loc[v["outage_flag"], "day_utc"].unique())
    res["outage_days_ref"] = sorted(str(x.date()) for x in b.loc[b["outage_day"].astype(bool), "day"].unique())
    res["outage_days_equal"] = res["outage_days"] == res["outage_days_ref"]
    if ref_summary:
        s = pd.read_csv(ref_summary)
        s = s[(s["approach"] == APPROACH) & (s["grid"] == GRID_STEP) & (s["outage"] == OUTAGE_MODE) & (s["table"] == "coin_year")
              & (s["alpha"].isna() | (s["alpha"] == ALPHA))]
        v["year"] = v["day"].dt.year
        g = v.groupby(["coin", "year"])
        mine = pd.DataFrame({"n_days": g.size(), "rv_mean": g["rv"].mean(), "jump_share": g["jump_flag"].mean(),
                             "drop_flag_share": g["abd_drop_flag"].mean(), "c_mean": g["c"].mean(),
                             "j_share_of_rv": g["j"].sum() / g["rv"].sum()}).reset_index()
        rows = []
        for met in ["n_days", "rv_mean", "jump_share", "drop_flag_share", "c_mean", "j_share_of_rv"]:
            ss = s[s["metric"] == met][["coin", "year", "value"]].merge(mine[["coin", "year", met]], on=["coin", "year"])
            rows.append({"metric": met, "cells": int(len(ss)), "max_abs_diff": float(np.abs(ss["value"] - ss[met]).max())})
        res["summary_coin_year_vs_ref"] = rows
    runner.write_json_atomic(out, res)
    log(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    c = sub.add_parser("check")
    c.add_argument("--ref-daily", required=True)
    c.add_argument("--ref-diag", required=True)
    c.add_argument("--ref-summary")
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    else:
        check_against(a.ref_daily, a.ref_diag, a.ref_summary, a.out)


if __name__ == "__main__":
    main()
