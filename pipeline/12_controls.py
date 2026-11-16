#!/usr/bin/env python3
"""
12_controls.py -- daily macro-financial and news controls on the 24/7 UTC calendar.

Input    data/control_var/: fred_VIXCLS, fred_T10Y2Y, fred_DGS10, fred_DGS2, fred_BAA10Y,
         fred_BAMLH0A0HYM2, fred_USEPUINDXD, fred_DTWEXBGS, lseg_DXY_daily, lseg_MOVE_daily,
         lseg_XAU_spot_daily, crypto_fear_and_greed_daily, wrds_ravenpack_crypto_2019_2023
         (Tether entity only). Read-only.
Output   data/pipeline/panel/controls_daily.parquet (+ .stats.json) -- one row per UTC day,
         1 Aug 2019 - 31 Jan 2023 (1,280 days). The coverage report
         (data/pipeline/panel/reports/controls_report_20261007.md) was written by hand from checks on this table.

Rules (MASTERPLAN_NOV15.md section 7, 'Decisions of 6 Oct 2026' point 10, frozen in
planning/pre_analysis_note_D8.md section 2): day-t values only, no HAR form.
Transforms: VIX and MOVE log level; term spread (T10Y2Y = DGS10 - DGS2) and BAA10Y levels;
10-year yield daily change; DXY and gold (XAU= mid) daily log returns; EPU log level lagged
one day; Tether news log(1 + distinct headlines) and its daily log change; weekend dummy.
Weekend and holiday levels are carried forward, so changes and returns on those days are 0.
Everything is z-scored INSIDE each estimation window by the engine; nothing here is
standardised, and no estimation is done here.

Calendar-day convention: days are UTC calendar days. FRED/LSEG date stamps are the market's
session date; the session closes inside the same UTC day (US cash close 21:00-22:00 UTC at
the latest), so the session date equals the UTC day on which the value becomes known.
Carry-forward: a weekday-only series is placed on the calendar, then forward-filled from its
own last observation (history before the window is used, so 1 Aug 2019 has a value where the
source has one). Weekend rows in weekday series are dropped (HY OAS has 12 ICE month-end
valuations on weekends) so that weekends are always carried.

Named parameters (open choices, see the report): BAA lag, RavenPack relevance threshold,
distinct-headline definition, DTWEXBGS point-in-time variant.

Usage (from code/):  python pipeline/12_controls.py
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

CODE_DIR = Path(__file__).resolve().parents[1]
CV = CODE_DIR / "data" / "control_var"
OUT = Path(os.environ.get("THESIS_PIPELINE_OUT", CODE_DIR / "data" / "pipeline")) / "panel"
OUT_FILE = OUT / "controls_daily.parquet"

WINDOW_START, WINDOW_END, N_DAYS = "2019-08-01", "2023-01-31", 1280   # as config.py
FIRST_ORIGIN = "2019-08-30"        # first in-sample forecast origin (30 days of RV history)

# ---- named parameters (open choices) ---------------------------------------------------
RP_ENTITY_ID = "9D4B6F"            # RavenPack entity 'Tether'
RP_MIN_RELEVANCE = 75              # brief's threshold; the frozen documents do not state one
DISTINCT_BY = "headline_string"    # exact headline string per UTC day (not per entity row)
HY_DROP_WEEKEND_ROWS = True        # ICE month-end weekend valuations are treated as closed days
EPU_LAG_DAYS = 1                   # calendar days (EPU is published for all 7 days)

US_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def read_fred(name: str) -> pd.Series:
    d = pd.read_csv(CV / f"fred_{name}.csv")
    s = pd.Series(pd.to_numeric(d[name], errors="coerce").to_numpy(),
                  index=pd.to_datetime(d["observation_date"]), name=name)
    assert s.index.is_unique and s.index.is_monotonic_increasing
    return s.dropna()


def read_lseg(name: str, col: str) -> pd.Series:
    d = pd.read_csv(CV / f"lseg_{name}_daily.csv")
    s = pd.Series(pd.to_numeric(d[col], errors="coerce").to_numpy(), index=pd.to_datetime(d["date"]), name=name)
    assert s.index.is_unique and s.index.is_monotonic_increasing
    return s.dropna()


def to_calendar(s: pd.Series, cal: pd.DatetimeIndex, drop_weekend: bool = True):
    """Forward-filled calendar series and the 'own observation on this day' mask.
    The fill starts at the source's first observation (history before the window counts)."""
    if drop_weekend:
        s = s[s.index.dayofweek < 5]
    full = pd.date_range(min(s.index.min(), cal[0]), cal[-1], freq="D")
    raw = s.reindex(full)
    own = raw.notna()
    lvl = raw.ffill()
    return lvl, own


def pit_calendar(s: pd.Series, avail: pd.DatetimeIndex, cal: pd.DatetimeIndex) -> pd.Series:
    """Level known at the END of each UTC calendar day: observation d counts from its
    availability date onward (later observation overrides earlier on the same availability day)."""
    t = pd.Series(s.to_numpy(), index=avail)
    t = t[~t.index.duplicated(keep="last")].sort_index()
    full = pd.date_range(min(t.index.min(), cal[0]), cal[-1], freq="D")
    return t.reindex(full).ffill()


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    cal = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    assert len(cal) == N_DAYS, len(cal)
    win = lambda x: x.reindex(cal)
    cols: dict[str, pd.Series] = {}
    flags: dict[str, pd.Series] = {}
    meta: dict = {"series": {}}

    # ---- main controls ----------------------------------------------------------------
    vix, vix_own = to_calendar(read_fred("VIXCLS"), cal)
    move, move_own = to_calendar(read_lseg("MOVE", "MOVE"), cal)
    t10y2y, ts_own = to_calendar(read_fred("T10Y2Y"), cal)
    baa, baa_own = to_calendar(read_fred("BAA10Y"), cal)
    dgs10, dgs10_own = to_calendar(read_fred("DGS10"), cal)
    dxy, dxy_own = to_calendar(read_lseg("DXY", "DXY"), cal)
    xau_raw = pd.read_csv(CV / "lseg_XAU_spot_daily.csv")
    assert np.allclose(xau_raw["mid"], (xau_raw["Bid"] + xau_raw["Ask"]) / 2, atol=1e-3)
    xau, xau_own = to_calendar(pd.Series(((xau_raw["Bid"] + xau_raw["Ask"]) / 2).to_numpy(),
                                         index=pd.to_datetime(xau_raw["date"])), cal)

    cols["vix"] = win(vix); cols["log_vix"] = win(np.log(vix))
    cols["move"] = win(move); cols["log_move"] = win(np.log(move))
    cols["term_spread"] = win(t10y2y)
    cols["baa10y"] = win(baa)
    cols["dgs10"] = win(dgs10); cols["d_dgs10"] = win(dgs10.diff())
    cols["dxy"] = win(dxy); cols["ret_dxy"] = win(np.log(dxy).diff())
    cols["xau_mid"] = win(xau); cols["ret_gold"] = win(np.log(xau).diff())
    for k, o in [("vix", vix_own), ("move", move_own), ("term_spread", ts_own), ("baa10y", baa_own),
                 ("dgs10", dgs10_own), ("dxy", dxy_own), ("gold", xau_own)]:
        flags[f"cf_{k}"] = ~win(o).astype(bool)

    # BAA10Y one-business-day-lag variant: observation d is usable from the next US business day
    baa_s = read_fred("BAA10Y"); baa_s = baa_s[baa_s.index.dayofweek < 5]
    cols["baa10y_lag1"] = win(pit_calendar(baa_s, baa_s.index + US_BDAY, cal))

    # EPU: all 7 days; value of day t-1 is used on day t
    epu = read_fred("USEPUINDXD")
    full = pd.date_range(epu.index.min(), cal[-1], freq="D")
    assert epu.reindex(full).notna().loc[cal[0] - pd.Timedelta(days=EPU_LAG_DAYS):].all(), "EPU gaps"
    assert (epu > 0).all()
    cols["epu"] = win(epu.reindex(full))
    cols["log_epu_lag1"] = win(np.log(epu.reindex(full)).shift(EPU_LAG_DAYS))
    flags["cf_epu"] = pd.Series(False, index=cal)       # a 7-day series: nothing is carried

    # ---- Tether news (RavenPack, market level) ------------------------------------------
    rp = pd.read_csv(CV / "wrds_ravenpack_crypto_2019_2023.csv",
                     usecols=["rp_entity_id", "rpa_date_utc", "timestamp_utc", "entity_name", "relevance", "headline"],
                     dtype={"rp_entity_id": str, "rpa_date_utc": str, "timestamp_utc": str, "entity_name": str, "headline": str})
    rp = rp[rp["rp_entity_id"] == RP_ENTITY_ID].copy()
    assert set(rp["entity_name"]) == {"Tether"}, set(rp["entity_name"])
    ts = pd.to_datetime(rp["timestamp_utc"], utc=True, format="ISO8601")   # explicit UTC parse
    assert str(ts.dt.tz) == "UTC"
    rp["day"] = ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    assert (rp["day"].dt.strftime("%Y-%m-%d") == rp["rpa_date_utc"]).all(), "rpa_date_utc is not the UTC day"
    rp["relevance"] = pd.to_numeric(rp["relevance"])
    assert rp["headline"].notna().all()
    n_rows_all = len(rp)
    rp = rp[rp["relevance"] >= RP_MIN_RELEVANCE]
    meta["tether_rows_total"] = n_rows_all
    meta["tether_rows_after_relevance"] = int(len(rp))
    assert rp["day"].between(cal[0], cal[-1]).all()
    n_head = rp.groupby("day")["headline"].nunique().reindex(cal, fill_value=0)
    n_rows = rp.groupby("day").size().reindex(cal, fill_value=0)
    cols["tether_n_headlines"] = n_head.astype("int64")
    cols["tether_log1p_headlines"] = np.log1p(n_head)
    cols["tether_dlog_headlines"] = np.log1p(n_head).diff()      # first day NaN (no news before 1 Aug 2019 in the file)
    cols["tether_n_rows"] = n_rows.astype("int64")
    cols["rob_tether_log1p_rows"] = np.log1p(n_rows)             # robustness: count per entity row
    meta["tether_days_zero_headlines"] = int((n_head == 0).sum())

    # ---- weekend dummy ------------------------------------------------------------------
    cols["weekend"] = pd.Series((cal.dayofweek >= 5).astype("int8"), index=cal)

    # ---- robustness series --------------------------------------------------------------
    hy_s = read_fred("BAMLH0A0HYM2")
    meta["hy_weekend_rows_in_window"] = int(((hy_s.index.dayofweek >= 5) & (hy_s.index >= cal[0]) & (hy_s.index <= cal[-1])).sum())
    hy, hy_own = to_calendar(hy_s, cal, drop_weekend=HY_DROP_WEEKEND_ROWS)
    cols["rob_hy_oas"] = win(hy); flags["cf_hy_oas"] = ~win(hy_own).astype(bool)
    hy_w = hy_s[hy_s.index.dayofweek < 5]
    cols["rob_hy_oas_lag1"] = win(pit_calendar(hy_w, hy_w.index + US_BDAY, cal))

    dtw = read_fred("DTWEXBGS")
    dtw_lvl, dtw_own = to_calendar(dtw, cal)
    cols["rob_dtwexbgs"] = win(dtw_lvl)
    cols["rob_dtwexbgs_logret"] = win(np.log(dtw_lvl).diff())
    flags["cf_dtwexbgs"] = ~win(dtw_own).astype(bool)
    # point-in-time variant: the Fed publishes the previous business week on Monday 16:15 ET
    # (a Monday holiday moves the release to the next business day)
    monday_next = dtw.index - pd.to_timedelta(dtw.index.dayofweek, unit="D") + pd.Timedelta(days=7)
    avail = pd.DatetimeIndex([US_BDAY.rollforward(m) for m in monday_next])
    dtw_pit = pit_calendar(dtw, avail, cal)
    cols["rob_dtwexbgs_logret_pit"] = win(np.log(dtw_pit).diff())

    g = pd.read_csv(CV / "crypto_fear_and_greed_daily.csv")
    gts = pd.to_datetime(g["timestamp"], unit="s", utc=True)
    assert (gts.dt.tz_localize(None).dt.normalize() == pd.to_datetime(g["date"])).all()
    gs = pd.Series(g["value"].to_numpy(dtype=float), index=pd.to_datetime(g["date"])).sort_index()
    assert gs.index.is_unique
    cols["rob_fng"] = gs.reindex(cal)
    flags["cf_fng"] = cols["rob_fng"].isna()                      # no own value that day (not filled)

    # ---- assemble -----------------------------------------------------------------------
    df = pd.DataFrame(cols, index=cal)
    for k, v in flags.items():
        df[k] = v.astype(bool)
    df.index.name = "day"
    df = df.reset_index()
    df["day"] = df["day"].dt.date
    df.insert(1, "in_forecast_window", pd.to_datetime(df["day"]) >= pd.Timestamp(FIRST_ORIGIN))

    # ---- invariants (asserted, not printed) -------------------------------------------------
    assert len(df) == N_DAYS and df["day"].is_unique
    assert df["day"].iloc[0].isoformat() == WINDOW_START and df["day"].iloc[-1].isoformat() == WINDOW_END
    fw = df[df["in_forecast_window"]]
    no_na = [c for c in df.columns if c not in ("rob_fng",) and df[c].dtype != bool]
    for c in no_na:
        assert fw[c].notna().all(), f"missing values in forecast window: {c}"
    assert fw["rob_fng"].notna().all()
    # carried days: changes and returns are exactly 0
    for lvl, chg in [("dgs10", "d_dgs10"), ("dxy", "ret_dxy"), ("xau_mid", "ret_gold")]:
        key = {"dgs10": "cf_dgs10", "dxy": "cf_dxy", "xau_mid": "cf_gold"}[lvl]
        sub = df[df[key] & df[chg].notna()]
        assert (sub[chg] == 0).all(), chg
    assert (df.loc[df["weekend"] == 1, ["d_dgs10", "ret_dxy", "ret_gold"]].fillna(0) == 0).all().all()
    assert df["weekend"].sum() == int((cal.dayofweek >= 5).sum())

    # ---- write (temp -> replace) ---------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp, compression="zstd")
    chk = pq.read_table(tmp).to_pandas()
    assert len(chk) == N_DAYS and list(chk.columns) == list(df.columns)
    os.replace(tmp, OUT_FILE)

    meta.update({
        "step": "12_controls",
        "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)), "columns": list(df.columns),
        "parameters": {"RP_ENTITY_ID": RP_ENTITY_ID, "RP_MIN_RELEVANCE": RP_MIN_RELEVANCE, "DISTINCT_BY": DISTINCT_BY,
                       "HY_DROP_WEEKEND_ROWS": HY_DROP_WEEKEND_ROWS, "EPU_LAG_DAYS": EPU_LAG_DAYS,
                       "FIRST_ORIGIN": FIRST_ORIGIN},
        "inputs": {p.name: md5(p) for p in sorted(CV.glob("*.csv"))
                   if p.stem in {"fred_VIXCLS", "fred_T10Y2Y", "fred_DGS10", "fred_DGS2", "fred_BAA10Y", "fred_BAMLH0A0HYM2",
                                 "fred_USEPUINDXD", "fred_DTWEXBGS", "lseg_DXY_daily", "lseg_MOVE_daily",
                                 "lseg_XAU_spot_daily", "crypto_fear_and_greed_daily", "wrds_ravenpack_crypto_2019_2023"}},
        "code_md5": md5(Path(__file__).resolve()), "output_md5": md5(OUT_FILE),
    })
    tmp2 = OUT / "controls_daily.parquet.stats.json.tmp"
    tmp2.write_text(json.dumps(meta, indent=1, default=str))
    os.replace(tmp2, OUT / "controls_daily.parquet.stats.json")
    print("wrote", OUT_FILE, len(df), "rows", len(df.columns), "columns")


if __name__ == "__main__":
    main()
