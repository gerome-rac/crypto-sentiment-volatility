#!/usr/bin/env python3
"""event_days.py -- the two days with the largest gain of the attention step (29 Jan 2021, 7 Sep 2021): what R0 and R1
forecast, what happened, and how unusual attention was on the last day the forecast could use. Written 14 Nov 2026 for
the Discussion (attention paragraph). Reads the main run and the main panel; writes two CSV files next to this script.

  event_days_forecasts.csv   target, coin, log forecast of R0 and R1, outcome, gain in squared error
  event_days_attention.csv   target, coin, series, z: (log(1 + x) on the day before the target - mean) / SD,
                             mean and SD over the 364 days before that day (M_ = market series)

Checks (asserted): the daily cross-coin mean gains sum to 7.27 over the 885 days and the two days are the two largest.
Usage (from the code/ folder): python <this file> <main_run_dir> <panel.parquet> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run, panel, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
f = pd.read_parquet(run / "forecasts.parquet"); f["target"] = pd.to_datetime(f["target"])
w = f[f.rung.isin(["R0", "R1"])].pivot_table(index=["target", "coin"], columns="rung", values="f_log")
w["y"] = f[f.rung == "R0"].set_index(["target", "coin"]).y
w["gain"] = (w.y - w.R0) ** 2 - (w.y - w.R1) ** 2
daily = w.groupby("target").gain.mean()
DAYS = [pd.Timestamp("2021-01-29"), pd.Timestamp("2021-09-07")]
assert len(daily) == 885 and abs(daily.sum() - 7.27) < 0.005
assert set(daily.sort_values(ascending=False).index[:2]) == set(DAYS)

p = pd.read_parquet(panel)
dc = [c for c in p.columns if c in ("date", "day", "utc_date")][0]
p[dc] = pd.to_datetime(p[dc])
A = ["att_reddit", "att_stocktwits", "att_twitter", "att_google", "att_wiki", "att_news",
     "mkt_reddit", "mkt_stocktwits", "mkt_twitter", "mkt_google_cryptocurrency", "mkt_wiki"]
rows = []
for day in DAYS:
    org = day - pd.Timedelta(days=1)
    for coin, g in p.groupby("coin"):
        g = g.set_index(dc).sort_index()
        ref = g.loc[org - pd.Timedelta(days=364): org - pd.Timedelta(days=1)]
        assert len(ref) == 364
        for v in A:
            x, rr = np.log1p(g[v].astype(float)), np.log1p(ref[v].astype(float)).dropna()
            rows.append({"target": day.date(), "coin": coin, "series": v.replace("mkt_", "M_"),
                         "z": (x.loc[org] - rr.mean()) / rr.std(), "n_ref": len(rr)})
att = pd.DataFrame(rows)
fc = w.loc[DAYS].reset_index()
fc["share_of_day"] = fc.gain / fc.groupby("target").gain.transform("sum")
out.mkdir(parents=True, exist_ok=True)
fc.to_csv(out / "event_days_forecasts.csv", index=False)
att.to_csv(out / "event_days_attention.csv", index=False)
pd.set_option("display.width", 250)
print(fc.round(3).to_string(index=False))
print(att.pivot_table(index=["target", "coin"], columns="series", values="z").round(1).to_string())
print("daily mean gain:", {str(d.date()): round(float(daily[d]), 2) for d in DAYS}, "| n_ref min", int(att.n_ref.min()))
