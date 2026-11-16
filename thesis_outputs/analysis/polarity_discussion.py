#!/usr/bin/env python3
"""polarity_discussion.py -- descriptive numbers for the polarity paragraph of the Discussion (15 Nov 2026).

1. pol_selection.csv   for each of the nine polarity terms of R2: number of the 885 estimation windows in which it is
                       selected, sign of its coefficient, first and last month of the forecast origin.
2. june2022_week.csv   13-19 June 2022: daily gain of R2 over R1 in squared error (cross-coin mean), mean log forecast
                       of R2 minus R1, BTC close.
3. june2022_polarity.csv  pooled polarity per coin on 11-18 June 2022 in standard deviations from its mean of the 365
                       days up to 11 June 2022.

Checks (asserted): gain of the polarity step 2.27 over the 885 days and 1.48 in the week; one-day level selected in
4.7% and one-day centered square in 14.1% of the windows (the figures of Appendix D).
Usage (from the code/ folder): python <this file> <main_run_dir> <panel.parquet> <state_series.csv> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run, panel, state_csv, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
c2 = pd.read_parquet(run / "coefs_R2.parquet"); c2["origin"] = pd.to_datetime(c2["origin"]); c2 = c2.set_index("origin").sort_index()
assert len(c2) == 885
rows = []
for t in [c for c in c2.columns if c.startswith("pol")]:
    s = c2[t][c2[t] != 0]
    rows.append({"term": t, "windows": len(s), "share_pct": 100 * len(s) / 885, "negative": int((s < 0).sum()), "positive": int((s > 0).sum()),
                 "first_origin_month": str(s.index.min().to_period("M")) if len(s) else "", "last_origin_month": str(s.index.max().to_period("M")) if len(s) else ""})
sel = pd.DataFrame(rows).set_index("term")
assert round(sel.loc["pol_h1", "share_pct"], 1) == 4.7 and round(sel.loc["pol_sq_h1", "share_pct"], 1) == 14.1

f = pd.read_parquet(run / "forecasts.parquet"); f["target"] = pd.to_datetime(f["target"])
w = f[f.rung.isin(["R1", "R2"])].pivot_table(index=["target", "coin"], columns="rung", values="f_log")
w["y"] = f[f.rung == "R1"].set_index(["target", "coin"]).y
w["gain"] = (w.y - w.R1) ** 2 - (w.y - w.R2) ** 2
daily = w.groupby("target").gain.mean()
assert abs(daily.sum() - 2.27) < 0.005 and abs(daily["2022-06-13":"2022-06-19"].sum() - 1.48) < 0.005
st = pd.read_csv(state_csv, parse_dates=["day"]).set_index("day")
days = pd.date_range("2022-06-10", "2022-06-19")
wk = pd.DataFrame({"gain_R2_over_R1": daily.reindex(days), "mean_logforecast_R2_minus_R1": (w.R2 - w.R1).groupby("target").mean().reindex(days),
                   "btc_close": st.btc_close.reindex(days)}).rename_axis("day").reset_index()

p = pd.read_parquet(panel); dc = [c for c in p.columns if c in ("date", "day", "utc_date")][0]; p[dc] = pd.to_datetime(p[dc])
z = []
for coin, g in p.groupby("coin"):
    g = g.set_index(dc).sort_index(); ref = g.loc["2021-06-12":"2022-06-11", "pol"]; assert len(ref) == 365
    z.append({"coin": coin, **{d: (g.loc[d, "pol"] - ref.mean()) / ref.std() for d in ("2022-06-11", "2022-06-12", "2022-06-13", "2022-06-14", "2022-06-15", "2022-06-16", "2022-06-17", "2022-06-18")}})
z = pd.DataFrame(z)
out.mkdir(parents=True, exist_ok=True)
sel.reset_index().to_csv(out / "pol_selection.csv", index=False); wk.to_csv(out / "june2022_week.csv", index=False); z.to_csv(out / "june2022_polarity.csv", index=False)
pd.set_option("display.width", 250)
print(sel.round(1).to_string()); print(wk.round(3).to_string(index=False)); print(z.round(1).to_string(index=False))
print("BTC close 10 -> 18 June 2022:", round(100 * (st.btc_close["2022-06-18"] / st.btc_close["2022-06-10"] - 1), 1), "%")
