#!/usr/bin/env python3
"""emotion_discussion.py -- descriptive numbers for the emotions paragraph of the Discussion (14 Nov 2026).

1. leadlag.csv        within-coin correlation of the one-day value of each emotion and of polarity with the log RV of
                      the previous, the same and the next day (6,250 coin-days, coin means removed). The next-day
                      column must reproduce Table "How far the text measures are spanned" (asserted for fear, anger).
2. dropped.csv        for the main run and the elastic-net run: windows in which R2 keeps at least one attention term
                      and R3 keeps none, the share of the loss of R3 relative to R2 that arises on their forecast days,
                      and the R2 -> R3 step (loss change, Clark-West p).
3. market_state.csv   correlation of each measure with the market state log(BTC close / 200-day average), per coin,
                      on the days with a defined state (the numbers quoted in the Discussion; in no table).

4. sep2021_emotions.csv  each measure on 6, 7 and 8 September 2021, per coin, in standard deviations from its mean of
                      the 365 days up to 5 September 2021 (6 September is the last day the forecast for 7 September uses).

Usage (from the code/ folder): python <this file> <panel.parquet> <runs_dir> <main_run> <enet_run> <state_csv> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from model.tests_eval import loss_frame  # noqa: E402

panel, runs, main_run, enet_run, state_csv, out = sys.argv[1:7]
runs, out = Path(runs), Path(out)
E = ["emo_fear", "emo_anger", "emo_sad", "emo_disgust", "emo_surprise", "emo_happy", "pol"]

p = pd.read_parquet(panel)
dc = [c for c in p.columns if c in ("date", "day", "utc_date")][0]
p[dc] = pd.to_datetime(p[dc]); p = p.sort_values(["coin", dc])
p["lrv"] = np.log(p["rv"]); g = p.groupby("coin")
p["lrv_next"], p["lrv_prev"] = g["lrv"].shift(-1), g["lrv"].shift(1)
d = p[(p[dc] >= "2019-08-30") & (p[dc] <= "2023-01-30")].copy()
assert len(d) == 6250 and (d.groupby("coin").size() == 1250).all()


def wc(a, b):
    x = d[[a, b, "coin"]].dropna()
    return float(np.corrcoef(x[a] - x.groupby("coin")[a].transform("mean"), x[b] - x.groupby("coin")[b].transform("mean"))[0, 1])


ll = pd.DataFrame([{"measure": e, "prev_day": wc(e, "lrv_prev"), "same_day": wc(e, "lrv"), "next_day": wc(e, "lrv_next")} for e in E])
assert abs(ll.set_index("measure").next_day["emo_fear"] - 0.313) < 5e-4 and abs(ll.set_index("measure").next_day["emo_anger"] - 0.274) < 5e-4

rows = []
for name, run in (("adaptive LASSO (main)", main_run), ("adaptive elastic net", enet_run)):
    r = runs / run
    f = pd.read_parquet(r / "forecasts.parquet"); f["target"] = pd.to_datetime(f["target"])
    L = {k: loss_frame(f[f.rung == k]).groupby("target")[["se", "qlike"]].mean() for k in ("R2", "R3")}
    d32 = L["R3"] - L["R2"]
    c2, c3 = pd.read_parquet(r / "coefs_R2.parquet"), pd.read_parquet(r / "coefs_R3.parquet")
    for c in (c2, c3): c["origin"] = pd.to_datetime(c["origin"])
    att = [x for x in c3.columns if x.startswith(("att_", "mkt_"))]
    emo = [x for x in c3.columns if x.startswith("emo_")]
    c2, c3 = c2.set_index("origin").sort_index(), c3.set_index("origin").sort_index()
    assert len(c2) == 885 == len(c3) and (c2.index == c3.index).all()
    drop = (c2[att] != 0).any(axis=1) & ~(c3[att] != 0).any(axis=1)
    noemo = ~(c3[emo] != 0).any(axis=1)
    tgt = lambda m: m.index[m.to_numpy()] + pd.Timedelta(days=1)
    nt = pd.read_csv(r / "nested_tests.csv").set_index("test")
    rows.append({"run": name, "windows_R2_attention_R3_none": int(drop.sum()), "windows_no_emotion_term": int(noemo.sum()),
                 "loss_share_se_dropped": d32.se.reindex(tgt(drop)).sum() / d32.se.sum(),
                 "loss_share_qlike_dropped": d32.qlike.reindex(tgt(drop)).sum() / d32.qlike.sum(),
                 "loss_share_se_no_emotion": d32.se.reindex(tgt(noemo)).sum() / d32.se.sum(),
                 "loss_share_qlike_no_emotion": d32.qlike.reindex(tgt(noemo)).sum() / d32.qlike.sum(),
                 "mse_change_pct": 100 * (L["R3"].se.mean() / L["R2"].se.mean() - 1),
                 "qlike_change_pct": 100 * (L["R3"].qlike.mean() / L["R2"].qlike.mean() - 1), "cw_p": nt.loc["H3", "cw_p"]})
dr = pd.DataFrame(rows)

st = pd.read_csv(state_csv); st["day"] = pd.to_datetime(st["day"])
m = d.merge(st[["day", "state"]], left_on=dc, right_on="day", how="left"); m = m[m.state.notna()]
ms = pd.DataFrame({c: {e: np.corrcoef(gg[e], gg["state"])[0, 1] for e in E} for c, gg in m.groupby("coin")}).rename_axis("measure").reset_index()

ev = []
for coin, gg in p.groupby("coin"):
    gg = gg.set_index(dc).sort_index(); ref = gg.loc["2020-09-06":"2021-09-05"]
    assert len(ref) == 365
    for day in ("2021-09-06", "2021-09-07", "2021-09-08"):
        ev.append({"coin": coin, "day": day, **{e: (gg.loc[day, e] - ref[e].mean()) / ref[e].std() for e in E}})
ev = pd.DataFrame(ev)

out.mkdir(parents=True, exist_ok=True)
ll.to_csv(out / "leadlag.csv", index=False); dr.to_csv(out / "dropped.csv", index=False); ms.to_csv(out / "market_state.csv", index=False)
ev.to_csv(out / "sep2021_emotions.csv", index=False)
print(ev.round(1).to_string(index=False))
pd.set_option("display.width", 250)
print(ll.round(3).to_string(index=False)); print(dr.round(3).T.to_string()); print(ms.round(2).to_string(index=False), "| days with a state per coin:", int(m.groupby("coin").size().iloc[0]))
