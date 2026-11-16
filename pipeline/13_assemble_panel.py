#!/usr/bin/env python3
"""
13_assemble_panel.py -- step 13: merge the four daily blocks into the estimation panel.

Inputs  (read-only, all under data/pipeline/panel/):
    volatility_daily.parquet          step 09 (main C and J = library A1, Deviation 1 of the D8 note)
    attention_daily.parquet           step 11 (raw daily counts, Wikipedia views, RavenPack headlines)
    google_daily_provisional.parquet  step 11b (Google Trends; PROVISIONAL until draws 2-5 and the
                                      open combination point are settled; GOOGLE_VARIANT below)
    text_daily.parquet                step 10 (variant 'main', platform 'POOLED' = D15 index)
    controls_daily.parquet            step 12 (day-t transformed controls)
Output  data/pipeline/panel/har_panel.parquet: one row per coin and UTC day (5 x 1,280), raw daily
        ingredients under the estimation engine's names (code/model/variable_spec_A00.csv), market
        variables repeated on every coin row; plus har_panel.parquet.stats.json.

Nothing data-driven is computed here (no averages, logs, centring, z-scores or imputation): the
engine does every in-window step. Twitter counts are left missing on the 76 zero-capture days
(flag twitter_absent; the engine imputes them with the current-window average, 6 Oct (4)).

    python pipeline/13_assemble_panel.py            (from code/)
Options (28 Oct 2026; defaults unchanged): --google <parquet> (step-11b table to use; default
google_daily_provisional.parquet), --out <parquet> (default har_panel.parquet; an explicit --out never
overwrites an existing file), --reference <parquet> (assert that every non-Google column equals that panel
exactly and that only att_google, mkt_google_* and google_provisional differ). google_provisional is taken
from the Google table's 'provisional' column (False only for the five-draw series).
    final panel: --google data/pipeline/panel/google_daily_final_20261028.parquet
                 --out data/pipeline/panel/har_panel_final_20261028.parquet --reference data/pipeline/panel/har_panel.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "..", "data", "pipeline", "panel")
OUT = os.path.join(PANEL, "har_panel.parquet")
COINS = ["BTC", "ETH", "BNB", "XRP", "DOGE"]
FIRST, LAST = pd.Timestamp("2019-08-01"), pd.Timestamp("2023-01-31")
GOOGLE_VARIANT = {"window_combine": "mean_prop", "draw_set": "pooled"}   # Deviation 2: proportional Denton-Cholette
GOOGLE_COIN_TERM = {"BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "Binance", "XRP": "XRP", "DOGE": "Dogecoin"}
GOOGLE_MKT_TERM = {"mkt_google_cryptocurrency": "cryptocurrency", "mkt_google_crypto": "crypto"}
EMO = ["happy", "sad", "anger", "disgust", "surprise", "fear"]
GOOGLE_COLS = ["att_google", *GOOGLE_MKT_TERM, "google_provisional"]   # the only columns that may differ from --reference
CTRL = {"ctrl_vix": "log_vix", "ctrl_move": "log_move", "ctrl_term": "term_spread", "ctrl_baa10y": "baa10y_lag1",   # lagged one day: posted the next day (clarification of 7 Oct 2026, D8 note section 7)
       
        "ctrl_dgs10_chg": "d_dgs10", "ctrl_dxy_ret": "ret_dxy", "ctrl_gold_ret": "ret_gold", "ctrl_epu": "log_epu_lag1",
        "ctrl_tether_news": "tether_log1p_headlines", "ctrl_tether_news_chg": "tether_dlog_headlines", "ctrl_weekend": "weekend"}


def md5(p: str) -> str:
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def day(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s.astype(str) if s.dtype == object else s)
    if getattr(s.dt, "tz", None) is not None:
        assert str(s.dt.tz) == "UTC"
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    assert (s == s.dt.normalize()).all()
    return s.dt.normalize().astype("datetime64[ns]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--google", default=os.path.join(PANEL, "google_daily_provisional.parquet"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--reference", default=None)
    cli = ap.parse_args()
    out = os.path.abspath(cli.out)
    checks = []
    def check(cond, msg):
        checks.append({"check": msg, "passed": bool(cond)})
        if not cond:
            raise AssertionError(msg)
    if out != os.path.abspath(OUT):
        check(not os.path.exists(out) and not os.path.exists(out + ".stats.json"), f"an explicit --out never overwrites an existing file ({os.path.basename(out)})")
    gkey = os.path.basename(cli.google)
    inp = {f: os.path.join(PANEL, f) for f in ["volatility_daily.parquet", "attention_daily.parquet",
           "text_daily.parquet", "text_precision_daily.parquet", "controls_daily.parquet"]}
    inp = {**dict(list(inp.items())[:2]), gkey: os.path.abspath(cli.google), **dict(list(inp.items())[2:])}
    days = pd.date_range(FIRST, LAST, freq="D")
    grid = pd.MultiIndex.from_product([COINS, days], names=["coin", "day"]).to_frame(index=False)

    v = pd.read_parquet(inp["volatility_daily.parquet"]); v["day"] = day(v["day_utc"])
    vol = v[["coin", "day", "rv", "c", "j", "jump_flag", "outage_flag", "abd_drop_flag", "doge_2019_20"]]

    a = pd.read_parquet(inp["attention_daily.parquet"]); a["day"] = day(a["day"])
    a.loc[a["twitter_absent"], "twitter_n_posts_scaled"] = np.nan
    coin_att = a[a["series"].isin(COINS)].rename(columns={"series": "coin"})
    coin_att = coin_att[["coin", "day", "reddit_n_posts", "stocktwits_n_posts", "twitter_n_posts_scaled", "wiki_views",
                         "news_headlines_distinct", "twitter_absent", "twitter_full_capture", "wiki_bnb_is_exchange", "news_zero_day"]]
    coin_att = coin_att.rename(columns={"reddit_n_posts": "att_reddit", "stocktwits_n_posts": "att_stocktwits",
                                        "twitter_n_posts_scaled": "att_twitter", "wiki_views": "att_wiki",
                                        "news_headlines_distinct": "att_news"})
    mk = a[a["series"] == "MARKET"][["day", "reddit_n_posts", "stocktwits_n_posts", "twitter_n_posts_scaled", "wiki_views"]]
    mk = mk.rename(columns={"reddit_n_posts": "mkt_reddit", "stocktwits_n_posts": "mkt_stocktwits",
                            "twitter_n_posts_scaled": "mkt_twitter", "wiki_views": "mkt_wiki"})

    g = pd.read_parquet(inp[gkey]); g["day"] = day(g["day"])
    g = g[(g["window_combine"] == GOOGLE_VARIANT["window_combine"]) & (g["draw_set"] == GOOGLE_VARIANT["draw_set"])]
    check(g["provisional"].nunique() == 1, "Google: one 'provisional' value for the primary series")
    gprov = bool(g["provisional"].iloc[0])
    check(g.groupby("term").size().eq(len(days)).all(), "Google: one value per term and day")
    gc = pd.concat([g[g["term"] == t][["day", "gt_level"]].assign(coin=c) for c, t in GOOGLE_COIN_TERM.items()])
    gc = gc.rename(columns={"gt_level": "att_google"})
    for col, t in GOOGLE_MKT_TERM.items():
        mk = mk.merge(g[g["term"] == t][["day", "gt_level"]].rename(columns={"gt_level": col}), on="day", how="left")

    # D16 precision-weighted daily means (step 10b; clarification of 7 Oct 2026): pooled means of the six
    # classes and polarity come from text_precision_daily.parquet; the unweighted pooled means stay as *_raw.
    tp = pd.read_parquet(inp["text_precision_daily.parquet"]); tp["day"] = day(tp["day"])
    tp = tp[tp["series"].isin(COINS)].rename(columns={"series": "coin"})
    t = pd.read_parquet(inp["text_daily.parquet"])
    t = t[(t["variant"] == "main") & (t["platform"] == "POOLED") & t["series"].isin(COINS)].copy()
    t["day"] = day(t["day"])
    ren = {"series": "coin", "mean_polarity": "pol", "sd_polarity": "pol_sd"}
    ren.update({f"mean_{k}": f"emo_{k}" for k in EMO}); ren.update({f"sd_{k}": f"emo_{k}_sd" for k in EMO})
    txt = t.rename(columns=ren)[["coin", "day", "pol", "pol_sd"] + [f"emo_{k}" for k in EMO] + [f"emo_{k}_sd" for k in EMO]
                                + ["n_posts", "n_platforms_emo"]].rename(columns={"n_posts": "text_n_posts"})
    raw_cols = {"pol": "pol_raw", **{f"emo_{k_}": f"emo_{k_}_raw" for k_ in EMO}}
    txt = txt.rename(columns=raw_cols)
    pwm = tp[["coin", "day", "mean_polarity_pw"] + [f"mean_{k_}_pw" for k_ in EMO]].rename(
        columns={"mean_polarity_pw": "pol", **{f"mean_{k_}_pw": f"emo_{k_}" for k_ in EMO}})
    txt = txt.merge(pwm, on=["coin", "day"], how="left")

    k = pd.read_parquet(inp["controls_daily.parquet"]); k["day"] = day(k["day"])
    ctl = k[["day"] + list(CTRL.values())].rename(columns={v_: k_ for k_, v_ in CTRL.items()})

    df = (grid.merge(vol, on=["coin", "day"], how="left").merge(coin_att, on=["coin", "day"], how="left")
              .merge(gc, on=["coin", "day"], how="left").merge(txt, on=["coin", "day"], how="left")
              .merge(mk, on="day", how="left").merge(ctl, on="day", how="left"))
    # Deviation 3 of the D8 note (7 Oct 2026): the weekend dummy refers to the forecast day t+1 (known in
    # advance, no look-ahead); the origin-day dummy of controls_daily is kept as ctrl_weekend_origin.
    df["ctrl_weekend_origin"] = df["ctrl_weekend"].astype(float)
    df["ctrl_weekend"] = ((df["day"] + pd.Timedelta(days=1)).dt.dayofweek >= 5).astype(float)
    df["google_provisional"] = gprov

    # ---- checks ----
    check(len(df) == 6400 and not df.duplicated(["coin", "day"]).any(), "6,400 unique coin-days")
    check(df[["rv", "c", "j"]].notna().all().all() and (df["rv"] > 0).all(), "rv, c, j complete, rv > 0")
    check(np.allclose(df["c"] + df["j"], df["rv"]), "c + j = rv")
    check(df["att_twitter"].isna().eq(df["twitter_absent"]).all(), "att_twitter missing exactly on twitter_absent days")
    check(int(df.loc[df.coin == "BTC", "twitter_absent"].sum()) == 76, "76 zero-capture Twitter days")
    for c_ in ["att_reddit", "att_stocktwits", "att_wiki", "att_news", "mkt_reddit", "mkt_stocktwits", "mkt_wiki"]:
        check(df[c_].notna().all() and (df[c_] >= 0).all(), f"{c_} complete and non-negative")
    check(df["pol"].notna().all() and df[[f"emo_{k_}" for k_ in EMO]].notna().all().all(), "pooled polarity and emotion means complete")
    fw = df["day"] >= pd.Timestamp("2019-08-30")
    check(df.loc[fw, list(CTRL)].notna().all().all(), "controls complete from the first forecast origin (2019-08-30)")
    for c_ in [c for c in df.columns if c.startswith(("mkt_", "ctrl_"))]:
        check((df.pivot(index="day", columns="coin", values=c_).nunique(axis=1, dropna=False) <= 1).all(), f"{c_} equal across coins")
    na = {c_: int(df[c_].isna().sum()) for c_ in df.columns if df[c_].isna().any()}
    nonpos_google = {c_: int((df[c_] <= 0).sum()) for c_ in ["att_google", "mkt_google_cryptocurrency", "mkt_google_crypto"]}

    ref_diff = None
    if cli.reference:
        ref = pd.read_parquet(cli.reference)
        check(list(ref.columns) == list(df.columns), "reference: same columns in the same order")
        check(len(ref) == len(df) and ref["coin"].astype(str).tolist() == df["coin"].astype(str).tolist()
              and pd.to_datetime(ref["day"]).tolist() == df["day"].tolist(), "reference: same rows and keys (coin, day) in the same order")
        same = {c_: bool(df[c_].equals(ref[c_])) for c_ in df.columns}
        check(all(same[c_] for c_ in df.columns if c_ not in GOOGLE_COLS), "reference: every non-Google column equal (values, missing positions and dtype)")
        check({c_ for c_, s_ in same.items() if not s_} <= set(GOOGLE_COLS), "reference: only att_google, mkt_google_* and google_provisional differ")
        ref_diff = {"reference_md5": md5(cli.reference), "columns_differing": sorted(c_ for c_, s_ in same.items() if not s_),
                    "rows_differing": {c_: int((~((df[c_] == ref[c_]) | (df[c_].isna() & ref[c_].isna()))).sum()) for c_ in GOOGLE_COLS}}
    tmp = out + ".tmp"; df.to_parquet(tmp, index=False)
    back = pd.read_parquet(tmp)
    check(list(back.columns) == list(df.columns) and len(back) == len(df) and all(back[c_].equals(df[c_]) for c_ in df.columns), "written panel re-read equals the assembled frame")
    os.replace(tmp, out)
    stats = {"step": "13_assemble_panel", "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "output": os.path.relpath(out, os.path.join(HERE, "..")) if out.startswith(os.path.abspath(os.path.join(HERE, ".."))) else out,
             "output_md5": md5(out), "rows": int(len(df)), "columns": list(df.columns),
             "inputs": {f: md5(p) for f, p in inp.items()}, "script_md5": md5(os.path.abspath(__file__)),
             "google_variant": GOOGLE_VARIANT, "google_input": gkey, "google_provisional": gprov, "missing_values": na,
             "google_nonpositive_coin_days": nonpos_google, "reference_comparison": ref_diff, "checks": checks}
    tj = out + ".stats.json.tmp"; json.dump(stats, open(tj, "w"), indent=1, default=str); os.replace(tj, out + ".stats.json")
    print(f"wrote {out} ({len(df)} rows, md5 {stats['output_md5']}); missing: {na}; Google <= 0: {nonpos_google}; google_provisional: {gprov}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
