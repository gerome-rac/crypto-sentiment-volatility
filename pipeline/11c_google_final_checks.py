#!/usr/bin/env python3
"""
11c_google_final_checks.py -- checks and series-level evidence for the final Google Trends series (28 Oct 2026).

Reads only. No model of any kind is estimated. Writes descriptive tables for the task report
data/pipeline/panel/reports/google_final_20261028.md; nothing here changes a series or a rule.

Inputs (from code/):
  data/pipeline/panel/google_daily_final_20261028.parquet (+ .stats.json)   step 11b, draws 01-05
  data/pipeline/panel/google_daily_provisional.parquet                      step 11b, draws 01-02 (7 Oct 2026)
  data/raw/google_trends/draw_01..draw_05/                                  raw windows (read only)
  data/raw/google_trends/related_queries_20261028/                          step 08b (related queries)
  data/pipeline/panel/har_panel.parquet       Wikipedia views, platform counts and RV, only for section 4e
Sections
  1. Agreement (frozen rule: D8 note section 5; MASTERPLAN_NOV15.md section 7, 6 Oct (6)): mean pairwise
     correlation of daily log changes across the five per-draw series of the primary window combination
     ('mean_prop', Deviation 2), per term; a term below 0.8 needs extra draws. Pairwise values are reported.
  2. Non-positive days of the primary series (mean_prop, pooled), per term, with dates.
  3. 'crypto' evidence (D8 section 5: correlation with 'cryptocurrency', divergence in Nov 2021 - Feb 2022,
     rank of 'crypto.com' in Google's related queries; never forecast results): correlations of log levels
     and daily log changes over the sample, inside 2021-11-01..2022-02-28 and outside; the log ratio
     crypto/cryptocurrency by month, inside vs outside, and vs the four months before and after; its one-day
     deviation on Crypto.com event days; the same ratio from the raw weekly 5-year request (2019-07-01 start,
     one request per draw, no temporal disaggregation); ranks in the related-query lists.
  4. Level break around 1 Jan 2022 (diagnostic only; Google Trends notes 'an improvement to our data collection
     system' from 1 Jan 2022; Myburgh 2022, IJERPH 19(22):15396, found a level jump from that date in the Health
     Trends API; docstring citation corrected after the run of 28 Oct 2026, no code line changed): raw daily, weekly and monthly windows spanning New Year 2020-2023; (a) log change of the window
     mean (and median) over 14 days after vs 14 days before, with its percentile among all boundaries of the
     same window; (b) weekly windows, 1 and 4 weeks; (c) monthly Dec -> Jan change; (d) cross-draw noise (SD of
     log values across the five draws for the same window and day), 31 days after vs before; (e) the largest
     one-day deviations common to all seven terms (log value minus its centred 7-day median, mean over windows
     and draws), with Wikipedia views, platform post counts and BTC RV on the same days.
  5. Final (draws 01-05) vs provisional (draws 01-02), primary variant: correlation of log levels and of daily
     log changes per term.
Window lengths (14 days, 1/4 weeks, 31 days, 7-day median) and the event dates are descriptive choices of this
diagnostic, not rules; '<1' is read as 0.5 as in the primary series (Deviation 2).
Usage (from code/, env python):  python pipeline/11c_google_final_checks.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
PANEL = CODE / "data" / "pipeline" / "panel"
REPORTS = PANEL / "reports"
RAW = CODE / "data" / "raw" / "google_trends"
FINAL = PANEL / "google_daily_final_20261028.parquet"
PROV = PANEL / "google_daily_provisional.parquet"
HARP = PANEL / "har_panel.parquet"
RQ_DIR = RAW / "related_queries_20261028"
TAG = "20261028"
OUT_JSON = REPORTS / f"google_final_checks_{TAG}.json"
OUT_CSV = {k: REPORTS / f"google_{k}_{TAG}.csv" for k in
           ["agreement", "agreement_pairs", "crypto_ratio", "level_break", "final_vs_provisional"]}
DRAWS = ["01", "02", "03", "04", "05"]
TERMS = ["Bitcoin", "Ethereum", "Dogecoin", "XRP", "Binance", "cryptocurrency", "crypto"]
COIN_TERM = {"BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "Binance", "XRP": "XRP", "DOGE": "Dogecoin"}
FIRST, LAST = pd.Timestamp("2019-08-01"), pd.Timestamp("2023-01-31")
P0, P1 = pd.Timestamp("2021-11-01"), pd.Timestamp("2022-02-28")
BEFORE = (pd.Timestamp("2021-07-01"), pd.Timestamp("2021-10-31"))
AFTER = (pd.Timestamp("2022-03-01"), pd.Timestamp("2022-06-30"))
EVENTS = {"2021-11-17": "Staples Center renaming to Crypto.com Arena announced",
          "2021-12-25": "Crypto.com Arena name effective", "2022-02-13": "Super Bowl LVI"}
AGREEMENT_THRESHOLD = 0.8
NEW_YEARS = [pd.Timestamp(f"{y}-01-01") for y in (2020, 2021, 2022, 2023)]
FIRST_SUNDAY = {2020: pd.Timestamp("2020-01-05"), 2021: pd.Timestamp("2021-01-03"),
                2022: pd.Timestamp("2022-01-02"), 2023: pd.Timestamp("2023-01-01")}
K_DAYS, NOISE_DAYS = 14, 31
CHECKS: list[dict] = []


def check(cond: bool, what: str) -> None:
    CHECKS.append({"check": what, "passed": bool(cond)})
    if not cond:
        raise AssertionError(what)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def corr(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def read_windows(term: str, freq: str) -> pd.DataFrame:
    out = []
    for d in DRAWS:
        for f in sorted((RAW / f"draw_{d}").glob(f"{term}_{freq}_*.csv")):
            x = pd.read_csv(f, dtype=str)
            x["draw"] = d
            out.append(x)
    x = pd.concat(out, ignore_index=True)
    check(bool((x["keyword"] == term).all() and (x["freq"] == freq).all()), f"raw {term} {freq}: keyword and freq match")
    x["date"] = pd.to_datetime(x["date"])
    lt1 = x["hits"] == "<1"
    x["val"] = np.where(lt1, 0.5, pd.to_numeric(x["hits"].where(~lt1, "0")))
    x["lt1"] = lt1
    x["window"] = x["window_start"] + "_" + x["window_end"]
    return x


def boundary_jumps(g: pd.DataFrame, k: int, stat: str = "mean") -> pd.Series:
    """log(stat of k values from the boundary) - log(stat of k values before), at every boundary of one window."""
    g = g.sort_values("date")
    v, dates = g["val"].to_numpy(float), g["date"].to_numpy()
    f = np.mean if stat == "mean" else np.median
    if stat == "mean":
        res = {dates[i]: np.log(f(v[i:i + k])) - np.log(f(v[i - k:i])) for i in range(k, len(v) - k + 1)}
    else:
        lv = np.log(v)
        res = {dates[i]: f(lv[i:i + k]) - f(lv[i - k:i]) for i in range(k, len(v) - k + 1)}
    return pd.Series(res)


def main() -> int:
    t0 = time.time()
    st = json.load(open(str(FINAL) + ".stats.json"))
    check(st["outputs"][0]["md5"] == md5(FINAL), "final Google table md5 equals its stats file")
    check(st["PROVISIONAL"] is False and st["draws_used"] == DRAWS, "final Google table: draws 01-05, not provisional")
    fin = pd.read_parquet(FINAL)
    prov = pd.read_parquet(PROV)
    fin["day"] = pd.to_datetime(fin["day"]); prov["day"] = pd.to_datetime(prov["day"])
    pf, pp = fin[fin["is_primary"]], prov[prov["is_primary"]]
    check(set(pf["window_combine"]) == {"mean_prop"} and set(pf["draw_set"]) == {"pooled"}, "primary = mean_prop, pooled")
    check(pf.groupby("term").size().eq(1280).all() and pp.groupby("term").size().eq(1280).all(), "primary: 1,280 days per term (final and provisional)")
    res: dict = {}
    rows_lb: list[dict] = []

    # ---- 1. agreement ----
    prs, agr = [], []
    for term, g in fin[(fin["window_combine"] == "mean_prop") & fin["draw_set"].str.startswith("draw_")].groupby("term"):
        w = g.pivot(index="day", columns="draw_set", values="gt_level").sort_index()
        check(list(w.columns) == [f"draw_{d}" for d in DRAWS], f"agreement {term}: five per-draw series")
        dl = np.log(w.where(w > 0)).diff()
        cs = []
        for a, b in itertools.combinations(w.columns, 2):
            c, n = corr(dl[a], dl[b]); cs.append((f"{a[-2:]}-{b[-2:]}", c, n))
            prs.append({"term": term, "pair": f"{a[-2:]}-{b[-2:]}", "corr_dlog": r4(c), "n_obs": n})
        m = float(np.mean([c for _, c, _ in cs]))
        lo, hi = min(cs, key=lambda z: z[1]), max(cs, key=lambda z: z[1])
        agr.append({"term": term, "n_draws": w.shape[1], "n_pairs": len(cs), "mean_pairwise_corr_dlog": r4(m),
                    "min_pair": lo[0], "min_pair_corr": r4(lo[1]), "max_pair": hi[0], "max_pair_corr": r4(hi[1]),
                    "min_obs_per_pair": min(z[2] for z in cs), "below_threshold_0.8": bool(m < AGREEMENT_THRESHOLD)})
    agr = pd.DataFrame(agr)
    st_ag = {a["term"]: a["mean_pairwise_corr_dlog"] for a in st["agreement"] if a["window_combine"] == "mean_prop"}
    check(all(abs(st_ag[t] - v) < 1e-4 for t, v in zip(agr["term"], agr["mean_pairwise_corr_dlog"])), "agreement equals the 11b stats file")
    pst = json.load(open(str(PROV) + ".stats.json"))
    pag = {a["term"]: a["mean_pairwise_corr_dlog"] for a in pst["agreement"] if a["window_combine"] == "mean_prop"}
    agr["provisional_draws_01_02"] = agr["term"].map(pag)
    res["agreement"] = agr.to_dict(orient="records")
    res["agreement_blocked"] = bool(agr["below_threshold_0.8"].any())

    # ---- 2. non-positive days ----
    npd = {}
    for term, g in pf.groupby("term"):
        bad = g[g["gt_level"] <= 0].sort_values("day")
        npd[term] = {"count": int(len(bad)), "days": [str(d.date()) for d in bad["day"]],
                     "values": [r4(v) for v in bad["gt_level"]]}
    st_np = {d["term"]: d["days_nonpositive"] for d in st["days_nonpositive"] if d["window_combine"] == "mean_prop" and d["draw_set"] == "pooled"}
    check(all(st_np[t] == v["count"] for t, v in npd.items()), "non-positive counts equal the 11b stats file")
    res["nonpositive_days_primary"] = npd

    # ---- 3. 'crypto' evidence ----
    wide = pf.pivot(index="day", columns="term", values="gt_level").sort_index()
    c, cc = wide["crypto"], wide["cryptocurrency"]
    check(bool((c > 0).all() and (cc > 0).all()), "crypto and cryptocurrency positive on every day")
    lc, lcc = np.log(c), np.log(cc)
    inp = (wide.index >= P0) & (wide.index <= P1)
    ce = {}
    for nm, (a, b) in {"log_level": (lc, lcc), "dlog": (lc.diff(), lcc.diff())}.items():
        for per, msk in {"sample": np.ones(len(a), bool), "nov2021_feb2022": inp, "outside": ~inp}.items():
            r, n = corr(a[msk], b[msk]); ce[f"corr_{nm}_{per}"] = r4(r); ce[f"n_{nm}_{per}"] = n
    lr = lc - lcc
    bef = (lr.index >= BEFORE[0]) & (lr.index <= BEFORE[1]); aft = (lr.index >= AFTER[0]) & (lr.index <= AFTER[1])
    ce.update({"log_ratio_mean_in_period": r4(lr[inp].mean()), "log_ratio_mean_outside": r4(lr[~inp].mean()),
               "log_ratio_sd_outside": r4(lr[~inp].std()),
               "log_ratio_shift_in_sd": r4((lr[inp].mean() - lr[~inp].mean()) / lr[~inp].std()),
               "log_ratio_mean_jul_oct_2021": r4(lr[bef].mean()), "log_ratio_mean_mar_jun_2022": r4(lr[aft].mean()),
               "log_ratio_mean_2019_08_2020_12": r4(lr[lr.index <= "2020-12-31"].mean()),
               "log_ratio_mean_2022_03_2023_01": r4(lr[lr.index >= "2022-03-01"].mean())})
    ce["log_ratio_by_month"] = {str(k.date())[:7]: r4(v) for k, v in lr.resample("MS").mean().items()}
    dev = lr - lr.rolling(7, center=True).median()
    ce["event_day_log_ratio_deviation_from_7day_median"] = {d: {"event": e, "deviation": r4(dev.loc[d])} for d, e in EVENTS.items()}
    ce["sd_of_daily_deviation_sample"] = r4(dev.std())
    ce["top_ratio_days"] = {str(k.date()): r4(v) for k, v in np.exp(lr).sort_values(ascending=False).head(5).items()}
    # raw weekly, single 5-year request per draw starting 2019-07-01 (one normalisation per term and draw)
    wk = {}
    for term in ["crypto", "cryptocurrency"]:
        x = read_windows(term, "w")
        x = x[x["window_start"] == "2019-07-01"]
        check(x.groupby("draw").size().nunique() == 1 and x["draw"].nunique() == 5, f"weekly 5-year request {term}: one per draw")
        wk[term] = np.log(x.pivot(index="date", columns="draw", values="val"))
    wlr = (wk["crypto"] - wk["cryptocurrency"]).mean(axis=1)
    wlc, wlcc = wk["crypto"].mean(axis=1), wk["cryptocurrency"].mean(axis=1)
    wsel = (wlr.index >= FIRST - pd.Timedelta(days=6)) & (wlr.index <= LAST)
    wlr, wlc, wlcc = wlr[wsel], wlc[wsel], wlcc[wsel]
    winp = (wlr.index >= P0) & (wlr.index <= P1)
    wkev = {"source": "raw weekly request 2019-07-01..2024-07-01 (one per draw), log values averaged over the five draws",
            "n_weeks": int(len(wlr)), "n_weeks_in_period": int(winp.sum())}
    for nm, (a, b) in {"log_level": (wlc, wlcc), "dlog": (wlc.diff(), wlcc.diff())}.items():
        for per, msk in {"sample": np.ones(len(a), bool), "nov2021_feb2022": winp, "outside": ~winp}.items():
            r, n = corr(a[msk], b[msk]); wkev[f"corr_{nm}_{per}"] = r4(r); wkev[f"n_{nm}_{per}"] = n
    wbef = (wlr.index >= BEFORE[0]) & (wlr.index <= BEFORE[1]); waft = (wlr.index >= AFTER[0]) & (wlr.index <= AFTER[1])
    wkev.update({"log_ratio_mean_in_period": r4(wlr[winp].mean()), "log_ratio_mean_outside": r4(wlr[~winp].mean()),
                 "log_ratio_shift_in_sd": r4((wlr[winp].mean() - wlr[~winp].mean()) / wlr[~winp].std()),
                 "log_ratio_mean_jul_oct_2021": r4(wlr[wbef].mean()), "log_ratio_mean_mar_jun_2022": r4(wlr[waft].mean())})
    ce["weekly_raw"] = wkev
    ratio_tab = pd.DataFrame({"day": wide.index, "log_ratio_daily_final": lr.values,
                              "ratio_daily_final_rel_outside": np.exp(lr - lr[~inp].mean()).values, "in_period": inp})
    wk_tab = pd.DataFrame({"day": wlr.index, "log_ratio_weekly_raw": wlr.values,
                           "ratio_weekly_raw_rel_outside": np.exp(wlr - wlr[~winp].mean()).values})
    ratio_tab = ratio_tab.merge(wk_tab, on="day", how="outer").sort_values("day")
    # related queries
    rq = pd.concat([pd.read_csv(f) for f in sorted(RQ_DIR.glob("related_queries_*.csv"))], ignore_index=True)
    check(set(rq["window"]) == {"sample_2019_08_2023_01", "period_2021_11_2022_02", "comparison_2020_11_2021_02"}, "three related-query windows")
    check(bool(rq.groupby(["window", "list"]).size().eq(25).all()), "25 top and 25 rising queries per window")
    q = rq["query"].str.lower()
    is_cdc = q.str.contains("crypto.com", regex=False) | q.str.contains(r"\bcrypto com\b") | q.str.contains(r"\bcro\b") \
        | q.str.contains("cronos", regex=False) | q.str.contains("crypto arena", regex=False)
    brands = ["binance", "coinbase", "ftx", "kraken", "robinhood", "kucoin", "gemini", "bitvavo", "voyager", "pancakeswap",
              "uniswap", "bitfinex", "bybit", "okx", "okex", "huobi", "etoro", "webull", "blockfi", "celsius", "metamask"]
    brand = q.apply(lambda s: next((b for b in brands if b in s.split() or s.startswith(b + " ") or s == b), None))
    rq["crypto_com_related"] = is_cdc
    rq["exchange_or_brand"] = brand
    ce["related_queries_crypto_dot_com_exact"] = rq[q.str.contains("crypto.com", regex=False) | q.str.contains(r"\bcrypto com\b")].to_dict(orient="records")
    ce["related_queries_crypto_com_related"] = rq[is_cdc].to_dict(orient="records")
    ce["related_queries_exchange_or_brand"] = rq[brand.notna()].to_dict(orient="records")
    ce["related_queries_lists"] = {f"{w}|{l}": [f"{r.rank}. {r.query} ({r.google_value})" for r in g.itertuples()]
                                   for (w, l), g in rq.groupby(["window", "list"], sort=False)}
    res["crypto_evidence"] = ce

    # ---- 4. level break around 1 Jan 2022 ----
    rawd = {t: read_windows(t, "d") for t in TERMS}
    raww = {t: read_windows(t, "w") for t in TERMS}
    rawm = {t: read_windows(t, "m") for t in TERMS}
    for t in TERMS:
        for (d, w), g in rawd[t].groupby(["draw", "window"]):
            jm, jmed = None, None
            for b in NEW_YEARS:
                if not (g["date"].min() <= b - pd.Timedelta(days=K_DAYS) and g["date"].max() >= b + pd.Timedelta(days=K_DAYS - 1)):
                    continue
                if jm is None:
                    jm, jmed = boundary_jumps(g, K_DAYS, "mean"), boundary_jumps(g, K_DAYS, "median")
                for nm, j in (("daily_14d_mean", jm), ("daily_14d_median", jmed)):
                    rows_lb.append({"stat": nm, "term": t, "boundary": str(b.date()), "draw": d, "window": w,
                                    "value": float(j[b]), "pctile_in_window": float((j < j[b]).mean()), "n_boundaries": len(j)})
        for (d, w), g in raww[t].groupby(["draw", "window"]):
            for k in (1, 4):
                j = boundary_jumps(g, k, "mean")
                for yr, b in FIRST_SUNDAY.items():
                    if b in j.index:
                        rows_lb.append({"stat": f"weekly_{k}w_mean", "term": t, "boundary": str(b.date()), "draw": d, "window": w,
                                        "value": float(j[b]), "pctile_in_window": float((j < j[b]).mean()), "n_boundaries": len(j)})
        for (d, w), g in rawm[t].groupby(["draw", "window"]):
            g = g[(g["date"] >= "2019-07-01") & (g["date"] <= "2023-01-01")].sort_values("date")
            dl = pd.Series(np.log(g["val"].to_numpy(float)), index=g["date"].values).diff().dropna()
            for b in NEW_YEARS:
                if b in dl.index:
                    rows_lb.append({"stat": "monthly_dec_to_jan", "term": t, "boundary": str(b.date()), "draw": d, "window": w,
                                    "value": float(dl[b]), "pctile_in_window": float((dl < dl[b]).mean()), "n_boundaries": len(dl)})
        x = rawd[t].assign(lv=np.log(rawd[t]["val"]))
        for w, g in x.groupby("window"):
            p = g.pivot_table(index="date", columns="draw", values="lv")
            if p.shape[1] < 5:
                continue
            sd = p.std(axis=1)
            for b in NEW_YEARS:
                if not (sd.index.min() <= b - pd.Timedelta(days=NOISE_DAYS) and sd.index.max() >= b + pd.Timedelta(days=NOISE_DAYS - 1)):
                    continue
                pre = sd[(sd.index >= b - pd.Timedelta(days=NOISE_DAYS)) & (sd.index < b)].mean()
                post = sd[(sd.index >= b) & (sd.index < b + pd.Timedelta(days=NOISE_DAYS))].mean()
                rows_lb.append({"stat": "noise_sd_ratio_post_pre_31d", "term": t, "boundary": str(b.date()), "draw": "all", "window": w,
                                "value": float(post / pre) if pre > 0 else np.nan, "pctile_in_window": np.nan, "n_boundaries": np.nan,
                                "sd_pre": float(pre), "sd_post": float(post)})
    lb = pd.DataFrame(rows_lb)
    summ = (lb.groupby(["stat", "term", "boundary"])
              .agg(value_mean=("value", "mean"), value_min=("value", "min"), value_max=("value", "max"),
                   pctile_mean=("pctile_in_window", "mean"), n=("value", "size")).reset_index())
    res["level_break_summary"] = summ.round(4).to_dict(orient="records")
    # (e) common one-day deviations
    devs = {}
    for t in TERMS:
        parts = []
        for (d, w), g in rawd[t].groupby(["draw", "window"]):
            g = g.sort_values("date")
            lv = pd.Series(np.log(g["val"].to_numpy(float)), index=g["date"].values)
            parts.append(lv - lv.rolling(7, center=True).median())
        devs[t] = pd.concat(parts, axis=1).mean(axis=1)
    SP = pd.DataFrame(devs)
    SP = SP[(SP.index >= FIRST) & (SP.index <= LAST)]
    SP["min_over_terms"] = SP[TERMS].min(axis=1)
    hp = pd.read_parquet(HARP)
    hp["day"] = pd.to_datetime(hp["day"])
    def dev7(s: pd.Series) -> pd.Series:
        l = np.log(s.astype(float).where(s > 0))
        return l - l.rolling(7, center=True).median()
    wiki = pd.DataFrame({c: dev7(hp[hp["coin"] == c].set_index("day")["att_wiki"]) for c in COIN_TERM})
    btc = hp[hp["coin"] == "BTC"].set_index("day")
    other = pd.DataFrame({"wiki_cryptocurrency": dev7(btc["mkt_wiki"]), "reddit_market": dev7(btc["mkt_reddit"]),
                          "stocktwits_market": dev7(btc["mkt_stocktwits"]), "twitter_market": dev7(btc["mkt_twitter"]),
                          "btc_rv": dev7(btc["rv"])})
    top = SP.sort_values("min_over_terms", ascending=False).head(5)
    res["common_one_day_deviations_top5"] = [
        {"day": str(d.date()), "google_min_over_terms": r4(r["min_over_terms"]),
         "google_by_term": {t: r4(r[t]) for t in TERMS},
         "wikipedia_coin_articles": {c: r4(wiki.loc[d, c]) for c in COIN_TERM},
         **{k: r4(other.loc[d, k]) for k in other.columns}} for d, r in top.iterrows()]
    res["common_one_day_deviation_rank_of_2022_01_04"] = int((SP["min_over_terms"] > SP.loc["2022-01-04", "min_over_terms"]).sum()) + 1

    # ---- 5. final vs provisional ----
    fvp = []
    for term in TERMS:
        a = pf[pf["term"] == term].set_index("day")["gt_level"].sort_index()
        b = pp[pp["term"] == term].set_index("day")["gt_level"].sort_index()
        check(a.index.equals(b.index), f"final and provisional {term}: same days")
        la, lb_ = np.log(a.where(a > 0)), np.log(b.where(b > 0))
        r1, n1 = corr(la, lb_); r2, n2 = corr(la.diff(), lb_.diff())
        fvp.append({"term": term, "corr_log_level": r4(r1), "n_log_level": n1, "corr_dlog": r4(r2), "n_dlog": n2,
                    "max_abs_diff_log_level": r4((la - lb_).abs().max())})
    res["final_vs_provisional"] = fvp

    # ---- write ----
    REPORTS.mkdir(parents=True, exist_ok=True)
    tables = {"agreement": agr, "agreement_pairs": pd.DataFrame(prs), "crypto_ratio": ratio_tab,
              "level_break": summ.round(6), "final_vs_provisional": pd.DataFrame(fvp)}
    outs = []
    for k, df in tables.items():
        p = OUT_CSV[k]
        check(not p.exists(), f"new output name {p.name}")
        tmp = p.with_name(p.name + ".tmp")
        df.to_csv(tmp, index=False)
        check(len(pd.read_csv(tmp)) == len(df), f"{p.name} re-read")
        os.replace(tmp, p)
        outs.append({"path": str(p.relative_to(CODE)), "md5": md5(p), "rows": int(len(df))})
    check(not OUT_JSON.exists(), f"new output name {OUT_JSON.name}")
    tmp = OUT_JSON.with_name(OUT_JSON.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=str))
    check(json.loads(tmp.read_text())["agreement_blocked"] == res["agreement_blocked"], f"{OUT_JSON.name} re-read")
    os.replace(tmp, OUT_JSON)
    outs.append({"path": str(OUT_JSON.relative_to(CODE)), "md5": md5(OUT_JSON)})
    rq_files = sorted(p for p in RQ_DIR.iterdir() if p.is_file())
    stats = {"step": "11c_google_final_checks", "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "python": sys.version.split()[0], "code_md5": md5(Path(__file__)),
             "inputs": {"google_daily_final_20261028.parquet": md5(FINAL), "google_daily_provisional.parquet": md5(PROV),
                        "har_panel.parquet": md5(HARP),
                        "raw_draws": {f"draw_{d}": st["inputs"][f"draw_{d}"] for d in DRAWS},
                        "related_queries_20261028": {p.name: md5(p) for p in rq_files}},
             "parameters": {"AGREEMENT_THRESHOLD": AGREEMENT_THRESHOLD, "PERIOD": [str(P0.date()), str(P1.date())],
                            "BEFORE": [str(x.date()) for x in BEFORE], "AFTER": [str(x.date()) for x in AFTER],
                            "K_DAYS": K_DAYS, "NOISE_DAYS": NOISE_DAYS, "EVENTS": EVENTS},
             "outputs": outs, "checks": CHECKS, "seconds": round(time.time() - t0, 1)}
    sp = Path(str(OUT_JSON) + ".stats.json")
    tmp = sp.with_name(sp.name + ".tmp"); tmp.write_text(json.dumps(stats, indent=1, default=str)); os.replace(tmp, sp)
    print(json.dumps({"agreement_blocked": res["agreement_blocked"], "outputs": len(outs), "seconds": stats["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
