#!/usr/bin/env python3
"""
11_attention.py -- attention block (R1): daily ingredients per series and UTC day.

Rules (frozen 7 Oct 2026, planning/pre_analysis_note_D8.md; MASTERPLAN_NOV15.md section 7,
'Decisions of 6 Oct 2026' points (2), (4), (5); 'Decisions of 3 Oct 2026' points (4), (6);
D3, D14, D15, D16; register arms A56, A57, A66):

  * Platform attention (Audrino, Sigrist & Ballinari 2020): log(1 + posts) per coin-day, separately
    for Reddit, StockTwits and Twitter; posts = n_posts of data/pipeline/panel/daily_counts.parquet
    (after the bot pass). Series BTC, ETH, BNB, XRP, DOGE and MARKET (market level).
  * Twitter partial days are coverage-scaled: count / captured share of the UTC day (own rule,
    flagged; = old A19). Captured share = minutes_utc / 1440 from
    data/pipeline/twitter/units/twitter_coverage_utc.parquet (parameter TWITTER_COVERAGE_COL).
  * The 76 UTC days without any Twitter capture (minutes_utc = 0) are MISSING here (NaN) and flagged
    twitter_absent. Their imputation with the current-window average is an in-window step of the
    estimation engine and is not done here.
  * Robustness A56: within-day share per platform, coin posts / all posts of the platform's
    crypto-defined panel that day (series CRYPTO_SET), in logs (not logits); raw Twitter counts
    (the capture rate cancels). Zero coin-days: parameter SHARE_ZERO_RULE.
  * Robustness A57 (market level): log(1 + Reddit CRYPTO_SET + StockTwits CRYPTO_SET posts).
  * Wikipedia pageviews (en.wikipedia, all-access, agent = user): log level. BNB = article 'Binance'
    (the exchange; flagged wiki_bnb_is_exchange), XRP = 'Ripple (payment protocol)', MARKET =
    'Cryptocurrency'.
  * RavenPack news volume: log(1 + distinct headlines per coin and UTC day), relevance >= 75;
    robustness A66: log(1 + entity rows). UTC day = date of timestamp_utc (asserted equal to
    rpa_date_utc). Distinct = exact headline string (parameter HEADLINE_KEY). A coin-day without
    any RavenPack row has 0 headlines (flagged news_zero_day). MARKET rows carry no news volume.

Not done here (assembly / estimation engine): HAR averages, day-t log changes, z-scoring, the
window-average imputation of twitter_absent days. Google Trends: 11b_google_combine.py.

Output   data/pipeline/panel/attention_daily.parquet (+ .stats.json): one row per series x UTC day
         (6 x 1,280 = 7,680 rows).
Usage (from code/):  python pipeline/11_attention.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
DATA = CODE / "data"
IN_COUNTS = DATA / "pipeline" / "panel" / "daily_counts.parquet"
IN_COVERAGE = DATA / "pipeline" / "twitter" / "units" / "twitter_coverage_utc.parquet"
IN_WIKI = DATA / "control_var" / "wikipedia_pageviews_user_20261006.csv"
IN_RAVENPACK = DATA / "control_var" / "wrds_ravenpack_crypto_2019_2023.csv"
OUT = DATA / "pipeline" / "panel" / "attention_daily.parquet"

FIRST_DAY, LAST_DAY, N_DAYS = date(2019, 8, 1), date(2023, 1, 31), 1280
COINS = ["BTC", "ETH", "BNB", "XRP", "DOGE"]
SERIES = COINS + ["MARKET"]
PLATFORMS = ["reddit", "stocktwits", "twitter"]
N_ZERO_CAPTURE_EXPECTED = 76

# ---- named parameters (construction details the frozen documents do not settle; see report) ----
TWITTER_COVERAGE_COL = "minutes_utc"   # lower bound of captured minutes; 'minutes_utc_hi' = upper bound
SHARE_ZERO_RULE = "nan"                # log(share) on a coin-day with 0 posts: 'nan' (literal log 0 undefined)
HEADLINE_KEY = "exact"                 # distinct headlines = exact headline string (md5 of UTF-8 bytes)
RELEVANCE_MIN = 75
DUCKDB_MEMORY, DUCKDB_THREADS = "2GB", 2

WIKI_ARTICLES = {"BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "Binance", "XRP": "Ripple_(payment_protocol)",
                 "DOGE": "Dogecoin", "MARKET": "Cryptocurrency"}
RP_ENTITIES = {"BTC": "A25816", "ETH": "0A2CF4", "DOGE": "2070E1", "XRP": "547BE3", "BNB": "A2BF5C"}

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


def calendar() -> pd.DatetimeIndex:
    cal = pd.date_range(FIRST_DAY, LAST_DAY, freq="D")
    check(len(cal) == N_DAYS, f"calendar has {N_DAYS} UTC days")
    return cal


def platform_counts(cal) -> pd.DataFrame:
    d = pd.read_parquet(IN_COUNTS)
    d["day"] = pd.to_datetime(d["day"])
    check(d["day"].min() == cal[0] and d["day"].max() == cal[-1] and d["day"].nunique() == N_DAYS,
          "daily_counts covers the 1,280-day calendar")
    keep = d[d["series"].isin(SERIES + ["CRYPTO_SET"])]
    wide = keep.pivot_table(index=["series", "day"], columns="platform", values="n_posts", aggfunc="sum")
    check(not wide.isna().any().any(), "complete platform x series x day grid in daily_counts")
    wide = wide.astype("int64")
    tot = wide.xs("CRYPTO_SET", level="series")
    out = wide.drop(index="CRYPTO_SET", level="series").reset_index()
    out = out.rename(columns={p: f"{p}_n_posts" for p in PLATFORMS})
    for p in PLATFORMS:
        out = out.merge(tot[[p]].rename(columns={p: f"{p}_n_crypto_set"}), left_on="day", right_index=True)
    # totals against the source
    for p in PLATFORMS:
        for s in SERIES:
            src = int(d.loc[(d.platform == p) & (d.series == s), "n_posts"].sum())
            check(int(out.loc[out.series == s, f"{p}_n_posts"].sum()) == src, f"{p} {s}: posts add up to daily_counts ({src:,})")
    return out


def twitter_coverage(cal) -> pd.DataFrame:
    cv = pd.read_parquet(IN_COVERAGE)
    cv["day"] = pd.to_datetime(cv["utc_day"])
    check(len(cv) == N_DAYS and cv["day"].min() == cal[0] and cv["day"].max() == cal[-1],
          "twitter_coverage_utc covers the 1,280-day calendar")
    check(bool((cv["minutes_utc"].between(0, 1440)).all()), "minutes_utc in [0, 1440]")
    n0 = int((cv[TWITTER_COVERAGE_COL] == 0).sum())
    check(n0 == N_ZERO_CAPTURE_EXPECTED, f"zero-capture Twitter days = {N_ZERO_CAPTURE_EXPECTED} (found {n0})")
    return pd.DataFrame({"day": cv["day"], "twitter_minutes_utc": cv[TWITTER_COVERAGE_COL].astype("int64"),
                         "twitter_coverage_share": cv[TWITTER_COVERAGE_COL] / 1440.0,
                         "twitter_coverage_pct_utc": cv["coverage_pct_utc"]})


def wikipedia(cal) -> pd.DataFrame:
    w = pd.read_csv(IN_WIKI)
    check(set(w["agent"]) == {"user"} and set(w["access"]) == {"all-access"} and set(w["project"]) == {"en.wikipedia"},
          "Wikipedia file: en.wikipedia, all-access, agent = user")
    rows = []
    for s, art in WIKI_ARTICLES.items():
        x = w[(w["series"] == s) & (w["article"] == art)].copy()
        x["day"] = pd.to_datetime(x["date"])
        check(len(x) == N_DAYS and x["day"].nunique() == N_DAYS and x["day"].min() == cal[0] and x["day"].max() == cal[-1],
              f"Wikipedia {s} ('{art}'): 1,280 days")
        check(bool(x["views"].notna().all() and (x["views"] > 0).all()), f"Wikipedia {s}: no missing or zero views")
        rows.append(pd.DataFrame({"series": s, "day": x["day"], "wiki_views": x["views"].astype("int64")}))
    return pd.concat(rows, ignore_index=True)


def ravenpack(cal) -> tuple[pd.DataFrame, dict]:
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "attention_duckdb_spill"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'; SET threads={DUCKDB_THREADS}; SET temp_directory='{tmp}'")
    tz = con.execute("SELECT current_setting('TimeZone')").fetchone()[0]
    check(tz == "UTC", "DuckDB TimeZone = UTC")
    src = f"""read_csv('{IN_RAVENPACK}', header=true, all_varchar=true)"""
    con.execute(f"""CREATE TEMP TABLE rp AS
        SELECT rp_entity_id, rpa_date_utc, CAST(relevance AS INTEGER) AS relevance, headline,
               CAST(CAST(timestamp_utc || '+00' AS TIMESTAMPTZ) AS DATE) AS day_utc
        FROM {src}""")
    n_all, n_mis, n_null_head, rel_min = con.execute(
        "SELECT count(*), count_if(CAST(rpa_date_utc AS DATE) <> day_utc), count_if(headline IS NULL OR headline = ''), "
        "min(relevance) FROM rp").fetchone()
    check(n_mis == 0, "RavenPack: UTC date of timestamp_utc equals rpa_date_utc on every row")
    check(n_null_head == 0, "RavenPack: no NULL or empty headline")
    ents = "', '".join(RP_ENTITIES.values())
    q = f"""SELECT rp_entity_id, day_utc, count(*) AS n_rows,
                   count(DISTINCT md5(headline)) AS n_distinct, count(DISTINCT headline) AS n_distinct_str
            FROM rp WHERE rp_entity_id IN ('{ents}') AND relevance >= {RELEVANCE_MIN}
            GROUP BY 1, 2"""
    g = con.execute(q).df()
    check(bool((g["n_distinct"] == g["n_distinct_str"]).all()), "RavenPack: md5 keys and headline strings give the same distinct count")
    n_coin_rows, n_coin_rows_rel, n_below = con.execute(
        f"SELECT count(*), count_if(relevance >= {RELEVANCE_MIN}), count_if(relevance < {RELEVANCE_MIN}) FROM rp "
        f"WHERE rp_entity_id IN ('{ents}')").fetchone()
    per_ent = con.execute(f"""SELECT rp_entity_id, count(*) AS rows, count(DISTINCT headline) AS uniq,
             count(DISTINCT day_utc) AS days, min(day_utc) AS first, max(day_utc) AS last
             FROM rp WHERE rp_entity_id IN ('{ents}') AND relevance >= {RELEVANCE_MIN} GROUP BY 1""").df()
    con.close()
    inv = {v: k for k, v in RP_ENTITIES.items()}
    g["series"] = g["rp_entity_id"].map(inv)
    g["day"] = pd.to_datetime(g["day_utc"])
    check(g["day"].min() >= cal[0] and g["day"].max() <= cal[-1], "RavenPack: every coin row inside the 1,280-day window")
    grid = pd.MultiIndex.from_product([COINS, cal], names=["series", "day"]).to_frame(index=False)
    out = grid.merge(g[["series", "day", "n_rows", "n_distinct"]], on=["series", "day"], how="left")
    out["news_zero_day"] = out["n_rows"].isna()
    out["news_rows"] = out["n_rows"].fillna(0).astype("int64")
    out["news_headlines_distinct"] = out["n_distinct"].fillna(0).astype("int64")
    check(int(out["news_rows"].sum()) == n_coin_rows_rel, f"RavenPack: daily rows add up to the CSV ({n_coin_rows_rel:,})")
    per_ent["series"] = per_ent["rp_entity_id"].map(inv)
    info = {"rows_all_entities": int(n_all), "rows_five_coins": int(n_coin_rows), "rows_five_coins_relevance_ok": int(n_coin_rows_rel),
            "rows_five_coins_below_relevance": int(n_below), "min_relevance_all_rows": int(rel_min),
            "per_entity": per_ent.astype({"first": str, "last": str}).to_dict(orient="records")}
    return out[["series", "day", "news_rows", "news_headlines_distinct", "news_zero_day"]], info


def main() -> int:
    t0 = time.time()
    cal = calendar()
    pc = platform_counts(cal)
    cv = twitter_coverage(cal)
    df = pc.merge(cv, on="day", how="left")
    check(len(df) == len(SERIES) * N_DAYS, "6 series x 1,280 days")

    # Twitter: missing on zero-capture days, coverage-scaled otherwise
    df["twitter_absent"] = df["twitter_minutes_utc"] == 0
    df["twitter_full_capture"] = df["twitter_coverage_pct_utc"] >= 95.0   # D3/D15 'fully observed' (>= 95%)
    check(bool((df.loc[df.twitter_absent, "twitter_n_posts"] == 0).all()), "Twitter: 0 posts on every zero-capture day")
    check(bool((df.loc[df.twitter_absent, "twitter_n_crypto_set"] == 0).all()), "Twitter CRYPTO_SET: 0 posts on every zero-capture day")
    zero_captured = df[(~df.twitter_absent) & (df.twitter_n_posts == 0)]
    check(len(zero_captured) == 0, "Twitter: no captured coin-day with 0 posts")
    for c in ["twitter_n_posts", "twitter_n_crypto_set"]:
        df[c] = df[c].astype("float64")
        df.loc[df.twitter_absent, c] = np.nan
    df["twitter_n_posts_scaled"] = df["twitter_n_posts"] / df["twitter_coverage_share"]

    # primary attention levels (Audrino): log(1 + posts)
    df["att_reddit"] = np.log1p(df["reddit_n_posts"])
    df["att_stocktwits"] = np.log1p(df["stocktwits_n_posts"])
    df["att_twitter"] = np.log1p(df["twitter_n_posts_scaled"])
    check(int(df["att_twitter"].isna().sum()) == N_ZERO_CAPTURE_EXPECTED * len(SERIES),
          "att_twitter missing exactly on the 76 zero-capture days (x 6 series)")

    # A56: log within-day share (raw counts, capture cancels)
    for p in PLATFORMS:
        share = df[f"{p}_n_posts"] / df[f"{p}_n_crypto_set"]
        df[f"share_{p}"] = share
        if SHARE_ZERO_RULE == "nan":
            df[f"logshare_{p}"] = np.where(share > 0, np.log(share.where(share > 0)), np.nan)
        else:
            raise ValueError(SHARE_ZERO_RULE)
        df[f"share_zero_{p}"] = df[f"{p}_n_posts"] == 0
        check(bool((share.dropna() <= 1).all()), f"{p}: share <= 1")

    # A57: market level, total Reddit + StockTwits crypto volume (MARKET rows only)
    df["att_rs_total_crypto"] = np.where(df["series"] == "MARKET",
                                         np.log1p(df["reddit_n_crypto_set"] + df["stocktwits_n_crypto_set"]), np.nan)

    wk = wikipedia(cal)
    df = df.merge(wk, on=["series", "day"], how="left")
    df["att_wiki"] = np.log(df["wiki_views"])
    df["wiki_bnb_is_exchange"] = df["series"] == "BNB"

    rp, rp_info = ravenpack(cal)
    df = df.merge(rp, on=["series", "day"], how="left")
    df["news_zero_day"] = df["news_zero_day"].astype("boolean")
    df["att_news"] = np.log1p(df["news_headlines_distinct"])
    df["att_news_rows"] = np.log1p(df["news_rows"])
    check(bool(df.loc[df.series != "MARKET", "att_news"].notna().all()), "news volume set on every coin-day")
    check(bool(df.loc[df.series == "MARKET", "att_news"].isna().all()), "no news volume on MARKET rows")

    for c in ["reddit_n_posts", "stocktwits_n_posts", "reddit_n_crypto_set", "stocktwits_n_crypto_set"]:
        df[c] = df[c].astype("int64")
    for c in ["news_rows", "news_headlines_distinct"]:
        df[c] = df[c].astype("Int64")
    df["day"] = df["day"].dt.date
    order = ["series", "day",
             "att_reddit", "att_stocktwits", "att_twitter", "att_wiki", "att_news",
             "twitter_absent", "twitter_full_capture", "twitter_coverage_share", "twitter_minutes_utc", "twitter_coverage_pct_utc",
             "reddit_n_posts", "stocktwits_n_posts", "twitter_n_posts", "twitter_n_posts_scaled",
             "reddit_n_crypto_set", "stocktwits_n_crypto_set", "twitter_n_crypto_set",
             "share_reddit", "share_stocktwits", "share_twitter", "logshare_reddit", "logshare_stocktwits", "logshare_twitter",
             "share_zero_reddit", "share_zero_stocktwits", "share_zero_twitter",
             "att_rs_total_crypto", "wiki_views", "wiki_bnb_is_exchange",
             "news_headlines_distinct", "news_rows", "news_zero_day", "att_news_rows"]
    check(set(order) == set(df.columns), "column list complete")
    df = df[order].sort_values(["series", "day"]).reset_index(drop=True)

    # completeness: every coin-day complete except the flagged Twitter days
    core = ["att_reddit", "att_stocktwits", "att_wiki"]
    check(bool(df[core].notna().all().all()), "Reddit, StockTwits and Wikipedia attention complete on 7,680 rows")
    check(bool(df.loc[~df.twitter_absent, "att_twitter"].notna().all()), "att_twitter complete on captured days")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    back = pd.read_parquet(tmp)
    check(len(back) == len(df) == len(SERIES) * N_DAYS, "written table re-read: 7,680 rows")
    os.replace(tmp, OUT)

    summary = {
        "n_share_zero": {p: {s: int(df.loc[df.series == s, f"share_zero_{p}"].sum()) for s in SERIES} for p in PLATFORMS},
        "n_news_zero_days": {s: int(df.loc[df.series == s, "news_zero_day"].sum()) for s in COINS},
        "twitter_coverage_positive_min_minutes": int(df.loc[~df.twitter_absent, "twitter_minutes_utc"].min()),
        "twitter_partial_days_lt95": int(((~df.twitter_absent) & (~df.twitter_full_capture)).sum() // len(SERIES)),
        "twitter_full_days_ge95": int(df["twitter_full_capture"].sum() // len(SERIES)),
        "ravenpack": rp_info,
    }
    stats = {"step": "11_attention", "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "python": sys.version.split()[0], "duckdb_version": duckdb.__version__,
             "code_md5": md5(Path(__file__)),
             "parameters": {"TWITTER_COVERAGE_COL": TWITTER_COVERAGE_COL, "SHARE_ZERO_RULE": SHARE_ZERO_RULE,
                            "HEADLINE_KEY": HEADLINE_KEY, "RELEVANCE_MIN": RELEVANCE_MIN,
                            "DUCKDB_MEMORY": DUCKDB_MEMORY, "DUCKDB_THREADS": DUCKDB_THREADS},
             "inputs": [{"path": str(p.relative_to(CODE)), "bytes": p.stat().st_size, "md5": md5(p)}
                        for p in [IN_COUNTS, IN_COVERAGE, IN_WIKI, IN_RAVENPACK]],
             "outputs": [{"path": str(OUT.relative_to(CODE)), "bytes": OUT.stat().st_size, "md5": md5(OUT), "rows": len(df)}],
             "summary": summary, "checks": CHECKS, "seconds": round(time.time() - t0, 1)}
    st = Path(str(OUT) + ".stats.json")
    tmpj = Path(str(st) + ".tmp")
    tmpj.write_text(json.dumps(stats, indent=1, default=str))
    os.replace(tmpj, st)
    print(json.dumps({"rows": len(df), "md5": stats["outputs"][0]["md5"], "n_checks": len(CHECKS), "seconds": stats["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
