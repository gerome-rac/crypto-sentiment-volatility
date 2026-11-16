#!/usr/bin/env python3
"""
10_text_series.py -- step 10: daily text series (EmTract emotions, FinBERT polarity) per coin,
platform and UTC day, the D15 pooled index, and the pre-forecast-year calibrations (D16
reliability, Twitter hour post-stratification profile).

Written 7 Oct 2026 (after the D8 freeze of 06:53 UTC) from the frozen rules; it changes no
earlier step and no earlier output.

Inputs (all local, read only)
-----------------------------
    data/pipeline/<platform>/panel/assignment.parquet              step 04: series flags, bot flags (D21)
    data/pipeline/<platform>/emtract/emotion/*.parquet             step 06 merge: p_*, pc_*, flags
    data/pipeline/<platform>/polarity/finbert-L64/polarity/*.parquet  step 07 merge: polarity, polarity_c
    data/pipeline/twitter/units/twitter_coverage_utc.parquet       captured minutes per UTC day (D14)
    data/pipeline/panel/daily_counts.parquet                       step 05, for the count cross-check
    (the fixed EmTract empty-text score is read from twitter/emtract/emotion/units.parquet.stats.json, A21)

Outputs
-------
    data/pipeline/panel/text_daily.parquet          long in variant x series x platform x day, wide in variables
    data/pipeline/panel/text_calibration_preforecast.json   fixed calibration parameters (+ full-sample values for reporting)
    (each with the usual .stats.json next to the parquet)

Rules applied (MASTERPLAN_NOV15.md section 7 and planning/pre_analysis_note_D8.md section 2)
-------------------------------------------------------------------------------------------
* Posts: every post of the series after the bot pass (D21, 26 Sep: no unit rule, no account or
  subreddit weights); a post counts for each series it names (PJ3); Twitter retweet-only posts on
  their first-seen day as stored by step 02 (D13). Day = UTC calendar day of `ts`, derived here in a
  session with TimeZone UTC (asserted) and checked against the stored `day` (PF1).
* Text: the author's own text (`version = 'own'`), primary score `p_*` (EmTract truncation, decision
  (4) of 29 Sep) and FinBERT `polarity` = P(pos) - P(neg) (D19).
* Texts empty after EmTract's cleaning (incl. clean errors) and missing texts count as posts but are
  left out of both the EmTract and the FinBERT means, so that R2 and R3 average over the same posts
  (decision (2) of 3 Oct). A post enters the means only if both models scored it.
* Daily mean and within-day SD (sample SD, missing below 2 posts) per platform, series and day for
  the six emotions, neutral (checks only), polarity and EmTract valence (A68).
* Twitter: a day with minutes_utc = 0 is missing (twitter_absent), never 0 (D14); partial days enter
  through the hour post-stratified mean (D3/D16, 6 Oct (2)); profile fixed on the pre-forecast year.
* Pooled index (D15): equal-weighted mean of the available platform means; pooled SD = equal-weighted
  mean of the available platform SDs (6 Oct (3)); counts of platforms used are stored.
* D16: one-way random-effects decomposition of post scores on day per platform x series x variable on
  the pre-forecast year 30 Aug 2019 - 29 Aug 2020 -> sigma2_b, sigma2_w, n* for R(n) >= 0.7; R(n),
  the precision weight 1/(sigma2_w/n + sigma2_b) and thin_<var> are stored per platform row.
* NOT built here (estimated inside each rolling window by the engine): centred squares, HAR
  averages, z-scores, the A10 composition adjustment, imputation of Twitter zero-capture days.

Open construction details are single named parameters below (OPEN_*); see the task report.

Usage (from code/)
------------------
    python pipeline/10_text_series.py            # full run
    python pipeline/10_text_series.py --reuse-posts   # reuse the post-level scratch tables
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, runner  # noqa: E402
from pipeline.runner import check, log  # noqa: E402

STEP = "10_text_series"
CODE_SHA = runner.code_fingerprint(__file__)
RULES_SHA = "MASTERPLAN-e9dc7a487c6a98d25738b7c88b436f7f"   # frozen MASTERPLAN md5 (D8 note section 6)

SERIES = ["BTC", "ETH", "BNB", "XRP", "DOGE", "MARKET"]
COINS = SERIES[:5]
PLATFORMS = ["reddit", "stocktwits", "twitter"]
EMO = ["neutral", "happy", "sad", "anger", "disgust", "surprise", "fear"]
EMO6 = EMO[1:]
VARS = EMO + ["valence", "polarity"]          # stored mean_/sd_ columns
EMO_VARS = EMO + ["valence"]                  # share the emotion sample (n_emo)

PREFORECAST = ("2019-08-30", "2020-08-29")    # calibration year (6 Oct 2026 (1))
RELIABILITY_TARGET = 0.7                      # D16
FULL_DAY_PCT = 95.0                           # 'fully observed' Twitter day (D3/D15, 28 Sep 2026)

# --- literal readings of points the frozen documents leave open (task report, open points) ---
OPEN_STRATIFY_DAYS = "partial_lt95"   # stratify Twitter days with 0 < coverage < 95%; alternative 'all_captured'
OPEN_PROFILE_SCOPE = "corpus"         # one Twitter hour profile over all series (HAR_VARIANTS T4 'corpus-average');
                                      # alternative 'series' (per-series profiles are written to the json, unused)
OPEN_PROFILE_DAYS = "full_ge95"       # profile from fully observed days of the calibration year
OPEN_STRATIFIED_SD = False            # within-day SD is unstratified on every day
OPEN_POOLED_RELIABILITY = None        # no reliability / weight / thin flag on pooled rows
OPEN_A21_POLARITY = "main"            # A21 changes the emotion means only; polarity keeps the main sample
OPEN_A48_POLARITY = "main"            # A48/A49: FinBERT has no quoted/combined score; polarity keeps the main sample

OUT = config.OUT_DIR / "panel" / "text_daily.parquet"
CAL = config.OUT_DIR / "panel" / "text_calibration_preforecast.json"
SCRATCH = config.SCRATCH_DIR / "text_series"


def src(platform: str, kind: str) -> List[str]:
    if kind == "assign":
        return [str(config.out_path(platform, "panel", "assignment.parquet"))]
    if kind == "emo":
        return sorted(glob.glob(str(config.out_path(platform, "emtract", "emotion", "*.parquet"))))
    if kind == "pol":
        return sorted(glob.glob(str(config.out_path(platform, "polarity", "finbert-L64", "polarity", "*.parquet"))))
    raise ValueError(kind)


def rp(files: List[str]) -> str:
    return "read_parquet([" + ", ".join(f"'{f}'" for f in files) + "])"


# ---------------------------------------------------------------------------
# 1. Post-level tables (scratch database, outside the repository)
# ---------------------------------------------------------------------------

EXTRA_COLS = {
    "reddit": ["kind", "edited", "rbinance_support", "bnb_strict", "bsc_context"] + [f"textgate_{c.lower()}" for c in COINS],
    "stocktwits": ["equity_only", "bnb_strict", "bsc_context"],
    "twitter": ["authored", "bnb_strict", "bsc_context"],
}
HAS_QUOTES = {"reddit": True, "stocktwits": False, "twitter": True}


def build_posts(con, platform: str, checks: List[Dict]) -> Dict[str, int]:
    a, e, f = rp(src(platform, "assign")), rp(src(platform, "emo")), rp(src(platform, "pol"))
    sc = ", ".join(f"a.in_{s.lower()}" for s in SERIES)
    extra = ", ".join(f"a.{c}" for c in EXTRA_COLS[platform])
    pcols = ", ".join(f"e.p_{x}, coalesce(e.pc_{x}, e.p_{x}) AS pc_{x}" for x in EMO)
    qsel, qjoin = "", ""
    if HAS_QUOTES[platform]:
        qsel = (", q.row_id IS NOT NULL AS has_quote, "
                + ", ".join(f"q.p_{x} AS q_{x}" for x in EMO) + ", "
                + ", ".join(f"b.p_{x} AS b_{x}" for x in EMO))
        qjoin = (f" LEFT JOIN (SELECT * FROM {e} WHERE version = 'quoted') q ON q.row_id = a.post_id"
                 f" LEFT JOIN (SELECT * FROM {e} WHERE version = 'both') b ON b.row_id = a.post_id"
                 )
    con.execute(f"DROP TABLE IF EXISTS posts_{platform}")
    # Reddit's score tables are partitioned by the assignment's coin_label: join partition by partition
    # (memory); StockTwits and Twitter are joined whole.
    if platform == "reddit":
        parts = [(Path(x).stem, f"a.coin_label = '{Path(x).stem}'", rp([x]),
                  rp([str(config.out_path(platform, "polarity", "finbert-L64", "polarity", Path(x).name))]))
                 for x in src(platform, "emo")]
    else:
        parts = [("all", "TRUE", e, f)]
    for i, (name, cond, e_p, f_p) in enumerate(parts):
        qsel_p, qjoin_p = qsel, qjoin.replace(e, e_p).replace(f, f_p) if qjoin else ""
        sql = f"""SELECT a.post_id, a.ts, CAST(a.ts AS DATE) AS day, CAST(hour(a.ts) AS TINYINT) AS hr, a.day AS day_stored,
               {sc}, a.bot_any, a.text_key, a.n_coins, {extra},
               e.row_id IS NOT NULL AS has_emo_row, coalesce(e.is_empty_after_clean, FALSE) AS is_empty,
               coalesce(e.is_missing_text, FALSE) AS is_missing, {pcols},
               f.polarity AS pol, coalesce(f.polarity_c, f.polarity) AS pol_c{qsel_p}
        FROM {a} a
        LEFT JOIN (SELECT * FROM {e_p} WHERE version = 'own') e ON e.row_id = a.post_id
        LEFT JOIN (SELECT row_id, polarity, polarity_c FROM {f_p} WHERE version = 'own') f ON f.row_id = a.post_id{qjoin_p}
        WHERE {cond} AND CAST(a.ts AS DATE) BETWEEN DATE '{config.WINDOW_START}' AND DATE '{config.WINDOW_END}'"""
        con.execute((f"CREATE TABLE posts_{platform} AS " if i == 0 else f"INSERT INTO posts_{platform} ") + sql)
        log(f"{platform}: post table part {name}")
    t = f"posts_{platform}"
    n, n_ids, n_dmis, n_noemo, n_emo_nopol, n_pol_noemo = con.execute(f"""SELECT count(*), count(DISTINCT post_id),
        count_if(day <> day_stored), count_if(NOT has_emo_row), count_if(p_happy IS NOT NULL AND pol IS NULL),
        count_if(p_happy IS NULL AND pol IS NOT NULL) FROM {t}""").fetchone()
    n_assign = con.execute(f"SELECT count(*) FROM {a}").fetchone()[0]
    check(n == n_assign, f"{platform}: every assignment row is in the window and kept ({n:,})", checks)
    check(n_ids == n, f"{platform}: one row per post", checks)
    check(n_dmis == 0, f"{platform}: UTC day from ts equals the stored day on every post", checks)
    check(n_noemo == 0, f"{platform}: every post has its own-text EmTract row", checks)
    nv = con.execute(f"SELECT count_if(version <> 'own') FROM {f}").fetchone()[0]
    check(nv == 0, f"{platform}: FinBERT scored the own text only (A48/A49 change the emotion means only)", checks)
    return {"posts": int(n), "emo_scored_no_polarity": int(n_emo_nopol), "polarity_scored_no_emo": int(n_pol_noemo)}


def prob_sum_check(con, platform: str, checks: List[Dict]) -> Dict[str, float]:
    e = rp(src(platform, "emo"))
    s_p = " + ".join(f"p_{x}" for x in EMO)
    s_c = " + ".join(f"pc_{x}" for x in EMO)
    mp, mc, n = con.execute(f"""SELECT max(abs(({s_p}) - 1)), max(abs(({s_c}) - 1)), count(p_happy)
                                FROM {e}""").fetchone()
    check(mp is not None and mp <= 1e-5, f"{platform}: EmTract p_* sum to 1 within 1e-5 on all {n:,} scored inputs (max {mp:.2e})", checks)
    check(mc is None or mc <= 1e-5, f"{platform}: EmTract pc_* sum to 1 within 1e-5 (max {mc if mc is None else f'{mc:.2e}'})", checks)
    f = rp(src(platform, "pol"))
    mf, mpol = con.execute(f"""SELECT max(abs(p_positive + p_negative + p_neutral - 1)),
                                      max(abs(polarity - (p_positive - p_negative))) FROM {f}""").fetchone()
    check(mf <= 1e-5, f"{platform}: FinBERT p_* sum to 1 within 1e-5 (max {mf:.2e})", checks)
    check(mpol <= 1e-5, f"{platform}: polarity = p_positive - p_negative within 1e-5 (max {mpol:.2e})", checks)
    return {"emtract_p_max_abs_dev": float(mp), "emtract_pc_max_abs_dev": None if mc is None else float(mc),
            "finbert_p_max_abs_dev": float(mf), "finbert_polarity_max_abs_dev": float(mpol), "n_scored_inputs": int(n)}


# ---------------------------------------------------------------------------
# 2. Variants (main + the robustness arms that follow mechanically from the same posts)
# ---------------------------------------------------------------------------

def _ok(prefix: str, pol: str) -> str:
    return f"({prefix}happy IS NOT NULL AND {pol} IS NOT NULL)"


MAIN_OK = _ok("p_", "pol")


def scores(variant: str, platform: str, empty_score: Dict[str, float]) -> Dict[str, str]:
    """SQL expression per variable: the post's score, NULL when the post is not in that mean."""
    out: Dict[str, str] = {}
    if variant == "a13_chunk":
        for x in EMO:
            out[x] = f"CASE WHEN {MAIN_OK} THEN pc_{x} END"
        out["polarity"] = f"CASE WHEN {MAIN_OK} THEN pol_c END"
    elif variant in ("a48_quoted", "a49_both") and HAS_QUOTES[platform]:
        # FinBERT scored the own text only (D19 settings), so A48/A49 change the emotion means only;
        # polarity keeps the main sample (OPEN_A48_POLARITY)
        k = "q_" if variant == "a48_quoted" else "b_"
        for x in EMO:
            out[x] = f"CASE WHEN has_quote THEN {k}{x} WHEN {MAIN_OK} THEN p_{x} END"
        out["polarity"] = f"CASE WHEN {MAIN_OK} THEN pol END"
    elif variant == "a21_with_empty":
        for x in EMO:
            out[x] = f"CASE WHEN {MAIN_OK} THEN p_{x} WHEN is_empty AND NOT is_missing THEN {empty_score[x]!r} END"
        out["polarity"] = f"CASE WHEN {MAIN_OK} THEN pol END"
    else:
        for x in EMO:
            out[x] = f"CASE WHEN {MAIN_OK} THEN p_{x} END"
        out["polarity"] = f"CASE WHEN {MAIN_OK} THEN pol END"
    out["valence"] = (f"({out['happy']}) - ({out['sad']}) - ({out['anger']}) - ({out['disgust']}) - ({out['fear']})")
    return out


# name -> (platforms, series, extra filter, membership override, bot filter on, dedup)
VARIANTS = {
    "main":               (PLATFORMS, SERIES, None, None, True, False),
    "a43_with_bots":      (PLATFORMS, SERIES, None, None, False, False),
    "a13_chunk":          (PLATFORMS, SERIES, None, None, True, False),
    "a21_with_empty":     (PLATFORMS, SERIES, None, None, True, False),
    "a44_dedup_text_day": (PLATFORMS, SERIES, None, None, True, True),
    "a28_single_ticker":  (PLATFORMS, COINS, "n_coins = 1", None, True, False),
    "a29_bnb_strict":     (PLATFORMS, ["BNB"], "bnb_strict", None, True, False),
    "a39_no_bsc_context": (PLATFORMS, ["BNB"], "NOT bsc_context", None, True, False),
    "a38_no_equity_only": (["stocktwits"], SERIES, "NOT equity_only", None, True, False),
    "a46_authored_only":  (["twitter"], SERIES, "authored", None, True, False),
    "a48_quoted":         (["reddit", "twitter"], SERIES, None, None, True, False),
    "a49_both":           (["reddit", "twitter"], SERIES, None, None, True, False),
    "a23_textgate":       (["reddit"], COINS, None, "textgate", True, False),
    "a26_posts_only":     (["reddit"], SERIES, "kind = 'posts'", None, True, False),
    "a45_no_edited":      (["reddit"], SERIES, "NOT edited", None, True, False),
    "a41_market_support": (["reddit"], ["MARKET"], None, "market_support", True, False),
}
DERIVED_VARIANTS = {"a20_unstratified": "main"}   # Twitter raw (unstratified) mean, no extra pass


def member(series: str, override) -> str:
    s = series.lower()
    if override == "textgate" and series in COINS:
        return f"textgate_{s}"
    if override == "market_support" and series == "MARKET":
        return "(in_market OR rbinance_support)"
    return f"in_{s}"


def hourly_sql(platform: str, variant: str, empty_score) -> str:
    plats, ser, filt, override, bots, dedup = VARIANTS[variant]
    sc = scores(variant, platform, empty_score)
    where = []
    if bots:
        where.append("NOT bot_any")
    if filt:
        where.append(filt)
    w = (" AND " + " AND ".join(where)) if where else ""
    parts = [f"SELECT '{s}' AS series, * FROM posts_{platform} WHERE {member(s, override)}{w}" for s in ser]
    base = " UNION ALL ".join(parts)
    if dedup:   # A44: exact-duplicate texts (text_key, md5 of the text) once per series and UTC day; first post kept
        base = (f"SELECT * FROM ({base}) QUALIFY text_key IS NULL OR "
                f"row_number() OVER (PARTITION BY series, day, text_key ORDER BY ts, post_id) = 1")
    aggs = [f"CAST(count(*) AS BIGINT) AS n_posts", "CAST(count_if(is_empty) AS BIGINT) AS n_empty",
            "CAST(count_if(is_missing) AS BIGINT) AS n_missing",
            f"CAST(count({sc['happy']}) AS BIGINT) AS n_emo", f"CAST(count({sc['polarity']}) AS BIGINT) AS n_pol"]
    for x in VARS:
        aggs.append(f"sum(CAST({sc[x]} AS DOUBLE)) AS s_{x}")
        aggs.append(f"sum(CAST({sc[x]} AS DOUBLE) * CAST({sc[x]} AS DOUBLE)) AS q_{x}")
    return (f"SELECT '{variant}' AS variant, '{platform}' AS platform, series, day, hr, {', '.join(aggs)} "
            f"FROM ({base}) GROUP BY series, day, hr")


# ---------------------------------------------------------------------------
# 3. Twitter hour profile (calibration year, fully observed days)
# ---------------------------------------------------------------------------

def hour_profiles(con, lo: str, hi: str) -> Dict[str, object]:
    cov = f"read_parquet('{config.out_path('twitter', 'units', 'twitter_coverage_utc.parquet')}')"
    anyser = " OR ".join(f"in_{s.lower()}" for s in SERIES)
    days = f"(SELECT utc_day FROM {cov} WHERE coverage_pct_utc >= {FULL_DAY_PCT} AND utc_day BETWEEN DATE '{lo}' AND DATE '{hi}')"
    rows = con.execute(f"""SELECT hr, count(*) FROM posts_twitter
        WHERE ({anyser}) AND NOT bot_any AND {MAIN_OK} AND day IN {days} GROUP BY hr ORDER BY hr""").fetchall()
    n = sum(r[1] for r in rows)
    corpus = {int(h): c / n for h, c in rows}
    check(len(corpus) == 24, f"hour profile {lo}..{hi}: all 24 hours present")
    per_series = {}
    for s in SERIES:
        rs = con.execute(f"""SELECT hr, count(*) FROM posts_twitter WHERE in_{s.lower()} AND NOT bot_any AND {MAIN_OK}
            AND day IN {days} GROUP BY hr ORDER BY hr""").fetchall()
        ns = sum(r[1] for r in rs)
        per_series[s] = {int(h): c / ns for h, c in rs}
    nd = con.execute(f"SELECT count(*) FROM {days}").fetchone()[0]
    return {"days_used": int(nd), "posts_used": int(n), "corpus": corpus, "per_series": per_series}


# ---------------------------------------------------------------------------
# 4. Daily aggregation from the hourly sums
# ---------------------------------------------------------------------------

def daily_sql(profile: Dict[int, float]) -> str:
    prof = "(VALUES " + ", ".join(f"({h}, {w!r})" for h, w in sorted(profile.items())) + ") pr(hr, pi)"
    cols = ["sum(n_posts) AS n_posts", "sum(n_empty) AS n_empty", "sum(n_missing) AS n_missing",
            "sum(n_emo) AS n_emo", "sum(n_pol) AS n_pol",
            "CAST(count_if(n_emo > 0) AS INTEGER) AS n_hours_emo"]
    for x in VARS:
        n = "n_pol" if x == "polarity" else "n_emo"
        cols.append(f"sum(s_{x}) / nullif(sum({n}), 0) AS raw_mean_{x}")
        cols.append(f"CASE WHEN sum({n}) >= 2 THEN sqrt(greatest((sum(q_{x}) - sum(s_{x}) * sum(s_{x}) / sum({n}))"
                    f" / (sum({n}) - 1), 0)) END AS sd_{x}")
        cols.append(f"sum(CASE WHEN {n} > 0 THEN pi * s_{x} / {n} END) / nullif(sum(CASE WHEN {n} > 0 THEN pi END), 0)"
                    f" AS strat_mean_{x}")
    return (f"SELECT variant, platform, series, day, {', '.join(cols)} FROM hourly h JOIN {prof} USING (hr) "
            f"GROUP BY variant, platform, series, day")


# ---------------------------------------------------------------------------
# 5. D16 variance decomposition (one-way random effects, ANOVA estimator)
# ---------------------------------------------------------------------------

def decompose(g: pd.DataFrame, x: str) -> Dict[str, float]:
    n_col = "n_pol" if x == "polarity" else "n_emo"
    d = g[g[n_col] > 0]
    n = d[n_col].astype(float).values
    m = d[f"raw_mean_{x}"].values
    sd = d[f"sd_{x}"].fillna(0.0).values
    N, D = n.sum(), len(n)
    if D < 2 or N - D <= 0:
        return {"days": int(D), "posts": float(N), "sigma2_w": None, "sigma2_b": None, "sigma2_b_raw": None, "n_star": None}
    grand = (n * m).sum() / N
    msw = ((n - 1) * sd ** 2).sum() / (N - D)
    msb = (n * (m - grand) ** 2).sum() / (D - 1)
    n0 = (N - (n ** 2).sum() / N) / (D - 1)
    s2b_raw = (msb - msw) / n0
    s2b = max(s2b_raw, 0.0)
    R = RELIABILITY_TARGET
    nstar = (R / (1 - R)) * msw / s2b if s2b > 0 else math.inf
    return {"days": int(D), "posts": float(N), "grand_mean": float(grand), "msw": float(msw), "msb": float(msb),
            "n0": float(n0), "sigma2_w": float(msw), "sigma2_b": float(s2b), "sigma2_b_raw": float(s2b_raw),
            "n_star": float(nstar), "n_star_ceil": (int(math.ceil(nstar)) if math.isfinite(nstar) else None),
            "median_n": float(np.median(n)), "median_R": float(np.median(s2b / (s2b + msw / n))) if s2b > 0 else 0.0}


def calibrate(daily: pd.DataFrame, lo: str, hi: str) -> Dict[str, Dict[str, Dict[str, dict]]]:
    d = daily[(daily.variant == "main") & (daily.day >= pd.Timestamp(lo)) & (daily.day <= pd.Timestamp(hi))]
    d = d[~((d.platform == "twitter") & d.twitter_absent)]
    out: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for (p, s), g in d.groupby(["platform", "series"]):
        out.setdefault(p, {})[s] = {x: decompose(g, x) for x in VARS}
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(reuse_posts: bool) -> int:
    t0 = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    db = SCRATCH / "text_series.duckdb"
    if not reuse_posts and db.exists():
        db.unlink()
    con = runner.connect(memory="4GB", threads=4, database=db)
    check(con.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC", "session TimeZone is UTC")
    checks: List[Dict] = []
    counts: Dict[str, object] = {}

    inputs_files = [config.out_path("twitter", "units", "twitter_coverage_utc.parquet"),
                    config.OUT_DIR / "panel" / "daily_counts.parquet"]
    for p in PLATFORMS:
        inputs_files += [Path(x) for x in src(p, "assign") + src(p, "emo") + src(p, "pol")]
    log("md5 of the inputs ...")
    inputs = runner.describe_inputs(inputs_files)

    stats_e = json.loads((config.out_path("twitter", "emtract", "emotion", "units.parquet.stats.json")).read_text())
    empty_score = {k: float(v) for k, v in stats_e["empty_text_score"]["reference_predict_empty"]["p"].items()}

    # 1. post tables and per-post checks
    for p in PLATFORMS:
        if reuse_posts and con.execute(f"SELECT count(*) FROM information_schema.tables WHERE table_name = 'posts_{p}'").fetchone()[0]:
            log(f"{p}: reusing post table")
            counts[f"posts_{p}"] = {"posts": int(con.execute(f"SELECT count(*) FROM posts_{p}").fetchone()[0])}
        else:
            log(f"{p}: building post table ...")
            counts[f"posts_{p}"] = build_posts(con, p, checks)
        counts[f"probsum_{p}"] = prob_sum_check(con, p, checks)
        log(f"{p}: {counts[f'posts_{p}']}")

    # 2. hourly sums for every variant
    con.execute("DROP TABLE IF EXISTS hourly")
    first = True
    for v, (plats, *_rest) in VARIANTS.items():
        for p in plats:
            sql = hourly_sql(p, v, empty_score)
            con.execute(("CREATE TABLE hourly AS " if first else "INSERT INTO hourly ") + sql)
            first = False
        log(f"hourly sums: {v}")

    # 3. Twitter hour profiles (pre-forecast year: applied; full sample: reported only)
    prof_pre = hour_profiles(con, *PREFORECAST)
    prof_full = hour_profiles(con, config.WINDOW_START, config.WINDOW_END)
    profile = prof_pre["corpus"] if OPEN_PROFILE_SCOPE == "corpus" else None
    check(profile is not None, "profile scope implemented: corpus")

    # 4. daily table
    daily = con.execute(daily_sql(profile)).df()
    daily["day"] = pd.to_datetime(daily["day"])
    cov = con.execute(f"""SELECT utc_day AS day, coverage_pct_utc, minutes_utc FROM
        read_parquet('{config.out_path('twitter', 'units', 'twitter_coverage_utc.parquet')}')""").df()
    cov["day"] = pd.to_datetime(cov["day"])
    check(len(cov) == config.N_DAYS, "coverage table has 1,280 days", checks)

    # complete grid: every variant x platform x series x day (variants copied from main where they do not apply)
    cal = pd.date_range(config.WINDOW_START, config.WINDOW_END, freq="D")
    check(len(cal) == config.N_DAYS, "1,280 UTC days", checks)
    grids = []
    allv = list(VARIANTS) + list(DERIVED_VARIANTS)
    for v in allv:
        plats, ser = (VARIANTS[v][0], VARIANTS[v][1]) if v in VARIANTS else ([], [])
        for p in PLATFORMS:
            for s in SERIES:
                grids.append((v, p, s, (p in plats and s in ser)))
    g = pd.DataFrame(grids, columns=["variant", "platform", "series", "variant_applies"])
    g["src_variant"] = np.where(g.variant_applies, g.variant, "main")
    g = g.merge(pd.DataFrame({"day": cal}), how="cross")
    daily = daily.rename(columns={"variant": "src_variant"})
    full = g.merge(daily, on=["src_variant", "platform", "series", "day"], how="left")
    for c in ["n_posts", "n_empty", "n_missing", "n_emo", "n_pol", "n_hours_emo"]:
        full[c] = full[c].fillna(0).astype("int64")
    full = full.merge(cov, on="day", how="left")
    full["twitter_coverage_share"] = full.coverage_pct_utc / 100.0
    full["twitter_absent"] = full.minutes_utc == 0
    tw = full.platform == "twitter"
    strat_day = tw & (full.minutes_utc > 0) & (
        (full.coverage_pct_utc < FULL_DAY_PCT) if OPEN_STRATIFY_DAYS == "partial_lt95" else True)
    full["twitter_stratified"] = strat_day & (full.variant != "a20_unstratified")
    for x in VARS:
        full[f"mean_{x}"] = np.where(full.twitter_stratified, full[f"strat_mean_{x}"], full[f"raw_mean_{x}"])
        absent = tw & full.twitter_absent
        full.loc[absent, [f"mean_{x}", f"sd_{x}"]] = np.nan
    n_abs_posts = int(full.loc[tw & full.twitter_absent & (full.variant == "main"), "n_posts"].sum())
    check(n_abs_posts == 0, "no Twitter post on a zero-capture day", checks)

    # 5. D16 calibration on the pre-forecast year (applied), full sample (reported only)
    cal_pre = calibrate(full, *PREFORECAST)
    cal_full = calibrate(full, config.WINDOW_START, config.WINDOW_END)
    for x in VARS:
        n_col = "n_pol" if x == "polarity" else "n_emo"
        key = full.platform + "|" + full.series
        m_w = {f"{p}|{s}": cal_pre[p][s][x]["sigma2_w"] for p in cal_pre for s in cal_pre[p]}
        m_b = {f"{p}|{s}": cal_pre[p][s][x]["sigma2_b"] for p in cal_pre for s in cal_pre[p]}
        m_n = {f"{p}|{s}": cal_pre[p][s][x]["n_star"] for p in cal_pre for s in cal_pre[p]}
        sw = key.map(m_w).astype(float)
        sb = key.map(m_b).astype(float)
        ns = key.map(m_n).astype(float)
        n = full[n_col].astype(float).where(full[n_col] > 0)
        n = n.where(~(tw & full.twitter_absent))
        full[f"rel_{x}"] = sb / (sb + sw / n)
        full[f"pw_{x}"] = 1.0 / (sw / n + sb)
        full[f"thin_{x}"] = (full[n_col] < ns) & n.notna()
    # 6. pooled index (D15)
    pk = ["variant", "series", "day"]
    plat = full.copy()
    agg = {f"mean_{x}": "mean" for x in VARS} | {f"sd_{x}": "mean" for x in VARS} | {
        "n_posts": "sum", "n_empty": "sum", "n_missing": "sum", "n_emo": "sum", "n_pol": "sum", "variant_applies": "max"}
    pooled = plat.groupby(pk).agg(agg).reset_index()     # pandas mean skips NaN: mean of the available platforms
    cnt = plat.groupby(pk).agg(n_platforms_emo=("mean_happy", "count"), n_platforms_pol=("mean_polarity", "count"),
                               n_platforms_sd_emo=("sd_happy", "count"), n_platforms_sd_pol=("sd_polarity", "count")).reset_index()
    used = plat[plat.mean_happy.notna()].groupby(pk).platform.agg(lambda s: "+".join(sorted(x[0].upper() for x in s)))
    pooled = pooled.merge(cnt, on=pk).merge(used.rename("platforms_used").reset_index(), on=pk, how="left")
    pooled["platform"] = "POOLED"
    pooled = pooled.merge(cov, on="day", how="left")
    pooled["twitter_coverage_share"] = pooled.coverage_pct_utc / 100.0
    pooled["twitter_absent"] = pooled.minutes_utc == 0
    twp = plat[plat.platform == "twitter"][pk + ["twitter_stratified"]]
    pooled = pooled.merge(twp, on=pk, how="left")
    plat["n_platforms_emo"] = plat.mean_happy.notna().astype(int)
    plat["n_platforms_pol"] = plat.mean_polarity.notna().astype(int)
    plat["n_platforms_sd_emo"] = plat.sd_happy.notna().astype(int)
    plat["n_platforms_sd_pol"] = plat.sd_polarity.notna().astype(int)
    plat["platforms_used"] = np.where(plat.mean_happy.notna(), plat.platform.str[0].str.upper(), None)
    out = pd.concat([plat, pooled], ignore_index=True, sort=False)
    # 7. mechanical transforms of the daily means (A59 log-ratios, A60 certainty index)
    for k in EMO6:
        out[f"lr_{k}"] = np.log(out[f"mean_{k}"] / out["mean_neutral"])
    out["certainty"] = np.log((out.mean_anger + out.mean_happy + out.mean_disgust) / (out.mean_fear + out.mean_surprise))
    out["empty_share"] = np.where(out.n_posts > 0, out.n_empty / out.n_posts.where(out.n_posts > 0), np.nan)  # A37

    keep = (["variant", "series", "platform", "day", "variant_applies", "n_posts", "n_empty", "n_missing", "n_emo", "n_pol",
             "empty_share"] + [f"mean_{x}" for x in VARS] + [f"sd_{x}" for x in VARS]
            + [f"lr_{k}" for k in EMO6] + ["certainty", "twitter_coverage_share", "twitter_absent", "twitter_stratified",
                                           "n_hours_emo", "n_platforms_emo", "n_platforms_pol", "n_platforms_sd_emo",
                                           "n_platforms_sd_pol", "platforms_used"]
            + [f"{a}_{x}" for x in VARS for a in ("rel", "pw", "thin")])
    out = out[keep].sort_values(["variant", "series", "platform", "day"]).reset_index(drop=True)
    out["day"] = out["day"].dt.date
    out["n_hours_emo"] = out["n_hours_emo"].astype("Int64")
    out.loc[out.platform == "POOLED", "n_hours_emo"] = pd.NA
    for c in ["twitter_stratified", "variant_applies", "twitter_absent"] + [f"thin_{x}" for x in VARS]:
        out[c] = out[c].astype("boolean")
    for c in [f"thin_{x}" for x in VARS] + [f"rel_{x}" for x in VARS] + [f"pw_{x}" for x in VARS]:
        out.loc[out.platform == "POOLED", c] = pd.NA

    # 8. checks
    exp_rows = len(allv) * len(SERIES) * (len(PLATFORMS) + 1) * config.N_DAYS
    check(len(out) == exp_rows, f"complete grid: {exp_rows:,} rows", checks)
    dc = con.execute(f"SELECT platform, series, day, n_posts, n_posts_with_bots FROM "
                     f"read_parquet('{config.OUT_DIR / 'panel' / 'daily_counts.parquet'}') WHERE series IN "
                     f"({', '.join(repr(s) for s in SERIES)})").df()
    dc["day"] = pd.to_datetime(dc["day"]).dt.date
    for v, col in (("main", "n_posts"), ("a43_with_bots", "n_posts_with_bots")):
        mine = out[(out.variant == v) & (out.platform != "POOLED")][["platform", "series", "day", "n_posts"]]
        mm = mine.merge(dc, on=["platform", "series", "day"], how="outer", suffixes=("", "_dc"), indicator=True)
        diff = int((mm["n_posts"] != mm[col]).sum())
        check(len(mm) == len(PLATFORMS) * len(SERIES) * config.N_DAYS and (mm._merge == "both").all() and diff == 0,
              f"{v}: n_posts equal daily_counts.{col} on all {len(mm):,} platform x series x days", checks)
    pm = out[(out.variant == "main") & (out.platform == "POOLED")]
    empty_emo = int((pm.n_platforms_emo == 0).sum())
    empty_pol = int((pm.n_platforms_pol == 0).sum())
    sd_missing = pm[pm.n_platforms_sd_emo == 0][["series", "day"]]
    counts["pooled_main_empty_emotion_days"] = empty_emo
    counts["pooled_main_empty_polarity_days"] = empty_pol
    counts["pooled_main_sd_missing_days"] = [f"{r.series} {r.day}" for r in sd_missing.itertuples()]
    check(empty_emo == 0 and empty_pol == 0, "no coin-day empty in the main pooled index (emotions and polarity)", checks)

    # Twitter monthly means against the 6 Oct check (distinct posts in any series, bot-filtered, own text)
    anyser = " OR ".join(f"in_{s.lower()}" for s in SERIES)
    mon = con.execute(f"""SELECT date_trunc('month', day) AS month, count(*) AS n_main,
        {', '.join(f'avg(p_{x}) AS m_{x}' for x in EMO)} FROM posts_twitter
        WHERE ({anyser}) AND NOT bot_any AND NOT is_missing AND NOT is_empty AND p_happy IS NOT NULL
        GROUP BY 1 ORDER BY 1""").df()
    counts["twitter_monthly_check"] = mon.assign(month=mon.month.astype(str)).round(6).to_dict(orient="records")
    log(f"done in {time.time() - t0:.0f} s; writing")

    # 9. write
    cal_obj = {
        "written_by": STEP, "code_sha": CODE_SHA, "calibration_window": list(PREFORECAST),
        "rule_sources": "MASTERPLAN_NOV15.md section 7 D16, D3, decisions of 6 Oct 2026 (1)-(2); "
                        "HAR_VARIANTS_AND_THRESHOLDS_22SEP.md section 3, T4, T5; pre_analysis_note_D8.md section 3",
        "reliability_target": RELIABILITY_TARGET, "full_day_pct": FULL_DAY_PCT,
        "estimator": "one-way random effects, ANOVA (method of moments): sigma2_w = MSW, "
                     "sigma2_b = max((MSB - MSW) / n0, 0), n0 = (N - sum n_d^2 / N) / (D - 1); "
                     "n* = (R / (1 - R)) * sigma2_w / sigma2_b; R(n) = sigma2_b / (sigma2_b + sigma2_w / n); "
                     "precision weight = 1 / (sigma2_w / n + sigma2_b)",
        "sample": "variant main; platform rows; days with >= 1 post in the mean; Twitter zero-capture days excluded; "
                  "Twitter day means unstratified",
        "open_parameters": {"OPEN_STRATIFY_DAYS": OPEN_STRATIFY_DAYS, "OPEN_PROFILE_SCOPE": OPEN_PROFILE_SCOPE,
                            "OPEN_PROFILE_DAYS": OPEN_PROFILE_DAYS, "OPEN_STRATIFIED_SD": OPEN_STRATIFIED_SD,
                            "OPEN_POOLED_RELIABILITY": OPEN_POOLED_RELIABILITY, "OPEN_A21_POLARITY": OPEN_A21_POLARITY,
                            "OPEN_A48_POLARITY": OPEN_A48_POLARITY},
        "twitter_hour_profile": {"applied": "corpus", "preforecast": prof_pre},
        "d16_reliability": cal_pre,
        "emtract_empty_text_score_A21": empty_score,
        "reported_only_full_sample": {"twitter_hour_profile": prof_full, "d16_reliability": cal_full,
                                      "window": [config.WINDOW_START, config.WINDOW_END]},
    }
    tmp = runner.tmp_path(OUT)
    out.to_parquet(tmp, index=False, compression="zstd")
    back = pd.read_parquet(tmp)
    check(len(back) == len(out), "written table reads back with all rows", checks)
    runner.commit(tmp, OUT)
    runner.write_json_atomic(CAL, cal_obj)
    counts["rows"] = int(len(out))
    counts["variants"] = allv
    runner.write_stats(OUT, step=STEP, inputs=inputs, code_sha=CODE_SHA, rules_sha=RULES_SHA, con=con, counts=counts,
                       checks=checks, extra={"seconds": round(time.time() - t0, 1),
                                             "calibration_md5": runner.file_md5(CAL)})
    con.close()
    log(f"{OUT}: {len(out):,} rows; {CAL}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse-posts", action="store_true", help="reuse the post tables of the scratch database")
    args = ap.parse_args(argv)
    log(f"{STEP}: code {CODE_SHA}")
    return run(args.reuse_posts)


if __name__ == "__main__":
    raise SystemExit(main())
