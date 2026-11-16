#!/usr/bin/env python3
"""
05_daily_counts.py -- step 5: daily post counts per platform, series and UTC day.

Input    data/pipeline/<platform>/panel/assignment.parquet (step 4) for every platform that
         has one; for Twitter also units/retweet_events_daily.parquet; for Reddit also the
         coarse tables (dead rows).
Output   data/pipeline/panel/daily_counts.parquet: one row per platform, series and day on
         the complete 1,280-day UTC calendar (a day without posts has 0).

Columns
-------
    n_posts            posts that count for the series that day, after the bot filter
    n_posts_with_bots  the same before the bot filter (robustness A43)
    n_bot_flagged      posts flagged by the bot filter
    n_accounts         distinct accounts behind n_posts
    n_dead             Reddit only: removed or deleted rows of the series' group that day
                       (total activity, robustness A24); NULL on the other platforms
Series: BTC, ETH, BNB, XRP, DOGE, MARKET and CRYPTO_SET (a post in any of them, counted
once); Twitter also <SERIES>_RETWEET_EVENTS, every retweet event on its own day
(robustness A12).

Checks: 1,280 days; the complete platform x series x day grid; per platform and series the
daily counts add up to the assignment; Reddit's n_dead adds up to the dead rows of each group.

Usage (from code/)
------------------
    python pipeline/05_daily_counts.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, runner, series  # noqa: E402
from pipeline import rules as R  # noqa: E402
from pipeline.runner import check, log  # noqa: E402
from pipeline.series import SERIES  # noqa: E402

STEP = "05_daily_counts"
CODE_SHA = runner.code_fingerprint(__file__, series.__file__)
RULES_SHA = R.section_sha("series") + "-" + R.section_sha("filter")


def daily_counts() -> int:
    parts = [p for p in config.PLATFORMS if config.out_path(p, "panel", "assignment.parquet").exists()]
    check(len(parts) > 0, "at least one platform has an assignment")
    import glob as _glob
    dead_src = sorted(_glob.glob(str(config.out_path("reddit", "coarse", "coin=*", "data_0.parquet")))) if "reddit" in parts else []
    inputs = runner.describe_inputs([config.out_path(p, "panel", "assignment.parquet") for p in parts] + dead_src
                                    + ([config.out_path("twitter", "units", "retweet_events_daily.parquet")]
                                       if "twitter" in parts else []))   # read for A12; listed since 28 Sep 2026
    final = config.OUT_DIR / "panel" / "daily_counts.parquet"
    con = runner.connect()
    t0 = time.time()
    selects = []
    for p in parts:
        a = f"read_parquet('{config.out_path(p, 'panel', 'assignment.parquet')}')"
        for s in SERIES:
            selects.append(f"""SELECT '{p}' AS platform, '{s.upper()}' AS series, day,
                   count_if(NOT bot_any) AS n_posts, count(*) AS n_posts_with_bots, count_if(bot_any) AS n_bot_flagged,
                   count(DISTINCT account) FILTER (WHERE NOT bot_any) AS n_accounts
                FROM {a} WHERE in_{s} GROUP BY day""")
            if p == "twitter":   # A12: every retweet event on its own day, units after the bot pass
                ev = f"read_parquet('{config.out_path('twitter', 'units', 'retweet_events_daily.parquet')}')"
                selects.append(f"""SELECT 'twitter' AS platform, '{s.upper()}_RETWEET_EVENTS' AS series,
                       e.retweet_day AS day, sum(e.n_events) FILTER (WHERE NOT u.bot_any) AS n_posts,
                       sum(e.n_events) AS n_posts_with_bots, sum(e.n_events) FILTER (WHERE u.bot_any) AS n_bot_flagged,
                       CAST(NULL AS BIGINT) AS n_accounts
                    FROM {ev} e JOIN {a} u ON u.post_id = e.unit_id WHERE u.in_{s} GROUP BY 1, 2, 3""")
        selects.append(f"""SELECT '{p}' AS platform, 'CRYPTO_SET' AS series, day,
               count_if(NOT bot_any) AS n_posts, count(*) AS n_posts_with_bots, count_if(bot_any) AS n_bot_flagged,
               count(DISTINCT account) FILTER (WHERE NOT bot_any) AS n_accounts
            FROM {a} WHERE {' OR '.join('in_' + s for s in SERIES)} GROUP BY day""")
    series_list = ", ".join(f"('{s.upper()}')" for s in SERIES + ["crypto_set"])
    plat_list = ", ".join(f"('{p}')" for p in parts)
    extra_grid = ""
    if "twitter" in parts:
        extra_grid = " UNION ALL SELECT 'twitter', s.series, cal.day FROM (VALUES " + ", ".join(
            f"('{s.upper()}_RETWEET_EVENTS')" for s in SERIES) + ") s(series) CROSS JOIN cal"
    if dead_src:   # A24: dead rows by partition (coin_label) and UTC day; CRYPTO_SET = the five coin partitions
        rc = "read_parquet([" + ", ".join(f"'{f}'" for f in dead_src) + "], hive_partitioning = false)"
        dead_sql = (f"SELECT 'reddit' AS platform, coin_label AS series, CAST(ts AS DATE) AS day, count(*) AS n_dead "
                    f"FROM {rc} WHERE is_dead GROUP BY 1, 2, 3 UNION ALL "
                    f"SELECT 'reddit', 'CRYPTO_SET', CAST(ts AS DATE), count(*) FROM {rc} "
                    f"WHERE is_dead AND coin_label <> 'MARKET' GROUP BY 3")
    else:
        dead_sql = ("SELECT CAST(NULL AS VARCHAR) AS platform, CAST(NULL AS VARCHAR) AS series, CAST(NULL AS DATE) AS day, "
                    "CAST(NULL AS BIGINT) AS n_dead WHERE FALSE")
    tmp = runner.tmp_path(final)
    con.execute(f"""COPY (
        WITH cal AS (SELECT CAST(d AS DATE) AS day FROM generate_series(TIMESTAMP '{config.WINDOW_START}',
                                                                       TIMESTAMP '{config.WINDOW_END}', INTERVAL 1 DAY) t(d)),
             grid AS (SELECT p.platform, s.series, cal.day FROM (VALUES {plat_list}) p(platform)
                      CROSS JOIN (VALUES {series_list}) s(series) CROSS JOIN cal{extra_grid}),
             c AS ({' UNION ALL '.join(selects)}),
             d AS ({dead_sql})
        SELECT g.platform, g.series, g.day, CAST(coalesce(c.n_posts, 0) AS BIGINT) AS n_posts,
               CAST(coalesce(c.n_posts_with_bots, 0) AS BIGINT) AS n_posts_with_bots,
               CAST(coalesce(c.n_bot_flagged, 0) AS BIGINT) AS n_bot_flagged,
               -- BIGINT: count_if and sum give HUGEINT, which Parquet stores as DOUBLE; retweet-event rows
               -- have no account count (NULL, not 0) -- both 28 Sep 2026
               CASE WHEN suffix(g.series, '_RETWEET_EVENTS') THEN CAST(NULL AS BIGINT)
                    ELSE CAST(coalesce(c.n_accounts, 0) AS BIGINT) END AS n_accounts,
               CASE WHEN g.platform = 'reddit' THEN coalesce(d.n_dead, 0) END AS n_dead
        FROM grid g LEFT JOIN c USING (platform, series, day) LEFT JOIN d USING (platform, series, day)
        ORDER BY 1, 2, 3) TO '{tmp}' (FORMAT PARQUET)""")
    o = f"read_parquet('{tmp}')"
    checks: List[Dict[str, object]] = []
    n, days = con.execute(f"SELECT count(*), count(DISTINCT day) FROM {o}").fetchone()
    check(days == config.N_DAYS, f"{config.N_DAYS} UTC days", checks)
    n_series = len(parts) * (len(SERIES) + 1) + (len(SERIES) if "twitter" in parts else 0)
    check(n == config.N_DAYS * n_series, "complete platform x series x day grid", checks)
    for p in parts:
        a = f"read_parquet('{config.out_path(p, 'panel', 'assignment.parquet')}')"
        for s in SERIES:
            want = con.execute(f"SELECT coalesce(count_if(NOT bot_any), 0) FROM {a} WHERE in_{s} "
                               f"AND day BETWEEN DATE '{config.WINDOW_START}' AND DATE '{config.WINDOW_END}'").fetchone()[0]
            got = con.execute(f"SELECT coalesce(sum(n_posts), 0) FROM {o} WHERE platform = '{p}' AND series = '{s.upper()}'").fetchone()[0]
            check(int(got) == int(want), f"{p} {s}: daily counts add up to the assignment ({int(got):,})", checks)
    if dead_src:
        for part in list(R.COINS) + ["MARKET"]:
            want = con.execute(f"SELECT coalesce(count_if(is_dead), 0) FROM {rc} WHERE coin_label = '{part}'").fetchone()[0]
            got = con.execute(f"SELECT coalesce(sum(n_dead), 0) FROM {o} WHERE platform = 'reddit' AND series = '{part}'").fetchone()[0]
            check(int(got) == int(want), f"reddit {part}: n_dead adds up to the dead rows of the partition ({int(got):,})", checks)
    totals = {f"{p}/{s}": int(v) for p, s, v in con.execute(
        f"SELECT platform, series, sum(n_posts) FROM {o} GROUP BY 1, 2 ORDER BY 1, 2").fetchall()}
    runner.commit(tmp, final)
    runner.write_stats(final, step=STEP, inputs=inputs, code_sha=CODE_SHA, rules_sha=RULES_SHA, con=con,
                       counts={"rows": int(n), "totals_after_bot_pass": totals}, checks=checks,
                       extra={"seconds": round(time.time() - t0, 1)})
    con.close()
    log(f"daily counts: {n:,} rows for {', '.join(parts)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    log(f"{STEP}: code {CODE_SHA}, rules {RULES_SHA}")
    return daily_counts()


if __name__ == "__main__":
    raise SystemExit(main())
