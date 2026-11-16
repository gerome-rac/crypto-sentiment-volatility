"""Shared by the three series scripts (04_series_reddit.py, 04_series_stocktwits.py, 04_series_twitter.py).

assign() runs a platform's assignment query and writes the one-row-per-post table through
the safe runner: temporary name, checks (one row per post, every post has its bot flags,
no post in a coin series and in the market), commit, stats file with the count of every
flag and the size of every series after the bot filter. link_evidence() and links_of()
give the robustness columns for coin words inside links (A40).
"""
from __future__ import annotations

import time
from typing import Dict, List

from pipeline import config, runner
from pipeline import rules as R
from pipeline.runner import check, log

SERIES = [c.lower() for c in R.COINS] + ["market"]


def link_evidence(coin: str, links: str, ctx: str) -> str:
    """A40: coin c from words inside links (tier 1/2, or tier 3 with crypto context in the text)."""
    parts = [R.match(links, R.COIN_PATTERNS[(coin, t)]) for t in ("t1", "t2") if R.COIN_PATTERNS[(coin, t)]]
    if coin != "BNB" and R.COIN_PATTERNS[(coin, "t3")]:
        parts.append(f"({R.match(links, R.COIN_PATTERNS[(coin, 't3')])} AND {ctx})")
    return "(" + " OR ".join(parts) + ")"


def links_of(text_expr: str) -> str:
    return f"array_to_string(regexp_extract_all(py_html_unescape(coalesce({text_expr}, '')), {R.dq(R.URL_RX)}), ' ')"


def assign(platform: str, sql: str, step: str, code_sha: str, rules_sha: str) -> int:
    """Write <platform>/panel/assignment.parquet from the platform's assignment query (step 4)."""
    import glob as _glob
    if platform == "twitter":
        srcs = [str(config.out_path("twitter", "units", "units.parquet"))]
    else:
        srcs = sorted(_glob.glob(str(config.out_path(platform, "coarse", "*", "data_0.parquet"))))
    srcs.append(str(config.out_path(platform, "bot", "post_flags.parquet")))
    inputs = runner.describe_inputs(srcs)
    final = config.out_path(platform, "panel", "assignment.parquet")
    if runner.is_current(final, inputs, code_sha, rules_sha):
        log(f"{platform}: assignment current, skipped")
        return 0
    t0 = time.time()
    con = runner.connect()
    R.register_udfs(con)
    tmp = runner.tmp_path(final)
    con.execute(f"COPY ({sql}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    o = f"read_parquet('{tmp}')"
    checks: List[Dict[str, object]] = []
    n, n_ids, n_noflag = con.execute(f"SELECT count(*), count(DISTINCT post_id), count_if(bot_any IS NULL) FROM {o}").fetchone()
    check(n_ids == n, f"one row per post ({n:,})", checks)
    check(n_noflag == 0, "every post has its bot flags", checks)
    both = con.execute(f"SELECT count_if(in_market AND ({' OR '.join('in_' + c.lower() for c in R.COINS)})) FROM {o}").fetchone()[0]
    check(both == 0, "no post is in a coin series and in the market", checks)
    cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {o}").fetchall()]
    boolean_cols = [c for c in cols if c.startswith(("in_", "bot_", "textgate_", "link_", "reply_parent_"))
                    or c in ("equity_only", "bnb_strict", "bsc_context", "coin_only_in_link", "rbinance_support",
                             "rbtc_bitcoin_cash", "rbtc_no_ticker", "ambiguous_extra", "ambiguous_multi", "edited",
                             "invite_link", "is_reply", "body_null", "authored", "authored_before_window",
                             "own_text_hit")]
    vals = con.execute("SELECT " + ", ".join(f"count_if({c})" for c in boolean_cols) + f" FROM {o}").fetchone()
    main = con.execute("SELECT " + ", ".join(f"count_if(in_{s} AND NOT bot_any)" for s in SERIES) + f" FROM {o}").fetchone()
    counts = {"posts": int(n), "flag_counts": dict(zip(boolean_cols, (int(v) for v in vals))),
              "series_after_bot_pass": dict(zip(SERIES, (int(v) for v in main)))}
    runner.commit(tmp, final)
    runner.write_stats(final, step=step, inputs=inputs, code_sha=code_sha, rules_sha=rules_sha, con=con,
                       counts=counts, checks=checks, extra={"platform": platform, "seconds": round(time.time() - t0, 1)})
    con.close()
    log(f"{platform}: assignment of {n:,} posts in {time.time() - t0:.0f} s; after the bot pass: "
        + ", ".join(f"{s} {v:,}" for s, v in counts["series_after_bot_pass"].items()))
    return 0
