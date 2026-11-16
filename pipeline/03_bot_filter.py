#!/usr/bin/env python3
"""
03_bot_filter.py -- step 3: the bot and spam filter, the same rules on all three platforms.

Input    the posts of step 2 (Reddit and StockTwits: the coarse tables; Twitter: the units).
Output   data/pipeline/<platform>/bot/post_flags.parquet (one row per post, one flag per
         rule, and bot_any), campaign_groups.parquet, campaign_account_days.parquet and
         template_account_days.parquet, each with a stats file.

How the rules are applied
-------------------------
* Every rule runs on ALL posts of a platform and writes its own flag. No rule runs only on
  what another rule left, and no post is deleted (decided 25-26 Sep 2026). A post is a bot
  post if any rule except invite flags it (bot_any); step 4 leaves such posts out of the
  main series, and one robustness arm keeps them.
* Every flag uses only information up to the post's own UTC day: an account's window is
  its 200 most recent posts up to and including that day, and a campaign group is found on
  one UTC day. No later post changes an earlier flag.
* Rules that do not exist on a platform are left out (StockTwits has no user names and no
  AutoModerator).

Rules and their sources
-----------------------
automod    Reddit: the author is AutoModerator (arXiv 2601.14423).
self_id    Reddit: the own text contains "performed automatically" or "i am a bot"
           (AutoModerator's disclaimer; self-identified bots, Yang et al. 2020).
name       Reddit: "bot" is a separate token of the user name (arXiv 2601.14423).
keyword    Kraaijeveld & De Smedt (2020) as adapted in Electronic Markets (2025, 35:64): at
           least 2 of "give away"/"giving away"; "pump" with "join" or "register"; more than
           5 hashtags; more than 5 crypto tickers; "bot" in the user name (StockTwits has no
           names: 2 of 4). Per post. The crypto ticker list is every StockTwits .X symbol
           with the suffix removed, so StockTwits runs first.
campaign   Chen & Subramanian (arXiv 1804.05232): a fingerprint posted by at least 20
           accounts on one UTC day forms a group; a member account is a bot that day if at
           least 60% of its 200 most recent posts up to that day carry fingerprints found in
           the windows of at least 3 of the group's accounts; all its posts that day are
           flagged.
template   Electronic Markets (2025, 35:64): two posts are one template if difflib's
           SequenceMatcher ratio exceeds 0.7; an account is a bot on a day if more than half
           of the pairs among its 200 most recent posts up to that day are. Computed
           exactly; rapidfuzz's Indel similarity (an upper bound of the ratio) only selects
           the pairs difflib has to check.
invite     Telegram/Discord invite links (Nizzoli et al. 2020): flagged and reported, never
           part of bot_any.

Choices within the published rules, stated so they can be checked
------------------------------------------------------------------
* texts are HTML-decoded before comparison; a pair with an empty text is never similar
  (a post without own words is no copy of a template);
* difflib is called as SequenceMatcher(None, older, newer).ratio();
* account windows are ordered by (timestamp, post id); Reddit rows without an author
  (deleted accounts) take part in the per-post rules only;
* the fingerprint (step 2) is the own text lowercased, without links, mentions,
  punctuation and emojis, with at least FINGERPRINT_MIN_WORDS words.

The template rule is the slow one (measured with 4 workers: Reddit 51 min, StockTwits
56 min, Twitter 13 min). Its result is cached: it is
recomputed only when its code, its rules or the texts, accounts or days of the posts
change (template_sha plus a content digest); --only-template computes just this rule.

Decisions: MASTERPLAN D21; Reddit defect map P22; StockTwits defect map PG4 (information up to the
post's day).

Usage (from code/)
------------------
    python pipeline/03_bot_filter.py stocktwits [--workers 8]
    python pipeline/03_bot_filter.py reddit
    python pipeline/03_bot_filter.py twitter
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, runner  # noqa: E402
from pipeline import rules as R  # noqa: E402
from pipeline.runner import check, log  # noqa: E402

STEP = "03_bot_filter"
CODE_SHA = runner.code_fingerprint(__file__)
RULES_SHA = R.section_sha("bot") + "-" + R.section_sha("filter")
N_SHARDS = 64

# Per platform: where the posts are and what each column is called.
SPECS = {
    "reddit": dict(
        src=lambda: str(config.out_path("reddit", "coarse", "coin=*", "data_0.parquet")),
        where="NOT is_dead", post_id="id", account="author_hash", ts="ts", own="own_text",
        name_has="name_has_bot", extra=["is_automoderator", "name_bot_token"], tickers="cashtags"),
    "stocktwits": dict(
        src=lambda: str(config.out_path("stocktwits", "coarse", "month=*", "data_0.parquet")),
        where="TRUE", post_id="message_id", account="CAST(user_id AS VARCHAR)", ts="created_at",
        own="message_body", name_has=None, extra=[], tickers="n_crypto_symbols"),
    "twitter": dict(
        src=lambda: str(config.out_path("twitter", "units", "units.parquet")),
        where="TRUE", post_id="unit_id", account="author_id", ts="panel_ts", own="own_text",
        name_has="author_name_has_bot", extra=[], tickers="cashtags"),
}


# ---------------------------------------------------------------------------
# Template rule: workers (run in separate processes)
# ---------------------------------------------------------------------------
# Similar pairs are stored as a band matrix B (one row per post, one column per
# distance 1..199): B[j, k] = 1 if post j and post j-k-1 of the same account are
# one template. Accounts with more than HEAVY_POSTS posts are split into chunks
# that run in parallel; their window counts are taken once all chunks are done.

HEAVY_POSTS = 25_000


def _ratio_fn():
    """difflib's SequenceMatcher ratio; cydifflib (a compiled port, identical results) when installed."""
    try:
        import cydifflib
        return (lambda a, b: cydifflib.SequenceMatcher(None, a, b).ratio()), "cydifflib"
    except ImportError:
        return (lambda a, b: difflib.SequenceMatcher(None, a, b).ratio()), "difflib"


def _band(texts: List[str], s: int, e: int):
    """Rows s..e-1 of the band matrix of one account (texts: all its posts in order)."""
    import numpy as np
    from rapidfuzz.distance import Indel
    from rapidfuzz.process import cdist

    ratio, _ = _ratio_fn()
    w, thr = R.TEMPLATE_WINDOW, R.TEMPLATE_RATIO
    B = np.zeros((e - s, w - 1), dtype=np.uint8)
    cache: Dict[Tuple[str, str], bool] = {}
    for bs in range(s, e, w):
        be = min(e, bs + w)
        lo = max(0, bs - (w - 1))
        if be - lo < 2:
            continue
        m = cdist(texts[bs:be], texts[lo:be], scorer=Indel.normalized_similarity, score_cutoff=thr - 0.01,
                  dtype=np.float32, workers=1)
        rows, cols = np.nonzero(m > thr - 0.01)
        j = bs + rows
        i = lo + cols
        d = j - i
        keep = (d >= 1) & (d <= w - 1)
        for jj, ii in zip(j[keep].tolist(), i[keep].tolist()):
            a, b = texts[ii], texts[jj]
            if not a or not b:
                continue                       # a post without own words is no copy of a template
            if a == b:
                ok = True
            else:
                ok = cache.get((a, b))
                if ok is None:
                    ok = ratio(a, b) > thr      # SequenceMatcher(None, older, newer)
                    if len(cache) < 1_000_000:
                        cache[(a, b)] = ok
            if ok:
                B[jj - s, jj - ii - 1] = 1
    return B


def _windows(days, B) -> List[Tuple[int, int, int, int]]:
    """Window of the 200 most recent posts ending at each day's last post: (day, posts, pairs, similar pairs)."""
    import numpy as np
    w = R.TEMPLATE_WINDOW
    days = np.asarray(days)
    ks = np.arange(w - 1)
    ends = np.nonzero(np.r_[days[1:] != days[:-1], True])[0]
    Bb = B.astype(bool)
    out = []
    lo, hi, count = 0, -1, 0
    for r in ends.tolist():
        if r > hi:                              # add posts hi+1..r: their partners inside [lo, r]
            rows = np.arange(hi + 1, r + 1)
            mask = ks[None, :] < (rows - lo)[:, None]
            count += int(np.count_nonzero(Bb[hi + 1:r + 1] & mask))
            hi = r
        new_lo = max(0, r - w + 1)
        if new_lo > lo:                         # drop posts lo..new_lo-1: their pairs with later posts
            ii = np.arange(lo, new_lo)
            jj = ii[:, None] + ks[None, :] + 1
            valid = jj <= hi
            count -= int(np.count_nonzero(Bb[np.where(valid, jj, 0), ks[None, :]] & valid))
            lo = new_lo
        size = hi - lo + 1
        out.append((int(days[r]), size, size * (size - 1) // 2, count))
    return out


def _read_shard(shard_path: str, account=None):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(shard_path, columns=["account", "rk", "day", "txt"])
    if account is not None:
        t = t.filter(pc.equal(t.column("account"), account))
    t = t.sort_by([("account", "ascending"), ("rk", "ascending")])     # partitioned writes keep no order
    return (t.column("account").to_pylist(), t.column("day").cast(pa.int32()).to_pylist(),
            [x or "" for x in t.column("txt").to_pylist()])


def _write_results(rows, out_path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    acc, day, size, pairs, sim = (list(x) for x in zip(*rows)) if rows else ([], [], [], [], [])
    out = pa.table({"account": pa.array(acc, pa.string()), "day": pa.array(day, pa.int32()).cast(pa.date32()),
                    "window_posts": pa.array(size, pa.int32()), "pairs": pa.array(pairs, pa.int64()),
                    "similar_pairs": pa.array(sim, pa.int64())})
    pq.write_table(out, out_path + ".tmp", compression="zstd")
    os.replace(out_path + ".tmp", out_path)


def template_light(shard_path: str, out_path: str, heavy: List[str]) -> Dict[str, int]:
    """Every account of a shard except the heavy ones, start to finish."""
    acc, day, txt = _read_shard(shard_path)
    heavy_set = set(heavy)
    rows, n_posts, start, n = [], 0, 0, len(acc)
    while start < n:
        end = start
        a = acc[start]
        while end < n and acc[end] == a:
            end += 1
        if a not in heavy_set:
            B = _band(txt[start:end], 0, end - start)
            rows.extend((a,) + r for r in _windows(day[start:end], B))
            n_posts += end - start
        start = end
    _write_results(rows, out_path)
    return {"posts": n_posts, "account_days": len(rows)}


def template_chunk(shard_path: str, account: str, s: int, e: int, out_npy: str) -> Dict[str, int]:
    """Rows s..e-1 of a heavy account's band matrix."""
    import numpy as np
    _, _, txt = _read_shard(shard_path, account)
    B = _band(txt, s, min(e, len(txt)))
    np.save(out_npy + ".tmp.npy", B)
    os.replace(out_npy + ".tmp.npy", out_npy)
    return {"rows": int(B.shape[0])}


def template_heavy_finish(shard_path: str, account: str, npys: List[str], out_path: str) -> Dict[str, int]:
    import numpy as np
    _, day, _ = _read_shard(shard_path, account)
    B = np.concatenate([np.load(p) for p in npys])
    assert B.shape[0] == len(day), (account, B.shape, len(day))
    rows = [(account,) + r for r in _windows(day, B)]
    _write_results(rows, out_path)
    return {"posts": len(day), "account_days": len(rows)}


def p_select(platform: str, src: str, n_tick: str) -> str:
    """One row per post: id, account, time, UTC day, fingerprint, rule inputs, decoded own text, rank in account."""
    spec = SPECS[platform]
    extra = "".join(f", {c}" for c in spec["extra"])
    name_has = spec["name_has"] or "CAST(NULL AS BOOLEAN)"
    st = ", n_crypto_symbols" if platform == "stocktwits" else ""
    return f"""
          SELECT post_id, account, ts, CAST(ts AS DATE) AS day, fingerprint AS fp, n_hashtags, {n_tick} AS n_tickers,
                 name_has_bot, own_dec{extra},
                 CASE WHEN account IS NULL THEN NULL
                      ELSE row_number() OVER (PARTITION BY account ORDER BY ts, post_id) END AS rk
          FROM (SELECT {spec['post_id']} AS post_id, {spec['account']} AS account, {spec['ts']} AS ts, fingerprint,
                       n_hashtags, cashtags{st}, {name_has} AS name_has_bot,
                       py_html_unescape(coalesce({spec['own']}, '')) AS own_dec{extra}
                FROM read_parquet('{src}', union_by_name = true) WHERE {spec['where']})"""


def template_sha() -> str:
    """Digest of everything the template result depends on besides its input files."""
    import hashlib
    import inspect
    parts = [inspect.getsource(f) for f in (p_select, _band, _windows, _read_shard, _write_results, template_light,
                                            template_chunk, template_heavy_finish)]
    parts.append(str(HEAVY_POSTS))
    parts += [repr(SPECS[k]["where"]) + repr(SPECS[k]["account"]) + repr(SPECS[k]["own"]) for k in sorted(SPECS)]
    # only the template's own rules and the decoding it reads through; the coarse-pass rules do not enter
    parts += [str(N_SHARDS), repr((R.TEMPLATE_WINDOW, R.TEMPLATE_RATIO, R.TEMPLATE_SHARE, R.RULES_VERSION)),
              inspect.getsource(R.register_udfs)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def crypto_tickers(con) -> List[str]:
    """The crypto ticker list: every StockTwits .X symbol, suffix removed (keyword rule, tickers)."""
    src = config.out_path("stocktwits", "coarse", "month=*", "data_0.parquet")
    rows = con.execute(f"SELECT DISTINCT regexp_replace(s, '\\.X$', '') AS t FROM "
                       f"(SELECT unnest(crypto_symbols) AS s FROM read_parquet('{src}')) ORDER BY t").fetchall()
    return [r[0] for r in rows]


def run(platform: str, workers: int, only_template: bool = False) -> int:
    spec = SPECS[platform]
    out_dir = config.out_path(platform, "bot")
    scratch = config.SCRATCH_DIR / f"bot_{platform}"
    scratch.mkdir(parents=True, exist_ok=True)
    src = spec["src"]()
    import glob as _glob
    inputs = runner.describe_inputs(_glob.glob(src))
    check(len(inputs) > 0, f"inputs found: {src}")
    final = out_dir / "post_flags.parquet"
    if platform != "stocktwits" and not only_template:
        st_inputs = runner.describe_inputs(_glob.glob(SPECS["stocktwits"]["src"]()))
        inputs = inputs + st_inputs          # the ticker list comes from StockTwits
    if not only_template and runner.is_current(final, inputs, CODE_SHA, RULES_SHA):
        log(f"{platform}: current, skipped")
        return 0
    t0 = time.time()
    con = runner.connect()
    R.register_udfs(con)
    checks: List[Dict[str, object]] = []
    counts: Dict[str, object] = {}

    # ---- P: one row per post with what the rules need -----------------------
    p_path = scratch / "P.parquet"
    if platform == "stocktwits":
        n_tick = "n_crypto_symbols"
    else:
        tick = crypto_tickers(con) if not only_template else []
        counts["crypto_ticker_list"] = len(tick)
        con.execute("CREATE TEMP TABLE TK AS SELECT unnest(?::VARCHAR[]) AS t", [tick])
        n_tick = "len(list_intersect(cashtags, (SELECT list(t) FROM TK)))"
    con.execute(f"COPY ({p_select(platform, src, n_tick)}) TO '{p_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    P = f"read_parquet('{p_path}')"
    n_posts, n_ids, n_acc_null = con.execute(f"SELECT count(*), count(DISTINCT post_id), count_if(account IS NULL) FROM {P}").fetchone()
    check(n_ids == n_posts, f"post ids unique ({n_posts:,})", checks)
    counts.update({"posts": int(n_posts), "posts_without_account": int(n_acc_null)})
    log(f"{platform}: {n_posts:,} posts prepared in {time.time() - t0:.0f} s")

    # ---- template rule (Electronic Markets 2025) ----------------------------------
    # The result depends only on the platform's own posts, so it is cached with its
    # own stats and reused when only other rules or the ticker list changed.
    tad_final = out_dir / "template_account_days.parquet"
    t_sha = template_sha()
    # keyed on the content the rule reads, so an identical re-run of step 2 keeps it current
    n_c, x_c, s_c = con.execute(f"""SELECT count(*), bit_xor(hash(account, rk, day, own_dec)),
                                          sum(hash(account, rk, day, own_dec) % 1000000007)
                                   FROM {P} WHERE account IS NOT NULL""").fetchone()
    template_inputs = [{"path": f"{platform}: posts with an account (account, rk, day, own_dec)",
                        "bytes": int(n_c), "md5": f"{x_c}-{s_c}"}]
    if runner.is_current(tad_final, template_inputs, t_sha, t_sha):
        log(f"{platform}: template rule: cached result is current, reused")
    else:
        shard_dir = scratch / "template_in"
        res_dir = scratch / "template_out"
        for d in (shard_dir, res_dir):
            if d.exists():
                import shutil
                shutil.rmtree(d)
            d.mkdir(parents=True)
        con.execute(f"""COPY (SELECT account, rk, day, own_dec AS txt, hash(account) % {N_SHARDS} AS shard
                              FROM {P} WHERE account IS NOT NULL)
                        TO '{shard_dir}' (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (shard))""")
        shards = {int(p.parent.name.split("=")[1]): p for p in shard_dir.glob("shard=*/*.parquet")}
        heavy = con.execute(f"""SELECT account, CAST(hash(account) % {N_SHARDS} AS INTEGER) AS shard, count(*) AS n
                                FROM {P} WHERE account IS NOT NULL GROUP BY 1, 2 HAVING count(*) > {HEAVY_POSTS}
                                ORDER BY n DESC""").fetchall()
        heavy_by_shard: Dict[int, List[str]] = {}
        for acc, sh, n in heavy:
            heavy_by_shard.setdefault(sh, []).append(acc)
        _, impl = _ratio_fn()
        log(f"{platform}: template rule: {len(shards)} shards, {len(heavy)} accounts with more than {HEAVY_POSTS:,} "
            f"posts split into chunks, {workers} workers, ratio by {impl}")
        agg = {"posts": 0, "account_days": 0}
        chunk_files: Dict[str, List[str]] = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for k, (acc, sh, n) in enumerate(heavy):          # the long units first
                for c, s in enumerate(range(0, n, HEAVY_POSTS)):
                    npy = str(res_dir / f"heavy{k:04d}_chunk{c:04d}.npy")
                    chunk_files.setdefault(acc, []).append(npy)
                    futs[ex.submit(template_chunk, str(shards[sh]), acc, s, s + HEAVY_POSTS, npy)] = ("chunk", acc)
            for sh, p in sorted(shards.items()):
                futs[ex.submit(template_light, str(p), str(res_dir / f"light_{sh:03d}.parquet"),
                               heavy_by_shard.get(sh, []))] = ("light", sh)
            done = 0
            for f in as_completed(futs):
                r = f.result()
                if futs[f][0] == "light":
                    agg["posts"] += r["posts"]
                    agg["account_days"] += r["account_days"]
                done += 1
                if done % 16 == 0 or done == len(futs):
                    log(f"{platform}: template {done}/{len(futs)} work units, {time.time() - t0:.0f} s")
            fin = {ex.submit(template_heavy_finish, str(shards[sh]), acc, chunk_files[acc],
                             str(res_dir / f"heavy_{k:04d}.parquet")): acc for k, (acc, sh, n) in enumerate(heavy)}
            for f in as_completed(fin):
                r = f.result()
                agg["posts"] += r["posts"]
                agg["account_days"] += r["account_days"]
        for p in res_dir.glob("*.npy"):
            p.unlink()
        agg["ratio_implementation"] = impl
        agg["heavy_accounts"] = len(heavy)
        t_checks: List[Dict[str, object]] = []
        check(agg["posts"] == n_posts - n_acc_null, "template: every post with an account was read once", t_checks)
        tp = runner.tmp_path(tad_final)
        con.execute(f"""COPY (SELECT *, pairs > 0 AND similar_pairs / pairs > {R.TEMPLATE_SHARE} AS is_bot
                              FROM read_parquet('{res_dir}/*.parquet') ORDER BY account, day) TO '{tp}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        runner.commit(tp, tad_final)
        runner.write_stats(tad_final, step=STEP + ":template", inputs=template_inputs, code_sha=t_sha, rules_sha=t_sha,
                           con=con, counts=agg, checks=t_checks,
                           extra={"platform": platform, "seconds": round(time.time() - t0, 1), "workers": workers})
        import shutil
        shutil.rmtree(shard_dir)
        shutil.rmtree(res_dir)
    if only_template:
        con.close()
        log(f"{platform}: template result written; the other rules run with the full pass")
        return 0
    con.execute(f"CREATE TEMP TABLE TAD AS SELECT * FROM read_parquet('{tad_final}')")
    n_tad, n_tad_bot, n_ad_posts = con.execute(
        f"SELECT count(*), coalesce(count_if(is_bot), 0), (SELECT count(DISTINCT (account, day)) FROM {P} WHERE account IS NOT NULL) FROM TAD"
    ).fetchone()
    check(n_tad == n_ad_posts, f"template: one result per account-day ({n_tad:,})", checks)
    counts["template"] = {"account_days": int(n_tad), "bot_account_days": int(n_tad_bot)}
    log(f"{platform}: template rule: {n_tad_bot:,} of {n_tad:,} account-days flagged, {time.time() - t0:.0f} s")

    # ---- post-level rules ---------------------------------------------------
    kw = R.keyword_criteria("own_dec", "n_hashtags", "n_tickers", "name_has_bot" if spec["name_has"] else None)
    kw_sel = ", ".join(f"{e} AS {n}" for n, e in kw)
    kw_sum = " + ".join(f"CAST({n} AS INTEGER)" for n, _ in kw)
    reddit_cols = ""
    if platform == "reddit":
        self_id = " OR ".join(f"contains(lower(own_dec), '{p}')" for p in R.SELF_ID_PHRASES)
        reddit_cols = (f", coalesce(is_automoderator, FALSE) AS bot_automod, ({self_id}) AS bot_self_id, "
                       "coalesce(name_bot_token, FALSE) AS bot_name")
    con.execute(f"""CREATE TEMP TABLE PL AS
        SELECT post_id, account, day, {kw_sel}, {R.match('own_dec', R.INVITE_RX)} AS invite_link{reddit_cols}
        FROM {P}""")
    con.execute("ALTER TABLE PL ADD COLUMN n_kw_criteria INTEGER")
    con.execute(f"UPDATE PL SET n_kw_criteria = {kw_sum}")
    log(f"{platform}: post-level rules done in {time.time() - t0:.0f} s")

    # ---- campaign rule (Chen & Subramanian 2018), exactly as prototyped 25 Sep --
    a, w, alpha, beta = R.CAMPAIGN_MIN_ACCOUNTS, R.CAMPAIGN_WINDOW, R.CAMPAIGN_ALPHA, R.CAMPAIGN_BETA
    con.execute(f"CREATE TEMP TABLE CP AS SELECT account, day, fp, rk FROM {P} WHERE account IS NOT NULL")
    con.execute(f"""CREATE TEMP TABLE GS AS SELECT fp, day, count(DISTINCT account) AS n_accounts, count(*) AS n_posts
                    FROM CP WHERE fp IS NOT NULL GROUP BY fp, day HAVING count(DISTINCT account) >= {a}""")
    con.execute("CREATE TEMP TABLE GM AS SELECT DISTINCT c.fp AS gfp, c.day, c.account FROM CP c JOIN GS g USING (fp, day)")
    con.execute("CREATE TEMP TABLE AD AS SELECT c.account, c.day, max(c.rk) AS r FROM CP c "
                "JOIN (SELECT DISTINCT account, day FROM GM) x USING (account, day) GROUP BY 1, 2")
    con.execute(f"""CREATE TEMP TABLE W AS SELECT ad.account, ad.day, c.fp AS wfp
                    FROM AD ad CROSS JOIN range({w}) t(off) JOIN CP c ON c.account = ad.account AND c.rk = ad.r - t.off""")
    con.execute("CREATE TEMP TABLE WS AS SELECT account, day, count(*) AS ws FROM W GROUP BY 1, 2")
    con.execute("CREATE TEMP TABLE GW AS SELECT gm.gfp, gm.day, gm.account, w.wfp FROM GM gm JOIN W w USING (account, day)")
    con.execute(f"""CREATE TEMP TABLE CSET AS SELECT gfp, day, wfp FROM GW WHERE wfp IS NOT NULL
                    GROUP BY 1, 2, 3 HAVING count(DISTINCT account) >= {alpha}""")
    con.execute("""CREATE TEMP TABLE OV AS
        SELECT gw.gfp, gw.day, gw.account, count(c.wfp) AS hits, any_value(ws.ws) AS ws,
               count(c.wfp) / any_value(ws.ws) AS overlap
        FROM GW gw LEFT JOIN CSET c ON c.gfp = gw.gfp AND c.day = gw.day AND c.wfp = gw.wfp
        JOIN WS ws ON ws.account = gw.account AND ws.day = gw.day
        GROUP BY 1, 2, 3""")
    con.execute(f"""CREATE TEMP TABLE CAD AS SELECT account, day, max(overlap) AS max_overlap, count(*) AS n_groups,
                           max(overlap) >= {beta} AS is_bot FROM OV GROUP BY 1, 2""")
    g_n, g_posts, m_n, cad_n, cad_bot = con.execute(
        "SELECT (SELECT count(*) FROM GS), (SELECT coalesce(sum(n_posts), 0) FROM GS), (SELECT count(*) FROM GM), "
        "(SELECT count(*) FROM CAD), (SELECT coalesce(count_if(is_bot), 0) FROM CAD)").fetchone()
    counts["campaign"] = {"groups": int(g_n), "group_posts": int(g_posts), "memberships": int(m_n),
                          "member_account_days": int(cad_n), "bot_account_days": int(cad_bot)}
    log(f"{platform}: campaign rule: {g_n:,} groups, {cad_bot:,} of {cad_n:,} member account-days flagged, "
        f"{time.time() - t0:.0f} s")

    # ---- combine ------------------------------------------------------------------
    excl = ["bot_keyword", "bot_campaign", "bot_template"] + (["bot_automod", "bot_self_id", "bot_name"] if platform == "reddit" else [])
    tmp = runner.tmp_path(final)
    con.execute(f"""COPY (
        SELECT pl.*, pl.n_kw_criteria >= {R.KW_MIN_CRITERIA} AS bot_keyword,
               coalesce(c.is_bot, FALSE) AS bot_campaign, coalesce(t.is_bot, FALSE) AS bot_template
        FROM PL pl
        LEFT JOIN CAD c ON c.account = pl.account AND c.day = pl.day
        LEFT JOIN TAD t ON t.account = pl.account AND t.day = pl.day
        ORDER BY pl.day, pl.post_id) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    con.execute(f"CREATE TEMP VIEW F AS SELECT *, ({' OR '.join(excl)}) AS bot_any FROM read_parquet('{tmp}')")
    n_f, n_fid = con.execute("SELECT count(*), count(DISTINCT post_id) FROM F").fetchone()
    check(n_f == n_posts and n_fid == n_posts, "one flag row per post", checks)
    nulls = con.execute("SELECT count_if(" + " OR ".join(f"{c} IS NULL" for c in excl + ["invite_link"]) + ") FROM F").fetchone()[0]
    check(nulls == 0, "no NULL flag", checks)
    flag_counts = con.execute("SELECT " + ", ".join(f"count_if({c})" for c in excl + ["invite_link", "bot_any"]) + " FROM F").fetchone()
    counts["flags"] = dict(zip(excl + ["invite_link", "bot_any"], (int(v) for v in flag_counts)))
    only = {}
    for c in excl:
        others = " OR ".join(o for o in excl if o != c) or "FALSE"
        only[c] = int(con.execute(f"SELECT count_if({c} AND NOT ({others})) FROM F").fetchone()[0])
    counts["flag_only_this_rule"] = only
    counts["keyword_criteria"] = dict(zip([n for n, _ in kw], (int(v) for v in con.execute(
        "SELECT " + ", ".join(f"count_if({n})" for n, _ in kw) + " FROM F").fetchone())))
    # the flag table itself: bot_any is stored so step 4 needs no rule logic
    final_tmp2 = runner.tmp_path(out_dir / "post_flags_any.parquet")
    con.execute(f"COPY (SELECT * FROM F) TO '{final_tmp2}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    for name, rel in (("campaign_groups", "GS"), ("campaign_account_days", "CAD")):
        p = out_dir / f"{name}.parquet"
        tp = runner.tmp_path(p)
        con.execute(f"COPY (SELECT * FROM {rel}) TO '{tp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        runner.commit(tp, p)
        runner.write_stats(p, step=STEP, inputs=inputs, code_sha=CODE_SHA, rules_sha=RULES_SHA, con=con,
                           counts={"rows": int(con.execute(f"SELECT count(*) FROM {rel}").fetchone()[0])}, checks=[])
    os.remove(tmp)
    runner.commit(final_tmp2, final)
    runner.write_stats(final, step=STEP, inputs=inputs, code_sha=CODE_SHA, rules_sha=RULES_SHA, con=con,
                       counts=counts, checks=checks,
                       extra={"platform": platform, "seconds": round(time.time() - t0, 1), "workers": workers})
    con.close()
    log(f"{platform}: bot pass done in {time.time() - t0:.0f} s: " + json.dumps(counts["flags"]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("platform", choices=sorted(SPECS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--only-template", action="store_true",
                    help="compute and cache only the template rule (it does not depend on the StockTwits ticker list)")
    args = ap.parse_args(argv)
    log(f"{STEP} {args.platform}: code {CODE_SHA}, rules {RULES_SHA}")
    return run(args.platform, args.workers, args.only_template)


if __name__ == "__main__":
    raise SystemExit(main())
