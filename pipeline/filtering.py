"""Shared by the three filter scripts (02_filter_reddit.py, 02_filter_stocktwits.py, 02_filter_twitter.py).

The rules themselves are in rules.py. This module holds what the filter scripts share
besides the rules: the Parquet write settings, the counts of the computed columns that go
into every stats file, and the check that no computed column is NULL.
"""
from __future__ import annotations

from typing import Dict

from pipeline import rules as R

# Boolean evidence columns (n_ambiguous_terms is a count) and the Parquet write settings.
BOOL_EVIDENCE = [n for n in R.evidence_names() if n != "n_ambiguous_terms"]
PARQUET = "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"


def lit(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def evidence_counts(con, rel: str, where: str = "TRUE") -> Dict[str, int]:
    exprs = [f"count_if({n}) AS {n}" for n in BOOL_EVIDENCE]
    exprs += ["count_if(n_ambiguous_terms >= 1) AS ambiguous_ge1",
              f"count_if(n_ambiguous_terms >= {R.AMBIGUOUS_FLAG_MIN_TERMS}) AS ambiguous_ge2",
              "count(fingerprint) AS with_fingerprint", "count_if(is_url_only) AS url_only",
              "count_if(n_hashtags > 5) AS hashtags_gt5"]
    row = con.execute(f"SELECT {', '.join(exprs)} FROM {rel} WHERE {where}").fetchone()
    names = [e.split(" AS ")[-1] for e in exprs]
    return {k: int(v) for k, v in zip(names, row)}


def null_evidence(con, rel: str) -> int:
    cols = R.evidence_names() + ["n_chars", "n_tokens", "n_hashtags", "n_mentions", "cashtags", "is_url_only"]
    cond = " OR ".join(f"{n} IS NULL" for n in cols)
    return int(con.execute(f"SELECT count_if({cond}) FROM {rel}").fetchone()[0])
