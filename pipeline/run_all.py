#!/usr/bin/env python3
"""run_all.py -- run the text pipeline in its fixed order (steps 01-05).

Every step skips an output that is already current (same inputs, code and rules, as
recorded in its stats file), so running this again repeats only what a change affects.
The run stops at the first failing step. The rule tests run first.

What each step needs
--------------------
01 reddit       the local copy of the Reddit archive (config.REDDIT_RAW_DIR); with
                --offline-manifest also its checked manifest (config.REDDIT_RAW_MANIFEST)
02 stocktwits   a local copy of StockTwits Stage A (--stocktwits-src), fetched with
                `rclone copy <remote>:stocktwits_parquet <dir>`
02 twitter      Twitter Stage A on Drive (fetched day by day by the step itself) and the
                capture ledgers (config.TWITTER_LEDGER_DIR)
03 - 05         only the outputs of the steps before them
Step 01 for StockTwits and Twitter is not part of this run: it ran once, on VMs, from
archives that are not stored locally (see the headers of 01_extract_stocktwits.py and
01_extract_twitter.py). Their outputs (Stage A) are the inputs of step 02.

Steps that this script does not run (added 2 Oct 2026): EmTract's stage 1 needs a second environment
(emtract-ref) and the model scoring a GPU, so they are run one subcommand at a time (README.md "Run
order"; each script's header lists its subcommands and the commands of the full run). `--list` prints
them after the steps above, in their run order:
04 dlt          pipeline/04_series_dlt.py           the DLT-Tweets corpus through the rules of 02-04
06 EmTract      pipeline/06_emotion_emtract.py      EmTract emotion scores (environments emtract-ref, thesis)
07 FinBERT      pipeline/07_polarity_finbert.py     FinBERT polarity (environment thesis)

Usage (from code/)
------------------
    python pipeline/run_all.py --list                     # print the steps and exit
    python pipeline/run_all.py                            # everything
    python pipeline/run_all.py --platforms reddit         # one platform
    python pipeline/run_all.py --from 03                  # steps 03-05, from the stored outputs of 02
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pipeline import config  # noqa: E402


def steps(args):
    """(number, name, command) in run order."""
    py = sys.executable
    step = lambda f: str(HERE / f)  # noqa: E731
    raw, run_dir = str(config.REDDIT_RAW_DIR), str(config.out_path("reddit", "extract", "_run"))
    out = []
    if "reddit" in args.platforms:
        verify = [py, step("01_extract_reddit.py"), "verify", "--raw", raw, "--run-dir", run_dir]
        if args.offline_manifest:
            verify += ["--manifest", str(config.REDDIT_RAW_MANIFEST)]
        out += [(1, "01 reddit verify", verify),
                (1, "01 reddit parse", [py, step("01_extract_reddit.py"), "parse", "--raw", raw, "--run-dir", run_dir,
                                        "--shard-dir", str(config.SCRATCH_DIR / "reddit_shards"), "--workers", "6"]),
                (1, "01 reddit finalize", [py, step("01_extract_reddit.py"), "finalize", "--run-dir", run_dir,
                                           "--out", str(config.out_path("reddit", "extract")), "--spill", str(config.SPILL_DIR)]),
                (2, "02 reddit", [py, step("02_filter_reddit.py")])]
    if "stocktwits" in args.platforms:
        out += [(2, "02 stocktwits", [py, step("02_filter_stocktwits.py"), "--src", args.stocktwits_src])]
    if "twitter" in args.platforms:
        out += [(2, "02 twitter coarse", [py, step("02_filter_twitter.py"), "coarse"]),
                (2, "02 twitter units", [py, step("02_filter_twitter.py"), "units"])]
    # the bot filter reads the StockTwits ticker list, so StockTwits goes first
    for p in ("stocktwits", "reddit", "twitter"):
        if p in args.platforms:
            out.append((3, f"03 {p}", [py, step("03_bot_filter.py"), p, "--workers", str(args.workers)]))
    for p in ("reddit", "stocktwits", "twitter"):
        if p in args.platforms:
            out.append((4, f"04 {p}", [py, step(f"04_series_{p}.py")]))
    out.append((5, "05 daily counts", [py, step("05_daily_counts.py")]))
    return [s for s in out if s[0] >= args.start]


SEPARATE = [  # (name, command): steps run one subcommand at a time, in run order (see the header)
    ("04 dlt", "pipeline/04_series_dlt.py tickers | units | profile | precheck"),
    ("06 emtract stage 1", "pipeline/06_emotion_emtract.py tokenize <platform>            [env emtract-ref]"),
    ("06 emtract stage 2", "pipeline/06_emotion_emtract.py pack | score | collect          [env thesis; score on a GPU]"),
    ("06 emtract merge", "pipeline/06_emotion_emtract.py merge ... ; text-key-index ; text-key-list   [env thesis]"),
    ("06 emtract dlt", "pipeline/06_emotion_emtract.py dlt clean | match | tokenize | verify | completeness | merge-keys | build | refscore"),
    ("06 emtract news", "pipeline/06_emotion_emtract.py news prepare | stage1 | score | refcheck"),
    ("07 finbert", "pipeline/07_polarity_finbert.py selfcheck | tokenize | pack | score | collect | merge   [env thesis; score on a GPU]"),
]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platforms", nargs="*", default=list(config.PLATFORMS), choices=list(config.PLATFORMS))
    ap.add_argument("--from", dest="start", type=int, default=1, choices=range(1, 6), metavar="STEP",
                    help="first step to run (1-5)")
    ap.add_argument("--stocktwits-src", default=str(config.SCRATCH_DIR / "stocktwits_parquet"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--offline-manifest", action="store_true",
                    help="verify the Reddit archive against the stored manifest instead of the bucket listing")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    plan = [(0, "tests", [sys.executable, str(HERE.parent / "tests" / "test_rules.py")])] + steps(args)
    if args.list:
        for _, name, cmd in plan:
            print(f"{name:22s} {' '.join(cmd[1:])}")
        print("\nrun separately (README.md; each script's header):")
        for name, cmd in SEPARATE:
            print(f"{name:22s} {cmd}")
        return 0
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    for _, name, cmd in plan:
        t0 = time.time()
        print(f"== {name}", flush=True)
        res = subprocess.run(cmd, cwd=str(HERE.parent))
        if res.returncode != 0:
            print(f"== {name} FAILED (exit {res.returncode}) after {time.time() - t0:.0f} s", flush=True)
            return res.returncode
        print(f"== {name} ok in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
