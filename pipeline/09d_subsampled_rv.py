#!/usr/bin/env python3
"""
09d_subsampled_rv.py -- step 9d: subsampled 5-minute RV per coin and UTC day, the alternative evaluation target of
robustness arm A62 (register A62: "evaluation target: subsampled 5-min RV (average over the five 1-min offsets)",
evaluation only, "same for every model variant"; MASTERPLAN_NOV15.md section 7, Decisions of 6 Oct 2026 (1):
"only a far, lowest-priority robustness arm as an alternative evaluation target, applied the same way to every model
variant"; Patton 2011). Written 28 Oct 2026 (engine-arms track). Nothing here enters A00.

Input    data/raw/binance_spot_1m/<PAIR>/<PAIR>-1m-YYYY-MM.zip (Binance spot 1-minute klines, read-only), and
         data/pipeline/panel/volatility_daily.parquet (only to check that grid 0 reproduces the main RV).
Output   data/pipeline/panel/volatility_subsampled_daily.parquet (a new name; the step refuses to overwrite it),
         one row per coin and UTC day (5 x 1,280 = 6,400 rows), with its .stats.json.

Construction: the grid and outage rules of 09_volatility.py, unchanged, through the same library
(pipeline/cj_methods.py, unchanged; entry RV02 "Subsampled 5-min RV over the 1-minute offsets (RV5_ss)").
  - Prices: previous tick (a missing minute carries the previous close); library loader with outage_rule 'script'
    and a 5-minute minimum run, as 09_volatility.py; returns 100 * diff(log price).
  - Grids: five 5-minute grids starting at minutes 0, 1, 2, 3, 4 after 00:00 UTC. Grid 0 is the main base grid
    (288 returns, the first from the previous day's 23:59 close).
  - Outage days: RV is the full-day RV over all returns of each grid, so the gap enters as one catch-up return, as
    in the main target (MASTERPLAN 5-6 Oct (6): "RV stays the full-day RV"). The split concerns BV, TQ and the jump
    test only, which this step does not compute.
  - Edge rule: OPEN (library choice O3; volatility_report_20261007.md section 6 point 3;
    planning/open_points_panel_build_20261007.md section D). A grid shifted by s = 1..4 minutes holds 287 full
    5-minute returns inside the UTC day plus an s-minute piece after midnight and a (5 - s)-minute piece before the
    next midnight. All three published rules are written, one column each; none is chosen here:
        rv5_ss_E1   288 * sum_s RV_s / sum_s N_s: pooled squared returns rescaled by return counts (MFE toolbox,
                    realized_variance code)
        rv5_ss_E2   mean_s (RV_s + the two short edge pieces): every grid covers 00:00-24:00 (Patton & Sheppard
                    2015, p. 686 analogue)
        rv5_ss_E3   mean_s (RV_s * 288 / N_s): each grid scaled to the full day (Andersen, Dobrev & Schaumburg 2012,
                    WP p. 12)
        rv5_base    grid 0 alone; equal to volatility_daily.rv on every coin-day (checked)
    RV_s = sum of the squared full-interval returns of grid s, N_s its number of returns (288 for s = 0, else 287).

Columns: coin, day (UTC calendar day, naive midnight), rv5_ss_E1, rv5_ss_E2, rv5_ss_E3, rv5_base, n_returns_all
(returns over the five grids, 1,436). Only new columns, so the table can be passed to the engine as an extra table
(python -m model.run ... --extra-table data/pipeline/panel/volatility_subsampled_daily.parquet --eval-target <column>).

Subcommand (from code/):  python pipeline/09d_subsampled_rv.py build
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import cj_methods as cjm  # noqa: E402
from pipeline import config, runner  # noqa: E402
from pipeline.runner import check, log  # noqa: E402

STEP = "09d_subsampled_rv"
CODE_SHA = runner.code_fingerprint(__file__, cjm.__file__)
RULES_SHA = "cj_methods-" + runner.file_md5(cjm.__file__)[:16]

# ---- construction parameters (one name each; grid and outage settings as 09_volatility.py) -------------------
APPROACH = "RV02"              # library entry: subsampled 5-minute RV over the 1-minute offsets
GRID_STEP = 5                  # minutes; 288 returns on grid 0
OFFSETS = "all"                # grids starting at minutes 0..4
OUTAGE_MODE = "split"          # as 09_volatility.py; RV02's full-day RV does not depend on it (checked below)
OUTAGE_RULE = "script"         # outage-minute mask of the library loader, as 09_volatility.py
OUTAGE_MIN_RUN = 5             # as 09_volatility.py
EDGE_RULES = ("E1", "E2", "E3")   # all three; the choice is open (O3)
LIBRARY_MD5 = "9abc2fb6048dc712f9c6f06edd223c80"   # the verified library copy named in 09_volatility.py
PAIRS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT", "XRP": "XRPUSDT", "DOGE": "DOGEUSDT"}
RAW = config.RAW_DIR / "binance_spot_1m"
MAIN = config.OUT_DIR / "panel" / "volatility_daily.parquet"
FINAL = config.OUT_DIR / "panel" / "volatility_subsampled_daily.parquet"
N0 = 1440 // GRID_STEP


def build() -> None:
    t_start = time.time()
    checks: list = []
    check(not FINAL.exists(), f"{FINAL} does not exist yet (new outputs get new names; never overwritten)", checks)
    check(runner.file_md5(cjm.__file__) == LIBRARY_MD5, "C/J library is the verified, unchanged copy", checks)
    check(GRID_STEP == 5 and N0 == 288, "grid 5 minutes, 288 returns on grid 0", checks)
    t0 = pd.Timestamp(config.WINDOW_START, tz="UTC")
    check(str(t0.tz) == "UTC", "window start is a UTC timestamp", checks)
    n_days = config.N_DAYS

    main = pd.read_parquet(MAIN)
    main["day"] = pd.to_datetime(main["day_utc"])
    check(len(main) == 5 * n_days and not main.duplicated(["coin", "day"]).any(), "main table: 6,400 unique coin-days",
          checks)

    log("loading 1-minute klines (library loader, read-only)")
    panels = cjm.load_binance_1m(str(RAW), pairs=tuple(PAIRS.values()), start=config.WINDOW_START,
                                 end=config.WINDOW_END, outage_rule=OUTAGE_RULE, min_run=OUTAGE_MIN_RUN)
    zips = sorted(str(p) for pr in PAIRS.values() for p in (RAW / pr).glob(f"{pr}-1m-*.zip"))
    check(len(zips) == 5 * 42, f"210 monthly zip files (got {len(zips)})", checks)
    inputs = runner.describe_inputs(zips + [str(MAIN)])

    frames = []
    for coin, pair in PAIRS.items():
        d = panels[pair]
        check(d.n_days == n_days and d.contiguous, f"{coin}: {n_days} consecutive UTC days", checks)
        res = {}
        for e in EDGE_RULES:
            out = cjm.run(APPROACH, d, grid_step=GRID_STEP, outage=OUTAGE_MODE, edge_rule=e, offsets=OFFSETS)
            res[e] = out
        o1 = res["E1"]
        for e in EDGE_RULES[1:]:
            check(bool(np.array_equal(res[e]["day"].to_numpy(), o1["day"].to_numpy())), f"{coin}: same days, {e} vs E1",
                  checks)
            check(bool(np.array_equal(res[e]["rv0"].to_numpy(), o1["rv0"].to_numpy())), f"{coin}: same grid-0 RV, {e}",
                  checks)
        # RV02's full-day RV must not depend on the outage mode (the split only concerns BV/TQ/test)
        cf = cjm.run(APPROACH, d, grid_step=GRID_STEP, outage="none", edge_rule="E1", offsets=OFFSETS)
        check(bool(np.array_equal(cf["rv"].to_numpy(), o1["rv"].to_numpy())), f"{coin}: E1 RV equal under outage "
              "'split' and 'none' (full-day RV, catch-up return included)", checks)
        # n_used (split mode) counts the returns untouched by an outage: all 1,436 (288 + 4 x 287) on days without an
        # outage, fewer on outage days. The RV itself always uses all 1,436 returns (checked above against 'none').
        n_obs = o1["n_used"].to_numpy(float)
        out_day = o1["outage_min"].to_numpy() > 0
        n_all = N0 + 4 * (N0 - 1)
        check(bool(np.all(n_obs[~out_day] == n_all) and np.all(n_obs[out_day] < n_all)),
              f"{coin}: 1,436 returns over the five grids, all untouched on days without an outage", checks)
        days = pd.to_datetime(o1["day"])
        frames.append(pd.DataFrame({
            "coin": coin, "day": days.dt.normalize().astype("datetime64[ns]"),
            "rv5_ss_E1": o1["rv"].to_numpy(float), "rv5_ss_E2": res["E2"]["rv"].to_numpy(float),
            "rv5_ss_E3": res["E3"]["rv"].to_numpy(float), "rv5_base": o1["rv0"].to_numpy(float),
            "n_returns_all": np.full(len(o1), n_all, np.int64),
        }))
    df = pd.concat(frames, ignore_index=True)

    # ---- invariants ---------------------------------------------------------------------------------------------
    cal = pd.date_range(config.WINDOW_START, config.WINDOW_END, freq="D")
    check(len(df) == 5 * n_days, f"{len(df)} rows = 6,400", checks)
    check(all(np.array_equal(g["day"].to_numpy(), cal.to_numpy()) for _, g in df.groupby("coin")),
          "complete UTC calendar per coin", checks)
    check(not df.duplicated(["coin", "day"]).any(), "one row per coin and day", checks)
    num = ["rv5_ss_E1", "rv5_ss_E2", "rv5_ss_E3", "rv5_base"]
    check(bool(np.isfinite(df[num].to_numpy()).all() and (df[num].to_numpy() > 0).all()), "all RV columns finite and > 0",
          checks)
    m = df.merge(main[["coin", "day", "rv", "outage_flag"]], on=["coin", "day"], how="left", validate="one_to_one")
    check(int(m["rv"].isna().sum()) == 0, "every coin-day found in volatility_daily", checks)
    rel = np.abs(m["rv5_base"].to_numpy() - m["rv"].to_numpy()) / m["rv"].to_numpy()
    check(bool(np.max(rel) <= 1e-12), f"grid 0 reproduces volatility_daily.rv on all 6,400 coin-days "
          f"(max relative difference {np.max(rel):.2e})", checks)
    n_out = int(m["outage_flag"].astype(bool).sum())
    check(n_out == 90, "90 outage coin-days in volatility_daily (their RV keeps the catch-up return, as the main RV)",
          checks)
    # only new columns: an extra table must not repeat panel columns (model.panel.load_tables refuses overlaps)
    check(set(df.columns) == {"coin", "day", "rv5_ss_E1", "rv5_ss_E2", "rv5_ss_E3", "rv5_base", "n_returns_all"},
          "output holds only coin, day and the new RV columns", checks)

    tmp = runner.tmp_path(FINAL)
    df.to_parquet(tmp, index=False)
    back = pd.read_parquet(tmp)
    check(back.shape == df.shape and all(np.array_equal(back[c].to_numpy(), df[c].to_numpy()) for c in num),
          "re-read equals the table", checks)
    check(not FINAL.exists(), "output still absent before the move (no overwrite)", checks)
    runner.commit(tmp, FINAL)

    def _rel(a, b):
        return float(np.max(np.abs(df[a] / df[b] - 1.0)))
    counts = {"rows": int(len(df)), "outage_coin_days": n_out,
              "max_rel_diff_between_edge_rules": {"E2_vs_E1": _rel("rv5_ss_E2", "rv5_ss_E1"),
                                                  "E3_vs_E1": _rel("rv5_ss_E3", "rv5_ss_E1"),
                                                  "E3_vs_E2": _rel("rv5_ss_E3", "rv5_ss_E2")}}
    params = dict(APPROACH=APPROACH, GRID_STEP=GRID_STEP, OFFSETS=OFFSETS, OUTAGE_MODE=OUTAGE_MODE,
                  OUTAGE_RULE=OUTAGE_RULE, OUTAGE_MIN_RUN=OUTAGE_MIN_RUN, EDGE_RULES=list(EDGE_RULES),
                  edge_rule_status="open (library choice O3); all three written, none chosen",
                  library_version=cjm.__version__, library_md5=runner.file_md5(cjm.__file__))
    runner.write_stats(FINAL, step=STEP, inputs=inputs, code_sha=CODE_SHA, rules_sha=RULES_SHA, con=None,
                       counts=counts, checks=checks, extra={"parameters": params, "code_md5": runner.file_md5(__file__)})
    log(f"wrote {FINAL} ({len(df)} rows, md5 {runner.file_md5(FINAL)}) in {time.time() - t_start:.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    a = ap.parse_args()
    if a.cmd == "build":
        build()


if __name__ == "__main__":
    main()
