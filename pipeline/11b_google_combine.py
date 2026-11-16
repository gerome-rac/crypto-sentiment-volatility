#!/usr/bin/env python3
"""
11b_google_combine.py -- Google Trends attention series (PROVISIONAL unless all five draws 01-05 are used).

Rules (frozen 7 Oct 2026: MASTERPLAN_NOV15.md section 7, 'Decisions of 6 Oct 2026' point (6);
planning/pre_analysis_note_D8.md section 4): Eichenauer, Indergand, Martinez & Sax (2022) procedure,
trendecon 0.3.2; option-B windows as fetched by 08_google_trends_fetch.R (29 daily 6-month windows
every 45 days, 7 weekly 5-year windows every 26 weeks, 2 monthly windows; 266 requests per draw);
'daily, weekly and monthly downloads, averaged over windows and draws, then temporal disaggregation
of the daily path to the weekly and monthly series'; the ts_gtrends_mwd() bug is avoided by
combining the frequencies as in proc_combine_freq(). Terms: Bitcoin, Ethereum, Dogecoin, XRP,
Binance (BNB, exchange attention, flagged), cryptocurrency and crypto (market level).
Agreement check: mean pairwise correlation of daily log changes across draws per term; a term
below 0.8 gets more draws (own threshold, flagged).

What this script does
  1. finds the complete draws (fetch_log.csv: 266 requests, all 'ok', every file present with the
     logged number of rows); incomplete draws are skipped and reported. Raw folders are only read.
  2. zero shares of the raw daily window values per term and draw ('<1' and '0').
  3. calls 11b_google_combine.R (env 'gtrends') once per WINDOW_COMBINE variant; each run combines
     every draw alone and all draws pooled (windows of all draws averaged by date, then td).
  4. builds 'mean_of_draws' = mean over the per-draw combined series (td per draw, then average).
  5. agreement check per term and variant; series-level evidence for the 'crypto' rule (no decision).
  6. writes data/pipeline/panel/google_daily_provisional.parquet (+ .stats.json), or the file named by --out.
     The column 'provisional' and the stats key PROVISIONAL are False only when all five draws of the
     frozen rule (01-05) are used (28 Oct 2026; MASTERPLAN 6 Oct (6): five draws on five days).

Named parameters (details the frozen text does not settle; see the task report):
  PRIMARY_WINDOW_COMBINE = 'mean_prop' (Deviation 2 of the D8 note, 7 Oct 2026): window mean, then the
                                    proportional Denton-Cholette step ('<1' and daily averages < 0.5 set to 0.5);
                                    'mean' = trendecon as written (additive 'fast', the frozen-literal series,
                                    kept for comparison: is_frozen_literal); 'chain' = windows chained first
  PRIMARY_DRAW_SET       = 'pooled' frozen text: averaged over windows AND draws, then td;
                                    'mean_of_draws' = td per draw, then averaged (task brief's order)
  FETCH_FROM, FETCH_TO   = 2019-07-01, 2023-02-28 (decided range); output kept 2019-08-01..2023-01-31
Not done here: division by the first-day value, logs/log changes for the model, HAR form, z-scoring
(assembly / in-window steps). log_value is stored only as a convenience (NaN where value <= 0).
Usage (from code/, env gtrends must provide Rscript):  python pipeline/11b_google_combine.py [--draws 01,02] [--out <path>]
  final series (28 Oct 2026): --draws 01,02,03,04,05 --out data/pipeline/panel/google_daily_final_20261028.parquet
  an explicit --out never overwrites an existing file; the default output path is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
RAW = CODE / "data" / "raw" / "google_trends"
R_SCRIPT = CODE / "pipeline" / "11b_google_combine.R"
OUT = CODE / "data" / "pipeline" / "panel" / "google_daily_provisional.parquet"

TERMS = {"Bitcoin": "BTC", "Ethereum": "ETH", "Dogecoin": "DOGE", "XRP": "XRP", "Binance": "BNB",
         "cryptocurrency": "MARKET", "crypto": "MARKET"}
N_REQUESTS = 266
FETCH_FROM, FETCH_TO = date(2019, 7, 1), date(2023, 2, 28)
FIRST_DAY, LAST_DAY, N_DAYS = date(2019, 8, 1), date(2023, 1, 31), 1280
WINDOW_COMBINE_VARIANTS = ["mean", "chain", "mean_prop"]
FROZEN_WINDOW_COMBINE = "mean"
PRIMARY_WINDOW_COMBINE = "mean_prop"   # Deviation 2
PRIMARY_DRAW_SET = "pooled"
AGREEMENT_THRESHOLD = 0.8
FINAL_DRAWS = ["01", "02", "03", "04", "05"]   # frozen rule: five draws on five days; PROVISIONAL unless all are used
CRYPTO_PERIOD = (date(2021, 11, 1), date(2022, 2, 28))

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


def draw_status(folder: Path) -> dict:
    log = folder / "fetch_log.csv"
    if not log.exists():
        return {"draw": folder.name, "complete": False, "reason": "no fetch_log.csv"}
    fl = pd.read_csv(log)
    ok = fl[fl["status"] == "ok"].drop_duplicates(["term", "freq", "start", "end"])
    missing, badrows = 0, 0
    for r in ok.itertuples():
        f = folder / f"{r.term}_{r.freq}_{r.start}_{r.end}.csv"
        if not f.exists():
            missing += 1
            continue
        if len(pd.read_csv(f, dtype=str)) != r.n_rows:
            badrows += 1
    complete = len(ok) == N_REQUESTS and missing == 0 and badrows == 0 and set(ok["term"]) == set(TERMS)
    return {"draw": folder.name, "complete": bool(complete), "requests_ok": int(len(ok)),
            "log_rows": int(len(fl)), "n_error_rows": int((fl["status"] != "ok").sum()),
            "files_missing": missing, "files_row_mismatch": badrows,
            "last_request_utc": str(fl["utc_time"].max())}


def zero_shares(draw: str) -> list[dict]:
    out = []
    for term in TERMS:
        fs = sorted((RAW / f"draw_{draw}").glob(f"{term}_d_*.csv"))
        x = pd.concat([pd.read_csv(f, dtype=str) for f in fs])
        x = x[x["keyword"] == term]
        z = x["hits"].isin(["<1", "0"])
        out.append({"draw": draw, "term": term, "daily_window_days": int(len(x)), "zero_or_lt1": int(z.sum()),
                    "share": round(float(z.mean()), 4), "n_daily_windows": len(fs),
                    "first_day": str(x["date"].min()), "last_day": str(x["date"].max())})
    return out


def run_r(draws: list[str], work: Path, rscript: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    series, diags = [], []
    for wc in WINDOW_COMBINE_VARIANTS:
        cmd = [rscript, str(R_SCRIPT), str(RAW), str(work), ",".join(draws), FETCH_FROM.isoformat(),
               FETCH_TO.isoformat(), ",".join(TERMS), wc]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stdout + r.stderr)
            raise RuntimeError(f"R combine failed ({wc})")
        series.append(pd.read_csv(work / f"google_combined_long_{wc}.csv", parse_dates=["time"]))
        diags.append(pd.read_csv(work / f"google_td_diagnostics_{wc}.csv"))
    return pd.concat(series, ignore_index=True), pd.concat(diags, ignore_index=True)


def dlog(s: pd.Series) -> pd.Series:
    v = s.where(s > 0)
    return np.log(v).diff()


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", default="auto", help="comma-separated draw ids, or 'auto' = every complete draw")
    ap.add_argument("--rscript", default=os.environ.get("RSCRIPT", "Rscript"))
    ap.add_argument("--out", default=str(OUT), help="output parquet (default: google_daily_provisional.parquet)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out != OUT.resolve():
        check(not out.exists() and not Path(str(out) + ".stats.json").exists(),
              f"an explicit --out never overwrites an existing file ({out.name})")

    status = [draw_status(p) for p in sorted(RAW.glob("draw_*")) if p.is_dir()]
    complete = [s["draw"].replace("draw_", "") for s in status if s["complete"]]
    draws = complete if a.draws == "auto" else a.draws.split(",")
    check(len(draws) >= 1, "at least one complete draw")
    check(all(d in complete for d in draws), f"every requested draw is complete ({draws})")
    zs = [z for d in draws for z in zero_shares(d)]
    provisional = not set(FINAL_DRAWS) <= set(draws)
    inputs = {}
    for d in draws:
        fs = sorted(p for p in (RAW / f"draw_{d}").glob("*.csv") if p.name != "fetch_log.csv")
        inputs[f"draw_{d}"] = {"fetch_log_md5": md5(RAW / f"draw_{d}" / "fetch_log.csv"), "n_window_files": len(fs),
                               "window_files_md5_digest": hashlib.md5("".join(f"{p.name}:{md5(p)}\n" for p in fs).encode()).hexdigest()}

    work = Path(tempfile.mkdtemp(prefix="gt_combine_"))
    try:
        long, diag = run_r(draws, work, a.rscript)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    long = long.rename(columns={"time": "day", "variant": "draw_set"})
    long["day"] = long["day"].dt.date
    check(set(long["day"].map(lambda d: d)) >= set(pd.date_range(FIRST_DAY, LAST_DAY).date), "combined daily path covers the sample")
    check(long["day"].min() == FETCH_FROM and long["day"].max() == FETCH_TO, "combined daily path spans the fetch range 2019-07-01..2023-02-28")
    check(bool((diag["max_abs_month_constraint"] < 1e-6).all()), "combined series reproduce the monthly means (td constraint)")

    # mean over per-draw series (td per draw, then average)
    per_draw = long[long["draw_set"].str.startswith("draw_")]
    mod = (per_draw.groupby(["term", "window_combine", "day"], as_index=False)
           .agg(value=("value", "mean"), daily_avg=("daily_avg", "mean"), n_windows=("n_windows", "sum"), wd=("wd", "mean")))
    mod["draw_set"] = "mean_of_draws"
    full = pd.concat([long, mod], ignore_index=True)
    full = full[(full["day"] >= FIRST_DAY) & (full["day"] <= LAST_DAY)].copy()
    full["series"] = full["term"].map(TERMS)
    full["value_nonpositive"] = full["value"] <= 0
    full["log_value"] = np.where(full["value"] > 0, np.log(full["value"].where(full["value"] > 0)), np.nan)
    full["is_frozen_literal"] = (full["window_combine"] == FROZEN_WINDOW_COMBINE) & (full["draw_set"] == PRIMARY_DRAW_SET)
    full["is_primary"] = (full["window_combine"] == PRIMARY_WINDOW_COMBINE) & (full["draw_set"] == PRIMARY_DRAW_SET)
    full["bnb_is_exchange_term"] = full["term"] == "Binance"
    full["provisional"] = provisional
    full["draws_used"] = ",".join(draws)
    full["n_draws"] = len(draws)
    full = full.rename(columns={"value": "gt_level", "log_value": "gt_log_level", "daily_avg": "daily_window_avg",
                                "n_windows": "n_daily_window_values", "wd": "gt_level_weekly_step"})
    cols = ["term", "series", "day", "window_combine", "draw_set", "is_frozen_literal", "is_primary", "gt_level", "gt_log_level",
            "value_nonpositive", "daily_window_avg", "n_daily_window_values", "gt_level_weekly_step",
            "bnb_is_exchange_term", "provisional", "n_draws", "draws_used"]
    full = full[cols].sort_values(["window_combine", "draw_set", "term", "day"]).reset_index(drop=True)
    n_sets = len(draws) + 1 + 1
    check(len(full) == len(TERMS) * N_DAYS * len(WINDOW_COMBINE_VARIANTS) * n_sets,
          "rows = 7 terms x 1,280 days x 3 window variants x (draws + pooled + mean_of_draws)")
    check(full.groupby(["term", "window_combine", "draw_set"])["day"].nunique().eq(N_DAYS).all(), "every series has 1,280 days")

    # nonpositive days per term / variant (log undefined)
    nonpos = (full.groupby(["window_combine", "draw_set", "term"])["value_nonpositive"].sum().astype(int)
              .reset_index().rename(columns={"value_nonpositive": "days_nonpositive"}))

    # agreement check across draws
    agree = []
    for (wc, term), g in full[full["draw_set"].str.startswith("draw_")].groupby(["window_combine", "term"]):
        w = g.pivot(index="day", columns="draw_set", values="gt_level").sort_index()
        dl = w.apply(dlog)
        pairs = list(itertools.combinations(w.columns, 2))
        cors = [dl[[p, q]].dropna().corr().iloc[0, 1] for p, q in pairs]
        nobs = [len(dl[[p, q]].dropna()) for p, q in pairs]
        mean_c = float(np.mean(cors)) if cors else np.nan
        agree.append({"window_combine": wc, "term": term, "n_draws": w.shape[1], "n_pairs": len(pairs),
                      "mean_pairwise_corr_dlog": None if np.isnan(mean_c) else round(mean_c, 4),
                      "min_obs_per_pair": min(nobs) if nobs else None,
                      "below_threshold": None if np.isnan(mean_c) else bool(mean_c < AGREEMENT_THRESHOLD)})

    # 'crypto' vs 'cryptocurrency': series-level evidence only
    ev = []
    for wc in WINDOW_COMBINE_VARIANTS:
        for ds in sorted(full["draw_set"].unique()):
            g = full[(full["window_combine"] == wc) & (full["draw_set"] == ds)]
            w = g.pivot(index="day", columns="term", values="gt_level").sort_index()
            c, cc = w["crypto"], w["cryptocurrency"]
            pos = (c > 0) & (cc > 0)
            lc, lcc = np.log(c.where(pos)), np.log(cc.where(pos))
            r = lc - lcc
            idx = pd.Index(w.index)
            inp = (idx >= CRYPTO_PERIOD[0]) & (idx <= CRYPTO_PERIOD[1])
            d1, d2 = lc.diff(), lcc.diff()
            ev.append({"window_combine": wc, "draw_set": ds,
                       "corr_level": round(float(c.corr(cc)), 4),
                       "corr_log_level": round(float(lc.corr(lcc)), 4),
                       "corr_dlog": round(float(d1.corr(d2)), 4),
                       "corr_dlog_in_period": round(float(d1[inp].corr(d2[inp])), 4),
                       "corr_dlog_outside": round(float(d1[~inp].corr(d2[~inp])), 4),
                       "log_ratio_mean_in_period": round(float(r[inp].mean()), 4),
                       "log_ratio_mean_outside": round(float(r[~inp].mean()), 4),
                       "log_ratio_sd_outside": round(float(r[~inp].std()), 4),
                       "log_ratio_shift_in_sd": round(float((r[inp].mean() - r[~inp].mean()) / r[~inp].std()), 3),
                       "days_both_positive": int(pos.sum())})
    # the same on the raw weekly window averages (no td involved)
    wk = {}
    for term in ["crypto", "cryptocurrency"]:
        fr = [pd.read_csv(f, dtype=str) for d in draws for f in sorted((RAW / f"draw_{d}").glob(f"{term}_w_*.csv"))]
        x = pd.concat(fr)
        x["v"] = pd.to_numeric(x["hits"].replace("<1", "0"))
        wk[term] = x.groupby("date")["v"].mean()
    wkd = pd.DataFrame(wk)
    wkd.index = pd.to_datetime(wkd.index).date
    wkd = wkd[(wkd.index >= FIRST_DAY) & (wkd.index <= LAST_DAY)]
    pos = (wkd > 0).all(axis=1)
    rr = np.log(wkd["crypto"][pos]) - np.log(wkd["cryptocurrency"][pos])
    inp = (rr.index >= CRYPTO_PERIOD[0]) & (rr.index <= CRYPTO_PERIOD[1])
    weekly_ev = {"source": "raw weekly windows averaged by date (draws used)", "n_weeks": int(len(wkd)),
                 "corr_level": round(float(wkd["crypto"].corr(wkd["cryptocurrency"])), 4),
                 "corr_dlog": round(float(np.log(wkd["crypto"][pos]).diff().corr(np.log(wkd["cryptocurrency"][pos]).diff())), 4),
                 "log_ratio_mean_in_period": round(float(rr[inp].mean()), 4),
                 "log_ratio_mean_outside": round(float(rr[~inp].mean()), 4),
                 "log_ratio_shift_in_sd": round(float((rr[inp].mean() - rr[~inp].mean()) / rr[~inp].std()), 3),
                 "peak_ratio_week": str(rr.idxmax()), "peak_log_ratio": round(float(rr.max()), 4)}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    full.to_parquet(tmp, index=False)
    check(len(pd.read_parquet(tmp)) == len(full), "written table re-read")
    os.replace(tmp, out)
    stats = {"step": "11b_google_combine", "PROVISIONAL": provisional,
             "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "python": sys.version.split()[0], "code_md5": {"py": md5(Path(__file__)), "R": md5(R_SCRIPT)},
             "parameters": {"PRIMARY_WINDOW_COMBINE": PRIMARY_WINDOW_COMBINE, "FROZEN_WINDOW_COMBINE": FROZEN_WINDOW_COMBINE, "PRIMARY_DRAW_SET": PRIMARY_DRAW_SET,
                            "FETCH_FROM": str(FETCH_FROM), "FETCH_TO": str(FETCH_TO),
                            "AGREEMENT_THRESHOLD": AGREEMENT_THRESHOLD, "CRYPTO_PERIOD": [str(x) for x in CRYPTO_PERIOD]},
             "draw_status": status, "draws_used": draws, "inputs": inputs, "raw_zero_shares": zs,
             "td_diagnostics": diag.to_dict(orient="records"), "days_nonpositive": nonpos.to_dict(orient="records"),
             "agreement": agree, "crypto_evidence_daily": ev, "crypto_evidence_weekly_raw": weekly_ev,
             "crypto_related_queries_rank": "not computed here: the draws fetch interest over time only (onlyInterest = TRUE in 08_google_trends_fetch.R); related queries are a separate fetch (08b_google_related_queries.R, 28 Oct 2026)",
             "outputs": [{"path": str(out.relative_to(CODE)) if out.is_relative_to(CODE) else str(out), "bytes": out.stat().st_size,
                          "md5": md5(out), "rows": len(full)}],
             "checks": CHECKS, "seconds": round(time.time() - t0, 1)}
    st = Path(str(out) + ".stats.json")
    tj = Path(str(st) + ".tmp")
    tj.write_text(json.dumps(stats, indent=1, default=str))
    os.replace(tj, st)
    print(json.dumps({"rows": len(full), "md5": stats["outputs"][0]["md5"], "draws": draws, "provisional": provisional,
                      "seconds": stats["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
