#!/usr/bin/env python3
"""
10b_text_precision.py -- step 10b: D16 precision-weighted daily means of the text variables.

Rule (clarification of D16, confirmed by Gerome on 7 Oct 2026; planning/pre_analysis_note_D8.md section 7):
thin days are precision-weighted, never dropped. For every platform p, series s (coin or MARKET), variable v
(the six EmTract classes and the FinBERT polarity) and UTC day t with at least one post:

    level  = mean of the platform's raw daily means of v over the available days t-30 .. t-1
             (no minimum count; if no day in that span has posts, the raw mean is kept)
    value  = level + R(n) * (raw mean - level)

R(n) = sigma2_b / (sigma2_b + sigma2_w / n) is the D16 reliability of the day's mean (column rel_<v> of
text_daily.parquet, sigma2 fixed from the pre-forecast year 30 Aug 2019 - 29 Aug 2020; R = 0 in the cells
whose between-day variance is not positive). This is regression calibration for a regressor measured with
classical error: the best linear prediction of the true daily mean given n posts.
The pooled index then follows D15 unchanged: the equal-weighted mean of the available platforms' values.
Within-day SDs are not adjusted; squares are formed by the engine from the adjusted means.

Input   data/pipeline/panel/text_daily.parquet (variant 'main', platform rows), read-only.
Output  data/pipeline/panel/text_precision_daily.parquet (+ .stats.json): series, day, mean_<v>_pw (pooled),
        mean_<v>_raw (pooled, = text_daily POOLED rows), n_platforms_<v>.
    python pipeline/10b_text_precision.py   (from code/)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "..", "data", "pipeline", "panel")
INP = os.path.join(PANEL, "text_daily.parquet")
OUT = os.path.join(PANEL, "text_precision_daily.parquet")
VARS = ["happy", "sad", "anger", "disgust", "surprise", "fear", "polarity"]
LEVEL_DAYS = 30            # own choice (flagged): span of the recent level, previous days only
FIRST, LAST = pd.Timestamp("2019-08-01"), pd.Timestamp("2023-01-31")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main() -> int:
    checks = []
    def check(c, m):
        checks.append({"check": m, "passed": bool(c)})
        if not c:
            raise AssertionError(m)
    t = pd.read_parquet(INP)
    t = t[t["variant"] == "main"].copy()
    t["day"] = pd.to_datetime(t["day"].astype(str))
    days = pd.date_range(FIRST, LAST, freq="D")
    plat = t[t["platform"] != "POOLED"]
    pool = t[t["platform"] == "POOLED"].set_index(["series", "day"]).sort_index()
    out = pd.MultiIndex.from_product([sorted(pool.index.get_level_values(0).unique()), days], names=["series", "day"]).to_frame(index=False)
    diag = []
    for v in VARS:
        vals, ones = {}, {}
        for (s, p), g in plat.groupby(["series", "platform"]):
            g = g.set_index("day").reindex(days)
            raw = g[f"mean_{v}"].astype(float)
            R = g[f"rel_{v}"].astype(float)
            level = raw.shift(1).rolling(LEVEL_DAYS, min_periods=1).mean()
            val = np.where(level.notna(), level + R * (raw - level), raw)
            val = pd.Series(val, index=days).where(raw.notna())
            one = raw.where(raw.notna())                     # R = 1 reproduces the raw mean
            vals[(s, p)] = val; ones[(s, p)] = one
            have = raw.notna()
            diag.append({"variable": v, "series": s, "platform": p, "days_with_posts": int(have.sum()),
                         "days_without_level": int((have & level.isna()).sum()),
                         "share_R_below_0_7": round(float((R[have] < 0.7).mean()), 4),
                         "median_R": round(float(R[have].median()), 4),
                         "mean_abs_adjustment": float((val[have] - raw[have]).abs().mean())})
        V = pd.DataFrame(vals); O = pd.DataFrame(ones)
        # pooled equal-weighted mean over the available platforms, per series
        pw_cols, chk_cols, npl = [], [], []
        for s in out["series"].unique():
            cols = [c for c in V.columns if c[0] == s]
            pw_cols.append(V[cols].mean(axis=1, skipna=True).rename(s))
            chk_cols.append(O[cols].mean(axis=1, skipna=True).rename(s))
            npl.append(O[cols].notna().sum(axis=1).rename(s))
        pw = pd.concat(pw_cols, axis=1).stack().rename(f"mean_{v}_pw")
        ck = pd.concat(chk_cols, axis=1).stack().rename("chk")
        nn = pd.concat(npl, axis=1).stack().rename(f"n_platforms_{v}")
        part = pd.concat([pw, ck, nn], axis=1)
        part.index.names = ["day", "series"]
        part = part.reset_index()
        stored = pool[f"mean_{v}"].rename("stored").reset_index()
        part = part.merge(stored, on=["series", "day"], how="left")
        m = part["stored"].notna() | part["chk"].notna()
        check(bool((part.loc[m, "stored"].isna() == part.loc[m, "chk"].isna()).all()), f"{v}: same available days as the stored pooled index")
        check(float((part["chk"] - part["stored"]).abs().max()) < 1e-12, f"{v}: R = 1 reproduces the stored pooled mean")
        check(bool((part[f"mean_{v}_pw"].isna() == part["stored"].isna()).all()), f"{v}: no pooled value lost or added")
        part = part.rename(columns={"stored": f"mean_{v}_raw"}).drop(columns="chk")
        out = out.merge(part, on=["series", "day"], how="left")
    cor = {f"{s}|{v}": round(float(out.loc[out.series == s, f"mean_{v}_pw"].corr(out.loc[out.series == s, f"mean_{v}_raw"])), 4)
           for s in out.series.unique() for v in VARS}
    tmp = OUT + ".tmp"; out.to_parquet(tmp, index=False); os.replace(tmp, OUT)
    stats = {"step": "10b_text_precision", "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "input_md5": md5(INP), "output_md5": md5(OUT), "script_md5": md5(os.path.abspath(__file__)),
             "LEVEL_DAYS": LEVEL_DAYS, "rows": int(len(out)), "checks": checks,
             "corr_pw_vs_raw_pooled": cor, "platform_diagnostics": diag}
    json.dump(stats, open(OUT + ".stats.json", "w"), indent=1, default=str)
    print(json.dumps({"rows": len(out), "md5": stats["output_md5"], "checks": len(checks)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
