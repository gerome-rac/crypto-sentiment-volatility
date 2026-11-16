#!/usr/bin/env python3
"""
09b_medrv_cj.py -- MedRV-based C/J split for the benchmark arms (provisional; written 7 Oct 2026,
after the first real model run, at Gerome's request). Adds no change to step 09 or its output.

Output: data/pipeline/panel/volatility_medrv_daily.parquet (coin x UTC day), columns
    rv, rv_obs                      as in volatility_daily (check: identical)
    medrv_obs, medrq_obs            MedRV / MedRQ of ADS (2012) on the 5-minute intervals untouched by an outage
                                    (library entry R04 with outage = 'split'; constants from the ADS 2009 working
                                    paper, flagged [FT-v] in cj_methods.py -- still to be checked against ADS 2012)
    z_med, jump_med                 max-adjusted ratio statistic with v = 0.96 and its decision at alpha = 0.999
    j_med, c_med                    tested split (arm A50): J = 1{Z > z} max(RV_obs - MedRV, 0), C = RV - J
    j_medu, c_medu                  untested split: J = max(RV_obs - MedRV, 0) on every day, C = RV - J
Usage (from code/): python pipeline/09b_medrv_cj.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import cj_methods as cjm  # noqa: E402
from pipeline import config, runner  # noqa: E402
from pipeline.runner import check, log  # noqa: E402

STEP = "09b_medrv_cj"
CODE_SHA = runner.code_fingerprint(__file__, cjm.__file__)
GRID_STEP, ALPHA, OUTAGE_MODE, OUTAGE_RULE, OUTAGE_MIN_RUN, DROP_RUN = 5, 0.999, "split", "script", 5, 20
PAIRS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT", "XRP": "XRPUSDT", "DOGE": "DOGEUSDT"}
RAW = config.RAW_DIR / "binance_spot_1m"
VOL = config.OUT_DIR / "panel" / "volatility_daily.parquet"
FINAL = config.OUT_DIR / "panel" / "volatility_medrv_daily.parquet"


def main() -> int:
    t0 = time.time(); checks = []
    check(abs(cjm.V_MED - 0.96) < 1e-12, "MedRV ratio-test constant v = 0.96", checks)
    panels = cjm.load_binance_1m(str(RAW), pairs=tuple(PAIRS.values()), start=config.WINDOW_START, end=config.WINDOW_END,
                                 outage_rule=OUTAGE_RULE, min_run=OUTAGE_MIN_RUN)
    vol = pd.read_parquet(VOL)
    frames = []
    for coin, pair in PAIRS.items():
        d = panels[pair]
        out = cjm.run("R04", d, grid_step=GRID_STEP, alpha=ALPHA, outage=OUTAGE_MODE, drop_run=DROP_RUN)
        rv = out["rv"].to_numpy(float); jump = out["jump"].to_numpy(bool)
        x = out["x_nn"].to_numpy(float); q = out["q_nn"].to_numpy(float)
        v = vol[vol.coin == coin].sort_values("day_utc")
        check(np.allclose(rv, v["rv"].to_numpy(float), rtol=1e-12, atol=1e-12), f"{coin}: rv identical to volatility_daily", checks)
        rv_obs = v["rv_obs"].to_numpy(float)
        j_raw = np.maximum(rv_obs - x, 0.0)
        check(np.allclose(np.where(jump, j_raw, 0.0), out["j"].to_numpy(float), rtol=1e-12, atol=1e-12),
              f"{coin}: tested J equals the library's R04 J (RV_obs basis)", checks)
        f = pd.DataFrame({"coin": coin, "day_utc": pd.to_datetime(out["day"]).dt.date, "rv": rv, "rv_obs": rv_obs,
                          "medrv_obs": x, "medrq_obs": q, "z_med": out["stat"].to_numpy(float), "jump_med": jump,
                          "j_med": np.where(jump, j_raw, 0.0), "j_medu": j_raw})
        f["c_med"] = f.rv - f.j_med; f["c_medu"] = f.rv - f.j_medu
        frames.append(f)
        log(f"{coin}: jump days {int(jump.sum())} ({jump.mean():.2%}); untested J > 0 on {(j_raw > 0).mean():.2%} of days")
    df = pd.concat(frames, ignore_index=True)
    num = ["rv", "medrv_obs", "c_med", "j_med", "c_medu", "j_medu"]
    check(len(df) == 5 * config.N_DAYS and np.isfinite(df[num].to_numpy()).all(), "6,400 finite rows", checks)
    check(bool((df.c_med > 0).all() and (df.c_medu > 0).all()), "c_med, c_medu > 0", checks)
    tmp = FINAL.with_suffix(".parquet.tmp"); df.to_parquet(tmp, index=False); tmp.replace(FINAL)
    runner.write_stats(FINAL, step=STEP, inputs=[{"path": str(VOL)}], code_sha=CODE_SHA, rules_sha="R04-provisional",
                       con=None, counts={"rows": len(df), "seconds": round(time.time() - t0, 1),
                       "jump_rate_med": {c: round(float(g.jump_med.mean()), 4) for c, g in df.groupby("coin")}}, checks=checks)
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
