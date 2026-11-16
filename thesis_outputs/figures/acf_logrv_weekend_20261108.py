#!/usr/bin/env python
"""acf_logrv_weekend_20261108.py - sample ACF of daily log RV per coin, lags 1-100, before and after removing the
weekend effect (input for fig_acf_logrv_weekend_20261108.R).

Unadjusted: the ACF of log RV_{i,t} as in Figure A.1 (cj_fig_acf_logrv_20261022.pdf). Checked on 8 Nov 2026 against
the bar heights stored in that PDF: identical to 4 decimals for all 5 coins x 100 lags.
Adjusted: the ACF of the residuals of one pooled OLS regression of log RV_{i,t} on coin intercepts and a weekend
indicator W_t = 1 if day t is a Saturday or Sunday (UTC), with one common coefficient - the form in which the model
uses the weekend indicator (ctrl_weekend marks the forecast day t+1, i.e. W of the day whose RV is explained).
ACF: r_k = sum_{t=1}^{N-k} (x_t - xbar)(x_{t+k} - xbar) / sum_t (x_t - xbar)^2, N = 1,280 days per coin.

Usage: python acf_logrv_weekend_20261108.py <har_panel_final_20261028.parquet> <out.csv>
Writes <out.csv> (temp file -> check -> replace) with columns coin, lag, acf_raw, acf_adj, weekend_coef, n_days.
"""
import os
import sys

import numpy as np
import pandas as pd

panel, out = sys.argv[1:3]
COINS = ["BTC", "ETH", "BNB", "XRP", "DOGE"]
K = 100

p = pd.read_parquet(panel, columns=["coin", "day", "rv", "ctrl_weekend", "ctrl_weekend_origin"])
p = p.sort_values(["coin", "day"]).reset_index(drop=True)
assert set(p.coin) == set(COINS) and p.groupby("coin").size().eq(1280).all()
assert p.groupby("coin").day.apply(lambda d: (d.diff().dropna() == pd.Timedelta(days=1)).all()).all()
assert (p.rv > 0).all()
p["y"] = np.log(p.rv)
p["W"] = pd.to_datetime(p.day).dt.dayofweek.isin([5, 6]).astype(float)
assert (p.W == p.ctrl_weekend_origin).all()
nxt = p.groupby("coin").W.shift(-1)
assert (nxt.dropna() == p.ctrl_weekend[nxt.notna()]).all()          # the model's dummy is W of day t+1

X = pd.get_dummies(p.coin).astype(float)
X["W"] = p.W
beta = np.linalg.lstsq(X.to_numpy(), p.y.to_numpy(), rcond=None)[0]
p["e"] = p.y - X.to_numpy() @ beta


def acf(x, k_max=K):
    x = np.asarray(x, float) - np.mean(x)
    d = (x * x).sum()
    return np.array([(x[:-k] * x[k:]).sum() / d for k in range(1, k_max + 1)])


rows = []
for c in COINS:
    g = p[p.coin == c]
    rows.append(pd.DataFrame(dict(coin=c, lag=np.arange(1, K + 1), acf_raw=acf(g.y), acf_adj=acf(g.e),
                                  weekend_coef=beta[-1], n_days=len(g))))
df = pd.concat(rows, ignore_index=True)
assert len(df) == 5 * K and df[["acf_raw", "acf_adj"]].abs().le(1).all().all()

tmp = out + ".tmp"
df.to_csv(tmp, index=False, float_format="%.6f")
chk = pd.read_csv(tmp)
assert chk.shape == df.shape and np.allclose(chk.acf_adj, df.acf_adj, atol=1e-6)
os.replace(tmp, out)
print(f"common weekend coefficient {beta[-1]:.4f}; wrote {out}")
