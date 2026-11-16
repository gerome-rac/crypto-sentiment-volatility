"""Synthetic fixtures only (no real data): a log-HAR volatility panel for the five coins over the
1,280 sample days, and synthetic attention/text/control tables in the default spec's columns."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import config, panel  # noqa: E402


def synth_volatility(seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range(config.SAMPLE_FIRST_DAY, config.SAMPLE_LAST_DAY, freq="D")
    rows = []
    for ci, coin in enumerate(config.COINS):
        lv = np.zeros(len(days))
        lv[:30] = rng.normal(1.0, 0.5, 30)
        for t in range(30, len(days)):
            lv[t] = 0.3 + 0.35 * lv[t - 1] + 0.3 * lv[t - 7:t].mean() + 0.2 * lv[t - 30:t].mean() + rng.normal(0, 0.6)
        rv = np.exp(lv + 0.2 * ci)
        jump = rng.random(len(days)) < 0.1
        j = np.where(jump, rv * rng.uniform(0.1, 0.4, len(days)), 0.0)
        rows.append(pd.DataFrame(dict(coin=coin, day=days, rv=rv, c=rv - j, j=j)))
    return pd.concat(rows, ignore_index=True)


def synth_tables(vol, seed=1, plant=None):
    rng = np.random.default_rng(seed)
    days = pd.date_range(config.SAMPLE_FIRST_DAY, config.SAMPLE_LAST_DAY, freq="D")
    C = list(config.COINS)
    idx = pd.MultiIndex.from_product([C, days], names=["coin", "day"])
    n = len(idx)
    att = pd.DataFrame(index=idx)
    for nm in ["att_reddit", "att_stocktwits", "att_twitter", "att_news"]:
        att[nm] = rng.poisson(50, n).astype(float)
    for nm in ["att_wiki", "att_google"]:
        att[nm] = np.exp(rng.normal(0, 0.3, n))
    mk = pd.DataFrame(index=days)
    for nm in ["mkt_reddit", "mkt_stocktwits", "mkt_twitter"]:
        mk[nm] = rng.poisson(500, len(days)).astype(float)
    for nm in ["mkt_wiki", "mkt_google_cryptocurrency", "mkt_google_crypto"]:
        mk[nm] = np.exp(rng.normal(0, 0.3, len(days)))
    absent = np.zeros(len(days), bool)
    absent[rng.choice(len(days), 76, replace=False)] = True
    mk["twitter_absent"] = absent
    for nm in mk.columns:
        att[nm] = np.tile(mk[nm].to_numpy(), len(C))
    att.loc[att.twitter_absent, "att_twitter"] = 0.0
    att.loc[att.twitter_absent, "mkt_twitter"] = 0.0
    txt = pd.DataFrame(index=idx)
    txt["pol"] = rng.normal(0.1, 0.1, n)
    txt["pol_sd"] = np.abs(rng.normal(0.5, 0.05, n))
    E = rng.dirichlet([6, 1, 0.3, 0.3, 0.5, 0.4, 4], n)
    for j, e in enumerate(["happy", "sad", "anger", "disgust", "surprise", "fear"]):
        txt[f"emo_{e}"] = E[:, j]
        txt[f"emo_{e}_sd"] = np.abs(rng.normal(0.2, 0.02, n))
    ctl = pd.DataFrame({"day": days})
    for nm in ["ctrl_vix", "ctrl_move", "ctrl_term", "ctrl_baa10y", "ctrl_dgs10_chg", "ctrl_dxy_ret",
               "ctrl_gold_ret", "ctrl_epu", "ctrl_tether_news", "ctrl_tether_news_chg"]:
        ctl[nm] = rng.normal(0, 1, len(days))
    ctl["ctrl_weekend"] = (days.dayofweek >= 5).astype(float)
    if plant is not None:
        plant(txt, att, rng)
    return dict(volatility_daily=vol, text_daily=txt.reset_index(), attention_daily=att.reset_index(),
                controls_daily=ctl)


@pytest.fixture(scope="session")
def vol():
    return synth_volatility()


@pytest.fixture(scope="session")
def tables(vol):
    return synth_tables(vol)


@pytest.fixture(scope="session")
def specs():
    return panel.load_variable_spec()
