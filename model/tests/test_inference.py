"""In-sample inference: Driscoll-Kraay against a hand computation, the H4 bootstrap test."""
import numpy as np
import pandas as pd
import pytest

from model import har, inference, panel
from model.config import EngineConfig


def test_driscoll_kraay_hand():
    from linearmodels.panel import PanelOLS
    rng = np.random.default_rng(0)
    C, T = 4, 300
    f = rng.normal(size=T)
    x = rng.normal(size=(C, T)) + f
    e = rng.normal(size=(C, T)) + 0.5 * np.r_[0, f[:-1]]
    y = 0.5 * x + e + np.arange(C)[:, None]
    idx = pd.MultiIndex.from_product([range(C), pd.date_range("2020-01-01", periods=T)])
    r = PanelOLS(pd.Series(y.ravel(), idx), pd.DataFrame({"x": x.ravel()}, idx), entity_effects=True).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=5)
    xd, yd = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    b = (xd * yd).sum() / (xd ** 2).sum()
    h = (xd * (yd - b * xd)).sum(0)
    L = 5
    S = h @ h + sum(2 * (1 - l / (L + 1)) * h[l:] @ h[:-l] for l in range(1, L + 1))
    V = S / ((xd ** 2).sum()) ** 2 * (C * T) / (C * T - 1 - C)     # linearmodels' df scale
    assert r.params.iloc[0] == pytest.approx(b)
    assert r.std_errors.iloc[0] == pytest.approx(np.sqrt(V), rel=1e-10)


def _plant(sign_fear, sign_anger):
    def f(txt, att, rng):
        # informative fear/anger: shift the day-t probability with the next day's volatility shock
        u = txt["_u"].to_numpy()
        txt["emo_fear"] = np.clip(txt["emo_fear"] + sign_fear * 0.01 * u, 1e-4, None)
        txt["emo_anger"] = np.clip(txt["emo_anger"] + sign_anger * 0.01 * u, 1e-4, None)
    return f


@pytest.mark.parametrize("sf,sa,expect", [(1, -1, True), (0, 0, False)])
def test_h4_bootstrap(vol, sf, sa, expect):
    import conftest
    v = vol.copy()                      # COINS order, as the synthetic tables
    lv = np.log(v.rv)
    shock = (lv.groupby(v.coin).shift(-1) - lv.groupby(v.coin).transform(lambda s: s.rolling(30, 1).mean())).fillna(0)
    def plant(txt, att, rng):
        txt["_u"] = shock.to_numpy() / shock.std()
        _plant(sf, sa)(txt, att, rng)
        txt.drop(columns="_u", inplace=True)
    tabs = conftest.synth_tables(v, seed=4, plant=plant)
    specs = panel.load_variable_spec()
    cfg = EngineConfig(boot_reps=299)
    b = har.FeatureBuilder(panel.build_grid(tabs, specs), specs, cfg)
    ins = inference.ols_dk_nw(b, "R3", cfg)
    h4 = inference.h4_test(ins, b, cfg)
    assert h4["block_length"] >= 1
    if expect:
        assert h4["p_h4"] < 0.05 and (h4["table"].estimate > 0).all()
    else:
        assert h4["p_h4"] > 0.01
    assert set(ins["wald"]["cov"]) == {"DK", "NW"} and (ins["wald"].q == 54).all()


def test_dk_cov_within_matches_linearmodels():
    from linearmodels.panel import PanelOLS
    rng = np.random.default_rng(9)
    C, T, k = 5, 200, 3
    f = rng.normal(size=T)
    Z = rng.normal(size=(C * T, k)) + np.repeat(f, C)[:, None] * 0.5
    coin = np.tile(np.arange(C), T)
    day = np.repeat(np.arange(T), C)
    y = Z @ np.array([0.3, 0.0, -0.2]) + coin * 0.2 + rng.normal(size=C * T) + np.repeat(f, C)
    b, V = inference.dk_cov_within(Z, y, coin, day, 7)
    idx = pd.MultiIndex.from_arrays([coin, pd.date_range("2020-01-01", periods=T)[day]])
    r = PanelOLS(pd.Series(y, idx), pd.DataFrame(Z, idx), entity_effects=True).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=7)
    assert np.allclose(b, r.params.to_numpy())
    assert np.allclose(V, r.cov.to_numpy(), rtol=1e-8)
