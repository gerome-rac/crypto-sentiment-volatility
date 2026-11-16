"""Unit tests of the forecast-evaluation statistics against hand-computed or library values."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from model import tests_eval as te
from model.config import EngineConfig, hac_default_lags


def test_hac_default_lags():
    assert hac_default_lags(885) == 6
    assert hac_default_lags(1250) == 7


def test_hac_var_hand():
    e = np.array([1.0, -1.0, 2.0, 0.0])          # mean 0.5 -> demeaned 0.5,-1.5,1.5,-0.5
    d = e - e.mean()
    g0 = np.dot(d, d) / 4
    g1 = np.dot(d[1:], d[:-1]) / 4
    assert te.hac_var(e, 0) == pytest.approx(g0)
    assert te.hac_var(e, 1) == pytest.approx(g0 + 2 * 0.5 * g1)
    assert te.hac_var(e, 1, divisor="T-l") == pytest.approx(g0 + 2 * 0.5 * np.dot(d[1:], d[:-1]) / 3)


def test_clark_west_hand():
    y = np.array([1.0, 2.0, 0.5, 1.5, 1.0, 2.5])
    f1 = np.array([1.2, 1.5, 0.9, 1.1, 1.3, 1.9])
    f2 = np.array([1.1, 1.8, 0.6, 1.4, 1.2, 2.2])
    adj = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    t = adj.mean() / np.sqrt(np.var(adj) / len(adj))
    r = te.clark_west(y, f1, f2, lags=0)
    assert r["cw_stat"] == pytest.approx(t)
    assert r["p_one_sided"] == pytest.approx(1 - stats.norm.cdf(t))
    # identical forecasts: degenerate, never a rejection
    r0 = te.clark_west(y, f1, f1, lags=0)
    assert r0["degenerate"] and r0["p_one_sided"] == 1.0


def test_clark_west_panel_daymean():
    rng = np.random.default_rng(0)
    y, f1, f2 = rng.normal(size=(3, 40))
    g = np.repeat(np.arange(8), 5)
    adj = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    a = pd.Series(adj).groupby(g).mean().to_numpy()
    r = te.clark_west(y, f1, f2, groups=g, lags=2)
    assert r["T"] == 8
    assert r["cw_stat"] == pytest.approx(a.mean() / np.sqrt(te.hac_var(a, 2) / 8))


def test_dm_hln_hand():
    la = np.array([1.0, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7, 1.0])
    lb = np.array([0.9, 0.9, 1.0, 0.8, 1.0, 1.1, 0.8, 0.9])
    d = la - lb
    T = len(d)
    dm = d.mean() / np.sqrt(np.var(d) / T)
    k = np.sqrt((T + 1 - 2 + 0) / T)                # HLN factor, h = 1
    r = te.dm_hln(la, lb, h=1, variance="hln")
    assert r["dm"] == pytest.approx(dm)
    assert r["dm_hln"] == pytest.approx(dm * k)
    assert r["p_two_sided"] == pytest.approx(2 * (1 - stats.t.cdf(abs(dm * k), T - 1)))


def test_dm_matches_6oct_reference_convention():
    """The 6 Oct script's dm(): Bartlett L = 7 with the mean of the cross products (T - l)."""
    rng = np.random.default_rng(3)
    d = rng.normal(0.01, 1, 885)
    L = 7
    e = d - d.mean()
    s = (e ** 2).mean() + sum(2 * (1 - l / (L + 1)) * (e[l:] * e[:-l]).mean() for l in range(1, L + 1))
    ref = d.mean() / np.sqrt(s / len(d))
    assert te.dm_hln(d, np.zeros_like(d), lags=7, divisor="T-l")["dm"] == pytest.approx(ref)


def test_gw_unconditional_equals_uncentred_dm():
    rng = np.random.default_rng(1)
    la, lb = rng.normal(1, 1, 300), rng.normal(1, 1, 300)
    d = la - lb
    r = te.giacomini_white(la, lb, conditional=False)
    assert r["gw_stat"] == pytest.approx(len(d) * d.mean() ** 2 / np.mean(d ** 2))
    assert r["df"] == 1
    rc = te.giacomini_white(la, lb, conditional=True)
    assert rc["df"] == 2 and 0 <= rc["p"] <= 1


def test_holm_hand():
    r = te.holm([0.01, 0.04, 0.03, 0.20], 0.05)
    assert np.allclose(r["p_holm"], [0.04, 0.09, 0.09, 0.20])
    assert r["reject"] == [True, False, False, False]


def test_mcs_wrapper_matches_arch_and_excludes_bad_model():
    from arch.bootstrap import MCS
    rng = np.random.default_rng(2)
    base = rng.chisquare(1, 500)
    L = pd.DataFrame({"a": base, "b": base + rng.normal(0, 0.05, 500), "bad": base + 0.5 + rng.normal(0, 0.05, 500)})
    t = te.mcs(L, 0.10, reps=2000, block_size=5, seed=7)
    m = MCS(L, size=0.10, reps=2000, block_size=5, method="R", bootstrap="stationary", seed=7)
    m.compute()
    assert set(t[t.in_mcs].model) == set(m.included)
    assert "bad" not in set(t[t.in_mcs].model)
    assert np.allclose(t.set_index("model").mcs_pvalue.sort_index(), m.pvalues.iloc[:, 0].sort_index())


def test_qlike_and_oos_r2():
    assert np.allclose(te.qlike([1.0, 2.0], [1.0, 2.0]), 0)
    assert te.qlike([2.0], [1.0])[0] == pytest.approx(2 - np.log(2) - 1)
    assert te.oos_r2([1, 1], [2, 2]) == pytest.approx(0.5)


def test_confirmatory_rules():
    nested = pd.DataFrame([
        dict(test="H1", cw_p=0.001, dm_qlike=-3.0, dm_qlike_p2=0.001, dm_mse_p2=0.01),
        dict(test="H2", cw_p=0.40, dm_qlike=0.1, dm_qlike_p2=0.6, dm_mse_p2=0.7,
             dm_mse_p_big_better=0.65, dm_qlike_p_big_better=0.7),   # 7 Oct 2026: Deviation 4 uses the one-sided p
        dict(test="H3", cw_p=0.30, dm_qlike=0.5, dm_qlike_p2=0.5, dm_mse_p2=0.5),
    ])
    mcs = pd.DataFrame(dict(loss=["MSE_log", "QLIKE"], model=["R1", "R1"], in_mcs=[True, True], alpha=[0.10, 0.10]))
    c = te.confirmatory(nested, mcs, 0.02, 0.05, mcs).set_index("hypothesis")   # MCS of {R1, R2} (Deviation 4)
    assert c.loc["H1", "p_holm"] == pytest.approx(0.004)
    assert c.loc["H1", "verdict"] == "supported on MSE, not robust to the loss"
    assert c.loc["H2", "verdict"] == "supported"
    assert c.loc["H3", "verdict"] == "not supported"
    assert c.loc["H4", "p_holm"] == pytest.approx(0.06)       # 0.02 * 3 after 0.001 * 4
