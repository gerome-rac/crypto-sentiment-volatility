"""Recentered bootstrap for the in-sample adaptive LASSO (Audrino & Camponovo 2018), synthetic data only."""
import numpy as np

from model.config import EngineConfig
from model.inference import _recentered_fit, recentered_bootstrap
from model.lasso import fit_window


def synth_panel(T=250, n_coins=5, beta=(0.4, -0.3, 0.25, 0, 0, 0, 0, 0), seed=0):
    """Daily panel: penalised AR(1) regressors with a common component, two unpenalised regressors,
    coin effects, and errors with a common daily shock (the dependence the bootstrap must respect)."""
    rng = np.random.default_rng(seed)
    p = len(beta)
    n = T * n_coins
    day = np.repeat(np.arange(T), n_coins)
    coin = np.tile(np.arange(n_coins), T)
    common = np.zeros((T, p))
    for t in range(1, T):
        common[t] = 0.5 * common[t - 1] + rng.normal(0, 1, p)
    Xp = common[day] + rng.normal(0, 1, (n, p))
    Xu = rng.normal(0, 1, (n, 2))
    shock = rng.normal(0, 0.5, T)[day]
    y = Xp @ np.asarray(beta) + Xu @ np.array([0.5, -0.2]) + 0.3 * coin + shock + rng.normal(0, 1, n)
    X = np.column_stack([Xu, Xp])
    pen = np.r_[np.zeros(2, bool), np.ones(p, bool)]
    return X, y, pen, coin, day


def test_zero_recentering_equals_plain_adaptive_lasso():
    cfg = EngineConfig()
    X, y, pen, coin, day = synth_panel(seed=1)
    al = fit_window(X, y, pen, coin, day, 5, cfg)
    got = _recentered_fit(X, y, pen, coin, 5, cfg, al.lam, al.ridge_alpha_rel, np.zeros(X.shape[1]))
    ref = fit_window(X, y, pen, coin, day, 5, cfg, fixed_lambda=al.lam, fixed_ridge_rel=al.ridge_alpha_rel).coef_raw
    assert np.allclose(got, ref, atol=1e-8)
    assert np.allclose(got, al.coef_raw, atol=1e-8)


def test_unselected_columns_get_p_one_and_signal_columns_reject():
    cfg = EngineConfig()
    X, y, pen, coin, day = synth_panel(T=400, beta=(0.6, -0.5, 0.4, 0, 0, 0, 0, 0), seed=2)
    r = recentered_bootstrap(X, y, pen, coin, day, 5, cfg, reps=199, seed=7)
    pp = r["p"][2:]
    sel = r["theta_al"][2:] != 0
    assert np.all(pp[~sel] == 1.0)                       # a zero estimate is never significant
    assert np.all(pp[:3] < 0.05)                          # strong true effects are found
    assert np.all((r["ci_lo"] <= r["ci_hi"]))


def test_null_rejection_rate_is_controlled():
    """Small Monte Carlo: share of 5% rejections among truly zero coefficients stays near or below 5%."""
    cfg = EngineConfig()
    rej0, rej1, n0, n1 = 0, 0, 0, 0
    for s in range(12):
        X, y, pen, coin, day = synth_panel(T=250, seed=100 + s)
        r = recentered_bootstrap(X, y, pen, coin, day, 5, cfg, reps=99, seed=s)
        p = r["p"][2:]
        rej1 += int((p[:3] < 0.05).sum()); n1 += 3
        rej0 += int((p[3:] < 0.05).sum()); n0 += 5
    assert rej0 / n0 <= 0.10, rej0 / n0
    assert rej1 / n1 >= 0.80, rej1 / n1
