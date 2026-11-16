"""Design builder (HAR form, in-window transforms, no look-ahead), OOS schedule and the
adaptive LASSO."""
import numpy as np
import pandas as pd
import pytest

from model import config, har, lasso, oos, panel
from model.config import EngineConfig


def _builder(tables, specs, cfg=None):
    cfg = cfg or EngineConfig()
    grid = panel.build_grid(tables, specs)
    return har.FeatureBuilder(grid, specs, cfg), grid


def test_rolling_nanmean_matches_pandas():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 60))
    x[0, [3, 4, 10]] = np.nan
    for k in (1, 7, 30):
        ref = pd.DataFrame(x.T).rolling(k, min_periods=1).mean().to_numpy().T
        assert np.allclose(har.rolling_nanmean(x, k), ref, equal_nan=True)


def test_utc_assertion():
    s = pd.Series(pd.to_datetime(["2020-01-01 00:00"]).tz_localize("Europe/Zurich"))
    with pytest.raises(AssertionError):
        panel.utc_days(s)
    with pytest.raises(AssertionError):
        panel.utc_days(pd.Series(pd.to_datetime(["2020-01-01 13:00"])))
    ok = panel.utc_days(pd.Series(pd.to_datetime(["2020-01-01"]).tz_localize("UTC")))
    assert ok.iloc[0] == pd.Timestamp("2020-01-01")


def test_schedule_counts():
    cfg = EngineConfig()
    vol = __import__("conftest").synth_volatility()
    sp = [v for v in panel.load_variable_spec() if v.block in ("har", "har_rv")]
    b, _ = _builder({"volatility_daily": vol}, sp, cfg)
    s = oos.schedule(b, cfg)
    assert len(s) == config.N_FORECAST_DAYS == 885
    assert b.days[s[0][0] + 1] == pd.Timestamp("2020-08-30") and b.days[s[-1][0] + 1] == pd.Timestamp("2023-01-31")
    assert all(len(tr) == 365 and tr[-1] == te - 1 for te, tr in s)
    assert b.days[s[0][1][0]] == pd.Timestamp("2019-08-30")
    s250 = oos.schedule(b, EngineConfig(window=250))
    assert len(s250) == 885 and all(len(tr) == 250 for _, tr in s250)
    se = oos.schedule(b, EngineConfig(window="expanding"))
    assert len(se[0][1]) == 365 and len(se[-1][1]) == 365 + 884
    d = b.design("R0", s[0][1], np.array([s[0][0]]))
    assert d.X_tr.shape == (1825, 6) and d.X_te.shape == (5, 6)


def test_har_features_hand(vol, specs):
    sp = [v for v in specs if v.block in ("har", "har_rv")]
    b, _ = _builder({"volatility_daily": vol}, sp)
    tr = np.arange(29, 394)
    d = b.design("R0", tr, np.array([394]))
    x = vol[vol.coin == "ETH"].set_index("day")
    t = b.days[394]
    i = d.names.index("c_h7")
    j = d.names.index("j_h30")
    ce = list(config.COINS).index("ETH")
    assert d.X_te[ce, i] == pytest.approx(np.log(x.c.loc[t - pd.Timedelta(days=6): t].mean()))
    assert d.X_te[ce, j] == pytest.approx(np.log1p(x.j.loc[t - pd.Timedelta(days=29): t].mean()))
    assert d.y_te[ce] == pytest.approx(np.log(x.rv.loc[t + pd.Timedelta(days=1)]))


def test_imputation_and_centring_inside_window(tables, specs):
    b, grid = _builder(tables, specs)
    tr = np.arange(100, 465)
    info = {}
    F, names, *_ = b.features("R3", tr, info)
    # Twitter zero-capture days imputed with the window mean of the present days, per coin
    tw = b.raw["att_twitter"]
    a = b.flags["twitter_absent"]
    exp = np.nanmean(np.where(a, np.nan, tw)[:, tr], axis=1)
    assert np.allclose(info["impute_value"]["att_twitter"], exp)
    # centred square of fear: each coin's own window mean (default since 7 Oct 2026), (x - m_c)^2 then averages
    m = np.nanmean(b.raw["emo_fear"][:, tr], axis=1)
    assert np.allclose(info["centre"]["emo_fear"], m)
    k = names.index("emo_fear_sq_h7")
    t = 300
    assert F[k][2, t] == pytest.approx(np.mean((b.raw["emo_fear"][2, t - 6: t + 1] - m[2]) ** 2))


def test_no_look_ahead(tables, specs, vol):
    """Changing any raw value after the forecast origin leaves the forecast unchanged."""
    cfg = EngineConfig(cv_folds=3, n_lambda=15)
    b1, grid = _builder(tables, specs, cfg)
    te = 600
    tr = np.arange(te - 365, te)
    r1 = oos._one(b1, cfg, "R3", te, tr)
    t2 = {k: v.copy() for k, v in tables.items()}
    cut = b1.days[te]
    for k, v in t2.items():
        num = [c for c in v.columns if c not in ("coin", "day") and v[c].dtype.kind == "f"]
        late = v.day > cut
        v.loc[late, num] = v.loc[late, num] * 3.0 + 1.0
    b2, _ = _builder(t2, specs, cfg)
    r2 = oos._one(b2, cfg, "R3", te, tr)
    assert np.allclose(r1["f"], r2["f"])
    # ...but changing the origin day's own regressors changes it
    t3 = {k: v.copy() for k, v in tables.items()}
    t3["volatility_daily"].loc[t3["volatility_daily"].day == cut, "c"] *= 5
    b3, _ = _builder(t3, specs, cfg)
    assert not np.allclose(oos._one(b3, cfg, "R3", te, tr)["f"], r1["f"])


def test_placebo_shift(tables, specs):
    grid = panel.build_grid(tables, specs)
    g = panel.placebo_shift(grid, specs, 365)
    x0 = grid.xs("BTC").emo_fear.to_numpy()
    x1 = g.xs("BTC").emo_fear.to_numpy()
    assert np.allclose(x1[365:], x0[:-365]) and np.allclose(x1[:365], x0[-365:])
    assert np.allclose(g.att_reddit, grid.att_reddit)          # attention not shifted
    gl = panel.placebo_shift(grid, specs, 365, mode="lag")
    assert np.isnan(gl.xs("BTC").pol.to_numpy()[:365]).all()


def test_day_folds_never_random():
    day = np.repeat(np.arange(100), 5)
    for tr, va in lasso.day_folds(day, "forward", 5):
        assert day[tr].max() < day[va].min()
        assert set(np.unique(day[va])) == set(range(day[va].min(), day[va].max() + 1))
    for tr, va in lasso.day_folds(day, "blocked", 5, gap=3):
        v0, v1 = day[va].min(), day[va].max()
        assert not np.any((day[tr] >= v0 - 3) & (day[tr] <= v1 + 3))
    with pytest.raises(ValueError):
        lasso.day_folds(day, "random", 5)


def test_ols_and_adaptive_lasso():
    rng = np.random.default_rng(5)
    nd, C = 365, 5
    day = np.repeat(np.arange(nd), C)
    coin = np.tile(np.arange(C), nd)
    U = rng.normal(size=(nd * C, 3))
    P = rng.normal(size=(nd * C, 20))
    y = U @ np.array([0.5, 0.2, 0.1]) + 0.3 * P[:, 0] - 0.2 * P[:, 1] + coin * 0.1 + rng.normal(0, 1, nd * C)
    X = np.column_stack([U, P])
    pen = np.r_[np.zeros(3, bool), np.ones(20, bool)]
    cfg = EngineConfig()
    f_ols = lasso.fit_window(X, y, pen, coin, day, C, cfg, estimator="ols")
    D = lasso.fe_dummies(coin, C)
    b, *_ = np.linalg.lstsq(np.column_stack([X, D]), y, rcond=None)
    assert np.allclose(f_ols.predict(X, coin), np.column_stack([X, D]) @ b)
    f = lasso.fit_window(X, y, pen, coin, day, C, cfg)
    sel = f.coef_z[3:] != 0
    assert sel[0] and sel[1] and sel[2:].sum() <= 6
    assert np.all(f.coef_z[:3] != 0)                       # unpenalised terms always kept
    # very large lambda: penalised block zero, unpenalised = OLS on U and FE
    fz = lasso.fit_window(X, y, pen, coin, day, C, cfg, fixed_lambda=1e6, fixed_ridge_rel=1.0)
    assert np.all(fz.coef_z[3:] == 0)
    bu, *_ = np.linalg.lstsq(np.column_stack([U, D]), y, rcond=None)
    assert np.allclose(fz.predict(X, coin), np.column_stack([U, D]) @ bu)
    fe = lasso.fit_window(X, y, pen, coin, day, C, cfg, estimator="enet")
    assert fe.coef_z[3] != 0


def test_per_coin_unit_and_duan(vol, specs):
    sp = [v for v in specs if v.block in ("har", "har_rv")]
    cfg = EngineConfig(unit="per_coin")
    b, _ = _builder({"volatility_daily": vol}, sp, cfg)
    te, tr = oos.schedule(b, cfg)[0]
    r = oos._one(b, cfg, "R0", te, tr)
    assert r["coef_z"].shape[0] == 5 and np.isfinite(r["f"]).all()
    d = b.design("R0", tr, np.array([te]), coins_idx=np.array([3]))
    f = lasso.fit_window(d.X_tr, d.y_tr, d.penalised, np.zeros(len(d.y_tr), int), d.day_tr, 1, cfg)
    assert r["f"][3] == pytest.approx(f.predict(d.X_te, np.zeros(1, int))[0])
    assert f.duan == pytest.approx(np.mean(np.exp(f.resid)))


def test_composition_hook_level_shift(tables, specs):
    t = {k: v.copy() for k, v in tables.items()}
    tx = t["text_daily"]
    tx["pol_2p"] = tx["pol"] - 0.05
    absent = t["attention_daily"].set_index(["coin", "day"]).twitter_absent
    tx["twitter_absent"] = absent.reindex(pd.MultiIndex.from_frame(tx[["coin", "day"]])).to_numpy()
    tx["twitter_full_capture"] = ~tx["twitter_absent"]
    cfg = EngineConfig(composition_adjustment="level_shift")
    grid = panel.build_grid(t, specs, extra_cols=("twitter_full_capture",))
    for c in ("pol_2p",):
        grid[c] = t["text_daily"].set_index(["coin", "day"])[c].reindex(grid.index).to_numpy()
    b = har.FeatureBuilder(grid, specs, cfg)
    info = {}
    x = b._window_series(b.specs["pol"], np.arange(100, 465), info)
    assert info["composition_shift"]["pol"] == pytest.approx(0.05)
    a = b.flags["twitter_absent"]
    assert np.allclose(x[a], b.raw["pol"][a] + 0.05) and np.allclose(x[~a], b.raw["pol"][~a])
