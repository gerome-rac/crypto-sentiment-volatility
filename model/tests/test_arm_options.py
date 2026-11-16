"""Added 28 Oct 2026 (engine-arms track): engine options for register arms A53 (drop rule), A61 (Duan QLIKE) and
A62 (evaluation target), the extra-table interface, the fixed-effect identification guard (open question A52) and
the per-coin in-sample guard (A01). Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from model import config, har, inference, oos, panel, tests_eval
from model import run as runmod
from model.config import EngineConfig


def _vol_specs():
    return [v for v in panel.load_variable_spec() if v.block in ("har", "har_rv")]


def _builder(tables, cfg, specs=None, extra_cols=(), value_cols=()):
    sp = specs or _vol_specs()
    grid = panel.build_grid(tables, sp, extra_cols=extra_cols, value_cols=value_cols)
    return har.FeatureBuilder(grid, sp, cfg)


def test_new_spec_arms_and_a00_defaults():
    d = EngineConfig()
    assert d.drop_days_flag is None and d.eval_target == "rv" and d.qlike_backtransform == "lognormal"
    a00 = EngineConfig.for_spec("A00")
    assert a00.drop_days_flag is None and a00.eval_target == "rv" and a00.qlike_backtransform == "lognormal"
    assert EngineConfig.for_spec("A61").qlike_backtransform == "duan"
    assert EngineConfig.for_spec("A53").drop_days_flag == "abd_drop_flag"
    with pytest.raises(ValueError, match="eval_target"):
        EngineConfig.for_spec("A62")                       # edge rule open: the column must be named
    assert EngineConfig.for_spec("A62", eval_target="rv5_ss_E1").eval_target == "rv5_ss_E1"
    for arm in ("A61", "A62", "A53"):                      # one axis changed relative to A00
        kw = config.SPEC_ARMS[arm]
        assert kw["window"] == 365 and kw["unit"] == "pooled" and "rung_blocks" not in kw


def test_qlike_duan_backtransform_uses_smearing_factor():
    rng = np.random.default_rng(4)
    n = 50
    f = rng.normal(1, 0.3, n)
    fc = pd.DataFrame(dict(coin="BTC", target=pd.date_range("2021-01-01", periods=n), y=f + rng.normal(0, .5, n),
                           f_log=f, rv_next=np.exp(f + rng.normal(0, .5, n)), sigma2=rng.uniform(0.1, 0.5, n),
                           duan=rng.uniform(1.05, 1.4, n)))
    fc["F_lognormal"] = np.exp(fc.f_log + fc.sigma2 / 2)
    fc["F_duan"] = np.exp(fc.f_log) * fc.duan
    ln, dn = tests_eval.loss_frame(fc, "lognormal"), tests_eval.loss_frame(fc, "duan")
    assert np.allclose(dn.qlike, tests_eval.qlike(fc.rv_next, fc.F_duan))
    assert np.allclose(ln.qlike, tests_eval.qlike(fc.rv_next, fc.F_lognormal))
    assert np.array_equal(ln.se, dn.se) and not np.allclose(ln.qlike, dn.qlike)


def test_extra_table_adds_columns_only(tmp_path, vol):
    merged = tmp_path / "panel.parquet"
    vol.to_parquet(merged, index=False)
    ex = vol[["coin", "day"]].copy()
    ex["rv_alt"] = vol.rv * 1.1
    p_ex = tmp_path / "extra.parquet"
    ex.to_parquet(p_ex, index=False)
    t = panel.load_tables(str(merged), extra=[str(p_ex)])
    assert set(t) == {"merged", "extra_0"}
    g = panel.build_grid(t, _vol_specs(), value_cols=("rv_alt",))
    assert g["rv_alt"].dtype == float and np.allclose(g["rv_alt"].to_numpy(), g["rv"].to_numpy() * 1.1)
    p_bad = tmp_path / "bad.parquet"
    vol[["coin", "day", "rv"]].to_parquet(p_bad, index=False)
    with pytest.raises(ValueError, match="already in the panel"):
        panel.load_tables(str(merged), extra=[str(p_bad)])
    assert set(panel.load_tables(str(merged))) == {"merged"}             # default unchanged


def test_eval_target_changes_evaluation_only(vol):
    v = vol.copy()
    v["rv_alt"] = v.rv * np.exp(np.random.default_rng(9).normal(0, 0.1, len(v)))
    base = _builder({"v": v}, EngineConfig())
    cfg = EngineConfig(eval_target="rv_alt")
    alt = _builder({"v": v}, cfg, value_cols=("rv_alt",))
    f0 = oos.run_oos(base, EngineConfig(), "HARCJ", n_jobs=1, max_days=4)[0]
    f1 = oos.run_oos(alt, cfg, "HARCJ", n_jobs=1, max_days=4)[0]
    assert np.array_equal(f0.f_log, f1.f_log) and np.array_equal(f0.sigma2, f1.sigma2)   # same fit
    assert np.array_equal(f1.y_fit, f0.y) and np.array_equal(f1.rv_next_fit, f0.rv_next)
    s = v.set_index(["coin", "day"]).rv_alt
    exp_rv = s.loc[list(zip(f1.coin, f1.target))].to_numpy()
    assert np.allclose(f1.rv_next, exp_rv) and np.allclose(f1.y, np.log(exp_rv))
    assert list(f0.columns) == [c for c in f1.columns if c not in ("y_fit", "rv_next_fit")]
    with pytest.raises(KeyError, match="eval_target"):
        _builder({"v": v}, cfg)                                        # column not loaded


def test_drop_rule_rows_averages_and_forecasts(vol):
    v = vol.copy()
    v["x_att"] = np.random.default_rng(3).normal(size=len(v))
    drop_days = pd.to_datetime(["2020-03-10", "2020-09-02"])
    v["dropflag"] = v.day.isin(drop_days)
    specs = _vol_specs() + [panel.VarSpec("x_att", "v", "attention", "coin", True, "level", False, False, "")]
    cfg = EngineConfig(drop_days_flag="dropflag", first_target="2020-09-01")
    b = _builder({"v": v}, cfg, specs=specs, extra_cols=("dropflag",))
    i1, i2 = b.day_index("2020-03-10"), b.day_index("2020-09-02")
    # (a) volatility values and the target are missing on flagged days; other blocks are untouched
    assert np.isnan(b.raw["c"][:, i1]).all() and np.isnan(b.raw["j"][:, i2]).all() and np.isnan(b.raw["rv"][:, i2]).all()
    assert np.isfinite(b.raw["x_att"][:, [i1, i2]]).all()
    assert np.isnan(b.y[:, i1 - 1]).all() and np.isfinite(b.y[:, i1]).all()
    # (b) the 7-day HAR average after a flagged day runs over the observed days
    tr = np.arange(i1 - 200, i1 + 20)
    F, names, *_ = b.features("HARCJ", tr, {})
    eth = list(b.coins).index("ETH")
    raw_c = v[v.coin == "ETH"].set_index("day").c
    win = raw_c.loc[b.days[i1 + 3] - pd.Timedelta(days=6): b.days[i1 + 3]]
    win = win[~win.index.isin(drop_days)]
    assert len(win) == 6 and F[names.index("c_h7")][eth, i1 + 3] == pytest.approx(np.log(win.mean()))
    # (c) training and in-sample rows: none with a flagged origin or target day
    d = b.design("HARCJ", tr, np.array([i1 + 20]))
    assert not np.isin(d.day_tr, [i1 - 1, i1]).any() and len(d.y_tr) == 5 * (len(tr) - 2) and d.te_ok.all()
    dis = inference.insample_design(b, "HARCJ")
    assert not np.isin(dis.day_tr, [i1 - 1, i1, i2 - 1, i2]).any()
    # (d) forecasts for targets 1-4 Sep 2020: 2 Sep (flagged target) and 3 Sep (flagged origin) do not exist
    fc, coefs, meta, _, _ = oos.run_oos(b, cfg, "HARCJ", n_jobs=1, max_days=4)
    assert sorted(fc.target.unique()) == list(pd.to_datetime(["2020-09-01", "2020-09-04"])) and len(fc) == 10
    assert set(meta.origin) == set(fc.origin) == set(coefs.origin)
    n_rows = meta.set_index("origin").n_rows
    assert n_rows.loc[pd.Timestamp("2020-08-31")] == 1825 - 10          # origins 9, 10 Mar 2020 leave the window
    assert n_rows.loc[pd.Timestamp("2020-09-03")] == 1825 - 20          # ... and 1, 2 Sep 2020
    full = oos.forecast_coin_days(b, cfg)
    assert not full.target.isin(drop_days).any() and not full.origin.isin(drop_days).any()
    assert set(zip(fc.coin, fc.target)) <= set(zip(full.coin, full.target))


def test_coin_without_training_rows_is_refused(vol):
    v = vol.copy()
    v["dropflag"] = (v.coin == "DOGE") & (v.day <= pd.Timestamp("2020-12-31"))
    cfg = EngineConfig(drop_days_flag="dropflag", first_target="2021-01-02")
    b = _builder({"v": v}, cfg, extra_cols=("dropflag",))
    with pytest.raises(ValueError, match="not identified"):
        oos.run_oos(b, cfg, "HARCJ", n_jobs=1, max_days=1)


def test_per_coin_arm_requires_no_insample():
    with pytest.raises(SystemExit, match="no-insample"):
        runmod.main(["--spec", "A01", "--panel", "does_not_exist.parquet"])


def test_forecast_coin_days_default_is_the_a00_set(vol):
    b = _builder({"v": vol}, EngineConfig())
    fcd = oos.forecast_coin_days(b, EngineConfig())
    assert len(fcd) == 5 * config.N_FORECAST_DAYS
    assert fcd.target.min() == config.FIRST_TARGET and fcd.target.max() == config.LAST_TARGET
