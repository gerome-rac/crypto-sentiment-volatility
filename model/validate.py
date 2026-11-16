"""Validation of the engine on R0 (real C/J daily data) and on synthetic regressors only.
No real text, attention or control series are read.

    python -m model.validate r0          --cj <cj_focused_empirical_daily_20261005.parquet> --out <dir>
    python -m model.validate ladder      --cj <...> --out <dir> [--plant]
    python -m model.validate cw_size     --cj <...> --out <dir> --reps 500
    python -m model.validate cw_size_lasso --cj <...> --out <dir> --reps 40
    python -m model.validate wald_size   --cj <...> --out <dir> --reps 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config, har, inference, oos, panel, tests_eval
from .config import EngineConfig
from .run import md5

# 6 Oct 2026 reference values (har_insample_descriptive_20261006.csv, har_model_comparison_20261006.csv)
REF = {
    ("insample", "HAR-CJ", "R2_desc"): 0.634685, ("insample", "HAR-CJ", "MSE_log"): 0.453385,
    ("insample", "HAR-CJ", "QLIKE"): 0.325759, ("insample", "HAR-CJ", "R2_comp"): 0.634627,
    ("insample", "HAR-RV", "R2_desc"): 0.632034, ("insample", "HAR-RV", "MSE_log"): 0.456676,
    ("insample", "HAR-RV", "QLIKE"): 0.329556, ("insample", "HAR-RV", "R2_comp"): 0.631975,
    ("oos", "HAR-CJ", "MSE_log"): 0.464004, ("oos", "HAR-CJ", "QLIKE"): 0.327177, ("oos", "HAR-CJ", "R2_oos_coinmean"): 0.607691,
    ("oos", "HAR-RV", "MSE_log"): 0.466210, ("oos", "HAR-RV", "QLIKE"): 0.330938, ("oos", "HAR-RV", "R2_oos_coinmean"): 0.605826,
    ("oos", "HAR-RV", "t_MSE_vs_main"): 1.348943, ("oos", "HAR-RV", "t_QLIKE_vs_main"): 1.745403,
    ("insample_excl_doge1920", "HAR-CJ", "MSE_log"): 0.460185,
}


def _write(df, path):
    tmp = path + ".tmp"
    (df.to_parquet if path.endswith(".parquet") else df.to_csv)(tmp, index=False)
    os.replace(tmp, path)


def r0_builder(cj, cfg, extra_specs=()):
    vol = panel.volatility_from_cj_artifact(cj)
    specs = [v for v in panel.load_variable_spec() if v.block in ("har", "har_rv")] + list(extra_specs)
    return vol, specs


def cmd_r0(a):
    cfg = EngineConfig()
    vol, specs = r0_builder(a.cj, cfg)
    os.makedirs(a.out, exist_ok=True)
    pdir = os.path.join(a.out, "r0_validation_panel")
    os.makedirs(pdir, exist_ok=True)
    _write(vol, os.path.join(pdir, "volatility_daily.parquet"))
    B = har.FeatureBuilder(panel.build_grid({"v": vol}, specs), specs, cfg)
    rows = []
    for rung, lab in (("R0", "HAR-CJ"), ("HARRV", "HAR-RV")):
        d = inference.insample_design(B, rung)
        from .lasso import fit_window
        f = fit_window(d.X_tr, d.y_tr, d.penalised, d.coin_tr, d.day_tr, B.C, cfg)
        res = d.y_tr - f.predict(d.X_tr, d.coin_tr)
        F = np.exp(d.y_tr - res + res.var() / 2)
        rvn = B.rv_next[d.coin_tr, d.day_tr]
        vals = dict(R2_desc=1 - res.var() / np.var(d.y_tr, ddof=1), R2_comp=1 - res.var() / np.var(d.y_tr),
                    MSE_log=np.mean(res ** 2), QLIKE=tests_eval.qlike(rvn, F).mean())
        for k, v in vals.items():
            rows.append(dict(sample="insample", model=lab, stat=k, engine=v, n=len(res)))
        if rung == "R0":
            excl = ~((np.array(B.coins)[d.coin_tr] == "DOGE") & (B.days[d.day_tr].year <= 2020))
            f2 = fit_window(d.X_tr[excl], d.y_tr[excl], d.penalised, d.coin_tr[excl], d.day_tr[excl], B.C, cfg)
            r2 = d.y_tr[excl] - f2.predict(d.X_tr[excl], d.coin_tr[excl])
            rows.append(dict(sample="insample_excl_doge1920", model=lab, stat="MSE_log", engine=np.mean(r2 ** 2), n=int(excl.sum())))
    t0 = time.perf_counter()
    fcs, walls = {}, {}
    for rung in ("R0", "HARRV"):
        fc, coefs, meta, wall, _ = oos.run_oos(B, cfg, rung, n_jobs=1)
        fcs[rung], walls[rung] = fc, wall
    Ls = {k: tests_eval.loss_frame(v).sort_values(["target", "coin"]).reset_index(drop=True) for k, v in fcs.items()}
    for rung, lab in (("R0", "HAR-CJ"), ("HARRV", "HAR-RV")):
        L = Ls[rung]
        r2 = 1 - L.se.sum() / ((L.y - L.groupby("coin").y.transform("mean")) ** 2).sum()
        for k, v in dict(MSE_log=L.se.mean(), QLIKE=L.qlike.mean(), R2_oos_coinmean=r2).items():
            rows.append(dict(sample="oos", model=lab, stat=k, engine=v, n=len(L)))
    dse = {k: tests_eval.day_mean(L.se, L.target) for k, L in Ls.items()}
    dql = {k: tests_eval.day_mean(L.qlike, L.target) for k, L in Ls.items()}
    rows.append(dict(sample="oos", model="HAR-RV", stat="t_MSE_vs_main",
                     engine=tests_eval.dm_hln(dse["HARRV"], dse["R0"], lags=7, divisor="T-l")["dm"], n=885))
    rows.append(dict(sample="oos", model="HAR-RV", stat="t_QLIKE_vs_main",
                     engine=tests_eval.dm_hln(dql["HARRV"], dql["R0"], lags=7, divisor="T-l")["dm"], n=885))
    out = pd.DataFrame(rows)
    out["reference"] = [REF.get((r.sample, r.model, r.stat), np.nan) for r in out.itertuples()]
    out["abs_diff"] = (out.engine - out.reference).abs()
    out["match_6dp"] = out.abs_diff < 5e-7
    _write(out, os.path.join(a.out, "validation_r0_reproduction.csv"))
    _write(pd.concat(fcs.values(), ignore_index=True), os.path.join(a.out, "validation_r0_forecasts.parquet"))
    print(out.to_string())
    print(json.dumps({f"oos_{k}_seconds_1core": v for k, v in walls.items()}))


def _noise_tables(vol, seed, plant=False, strength=0.006):
    """Synthetic regressors with the default spec's columns (R1-R3 dimensions). With plant=True,
    emo_fear (+) and emo_anger (-) carry the next day's log-RV shock."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))
    import conftest
    v = vol.copy()
    lv = np.log(v.rv)
    shock = (lv.groupby(v.coin).shift(-1) - lv.groupby(v.coin).transform(lambda s: s.rolling(30, 1).mean())).fillna(0)
    def pl(txt, att, rng):
        u = shock.to_numpy() / shock.std()
        txt["emo_fear"] = np.clip(txt["emo_fear"] + strength * u, 1e-4, None)
        txt["emo_anger"] = np.clip(txt["emo_anger"] - (2 / 3) * strength * u, 1e-4, None)
    order = pd.Categorical(v.coin, list(config.COINS))
    v = v.assign(_o=order).sort_values(["_o", "day"]).drop(columns="_o").reset_index(drop=True)
    return conftest.synth_tables(v, seed=seed, plant=pl if plant else None)


def cmd_ladder(a):
    from .run import main as run_main
    vol = panel.volatility_from_cj_artifact(a.cj)
    tabs = _noise_tables(vol, seed=11, plant=a.plant, strength=a.strength)
    lab = (f"SYNTH_PLANTED{int(round(a.strength * 1000)):03d}" if a.plant else "SYNTH_NOISE") + ("_PLACEBO" if a.placebo else "")
    pdir = os.path.join(a.out, f"panel_{lab.lower()}")
    os.makedirs(pdir, exist_ok=True)
    for k, t in tabs.items():
        _write(t, os.path.join(pdir, f"{k}.parquet"))
    argv = ["--spec", "A00", "--rung", "all", "--panel", pdir, "--label", lab, "--out-root", a.out,
            "--n-jobs", str(a.n_jobs)]
    if a.placebo:
        argv += ["--placebo-shift-days", "365"]
    run_main(argv)


def _cw_rep(cj_vol, specs_base, rep, cfg, n_noise, f0):
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(1000 + rep)
    vol = cj_vol
    names = [f"noise{i}" for i in range(n_noise)]
    noise = pd.DataFrame({"coin": vol.coin, "day": vol.day, **{n: rng.normal(size=len(vol)) for n in names}})
    sp = specs_base + [panel.VarSpec(n, "noise", "attention", "coin", True, "level", False, False, "") for n in names]
    B = har.FeatureBuilder(panel.build_grid({"v": vol, "n": noise}, sp), sp, cfg)
    with threadpool_limits(1):
        fc, coefs, meta, wall, _ = oos.run_oos(B, cfg, "R1", n_jobs=1)
    fc = fc.sort_values(["target", "coin"]).reset_index(drop=True)
    cw = tests_eval.clark_west(fc.y, f0.f_log, fc.f_log, groups=fc.target, cfg=cfg)
    L1, L0 = tests_eval.loss_frame(fc), tests_eval.loss_frame(f0)
    dm = tests_eval.dm_hln(tests_eval.day_mean(L0.se, L0.target), tests_eval.day_mean(L1.se, L1.target), cfg=cfg)
    sel = (coefs[[c for c in coefs.columns if c.startswith("noise")]] != 0).to_numpy()
    return dict(rep=rep, cw_stat=cw["cw_stat"], cw_p=cw["p_one_sided"], degenerate=cw["degenerate"],
                dm_p2=dm["p_two_sided"], dm_p_big_better=dm["p_b_better"],
                r2_oos=tests_eval.oos_r2(L1.se, L0.se),
                sel_col_freq=float(sel.mean()), sel_any_freq=float(sel.any(axis=1).mean()), seconds=wall)


def _cw_mc(a, estimator, n_noise, tag):
    cfg = EngineConfig(estimator=estimator, cv_folds=5)
    # the noise block sits in the "attention" slot of R1; the builder holds no other R1 block
    vol = panel.volatility_from_cj_artifact(a.cj)
    vol = vol[["coin", "day", "rv", "c", "j"]]
    specs = [v for v in panel.load_variable_spec() if v.block in ("har", "har_rv")]
    B0 = har.FeatureBuilder(panel.build_grid({"v": vol}, specs), specs, cfg)
    f0 = oos.run_oos(B0, cfg, "R0", n_jobs=1)[0].sort_values(["target", "coin"]).reset_index(drop=True)
    t0 = time.perf_counter()
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(
        delayed(_cw_rep)(vol, specs, r, cfg, n_noise, f0) for r in range(a.reps))
    out = pd.DataFrame(res)
    out["estimator"], out["n_noise_vars"], out["n_noise_cols"] = estimator, n_noise, 3 * n_noise
    _write(out, os.path.join(a.out, f"validation_{tag}.csv"))
    summ = dict(reps=len(out), estimator=estimator, n_noise_cols=3 * n_noise,
                cw_reject_05=float((out.cw_p < 0.05).mean()), cw_reject_10=float((out.cw_p < 0.10).mean()),
                dm_reject_05_two_sided=float((out.dm_p2 < 0.05).mean()),
                dm_reject_05_one_sided=float((out.dm_p_big_better < 0.05).mean()),
                degenerate_share=float(out.degenerate.mean()), mean_sel_col_freq=float(out.sel_col_freq.mean()),
                mean_sel_any_freq=float(out.sel_any_freq.mean()), mean_r2_oos=float(out.r2_oos.mean()),
                wall_seconds=time.perf_counter() - t0)
    print(json.dumps(summ, indent=1))
    with open(os.path.join(a.out, f"validation_{tag}_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)


def _wald_rep(vol, rep, cfg, boot_reps=0):
    from threadpoolctl import threadpool_limits
    cfg30 = EngineConfig(dk_bandwidth=30)
    tabs = _noise_tables(vol, seed=5000 + rep)
    specs = panel.load_variable_spec()
    B = har.FeatureBuilder(panel.build_grid(tabs, specs), specs, cfg)
    out = []
    with threadpool_limits(1):
        for rung in ("R1", "R3"):
            ins = inference.ols_dk_nw(B, rung, cfg)
            for r in ins["wald"].itertuples():
                out.append(dict(rep=rep, rung=rung, block=r.block, cov=r.cov, q=r.q, wald=r.wald, p=r.p))
            if boot_reps:
                wb = inference.wald_bootstrap(ins, B, cfg, boot_reps)
                for r in wb.itertuples():
                    out.append(dict(rep=rep, rung=rung, block=r.block, cov=r.cov, q=r.q, wald=r.wald, p=r.p))
            if rung == "R3":
                c = ins["coef"]
                out.append(dict(rep=rep, rung=rung, block="emo_fear_h1 single", cov="DK", q=1, wald=np.nan,
                                p=float(c.loc[c.variable == "emo_fear_h1", "p_dk"].iloc[0])))
                ins30 = inference.ols_dk_nw(B, rung, cfg30)
                for r in ins30["wald"].itertuples():
                    if r.cov == "DK":
                        out.append(dict(rep=rep, rung=rung, block=r.block, cov="DK_bw30", q=r.q, wald=r.wald, p=r.p))
    return out


def cmd_wald_size(a):
    cfg = EngineConfig()
    vol = panel.volatility_from_cj_artifact(a.cj)[["coin", "day", "rv", "c", "j"]]
    t0 = time.perf_counter()
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(_wald_rep)(vol, r, cfg, a.boot_reps) for r in range(a.reps))
    out = pd.DataFrame([x for r in res for x in r])
    tag = f"_boot{a.boot_reps}" if a.boot_reps else ""
    _write(out, os.path.join(a.out, f"validation_wald_size{tag}.csv"))
    summ = out.groupby(["rung", "block", "cov", "q"]).p.agg(reject_05=lambda p: (p < 0.05).mean(),
                                                           reject_10=lambda p: (p < 0.10).mean(), n="size").reset_index()
    summ["wall_seconds"] = time.perf_counter() - t0
    _write(summ, os.path.join(a.out, f"validation_wald_size{tag}_summary.csv"))
    print(summ.to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m model.validate")
    ap.add_argument("cmd", choices=["r0", "ladder", "cw_size", "cw_size_lasso", "wald_size"])
    ap.add_argument("--cj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--plant", action="store_true")
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--n-noise", type=int, default=None)
    ap.add_argument("--strength", type=float, default=0.006, help="planted fear loading per SD of the next-day shock")
    ap.add_argument("--boot-reps", type=int, default=199)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "r0":
        cmd_r0(a)
    elif a.cmd == "ladder":
        cmd_ladder(a)
    elif a.cmd == "cw_size":
        _cw_mc(a, "ols", a.n_noise or 2, "cw_size_ols")
    elif a.cmd == "cw_size_lasso":
        _cw_mc(a, "adaptive_lasso", a.n_noise or 10, "cw_size_lasso")
    elif a.cmd == "wald_size":
        cmd_wald_size(a)


if __name__ == "__main__":
    sys.exit(main())
