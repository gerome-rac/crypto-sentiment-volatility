"""Arm A73: state-dependent tests of the ladder on a continuous market state (register rows A73-R0..R3).

POST-RESULT ANALYSIS, written 9 Nov 2026 at Gerome's request, after the bull/bear split (arm A07) had been seen. The
state variable was chosen after the results were known. Nothing here is confirmatory and nothing here changes a D8
section 4 verdict.

State. s_t = log(BTC close_t / MA200_t): the BTCUSDT close of UTC day t against the simple average of that close and
the 199 before it, the moving average of the 1-200 rule of Brock, Lakonishok & LeBaron (1992) used in arm A07. Taking
the log distance instead of its sign is our own choice. s_t is known at the end of the forecast origin day t and is
undefined before the 200th close. It is standardised on the days used, which changes no test statistic.

Part 1 (SQ1-SQ3, stored forecasts only). For each nested step and loss, d_{t+1} is the daily cross-coin mean loss of
the smaller rung minus that of the larger one, as in model.tests_eval.evaluate. The Giacomini & White (2006, pp. 1551,
1553) statistic n Zbar' Omega^-1 Zbar with Z = h_t d_{t+1} is computed for h_t = (1, s_t) and h_t = (1, d_t, s_t). The
slope of OLS d_{t+1} on (1, s_t) is reported with a Newey-West covariance and the engine's lag rule, also for the
Clark-West adjusted differential. A positive slope means the larger rung gains more when BTC is above its average.
Check: with h_t = (1, d_t) the stored gw_mse_p and gw_qlike_p of nested_tests.csv are reproduced.

Part 2 (SQ4, R3 OLS, days with a defined state). Pooled OLS with coin effects and the in-sample design of
model.inference.insample_design, plus s_t and interactions of s_t with a chosen set of regressors, Driscoll-Kraay
covariance as in model.inference.ols_dk_nw. 'parsimonious' interacts the six HAR-CJ terms and the six level terms of
fear and anger; 'full_interaction' interacts all regressors. The test is the slope of
Delta(s) = Cum(fear) - Cum(anger) in s, two-sided. Coefficients are in raw (probability) units, per standard deviation
of the state for the slopes. Check: without state terms on all rows, the stored theta1 and theta2 of h4_bootstrap.csv
are reproduced.

Usage, from code/ (paths from the command line only):
  python -m model.arm_state_dependent run --run-dir <run folder> --panel <panel parquet> --variable-spec <spec csv>
      --klines-dir <folder with BTCUSDT-1m-YYYY-MM.zip> --out-root <folder for the new run folder>
      [--label <label>] [--stamp <UTC stamp>] [--expect-forecasts-md5 <md5>] [--expect-panel-md5 <md5>]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .arm_regime_split import (btc_daily_close, build_builder, daily_differentials, insample_check, load_run, md5,
                               regime_dating, write_csv, write_md5sums)
from .config import NESTED_COMPARISONS, EngineConfig, hac_default_lags
from .tests_eval import day_mean, holm, loss_frame

FEAR = ["emo_fear_h1", "emo_fear_h7", "emo_fear_h30"]
ANGER = ["emo_anger_h1", "emo_anger_h7", "emo_anger_h30"]
EXCL_DAY = pd.Timestamp("2021-09-07")         # the day with the largest QLIKE gains (Appendix D); sensitivity only
EXCL_LABEL = "excl_2021-09-07"
STEPS = ("H1", "H2", "H3")


def state_series(klines_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
    d = regime_dating(btc_daily_close(klines_dir))
    s = np.log(d["btc_close"].to_numpy(float) / d["ma200"].to_numpy(float))
    tab = d[["day", "btc_close", "ma200"]].assign(state=s)
    return tab, pd.Series(s, index=pd.DatetimeIndex(d["day"]), name="state")


# ----------------------------------------------------------------------------- part 1
def gw(d_next: np.ndarray, H: np.ndarray) -> Tuple[float, int, float]:
    """Giacomini-White statistic for one-step forecasts, as model.tests_eval.giacomini_white."""
    Z = H * d_next[:, None]
    n, q = Z.shape
    zbar = Z.mean(0)
    stat = float(n * zbar @ np.linalg.solve(Z.T @ Z / n, zbar))
    return stat, q, float(stats.chi2.sf(stat, q))


def hac_slope(y: np.ndarray, z: np.ndarray, lags: int) -> Dict[str, float]:
    """OLS of y on (1, z) with a Newey-West covariance, as model.arm_regime_split.regime_difference_tests."""
    X = np.column_stack([np.ones(len(y)), z])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    g = X * (y - X @ beta)[:, None]
    S = g.T @ g
    for l in range(1, lags + 1):
        G = g[l:].T @ g[:-l]
        S += (1.0 - l / (lags + 1.0)) * (G + G.T)
    V = XtXi @ S @ XtXi
    se = float(np.sqrt(V[1, 1]))
    t = float(beta[1] / se)
    return dict(const=float(beta[0]), slope=float(beta[1]), slope_se_hac=se, slope_t=t,
                slope_p=float(2 * stats.norm.sf(abs(t))))


def part1(fcs: Dict[str, pd.DataFrame], cfg: EngineConfig, state: pd.Series, run_dir: str) -> Dict[str, pd.DataFrame]:
    f0 = fcs["R0"]
    assert ((pd.to_datetime(f0["target"]) - pd.to_datetime(f0["origin"])) == pd.Timedelta(days=1)).all()
    L = {k: loss_frame(v, cfg.qlike_backtransform).sort_values(["target", "coin"]).reset_index(drop=True)
         for k, v in fcs.items()}
    tgt = L["R0"]["target"]
    days = pd.DatetimeIndex(np.sort(tgt.unique()))
    D = {k: {lo: day_mean(L[k][lo], tgt) for lo in ("se", "qlike")} for k in L}
    s = state.reindex(days - pd.Timedelta(days=1)).to_numpy(float)
    assert np.isfinite(s).all(), "the state is undefined on a forecast origin day"
    zs = (s - s.mean()) / s.std()
    lags = int(cfg.hac_lags) if cfg.hac_lags is not None else hac_default_lags(len(days))
    stored = pd.read_csv(os.path.join(run_dir, "nested_tests.csv")).set_index("test")
    dd = daily_differentials(fcs)
    rep, rows, terc = [], [], []
    bins = pd.qcut(zs, 3, labels=["low", "middle", "high"]).astype(str)
    for small, big, lab in NESTED_COMPARISONS:
        adj = dd[lab][2]
        assert list(adj.index) == list(days)
        for loss, col in (("MSE_log", "se"), ("QLIKE", "qlike")):
            d = D[small][col] - D[big][col]
            _, _, p = gw(d[1:], np.column_stack([np.ones(len(d) - 1), d[:-1]]))
            ps = float(stored.loc[lab, "gw_mse_p" if col == "se" else "gw_qlike_p"])
            rep.append(dict(test=lab, loss=loss, gw_p_recomputed=p, gw_p_stored=ps, abs_diff=abs(p - ps)))
            for sample in ("all", EXCL_LABEL):
                m = np.ones(len(d), bool) if sample == "all" else np.asarray(days != EXCL_DAY)
                st1, q1, p1 = gw(d[m], np.column_stack([np.ones(m.sum()), zs[m]]))
                m2 = m[1:] & m[:-1]                     # h_t = (1, d_t, s_t): neither d_t nor d_{t+1} on the excluded day
                st2, q2, p2 = gw(d[1:][m2], np.column_stack([np.ones(m2.sum()), d[:-1][m2], zs[1:][m2]]))
                rows.append(dict(test=lab, small=small, big=big, series=loss, sample=sample, n_days=int(m.sum()),
                                 hac_lags=lags, mean_d=float(d[m].mean()), gw_1s_stat=st1, gw_1s_df=q1, gw_1s_p=p1,
                                 gw_1ds_stat=st2, gw_1ds_df=q2, gw_1ds_p=p2, **hac_slope(d[m], zs[m], lags)))
            for b in ("low", "middle", "high"):
                k = np.asarray(bins == b)
                terc.append(dict(test=lab, series=loss, state_tercile=b, n_days=int(k.sum()),
                                 mean_state_z=float(zs[k].mean()), mean_d=float(d[k].mean())))
        for sample in ("all", EXCL_LABEL):
            m = np.ones(len(days), bool) if sample == "all" else np.asarray(days != EXCL_DAY)
            rows.append(dict(test=lab, small=small, big=big, series="CW_adjusted", sample=sample, n_days=int(m.sum()),
                             hac_lags=lags, mean_d=float(adj.to_numpy()[m].mean()),
                             **hac_slope(adj.to_numpy()[m], zs[m], lags)))
        for b in ("low", "middle", "high"):
            k = np.asarray(bins == b)
            terc.append(dict(test=lab, series="CW_adjusted", state_tercile=b, n_days=int(k.sum()),
                             mean_state_z=float(zs[k].mean()), mean_d=float(adj.to_numpy()[k].mean())))
    rep = pd.DataFrame(rep)
    assert (rep["abs_diff"] < 1e-10).all(), "the stored Giacomini-White p-values are not reproduced"
    out = pd.DataFrame(rows)
    for col in ("slope_p", "gw_1s_p", "gw_1ds_p"):
        out[col + "_holm"] = np.nan
    for (series, sample), g in out[out["test"].isin(STEPS)].groupby(["series", "sample"]):
        for col in ("slope_p", "gw_1s_p", "gw_1ds_p"):
            if g[col].notna().all():
                out.loc[g.index, col + "_holm"] = holm(g[col].to_numpy(), cfg.holm_alpha)["p_holm"]
    info = dict(state_mean=float(s.mean()), state_sd=float(s.std()), first_target=str(days[0].date()),
                last_target=str(days[-1].date()), n_days=len(days))
    return dict(reproduction=rep, tests=out, terciles=pd.DataFrame(terc), info=info)


# ----------------------------------------------------------------------------- part 2
def fit(builder, cfg: EngineConfig, state: pd.Series, spec: str) -> Dict:
    """spec: 'base_all_rows' (no state; all rows), 'base_state_days' (no state; days with a defined state),
    'parsimonious', 'full_interaction'."""
    from linearmodels.panel import PanelOLS
    from .inference import _zdesign, dk_cov_within, insample_design
    d = insample_design(builder, "R3")
    Z, names, mu, sd, keep = _zdesign(d)
    blocks = [b for b, k in zip(d.blocks, keep) if k]
    origin = builder.days[d.day_tr]
    s = state.reindex(origin).to_numpy(float)
    use = np.ones(len(s), bool) if spec == "base_all_rows" else np.isfinite(s)
    ou = origin[use]
    assert len(np.unique(ou)) == len(pd.date_range(ou.min(), ou.max(), freq="D")), "the days used are not contiguous"
    Zu, yu = Z[use], d.y_tr[use]
    coins = np.array(builder.coins)[d.coin_tr][use]
    cols, X, scale = list(names), [Zu], list(sd)
    state_mean, state_sd = np.nan, np.nan
    if spec in ("parsimonious", "full_interaction"):
        su = s[use]
        state_mean, state_sd = float(su.mean()), float(su.std())
        zs = (su - state_mean) / state_sd
        if spec == "parsimonious":
            inter = [n for n, b in zip(names, blocks) if b == "har"] + FEAR + ANGER
            assert len(inter) == 12, inter
        else:
            inter = list(names)
        ix = [names.index(t) for t in inter]
        X += [zs[:, None], Zu[:, ix] * zs[:, None]]
        cols += ["state"] + [f"{t}|state" for t in inter]
        scale += [1.0] + [float(sd[i]) for i in ix]
    Xm = np.hstack(X)
    idx = pd.MultiIndex.from_arrays([coins, ou], names=["coin", "day"])
    kw = dict(cov_type="kernel", kernel="bartlett")
    if cfg.dk_bandwidth is not None:
        kw["bandwidth"] = cfg.dk_bandwidth
    res = PanelOLS(pd.Series(yu, index=idx, name="y"), pd.DataFrame(Xm, index=idx, columns=cols),
                   entity_effects=True).fit(**kw)
    b, V = res.params.to_numpy(), res.cov.to_numpy()
    grp = pd.factorize(coins)[0]
    ud, day_pos = np.unique(ou, return_inverse=True)
    bw = cfg.dk_bandwidth if cfg.dk_bandwidth is not None else float(np.floor(4 * (len(ud) / 100) ** (2 / 9)))
    b2, V2 = dk_cov_within(Xm, yu, grp, day_pos, bw)
    chk_b, chk_se = float(np.max(np.abs(b2 - b))), float(np.max(np.abs(np.sqrt(np.diag(V2)) / np.sqrt(np.diag(V)) - 1)))
    assert chk_b < 1e-8 and chk_se < 1e-6, (spec, chk_b, chk_se)
    sc = np.asarray(scale)
    braw, Vraw = b / sc, V / np.outer(sc, sc)
    se = np.sqrt(np.diag(Vraw))
    coef = pd.DataFrame(dict(spec=spec, variable=cols, coef_raw=braw, se_dk_raw=se, t_dk=braw / se,
                             p_dk=2 * stats.norm.sf(np.abs(braw / se))))
    info = dict(spec=spec, n_rows=int(len(yu)), n_days=int(len(ud)), first_origin=str(pd.Timestamp(ou.min()).date()),
                last_origin=str(pd.Timestamp(ou.max()).date()), k=int(Xm.shape[1]), dk_bandwidth=bw,
                r2_within=float(res.rsquared_within), state_mean=state_mean, state_sd=state_sd,
                max_abs_diff_beta_vs_engine_dk=chk_b, max_rel_diff_se_vs_engine_dk=chk_se)
    return dict(b=braw, V=Vraw, cols=cols, coef=coef, info=info)


def combos(f: Dict) -> pd.DataFrame:
    b, V, cols, spec = f["b"], f["V"], f["cols"], f["info"]["spec"]

    def comb(label, plus, minus=()):
        a = np.zeros(len(cols))
        for t, w in plus:
            a[cols.index(t)] += w
        for t, w in minus:
            a[cols.index(t)] -= w
        est, se = float(a @ b), float(np.sqrt(a @ V @ a))
        return dict(spec=spec, quantity=label, estimate=est, se_dk=se, z=est / se, p_two_sided=float(2 * stats.norm.sf(abs(est / se))))

    main = lambda terms: [(t, 1.0) for t in terms]
    if "state" not in cols:
        return pd.DataFrame([comb("Cum(fear)", main(FEAR)), comb("Cum(anger)", main(ANGER)),
                             comb("Delta = Cum(fear) - Cum(anger)", main(FEAR), main(ANGER))])
    sl = lambda terms, w=1.0: [(f"{t}|state", w) for t in terms]
    rows = [comb("state main effect", [("state", 1.0)]),
            comb("Cum(fear) at the mean state", main(FEAR)),
            comb("Cum(anger) at the mean state", main(ANGER)),
            comb("Delta = Cum(fear) - Cum(anger) at the mean state", main(FEAR), main(ANGER)),
            comb("slope of Cum(fear) per SD of the state", sl(FEAR)),
            comb("slope of Cum(anger) per SD of the state", sl(ANGER)),
            comb("slope of Delta per SD of the state (the test)", sl(FEAR), sl(ANGER))]
    for lab, w in (("-1 SD", -1.0), ("+1 SD", 1.0)):
        rows += [comb(f"Cum(fear) at {lab}", main(FEAR) + sl(FEAR, w)),
                 comb(f"Cum(anger) at {lab}", main(ANGER) + sl(ANGER, w)),
                 comb(f"Delta at {lab}", main(FEAR) + sl(FEAR, w), main(ANGER) + sl(ANGER, w))]
    return pd.DataFrame(rows)


def check_stored_h4(base: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    hb = pd.read_csv(os.path.join(run_dir, "h4_bootstrap.csv"))
    rows = []
    for q, key in (("Cum(fear)", "theta1"), ("Delta = Cum(fear) - Cum(anger)", "theta2")):
        h = hb[hb["condition"].str.startswith(key)].iloc[0]
        r = base[base["quantity"] == q].iloc[0]
        rows.append(dict(quantity=q, estimate=r["estimate"], stored_estimate=h["estimate"], se_dk=r["se_dk"],
                         stored_dk_se=h["dk_se"], abs_diff_estimate=abs(r["estimate"] - h["estimate"]),
                         abs_diff_se=abs(r["se_dk"] - h["dk_se"])))
    out = pd.DataFrame(rows)
    assert (out[["abs_diff_estimate", "abs_diff_se"]].to_numpy() < 1e-8).all(), "the stored H4 row is not reproduced"
    return out


# ----------------------------------------------------------------------------- main run
def run(args) -> str:
    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(args.out_root, f"A73_{args.label}_{stamp}")
    os.makedirs(out, exist_ok=False)
    inputs = {f"run_dir/{f}": os.path.join(args.run_dir, f)
              for f in ("forecasts.parquet", "config.json", "nested_tests.csv", "h4_bootstrap.csv", "insample_coef.csv")}
    inputs.update({"panel": args.panel, "variable_spec": args.variable_spec})
    in_md5 = {k: md5(v) for k, v in inputs.items()}
    kl = sorted(glob.glob(os.path.join(args.klines_dir, "BTCUSDT-1m-*.zip")))
    in_md5.update({f"klines/{os.path.basename(f)}": md5(f) for f in kl})
    if args.expect_forecasts_md5:
        assert in_md5["run_dir/forecasts.parquet"] == args.expect_forecasts_md5, "forecasts.parquet md5 differs"
    if args.expect_panel_md5:
        assert in_md5["panel"] == args.expect_panel_md5, "panel md5 differs"
    tab, state = state_series(args.klines_dir)
    write_csv(tab, os.path.join(out, "state_series.csv"))
    cfg, fcs, _ = load_run(args.run_dir)

    p1 = part1(fcs, cfg, state, args.run_dir)
    write_csv(p1["reproduction"], os.path.join(out, "part1_gw_reproduction_check.csv"))
    write_csv(p1["tests"], os.path.join(out, "part1_state_tests.csv"))
    write_csv(p1["terciles"], os.path.join(out, "part1_state_terciles.csv"))

    builder = build_builder(args.panel, args.variable_spec, cfg)
    write_csv(insample_check(builder, cfg, args.run_dir), os.path.join(out, "part2_insample_reproduction_check.csv"))
    fits = {s: fit(builder, cfg, state, s) for s in ("base_all_rows", "base_state_days", "parsimonious", "full_interaction")}
    write_csv(check_stored_h4(combos(fits["base_all_rows"]), args.run_dir), os.path.join(out, "part2_h4_reproduction_check.csv"))
    write_csv(pd.concat([combos(f) for f in fits.values()], ignore_index=True), os.path.join(out, "part2_state_lincomb.csv"))
    write_csv(pd.concat([f["coef"] for f in fits.values() if f["info"]["spec"] != "base_all_rows"], ignore_index=True),
              os.path.join(out, "part2_state_coef.csv"))
    write_csv(pd.DataFrame([f["info"] for f in fits.values()]), os.path.join(out, "part2_state_fit_info.csv"))

    cfg_out = dict(arm="A73", label=args.label, stamp_utc=stamp, post_result=True,
                   note="post-result analysis: the state variable was chosen on 9 Nov 2026, after the results were known",
                   state="log(BTCUSDT UTC close / 200-day simple moving average of the close), known at the origin day",
                   state_source="moving average of Brock, Lakonishok & LeBaron (1992); the continuous log distance is our own choice",
                   part1_state=p1["info"], sensitivity_day=str(EXCL_DAY.date()),
                   engine_config_of_evaluated_run=json.loads(cfg.to_json()), inputs_md5=in_md5)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg_out, fh, indent=2, default=str)
    with open(os.path.join(out, "commands.txt"), "w") as fh:
        fh.write("python -m model.arm_state_dependent " + " ".join(sys.argv[1:]) + "\n")
    write_md5sums(out, in_md5)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m model.arm_state_dependent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for a in ("--run-dir", "--panel", "--variable-spec", "--klines-dir", "--out-root"):
        r.add_argument(a, required=True)
    r.add_argument("--label", default="state")
    r.add_argument("--stamp", default=None)
    r.add_argument("--expect-forecasts-md5", default=None)
    r.add_argument("--expect-panel-md5", default=None)
    print(run(ap.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
