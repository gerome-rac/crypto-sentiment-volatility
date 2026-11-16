#!/usr/bin/env python
"""oos_appendix_tables_20261108.py - numbers, tables and figure data for the appendix section
"Further out-of-sample evaluation" (Appendix D), on the MedRV model (D8 Deviation 7, 8 Nov 2026).
Same computations as oos_appendix_tables_20261107.py (BV model); only the runs and two inputs differ.

Reads the stored outputs (no re-estimation): the MedRV ladder (final run A00_final_A50 of 28 Oct 2026) and the
  MedRV re-runs of 8 Nov 2026 of the 250-day window (A06), the expanding window (A55), the Duan back-transformation
  (A61) and the coin-by-coin estimation (A01, member of the sample/window family); the main window's Holm values come
  from confirmatory_sq4dk.csv (family of four with the two-sided SQ4 test). The sample/window family Holm is
  recomputed from the three family members (the 7 Nov version read it from the arm summary of 28 Oct; the
  recomputation reproduces that summary on the BV runs before anything is written).
Computes one new quantity: the dependence sensitivity of the Clark-West p-values (HAC bandwidths 6/14/30/60;
  stationary block bootstrap of the HAC t-statistic with mean block lengths 10/30/60 days), using the engine's
  own clark_west / hac_var (code/model/tests_eval.py) and stationary bootstrap (code/model/inference.py).
  The 6-lag p-values must reproduce the stored nested_tests.csv exactly before anything is written.
Writes (temp file -> checks -> replace):
  <tables_dir>/app_oos_*.tex              booktabs tables, values formatted by code
  <tables_dir>/src/app_oos_values_20261108.csv   every number shown, unformatted
  <fig_src_dir>/cum_loss_diff_medrv_20261108.csv   input of the cumulative loss figure
Usage: python oos_appendix_tables_20261108.py <code_dir> <tables_dir> <fig_src_dir>
       (run with the engine's environment, e.g. ~/.claude-science/conda/envs/harmodel/bin/python)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

code_dir, tab_dir, fig_src = map(Path, sys.argv[1:4])
sys.path.insert(0, str(code_dir))
from model.tests_eval import clark_west, hac_var, holm, loss_frame  # noqa: E402
from model.inference import stationary_block_bootstrap_days        # noqa: E402

RUNS = code_dir / "data" / "pipeline" / "model_runs"
def one(pattern):
    hits = sorted(p.name for p in RUNS.glob(pattern))
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


RUN = {"main": "A00_final_A50_20261028T134243Z", "w250": one("A06_final_A06_medrv_*"),
       "expanding": one("A55_final_A55_medrv_*"), "duan": one("A61_final_A61_medrv_*"),
       "coin": one("A01_final_A01_medrv_*")}
BV_FAMILY = {"A06": "A06_final_A06_20261028T115704Z", "A55": "A55_final_A55_20261028T120152Z",
             "A01": "A01_final_A01_20261028T120727Z"}
CONF_SQ4 = RUNS / "confirmatory_sq4dk_20261108" / "confirmatory_sq4dk.csv"
STEPS = [("H1", "R0", "R1", "SQ1"), ("H2", "R1", "R2", "SQ2"), ("H3", "R2", "R3", "SQ3")]
COINS = ["BTC", "ETH", "BNB", "XRP", "DOGE"]
LAGS = [6, 14, 30, 60]
BLOCKS = [10, 30, 60]
B, SEED = 9999, 20261107
values = []          # (table, row, column, value)


def rd(run, name):
    return pd.read_csv(RUNS / RUN[run] / name)


def fp(p):
    """p-value for the tables: three decimals, '<0.001' below."""
    return "$<$0.001" if p < 0.0005 else f"{p:.3f}"


def fpct(x):
    return f"{100 * x:+.2f}"


def write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    assert tmp.read_text(encoding="utf-8") == text
    os.replace(tmp, path)


# ---------------------------------------------------------------------------- main run: CW check
fc = pd.read_parquet(RUNS / RUN["main"] / "forecasts.parquet")
L = {r: loss_frame(fc[fc.rung == r]).sort_values(["target", "coin"]).reset_index(drop=True)
     for r in ["R0", "R1", "R2", "R3"]}
ref = L["R0"][["coin", "target"]]
assert all(L[r][["coin", "target"]].equals(ref) for r in L) and len(ref) == 885 * 5
tgt = ref["target"]
nested = rd("main", "nested_tests.csv").set_index("test")
dep_rows = []
for lab, small, big, sq in STEPS:
    y, f1, f2 = L[big].y.to_numpy(), L[small].f_log.to_numpy(), L[big].f_log.to_numpy()
    base = clark_west(y, f1, f2, groups=tgt, lags=6)
    assert abs(base["p_one_sided"] - nested.loc[lab, "cw_p"]) < 1e-12, (lab, base, nested.loc[lab, "cw_p"])
    assert abs(base["cw_stat"] - nested.loc[lab, "cw_stat"]) < 1e-10
    row = {"step": sq, "small": small, "big": big, "cw_stat_L6": base["cw_stat"]}
    for lag in LAGS:
        row[f"p_hac_L{lag}"] = clark_west(y, f1, f2, groups=tgt, lags=lag)["p_one_sided"]
    # daily cross-coin mean of the adjusted differential, as inside clark_west
    adj = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    c = pd.Series(adj).groupby(tgt.to_numpy()).mean().sort_index().to_numpy()
    T, m = len(c), c.mean()
    t_obs = m / np.sqrt(hac_var(c, 6) / T)
    assert abs(t_obs - base["cw_stat"]) < 1e-10
    for blk in BLOCKS:
        cnt = 0
        for idx in stationary_block_bootstrap_days(T, blk, B, SEED + blk):
            cs = c[idx]
            v = hac_var(cs, 6)
            if v > 0 and (cs.mean() - m) / np.sqrt(v / T) >= t_obs:
                cnt += 1
        row[f"p_boot_b{blk}"] = (1 + cnt) / (B + 1)
    dep_rows.append(row)
dep = pd.DataFrame(dep_rows)
for _, r in dep.iterrows():
    for k, v in r.items():
        if k.startswith("p_") or k.startswith("cw_"):
            values.append(("dependence", r["step"], k, v))

# ---------------------------------------------------------------------------- figure data
cl = pd.read_parquet(RUNS / RUN["main"] / "cum_loss_diff.parquet")
ls_main = rd("main", "loss_summary.csv").set_index("model")
assert len(cl) == 885
for r in ["R1", "R2", "R3"]:          # last cumulative value / 885 = full-sample loss difference to R0
    for col, lo in (("dse", "MSE_log"), ("dqlike", "QLIKE")):
        got = cl[f"cum_{col}_R0_minus_{r}"].iloc[-1] / 885
        want = ls_main.loc["R0", lo] - ls_main.loc[r, lo]
        assert abs(got - want) < 1e-10, (r, col, got, want)
fig = cl[["target"] + [f"cum_{a}_R0_minus_{r}" for a in ("dse", "dqlike") for r in ("R1", "R2", "R3")]]
fig_path = fig_src / "cum_loss_diff_medrv_20261108.csv"
write(fig_path, fig.to_csv(index=False))

# ---------------------------------------------------------------------------- per-coin table
pc = rd("main", "per_coin.csv")
lines = []
for lab, small, big, sq in STEPS:
    lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{sq}: {small} $\to$ {big}}}}} \\")
    for coin in COINS:
        r = pc[(pc.coin == coin) & (pc.test == lab)].iloc[0]
        lines.append(f"{coin} & {fpct(r.r2_oos)} & {r.cw_stat:.2f} & {fp(r.cw_p)} & {fp(r.cw_p_holm_within_test)} & "
                     f"{r.dm_qlike:+.2f} & {fp(r.dm_qlike_p2)} \\\\")
        for k in ("r2_oos", "cw_stat", "cw_p", "cw_p_holm_within_test", "dm_qlike", "dm_qlike_p2"):
            values.append(("per_coin", f"{sq}/{coin}", k, r[k]))
    lines.append(r"\addlinespace")
tab_coin = r"""\begin{table}[htbp]
\centering
\caption{Out-of-sample evaluation of the pooled forecasts by coin}\label{tab:app_percoin}
\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & OOS $R^2$ (\%) & \multicolumn{3}{c}{Clark--West} & \multicolumn{2}{c}{Diebold--Mariano, QLIKE} \\
\cmidrule(lr){3-5}\cmidrule(l){6-7}
Coin & & $t$ & $p$ & $p$ (Holm) & $t$ & $p$ \\
\midrule
""" + "\n".join(lines[:-1]) + r"""
\bottomrule
\end{tabular}
\srcnote{Own calculations from the final run (885 forecast days per coin, 30~August 2020--31~January 2023). OOS $R^2$: reduction of the larger rung's MSPE relative to the smaller rung's, in percent. Clark--West: one-sided test that the larger rung improves the forecasts, on the coin's own daily series; $p$ (Holm) adjusts across the five coins within each step. Diebold--Mariano on QLIKE with the small-sample correction of \textcite[p.~283]{harvey1997}: a positive $t$ favors the larger rung; $p$ is two-sided.}
\end{table}
"""
write(tab_dir / "app_oos_percoin.tex", tab_coin)

# ---------------------------------------------------------------------------- Giacomini-White table
gw_lines = []
for run, name in (("main", "365-day window (main)"), ("w250", "250-day window")):
    nt = rd(run, "nested_tests.csv").set_index("test")
    adj = {}
    for lo in ("mse", "qlike"):
        ok = [lab for lab, *_ in STEPS if not bool(nt.loc[lab, f"gw_{lo}_degenerate"])]
        hp = holm([nt.loc[lab, f"gw_{lo}_p"] for lab in ok])["p_holm"]
        adj[lo] = dict(zip(ok, hp))
    gw_lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{name}}}}} \\")
    for lab, small, big, sq in STEPS:
        cells = []
        for lo in ("mse", "qlike"):
            if bool(nt.loc[lab, f"gw_{lo}_degenerate"]):
                cells += ["--", "--", "--"]
            else:
                cells += [f"{nt.loc[lab, f'gw_{lo}_stat']:.2f}", fp(nt.loc[lab, f"gw_{lo}_p"]), fp(adj[lo][lab])]
                values += [("gw", f"{run}/{sq}", f"{lo}_stat", nt.loc[lab, f"gw_{lo}_stat"]),
                           ("gw", f"{run}/{sq}", f"{lo}_p", nt.loc[lab, f"gw_{lo}_p"]),
                           ("gw", f"{run}/{sq}", f"{lo}_p_holm", adj[lo][lab])]
        gw_lines.append(f"{sq}: {small} $\\to$ {big} & " + " & ".join(cells) + r" \\")
    gw_lines.append(r"\addlinespace")
tab_gw = r"""\begin{table}[htbp]
\centering
\caption{Conditional predictive ability of each rung step}\label{tab:app_gw}
\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & \multicolumn{3}{c}{MSPE} & \multicolumn{3}{c}{QLIKE} \\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
Step & $\chi^2(2)$ & $p$ & $p$ (Holm) & $\chi^2(2)$ & $p$ & $p$ (Holm) \\
\midrule
""" + "\n".join(gw_lines[:-1]) + r"""
\bottomrule
\end{tabular}
\srcnote{Own calculations from the final runs. Test of \textcite[pp.~1545--1546, 1554, Theorem~1]{giacomini2006} on the daily cross-coin mean loss differences, with the instruments $(1, \bar{d}_t)$ for $\bar{d}_{t+1}$; it rejects if the loss difference is predictable from its own past, i.e.\ if one rung is better in some market states. It is valid only for rolling windows of fixed length and is therefore not computed for the expanding window. $p$ (Holm) adjusts across the three steps within each loss and window.}
\end{table}
"""
write(tab_dir / "app_oos_gw.tex", tab_gw)

# ---------------------------------------------------------------------------- dependence table
dl = []
for _, r in dep.iterrows():
    dl.append(f"{r.step}: {r.small} $\\to$ {r.big} & " + " & ".join(fp(r[f'p_hac_L{lag}']) for lag in LAGS) + " & "
              + " & ".join(fp(r[f'p_boot_b{b}']) for b in BLOCKS) + r" \\")
tab_dep = r"""\begin{table}[htbp]
\centering
\caption{Sensitivity of the Clark--West $p$-values to serial dependence}\label{tab:app_dependence}
\small
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
 & \multicolumn{4}{c}{HAC bandwidth} & \multicolumn{3}{c}{Block bootstrap} \\
\cmidrule(lr){2-5}\cmidrule(l){6-8}
Step & 6 (main) & 14 & 30 & 60 & 10 & 30 & 60 \\
\midrule
""" + "\n".join(dl) + r"""
\bottomrule
\end{tabular}
\srcnote{Own calculations from the final run's forecasts. One-sided Clark--West $p$-values, unadjusted, on the 885 daily cross-coin means of the adjusted loss difference. Bandwidths and mean block lengths in days. Left: Bartlett HAC standard errors \parencite[pp.~703--704]{newey1987} with the pre-specified six lags and three longer bandwidths. Right: stationary block bootstrap \parencite[pp.~1303--1304]{politis1994} of the HAC $t$-statistic, recentered at the sample mean, with 9,999 replications \parencite[pp.~745, 758]{goncalvesBlockBootstrapHac2011}. The decision rule of \cref{sec:oos} uses the six-lag $p$-value.}
\end{table}
"""
write(tab_dir / "app_oos_dependence.tex", tab_dep)

# ---------------------------------------------------------------------------- estimation windows table
arm = pd.read_csv(RUNS / "_arm_results_final_20261028.csv").set_index("arm_id")


def family_holm(dirs):
    """Holm across the sample/window family members, per step, on their one-sided Clark-West p-values."""
    p = {k: pd.read_csv(RUNS / d / "nested_tests.csv").set_index("test") for k, d in dirs.items()}
    out = {}
    for lab, *_ in STEPS:
        keys = list(dirs)
        hp = holm([p[k].loc[lab, "cw_p"] for k in keys])["p_holm"]
        out.update({(k, lab): v for k, v in zip(keys, hp)})
    return out


bv_fh = family_holm(BV_FAMILY)            # must reproduce the arm summary of 28 Oct (BV runs)
for (k, lab), v in bv_fh.items():
    assert abs(v - arm.loc[k, f"{lab}_cw_p_holm_family"]) < 1e-12, (k, lab, v)
fh = family_holm({"A06": RUN["w250"], "A55": RUN["expanding"], "A01": RUN["coin"]})
c4 = pd.read_csv(CONF_SQ4)
c4 = c4[c4.run == RUN["main"]].set_index("subquestion")
conf = pd.DataFrame({"p_holm": [c4.loc[sq, "p_holm"] for *_, sq in STEPS]}, index=[lab for lab, *_ in STEPS])
nt_main = rd("main", "nested_tests.csv").set_index("test")
assert all(abs(c4.loc[sq, "p"] - nt_main.loc[lab, "cw_p"]) < 1e-12 for lab, _, _, sq in STEPS)
wl = []
for run, name, arm_id in (("main", "365-day rolling (main)", None), ("w250", "250-day rolling", "A06"),
                          ("expanding", "Expanding from 30 Aug 2019", "A55")):
    nt = rd(run, "nested_tests.csv").set_index("test")
    cells = []
    for lab, *_ in STEPS:
        p = nt.loc[lab, "cw_p"]
        ph = conf.loc[lab, "p_holm"] if arm_id is None else fh[(arm_id, lab)]
        r2 = nt.loc[lab, "r2_oos_big_vs_small"]
        cells += [fp(p), fp(ph), fpct(r2)]
        values += [("windows", f"{run}/{lab}", "cw_p", p), ("windows", f"{run}/{lab}", "p_holm", ph),
                   ("windows", f"{run}/{lab}", "r2_oos", r2)]
    wl.append(f"{name} & " + " & ".join(cells) + r" \\")
fam_size = {lab: 3 for lab, *_ in STEPS}
assert all(int(arm.loc["A06", f"{lab}_family_size"]) == 3 for lab, *_ in STEPS)
tab_win = r"""\begin{table}[htbp]
\centering
\caption{The ladder under alternative estimation windows}\label{tab:app_windows}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
 & \multicolumn{3}{c}{SQ1: R0 $\to$ R1} & \multicolumn{3}{c}{SQ2: R1 $\to$ R2} & \multicolumn{3}{c}{SQ3: R2 $\to$ R3} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}
Window & $p$ & $p$ (Holm) & $R^2$ (\%) & $p$ & $p$ (Holm) & $R^2$ (\%) & $p$ & $p$ (Holm) & $R^2$ (\%) \\
\midrule
""" + "\n".join(wl) + r"""
\bottomrule
\end{tabular}
\srcnote{Own calculations from the final runs; every variant forecasts the same 885 target days. $p$: one-sided Clark--West $p$-value. $p$ (Holm): for the main window, the confirmatory adjustment across the four subquestions (\cref{sec:oos}); for the other windows, the adjustment within the family of """ + str(fam_size["H1"]) + r""" sample and window variants, which also contains the coin-by-coin estimation of \cref{sec:model}. $R^2$: out-of-sample $R^2$ of the larger rung relative to the smaller one, in percent.}
\end{table}
"""
write(tab_dir / "app_oos_windows.tex", tab_win)

# ---------------------------------------------------------------------------- Duan table
ls_d = rd("duan", "loss_summary.csv").set_index("model")
nt_m, nt_d = rd("main", "nested_tests.csv").set_index("test"), rd("duan", "nested_tests.csv").set_index("test")
conf_d = rd("duan", "confirmatory_holm.csv").set_index("hypothesis")
assert np.allclose(ls_d.MSE_log.to_numpy(), ls_main.loc[ls_d.index, "MSE_log"].to_numpy(), rtol=0, atol=1e-14)
dul = []
for r in ["R0", "R1", "R2", "R3"]:
    dul.append(f"{r} & {ls_main.loc[r, 'QLIKE']:.4f} & {ls_d.loc[r, 'QLIKE']:.4f} & & & \\\\")
    values += [("duan", r, "qlike_lognormal", ls_main.loc[r, "QLIKE"]), ("duan", r, "qlike_duan", ls_d.loc[r, "QLIKE"])]
dul.append(r"\addlinespace")
for lab, small, big, sq in STEPS:
    qa = conf_d.loc[lab, "qlike_points_against"]          # NaN for SQ2, which has its own three-condition rule
    qa_txt = "--" if pd.isna(qa) else ("yes" if str(qa).strip().lower() == "true" else "no")
    assert pd.isna(qa) == (lab == "H2"), (lab, qa)
    dul.append(f"{sq}: {small} $\\to$ {big} & & & {nt_m.loc[lab, 'dm_qlike']:+.2f} ({fp(nt_m.loc[lab, 'dm_qlike_p2'])}) & "
               f"{nt_d.loc[lab, 'dm_qlike']:+.2f} ({fp(nt_d.loc[lab, 'dm_qlike_p2'])}) & {qa_txt} \\\\")
    values += [("duan", sq, "dm_qlike_lognormal", nt_m.loc[lab, "dm_qlike"]), ("duan", sq, "dm_qlike_p2_lognormal", nt_m.loc[lab, "dm_qlike_p2"]),
               ("duan", sq, "dm_qlike_duan", nt_d.loc[lab, "dm_qlike"]), ("duan", sq, "dm_qlike_p2_duan", nt_d.loc[lab, "dm_qlike_p2"])]
tab_duan = r"""\begin{table}[htbp]
\centering
\caption{QLIKE under the lognormal and the smearing back-transformation}\label{tab:app_duan}
\small
\begin{tabular}{@{}lrrrrc@{}}
\toprule
 & \multicolumn{2}{c}{Mean QLIKE} & \multicolumn{2}{c}{Diebold--Mariano $t$ ($p$)} & Reverses \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
 & Lognormal & Smearing & Lognormal & Smearing & under QLIKE \\
\midrule
""" + "\n".join(dul) + r"""
\bottomrule
\end{tabular}
\srcnote{Own calculations from the final runs. The log forecasts are identical; only their conversion to variance forecasts differs: the lognormal correction $\exp(\hat{f} + \hat{\sigma}_w^2/2)$ of \cref{eq:qlike_retransform} or the smearing estimator $\exp(\hat{f})\, n_w^{-1}\sum_{s \in w} \exp(\hat{\varepsilon}_s)$ of \textcite[pp.~605--606]{duan1983}, both computed inside each window. Diebold--Mariano on QLIKE with the small-sample correction of \textcite[p.~283]{harvey1997}: a positive $t$ favors the larger rung; $p$ is two-sided. ``Reverses'': the rule of \cref{sec:oos} that downgrades an MSPE improvement if QLIKE significantly favors the smaller rung; it does not apply to SQ2, which has its own rule.}
\end{table}
"""
write(tab_dir / "app_oos_duan.tex", tab_duan)

# ---------------------------------------------------------------------------- all values
vals = pd.DataFrame(values, columns=["table", "row", "column", "value"])
write(tab_dir / "src" / "app_oos_values_20261108.csv", vals.to_csv(index=False))
print(dep.to_string(index=False))
print("written:", sorted(p.name for p in tab_dir.glob("app_oos_*.tex")), "| values:", len(vals))
