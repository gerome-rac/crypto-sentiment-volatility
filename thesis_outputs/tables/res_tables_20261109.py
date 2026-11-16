#!/usr/bin/env python
"""res_tables_20261109.py - tables of Chapter 5 (Results) and the two appendix tables it points to, on the MedRV model
(D8 Deviation 7). Every number comes from stored run outputs or the stored panel; the formatting is done here.

Writes (temp file -> checks -> replace):
  <tables_dir>/res_text_descriptives.tex, res_ladder.tex, res_sq4.tex, res_answers.tex     Chapter 5
  <tables_dir>/app_validation.tex, app_regime.tex                                          Appendices B and D
  <tables_dir>/src/res_values_20261109.csv       every number shown, unformatted
  <fig_src_dir>/emotions_rv_btc_20261109.csv     input of the figure of Section 5.1
Usage: python res_tables_20261109.py <code_dir> <tables_dir> <fig_src_dir> <duckdb_tmp_dir>
"""
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

code_dir, tab_dir, fig_src, ddir = map(Path, sys.argv[1:5])
sys.path.insert(0, str(code_dir))
os.chdir(code_dir)
from model.arm_regime_split import build_builder, load_run                 # noqa: E402
from model.arm_state_dependent import ANGER, FEAR, fit, hac_slope          # noqa: E402

D = Path("data/pipeline")
RUNS = D / "model_runs"
PRIM = RUNS / "A00_final_A50_20261028T134243Z"
PANEL = D / "panel/arms/final_20261028/A50.parquet"
SPEC = "model/variable_spec_A50.csv"
A07 = RUNS / "A07_bullbear_medrv_20261109T084736Z"
A73 = RUNS / "A73_state_medrv_20261109T090959Z"
COINS = ["BTC", "ETH", "BNB", "XRP", "DOGE"]
EMO = [("fear", "Fear"), ("anger", "Anger"), ("sad", "Sad"), ("disgust", "Disgust"), ("surprise", "Surprise"), ("happy", "Happy")]
values = []


def val(tab, row, col, x):
    values.append(dict(table=tab, row=row, column=col, value=x))
    return x


def fp(p):
    return "$<$0.001" if p < 0.0005 else f"{p:.3f}"


def write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    assert tmp.read_text(encoding="utf-8") == text
    os.replace(tmp, path)


# --------------------------------------------------------------------------- Table: text measures
p = pd.read_parquet(PANEL)
p["day"] = pd.to_datetime(p["day"])
assert len(p) == 6400
s = p[(p.day >= "2019-08-30") & (p.day <= "2023-01-30")].copy()
assert len(s) == 6250
cols = [f"emo_{e}" for e, _ in EMO] + ["pol"]
sd = s[cols].std()                               # over all coin-days, as the SDs quoted in Section 4.3
assert abs(sd["emo_fear"] - 0.0139) < 0.0005 and abs(sd["emo_anger"] - 0.0092) < 0.0005, sd
rows = []
for c, lab in zip(cols, [l for _, l in EMO] + ["Polarity"]):
    cells = [f"{val('res_text', lab, k, s.loc[s.coin == k, c].mean()):.3f}" for k in COINS]
    rows.append(f"{lab} & " + " & ".join(cells) + f" & {val('res_text', lab, 'SD', sd[c]):.3f} \\\\")
n = [f"{val('res_text', 'posts', k, s.loc[s.coin == k, 'text_n_posts'].median()):,.0f}" for k in COINS]
tex = r"""\begin{table}[htbp]
\centering
\caption{Daily emotion and polarity measures by coin}\label{tab:res_text}
\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & BTC & ETH & BNB & XRP & DOGE & SD \\
\midrule
""" + "\n".join(rows) + r"""
\addlinespace
Posts per day (median) & """ + " & ".join(n) + r""" & \\
\bottomrule
\end{tabular}
\srcnote{Own calculations; 1{,}250 days per coin, 30~August 2019--30~January 2023. Means of the pooled daily index of \cref{sec:emotionmeasure}: the six probabilities in $[0,1]$ and FinBERT polarity in $[-1,1]$. SD: standard deviation over all coin-days.}
\end{table}
"""
write(tab_dir / "res_text_descriptives.tex", tex)

# --------------------------------------------------------------------------- figure data
b = p[p.coin == "BTC"].sort_values("day")
fd = pd.DataFrame(dict(day=b.day.dt.strftime("%Y-%m-%d"), log_rv=np.log(b.rv), fear=b.emo_fear, anger=b.emo_anger))
for c in ("fear", "anger"):
    fd[c + "_7d"] = fd[c].rolling(7, min_periods=7).mean()
assert len(fd) == 1280
fd.to_csv(fig_src / "emotions_rv_btc_20261109.csv.tmp", index=False)
os.replace(fig_src / "emotions_rv_btc_20261109.csv.tmp", fig_src / "emotions_rv_btc_20261109.csv")

# --------------------------------------------------------------------------- Table: the ladder
ls = pd.read_csv(PRIM / "loss_summary.csv").set_index("model")
nt = pd.read_csv(PRIM / "nested_tests.csv").set_index("test")
mc = pd.read_csv(PRIM / "mcs.csv")
cq = pd.read_csv(RUNS / "confirmatory_sq4dk_20261108/confirmatory_sq4dk.csv")
cq = cq[cq.run == PRIM.name].set_index("subquestion")
assert len(cq) == 4
mp = {(r.loss, r.model): r.mcs_pvalue for r in mc[mc.alpha == 0.10].itertuples()}
for r in mc[mc.alpha == 0.25].itertuples():
    assert abs(mp[(r.loss, r.model)] - r.mcs_pvalue) < 1e-12
STEP = {"R1": ("H1", "SQ1"), "R2": ("H2", "SQ2"), "R3": ("H3", "SQ3")}
NAME = {"HARRV": "HAR-RV", "HARCJ": "Pure HAR-CJ", "R0": "R0: HAR-CJ + controls", "R1": "R1: + attention", "R2": "R2: + polarity", "R3": "R3: + emotions"}
rows = []
for m in ("HARRV", "HARCJ", "R0", "R1", "R2", "R3"):
    r = ls.loc[m]
    c = [NAME[m], f"{val('res_ladder', m, 'MSE', r.MSE_log):.4f}", f"{val('res_ladder', m, 'QLIKE', r.QLIKE):.4f}",
         f"{100 * val('res_ladder', m, 'R2oos_vs_R0', r.R2_oos_vs_R0):+.2f}" if m != "R0" else "---"]
    if m in STEP:
        h, q = STEP[m]
        t = nt.loc[h]
        c += [f"{100 * val('res_ladder', m, 'R2_step', t.r2_oos_big_vs_small):+.2f}", fp(val("res_ladder", m, "cw_p", t.cw_p)),
              fp(val("res_ladder", m, "holm_p", cq.loc[q, "p_holm"])), fp(val("res_ladder", m, "dm_mse_p", t.dm_mse_p_big_better)),
              fp(val("res_ladder", m, "dm_qlike_p", t.dm_qlike_p_big_better))]
    else:
        c += ["---"] * 5
    for lo in ("MSE_log", "QLIKE"):
        c.append(f"{val('res_ladder', m, 'mcs_' + lo, mp[(lo, m)]):.3f}" if (lo, m) in mp else "---")
    rows.append(" & ".join(c) + r" \\")
rows.insert(2, r"\addlinespace")
tex = r"""\begin{table}[htbp]
\centering
\caption{Out-of-sample results of the model ladder}\label{tab:res_ladder}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
 & & & \multicolumn{2}{c}{OOS $R^2$ (\%)} & \multicolumn{2}{c}{Clark--West} & \multicolumn{2}{c}{DM (one-sided)} & \multicolumn{2}{c}{MCS $p$} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}
Model & MSE & QLIKE & vs R0 & vs rung below & $p$ & Holm & MSE & QLIKE & MSE & QLIKE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\srcnote{Own calculations; 885 forecast days per coin, 30~August 2020--31~January 2023; rolling 365-day windows. MSE: squared error of log RV; QLIKE on the variance level. OOS $R^2$ from the MSE. Clark--West: one-sided $p$ for the step to the rung; Holm: adjusted over the family of four confirmatory tests including that of \cref{sq:4}. DM: one-sided Diebold--Mariano $p$ that the larger rung is better. MCS: $p$-value of the model confidence set over R0--R3 and HAR-RV; a model is in the 90\% (75\%) set if it is at least 0.10 (0.25). The pure HAR-CJ model is not in the set because R0 contains it.}
\end{table}
"""
write(tab_dir / "res_ladder.tex", tex)

# --------------------------------------------------------------------------- Table: SQ4
cfg, fcs, _ = load_run(str(PRIM))
builder = build_builder(str(PANEL), SPEC, cfg)
f = fit(builder, cfg, pd.Series(dtype=float), "base_all_rows")
bh, V, cn = f["b"], f["V"], f["cols"]
def lc(plus, minus=()):
    a = np.zeros(len(cn))
    for t in plus:
        a[cn.index(t)] += 1
    for t in minus:
        a[cn.index(t)] -= 1
    return float(a @ bh), float(np.sqrt(a @ V @ a))
def cell(est, se, tab_row, col):
    val("res_sq4", tab_row, col + "_est", est); val("res_sq4", tab_row, col + "_se", se)
    return f"{est:+.1f} ({se:.1f})"
rows, H = [], [("h1", 0, "1-day"), ("h7", 1, "7-day"), ("h30", 2, "30-day")]
for k, i, lab in H:
    rows.append(f"{lab} & {cell(*lc([FEAR[i]]), lab, 'fear')} & {cell(*lc([ANGER[i]]), lab, 'anger')} & {cell(*lc([FEAR[i]], [ANGER[i]]), lab, 'delta')} \\\\")
cf, ca, cd = lc(FEAR), lc(ANGER), lc(FEAR, ANGER)
assert abs(cd[0] - cq.loc["SQ4", "delta"]) < 1e-8 and abs(cd[1] - cq.loc["SQ4", "delta_dk_se"]) < 1e-8
z = lambda x: 2 * __import__("scipy.stats", fromlist=["norm"]).norm.sf(abs(x[0] / x[1]))
rows += [r"\addlinespace", f"Cumulative & {cell(*cf, 'cum', 'fear')} & {cell(*ca, 'cum', 'anger')} & {cell(*cd, 'cum', 'delta')} \\\\"]
ci = lambda x: f"[{x[0] - 1.96 * x[1]:+.1f}, {x[0] + 1.96 * x[1]:+.1f}]"
rows += [f"95\\% interval & {ci(cf)} & {ci(ca)} & {ci(cd)} \\\\", f"$p$ (two-sided) & {fp(val('res_sq4','p','fear',z(cf)))} & {fp(val('res_sq4','p','anger',z(ca)))} & {fp(val('res_sq4','p','delta',z(cd)))} \\\\"]
tex = r"""\begin{table}[htbp]
\centering
\caption{Effects of fear and anger on next-day log realized volatility}\label{tab:res_sq4}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
 & Fear & Anger & Fear $-$ anger \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\srcnote{Own calculations; R3 estimated by OLS with coin effects on 6{,}250 coin-days. Coefficients in raw probability units with Driscoll--Kraay standard errors in parentheses. A coefficient of 10 means that moving one percentage point of a day's posts from neutral to the emotion raises next-day log RV by 0.10; the cumulative row adds the three horizons.}
\end{table}
"""
write(tab_dir / "res_sq4.tex", tex)

# --------------------------------------------------------------------------- Table: answers
m = pd.read_csv(RUNS / "_arm_results_medrv_20261108.csv")
assert len(m) == 44
same = {h: int(m[f"{h}_same_verdict_as_primary"].sum()) for h in ("H1", "H2", "H3", "H4")}
tested4 = int(m.H4_p.notna().sum())
val("res_answers", "SQ1", "same", same["H1"]); val("res_answers", "SQ2", "same", same["H2"]); val("res_answers", "SQ3", "same", same["H3"]); val("res_answers", "SQ4", "same", same["H4"]); val("res_answers", "SQ4", "tested", tested4)
q = cq
tex = r"""\begin{table}[!htbp]
\centering
\caption{Answers to the subquestions}\label{tab:res_answers}
\small
\begin{tabularx}{\textwidth}{@{}p{3.2cm}Xrr@{}}
\toprule
Subquestion & Answer & Holm $p$ & Same answer in robustness runs \\
\midrule
""" + "\n".join([
 f"\\cref{{sq:1}}: attention & Improves the forecasts on both losses & {fp(q.loc['SQ1','p_holm'])} & {same['H1']} of 44 \\\\",
 f"\\cref{{sq:2}}: polarity & Small gain in the Clark--West test, none in the model confidence set or the Diebold--Mariano tests; the expected answer of no gain is not confirmed & {fp(q.loc['SQ2','p_holm'])} & {same['H2']} of 44 \\\\",
 f"\\cref{{sq:3}}: emotions & No improvement & {fp(q.loc['SQ3','p_holm'])} & {same['H3']} of 44 \\\\",
 f"\\cref{{sq:4}}: fear and anger & No significant difference; the estimate points to a larger effect of anger & {fp(q.loc['SQ4','p_holm'])} & {same['H4']} of {tested4} \\\\"]) + r"""
\bottomrule
\end{tabularx}
\srcnote{Own calculations. Holm: adjusted over the four confirmatory tests. Robustness runs: the ladder re-estimated with other windows, text constructions, attention measures, controls, platform sets and jump measures (\cref{app:robustness-and-event-analyses}); same answer: the run's own decision rule gives the answer of the main specification.}
\end{table}
"""
write(tab_dir / "res_answers.tex", tex)

# --------------------------------------------------------------------------- Appendix B: validation
L = 7
c = duckdb.connect()
c.execute(f"SET TimeZone='UTC'; SET memory_limit='4GB'; SET threads=3; SET temp_directory='{ddir}'")
assert c.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"
parts = " UNION ALL ".join(f"SELECT day, post_id, '{k}' coin FROM read_parquet('{D}/stocktwits/panel/assignment.parquet') WHERE NOT bot_any AND in_{k.lower()}" for k in COINS)
bs = c.execute(f"""SELECT a.day, a.coin, sum(CASE WHEN s.sentiment='Bullish' THEN 1 ELSE 0 END)::BIGINT bull, count(*)::BIGINT n
  FROM ({parts}) a JOIN (SELECT message_id, sentiment FROM read_parquet('{D}/stocktwits/coarse/month=*/data_0.parquet') WHERE sentiment IN ('Bullish','Bearish')) s
  ON a.post_id = s.message_id GROUP BY 1,2""").df()
bs["day"] = pd.to_datetime(bs["day"]); bs["share"] = bs.bull / bs.n
assert bs.n.sum() > 5_000_000 and set(bs.coin) == set(COINS)
td = pd.read_parquet(D / "panel/text_daily.parquet")
td = td[(td.variant == "main") & (td.platform == "stocktwits") & td.series.isin(COINS)].copy()
td["day"] = pd.to_datetime(td["day"])
def corr(y, x):
    d = pd.concat([pd.Series(y), pd.Series(x)], axis=1).dropna().to_numpy()
    a, b = [(d[:, i] - d[:, i].mean()) / d[:, i].std() for i in (0, 1)]
    return hac_slope(a, b, L), len(a)
VARS = [(e, f"mean_{e}", l) for e, l in EMO] + [("polarity", "mean_polarity", "Polarity")]
share = {}
for e, col, lab in VARS:
    rs = []
    for k in COINS:
        t = td[td.series == k].set_index("day")[col]
        y = bs[bs.coin == k].set_index("day")["share"]
        idx = y.index.intersection(t.index)
        rr, nn = corr(y.reindex(idx).to_numpy(), t.reindex(idx).to_numpy())
        rs.append((rr["slope"], rr["slope_p"]))
    share[e] = rs
    for k, (r_, p_) in zip(COINS, rs):
        val("app_validation", lab, f"bull_share_r_{k}", r_); val("app_validation", lab, f"bull_share_p_{k}", p_)
fg = pd.read_csv("data/control_var/crypto_fear_and_greed_daily.csv", parse_dates=["date"]).set_index("date")["value"].astype(float).sort_index()
pb = p[p.coin == "BTC"].set_index("day")
fgr = {}
for e, col, lab in VARS:
    x = pb["pol" if e == "polarity" else f"emo_{e}"]
    y = fg.reindex(x.index + pd.Timedelta(days=1)).to_numpy()   # the index of day t+1 summarises day t
    rr, nn = corr(y, x.to_numpy())
    fgr[e] = (rr["slope"], rr["slope_p"], nn)
    val("app_validation", lab, "fg_r", rr["slope"]); val("app_validation", lab, "fg_p", rr["slope_p"])
vad = pd.read_csv(D / "panel/reports/vad_test29_correlations_20261007.csv")
vad = vad[(vad.variant == "v1_pooled_primary") & (vad.platform == "POOLED") & (vad.kind == "levels")]
dom = vad[vad.dim == "mean_d_v1"]
rows = []
for e, col, lab in VARS:
    rs = [a for a, _ in share[e]]
    nsig = sum(1 for a, b in share[e] if b < 0.05)
    d = dom[dom.emotion == e]
    dc = f"{d.r.min():+.2f} to {d.r.max():+.2f}" if len(d) else "---"
    if len(d):
        val("app_validation", lab, "dom_r_min", d.r.min()); val("app_validation", lab, "dom_r_max", d.r.max()); val("app_validation", lab, "dom_n_series", len(d)); val("app_validation", lab, "dom_max_p", d.p.max())
    rows.append(f"{lab} & {fgr[e][0]:+.2f} & {fp(fgr[e][1])} & {min(rs):+.2f} to {max(rs):+.2f} & {nsig} of 5 & {dc} \\\\")
tex = r"""\begin{table}[htbp]
\centering
\caption{Convergent and construct validity of the daily measures}\label{tab:app_validation}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{Fear \& Greed Index} & \multicolumn{2}{c}{StockTwits Bullish share} & Dominance \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Measure & $r$ & $p$ & $r$ across coins & $p<0.05$ & $r$ across series \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\srcnote{Own calculations; daily data, 1~August 2019--31~January 2023. Pearson correlations with Newey--West $p$-values (seven lags). Fear \& Greed Index (Alternative.me): index of the next day against the BTC pooled measure, because the value of a day summarises the day before. StockTwits Bullish share: share of Bullish among the labelled posts, per coin, against the same coin's StockTwits measure. Dominance: NRC-VAD v1 daily dominance against the pooled measure, for the five coins and the market series, levels.}
\end{table}
"""
write(tab_dir / "app_validation.tex", tex)

# --------------------------------------------------------------------------- Appendix D: market phases and state
SETS = [("lt20", "bull"), ("lt20", "bear"), ("ma200", "bull"), ("ma200", "bear")]
lin7 = pd.read_csv(A07 / "part2_lincomb_by_regime.csv"); lin7 = lin7[lin7.rung == "R3"]
st = pd.read_csv(A73 / "part1_state_tests.csv"); st = st[st["sample"] == "all"].set_index(["test", "series"])
lin3 = pd.read_csv(A73 / "part2_state_lincomb.csv")
rows = [r"\multicolumn{5}{@{}l}{\textit{Panel A: Clark--West $p$ on the forecast days of each phase}} \\"]
for t, lab in (("H1", "Attention"), ("H2", "Polarity"), ("H3", "Emotions")):
    c = []
    for rl, rg in SETS:
        v = pd.read_csv(A07 / f"part1_nested_tests_{rl}_{rg}.csv").set_index("test").loc[t, "cw_p"]
        c.append(fp(val("app_regime", lab, f"cw_{rl}_{rg}", v)))
    rows.append(f"{lab} & " + " & ".join(c) + r" \\")
rows += [r"\addlinespace", r"\multicolumn{5}{@{}l}{\textit{Panel B: fear minus anger, cumulative, in sample}} \\"]
est, pv = [], []
for rl, rg in SETS:
    r = lin7[(lin7.rule == rl) & (lin7.regime == rg) & lin7.quantity.str.startswith("H4 theta2")].iloc[0]
    est.append(f"{val('app_regime', 'delta', f'{rl}_{rg}', r.estimate):+.1f}"); pv.append(fp(val("app_regime", "delta_p", f"{rl}_{rg}", r.p_two_sided)))
rows += ["Difference & " + " & ".join(est) + r" \\", "$p$ (two-sided) & " + " & ".join(pv) + r" \\"]
d20 = lin7[(lin7.rule == "lt20") & (lin7.regime == "bull minus bear") & lin7.quantity.str.startswith("H4 theta2")].iloc[0]
dma = lin7[(lin7.rule == "ma200") & (lin7.regime == "bull minus bear") & lin7.quantity.str.startswith("H4 theta2")].iloc[0]
val("app_regime", "delta_equal_p", "lt20", d20.p_two_sided); val("app_regime", "delta_equal_p", "ma200", dma.p_two_sided)
rows.append(f"Equal across phases, $p$ & \\multicolumn{{2}}{{c}}{{{fp(d20.p_two_sided)}}} & \\multicolumn{{2}}{{c}}{{{fp(dma.p_two_sided)}}} \\\\")
pa = lin3[(lin3.spec == "parsimonious") & lin3.quantity.str.startswith("slope of Delta")].iloc[0]
fu = lin3[(lin3.spec == "full_interaction") & lin3.quantity.str.startswith("slope of Delta")].iloc[0]
rows2 = []
for t, lab in (("H1", "Attention"), ("H2", "Polarity"), ("H3", "Emotions")):
    a, b = st.loc[(t, "MSE_log")], st.loc[(t, "QLIKE")]
    for nm, x in (("MSE", a), ("QLIKE", b)):
        val("app_regime", lab, f"slope_p_{nm}", x.slope_p); val("app_regime", lab, f"gw_p_{nm}", x.gw_1s_p)
    rows2.append(f"{lab} & {fp(a.slope_p)} & {fp(b.slope_p)} & {fp(a.gw_1s_p)} & {fp(b.gw_1s_p)} \\\\")
val("app_regime", "delta_slope", "parsimonious", pa.estimate); val("app_regime", "delta_slope_p", "parsimonious", pa.p_two_sided)
val("app_regime", "delta_slope", "full", fu.estimate); val("app_regime", "delta_slope_p", "full", fu.p_two_sided)
tex = r"""\begin{table}[htbp]
\centering
\caption{Market phases and the market state}\label{tab:app_regime}
\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
 & \multicolumn{2}{c}{20\%/20\% filter} & \multicolumn{2}{c}{200-day average} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
 & Bull & Bear & Bull & Bear \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{5}{@{}l}{\textit{Panel C: dependence on the continuous market state, $p$}} \\
 & \multicolumn{2}{c}{Slope of the daily gain} & \multicolumn{2}{c}{Giacomini--White} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
 & MSE & QLIKE & MSE & QLIKE \\
""" + "\n".join(rows2) + f"""
\\addlinespace
\\multicolumn{{5}}{{@{{}}l}}{{Slope of fear minus anger per SD of the state: {pa.estimate:+.1f} ($p={pa.p_two_sided:.2f}$); all regressors interacted: {fu.estimate:+.1f} ($p={fu.p_two_sided:.2f}$)}} \\\\
\\bottomrule
\\end{{tabular}}
""" + r"""\srcnote{Own calculations; post-result analysis. Panels A and B: each day is assigned the phase of its target day; 20\%/20\% filter on the BTC close \parencite{lundeDurationDependenceStock2004}: a bull phase ends when the close falls 20\% below its running peak, a bear phase when it rises 20\% above its running trough (496 bull and 389 bear forecast days); 200-day average: the close above or below its average of 200 days \parencite{brockSimpleTechnicalTrading1992} (401 and 484). Panel B: R3 with all regressors interacted with the phase, Driscoll--Kraay; the 200-day rows use the 1{,}081 days on which the average is defined. Panel C: state = log distance of the BTC close from its 200-day average on the origin day; Giacomini--White test with the test function $(1, s_t)$; slope of fear minus anger from R3 with the state interacted with the HAR-CJ, fear and anger terms (first value) or all regressors (second value), 1{,}080 days.}
\end{table}
"""
write(tab_dir / "app_regime.tex", tex)
v = pd.DataFrame(values)
v.to_csv(tab_dir / "src/res_values_20261109.csv.tmp", index=False)
os.replace(tab_dir / "src/res_values_20261109.csv.tmp", tab_dir / "src/res_values_20261109.csv")
print(len(v), "values written")
