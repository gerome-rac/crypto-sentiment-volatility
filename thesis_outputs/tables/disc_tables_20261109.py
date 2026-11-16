#!/usr/bin/env python
"""disc_tables_20261109.py - the numbers Chapter 6 (Discussion) cites and their home in Appendix D, on the MedRV model
(D8 Deviation 7). Post-result and descriptive: everything comes from the stored run outputs, the stored panels and the
variable spec of the primary run; no volatility or forecast model is estimated beyond the in-sample OLS fits already
stored in the primary run (the spanning table adds 20 auxiliary OLS fits of the text measures on the R0, R1 and R2
regressors, which are not stored outputs), nothing is registered, and none of these numbers can change an answer to a
subquestion.

Writes (temp file -> checks -> replace):
  <tables_dir>/app_grid.tex, app_spanning.tex        Appendix D, tables tab:app_grid and tab:app_spanning
  <tables_dir>/src/disc_values_20261109.csv          name, value, source: every number used in the appendix text and
                                                     in the two tables, unformatted
<tmp_dir> receives only the log of the reproduction checks that passed.
Usage: python disc_tables_20261109.py <code_dir> <tables_dir> <tmp_dir>
Run with PYTHONDONTWRITEBYTECODE=1 so that nothing lands in code/model/__pycache__.
"""
import json
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import norm

code_dir, tab_dir, tmp_dir = [Path(a).resolve() for a in sys.argv[1:4]]
REG = code_dir.parent / "har_specification_register.csv"
sys.path.insert(0, str(code_dir))
os.chdir(code_dir)
from model.confirmatory_sq4 import holm                                           # noqa: E402
from model.config import FIRST_ORIGIN, LADDER, LAST_ORIGIN, EngineConfig          # noqa: E402
from model.har import FeatureBuilder                                              # noqa: E402
from model.inference import _h4_weights, _zdesign, dk_cov_within, insample_design  # noqa: E402
from model.panel import build_grid, load_tables, load_variable_spec               # noqa: E402
from model.run import specs_for_rungs                                             # noqa: E402
from model.tests_eval import clark_west, loss_frame                               # noqa: E402

D = Path("data/pipeline")
RUNS = D / "model_runs"
PRIM = RUNS / "A00_final_A50_20261028T134243Z"                  # the primary run
PANEL = D / "panel/arms/final_20261028/A50.parquet"
SPEC = "model/variable_spec_A50.csv"
GRID = RUNS / "_arm_results_medrv_20261108.csv"
Q4 = RUNS / "confirmatory_sq4dk_20261108/confirmatory_sq4dk.csv"
COINS = ["BTC", "ETH", "BNB", "XRP", "DOGE"]

values, CHECKS = [], []


def val(name, x, source):
    values.append(dict(name=name, value=float(x), source=source))
    return x


def check(label, got, exp, tol):
    assert abs(float(got) - float(exp)) <= tol, (label, got, exp, tol)
    CHECKS.append(f"{label}: {float(got):.8g} (expected {float(exp):.8g}, tol {tol})")


def check_true(label, cond):
    assert cond, label
    CHECKS.append(f"{label}: ok")


def fp(p):
    return "$<$0.001" if p < 0.0005 else f"{p:.3f}"


def write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    assert tmp.read_text(encoding="utf-8") == text
    assert not re.search(r"\bA\d\d\b|\bH[1-4]\b", text), f"internal label in {path.name}"  # no run labels may print
    os.replace(tmp, path)


# --------------------------------------------------------------------------- the primary run, reproduced
fc = pd.read_parquet(PRIM / "forecasts.parquet")
fc["target"] = pd.to_datetime(fc["target"])
fc["origin"] = pd.to_datetime(fc["origin"])
check_true("forecast target is origin + 1 day on every row", ((fc.target - fc.origin).dt.days == 1).all())
RUNG = {r: fc[fc.rung == r].sort_values(["target", "coin"]).reset_index(drop=True) for r in LADDER}
check_true("885 target days and 5 coins for R0-R3", all(len(g) == 885 * 5 and g.target.nunique() == 885 for g in RUNG.values()))
nt = pd.read_csv(PRIM / "nested_tests.csv").set_index("test")
EV_CW = {"H1": 0.0002962538879149, "H2": 0.0109353384503759, "H3": 0.8733984457442896}     # E3/E4, stored run
STEPS = {"H1": ("R0", "R1"), "H2": ("R1", "R2"), "H3": ("R2", "R3")}
cw_p = {}
for h, (a, b) in STEPS.items():
    A, B = RUNG[a], RUNG[b]
    check_true(f"{h}: same targets and y in both rungs", (A[["target", "coin"]].equals(B[["target", "coin"]])) and np.array_equal(A.y, B.y))
    r = clark_west(A.y.to_numpy(), A.f_log.to_numpy(), B.f_log.to_numpy(), groups=A.target.to_numpy(), lags=6)
    check(f"stored Clark-West p of the primary run, {h} (recomputed from forecasts vs nested_tests.csv)", r["p_one_sided"], nt.loc[h, "cw_p"], 1e-12)
    check(f"stored Clark-West p of the primary run, {h} (vs the value in the evidence file)", nt.loc[h, "cw_p"], EV_CW[h], 1e-12)
    cw_p[h] = float(nt.loc[h, "cw_p"])
cq = pd.read_csv(Q4)
cq = cq[cq.run == PRIM.name].set_index("subquestion")
check_true("confirmatory_sq4dk.csv has the four rows of the primary run", list(cq.index) == ["SQ1", "SQ2", "SQ3", "SQ4"])
check("SQ4 difference of the primary run", cq.loc["SQ4", "delta"], -29.76041911548083, 1e-9)
check("SQ4 Driscoll-Kraay SE of the primary run", cq.loc["SQ4", "delta_dk_se"], 16.7647, 1e-3)
check("SQ4 p of the primary run", cq.loc["SQ4", "p"], 0.075867, 1e-5)
for s, h in (("SQ1", "H1"), ("SQ2", "H2"), ("SQ3", "H3")):
    check(f"{s} p of the primary run in confirmatory_sq4dk.csv", cq.loc[s, "p"], cw_p[h], 1e-12)
for h, nm in (("H1", "sq1"), ("H2", "sq2"), ("H3", "sq3")):
    val(f"main.{nm}_cw_p", cw_p[h], "primary run, nested_tests.csv (reproduced from forecasts.parquet with six HAC lags)")
val("main.sq4_delta", cq.loc["SQ4", "delta"], "primary run, confirmatory_sq4dk.csv")
val("main.sq4_p", cq.loc["SQ4", "p"], "primary run, confirmatory_sq4dk.csv")
val("main.sq2_holm_p", cq.loc["SQ2", "p_holm"], "primary run, confirmatory_sq4dk.csv")

# --------------------------------------------------------------------------- grid of robustness runs
m = pd.read_csv(GRID)
check_true("grid csv has 44 runs", len(m) == 44)
same = {h: int(m[f"{h}_same_verdict_as_primary"].sum()) for h in ("H1", "H2", "H3", "H4")}
tested4 = int(m.H4_p.notna().sum())
same4 = int(m.loc[m.H4_p.notna(), "H4_same_verdict_as_primary"].sum())
check("same answer as the main specification, SQ1 (of 44)", same["H1"], 44, 0)
check("same answer as the main specification, SQ2 (of 44)", same["H2"], 21, 0)
check("same answer as the main specification, SQ3 (of 44)", same["H3"], 43, 0)
check("runs with a fear-anger test", tested4, 42, 0)
check("same answer as the main specification, SQ4 (of the 42 tested)", same4, 41, 0)
check("all 44 runs: SQ1 raw Clark-West p below 0.05", int((m.H1_cw_p < 0.05).sum()), 44, 0)
for name, h in (("grid.same_sq1", "H1"), ("grid.same_sq2", "H2"), ("grid.same_sq3", "H3")):
    val(name, same[h], "grid csv, count of same-answer flags among the 44 runs")
val("grid.same_sq4", same4, "grid csv, count of same-answer flags among the 42 runs with a fear-anger test")
val("grid.n_runs", len(m), "grid csv")
val("grid.n_tested_sq4", tested4, "grid csv")
val("grid.n_sq1_significant", int((m.H1_cw_p < 0.05).sum()), "grid csv, raw Clark-West p of the attention step below 0.05")
val("grid.n_different_sq2", 44 - same["H2"], "grid csv")
val("grid.n_different_sq3", 44 - same["H3"], "grid csv")
val("grid.n_different_sq4", tested4 - same4, "grid csv")

reg = pd.read_csv(REG)
reg_txt = {a: " ".join(str(v) for v in g[["arm", "data_rule", "purpose"]].values.ravel()).lower() for a, g in reg.groupby("arm_id")}

# run_id in the grid csv, plain description (one line, no run labels), expected csv family, keyword in the register row
EXP = "sample/window"; NON = "none (not assigned in the plan note)"; TXT = "text construction"; ATT = "attention"; CON = "controls"; PLA = "platforms/events"; VOL = "volatility"
RUNS_BY_FAMILY = [
    ("Estimation window, sample and evaluation", [
        ("A06", "250-day rolling window", EXP, "250d"),
        ("A55", "Expanding window from 30 August 2019", EXP, "expanding"),
        ("A61", "QLIKE back-transformation by smearing (forecasts unchanged)", NON, "smearing"),
    ]),
    ("Text constructions", [
        ("A13", "Long texts: chunk mean instead of truncation", TXT, "chunk-mean"),
        ("A20", "Daily means without post-stratification by hour", TXT, "post-stratification"),
        ("A21", "Emotion means include texts empty after cleaning", TXT, "empty-after-clean"),
        ("A59", "Emotions as log-ratios against neutral", TXT, "log-ratio"),
        ("A60", "Certainty index in place of the six emotions", TXT, "certainty"),
        ("A68", "EmTract valence in place of FinBERT polarity", TXT, "valence"),
        ("A48", "Emotions of quote posts scored on the quoted text only", TXT, "quoted text only"),
        ("A49", "Emotions of quote posts scored on own and quoted text together", TXT, "own and quoted"),
        ("A46", "Twitter text measures from authored posts only (no retweet-only posts)", TXT, "authored posts only"),
        ("A26", "Reddit text measures from submissions only, comments excluded", TXT, "comments excluded"),
        ("A45", "Reddit text measures without posts edited after posting", TXT, "edited after posting"),
        ("A23", "Reddit coin measures from text-filtered posts, not by subreddit", TXT, "text-gated"),
        ("A41", "Reddit market measure includes r/binance support posts", TXT, "r/binance"),
        ("A38", "StockTwits coin and market measures without equity-ticker posts", TXT, "equity"),
        ("A42", "StockTwits replies without a coin inherit the parent's coin", NON, "inherit"),
        ("A29", "BNB assigned by cashtag, hashtag or full name only", TXT, "tier-1"),
        ("A39", "BNB measure without posts using BNB Chain terms", TXT, "bsc_context"),
        ("A28", "Single-coin posts only", TXT, "single-ticker"),
        ("A40", "Coin words inside links count again (StockTwits and Twitter)", NON, "inside links"),
        ("A47", "Market measure (StockTwits and Twitter) adds posts with several ambiguous terms", NON, "ambiguous"),
        ("A44", "Exact duplicate texts counted once per UTC day in the text measures", TXT, "exact-duplicate"),
        ("A43", "Posts flagged by the bot filter kept", TXT, "bot"),
    ]),
    ("Attention measures", [
        ("A56", "Attention as the log within-day share of posts", ATT, "within-day share"),
        ("A57", "Market attention adds total Reddit and StockTwits volume", ATT, "total reddit"),
        ("A58", "Google Trends in daily log changes only", ATT, "log change"),
        ("A66", "News and Tether news volume per entity row, not per headline", ATT, "entity row"),
        ("A24", "Reddit attention as total activity, removed posts included", NON, "total activity"),
    ]),
    ("Controls", [
        ("A71", "Controls enter only at R1 (R0 is the pure HAR-CJ)", CON, "controls from r1"),
        ("A63", "High-yield spread in place of the BAA spread", CON, "high-yield"),
        ("A65", r"Fear \& Greed Index added to the controls", CON, "fear & greed"),
        ("A67", "FinBERT polarity of news headlines added in R2", CON, "headline polarity"),
        ("A37", "Share of texts empty after cleaning added as a control", NON, "empty-after-clean share"),
    ]),
    ("Platform sets", [
        ("A04_Reddit", "Text measures from Reddit only", NON, "single platform"),
        ("A04_StockTwits", "Text measures from StockTwits only", NON, "single platform"),
        ("A04_Twitter", "Text measures from Twitter only", NON, "single platform"),
        ("A03_noReddit", "Text measures without Reddit", PLA, "leave-one-platform-out"),
        ("A03_noStockTwits", "Text measures without StockTwits", PLA, "leave-one-platform-out"),
        ("A03_noTwitter", "Text measures without Twitter", PLA, "leave-one-platform-out"),
    ]),
    ("Jump measures", [
        ("A00_BV", "Bipower variation in place of MedRV", VOL, None),
        ("A54", "MedRV replaces the continuous and jump terms in R0", VOL, "medrv replaces c and j"),
    ]),
    ("Coin by coin", [
        ("A01", "Separate ladder for each coin (out of sample only)", EXP, "per-coin"),
    ]),
]
order = [r for _, rows in RUNS_BY_FAMILY for r in rows]
check_true("every run of the grid csv appears exactly once in the table definition",
           sorted(r[0] for r in order) == sorted(m.run_id) and len(order) == 44)
for run, label, fam, kw in order:
    row = m[m.run_id == run].iloc[0]
    assert row.family == fam, (run, row.family, fam)
    if kw is not None:
        assert kw in reg_txt[row.arm_id], (run, kw)
check_true("the csv family of every run matches the table definition, and each description carries a keyword of its register row", True)

# stored outputs of each run agree with the grid csv
prim_fc = fc.set_index(["rung", "coin", "target"]).sort_index()
d_max = {}
same_r0r1 = {}
for run, label, fam, kw in order:
    row = m[m.run_id == run].iloc[0]
    ntr = pd.read_csv(RUNS / row.run_dir / "nested_tests.csv").set_index("test")
    for h in ("H1", "H2", "H3"):
        assert abs(ntr.loc[h, "cw_p"] - row[f"{h}_cw_p"]) < 1e-12, (run, h)
    if pd.notna(row.H4_p):
        hb = pd.read_csv(RUNS / row.run_dir / "h4_bootstrap.csv")
        t2 = hb[hb.condition.str.startswith("theta2")].iloc[0]
        assert abs(t2.estimate - row.H4_delta) < 1e-9 and abs(t2.dk_se - row.H4_delta_dk_se) < 1e-9, run
        assert abs(2 * norm.sf(abs(t2.estimate / t2.dk_se)) - row.H4_p) < 1e-9, run
    f = pd.read_parquet(RUNS / row.run_dir / "forecasts.parquet", columns=["rung", "coin", "target", "f_log"])
    f["target"] = pd.to_datetime(f["target"])
    f = f.set_index(["rung", "coin", "target"]).sort_index()
    check_true(f"run '{label}': forecasts the same 885 target days", f.xs("R0", level="rung").index.get_level_values("target").nunique() == 885)
    dmx = 0.0
    for rg in ("R0", "R1"):
        a = prim_fc.xs(rg, level="rung")["f_log"]
        b = f.xs(rg, level="rung")["f_log"]
        assert a.index.equals(b.index), (run, rg)
        dmx = max(dmx, float(np.abs(a.to_numpy() - b.to_numpy()).max()))
    d_max[run] = dmx
    same_r0r1[run] = dmx < 1e-9
check_true("the grid csv agrees with each run's nested_tests.csv and h4_bootstrap.csv (44 runs)", True)
n_unch = int(sum(same_r0r1.values()))
check("runs that leave R0 and R1 unchanged (forecasts identical to the main run)", n_unch, 20, 0)
print("largest absolute forecast difference R0/R1 vs main, unchanged runs (max) and others (min):",
      max(d_max[r] for r in same_r0r1 if same_r0r1[r]), min(d_max[r] for r in same_r0r1 if not same_r0r1[r]))
assert max(d_max[r] for r in same_r0r1 if same_r0r1[r]) < 1e-9 < 1e-6 < min(d_max[r] for r in same_r0r1 if not same_r0r1[r])
for run, label, fam, kw in order:
    row = m[m.run_id == run].iloc[0]
    if same_r0r1[run]:
        assert row.H1_cw_p == nt.loc["H1", "cw_p"], (run, row.H1_cw_p)
    else:
        assert abs(row.H1_cw_p - nt.loc["H1", "cw_p"]) > 1e-12, (run, row.H1_cw_p)
check_true("the 20 dagger runs are exactly the runs whose stored SQ1 p equals the primary run's", True)
val("grid.n_unchanged_r0r1", n_unch, "stored forecasts of each run compared with the main run: R0 and R1 identical (max abs. difference below 1e-9)")
val("grid.n_can_differ", 44 - n_unch, "44 minus the runs that leave R0 and R1 unchanged")

# what the descriptions above claim about the run they name (added 9 Nov 2026 after the independent check): each run's
# panel against the primary panel; differences at or below 1e-12 are floating-point noise (the two quote-post runs: at most 1.1e-16 on polarity)
ARM_PANELS = D / "panel/arms/final_medrv_20261108"
base_panel = pd.read_parquet(PANEL).set_index(["coin", "day"]).sort_index()
_arm_cache = {}


def panel_diff(run, col):
    """Coin-days per coin on which the run's panel column differs from the primary panel by more than 1e-12."""
    if run not in _arm_cache:
        _arm_cache[run] = pd.read_parquet(ARM_PANELS / f"{run}.parquet").set_index(["coin", "day"]).reindex(base_panel.index)
    a, b = pd.to_numeric(base_panel[col], errors="coerce"), pd.to_numeric(_arm_cache[run][col], errors="coerce")
    assert (a.isna() == b.isna()).all(), (run, col)
    return (a - b).abs().gt(1e-12).groupby(level="coin").sum().reindex(COINS)


n = panel_diff("A38", "att_stocktwits")
check_true(f"equity-ticker run: StockTwits post counts change for all five coins ({n.to_dict()})", (n > 0).all())
n = panel_diff("A38", "mkt_stocktwits")
check_true(f"equity-ticker run: the StockTwits market measure changes on every coin-day ({int(n.sum())})", int(n.sum()) == len(base_panel))
n = panel_diff("A38", "pol")
check_true(f"equity-ticker run: the pooled text measures change for all five coins (polarity, {n.to_dict()})", (n > 0).all())
for c_ in ("att_reddit", "att_twitter", "mkt_reddit", "mkt_twitter"):
    check_true(f"equity-ticker run: {c_} unchanged", int(panel_diff("A38", c_).sum()) == 0)
att_mkt = [c_ for c_ in base_panel.columns if c_.startswith(("att_", "mkt_"))]
check_true(f"{len(att_mkt)} attention and market-attention columns in the primary panel", len(att_mkt) >= 10)
for run in ("A03_noReddit", "A03_noStockTwits", "A03_noTwitter"):
    check_true(f"{run}: R0 and R1 forecasts identical to the main run", same_r0r1[run])
    for c_ in att_mkt:
        assert int(panel_diff(run, c_).sum()) == 0, (run, c_)
    check_true(f"{run}: no attention or market-attention column changes (only the text index does)", True)
for c_, changes in (("att_stocktwits", True), ("att_twitter", True), ("att_reddit", False), ("mkt_reddit", False), ("mkt_stocktwits", False), ("mkt_twitter", False)):
    tot = int(panel_diff("A40", c_).sum())
    check_true(f"run with coin words inside links: {c_} {'changes' if changes else 'unchanged'} ({tot} coin-days)", (tot > 0) if changes else (tot == 0))
for run in ("A48", "A49"):
    for c_ in ("pol", "pol_raw", "pol_sd", "text_n_posts"):
        check_true(f"{run}: {c_} unchanged", int(panel_diff(run, c_).sum()) == 0)
    check_true(f"{run}: the emotion measures change (fear, {int(panel_diff(run, 'emo_fear').sum())} coin-days)", int(panel_diff(run, "emo_fear").sum()) > 0)
    f2 = pd.read_parquet(RUNS / m[m.run_id == run].iloc[0].run_dir / "forecasts.parquet", columns=["rung", "coin", "target", "f_log"])
    f2["target"] = pd.to_datetime(f2["target"])
    f2 = f2[f2.rung == "R2"].set_index(["coin", "target"]).sort_index().f_log
    a2 = prim_fc.xs("R2", level="rung")["f_log"]
    assert a2.index.equals(f2.index), run
    check_true(f"{run}: R2 forecasts equal the main run's (max abs. difference {float(np.abs(a2.to_numpy() - f2.to_numpy()).max()):.2e})",
               float(np.abs(a2.to_numpy() - f2.to_numpy()).max()) < 1e-9)
d66 = m[m.run_id == "A66"].iloc[0].run_dir
co0, co1 = list(pd.read_parquet(RUNS / d66 / "coefs_R0.parquet").columns), list(pd.read_parquet(RUNS / d66 / "coefs_R1.parquet").columns)
check_true("news run: R0 uses the Tether news control per entity row (level and daily change), not per headline",
           {"ctrl_tether_news_rows", "ctrl_tether_news_rows_chg"} <= set(co0) and not {"ctrl_tether_news", "ctrl_tether_news_chg"} & set(co0))
check_true("news run: R1 uses news volume per entity row, not per headline", any(c_.startswith("att_news_rows_n") for c_ in co1) and not any(c_.startswith("att_news_h") for c_ in co1))
check_true("news run: changes R0 and R1 (no dagger)", not same_r0r1["A66"])

# added 9 Nov 2026 after the second independent check: the market run names its two platforms, the four text-only runs name the
# text measures, and the BNB run names every form of the flag it uses (hashtags as well as the cashtag and the full name)
text_cols = [c_ for c_ in base_panel.columns if c_.startswith(("emo_", "pol"))]
check("text-measure columns in the primary panel", len(text_cols), 21, 0)
for c_, changes in (("mkt_stocktwits", True), ("mkt_twitter", True), ("mkt_reddit", False)):
    tot = int(panel_diff("A47", c_).sum())
    check_true(f"run with several ambiguous terms: {c_} {'changes' if changes else 'unchanged'} ({tot} coin-days)", (tot > 0) if changes else (tot == 0))
for c_ in att_mkt + text_cols:
    if c_ not in ("mkt_stocktwits", "mkt_twitter"):
        assert int(panel_diff("A47", c_).sum()) == 0, c_
check_true("run with several ambiguous terms: no other attention, market-attention or text column changes", True)
for run in ("A46", "A26", "A45", "A44"):
    check_true(f"{run}: R0 and R1 forecasts identical to the main run", same_r0r1[run])
    for c_ in att_mkt:
        assert int(panel_diff(run, c_).sum()) == 0, (run, c_)
    for c_ in ("pol", "emo_fear", "emo_anger"):
        assert int(panel_diff(run, c_).sum()) > 0, (run, c_)
    check_true(f"{run}: only the text measures change (no attention or market-attention column does; polarity, fear and anger do)", True)
rules_src = (code_dir / "pipeline/rules.py").read_text(encoding="utf-8")
m_t1 = re.findall(r'\("BNB", "t1"\): r"([^"]+)"', rules_src)
assert len(m_t1) == 1
t1_bnb = re.compile(m_t1[0])
for s_, hit in (("$bnb", True), ("$bnb.x", True), ("#bnb", True), ("#binancecoin", True), ("binance coin", True), ("binancecoin", True), ("bnb", False), ("binance", False)):
    assert bool(t1_bnb.search(s_)) == hit, s_
check_true("BNB run: tier 1 is the cashtag, the hashtags #bnb and #binancecoin and the full name, not bare 'bnb' or 'binance'", True)
for f_, expr in (("04_series_reddit.py", "(x.hit_bnb_t1) AS bnb_strict"), ("04_series_twitter.py", "a.hit_bnb_t1 AS bnb_strict"), ("04_series_stocktwits.py", "(a.cash_bnb OR a.hit_bnb_t1) AS bnb_strict")):
    assert (code_dir / "pipeline" / f_).read_text(encoding="utf-8").count(expr) == 1, f_
check_true("BNB run: its flag is the tier-1 hit on every platform (StockTwits: or the BNB cashtag)", True)
for c_ in ("att_reddit", "att_stocktwits", "att_twitter"):
    n = panel_diff("A29", c_)
    check_true(f"BNB run: {c_} changes for BNB only ({n.to_dict()})", n["BNB"] > 0 and int(n.drop("BNB").sum()) == 0)

# marks: dagger on the SQ1 number of each unchanged run, double dagger on each number whose run's own rule differs
DAG, DD = r"\rlap{$^{\dagger}$}", r"\rlap{$^{\ddagger}$}"
n_dd = {h: 0 for h in ("H1", "H2", "H3", "H4")}
slug = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower().replace(r"\&", "and")).strip("_")
tbl = ["Main specification & " + " & ".join([fp(cw_p["H1"]), fp(cw_p["H2"]), fp(cw_p["H3"]), f"{cq.loc['SQ4', 'delta']:+.1f}", fp(cq.loc["SQ4", "p"])]) + r" \\"]
for fam_name, fam_rows in RUNS_BY_FAMILY:
    tbl += [r"\addlinespace", rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{fam_name}}}}} \\"]
    for run, label, fam, kw in fam_rows:
        row = m[m.run_id == run].iloc[0]
        nm = slug(label)
        c1 = fp(val(f"grid.{nm}.sq1_p", row.H1_cw_p, f"grid csv, run '{label}', Clark-West p of the attention step")) + (DAG if same_r0r1[run] else "")
        if not row.H1_same_verdict_as_primary:
            n_dd["H1"] += 1
            c1 += DD
        c2 = fp(val(f"grid.{nm}.sq2_p", row.H2_cw_p, f"grid csv, run '{label}', Clark-West p of the polarity step"))
        if not row.H2_same_verdict_as_primary:
            n_dd["H2"] += 1
            c2 += DD
        c3 = fp(val(f"grid.{nm}.sq3_p", row.H3_cw_p, f"grid csv, run '{label}', Clark-West p of the emotion step"))
        if not row.H3_same_verdict_as_primary:
            n_dd["H3"] += 1
            c3 += DD
        if pd.notna(row.H4_p):
            d = val(f"grid.{nm}.sq4_delta", row.H4_delta, f"grid csv, run '{label}', difference of the cumulative effects of fear and anger")
            c4 = f"{d:+.2f}" if run == "A59" else f"{d:+.1f}"
            c5 = fp(val(f"grid.{nm}.sq4_p", row.H4_p, f"grid csv, run '{label}', two-sided Driscoll-Kraay p of the difference"))
            if not row.H4_same_verdict_as_primary:
                n_dd["H4"] += 1
                c5 += DD
        else:
            check_true(f"run '{label}' has no fear-anger test", pd.isna(row.H4_delta))
            c4, c5 = "---", "---"
        tbl.append(f"{label} & {c1} & {c2} & {c3} & {c4} & {c5} \\\\")
check("double daggers, SQ1", n_dd["H1"], 0, 0)
check("double daggers, SQ2", n_dd["H2"], 23, 0)
check("double daggers, SQ3", n_dd["H3"], 1, 0)
check("double daggers, SQ4", n_dd["H4"], 1, 0)
differ3 = [r for r, *_ in order if not m[m.run_id == r].iloc[0].H3_same_verdict_as_primary]
differ4 = [r for r, *_ in order if not m[m.run_id == r].iloc[0].H4_same_verdict_as_primary and pd.notna(m[m.run_id == r].iloc[0].H4_p)]
check_true(f"the run with a different SQ3 answer is the StockTwits-only run: {differ3}", differ3 == ["A04_StockTwits"])
check_true(f"the run with a different SQ4 answer is the Reddit-only run: {differ4}", differ4 == ["A04_Reddit"])

tex = r"""\begin{table}[htbp]
\centering
\caption{All robustness runs}\label{tab:app_grid}
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{@{}lrrrrr@{\hspace{5pt}}}
\toprule
 & \multicolumn{3}{c}{Clark--West $p$} & \multicolumn{2}{c}{Fear $-$ anger} \\
\cmidrule(lr){2-4}\cmidrule(l){5-6}
What changes & SQ1 & SQ2 & SQ3 & Difference & $p$ \\
\midrule
""" + "\n".join(tbl) + r"""
\bottomrule
\end{tabular}
\srcnote{Own calculations from the stored runs; every run forecasts the same 885 target days. All $p$-values are unadjusted. Clark--West: one-sided $p$-value of the step to the rung (SQ1: R0 $\to$ R1, SQ2: R1 $\to$ R2, SQ3: R2 $\to$ R3). Fear $-$ anger: difference between the cumulative effects of fear and anger on next-day log RV, in raw probability units (in the log-ratio run in units of the log-ratio), with the two-sided Driscoll--Kraay $p$-value; a dash marks a run without this test. $^{\dagger}$~The run leaves R0 and R1 unchanged, so its SQ1 step repeats the main result by construction. $^{\ddagger}$~The run's own decision rule gives a different answer from the main specification on this number.}
\end{table}
"""
assert tex.count(r"\\") >= 45 + 7
write(tab_dir / "app_grid.tex", tex)
CHECKS.append("app_grid.tex written")

# --------------------------------------------------------------------------- spanning of the text measures (in sample)
cfg = EngineConfig.for_spec("A00", label="final_A50")
stored = json.load(open(PRIM / "config.json"))
mine = json.loads(cfg.to_json())
diff = {k: (mine.get(k), stored.get(k)) for k in sorted(set(mine) | set(stored)) if mine.get(k) != stored.get(k)}
check_true(f"engine config equals the primary run's config.json except n_jobs ({diff})", set(diff) <= {"n_jobs"})
specs = specs_for_rungs(load_variable_spec(SPEC), list(LADDER) + ["HARRV", "HARCJ"], cfg)
grid = build_grid(load_tables(str(PANEL)), specs)
builder = FeatureBuilder(grid, specs, cfg)
o0, o1 = builder.day_index(FIRST_ORIGIN), builder.day_index(LAST_ORIGIN)
tr, te = np.arange(o0, o1 + 1), np.array([o1])
designs = {r: builder.design(r, tr, te) for r in LADDER}
d0 = designs["R0"]
for r in LADDER:
    assert np.array_equal(designs[r].coin_tr, d0.coin_tr) and np.array_equal(designs[r].day_tr, d0.day_tr)
check("in-sample coin-days", len(d0.y_tr), 6250, 0)
check("in-sample days", len(np.unique(d0.day_tr)), 1250, 0)
coin, y = d0.coin_tr, d0.y_tr


def dm(x):
    x = np.asarray(x, float)
    C = int(coin.max()) + 1
    if x.ndim == 1:
        return x - (np.bincount(coin, weights=x, minlength=C) / np.bincount(coin, minlength=C))[coin]
    return np.column_stack([dm(x[:, j]) for j in range(x.shape[1])])


def within_fit(yy, X):
    yd, Xd = dm(yy), dm(X)
    b, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    e = yd - Xd @ b
    return 1.0 - float(e @ e) / float(yd @ yd), e, b


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


coef = pd.read_csv(PRIM / "insample_coef.csv")
fit_s = pd.read_csv(PRIM / "insample_fit.csv").set_index("rung")
EV_K = {"R0": 17, "R1": 62, "R2": 71, "R3": 125}
for rg in LADDER:
    dd = designs[rg]
    assert dd.X_tr.shape[1] == EV_K[rg], rg
    rr2, e, b = within_fit(y, dd.X_tr)
    s = coef[coef.rung == rg].set_index("variable")["beta_raw"]
    assert list(dd.names) == list(s.index), rg
    check(f"in-sample R^2 within of {rg}", rr2, fit_s.loc[rg, "R2_within"], 1e-8)
    check(f"in-sample MSE of {rg}", float(np.mean(e ** 2)), fit_s.loc[rg, "MSE_log"], 1e-8)
    check(f"in-sample coefficients of {rg}, max abs difference", float(np.abs(pd.Series(b, index=dd.names) - s).max()), 0.0, 1e-6)
_, resid0, _ = within_fit(y, d0.X_tr)
d3 = designs["R3"]
pos3 = {n_: i for i, n_ in enumerate(d3.names)}
SER = [("Fear", "emo_fear_h1"), ("Anger", "emo_anger_h1"), ("Sadness", "emo_sad_h1"), ("Disgust", "emo_disgust_h1"),
       ("Surprise", "emo_surprise_h1"), ("Happiness", "emo_happy_h1"), ("Polarity", "pol_h1")]
EV_R2 = {"Fear": (0.2840, 0.3777, 0.5977), "Anger": (0.3395, 0.4326, 0.5593), "Sadness": (0.2780, 0.4152, 0.5335), "Disgust": (0.3115, 0.4256, 0.5822),
         "Surprise": (0.0882, 0.2690, 0.3278), "Happiness": (0.2734, 0.5205, 0.6343), "Polarity": (0.1393, 0.2356, None)}
EV_CORR = {"Fear": (0.3128, 0.0415), "Anger": (0.2743, 0.0239)}
SRC_SP = "in-sample design of the primary run (panel and variable spec of the primary run); coin-demeaned OLS and correlations"
sp_rows = []
for lab, col in SER:
    x = d3.X_tr[:, pos3[col]]
    r0 = within_fit(x, designs["R0"].X_tr)[0]
    r1 = within_fit(x, designs["R1"].X_tr)[0]
    r2_val = within_fit(x, designs["R2"].X_tr)[0] if lab != "Polarity" else np.nan
    c_rv = r(dm(x), dm(y))
    c_res = r(dm(x), dm(resid0))
    ev = EV_R2[lab]
    check(f"within R^2 of {lab} on the R0 regressors", r0, ev[0], 5e-5)
    check(f"within R^2 of {lab} on the R1 regressors", r1, ev[1], 5e-5)
    if ev[2] is not None:
        check(f"within R^2 of {lab} on the R2 regressors", r2_val, ev[2], 5e-5)
    if lab in EV_CORR:
        check(f"within correlation of {lab} with next-day log RV", c_rv, EV_CORR[lab][0], 5e-5)
        check(f"within correlation of {lab} with the R0 residual", c_res, EV_CORR[lab][1], 5e-5)
    sn = lab.lower()
    val(f"spanning.{sn}.r2_on_r0", r0, SRC_SP); val(f"spanning.{sn}.r2_on_r1", r1, SRC_SP)
    if lab != "Polarity":
        val(f"spanning.{sn}.r2_on_r2", r2_val, SRC_SP)
    val(f"spanning.{sn}.corr_next_day_log_rv", c_rv, SRC_SP); val(f"spanning.{sn}.corr_r0_residual", c_res, SRC_SP)
    r2c = f"{r2_val:.3f}" if lab != "Polarity" else "---"
    sp_rows.append(f"{lab} & {r0:.3f} & {r1:.3f} & {r2c} & {c_rv:.3f} & {c_res:.3f} \\\\")
val("spanning.n_coin_days", len(y), SRC_SP)
n_aux = 2 * len(SER) + sum(lab != "Polarity" for lab, _ in SER)
check("auxiliary OLS fits of the text measures on the R0, R1 and (emotions) R2 regressors", n_aux, 20, 0)
val("spanning.n_auxiliary_ols_fits", n_aux, "fits in this script that are not stored outputs: seven text measures on the R0 and R1 regressors, six emotions on the R2 regressors")
tex = r"""\begin{table}[htbp]
\centering
\caption{How far the text measures are spanned by the other regressors}\label{tab:app_spanning}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
 & \multicolumn{3}{c}{$R^2$ on the regressors of} & \multicolumn{2}{c}{Correlation with} \\
\cmidrule(lr){2-4}\cmidrule(l){5-6}
One-day value of & R0 & R1 & R2 & Next-day log RV & R0 residual \\
\midrule
""" + "\n".join(sp_rows[:6]) + r"""
\addlinespace
""" + sp_rows[6] + r"""
\bottomrule
\end{tabular}
\srcnote{Own calculations; in sample, descriptive; 6{,}250 coin-days, coin effects removed. $R^2$ of an OLS regression of the measure's one-day value on the 17, 62 or 71 regressors of R0, R1 or R2, after removing coin means; polarity is itself part of R2, hence the dash. Correlations are computed on the coin-demeaned series: with next-day log RV and with the residual of the in-sample R0 regression of next-day log RV.}
\end{table}
"""
write(tab_dir / "app_spanning.tex", tex)
CHECKS.append("app_spanning.tex written")

# --------------------------------------------------------------------------- 4(a) the loss of R3 relative to R2, by window type
blk = pd.read_csv(PRIM / "selection_freq_columns_R3.csv").set_index("column").block.to_dict()
coefs = {}
for rg in ("R1", "R2", "R3"):
    c = pd.read_parquet(PRIM / f"coefs_{rg}.parquet")
    c["origin"] = pd.to_datetime(c["origin"])
    coefs[rg] = c.set_index("origin").sort_index()
    assert len(coefs[rg]) == 885, rg
check_true("coefficient files of R1-R3 have 885 windows", True)


def cols(rg, block):
    return [n_ for n_ in coefs[rg].columns if blk.get(n_) == block]


emo_cols, pol_cols2, att_cols1 = cols("R3", "emotion"), cols("R2", "polarity"), cols("R1", "attention")
check("emotion columns in R3", len(emo_cols), 54, 0)
check("polarity columns in R2", len(pol_cols2), 9, 0)
check("attention columns in R1", len(att_cols1), 45, 0)
any_emo = (coefs["R3"][emo_cols].to_numpy() != 0).any(axis=1)
n_none = int((~any_emo).sum())
check("R3 windows with no emotion coefficient", n_none, 824, 0)
check("R3 windows", len(coefs["R3"]), 885, 0)
check("R3 windows with any emotion coefficient", int(any_emo.sum()), 61, 0)
val("r3loss.windows_no_emotion", n_none, "primary run, coefs_R3.parquet: windows whose 54 emotion coefficients are all zero")
val("r3loss.windows_total", len(coefs["R3"]), "primary run, coefs_R3.parquet")
val("r3loss.windows_any_emotion", int(any_emo.sum()), "primary run, coefs_R3.parquet")
val("r3loss.n_emotion_coefficients", len(emo_cols), "primary run, coefs_R3.parquet: emotion columns of R3")
loss = {}
for rg in LADDER:
    L = loss_frame(RUNG[rg])
    loss[rg] = L.groupby("target")[["se", "qlike"]].mean().sort_index()
    assert len(loss[rg]) == 885
d23 = loss["R2"] - loss["R3"]                         # positive: R3 better
d23.index = pd.DatetimeIndex(d23.index)
is_none = pd.Series(~any_emo, index=coefs["R3"].index + pd.Timedelta(days=1)).reindex(d23.index)
assert is_none.notna().all()
is_none = is_none.astype(bool)
SRC_L = "primary run, forecasts.parquet and coefs_R3.parquet; sum over the 885 days of the daily cross-coin mean loss of R2 minus that of R3"
for nm, col in (("se", "se"), ("qlike", "qlike")):
    tot = val(f"r3loss.{nm}_sum_all_days", d23[col].sum(), SRC_L)
    no = val(f"r3loss.{nm}_sum_no_emotion_windows", d23.loc[is_none, col].sum(), SRC_L)
    val(f"r3loss.{nm}_sum_emotion_windows", d23.loc[~is_none, col].sum(), SRC_L)
    val(f"r3loss.{nm}_share_no_emotion_windows", no / tot, SRC_L)
check("R2-to-R3 squared-error sum, all days", d23["se"].sum(), -3.565, 0.008)
check("R2-to-R3 squared-error sum, windows with no emotion coefficient", d23.loc[is_none, "se"].sum(), -3.011, 0.002)
check("R2-to-R3 QLIKE sum, all days", d23["qlike"].sum(), -7.130, 0.002)
check("R2-to-R3 QLIKE sum, windows with no emotion coefficient", d23.loc[is_none, "qlike"].sum(), -6.869, 0.002)
check("share of the squared-error loss in those windows", d23.loc[is_none, "se"].sum() / d23["se"].sum(), 3.011 / 3.565, 0.002)
check("share of the QLIKE loss in those windows", d23.loc[is_none, "qlike"].sum() / d23["qlike"].sum(), 6.869 / 7.130, 0.002)

# 4(b) the forecast of 7 September 2021, origin 6 September 2021
O = pd.Timestamp("2021-09-06"); T = O + pd.Timedelta(days=1)
assert (RUNG["R0"].target == T).sum() == 5


def n_sel(rg, block):
    return int((coefs[rg].loc[O, cols(rg, block)].to_numpy() != 0).sum())


SRC_B = "primary run, coefs_R1/R2/R3.parquet at origin 6 September 2021"
n1a, n2a, n2p = n_sel("R1", "attention"), n_sel("R2", "attention"), n_sel("R2", "polarity")
n3a, n3p, n3e = n_sel("R3", "attention"), n_sel("R3", "polarity"), n_sel("R3", "emotion")
check("7 Sep 2021: attention columns selected by R1", n1a, 15, 0)
check("7 Sep 2021: attention columns selected by R2", n2a, 21, 0)
check("7 Sep 2021: polarity columns selected by R2", n2p, 4, 0)
check("7 Sep 2021: attention columns selected by R3", n3a, 0, 0)
check("7 Sep 2021: polarity columns selected by R3", n3p, 0, 0)
check("7 Sep 2021: emotion columns selected by R3", n3e, 0, 0)
val("sep7.r1_attention_columns", n1a, SRC_B); val("sep7.r2_attention_columns", n2a, SRC_B); val("sep7.r2_polarity_columns", n2p, SRC_B)
val("sep7.r3_attention_columns", n3a, SRC_B); val("sep7.r3_polarity_columns", n3p, SRC_B); val("sep7.r3_emotion_columns", n3e, SRC_B)
fs7 = {rg: RUNG[rg][RUNG[rg].target == T].set_index("coin").f_log.reindex(COINS) for rg in ("R0", "R3")}
gap = float(np.abs(fs7["R3"] - fs7["R0"]).max())
check("7 Sep 2021: largest absolute gap between the R3 and R0 forecasts (log points)", gap, 0.0059, 0.0005)
val("sep7.max_abs_gap_r3_r0_log_points", gap, "primary run, forecasts.parquet, target 7 September 2021, largest over the five coins of |f(R3) - f(R0)|")

# 4(c) share of windows with at least one attention column, by forecast-origin quarter
sel_att = pd.Series((coefs["R1"][att_cols1].to_numpy() != 0).any(axis=1), index=coefs["R1"].index)
qshare = sel_att.groupby(pd.PeriodIndex(sel_att.index, freq="Q")).mean()
EV_Q = {"2020Q3": 90.9, "2020Q4": 100, "2021Q1": 100, "2021Q2": 100, "2021Q3": 100, "2021Q4": 50.0, "2022Q1": 14.4, "2022Q2": 8.8, "2022Q3": 4.3, "2022Q4": 4.3, "2023Q1": 0.0}
SRC_Q = "primary run, coefs_R1.parquet: share of windows with a forecast origin in the quarter that select at least one of the 45 attention columns"
check_true("forecast-origin quarters of R1: the expected eleven", sorted(str(p) for p in qshare.index) == sorted(EV_Q))
for p, v in qshare.items():
    check(f"attention selected, windows of {p}, %", 100 * v, EV_Q[str(p)], 0.05)
    val(f"attention_share.{p}", v, SRC_Q)
check("attention selected in every window from 2020Q4 to 2021Q3", int(qshare.loc["2020Q4":"2021Q3"].eq(1.0).sum()), 4, 0)

# 4(d) selection shares of the polarity and the emotion terms (columns with a non-zero share)
sf2 = pd.read_csv(PRIM / "selection_freq_columns_R2.csv")
sf3 = pd.read_csv(PRIM / "selection_freq_columns_R3.csv")
LABEL = {"sq": "centered square", "sd": "within-day SD"}
EV_SEL2 = {"pol_sq_h1": 14.1, "pol_sq_h7": 8.7, "pol_h1": 4.7, "pol_sd_h1": 4.0, "pol_h7": 3.8, "pol_sq_h30": 0.9, "pol_sd_h7": 0.9, "pol_h30": 0.1, "pol_sd_h30": 0.1}
# E4 prints 2.2 for emo_happy_sq_h30; the stored frequency is 19/885 = 2.147%, so the expected value here is 2.15 (see the report)
EV_SEL3 = {"emo_fear_sq_h1": 4.4, "emo_sad_sq_h1": 3.1, "emo_fear_sq_h30": 2.7, "emo_happy_sq_h30": 2.15, "emo_fear_h1": 1.8, "emo_happy_sq_h7": 1.2,
           "emo_happy_sq_h1": 0.2, "emo_anger_sq_h30": 0.2, "emo_disgust_sq_h1": 0.2}
for sf, block, rg, ev, tag in ((sf2, "polarity", "R2", EV_SEL2, "polarity"), (sf3, "emotion", "R3", EV_SEL3, "emotion")):
    sub = sf[(sf.block == block) & (sf.freq > 0)].sort_values("freq", ascending=False).set_index("column").freq
    check_true(f"{tag} terms with a non-zero selection share are the expected set ({len(sub)})", sorted(sub.index) == sorted(ev))
    val(f"selection.n_{tag}_terms_selected", len(sub), f"primary run, selection_freq_columns_{rg}.csv: columns of the block with a non-zero selection share")
    val(f"selection.n_{tag}_terms", int((sf.block == block).sum()), f"primary run, selection_freq_columns_{rg}.csv: columns of the block")
    for k, f_ in sub.items():
        share_rec = float((coefs[rg][k].to_numpy() != 0).mean())
        check(f"selection share of {k} ({rg}), stored vs recomputed from the coefficients", f_, share_rec, 1e-9)
        check(f"selection share of {k} ({rg}), %", 100 * f_, ev[k], 0.05)
        val(f"selection.{tag}.{k}", f_, f"primary run, selection_freq_columns_{rg}.csv")

# 4(e) Holm over the four tests of the polarity step under other inference for the Clark-West tests
dep = pd.read_csv(tab_dir / "src/app_oos_values_20261108.csv")
dep = dep[dep.table == "dependence"].set_index(["row", "column"]).value
ds = insample_design(builder, "R3")
Z, names, mu, sd, keep = _zdesign(ds)
a1, a2 = _h4_weights(names, sd, "raw")
_, inv = np.unique(ds.day_tr, return_inverse=True)


def sq4_at(bw):
    beta, V = dk_cov_within(Z, ds.y_tr, ds.coin_tr, inv, bw)
    dlt, se = float(a2 @ beta), float(np.sqrt(a2 @ V @ a2))
    fr, sef = float(a1 @ beta), float(np.sqrt(a1 @ V @ a1))
    an = float((a1 - a2) @ beta); sea = float(np.sqrt((a1 - a2) @ V @ (a1 - a2)))
    return dict(delta=dlt, se=se, p=2 * norm.sf(abs(dlt / se)), fear=fr, sef=sef, anger=an, sea=sea)


b7 = sq4_at(7)
check("SQ4 difference at Driscoll-Kraay bandwidth 7 (engine design)", b7["delta"], cq.loc["SQ4", "delta"], 1e-8)
check("SQ4 Driscoll-Kraay SE at bandwidth 7", b7["se"], cq.loc["SQ4", "delta_dk_se"], 1e-8)
check("SQ4 p at bandwidth 7", b7["p"], cq.loc["SQ4", "p"], 1e-8)
check("cumulative fear, stored", b7["fear"], cq.loc["SQ4", "cum_fear"], 1e-8)
check("cumulative anger, stored", b7["anger"], cq.loc["SQ4", "cum_anger"], 1e-8)
EV_SQ4_P = {14: 0.081, 30: 0.092, 60: 0.089}
SETTINGS = [("L6", "p_hac_L6", 7), ("L14", "p_hac_L14", 14), ("L30", "p_hac_L30", 30), ("L60", "p_hac_L60", 60), ("b10", "p_boot_b10", None), ("b30", "p_boot_b30", None), ("b60", "p_boot_b60", None)]
EV_H = {"L14": 0.049, "L30": 0.061, "L60": 0.053}
SRC_H = "Holm over the four tests with the SQ1 and SQ3 p of the same setting (latex/tables/src/app_oos_values_20261108.csv)"
# the Clark-West p at the longer bandwidths, recomputed from the stored forecasts
for s, key, bw in SETTINGS:
    if bw is not None:
        for sq, h in (("SQ1", "H1"), ("SQ2", "H2"), ("SQ3", "H3")):
            aa, bb = STEPS[h]
            A, B = RUNG[aa], RUNG[bb]
            L = int(key.split("L")[1])
            pr = clark_west(A.y.to_numpy(), A.f_log.to_numpy(), B.f_log.to_numpy(), groups=A.target.to_numpy(), lags=L)["p_one_sided"]
            check(f"{sq} Clark-West p, {L} lags (forecasts vs values file)", pr, dep[(sq, key)], 1e-9)
for s, key, bw in SETTINGS:
    p = dep[("SQ1", key)], dep[("SQ2", key)], dep[("SQ3", key)]
    if bw is not None:
        k4 = sq4_at(bw)
        sq4p = k4["p"]
        which = f"SQ4 p at Driscoll-Kraay bandwidth {bw}" if bw != 7 else "main SQ4 p (bandwidth 7)"
        if bw in EV_SQ4_P:
            check(f"SQ4 p at Driscoll-Kraay bandwidth {bw}", sq4p, EV_SQ4_P[bw], 0.0006)
        val(f"holm.sq4_p_used.{s}", sq4p, f"{which}, from the primary run's in-sample design and model.inference.dk_cov_within")
    else:
        sq4p = float(cq.loc["SQ4", "p"])
        val(f"holm.sq4_p_used.{s}", sq4p, "SQ4 p of the main run (a block-bootstrap setting has no Driscoll-Kraay counterpart)")
    adj = holm(np.array([p[0], p[1], p[2], sq4p]))
    check(f"Holm p of the polarity step does not depend on the SQ4 p used ({s})", adj[1], holm(np.array([p[0], p[1], p[2], 1.0]))[1], 1e-12)
    val(f"holm.sq2_p.{s}", adj[1], SRC_H + ("; SQ4 at the same bandwidth" if bw else "; SQ4 p of the main run"))
    if s in EV_H:
        check(f"Holm-adjusted p of the polarity step, {s}", adj[1], EV_H[s], 0.0006)
    if s == "L6":
        check("Holm-adjusted p of the polarity step, main setting (vs confirmatory table)", adj[1], cq.loc["SQ2", "p_holm"], 1e-9)
        check("Holm-adjusted p of the attention step, main setting (vs confirmatory table)", adj[0], cq.loc["SQ1", "p_holm"], 1e-9)
        check("Holm-adjusted p of the emotion step, main setting (vs confirmatory table)", adj[2], cq.loc["SQ3", "p_holm"], 1e-9)
    if s in ("b10", "b30", "b60"):
        check_true(f"Holm-adjusted p of the polarity step well below 0.05 under the block bootstrap ({s}: {adj[1]:.4f})", 0.003 <= adj[1] <= 0.008)

# fear and anger by text source (platform-only runs)
SRC_P = "grid csv and the run's h4_bootstrap.csv / insample_coef.csv"
EV_P = {"A04_Reddit": ("reddit", -25.28, 0.0016, 4.66, 29.94), "A04_StockTwits": ("stocktwits", 1.91, 0.839, 1.75, -0.16), "A04_Twitter": ("twitter", 38.78, 0.0175, 8.41, -30.37)}
for run, (nm, ev_d, ev_p, ev_f, ev_a) in EV_P.items():
    row = m[m.run_id == run].iloc[0]
    hb = pd.read_csv(RUNS / row.run_dir / "h4_bootstrap.csv")
    t1, t2 = hb[hb.condition.str.startswith("theta1")].iloc[0], hb[hb.condition.str.startswith("theta2")].iloc[0]
    cf, dlt = t1.estimate, t2.estimate
    ca = cf - dlt
    ic = pd.read_csv(RUNS / row.run_dir / "insample_coef.csv")
    ic = ic[ic.rung == "R3"].set_index("variable").beta_raw
    ca2 = ic[[f"emo_anger_h{k}" for k in (1, 7, 30)]].sum()
    cf2 = ic[[f"emo_fear_h{k}" for k in (1, 7, 30)]].sum()
    check(f"{nm}: cum. anger from h4_bootstrap vs insample_coef", ca, ca2, 1e-6)
    check(f"{nm}: cum. fear from h4_bootstrap vs insample_coef", cf, cf2, 1e-6)
    check(f"{nm}: difference vs the evidence", dlt, ev_d, 0.006)
    check(f"{nm}: p vs the evidence", row.H4_p, ev_p, 0.0006 if nm != "stocktwits" else 0.0006)
    check(f"{nm}: cum. fear vs the evidence", cf, ev_f, 0.006)
    check(f"{nm}: cum. anger vs the evidence", ca, ev_a, 0.006)
    val(f"platform.{nm}.delta", dlt, SRC_P); val(f"platform.{nm}.p", row.H4_p, SRC_P); val(f"platform.{nm}.cum_fear", cf, SRC_P); val(f"platform.{nm}.cum_anger", ca, SRC_P)

# --------------------------------------------------------------------------- values and the check log
v = pd.DataFrame(values)
check_true("every value is finite", np.isfinite(v.value.to_numpy(float)).all())
check_true("value names are unique", v.name.is_unique)
for s in ("r3loss.se_sum_all_days", "attention_share.2021Q4", "holm.sq2_p.L30", "platform.twitter.delta", "selection.emotion.emo_fear_sq_h1"):
    assert (v.name == s).sum() == 1, s
out = tab_dir / "src/disc_values_20261109.csv"
tmp = Path(str(out) + ".tmp")
v.to_csv(tmp, index=False)
back = pd.read_csv(tmp)
assert list(back.columns) == ["name", "value", "source"] and len(back) == len(v) and np.allclose(back.value, v.value)
os.replace(tmp, out)
log = tmp_dir / "disc_tables_20261109_checks.txt"
tmp_l = Path(str(log) + ".tmp")
tmp_l.write_text("\n".join(CHECKS) + "\n", encoding="utf-8")
os.replace(tmp_l, log)
print(len(v), "values written;", len(CHECKS), "checks passed; log:", log)
