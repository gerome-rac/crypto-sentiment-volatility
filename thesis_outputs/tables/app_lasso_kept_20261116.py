#!/usr/bin/env python
"""app_lasso_kept_20261116.py - Appendix C: the attention and polarity terms that the adaptive LASSO keeps when each
rung is estimated on all days, with the p-values of the recentered block bootstrap (Section 4.3).

Every number comes from the stored run A00_insample_recentered_medrv_20261108T095413Z (1,999 replications), whose
LASSO fit is checked here against the main run A00_final_A50_20261028T134243Z. The counts stated in Section 5.2 and
Table 5.3 (nine attention terms kept, two / three / five of them significant at the 1% / 5% / 10% level; one polarity
term; no emotion term) are asserted.

Writes (temp file -> check -> replace): <tables_dir>/app_lasso_kept.tex
Usage: python app_lasso_kept_20261116.py <code_dir> <tables_dir>
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

code_dir, tab_dir = map(Path, sys.argv[1:3])
RUNS = code_dir / "data/pipeline/model_runs"
BOOT = RUNS / "A00_insample_recentered_medrv_20261108T095413Z"
MAIN = RUNS / "A00_final_A50_20261028T134243Z"

LABEL = {"att_reddit": "Reddit posts, coin", "att_stocktwits": "StockTwits posts, coin", "att_twitter": "Twitter posts, coin",
         "att_google": "Google searches, coin", "att_wiki": "Wikipedia page views, coin", "att_news": "News headlines, coin",
         "mkt_reddit": "Reddit posts, market", "mkt_stocktwits": "StockTwits posts, market", "mkt_twitter": "Twitter posts, market",
         "mkt_wiki": "Wikipedia page views, market", "mkt_google_cryptocurrency": r"Google searches, \emph{cryptocurrency}",
         "mkt_google_crypto": r"Google searches, \emph{crypto}", "pol": "Polarity score", "pol_sd": "Polarity score, within-day SD"}
TERM = {"h1": "1-day", "h7": "7-day", "h30": "30-day", "dlog": "Daily change"}


def split(var):
    base, suffix = var.rsplit("_", 1)
    if base.endswith("_sq"):
        return LABEL[base[:-3]], TERM[suffix] + ", centered square"
    return LABEL[base], TERM[suffix]


def kept(rung, block, n_terms):
    d = pd.read_csv(BOOT / f"insample_lasso_recentered_{rung}.csv")
    m = pd.read_csv(MAIN / f"insample_lasso_{rung}.csv")
    assert list(d.variable) == list(m.variable)
    assert np.allclose(d.beta_z.to_numpy(), m.beta_z.to_numpy(), atol=1e-8, rtol=0), f"{rung}: LASSO fit differs from the main run"
    assert (d.boot_reps == 1999).all()
    w = d[d.variable == "ctrl_weekend"].iloc[0]          # beta_z = beta_raw x SD of the regressor (model/inference.py)
    assert 0.44 < w.beta_z / w.beta_raw < 0.46, w
    b = d[d.block == block]
    assert len(b) == n_terms, (rung, block, len(b))
    k = b[b.selected].sort_values(["p_value", "variable"])
    assert (k.beta_z != 0).all() and (b[~b.selected].beta_z == 0).all()
    return k


att, pol, emo = kept("R1", "attention", 45), kept("R2", "polarity", 9), kept("R3", "emotion", 54)
assert len(att) == 9 and [int((att.p_value < a).sum()) for a in (0.01, 0.05, 0.10)] == [2, 3, 5], att[["variable", "p_value"]]
assert len(pol) == 1 and float(pol.p_value.iloc[0]) < 0.10 and float(pol.p_value.iloc[0]) >= 0.05
assert len(emo) == 0
s = pd.read_csv(BOOT / "insample_lasso_recentered_summary.csv").set_index(["rung", "block"])
assert tuple(s.loc[("R1", "attention"), ["n_selected", "n_p_lt_01", "n_p_lt_05", "n_p_lt_10"]]) == (9, 2, 3, 5)
p = att.set_index("variable").p_value                      # the three terms named in Section 5.2
assert round(p["att_stocktwits_h1"], 3) == 0.001 and round(p["att_twitter_h30"], 3) == 0.001 and round(p["mkt_google_cryptocurrency_h1"], 3) == 0.016
b = att.set_index("variable").beta_z
assert b["att_stocktwits_h1"] > 0 and b["att_twitter_h30"] < 0 and b["mkt_google_cryptocurrency_h1"] > 0


def rows(k):
    out = []
    for r in k.itertuples():
        lab, term = split(r.variable)
        out.append(f"{lab} & {term} & ${r.beta_z:+.3f}$ & {r.p_value:.3f} \\\\")
    return out


tex = "\n".join([
    r"%% Table of 16 Nov 2026 (Appendix C). Written by latex/tables/src/app_lasso_kept_20261116.py from the stored run",
    r"%% A00_insample_recentered_medrv_20261108T095413Z (recentered block bootstrap, 1,999 replications).",
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Attention and polarity terms kept by the adaptive LASSO on all days}\label{tab:app_lasso_kept}",
    r"\small",
    r"\begin{tabular}{@{}llrr@{}}",
    r"\toprule",
    r"Variable & Term & Coefficient & $p$ \\",
    r"\midrule",
    r"\multicolumn{4}{@{}l}{\textit{Attention, in R1: 9 of 45 terms kept}} \\"] + rows(att) + [
    r"\addlinespace",
    r"\multicolumn{4}{@{}l}{\textit{Polarity, in R2: 1 of 9 terms kept}} \\"] + rows(pol) + [
    r"\addlinespace",
    r"\multicolumn{4}{@{}l}{\textit{Emotions, in R3: none of the 54 terms kept}} \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\srcnote{Own calculations; adaptive LASSO of \cref{sec:estimation} on all days, each block in the rung in which it enters. Coefficient: change in next-day log RV when the term rises by one standard deviation. $p$: recentered block bootstrap with $1{,}999$ replications (\cref{sec:estimation}). Variables and terms as in \cref{tab:app_variables}; daily change: log change on day $t$.}",
    r"\end{table}", ""])
out = tab_dir / "app_lasso_kept.tex"
tmp = out.with_suffix(".tex.tmp")
tmp.write_text(tex, encoding="utf-8")
assert tmp.read_text(encoding="utf-8") == tex
os.replace(tmp, out)
print("written:", out, "| attention", len(att), "| polarity", len(pol), "| emotions", len(emo))
