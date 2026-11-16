#!/usr/bin/env python
"""app_selection_terms_20261116.py - Appendix D: how often the adaptive LASSO selects each polarity term (in R2) and
each emotion term (in R3) across the 885 rolling estimation windows of the main run.

Every number comes from selection_freq_columns_R2.csv and selection_freq_columns_R3.csv of the main run
A00_final_A50_20261028T134243Z. The rates that the appendix text gave in words until 16 Nov 2026 are asserted.

Writes (temp file -> check -> replace): <tables_dir>/app_selection_terms.tex
Usage: python app_selection_terms_20261116.py <code_dir> <tables_dir>
"""
import os
import sys
from pathlib import Path

import pandas as pd

code_dir, tab_dir = map(Path, sys.argv[1:3])
MAIN = code_dir / "data/pipeline/model_runs/A00_final_A50_20261028T134243Z"
N = 885
H = ("h1", "h7", "h30")
EMO = [("fear", "Fear"), ("anger", "Anger"), ("sad", "Sadness"), ("disgust", "Disgust"), ("surprise", "Surprise"), ("happy", "Happiness")]


def rates(rung, block, n_terms):
    d = pd.read_csv(MAIN / f"selection_freq_columns_{rung}.csv")
    d = d[d.block == block].set_index("column").freq
    assert len(d) == n_terms, (rung, block, len(d))
    k = d * N
    assert ((k - k.round()).abs() < 1e-6).all()            # every rate is a count of windows out of 885
    return d


def cell(x):
    return "---" if x == 0 else f"{100 * x:.1f}"


def line(label, d, stem):
    kinds = [f"{stem}_{{h}}", f"{stem}_sq_{{h}}", f"{stem}_sd_{{h}}"]
    return label + " & " + " & ".join(cell(d[k.format(h=h)]) for k in kinds for h in H) + r" \\"


pol, emo = rates("R2", "polarity", 9), rates("R3", "emotion", 54)
r1 = lambda x: round(100 * x, 1)
assert [r1(pol[f"pol_sq_{h}"]) for h in H] == [14.1, 8.7, 0.9]
assert [r1(pol[f"pol_{h}"]) for h in H] == [4.7, 3.8, 0.1]
assert [r1(pol[f"pol_sd_{h}"]) for h in H] == [4.0, 0.9, 0.1]
sel = emo[emo > 0]
assert len(sel) == 9 and not any("_sd_" in c for c in sel.index), sel
want = {"emo_fear_h1": 1.8, "emo_fear_sq_h1": 4.4, "emo_fear_sq_h30": 2.7, "emo_sad_sq_h1": 3.1, "emo_happy_sq_h30": 2.1,
        "emo_happy_sq_h7": 1.2, "emo_happy_sq_h1": 0.2, "emo_anger_sq_h30": 0.2, "emo_disgust_sq_h1": 0.2}
assert {c: r1(v) for c, v in sel.items()} == want, sel
assert round(pol["pol_sq_h1"] * N) == 125

tex = "\n".join([
    r"%% Table of 16 Nov 2026 (Appendix D). Written by latex/tables/src/app_selection_terms_20261116.py from",
    r"%% selection_freq_columns_R2.csv and selection_freq_columns_R3.csv of the main run A00_final_A50_20261028T134243Z.",
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Selection rate of each polarity and emotion term across the 885 estimation windows (\%)}\label{tab:app_selterms}",
    r"\small",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
    r"\toprule",
    r" & \multicolumn{3}{c}{Level} & \multicolumn{3}{c}{Centered square} & \multicolumn{3}{c}{Within-day SD} \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
    r" & 1-day & 7-day & 30-day & 1-day & 7-day & 30-day & 1-day & 7-day & 30-day \\",
    r"\midrule",
    r"\multicolumn{10}{@{}l}{\textit{Polarity, in R2}} \\",
    line("Polarity score", pol, "pol"),
    r"\addlinespace",
    r"\multicolumn{10}{@{}l}{\textit{Emotions, in R3}} \\"] + [line(lab, emo, f"emo_{e}") for e, lab in EMO] + [
    r"\bottomrule",
    r"\end{tabular}",
    r"\srcnote{Own calculations; main specification. Share of the 885 rolling estimation windows in which the adaptive LASSO keeps the term, in percent; ---: never selected. Level and centered square: of the daily mean; within-day SD: standard deviation across the posts of a day; each as a 1-, 7-, and 30-day average (\cref{tab:app_variables}).}",
    r"\end{table}", ""])
out = tab_dir / "app_selection_terms.tex"
tmp = out.with_suffix(".tex.tmp")
tmp.write_text(tex, encoding="utf-8")
assert tmp.read_text(encoding="utf-8") == tex
os.replace(tmp, out)
print("written:", out)
