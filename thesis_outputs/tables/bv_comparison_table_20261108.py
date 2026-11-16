#!/usr/bin/env python
"""bv_comparison_table_20261108.py - appendix table: the confirmatory family with C/J from MedRV (primary since
D8 Deviation 7, 8 Nov 2026) and from adjacent BV + Huang-Tauchen (the measurement fixed at the D8 freeze).
Input: <code_dir>/data/pipeline/model_runs/confirmatory_sq4dk_20261108/confirmatory_sq4dk.csv (written by
python -m model.confirmatory_sq4 from the two final runs of 28 Oct 2026) and both runs' nested_tests.csv (OOS R2).
Writes <tables_dir>/app_oos_bv.tex (temp file -> check -> replace).
Usage: python bv_comparison_table_20261108.py <code_dir> <tables_dir>
"""
import os
import sys
from pathlib import Path

import pandas as pd

code_dir, tab_dir = map(Path, sys.argv[1:3])
RUNS = code_dir / "data" / "pipeline" / "model_runs"
MED, BV = "A00_final_A50_20261028T134243Z", "A00_final_20261028T115254Z"
c = pd.read_csv(RUNS / "confirmatory_sq4dk_20261108" / "confirmatory_sq4dk.csv").set_index(["run", "subquestion"])
nt = {r: pd.read_csv(RUNS / r / "nested_tests.csv").set_index("test") for r in (MED, BV)}
ROWS = [("SQ1", "H1", "Attention: R0 $\\to$ R1"), ("SQ2", "H2", "Polarity: R1 $\\to$ R2"),
        ("SQ3", "H3", "Emotions: R2 $\\to$ R3"), ("SQ4", None, "Fear $-$ anger, in sample")]


def fp(p):
    return "$<$0.001" if p < 0.0005 else f"{p:.3f}"


lines = []
for sq, h, lab in ROWS:
    cells = []
    for r in (MED, BV):
        row = c.loc[(r, sq)]
        r2 = "--" if h is None else f"{100 * nt[r].loc[h, 'r2_oos_big_vs_small']:+.2f}"
        cells += [fp(row.p), fp(row.p_holm), r2]
    lines.append(f"{sq} & {lab} & " + " & ".join(cells) + r" \\")
tex = r"""\begin{table}[htbp]
\centering
\caption{The confirmatory tests with MedRV and with bipower variation}\label{tab:app_bv}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{MedRV (main)} & \multicolumn{3}{c}{Bipower variation} \\
\cmidrule(lr){3-5}\cmidrule(l){6-8}
 & Step or contrast & $p$ & $p$ (Holm) & $R^2$ (\%) & $p$ & $p$ (Holm) & $R^2$ (\%) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\srcnote{Own calculations from the final runs. $\Cc$ and $\Jc$ from MedRV and the ratio test of \cref{eq:jumptestZ}, or from adjacent bipower variation and the ratio test of \textcite[p.~463, eq.~9]{huangRelativeContributionJumps2005}; everything else is identical. SQ1--SQ3: one-sided Clark--West $p$-value and out-of-sample $R^2$ of the larger rung relative to the smaller one. SQ4: two-sided Driscoll--Kraay test of \cref{eq:sq4_test}. $p$ (Holm): adjustment across the four subquestions within each measurement.}
\end{table}
"""
out = tab_dir / "app_oos_bv.tex"
tmp = out.with_suffix(".tex.tmp")
tmp.write_text(tex, encoding="utf-8")
assert tmp.read_text(encoding="utf-8") == tex
os.replace(tmp, out)
print("\n".join(lines))
