#!/usr/bin/env python
"""wald_size_table_20261107.py - appendix table 'Size of the joint Wald tests' from the engine validation of
7 Oct 2026 (code/model/validate.py wald_size): observed realized volatility, synthetic regressors unrelated to it
(model/tests/conftest.synth_tables), in-sample OLS of R1 and R3 as in the engine.
Inputs (in <validation_dir>): validation_wald_size_summary.csv (100 replications, asymptotic p-values) and
validation_wald_size_boot199_summary.csv (60 replications, block bootstrap with 199 draws each).
Writes <tables_dir>/app_wald_size.tex (temp file -> check -> replace).
Usage: python wald_size_table_20261107.py <validation_dir> <tables_dir>
"""
import os
import sys
from pathlib import Path

import pandas as pd

vdir, tdir = map(Path, sys.argv[1:3])
a = pd.read_csv(vdir / "validation_wald_size_summary.csv")
b = pd.read_csv(vdir / "validation_wald_size_boot199_summary.csv")
assert set(a.n) == {100} and set(b.n) == {60}


def row(df, block, cov):
    r = df[(df.block == block) & (df["cov"] == cov)]
    assert len(r) == 1, (block, cov)
    return r.iloc[0]


ROWS = [  # label, q, covariance, reference distribution, source frame, block, cov
    ("Attention block (R1)", 45, "Driscoll--Kraay", "$\\chi^2$", a, "attention", "DK"),
    ("", 45, "Newey--West within coin", "$\\chi^2$", a, "attention", "NW"),
    ("", 45, "Driscoll--Kraay", "block bootstrap", b, "attention", "DK_bootstrap"),
    ("Emotion block (R3)", 54, "Driscoll--Kraay", "$\\chi^2$", a, "emotion", "DK"),
    ("", 54, "Driscoll--Kraay, 30 lags", "$\\chi^2$", a, "emotion", "DK_bw30"),
    ("", 54, "Newey--West within coin", "$\\chi^2$", a, "emotion", "NW"),
    ("", 54, "Driscoll--Kraay", "block bootstrap", b, "emotion", "DK_bootstrap"),
    ("One coefficient (fear, 1 day)", 1, "Driscoll--Kraay", "normal", a, "emo_fear_h1 single", "DK"),
]
lines = []
for lab, q, cov_txt, ref, df, block, cov in ROWS:
    r = row(df, block, cov)
    lines.append(f"{lab} & {q} & {cov_txt} & {ref} & {r.reject_05:.3f} & {r.reject_10:.3f} & {int(r.n)} \\\\")
tex = r"""\begin{table}[htbp]
\centering
\caption{Rejection rates of the joint Wald tests when the null is true}\label{tab:app_wald_size}
\small
\begin{tabular}{@{}lrllrrr@{}}
\toprule
Test & $q$ & Standard errors & $p$-values from & 5\% & 10\% & Replications \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\srcnote{Own simulation with the estimation code (7~October 2026). Each replication keeps the observed realized volatility of the five coins and replaces all attention, polarity, emotion, and control variables by synthetic series unrelated to it, so that every block coefficient is zero; the rung is estimated by OLS on all days. Columns 5\% and 10\%: share of replications in which the test rejects at that level. Bootstrap rows: 199 bootstrap draws per replication (499 in the thesis's estimates).}
\end{table}
"""
out = tdir / "app_wald_size.tex"
tmp = out.with_suffix(".tex.tmp")
tmp.write_text(tex, encoding="utf-8")
assert tmp.read_text(encoding="utf-8") == tex
os.replace(tmp, out)
print("\n".join(lines))
