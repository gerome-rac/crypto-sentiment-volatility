"""Appendix B table: agreement of the two coders on the hand-labeled posts (15 Nov 2026).
Reads code/data/validation/sample_20261002/analysis_consensus_20261115/agreement_two_rounds_20261115.csv
(written by agreement_two_rounds_build.py there; its columns by round are not printed, Gerome's decision of 15 Nov)
and writes latex/tables/app_kappa.tex. Run from the thesis root: python latex/tables/src/app_kappa_20261115.py"""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
r = pd.read_csv(ROOT / "code/data/validation/sample_20261002/analysis_consensus_20261115/agreement_two_rounds_20261115.csv")
L = ["%% Table of 15 Nov 2026 (Appendix B): agreement of the two coders. Written by latex/tables/src/app_kappa_20261115.py from",
     "%% code/data/validation/sample_20261002/analysis_consensus_20261115/agreement_two_rounds_20261115.csv (columns for all posts).",
     r"\begin{table}[htbp]", r"\centering", r"\caption{Agreement of the two coders on the hand-labeled posts}\label{tab:app_kappa}", r"\small",
     r"\begin{tabular}{@{}lrrl@{}}", r"\toprule", r"Item & Same label (\%) & Cohen's $\kappa$ & Reading \\", r"\midrule"]
for x in r.itertuples(index=False):
    L.append(rf"{x.item} & {100 * x.same_all:.1f} & {x.kappa_all:.2f} & {x.reading_all} \\")
n_all, n_tone = int(r["n_all"].iloc[0]), int(r["n_all"].iloc[-1])
L += [r"\bottomrule", r"\end{tabular}",
      rf"\srcnote{{Own calculations; posts labeled by both coders, unweighted: ${n_all}$ for the emotion and ${n_tone}$ for the tone. Reading: $\kappa \geq 0.61$ reliable, $0.41$ to $0.60$ tentative, $\kappa \leq 0.40$ failed (\cref{{sec:validationdesign}}). Three valence classes: happy counts as positive, neutral and surprise as neutral, and sad, anger, disgust, and fear as negative.}}",
      r"\end{table}", ""]
(ROOT / "latex/tables/app_kappa.tex").write_text("\n".join(L), encoding="utf-8"); print("written latex/tables/app_kappa.tex")
