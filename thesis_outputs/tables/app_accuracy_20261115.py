"""Appendix B table: EmTract and FinBERT against the reference labels of the hand-labeled sample (15 Nov 2026).
Reads code/data/validation/sample_20261002/analysis_consensus_20261115/reference_vs_models_20261115.csv
(written by reference_vs_models_build.py in that folder) and writes latex/tables/app_accuracy.tex.
Run from the thesis root: python latex/tables/src/app_accuracy_20261115.py"""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
r = pd.read_csv(ROOT / "code/data/validation/sample_20261002/analysis_consensus_20261115/reference_vs_models_20261115.csv")
r["class"] = r["class"].fillna("")
def g(comp, metric, cls=""):
    m = r[r["comparison"].str.startswith(comp) & (r["metric"] == metric) & (r["class"] == cls) & (r["weights"] == "cell_equal") & (r["domain"] == "all")]
    assert len(m) == 1, (comp, metric, cls, len(m))
    return m.iloc[0]
def pct(x): return f"{100 * x['value']:.1f} & [{100 * x['ci_lo']:.1f}, {100 * x['ci_hi']:.1f}]"
def num(x): return f"{x['value']:.2f} & [{x['ci_lo']:.2f}, {x['ci_hi']:.2f}]"
n_emo = int(g("emotion7", "accuracy")["n"]); n_tone = int(g("tone3: FinBERT", "accuracy")["n"])
maj = 100 * g("emotion7", "majority_class_share", "neutral")["value"]
L = [
 "%% Table of 15 Nov 2026 (Appendix B): EmTract and FinBERT against the reference labels of the hand-labeled sample.",
 "%% Written by latex/tables/src/app_accuracy_20261115.py from",
 "%% code/data/validation/sample_20261002/analysis_consensus_20261115/reference_vs_models_20261115.csv (cell-equal weights).",
 r"\begin{table}[htbp]", r"\centering", r"\caption{EmTract and FinBERT against the reference labels}\label{tab:app_accuracy}", r"\small",
 r"\begin{tabular}{@{}lrl@{}}", r"\toprule", r" & Value & 95\% interval \\", r"\midrule",
 rf"\multicolumn{{3}}{{@{{}}l}}{{\textit{{Panel A: EmTract's top class against the reference emotion ({n_emo} posts)}}}} \\",
 rf"Same class, seven classes (\%) & {pct(g('emotion7', 'accuracy'))} \\",
 rf"Cohen's $\kappa$, seven classes & {num(g('emotion7', 'kappa'))} \\",
 rf"Same class, three valence classes (\%) & {pct(g('valence3', 'accuracy'))} \\",
 rf"Posts EmTract labels fear that carry the reference label fear (\%) & {pct(g('emotion7', 'precision', 'fear'))} \\",
 rf"Posts with the reference label fear that EmTract labels fear (\%) & {pct(g('emotion7', 'recall', 'fear'))} \\",
 rf"Posts EmTract labels anger that carry the reference label anger (\%) & {pct(g('emotion7', 'precision', 'anger'))} \\",
 rf"Posts with the reference label anger that EmTract labels anger (\%) & {pct(g('emotion7', 'recall', 'anger'))} \\",
 rf"AUC of the fear probability & {num(g('probability', 'auc', 'fear'))} \\",
 rf"AUC of the anger probability & {num(g('probability', 'auc', 'anger'))} \\",
 r"\addlinespace",
 rf"\multicolumn{{3}}{{@{{}}l}}{{\textit{{Panel B: FinBERT against the reference tone ({n_tone} posts)}}}} \\",
 rf"Same class, three classes (\%) & {pct(g('tone3: FinBERT', 'accuracy'))} \\",
 rf"Cohen's $\kappa$ & {num(g('tone3: FinBERT', 'kappa'))} \\",
 rf"AUC of the polarity score, positive against negative posts & {num(g('tone: FinBERT polarity', 'auc'))} \\",
 r"\bottomrule", r"\end{tabular}",
 rf"\srcnote{{Own calculations. Reference labels: the coders' labels as reviewed and fixed by the author. Every combination of platform and coin has the same weight, and the sampling weights within it are used. Intervals: bootstrap within the sampling strata, $2{{,}}000$ draws. Always choosing neutral agrees with the reference emotion for ${maj:.1f}\%$ of the posts. AUC: probability that a post with the label has a higher score than a post without it.}}",
 r"\end{table}", ""]
(ROOT / "latex/tables/app_accuracy.tex").write_text("\n".join(L), encoding="utf-8")
print("written latex/tables/app_accuracy.tex")
