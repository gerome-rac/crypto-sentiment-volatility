#!/usr/bin/env python3
"""selection_rates_20261112.py

Data for the figure "How often each variable is selected across the 885 estimation windows" (Appendix D) and the
counts of the variable table (Appendix A).

For every penalized variable, the share of the 885 rolling estimation windows of the main specification in which the
adaptive LASSO keeps at least one of its terms, taken in the rung in which the variable's block enters:
controls and attention in R1, polarity in R2, emotions in R3. The six HAR-CJ terms are unpenalized and not written.

Inputs (paths on the command line):
  <run_dir>       folder of the main run (selection_freq_variables_R{1,2,3}.csv, coefs_R{1,2,3}.parquet)
  <insample_dir>  folder of the in-sample adaptive LASSO fit with re-centered bootstrap
                  (insample_lasso_recentered_R{1,2,3}.csv; column p_value)
  <spec_csv>      variable specification of the main run (variable_spec_A50.csv)
  <out_csv>       output: block, variable, label, freq, sig_insample

Checks (asserted):
  - every stored selection rate equals the share recomputed from the coefficient paths (any term non-zero);
  - 885 windows; 17 / 62 / 71 / 125 regressors in R0 / R1 / R2 / R3, from the specification and from the paths;
  - the rates printed in Section 5.2 and Appendix D (48, 42, 40 per cent; 4.4, 3.1, 2.5 per cent) are reproduced;
  - the in-sample marks are two controls (weekend indicator, Tether news volume), the two attention terms of
    Table 5.3 and none in the polarity and emotion blocks.

Usage: python selection_rates_20261112.py <run_dir> <insample_dir> <spec_csv> <out_csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run_dir, ins_dir, spec_csv, out_csv = (Path(a) for a in sys.argv[1:5])

LABELS = {
    "ctrl_weekend": "Weekend indicator",
    "ctrl_move": "MOVE index",
    "ctrl_vix": "VIX",
    "ctrl_tether_news": "Tether news volume",
    "ctrl_gold_ret": "Gold return",
    "ctrl_baa10y": "BAA credit spread",
    "ctrl_dgs10_chg": "Change in the 10-year yield",
    "ctrl_dxy_ret": "Dollar index return",
    "ctrl_term": "Term spread",
    "ctrl_tether_news_chg": "Change in Tether news volume",
    "ctrl_epu": "Economic policy uncertainty",
    "mkt_reddit": "Reddit posts, market",
    "att_stocktwits": "StockTwits posts, coin",
    "att_google": "Google searches, coin",
    "att_reddit": "Reddit posts, coin",
    "mkt_google_cryptocurrency": "Google searches, “cryptocurrency”",
    "mkt_google_crypto": "Google searches, “crypto”",
    "mkt_stocktwits": "StockTwits posts, market",
    "mkt_twitter": "Twitter posts, market",
    "att_twitter": "Twitter posts, coin",
    "mkt_wiki": "Wikipedia page views, market",
    "att_news": "News headlines, coin",
    "att_wiki": "Wikipedia page views, coin",
    "pol": "Polarity score",
    "pol_sd": "Within-day dispersion of polarity",
    "emo_fear": "Fear",
    "emo_sad": "Sadness",
    "emo_happy": "Happiness",
    "emo_anger": "Anger",
    "emo_disgust": "Disgust",
    "emo_surprise": "Surprise",
}
ENTRY_RUNG = {"control": "R1", "attention": "R1", "polarity": "R2", "emotion": "R3"}
SUFFIXES = ("_h1", "_h7", "_h30", "_dlog", "_csq_h1", "_csq_h7", "_csq_h30")


def base_name(col: str, names: list[str]) -> str:
    """Variable to which a coefficient column belongs (longest specification name that prefixes the column)."""
    hits = [n for n in names if col == n or col.startswith(n + "_")]
    assert hits, col
    return max(hits, key=len)


spec = pd.read_csv(spec_csv)
spec = spec[spec.block != "har_rv"]
names = spec.name.tolist()
n_terms = (spec.har * 3 + spec.dlog + spec.csq * 3).where(spec.har == 1, 1)
per_block = n_terms.groupby(spec.block).sum()
assert per_block.to_dict() == {"attention": 45, "control": 11, "emotion": 54, "har": 6, "polarity": 9}, per_block
assert int(n_terms.sum()) == 125

rows = []
n_reg = {}
for rung in ("R1", "R2", "R3"):
    coefs = pd.read_parquet(run_dir / f"coefs_{rung}.parquet")
    assert len(coefs) == 885 and coefs.origin.is_unique, rung
    cols = [c for c in coefs.columns if c not in ("origin", "group")]
    n_reg[rung] = len(cols)
    owner = pd.Series({c: base_name(c, names) for c in cols})
    recomputed = {v: float((coefs[owner.index[owner == v]].to_numpy() != 0).any(axis=1).mean())
                  for v in owner.unique()}
    stored = pd.read_csv(run_dir / f"selection_freq_variables_{rung}.csv")
    assert set(stored.variable) == set(recomputed), rung
    for r in stored.itertuples():
        assert abs(r.freq - recomputed[r.variable]) < 1e-12, (rung, r.variable, r.freq, recomputed[r.variable])
        if ENTRY_RUNG.get(r.block) == rung:
            rows.append({"block": r.block, "variable": r.variable, "freq": r.freq})
assert n_reg == {"R1": 62, "R2": 71, "R3": 125}, n_reg

out = pd.DataFrame(rows)
assert out.variable.is_unique and len(out) == 11 + 12 + 2 + 12, len(out)

# In-sample marks: variables with a term that is significant at the 1% level (bootstrap p-value of the in-sample
# adaptive LASSO), each block in the rung in which it enters. Until 14 Nov 2026 the rule was a Holm-adjusted p-value
# below 5% within the block, applied to the three text blocks only; both rules select the same terms in every block
# (asserted below), and the controls are marked since 14 Nov 2026.
marks = set()
for rung, block in (("R1", "control"), ("R1", "attention"), ("R2", "polarity"), ("R3", "emotion")):
    t = pd.read_csv(ins_dir / f"insample_lasso_recentered_{rung}.csv")
    t = t[t.block == block]
    assert set(t.variable[t.p_value < 0.01]) == set(t.variable[t.p_holm_block < 0.05]), (rung, block)
    t = t[t.p_value < 0.01]
    marks |= {base_name(c, names) for c in t.variable}
assert marks == {"ctrl_weekend", "ctrl_tether_news", "att_stocktwits", "att_twitter"}, marks
out["sig_insample"] = out.variable.isin(marks).astype(int)

# The six within-day dispersions of the emotions are never selected and are not drawn.
sd = out[out.variable.str.startswith("emo_") & out.variable.str.endswith("_sd")]
assert len(sd) == 6 and (sd.freq == 0).all()
out = out[~out.variable.isin(sd.variable)].copy()
assert set(out.variable) == set(LABELS), set(out.variable) ^ set(LABELS)
out["label"] = out.variable.map(LABELS)

pct = (out.set_index("variable").freq * 100).round(1)
assert [round(pct[v]) for v in ("mkt_reddit", "att_stocktwits", "att_google")] == [48, 42, 40]
assert [pct[v] for v in ("emo_fear", "emo_sad", "emo_happy")] == [4.4, 3.1, 2.5]
assert pct["ctrl_weekend"] == 97.4 and pct["pol"] == 20.1

order = {"control": 0, "attention": 1, "polarity": 2, "emotion": 3}
out = out.assign(_o=out.block.map(order)).sort_values(["_o", "freq"], ascending=[True, False]).drop(columns="_o")
tmp = out_csv.with_suffix(".tmp.csv")
out[["block", "variable", "label", "freq", "sig_insample"]].to_csv(tmp, index=False)
back = pd.read_csv(tmp)
assert len(back) == 31 and np.allclose(back.freq, out.freq.to_numpy(), atol=0, rtol=0)
tmp.replace(out_csv)
print(out[["block", "label", "freq", "sig_insample"]].assign(freq=lambda d: (d.freq * 100).round(1)).to_string(index=False))
print("regressors per block:", per_block.to_dict(), "| per rung:", {"R0": 17, **n_reg})
