#!/usr/bin/env python3
"""
15b_sidearm_panels.py -- panel, variable specifications and run queue of the side arms of 12 Nov 2026.

The side arms change the emotion block (rung R3) of the primary ladder only (primary = former arm A50: C and J from
MedRV; data/pipeline/panel/arms/final_20261028/A50.parquet with model/variable_spec_A50.csv). One panel serves all
of them: the primary panel plus the new columns; each arm's variable specification names the columns that enter.

New columns (coin x day; from data/pipeline/panel/text_sidearms_daily.parquet, POOLED rows, step 15)
    disp_gini_between, disp_gini_total, disp_gini_within, disp_ent_between, disp_ent_total, disp_ent_within
    emo_<k>_pos, emo_<k>_neg       valenced emotions formed per post (six emotions k), precision-weighted, pooled
    int_pol_<k> = pol * emo_<k>    product of the two daily pooled values of the primary panel (day-level interaction)

Arms (every new variable enters as its 1-, 7- and 30-day average; block 'emotion', so it enters at R3)
    A75..A80          the 54 emotion terms of the primary plus one dispersion measure (3 terms)
    A75_alone..A80_alone   emotion block = the dispersion measure only (R3 = R2 + 3 terms)
    A81   + positive and negative surprise (6 terms)          A82   + all twelve valenced emotions (36 terms)
    A83   + polarity x surprise, day level (3 terms)          A84   + polarity x each emotion, day level (18 terms)

Checks (asserted before anything is written): 6,400 rows; every column of the primary panel unchanged; the new
columns matched on coin and day with no key lost; int_pol_<k> equals the product of the stored columns.

Outputs
    data/pipeline/panel/arms/sidearms_20261112/sidearms.parquet (+ .stats.json)
    model/variable_spec_<arm>.csv                       one per arm
    data/pipeline/model_runs/_queue_sidearms_20261112.csv
Usage (from code/):  python pipeline/15b_sidearm_panels.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
PANEL = CODE / "data" / "pipeline" / "panel"
SRC = PANEL / "arms" / "final_20261028" / "A50.parquet"
SIDE = PANEL / "text_sidearms_daily.parquet"
OUT = PANEL / "arms" / "sidearms_20261112" / "sidearms.parquet"
SPEC0 = CODE / "model" / "variable_spec_A50.csv"
QUEUE = CODE / "data" / "pipeline" / "model_runs" / "_queue_sidearms_20261112.csv"
EMO6 = ["happy", "sad", "anger", "disgust", "surprise", "fear"]
COINS = ["BTC", "ETH", "BNB", "XRP", "DOGE"]

DISP = {   # arm -> (column, note)
    "A75": ("disp_gini_between", "between-post emotion dispersion: root of the summed within-day variances of the seven "
                                 "EmTract probabilities (SD of the message signal, Antweiler and Frank 2004 eq. 7-8, over seven classes)"),
    "A76": ("disp_gini_total", "total emotion dispersion (Gini): 1 - sum of the squared mean probabilities of the day"),
    "A77": ("disp_gini_within", "within-post emotion dispersion (Gini): mean over posts of 1 - sum of squared probabilities"),
    "A78": ("disp_ent_between", "between-post emotion dispersion (entropy): Jensen-Shannon divergence of the day's posts"),
    "A79": ("disp_ent_total", "total emotion dispersion: entropy of the day's mean probabilities"),
    "A80": ("disp_ent_within", "within-post emotion dispersion: mean entropy of a post's probabilities"),
}
VALENCED = {
    "A81": ([f"emo_surprise_{s}" for s in ("pos", "neg")], "post level"),
    "A82": ([f"emo_{k}_{s}" for k in EMO6 for s in ("pos", "neg")], "post level"),
    "A83": (["int_pol_surprise"], "day level"),
    "A84": ([f"int_pol_{k}" for k in EMO6], "day level"),
}


def md5(p: Path) -> str:
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def note_for(col: str) -> str:
    if col.startswith("int_pol_"):
        k = col[len("int_pol_"):]
        return f"day-level interaction: pol x emo_{k}, product of the two daily pooled values of the primary panel (not centred)"
    k, s = col[len("emo_"):].rsplit("_", 1)
    cls = "positive" if s == "pos" else "negative"
    return (f"valenced emotion, post level: daily mean of P({k}) x FinBERT P({cls}) of the same post; "
            f"stratified, precision-weighted and pooled as the emotion means (step 15)")


def main() -> int:
    key = ["coin", "day"]
    base = pd.read_parquet(SRC)
    base["day"] = pd.to_datetime(base["day"])
    assert len(base) == 6400 and not base.duplicated(key).any()
    side = pd.read_parquet(SIDE)
    side = side[(side.platform == "POOLED") & side.series.isin(COINS)].rename(columns={"series": "coin"})
    side["day"] = pd.to_datetime(side["day"].astype(str))
    new_cols = [c for c, _ in DISP.values()] + VALENCED["A82"][0]
    assert not side.duplicated(key).any() and len(side) == 6400
    out = base.merge(side[key + new_cols], on=key, how="left", validate="one_to_one")
    assert len(out) == 6400 and out[key].equals(base[key])
    for k in EMO6:
        out[f"int_pol_{k}"] = out["pol"] * out[f"emo_{k}"]
    # checks
    for c in base.columns:
        a, b = base[c], out[c]
        same = a.equals(b) or (a.dtype.kind == "f" and np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))
        assert same, f"column {c} changed"
    for k in EMO6:
        assert np.allclose(out[f"int_pol_{k}"], out["pol"] * out[f"emo_{k}"], rtol=0, atol=0, equal_nan=True)
    added = [c for c in out.columns if c not in base.columns]
    assert set(added) == set(new_cols) | {f"int_pol_{k}" for k in EMO6}
    nan = {c: int(out[c].isna().sum()) for c in added}
    assert all(out[c].notna().sum() >= 6390 for c in added), nan      # the primary's SD columns miss 2 coin-days
    for c in VALENCED["A82"][0]:
        assert (out[c].dropna() >= 0).all() and (out[c].dropna() <= 1).all(), c
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + ".tmp")
    out.to_parquet(tmp, index=False)
    back = pd.read_parquet(tmp)
    assert back.shape == out.shape and list(back.columns) == list(out.columns)
    os.replace(tmp, OUT)

    # variable specifications
    spec0 = pd.read_csv(SPEC0, keep_default_na=False)
    assert list(spec0.columns) == ["name", "table", "block", "scope", "har", "transform", "dlog", "csq", "impute_flag", "notes"]
    assert (spec0.block == "emotion").sum() == 12 and set(spec0.name) <= set(out.columns)

    def row(name: str, note: str) -> dict:
        return {"name": name, "table": "har_panel", "block": "emotion", "scope": "coin", "har": 1, "transform": "level",
                "dlog": 0, "csq": 0, "impute_flag": "", "notes": f"side arm of 12 Nov 2026: {note}"}

    specs = {}
    for arm, (col, note) in DISP.items():
        specs[arm] = pd.concat([spec0, pd.DataFrame([row(col, note)])], ignore_index=True)
        specs[f"{arm}_alone"] = pd.concat([spec0[spec0.block != "emotion"], pd.DataFrame([row(col, note)])], ignore_index=True)
    for arm, (cols, _lvl) in VALENCED.items():
        specs[arm] = pd.concat([spec0, pd.DataFrame([row(c, note_for(c)) for c in cols])], ignore_index=True)
    queue = []
    for arm, sp in specs.items():
        assert set(sp.name) <= set(out.columns) and not sp.name.duplicated().any(), arm
        f = CODE / "model" / f"variable_spec_{arm}.csv"
        t = f.with_name(f.name + ".tmp")
        sp.to_csv(t, index=False)
        back_sp = pd.read_csv(t, keep_default_na=False)
        assert list(back_sp.name) == list(sp.name) and list(back_sp.block) == list(sp.block), arm
        os.replace(t, f)
        queue.append({"run_id": arm, "label": f"side_{arm}", "spec": "A00",
                      "panel": str(OUT.relative_to(CODE)), "variable_spec": f"model/variable_spec_{arm}.csv",
                      "extra_args": "--no-insample" if arm.endswith("_alone") else "",
                      "emotion_terms": int(sum((3 if r.har else 1) * (2 if r.csq else 1) for r in sp[sp.block == "emotion"].itertuples()))})
    q = pd.DataFrame(queue)
    t = QUEUE.with_name(QUEUE.name + ".tmp")
    q.to_csv(t, index=False)
    os.replace(t, QUEUE)
    stats = {"step": "15b_sidearm_panels", "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "source_md5": md5(SRC), "side_table_md5": md5(SIDE), "output_md5": md5(OUT), "script_md5": md5(Path(__file__)),
             "rows": int(len(out)), "added_columns": added, "missing_per_added_column": nan,
             "arms": {a: {"emotion_terms": int(e)} for a, e in zip(q.run_id, q.emotion_terms)},
             "checks": ["6,400 rows, key unchanged", "every primary column unchanged", "added columns as listed",
                        "interaction columns equal the product of the stored columns", "specs name only panel columns"]}
    json.dump(stats, open(str(OUT) + ".stats.json", "w"), indent=1)
    print(json.dumps({"rows": len(out), "added": len(added), "arms": len(specs), "missing": nan}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
