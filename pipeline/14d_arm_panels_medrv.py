#!/usr/bin/env python3
"""
14d_arm_panels_medrv.py -- MedRV versions of the robustness-arm panels (D8 Deviation 7, 8 Nov 2026).

After Deviation 7 the primary ladder takes C and J from MedRV and the MedRQ ratio test (former arm A50). Every other
arm panel of 28 Oct 2026 (data/pipeline/panel/arms/final_20261028/<arm>.parquet) still carries the BV-based c and j
of the A00 panel. This step writes a copy of each arm panel in which c and j are replaced by c_med and j_med of the
A50 panel (matched on coin and day); every other column is left as it is, so each arm's own variable spec applies
unchanged. Skipped: A50 (now the primary) and A54 (HAR-MedRV benchmark; uses MedRV lags, no c or j).

Checks (asserted before anything is written): same rows and column order as the source arm panel; every column other
than c and j identical to the source; the source's c and j identical to the A00 panel's (the arm did not change them);
new c, j identical to A50's c_med, j_med; c + j = rv; c > 0; j >= 0.

Output: data/pipeline/panel/arms/final_medrv_20261108/<arm>.parquet (temp file -> checks -> replace), each with a
.stats.json, plus MANIFEST.csv (arm, source md5, output md5).
Usage (from code/): python pipeline/14d_arm_panels_medrv.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "pipeline" / "panel"
SRC = ROOT / "arms" / "final_20261028"
OUT = ROOT / "arms" / "final_medrv_20261108"
BASE = ROOT / "har_panel_final_20261028.parquet"
A50 = SRC / "A50.parquet"
SKIP = {"A50", "A54"}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    key = ["coin", "day"]
    base = pd.read_parquet(BASE).set_index(key)
    med = pd.read_parquet(A50).set_index(key)[["c_med", "j_med", "rv"]]
    assert np.allclose(med.c_med + med.j_med, med.rv, rtol=1e-12, atol=1e-12)
    rows = []
    for src in sorted(SRC.glob("*.parquet")):
        arm = src.stem
        if arm in SKIP:
            continue
        df = pd.read_parquet(src)
        assert {"c", "j", "rv", "coin", "day"} <= set(df.columns), arm
        idx = pd.MultiIndex.from_frame(df[key])
        assert idx.is_unique and len(df) == 6400, arm
        b = base.reindex(idx)
        assert np.array_equal(df.c.to_numpy(), b.c.to_numpy()) and np.array_equal(df.j.to_numpy(), b.j.to_numpy()), \
            f"{arm}: c/j differ from the A00 panel"
        m = med.reindex(idx)
        assert m.notna().all().all() and np.array_equal(df.rv.to_numpy(), m.rv.to_numpy()), arm
        new = df.copy()
        new["c"] = m.c_med.to_numpy()
        new["j"] = m.j_med.to_numpy()
        other = [c for c in df.columns if c not in ("c", "j")]
        assert list(new.columns) == list(df.columns) and new[other].equals(df[other]), arm
        assert np.allclose(new.c + new.j, new.rv, rtol=1e-12, atol=1e-12), arm
        assert (new.c > 0).all() and (new.j >= 0).all(), arm
        out = OUT / src.name
        tmp = out.with_suffix(".parquet.tmp")
        new.to_parquet(tmp, index=False)
        chk = pd.read_parquet(tmp)
        assert chk.shape == new.shape and np.array_equal(chk.c.to_numpy(), new.c.to_numpy())
        os.replace(tmp, out)
        stats = dict(step="14d_arm_panels_medrv", written_utc=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                     source=str(src.relative_to(ROOT.parents[1])), source_md5=md5(src), a50_md5=md5(A50),
                     replaced={"c": "c_med of A50", "j": "j_med of A50"}, rows=len(new), output_md5=md5(out),
                     checks=["rows and columns as source", "other columns identical", "source c/j = A00 c/j",
                             "c + j = rv", "c > 0", "j >= 0"])
        with open(str(out) + ".stats.json", "w") as fh:
            json.dump(stats, fh, indent=1)
        rows.append(dict(arm=arm, source_md5=stats["source_md5"], output_md5=stats["output_md5"]))
        print(f"{arm}: ok")
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "MANIFEST.csv", index=False)
    print(f"{len(man)} panels in {time.time() - t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
