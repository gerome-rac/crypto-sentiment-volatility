"""Confirmatory family with the simplified SQ4 test (D8 Deviation 7, 8 Nov 2026).

SQ4 is now one two-sided Driscoll-Kraay z-test of
    Delta = Cum(fear) - Cum(anger) = sum_k (beta_fear,k - beta_anger,k) = 0,   k in {1, 7, 30},
on the in-sample R3 OLS fit (raw probability units), instead of the intersection-union test of
Cum(fear) > 0 and Cum(fear) - Cum(anger) > 0. Delta, its Driscoll-Kraay standard error and Cum(fear) are read from
the run's h4_bootstrap.csv (row 'theta2', column dk_se); nothing is re-estimated. SQ1-SQ3 keep their Clark-West
p-values and SQ2 its three-condition rule from the run's confirmatory_holm.csv; Holm is re-applied to the four
p-values. Before anything is written, the stored Holm values are reproduced from the stored p-values.

    cd code && python -m model.confirmatory_sq4 <run dir> [<run dir> ...] --out <new output dir>

Writes <out>/confirmatory_sq4dk.csv (one block of four rows per run), config.json and MD5SUMS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from .run import md5

ALPHA = 0.05


def holm(p: np.ndarray) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def one_run(run_dir: str) -> pd.DataFrame:
    ch = pd.read_csv(os.path.join(run_dir, "confirmatory_holm.csv")).set_index("hypothesis")
    hb = pd.read_csv(os.path.join(run_dir, "h4_bootstrap.csv"))
    assert list(ch.index) == ["H1", "H2", "H3", "H4"] and len(hb) == 2
    # the stored Holm values must follow from the stored p-values
    assert np.allclose(holm(ch.p.to_numpy()), ch.p_holm.to_numpy(), atol=1e-12), run_dir
    t1 = hb[hb.condition.str.startswith("theta1")].iloc[0]
    t2 = hb[hb.condition.str.startswith("theta2")].iloc[0]
    assert t1.units == "raw" and t2.units == "raw"
    z = t2.estimate / t2.dk_se
    p4 = 2.0 * norm.sf(abs(z))
    p = np.r_[ch.loc[["H1", "H2", "H3"], "p"].to_numpy(float), p4]
    ph = holm(p)
    h2 = ch.loc["H2"]
    h2_rest = bool(h2.R1_in_MCS10_of_R1R2_both_losses) and bool(h2.DM_no_improvement_of_R2_both)
    verdict = [
        "supported" if ph[0] < ALPHA and not bool(ch.loc["H1", "qlike_points_against"]) else "not supported",
        "supported" if ph[1] >= ALPHA and h2_rest else "not supported",
        "supported" if ph[2] < ALPHA and not bool(ch.loc["H3", "qlike_points_against"]) else "not supported",
        "fear and anger differ" if ph[3] < ALPHA else "no significant difference",
    ]
    out = pd.DataFrame(dict(run=os.path.basename(os.path.normpath(run_dir)), subquestion=["SQ1", "SQ2", "SQ3", "SQ4"],
                            test=["Clark-West R0->R1", "Clark-West R1->R2 (+ MCS and DM conditions)", "Clark-West R2->R3",
                                  "two-sided DK z-test of Cum(fear) - Cum(anger) = 0"],
                            p=p, p_holm=ph, verdict=verdict))
    out["cum_fear"] = [np.nan] * 3 + [t1.estimate]
    out["cum_fear_dk_se"] = [np.nan] * 3 + [t1.dk_se]
    out["cum_anger"] = [np.nan] * 3 + [t1.estimate - t2.estimate]
    out["delta"] = [np.nan] * 3 + [t2.estimate]
    out["delta_dk_se"] = [np.nan] * 3 + [t2.dk_se]
    out["z"] = [np.nan] * 3 + [z]
    out["previous_p_sq4_iut"] = [np.nan] * 3 + [ch.loc["H4", "p"]]
    out["previous_verdict"] = ch.verdict.to_numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m model.confirmatory_sq4")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=False)
    df = pd.concat([one_run(r) for r in a.runs], ignore_index=True)
    path = os.path.join(a.out, "confirmatory_sq4dk.csv")
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    chk = pd.read_csv(tmp)
    assert chk.shape == df.shape and np.allclose(chk.p_holm, df.p_holm)
    os.replace(tmp, path)
    inputs = {os.path.basename(os.path.normpath(r)): {f: md5(os.path.join(r, f)) for f in ("confirmatory_holm.csv", "h4_bootstrap.csv")}
              for r in a.runs}
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(dict(method=__doc__.strip().splitlines()[0], alpha=ALPHA, inputs=inputs), fh, indent=1)
    with open(os.path.join(a.out, "MD5SUMS"), "w") as fh:
        for f in sorted(os.listdir(a.out)):
            if f != "MD5SUMS":
                fh.write(f"{md5(os.path.join(a.out, f))}  {f}\n")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(df[["run", "subquestion", "p", "p_holm", "verdict", "previous_verdict"]].to_string(index=False))
        print(df.loc[df.subquestion == "SQ4", ["run", "cum_fear", "cum_anger", "delta", "delta_dk_se", "z", "p", "previous_p_sq4_iut"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
