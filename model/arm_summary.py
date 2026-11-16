"""Summary of the robustness-arm runs: per run the three Clark-West steps, the SQ4 test, Holm within the run and
Holm within each arm family (8 Nov 2026, for the MedRV re-runs of D8 Deviation 7).

Reads only stored run outputs (nested_tests.csv, confirmatory_holm.csv, h4_bootstrap.csv). Rules as in the summary of
28 Oct 2026 (`_arm_results_final_20261028.csv`): Holm across the run's own confirmatory p-values (four, or three if
the run has no fear/anger test); Holm within each family and hypothesis across its runs on the Clark-West p-values,
none for runs whose arm the plan note assigns to no family; verdicts as in the confirmatory rules (SQ1/SQ3: Holm p
< 0.05 and no QLIKE reversal; SQ2: Holm p >= 0.05, R1 in the 90% MCS of {R1, R2} on both losses and no DM gain).
SQ4: --sq4 iut keeps each run's stored intersection-union p (as on 28 Oct); --sq4 dk2 uses the two-sided
Driscoll-Kraay test of Cum(fear) - Cum(anger) = 0 from h4_bootstrap.csv (Deviation 7).

    cd code && python -m model.arm_summary --runs <csv: arm_id,run_id,family,run_dir> --sq4 dk2 \
        --primary <run dir> --out <csv> [--check-against <28 Oct summary csv>]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from .confirmatory_sq4 import holm

RUNS = os.path.join(os.path.dirname(__file__), "..", "data", "pipeline", "model_runs")
HYP = ["H1", "H2", "H3"]


def _bool(x):
    return isinstance(x, (bool, np.bool_)) and bool(x) or (isinstance(x, str) and x.strip().lower() == "true")


def run_row(run_dir: str, sq4: str) -> dict:
    nt = pd.read_csv(os.path.join(run_dir, "nested_tests.csv")).set_index("test")
    ch = pd.read_csv(os.path.join(run_dir, "confirmatory_holm.csv")).set_index("hypothesis")
    row = {}
    for h in HYP:
        row.update({f"{h}_cw_p": nt.loc[h, "cw_p"], f"{h}_dm_mse_p_big_better": nt.loc[h, "dm_mse_p_big_better"],
                    f"{h}_dm_qlike_p_big_better": nt.loc[h, "dm_qlike_p_big_better"],
                    f"{h}_r2_oos": nt.loc[h, "r2_oos_big_vs_small"]})
    has_h4 = pd.notna(ch.loc["H4", "p"])
    p4 = np.nan
    if has_h4 and sq4 == "iut":
        p4 = ch.loc["H4", "p"]
    elif has_h4 and sq4 == "dk2":
        hb = pd.read_csv(os.path.join(run_dir, "h4_bootstrap.csv"))
        t2 = hb[hb.condition.str.startswith("theta2")].iloc[0]
        p4 = 2.0 * norm.sf(abs(t2.estimate / t2.dk_se))
        row["H4_delta"], row["H4_delta_dk_se"] = t2.estimate, t2.dk_se
    row["H4_p"] = p4
    ps = [row[f"{h}_cw_p"] for h in HYP] + ([p4] if has_h4 else [])
    ph = holm(np.asarray(ps, float))
    for h, v in zip(HYP + (["H4"] if has_h4 else []), ph):
        row[f"{h}_p_holm_own"] = v
    qa = {h: _bool(ch.loc[h, "qlike_points_against"]) for h in ("H1", "H3")}
    h2_rest = _bool(ch.loc["H2", "R1_in_MCS10_of_R1R2_both_losses"]) and _bool(ch.loc["H2", "DM_no_improvement_of_R2_both"])
    row["H1_verdict_own_holm"] = "supported" if row["H1_p_holm_own"] < 0.05 and not qa["H1"] else "not supported"
    row["H2_verdict_own_holm"] = "supported" if row["H2_p_holm_own"] >= 0.05 and h2_rest else "not supported"
    row["H3_verdict_own_holm"] = "supported" if row["H3_p_holm_own"] < 0.05 and not qa["H3"] else "not supported"
    if not has_h4:
        row["H4_verdict_own_holm"] = ch.loc["H4", "verdict"]
    elif sq4 == "iut":
        row["H4_verdict_own_holm"] = "supported" if row["H4_p_holm_own"] < 0.05 else "not supported"
    else:
        row["H4_verdict_own_holm"] = "fear and anger differ" if row["H4_p_holm_own"] < 0.05 else "no significant difference"
    return row


def summarise(runs: pd.DataFrame, sq4: str, primary: str) -> pd.DataFrame:
    rows = []
    for r in runs.itertuples():
        d = r.run_dir if os.path.isabs(r.run_dir) else os.path.join(RUNS, r.run_dir)
        rows.append(dict(arm_id=r.arm_id, run_id=r.run_id, family=r.family, run_dir=os.path.basename(os.path.normpath(d)),
                         **run_row(d, sq4)))
    df = pd.DataFrame(rows)
    for h in HYP:
        df[f"{h}_cw_p_holm_family"] = np.nan
        df[f"{h}_family_size"] = df.groupby("family")["run_id"].transform("size")
        for fam, g in df.groupby("family"):
            if str(fam).startswith("none"):
                continue
            df.loc[g.index, f"{h}_cw_p_holm_family"] = holm(g[f"{h}_cw_p"].to_numpy(float))
    prim = run_row(primary, sq4)
    for h in HYP + ["H4"]:
        df[f"{h}_same_verdict_as_primary"] = df[f"{h}_verdict_own_holm"] == prim[f"{h}_verdict_own_holm"]
    df.attrs["primary"] = prim
    return df


def _fp(p):
    return "" if pd.isna(p) else ("<0.0001" if p < 0.0001 else f"{p:.4f}")


def markdown(df: pd.DataFrame, title: str, intro: str) -> str:
    prim = df.attrs["primary"]
    out = [f"# {title}", "", intro, "", "## Counts", ""]
    fam = df[~df.family.astype(str).str.startswith("none")]
    for h, name in (("H1", "attention, R0→R1"), ("H2", "polarity, R1→R2"), ("H3", "emotions, R2→R3")):
        raw = int((df[f"{h}_cw_p"] < 0.05).sum())
        adj = fam[fam[f"{h}_cw_p_holm_family"] < 0.05]
        out.append(f"- {h} ({name}): raw CW p < 0.05 in {raw} of {len(df)} runs; Holm-adjusted (family) p < 0.05 in "
                   f"{len(adj)} of {len(fam)} family runs" + (f": {', '.join(adj.run_id)}." if 0 < len(adj) < 12 else "."))
    h4 = df[df.H4_p.notna()]
    out.append(f"- H4 (SQ4 test) p < 0.05 in {int((h4.H4_p < 0.05).sum())} of {len(h4)} runs with a fear/anger test"
               + (f": {', '.join(h4[h4.H4_p < 0.05].run_id)}." if (h4.H4_p < 0.05).any() else "."))
    for h in ("H1", "H2", "H3", "H4"):
        n = int(df[f"{h}_same_verdict_as_primary"].sum())
        out.append(f"- {h}: own-Holm verdict equals the primary's ({prim[f'{h}_verdict_own_holm']}) in {n} of {len(df)} runs.")
    out += ["", "## By family", ""]
    cols = ["run", "H1 CW p", "H1 Holm (family)", "H1 OOS R²", "H2 CW p", "H2 Holm (family)", "H3 CW p",
            "H3 Holm (family)", "H3 OOS R²", "H4 p"]
    for f, g in df.groupby("family", sort=False):
        out += [f"### {f} ({len(g)} runs)", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in g.sort_values("run_id").itertuples():
            out.append(f"| {r.run_id} | {_fp(r.H1_cw_p)} | {_fp(r.H1_cw_p_holm_family)} | {100 * r.H1_r2_oos:+.2f}% | "
                       f"{_fp(r.H2_cw_p)} | {_fp(r.H2_cw_p_holm_family)} | {_fp(r.H3_cw_p)} | {_fp(r.H3_cw_p_holm_family)} | "
                       f"{100 * r.H3_r2_oos:+.2f}% | {_fp(r.H4_p)} |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m model.arm_summary")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--sq4", choices=["iut", "dk2"], required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-against", default=None)
    ap.add_argument("--md", default=None, help="also write a markdown summary to this path")
    ap.add_argument("--title", default="Robustness arms")
    ap.add_argument("--intro", default="")
    a = ap.parse_args(argv)
    runs = pd.read_csv(a.runs)
    df = summarise(runs, a.sq4, a.primary)
    if a.check_against:
        ref = pd.read_csv(a.check_against).set_index("run_id")
        got = df.set_index("run_id").loc[ref.index]
        num = [f"{h}_{c}" for h in HYP for c in ("cw_p", "p_holm_own", "cw_p_holm_family", "r2_oos")] + ["H4_p"]
        for c in num:
            assert np.allclose(got[c].to_numpy(float), ref[c].to_numpy(float), atol=1e-10, equal_nan=True), c
        for h in HYP:
            assert (got[f"{h}_verdict_own_holm"] == ref[f"{h}_verdict_own_holm"]).all(), h
            assert (got[f"{h}_same_verdict_as_primary"] == ref[f"{h}_same_verdict_as_A00"]).all(), h
        print(f"reproduces {a.check_against} ({len(ref)} runs)")
    tmp = a.out + ".tmp"
    df.to_csv(tmp, index=False)
    assert pd.read_csv(tmp).shape == df.shape
    os.replace(tmp, a.out)
    if a.md:
        text = markdown(df, a.title, a.intro)
        tmp = a.md + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        os.replace(tmp, a.md)
    print(f"wrote {a.out} ({len(df)} runs); primary verdicts:",
          {h: df.attrs["primary"][f"{h}_verdict_own_holm"] for h in HYP + ["H4"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
