#!/usr/bin/env python3
"""Step 14c: smoke test of the arm panels with a frozen copy of the engine (no forecast or test number is read).

For every run of an arm run list (pipeline/14_arm_panels.py writes arm_runs_<stamp>.csv):
  (1) the briefed run, --max-days 3 --no-insample: the out-of-sample loop must finish all six rungs (R0-R3, HARRV,
      HARCJ). The evaluation step cannot run on 3 forecast days (arch's optimal block length inside the MCS fails
      for any panel, A00 included), so that one error signature is accepted;
  (2) --max-days 30 --no-insample --mcs-reps 200 must exit 0 (the evaluation path runs);
  (3) the design columns of every rung (column names of coefs_<rung>.parquet; values are not read) must equal the
      names implied by the arm's variable spec under the frozen engine's RUNG_BLOCKS.
Run in env harmodel:
  cd code && python pipeline/14c_arm_smoke.py --runs data/pipeline/panel/arms/arm_runs_20261028.csv \
      --engine-dir <frozen engine copy containing model/> --out-root <scratch dir outside model_runs> --out <csv>
The frozen copy must be made before the run (model/ is untracked, so `git archive` cannot produce it; copy
code/model to a temporary directory and keep its md5 list).
"""
import argparse, importlib.util, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parent.parent          # code/
RUNGS = ["R0", "R1", "R2", "R3", "HARRV", "HARCJ"]
MCS_SIG = "optimal_block_length"
ENGINE: Path = None
PY: str = sys.executable
cfg = None


def load_engine(engine_dir: Path) -> None:
    global ENGINE, cfg
    ENGINE = Path(engine_dir).resolve()
    assert (ENGINE / "model" / "run.py").exists(), f"{ENGINE} has no model/run.py"
    s = importlib.util.spec_from_file_location("frozen_cfg", ENGINE / "model" / "config.py")
    cfg = importlib.util.module_from_spec(s); sys.modules["frozen_cfg"] = cfg; s.loader.exec_module(cfg)


def expected_names(spec_df, rung):
    blocks = cfg.RUNG_BLOCKS[rung]
    out = []
    for r in spec_df.itertuples(index=False):
        if r.block not in blocks:
            continue
        out += [f"{r.name}_h{k}" for k in (1, 7, 30)] if r.har == "1" else [r.name]
        if r.dlog == "1":
            out.append(f"{r.name}_dlog")
        if r.csq == "1":
            out += [f"{r.name}_sq_h{k}" for k in (1, 7, 30)]
    return out


def run_engine(panel, spec_path, label, days, out_root, extra=()):
    cmd = [PY, "-m", "model.run", "--spec", "A00", "--rung", "all", "--panel", str(panel), "--variable-spec", str(spec_path),
           "--label", label, "--max-days", str(days), "--no-insample", "--out-root", str(out_root), "--n-jobs", "1", *extra]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ENGINE, env=dict(os.environ, PYTHONPATH=str(ENGINE)), capture_output=True, text=True)
    log = p.stdout + p.stderr
    done = re.findall(r"^\[(\w+)\] (\d+) forecasts", log, flags=re.M)
    rd = [l for l in p.stdout.strip().splitlines() if l.startswith(str(out_root))]
    return dict(exit=p.returncode, seconds=round(time.time() - t0, 1), rungs_done=[r for r, _ in done],
                n_forecasts={r: int(n) for r, n in done}, run_dir=rd[-1] if rd else "",
                err=(log.strip().splitlines() or [""])[-1][:200] if p.returncode else "", mcs_sig=MCS_SIG in log)


def one(row, out_root):
    rid, panel, spec_path = row["run_id"], row["panel"], row["spec"]
    sp = pd.read_csv(spec_path, dtype=str, keep_default_na=False)
    r3 = run_engine(panel, spec_path, f"{rid}_d3", 3, out_root / "d3")
    r30 = run_engine(panel, spec_path, f"{rid}_d30", 30, out_root / "d30", ("--mcs-reps", "200"))
    design = {}
    rd = Path(r30["run_dir"]) if r30["run_dir"] else None
    if rd is None:
        cands = sorted((out_root / "d30").glob(f"A00_{rid}_d30_*"))
        rd = cands[-1] if cands else None
    for rung in RUNGS:
        got = [c for c in pq.read_schema(rd / f"coefs_{rung}.parquet").names if c not in ("origin", "group")] if rd else []
        exp = expected_names(sp, rung)
        design[rung] = dict(match=got == exp, n=len(got), missing=[c for c in exp if c not in got][:8],
                            extra=[c for c in got if c not in exp][:8])
    ok3 = set(r3["rungs_done"]) == set(RUNGS) and (r3["exit"] == 0 or r3["mcs_sig"])
    return dict(run_id=rid, d3_exit=r3["exit"], d3_rungs_done=" ".join(r3["rungs_done"]), d3_forecasts_per_rung=r3["n_forecasts"].get("R0"),
                d3_known_mcs_error=r3["mcs_sig"], d3_ok=ok3, d3_seconds=r3["seconds"], d30_exit=r30["exit"], d30_seconds=r30["seconds"],
                d30_err=r30["err"], design_match_all=all(d["match"] for d in design.values()),
                design_cols=json.dumps({k: v["n"] for k, v in design.items()}),
                design_mismatch=json.dumps({k: v for k, v in design.items() if not v["match"]}))


def read_runs(path: str, only: str):
    df = pd.read_csv(path)
    if "panel" not in df.columns:                      # arm run list of step 14: paths relative to code/
        df["panel"] = [str(p if os.path.isabs(p) else CODE / p) for p in df.panel_path]
        df["spec"] = [str(p if os.path.isabs(p) else CODE / p) for p in df.variable_spec]
    if only:
        keep = only.split(",")
        df = df[df.run_id.isin(keep)]
        assert len(df) == len(keep), f"unknown run ids in --only: {sorted(set(keep) - set(df.run_id))}"
    for p in list(df.panel) + list(df.spec):
        assert os.path.exists(p), p
    return df[["run_id", "panel", "spec"]].to_dict("records")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="arm run list (run_id, panel_path, variable_spec) or CSV with run_id, panel, spec")
    ap.add_argument("--engine-dir", required=True, help="frozen engine copy (directory that contains model/)")
    ap.add_argument("--out-root", required=True, help="scratch directory for the engine runs (not under model_runs)")
    ap.add_argument("--out", required=True, help="result CSV (must not exist)")
    ap.add_argument("--only", default="", help="comma-separated run ids")
    ap.add_argument("--workers", type=int, default=3, help="parallel engine runs (each with --n-jobs 1); at most 4")
    a = ap.parse_args()
    assert 1 <= a.workers <= 4, "at most 4 worker processes"
    out_root = Path(a.out_root).resolve()
    assert "model_runs" not in out_root.parts, "--out-root must lie outside model_runs"
    assert not os.path.exists(a.out), f"{a.out} exists; not overwritten"
    load_engine(Path(a.engine_dir))
    runs = read_runs(a.runs, a.only)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(lambda r: one(r, out_root), runs))
    df = pd.DataFrame(res)
    tmp = a.out + ".tmp"
    df.to_csv(tmp, index=False)
    assert len(pd.read_csv(tmp)) == len(runs)
    os.replace(tmp, a.out)
    print(df[["run_id", "d3_ok", "d3_known_mcs_error", "d30_exit", "design_match_all", "d3_seconds", "d30_seconds"]].to_string())
    ok = bool(df.d3_ok.all() and (df.d30_exit == 0).all() and df.design_match_all.all())
    print(f"{int(df.d3_ok.sum())}/{len(df)} 3-day runs ok, {int((df.d30_exit == 0).sum())}/{len(df)} 30-day runs exit 0, "
          f"{int(df.design_match_all.sum())}/{len(df)} designs match the specs")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
