"""Safe execution rules shared by every pipeline step.

Contract (decided 22-23 Sep 2026; StockTwits map PD7 and PD9, Twitter map D14/D17):

1. DuckDB runs with TimeZone=UTC (asserted), a memory limit, a thread limit and
   a spill directory, set in one place (``connect``).
2. A step writes its output to a temporary name next to the final one
   (``<dir>/_tmp/<name>``), never straight into place.
3. It runs its conservation checks against the temporary output.
4. Only when every check passes is the output moved into place (``commit``);
   a failed check raises ``CheckError`` and leaves the previous output untouched.
5. A stats file next to the output records the inputs (path, bytes, md5), the
   code and rules fingerprints, the settings, the counts and every check.
6. A step whose stats file matches its current inputs, code and output is
   skipped (``is_current``), so re-running the pipeline repeats no work.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import duckdb

from . import config

PathLike = Union[str, Path]


class CheckError(RuntimeError):
    """A conservation or consistency check failed; the output was not committed."""


def log(msg: str) -> None:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", flush=True)


def check(condition: bool, message: str, results: Optional[List[Dict[str, object]]] = None) -> None:
    """Record a check and stop the step if it fails."""
    if results is not None:
        results.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise CheckError(message)


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------

def connect(memory: Optional[str] = None, threads: Optional[int] = None,
            spill: Optional[PathLike] = None, database: Optional[PathLike] = None) -> duckdb.DuckDBPyConnection:
    """DuckDB with the pipeline settings; TimeZone is asserted. In memory by default; with `database`, a
    scratch database file, so that large intermediate tables can be moved to disk (the Twitter units of step 2)."""
    spill_dir = Path(spill or config.SPILL_DIR)
    spill_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database)) if database else duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit='{memory or config.DUCKDB_MEMORY}'")
    con.execute(f"SET threads={int(threads or config.DUCKDB_THREADS)}")
    con.execute(f"SET temp_directory='{spill_dir}'")
    tz = con.execute("SELECT current_setting('TimeZone')").fetchone()[0]
    if tz != "UTC":
        raise CheckError(f"DuckDB TimeZone is {tz!r}, expected 'UTC'")
    return con


def settings(con: duckdb.DuckDBPyConnection) -> Dict[str, str]:
    names = ["TimeZone", "memory_limit", "threads", "temp_directory"]
    return {n: str(con.execute(f"SELECT current_setting('{n}')").fetchone()[0]) for n in names} | {
        "duckdb_version": duckdb.__version__}


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def file_md5(path: PathLike, block: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_inputs(paths: Iterable[PathLike], with_md5: bool = True) -> List[Dict[str, object]]:
    """Name, bytes and (optionally) md5 of every input file, in a stable order."""
    out = []
    for p in sorted(str(x) for x in paths):
        rec: Dict[str, object] = {"path": p, "bytes": os.path.getsize(p)}
        if with_md5:
            rec["md5"] = file_md5(p)
        out.append(rec)
    return out


def code_fingerprint(*extra_files: PathLike) -> str:
    """sha256 over the calling step's own source and the settings that shape the data.

    The settings are the study window, the coins and the platforms; paths, memory
    and thread limits do not change a table and are left out, as are this module's
    checks. The rules a step applies are fingerprinted separately (rules.section_sha).
    """
    h = hashlib.sha256()
    h.update(repr((config.WINDOW_START, config.WINDOW_END, config.N_DAYS, config.COINS, config.PLATFORMS)).encode())
    for f in [Path(f) for f in extra_files]:
        if f.exists():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]




# ---------------------------------------------------------------------------
# Staged writes
# ---------------------------------------------------------------------------

def tmp_path(final: PathLike) -> Path:
    """Temporary location for an output: <dir>/_tmp/<name>."""
    final = Path(final)
    t = final.parent / "_tmp" / final.name
    t.parent.mkdir(parents=True, exist_ok=True)
    if t.is_dir():
        shutil.rmtree(t)
    elif t.exists():
        t.unlink()
    return t


def commit(tmp: PathLike, final: PathLike) -> None:
    """Move a checked temporary output into place (file or directory)."""
    tmp, final = Path(tmp), Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.is_dir() and tmp.is_dir():
        old = final.with_name(final.name + ".old")
        if old.exists():
            shutil.rmtree(old)
        os.replace(final, old)
        os.replace(tmp, final)
        shutil.rmtree(old)
    else:
        os.replace(tmp, final)
    tmpdir = tmp.parent
    if tmpdir.name == "_tmp" and tmpdir.exists() and not any(tmpdir.iterdir()):
        tmpdir.rmdir()


def write_json_atomic(path: PathLike, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = path.with_name(path.name + ".tmp")
    with open(t, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(t, path)


def stats_path(final: PathLike) -> Path:
    final = Path(final)
    return final.with_name(final.name + ".stats.json") if not final.is_dir() else final / "_stats.json"


def write_stats(final: PathLike, *, step: str, inputs: Sequence[Dict[str, object]], code_sha: str,
                rules_sha: str, con: Optional[duckdb.DuckDBPyConnection], counts: Dict[str, object],
                checks: List[Dict[str, object]], extra: Optional[Dict[str, object]] = None) -> None:
    final = Path(final)
    outputs = []
    if final.is_dir():
        for f in sorted(final.rglob("*.parquet")):
            outputs.append({"path": str(f), "bytes": f.stat().st_size, "md5": file_md5(f)})
    else:
        outputs.append({"path": str(final), "bytes": final.stat().st_size, "md5": file_md5(final)})
    rec = {
        "step": step,
        "written_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "code_sha": code_sha,
        "rules_sha": rules_sha,
        "settings": settings(con) if con is not None else {},
        "inputs": list(inputs),
        "outputs": outputs,
        "counts": counts,
        "checks": checks,
    }
    if extra:
        rec.update(extra)
    write_json_atomic(stats_path(final), rec)


def is_current(final: PathLike, inputs: Sequence[Dict[str, object]], code_sha: str, rules_sha: str) -> bool:
    """True if the output exists and its stats match the current inputs, code and file contents."""
    final = Path(final)
    sp = stats_path(final)
    if not final.exists() or not sp.exists():
        return False
    try:
        rec = json.loads(sp.read_text())
    except (OSError, ValueError):
        return False
    if rec.get("code_sha") != code_sha or rec.get("rules_sha") != rules_sha:
        return False
    if [(i["path"], i["bytes"], i.get("md5")) for i in rec.get("inputs", [])] != \
       [(i["path"], i["bytes"], i.get("md5")) for i in inputs]:
        return False
    for o in rec.get("outputs", []):
        p = Path(o["path"])
        if not p.exists() or p.stat().st_size != o["bytes"] or file_md5(p) != o["md5"]:
            return False
    return all(c.get("passed") for c in rec.get("checks", []))
