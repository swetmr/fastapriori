"""Measurement layer for the v0.2.0 experiment notebooks (10-19; Linux).

Protocol (same as bench/baseline_runners.py, which produced the v0.1.0.post1
grid): every run is a fresh process under `ulimit -v` + `timeout` +
`/usr/bin/time -v`; peak RSS comes from /usr/bin/time; in-process Python
methods report the library-call wall time (perf_counter inside the driver),
CLI methods the process wall time. Default: 1 warm-up + 3 measured runs,
median reported, as in the paper.

Results are appended row-by-row to `results/runs/<campaign>/<name>.csv`, so every
notebook can be interrupted and re-run: finished cells are skipped.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# [artifact patch] bundle layout: every harness module lives in bench/, the library
# source in lib/, results in results/runs/<campaign>/ (one folder per campaign, never mixed).
HARNESS_DIR = Path(__file__).resolve().parent
BENCH_DIR = HARNESS_DIR
BUNDLE_DIR = HARNESS_DIR.parent
REPO_DIR = Path(os.environ.get("FA_LIB_DIR", BUNDLE_DIR / "lib" / "fastapriori-0.2.0"))
WORK = Path(os.environ.get("FA_WORKDIR", BUNDLE_DIR / "_work")) / "harness"
CAMPAIGN = os.environ.get("FA_CAMPAIGN", "new_campaign")
RESULTS = BUNDLE_DIR / "results" / "runs" / CAMPAIGN
DRIVER = HARNESS_DIR / "exp_driver.py"
for p in (str(BENCH_DIR), str(HARNESS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import baseline_runners as br  # noqa: E402  (parse_time_output, df_to_spmf_file, ...)
import exp_data as pdd  # noqa: E402

TOOLS_BIN = br.TOOLS_BIN
DEFAULT_TIMEOUT_S = 1500
DEFAULT_MEMOUT_KB = int(float(os.environ.get("FA_MEMOUT_GB", "60")) * 1024 * 1024)  # [patch] ulimit -v; GiB via env

# LCM (Uno v5.3). The LCM driver of the v0.1.0.post1 runs was not archived;
# notebook 00 prints LCM's usage so this template can be checked against it.
# The bundled build, unless LCM_BIN explicitly names another: never whatever is first on PATH.
LCM_BIN = os.environ.get("LCM_BIN") or str(TOOLS_BIN / "lcm")
LCM_ARGS = os.environ.get("LCM_ARGS", "F -u {k} {input} {min_count}")

# ------------------------------------------------------------------ methods --
# kind: hyb | classic | pyfim | borgelt | lcm | hybcli
# needs: Rust functions that must exist in the installed extension.
METHODS: dict[str, dict] = {
    # HybridFIM as published (pairs v0, anchor-extend v3_adaptive)
    "hyb":             dict(kind="hyb"),
    "hyb_1t":          dict(kind="hyb", threads=1),
    # Suffix-only pair counting (+ singleton filtering), and its two halves
    "hyb_sfx":         dict(kind="hyb", bo={"pair_variant": "suffix_filtered"}, needs=["rust_compute_pairs_v2"]),
    "hyb_sfx_1t":      dict(kind="hyb", bo={"pair_variant": "suffix_filtered"}, threads=1, needs=["rust_compute_pairs_v2"]),
    "hyb_sfxonly_1t":  dict(kind="hyb", bo={"pair_variant": "suffix"}, threads=1, needs=["rust_compute_pairs_v2"]),
    "hyb_filt_1t":     dict(kind="hyb", bo={"pair_variant": "filtered"}, threads=1, needs=["rust_compute_pairs_v2"]),
    # k>=3 operators
    "hyb_eclat":       dict(kind="hyb", bo={"fast_variant": "eclat"}),
    "hyb_eclat_1t":    dict(kind="hyb", bo={"fast_variant": "eclat"}, threads=1),
    "hyb_seeded":      dict(kind="hyb", bo={"fast_variant": "eclat_seeded"}, needs=["rust_eclat_pipeline_seeded"]),
    "hyb_seeded_1t":   dict(kind="hyb", bo={"fast_variant": "eclat_seeded"}, threads=1, needs=["rust_eclat_pipeline_seeded"]),
    "hyb_prefilter":   dict(kind="hyb", bo={"impl_variant": "v5_prefilter"}),
    "classic":         dict(kind="classic"),
    # [artifact patch] candidate-intersection baseline pinned to one thread
    "classic_1t":      dict(kind="classic", threads=1),
    # External baselines
    "pyfim_apriori":   dict(kind="pyfim", algo="apriori"),
    "pyfim_eclat":     dict(kind="pyfim", algo="eclat"),
    "pyfim_fpgrowth":  dict(kind="pyfim", algo="fpgrowth"),
    "borgelt_apriori": dict(kind="borgelt", algo="apriori"),
    "borgelt_eclat":   dict(kind="borgelt", algo="eclat"),
    "borgelt_fpgrowth": dict(kind="borgelt", algo="fpgrowth"),
    # A1 check: same binaries without the -n upper bound
    "borgelt_apriori_nobound":  dict(kind="borgelt", algo="apriori", nobound=True),
    "borgelt_eclat_nobound":    dict(kind="borgelt", algo="eclat", nobound=True),
    "borgelt_fpgrowth_nobound": dict(kind="borgelt", algo="fpgrowth", nobound=True),
    "lcm":             dict(kind="lcm"),
    # Memory boundary: numpy-only process calling the Rust kernel directly
    "hybcli_load":     dict(kind="hybcli", load_only=True),
    "hybcli_pairs_v0": dict(kind="hybcli", fn="pairs_v0"),
    "hybcli_pairs_v2": dict(kind="hybcli", fn="pairs_v2", needs=["rust_compute_pairs_v2"]),
    "hybcli_anchor":   dict(kind="hybcli", fn="anchor"),
    "hybcli_seeded":   dict(kind="hybcli", fn="seeded", needs=["rust_eclat_pipeline_seeded"]),
    "hybcli_pairs_v0_1t": dict(kind="hybcli", fn="pairs_v0", threads=1),
    "hybcli_pairs_v2_1t": dict(kind="hybcli", fn="pairs_v2", threads=1, needs=["rust_compute_pairs_v2"]),
}

RESULT_COLUMNS = [
    "experiment", "dataset", "method", "k", "min_support", "run_id", "warmup",
    "status", "wall_s", "inner_wall_s", "process_wall_s", "user_cpu_s", "sys_cpu_s",
    "peak_rss_kb", "max_vsize_kb", "n_itemsets", "threads", "borgelt_read_s", "borgelt_mine_s",
    "borgelt_write_s", "lcm_counts", "stderr_tail", "build_id", "host", "timestamp_utc",
]


# ---------------------------------------------------------------- utilities --
def build_id() -> str:
    """Hash of the Rust sources and Python package: ties rows to a build."""
    h = hashlib.sha1()
    for p in sorted(list((REPO_DIR / "src").glob("*.rs")) + list((REPO_DIR / "fastapriori").rglob("*.py"))):
        h.update(p.read_bytes())
    # [artifact patch] also hash the installed extension binary and the harness itself, so a
    # stale build, a pip-installed wheel or a changed measurement layer gets a different id.
    try:
        h.update(Path(ext_path()).read_bytes())
    except ImportError:
        pass
    for p in (HARNESS_DIR / "exp_runners.py", HARNESS_DIR / "exp_driver.py", HARNESS_DIR / "baseline_runners.py"):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def ext_path() -> str:
    import importlib.util
    spec = importlib.util.find_spec("fastapriori._fastapriori_rs")
    if spec is None or spec.origin is None:
        raise ImportError("fastapriori Rust extension not installed (maturin develop --release)")
    return spec.origin


def missing_functions(methods: Iterable[str]) -> dict[str, list[str]]:
    import fastapriori._fastapriori_rs as rs
    out = {}
    for m in methods:
        miss = [f for f in METHODS[m].get("needs", []) if not hasattr(rs, f)]
        if miss:
            out[m] = miss
    return out


def preflight(methods: Iterable[str]) -> None:
    """Fail fast if the platform, tools or Rust build cannot run `methods`."""
    methods = list(methods)
    problems = []
    if not Path("/usr/bin/time").exists():
        problems.append("/usr/bin/time missing (apt install time); runs need Linux")
    for m in methods:
        spec = METHODS[m]
        if spec["kind"] == "borgelt" and not (TOOLS_BIN / spec["algo"]).exists():
            problems.append(f"{m}: Borgelt binary {TOOLS_BIN / spec['algo']} not found")
        if spec["kind"] == "lcm" and not Path(LCM_BIN).exists():
            problems.append(f"{m}: LCM binary not found (set LCM_BIN)")
        if spec["kind"] == "pyfim":
            try:
                import fim  # noqa: F401
            except ImportError:
                problems.append(f"{m}: pyfim not installed")
    for m, miss in missing_functions(m for m in methods if METHODS[m]["kind"] in ("hyb", "hybcli")).items():
        problems.append(f"{m}: rebuild the extension, missing {miss}")
    if problems:
        raise RuntimeError("preflight failed:\n  " + "\n  ".join(problems))
    print(f"preflight ok: {len(methods)} methods, build {build_id()}")


# -------------------------------------------------------------- input cache --
def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


_DF_CACHE: dict[str, pd.DataFrame] = {}


def dataset_df(name: str) -> pd.DataFrame:
    if name not in _DF_CACHE:
        _DF_CACHE[name] = pdd.load_df(name)
    return _DF_CACHE[name]


def parquet_for(name: str) -> Path:
    p = WORK / "parquet" / f"{_slug(name)}.parquet"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        dataset_df(name).to_parquet(p, index=False)
    return p


def npz_for(name: str) -> Path:
    """Encoded arrays exactly as the Python wrapper passes them to Rust."""
    p = WORK / "npz" / f"{_slug(name)}.npz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        df = dataset_df(name).dropna()
        items = sorted(df["item"].unique())
        enc = {it: i for i, it in enumerate(items)}
        tx = df["txn_id"]
        txn_ids = (tx.to_numpy(np.int64) if pd.api.types.is_integer_dtype(tx)
                   else pd.factorize(tx)[0].astype(np.int64))
        np.savez(p, txn_ids=txn_ids, item_ids=df["item"].map(enc).to_numpy(np.int32),
                 n_items=np.int64(len(items)))
    return p


def spmf_for(name: str) -> tuple[Path, int]:
    path, _ = br.df_to_spmf_file(name, dataset_df(name))
    n_txn_file = path.with_suffix(".ntxn")
    if not n_txn_file.exists():
        with path.open() as f:
            n_txn_file.write_text(str(sum(1 for _ in f)))
    return path, int(n_txn_file.read_text())


# ---------------------------------------------------------------- one run --
_BORGELT_COUNT = re.compile(r"\[(\d+)\s*set\(s\)\]")
_BORGELT_STEP = re.compile(r"^(?P<label>[^\n]*?)\s*\.\.\..*?done \[(?P<sec>[\d.]+)s\]", re.M)


def _borgelt_phases(text: str) -> dict:
    read = write = total = 0.0
    for m in _BORGELT_STEP.finditer(text):
        sec = float(m.group("sec"))
        total += sec
        label = m.group("label").strip().lower()
        if label.startswith("reading"):
            read += sec
        elif label.startswith("writing"):
            write += sec
    # [artifact fix] Borgelt's tools mine *during* their "writing" step (itemsets are
    # reported as they are found), so mining time is everything after reading. Rows recorded before
    # this fix (campaign vm_20261029 and earlier) hold total - reading - writing (only recoding and
    # sorting); for those, the mining time is borgelt_mine_s + borgelt_write_s.
    return dict(borgelt_read_s=read, borgelt_write_s=write, borgelt_mine_s=total - read)


def _status(returncode: int, stderr: str, timed_out: bool) -> str:
    if timed_out or returncode == 124:
        return "timeout"
    if returncode == 0:
        return "ok"     # [artifact patch] as bench/baseline_runners.py: stderr text never overrides rc 0
    low = stderr.lower()
    # [artifact patch] 134 (abort) and 139 (SIGSEGV) under `ulimit -v` are allocation
    # failures, as in bench/baseline_runners.py (_classify_exit).
    if returncode in (134, 137, 139, -6, -9, -11) or any(s in low for s in (
            "memoryerror", "cannot allocate", "bad_alloc", "out of memory",
            "memory allocation failed", "memory allocation of")):   # last: Rust's abort text
        return "oom"
    return "ok" if returncode == 0 else "error"


def run_once(method: str, dataset: str, k: int, min_support: float, *,
             timeout_s: int = DEFAULT_TIMEOUT_S, memout_kb: int = DEFAULT_MEMOUT_KB,
             threads: int | None = None) -> dict:
    """One fresh-process run. Returns a result row (without bookkeeping)."""
    spec = METHODS[method]
    kind = spec["kind"]
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="run_", dir=WORK))
    out_json, time_file = tmp / "out.json", tmp / "time.log"
    row: dict = {"lcm_counts": None}

    if kind in ("hyb", "classic", "pyfim", "hybcli"):
        dspec = dict(kind=kind, k=k, min_support=min_support, out_json=str(out_json),
                     bench_dir=str(BENCH_DIR), backend_options=spec.get("bo"),
                     algo=spec.get("algo"))
        if kind == "hybcli":
            dspec.update(npz=str(npz_for(dataset)), ext_path=ext_path(), fn=spec.get("fn"),
                         load_only=spec.get("load_only", False))
        else:
            dspec["parquet"] = str(parquet_for(dataset))
        (tmp / "spec.json").write_text(json.dumps(dspec))
        inner = [sys.executable, "-X", "utf8", "-u", str(DRIVER), str(tmp / "spec.json")]
    elif kind == "borgelt":
        spmf, _ = spmf_for(dataset)
        inner = [str(TOOLS_BIN / spec["algo"]), "-ts", f"-s{min_support * 100.0:.6f}", f"-m{k}"]
        if not spec.get("nobound"):
            inner.append(f"-n{k}")
        inner += [str(spmf), "/dev/null"]
    elif kind == "lcm":
        spmf, n_txn = spmf_for(dataset)
        mc = max(1, math.ceil(min_support * n_txn))
        inner = [LCM_BIN] + shlex.split(LCM_ARGS.format(k=k, input=str(spmf), min_count=mc))
    else:
        raise ValueError(kind)

    env = os.environ.copy()
    t = threads if threads is not None else spec.get("threads")
    if t is None and kind in ("hyb", "classic", "hybcli"):
        t = os.cpu_count()          # [patch] make the default thread count explicit and recorded
    if t is not None:
        env["RAYON_NUM_THREADS"] = str(t)
    shell = (f"ulimit -v {int(memout_kb)}; exec /usr/bin/time -v -o {shlex.quote(str(time_file))} "
             f"timeout {int(timeout_s)} " + " ".join(shlex.quote(x) for x in inner))
    timed_out = False
    try:
        proc = subprocess.run(["/bin/bash", "-c", shell], capture_output=True, text=True,
                              timeout=timeout_s + 60, env=env)
        rc, so, se = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        timed_out, rc = True, 124
        so, se = (e.stdout or ""), (e.stderr or "")
        so = so.decode() if isinstance(so, bytes) else so
        se = se.decode() if isinstance(se, bytes) else se
    tm = br.parse_time_output(time_file.read_text() if time_file.exists() else "")
    status = _status(rc, se, timed_out)
    # [artifact patch] Borgelt binaries can exit non-zero after a complete run; as in
    # bench/baseline_runners.py, a run that printed its final "[N set(s)] done" line is a success.
    if kind == "borgelt" and status == "error" and _BORGELT_COUNT.search(so + se) and "done" in (so + se):
        status = "ok"
    row.update(status=status, process_wall_s=tm.get("wall_s"), user_cpu_s=tm.get("user_cpu_s"),
               sys_cpu_s=tm.get("sys_cpu_s"), peak_rss_kb=tm.get("peak_rss_kb"),
               max_vsize_kb=tm.get("max_vsize_kb"),
               threads=t, stderr_tail=(se[-400:] if status != "ok" else None))

    row["inner_wall_s"] = np.nan
    row["n_itemsets"] = np.nan
    if status == "ok":
        if kind in ("hyb", "classic", "pyfim", "hybcli"):
            try:
                res = json.loads(out_json.read_text())
                row["inner_wall_s"] = res.get("inner_wall_s", np.nan)
                row["n_itemsets"] = res.get("n_itemsets", np.nan)
            except Exception as exc:
                row.update(status="error", stderr_tail=f"no driver output: {exc}; {se[-300:]}")
        elif kind == "borgelt":
            text = so + se
            m = _BORGELT_COUNT.search(text)
            row["n_itemsets"] = int(m.group(1)) if m else np.nan
            row.update(_borgelt_phases(text))
        elif kind == "lcm":
            # [artifact fix] the total can overflow 64 bits and print negative; it must
            # still be matched, or the per-size counts shift by one (Kosarak, k=2).
            ints = [int(x) for x in re.findall(r"^\s*(-?\d+)\s*$", so, re.M)]
            row["lcm_counts"] = json.dumps(ints)
            # LCM prints the total, then the count per itemset size 0,1,2,...
            row["n_itemsets"] = ints[1 + k] if len(ints) > 1 + k else np.nan
    in_process = kind in ("hyb", "classic", "pyfim", "hybcli")
    row["wall_s"] = row["inner_wall_s"] if in_process else row["process_wall_s"]
    shutil.rmtree(tmp, ignore_errors=True)
    return row


# ----------------------------------------------------------------- grids --
def results_path(experiment: str) -> Path:
    return RESULTS / f"{experiment}.csv"


def load_results(experiment: str) -> pd.DataFrame:
    p = results_path(experiment)
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=RESULT_COLUMNS)


def _append(experiment: str, row: dict) -> None:
    p = results_path(experiment)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).reindex(columns=RESULT_COLUMNS).to_csv(
        p, mode="a", header=not p.exists(), index=False)


def _cell_state(done: pd.DataFrame, dataset, method, k, s) -> tuple[set, str | None]:
    if len(done) == 0:
        return set(), None
    m = ((done.dataset == dataset) & (done.method == method) & (done.k == k)
         & (np.isclose(done.min_support.astype(float), float(s))))
    sub = done[m]
    good = set(sub[sub.status == "ok"].run_id.astype(int))
    # a failure counts only while no later success holds the same run id
    bad = sub[(sub.status != "ok") & ~sub.run_id.astype(int).isin(good)]
    return good, (bad.status.iloc[-1] if len(bad) else None)


def run_grid(experiment: str, datasets: Iterable[str], methods: Iterable[str],
             ks: Iterable[int], supports: Iterable[float], *, repeats: int = 3,
             warmup: int = 1, timeout_s: int = DEFAULT_TIMEOUT_S,
             memout_kb: int = DEFAULT_MEMOUT_KB, cascade: bool = True,
             retry_failed: bool = False, max_wall_for_repeats: float | None = None,
             verbose: bool = True) -> pd.DataFrame:
    """Run every (dataset, method, support, k) cell, resumable.

    cascade: once a (dataset, method, support) cell times out / OOMs at some
    k, larger k are recorded as 'skipped' rather than run.
    max_wall_for_repeats: if the warm-up takes longer than this, only one
    measured run is taken (big cells; the probe already shows the cost).
    """
    methods, ks, supports = list(methods), sorted(ks), list(supports)
    preflight(methods)
    bid, host = build_id(), os.uname().nodename if hasattr(os, "uname") else "?"
    datasets = list(datasets)
    write_manifest(experiment, dict(datasets=datasets, methods=methods, ks=ks, supports=supports,
                                    repeats=repeats, warmup=warmup, timeout_s=timeout_s,
                                    memout_kb=memout_kb, cascade=cascade,
                                    max_wall_for_repeats=max_wall_for_repeats), bid)
    done = load_results(experiment)
    # [artifact patch] never resume a campaign on a different build: medians would mix builds.
    other = sorted(set(done.get("build_id", pd.Series(dtype=str)).dropna().astype(str)) - {bid})
    if other and os.environ.get("FA_ALLOW_MIXED_BUILDS") != "1":
        raise RuntimeError(f"{results_path(experiment)} already holds rows from build(s) {other}, "
                           f"current build is {bid}. Start a new FA_CAMPAIGN (or set "
                           f"FA_ALLOW_MIXED_BUILDS=1 if you really mean to mix).")
    # [artifact patch] overall progress bar over (dataset, method, support, k) cells, with ETA;
    # cells already in the results file count as done immediately. Display only, no effect on timing.
    bar = None
    if verbose:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(datasets) * len(methods) * len(supports) * len(ks), desc=experiment,
                   unit="cell", dynamic_ncols=True)
    for ds in datasets:
        _DF_CACHE.clear()           # [patch] keep only the current dataset in the parent process
        for method in methods:
            for s in supports:
                blocked = None
                for k in ks:
                    if bar is not None:
                        bar.set_postfix_str(f"{ds} | {method} | k={k} s={s:g}", refresh=True)
                    try:
                        have, failed = _cell_state(done, ds, method, k, s)
                        if failed and not retry_failed:
                            blocked = failed if cascade and failed in ("timeout", "oom", "skipped") else blocked
                            continue
                        base = dict(experiment=experiment, dataset=ds, method=method, k=k,
                                    min_support=s, build_id=bid, host=host)
                        if blocked:
                            _append(experiment, dict(base, run_id=0, warmup=True, status="skipped",
                                                     stderr_tail=f"after {blocked} at smaller k",
                                                     timestamp_utc=datetime.now(timezone.utc).isoformat()))
                            continue
                        n_runs, run_id = warmup + repeats, 0
                        while run_id < n_runs:
                            if run_id in have:
                                run_id += 1
                                continue
                            t0 = time.time()
                            r = run_once(method, ds, k, s, timeout_s=timeout_s, memout_kb=memout_kb)
                            r.update(base, run_id=run_id, warmup=run_id < warmup,
                                     timestamp_utc=datetime.now(timezone.utc).isoformat())
                            _append(experiment, r)
                            if verbose:
                                bar.write(f"[{experiment}] {ds} | {method} | k={k} s={s:g} run {run_id}: "
                                          f"{r['status']} wall={r['wall_s']} rss={r['peak_rss_kb']} "
                                          f"({time.time() - t0:.1f}s)")
                            if r["status"] != "ok":
                                if cascade and r["status"] in ("timeout", "oom"):
                                    blocked = r["status"]
                                break
                            big = (max_wall_for_repeats is not None
                                   and (r["process_wall_s"] or 0) > max_wall_for_repeats)
                            if run_id == 0 and big:
                                n_runs = min(n_runs, warmup + 1)
                            run_id += 1
                    finally:
                        if bar is not None:
                            bar.update(1)
    if bar is not None:
        bar.close()
    return load_results(experiment)


def write_manifest(experiment: str, grid: dict, bid: str) -> None:
    """[artifact patch] Append one JSON line per run_grid call to RESULTS/manifest.jsonl:
    grid, build id, extension, library version, tool hashes and host fingerprint."""
    import importlib.metadata as md
    RESULTS.mkdir(parents=True, exist_ok=True)
    tools = {}
    for p in sorted(TOOLS_BIN.glob("*")) + ([Path(LCM_BIN)] if Path(LCM_BIN).exists() else []):
        if p.is_file():
            tools[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        version = md.version("fastapriori")
    except md.PackageNotFoundError:
        version = None
    ext_sha = hashlib.sha256(Path(ext_path()).read_bytes()).hexdigest()
    WORK.mkdir(parents=True, exist_ok=True)
    fs = subprocess.run(["stat", "-f", "-c", "%T", str(WORK)], capture_output=True, text=True).stdout.strip()
    if fs in ("nfs", "cifs", "smb2", "fuseblk", "fuse", "9p", "v9fs", "drvfs"):
        print(f"WARNING: work dir {WORK} is on '{fs}'. CLI baselines read their input from it inside "
              f"the timed region; set FA_WORKDIR to local disk for comparable timings.", flush=True)
    rec = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), campaign=CAMPAIGN,
               workdir=str(WORK), workdir_fs=fs,
               experiment=experiment, build_id=bid, fastapriori_version=version,
               extension=ext_path(), extension_sha256=ext_sha, tools_sha256=tools, lcm_args=LCM_ARGS,
               environment=br.collect_environment(), grid=grid)
    with (RESULTS / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Median over measured (non-warm-up) ok runs per cell."""
    if len(df) == 0:
        return df
    ok = df[(df.status == "ok") & (~df.warmup.astype(bool))]
    g = ok.groupby(["dataset", "method", "k", "min_support"])
    out = g.agg(wall_s=("wall_s", "median"), peak_rss_kb=("peak_rss_kb", "median"),
                n_itemsets=("n_itemsets", "first"), n_runs=("wall_s", "size"),
                process_wall_s=("process_wall_s", "median"),
                borgelt_read_s=("borgelt_read_s", "median"),
                borgelt_mine_s=("borgelt_mine_s", "median")).reset_index()
    # [artifact patch] report a failure only for cells without any successful measurement
    # (a cell retried with retry_failed=True keeps its old failure rows in the CSV).
    key = ["dataset", "method", "k", "min_support"]
    failed = df[df.status != "ok"].set_index(key)
    failed = failed[~failed.index.isin(out.set_index(key).index)].reset_index()
    fails = failed.groupby(key).status.first()
    out = out.merge(fails.rename("failure").reset_index(), how="outer",
                    on=["dataset", "method", "k", "min_support"])
    return out


def check_agreement(summary: pd.DataFrame, reference: str = "hyb") -> pd.DataFrame:
    """Cells where a method's size-k itemset count disagrees with `reference`.
    (LCM and *_nobound variants report other levels and are excluded.)"""
    s = summary[~summary.method.str.contains("nobound|lcm|hybcli_load", regex=True)]
    piv = s.pivot_table(index=["dataset", "k", "min_support"], columns="method",
                        values="n_itemsets", aggfunc="first")
    if reference not in piv:
        return pd.DataFrame()
    bad = piv.apply(lambda r: r.dropna().nunique() > 1, axis=1)
    return piv[bad]
