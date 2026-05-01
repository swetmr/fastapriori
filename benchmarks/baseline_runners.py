"""Subprocess-based benchmark runners for fastapriori + Borgelt C + SPMF Java + mlxtend + efficient-apriori.

Every method runs in a fresh subprocess wrapped by /usr/bin/time -v so that peak
RSS is kernel-reported and comparable across languages. The notebook
(rigorous_baseline_benchmark.ipynb) drives the config loop and calls into this
module.

Design rules (from plan stateless-soaring-reef.md):
  - Fresh subprocess for every run (no interpreter reuse).
  - /usr/bin/time -v is the single source of truth for wall time + peak RSS.
  - ulimit -v + timeout enforce memory/time caps without hanging the host.
  - Append-only CSV lets the notebook resume after Ctrl+C or GCP preemption.
  - Driver scripts live as string constants; they're written on demand to a
    workdir so nothing stateful leaks between runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

BENCHMARKS_DIR = Path(__file__).parent
TOOLS_DIR = BENCHMARKS_DIR / "tools"
TOOLS_BIN = TOOLS_DIR / "bin"
TOOLS_SRC = TOOLS_DIR / "src"
SPMF_JAR = TOOLS_DIR / "spmf.jar"
WORKDIR = BENCHMARKS_DIR / "_bench_workdir"
DATA_CACHE = WORKDIR / "spmf_cache"
DRIVERS_DIR = WORKDIR / "drivers"

RAW_CSV = BENCHMARKS_DIR / "results_rigorous_raw.csv"
AGG_CSV = BENCHMARKS_DIR / "results_rigorous.csv"
EXCLUDED_MD = BENCHMARKS_DIR / "rigorous_excluded.md"


def set_raw_csv_suffix(label: str, prefix: str = "results_rigorous") -> tuple[Path, Path]:
    """Rebind the module-level RAW_CSV/AGG_CSV paths so each notebook session
    writes to its own files. Call once at notebook startup BEFORE any
    run/append/load.

    Example:
        br.set_raw_csv_suffix("20260418_1530")
        -> RAW_CSV = benchmarks/results_rigorous_raw_20260418_1530.csv
        -> AGG_CSV = benchmarks/results_rigorous_20260418_1530.csv

    The bare prefix can also be overridden, which is how the thread-scaling
    notebook keeps its artifacts separate from the main benchmark:
        br.set_raw_csv_suffix("20260418_1530", prefix="results_thread_scaling")
    """
    global RAW_CSV, AGG_CSV
    suffix = f"_{label}" if label else ""
    RAW_CSV = BENCHMARKS_DIR / f"{prefix}_raw{suffix}.csv"
    AGG_CSV = BENCHMARKS_DIR / f"{prefix}{suffix}.csv"
    return RAW_CSV, AGG_CSV

ALL_METHODS = [
    # Two explicit fast-path variants for A/B comparison. Gated by
    # backend_options={"fast_variant": ...}; "eclat" currently raises
    # NotImplementedError until the Rust vertical-recursion port lands.
    # Plan is to collapse these once Eclat wins empirically; until then
    # both variants appear as distinct rows / bars in the rigorous plots.
    "fastapriori_fast_apriori",
    "fastapriori_fast_eclat",
    "fastapriori_fast",
    # Same Rust binary as fastapriori_fast, pinned to RAYON_NUM_THREADS=1.
    # Used at k=2 only, for like-for-like comparison against the intrinsically
    # serial C/Python baselines (Borgelt, efficient-apriori, pyfim). Both
    # rows appear in the raw CSV so plots can show the parallelism headroom
    # as a dashed/solid pair of the "same method in two configurations".
    "fastapriori_fast_1t",
    "fastapriori_classic",
    "efficient_apriori",
    "borgelt_apriori",
    "borgelt_fpgrowth",
    "borgelt_eclat",
    # PyFIM = Borgelt's C algorithms exposed as a CPython extension (pip install pyfim).
    # Same C hot loop as borgelt_* above but crosses a Python FFI boundary like
    # fastapriori does, which makes it the closest like-for-like comparator for
    # us: both sides pay the interpreter-boundary cost, so any remaining gap is
    # pure algorithm + implementation rather than subprocess overhead.
    "pyfim_apriori",
    "pyfim_fpgrowth",
    "pyfim_eclat",
    # pyECLAT (pip install pyECLAT): pure-Python Eclat implementation. Slow
    # and memory-hungry but pip-installable and widely cited -- included as
    # an ecosystem-complete Python reference (no FFI, no C).
    "pyeclat_eclat",
    # spmf-py (pip install spmf): thin Python wrapper that shells out to the
    # SPMF jar. Same Java core as spmf_fpgrowth / spmf_eclat above, just
    # invoked via the spmf-py Python API instead of a direct java command.
    # Kept separate so a reviewer can compare the two call paths.
    "spmf_py_fpgrowth",
    "spmf_py_eclat",
]

# Full canonical raw-CSV schema. Kept as a list so pd.DataFrame.reindex keeps
# column order stable even when a given row doesn't populate every field.
RAW_COLUMNS = [
    # identity / checkpoint key
    "dataset", "method", "algo", "k", "min_support", "run_id", "warmup", "timestamp_utc",
    # timing
    "wall_s", "user_cpu_s", "sys_cpu_s", "cpu_pct",
    # memory
    "peak_rss_kb", "avg_rss_kb", "max_vsize_kb", "page_size_kb",
    # resource pressure
    "major_page_faults", "minor_page_faults",
    "voluntary_ctx_switches", "involuntary_ctx_switches",
    "fs_inputs", "fs_outputs", "sock_sent", "sock_recv", "exit_status",
    # JVM
    "jvm_heap_used_mb", "jvm_heap_committed_mb", "jvm_gc_time_ms",
    # algorithm output
    "n_itemsets", "n_rules", "output_bytes", "status",
    # dataset stats
    "n_transactions", "n_items", "n_rows",
    "avg_items_per_txn", "median_items_per_txn",
    "max_items_per_txn", "std_items_per_txn", "density",
    # environment fingerprint
    "git_sha", "hostname", "cpu_model", "cpu_cores", "ram_gb",
    "python_version", "rust_version", "java_version", "jvm_opts",
    "os_release", "swappiness", "cpu_governor", "caches_dropped",
    # per-run env overrides (captured from effective_env in run_method)
    "env_rayon_num_threads",
    # debug
    "stderr_tail", "cmd",
]

# Tiered run counts per size class. We always run N+1 and drop the first
# (warm-up) so the reported N is exactly this many.
RUNS_BY_SIZE_CLASS = {"fast": 10, "medium": 5, "heavy": 3}


# -----------------------------------------------------------------------------
# Environment fingerprint (frozen once per notebook run)
# -----------------------------------------------------------------------------

def _safe_read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def _safe_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def collect_environment() -> dict[str, Any]:
    """One-time snapshot of the host environment. Safe to call on Windows
    (most fields degrade to empty strings)."""
    try:
        cpu_model = ""
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        ram_gb = 0.0
        if Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break

        git_sha = _safe_cmd(["git", "-C", str(BENCHMARKS_DIR.parent), "rev-parse", "HEAD"])
        java_version = _safe_cmd(["java", "-version"])
        rust_version = _safe_cmd(["rustc", "--version"])
        swappiness = _safe_read("/proc/sys/vm/swappiness")
        cpu_governor = _safe_read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")

        return {
            "git_sha": git_sha[:12],
            "hostname": socket.gethostname(),
            "cpu_model": cpu_model,
            "cpu_cores": os.cpu_count() or 0,
            "ram_gb": round(ram_gb, 2),
            "python_version": sys.version.split()[0],
            "rust_version": rust_version,
            "java_version": java_version.splitlines()[0] if java_version else "",
            "jvm_opts": os.environ.get("JAVA_OPTS", ""),
            "os_release": platform.platform(),
            "swappiness": swappiness,
            "cpu_governor": cpu_governor,
            "caches_dropped": os.environ.get("FASTAPRIORI_CACHES_DROPPED", "0"),
        }
    except Exception as e:
        return {"error": repr(e)}


# -----------------------------------------------------------------------------
# Dataset stats (computed once per dataset, cached)
# -----------------------------------------------------------------------------

_DATASET_STATS_CACHE: dict[str, dict[str, float]] = {}


def dataset_stats(dataset_name: str, df: pd.DataFrame) -> dict[str, float]:
    if dataset_name in _DATASET_STATS_CACHE:
        return _DATASET_STATS_CACHE[dataset_name]
    txn_col = df.columns[0]
    item_col = df.columns[1]
    per_txn = df.groupby(txn_col)[item_col].size()
    n_txn = len(per_txn)
    n_items = df[item_col].nunique()
    n_rows = len(df)
    stats = {
        "n_transactions": n_txn,
        "n_items": n_items,
        "n_rows": n_rows,
        "avg_items_per_txn": float(per_txn.mean()),
        "median_items_per_txn": float(per_txn.median()),
        "max_items_per_txn": int(per_txn.max()),
        "std_items_per_txn": float(per_txn.std()),
        "density": n_rows / (n_txn * max(n_items, 1)),
    }
    _DATASET_STATS_CACHE[dataset_name] = stats
    return stats


# -----------------------------------------------------------------------------
# Format converters
# -----------------------------------------------------------------------------

_SPMF_WRITER_VERSION = "v2-dedup"  # bump when df_to_spmf_file's format changes


def _fingerprint_df(dataset_name: str, df: pd.DataFrame) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update(dataset_name.encode())
    h.update(_SPMF_WRITER_VERSION.encode())
    h.update(str(len(df)).encode())
    h.update(str(df.iloc[0].to_dict()).encode())
    h.update(str(df.iloc[-1].to_dict()).encode())
    return h.hexdigest()


def df_to_spmf_file(dataset_name: str, df: pd.DataFrame) -> tuple[Path, dict[Any, int]]:
    """Write one transaction per line, space-separated integer item IDs. Items
    are remapped to a dense 1..n integer space for Borgelt/SPMF compatibility.

    Returns (path, item_mapping). Cached by dataset fingerprint so large
    datasets are only serialized once per session.
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint_df(dataset_name, df)
    path = DATA_CACHE / f"{dataset_name.replace(' ', '_')}_{fp}.spmf"
    mapping_path = path.with_suffix(".mapping.json")

    if path.exists() and mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        # JSON keys are strings; map back to original type if needed
        return path, mapping

    txn_col, item_col = df.columns[0], df.columns[1]
    unique_items = sorted(df[item_col].unique())
    mapping = {item: i + 1 for i, item in enumerate(unique_items)}

    # SPMF's frequent-itemset algorithms (FPGrowth_itemsets, Eclat, ...) treat
    # each line as a *set* of items and misbehave when an item appears twice on
    # the same line — on Online Retail (where a single invoice can list the
    # same stock code on multiple rows) this silently corrupts counts. Borgelt
    # also interprets transactions as sets, so dedup is safe for both tools.
    # Preserve first-seen order so the file stays diff-friendly.
    with path.open("w", encoding="utf-8") as f:
        for _, group in df.groupby(txn_col, sort=False):
            seen: set[int] = set()
            ids: list[str] = []
            for x in group[item_col]:
                i = mapping[x]
                if i in seen:
                    continue
                seen.add(i)
                ids.append(str(i))
            f.write(" ".join(ids) + "\n")

    mapping_path.write_text(json.dumps({str(k): v for k, v in mapping.items()}))
    return path, mapping


# -----------------------------------------------------------------------------
# /usr/bin/time -v parser
# -----------------------------------------------------------------------------

_ML = re.MULTILINE
_TIME_FIELD_PATTERNS = {
    # The label is "Elapsed (wall clock) time (h:mm:ss or m:ss):" — the parenthetical
    # contains a colon, so anchor on the closing ')' before the final ':'.
    "wall_s": re.compile(r"Elapsed \(wall clock\) time[^)]*\):\s*([^\r\n]+)", _ML),
    "user_cpu_s": re.compile(r"User time \(seconds\):\s*([\d.]+)", _ML),
    "sys_cpu_s": re.compile(r"System time \(seconds\):\s*([\d.]+)", _ML),
    "cpu_pct": re.compile(r"Percent of CPU this job got:\s*(\d+)%", _ML),
    "peak_rss_kb": re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)", _ML),
    "avg_rss_kb": re.compile(r"Average resident set size \(kbytes\):\s*(\d+)", _ML),
    "max_vsize_kb": re.compile(r"Maximum virtual memory \(kbytes\):\s*(\d+)", _ML),
    "page_size_kb": re.compile(r"Page size \(bytes\):\s*(\d+)", _ML),
    "major_page_faults": re.compile(r"Major \(requiring I/O\) page faults:\s*(\d+)", _ML),
    "minor_page_faults": re.compile(r"Minor \(reclaiming a frame\) page faults:\s*(\d+)", _ML),
    "voluntary_ctx_switches": re.compile(r"Voluntary context switches:\s*(\d+)", _ML),
    "involuntary_ctx_switches": re.compile(r"Involuntary context switches:\s*(\d+)", _ML),
    "fs_inputs": re.compile(r"File system inputs:\s*(\d+)", _ML),
    "fs_outputs": re.compile(r"File system outputs:\s*(\d+)", _ML),
    "sock_sent": re.compile(r"Socket messages sent:\s*(\d+)", _ML),
    "sock_recv": re.compile(r"Socket messages received:\s*(\d+)", _ML),
    "exit_status": re.compile(r"Exit status:\s*(\d+)", _ML),
}


def _parse_elapsed(s: str) -> float:
    """Parse 'h:mm:ss' or 'm:ss.xx' (GNU time formats) to seconds."""
    s = s.strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return float("nan")


def parse_time_output(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, pat in _TIME_FIELD_PATTERNS.items():
        m = pat.search(text)
        if m is None:
            result[key] = float("nan")
            continue
        raw = m.group(1).strip()
        if key == "wall_s":
            result[key] = _parse_elapsed(raw)
        elif key == "page_size_kb":
            result[key] = int(raw) / 1024.0
        else:
            try:
                result[key] = float(raw)
            except ValueError:
                result[key] = float("nan")
    # avg_rss is reported in some GNU time versions as always-0, which is a
    # known kernel-side bug; keep the parsed value regardless.
    return result


# -----------------------------------------------------------------------------
# Dependency detection
# -----------------------------------------------------------------------------

def _which(name: str) -> str | None:
    return shutil.which(name)


def detect_tools() -> dict[str, Any]:
    """Return a dict describing which baselines are actually runnable on this
    host. Used by the notebook's Section 0 to decide what to skip."""
    info: dict[str, Any] = {
        "usr_bin_time": Path("/usr/bin/time").exists(),
        "gcc": bool(_which("gcc")),
        "make": bool(_which("make")),
        "java": bool(_which("java")),
        "wget": bool(_which("wget")),
        "borgelt_apriori": (TOOLS_BIN / "apriori").exists(),
        "borgelt_fpgrowth": (TOOLS_BIN / "fpgrowth").exists(),
        "borgelt_eclat": (TOOLS_BIN / "eclat").exists(),
        "spmf_jar": SPMF_JAR.exists(),
    }
    # Python libs. `fim` is the import name of the pyfim package;
    # `pyECLAT` is pip install pyECLAT but import pyECLAT (case-sensitive);
    # `spmf` is the import name of the spmf-py wrapper.
    for lib in ("mlxtend", "efficient_apriori", "fastapriori", "fim",
                "pyECLAT", "spmf"):
        try:
            __import__(lib)
            info[lib] = True
        except ImportError:
            info[lib] = False
    # Rename for readability in the notebook's detection print-out.
    info["pyfim"] = info.pop("fim")
    info["pyeclat"] = info.pop("pyECLAT")
    info["spmf_py"] = info.pop("spmf")
    return info


# -----------------------------------------------------------------------------
# Driver scripts for Python methods (written to workdir on demand)
# -----------------------------------------------------------------------------

FASTAPRIORI_DRIVER = r"""
import json, sys, os, time
from pathlib import Path
BENCH = Path(r"{BENCH}")
sys.path.insert(0, str(BENCH))
import numpy as np
import pandas as pd
from load_datasets import ALL_DATASETS
from fastapriori import find_associations

DATASET = {DATASET!r}
K = {K}
MIN_SUPPORT = {MIN_SUPPORT}
ALGO = {ALGO!r}
FAST_VARIANT = {FAST_VARIANT!r}   # None, "apriori", or "eclat"
OUT_JSON = Path(r"{OUT_JSON}")

# Load dataset BEFORE the timer so I/O and parsing aren't charged to the library.
df = ALL_DATASETS[DATASET]()
txn_col, item_col = df.columns[0], df.columns[1]

# ---- Timed section ----
# Single call that computes k=2..K internally via the Rust compute_pipeline
# (one inverted-index build, one sweep). The previous level-by-level chain
# (`for kk in range(2, K+1): find_associations(...frequent_lower=...)`) was
# ~6x slower on Instacart k=3 because each call rebuilt the index from
# scratch and redid the lower-level counting. Eclat also runs k=2..K in one
# recursion internally, so the single call is the fair path for every
# fast_variant.
#
# low_memory=False skips the pandas groupby(item).nunique() pre-filter
# (~5-7 s on Instacart, redundant with the int32 downward-closure filter
# inside count_k_itemsets_internal). The benchmark VM has 60 GB RAM, so
# trading RAM for speed here is always the right call.  Explicit beats
# relying on core.py's "auto" default resolving correctly for every release.
backend_options = (
    None if FAST_VARIANT is None else {{"fast_variant": FAST_VARIANT}}
)

t0 = time.perf_counter()
result = find_associations(
    df,
    transaction_col=txn_col,
    item_col=item_col,
    k=K,
    min_support=MIN_SUPPORT,
    min_confidence=0.0,
    backend="auto",
    algo=ALGO,
    sorted_by=None,
    low_memory=False,
    backend_options=backend_options,
)
inner_wall_s = time.perf_counter() - t0
# ---- End timed section ----

# Count unique unordered K-itemsets for validation. Sort items within each
# row (so {{A,B}} == {{B,A}}), then drop duplicates via pandas — works for
# any dtype (Online Retail items are mixed-type strings like "85048", "POST"
# which break np.unique(axis=0)). Orders-of-magnitude faster than the old
# iterrows + frozenset loop that was dominating wall time at low support.
item_cols = [c for c in result.columns if c.startswith("antecedent") or c in ("item_A", "item_B", "consequent")]
if not item_cols:
    item_cols = list(result.columns[:K])

if len(result) == 0:
    n_itemsets = 0
else:
    items_mat = result[item_cols].to_numpy()
    # np.sort on object arrays uses Python comparison — works for homogenous
    # strings or homogenous ints within a row; an occasional mixed row would
    # raise, but within one dataset items share a dtype.
    items_sorted = np.sort(items_mat, axis=1)
    sorted_df = pd.DataFrame(items_sorted, columns=item_cols)
    n_itemsets = int(len(sorted_df.drop_duplicates()))

n_rules = int(len(result))
OUT_JSON.write_text(json.dumps({{
    "n_itemsets": int(n_itemsets),
    "n_rules": int(n_rules),
    "inner_wall_s": float(inner_wall_s),
}}))
"""

EFFICIENT_APRIORI_DRIVER = r"""
import json, sys, time
from pathlib import Path
BENCH = Path(r"{BENCH}")
sys.path.insert(0, str(BENCH))
from load_datasets import ALL_DATASETS
from efficient_apriori import apriori

DATASET = {DATASET!r}
K = {K}
MIN_SUPPORT = {MIN_SUPPORT}
OUT_JSON = Path(r"{OUT_JSON}")

df = ALL_DATASETS[DATASET]()
txn_col, item_col = df.columns[0], df.columns[1]
transactions = [tuple(g) for _, g in df.groupby(txn_col, sort=False)[item_col]]

t0 = time.perf_counter()
itemsets, rules = apriori(
    transactions,
    min_support=MIN_SUPPORT,
    min_confidence=0.0,
    max_length=K,
    output_transaction_ids=False,
    verbosity=0,
)
inner_wall_s = time.perf_counter() - t0

# Count only size-K itemsets to match Borgelt / fastapriori semantics.
n_itemsets = len(itemsets.get(K, {{}}))
OUT_JSON.write_text(json.dumps({{
    "n_itemsets": int(n_itemsets),
    "n_rules": int(len(rules)),
    "inner_wall_s": float(inner_wall_s),
}}))
"""

PYFIM_DRIVER = r"""
import json, sys, time
from pathlib import Path
BENCH = Path(r"{BENCH}")
sys.path.insert(0, str(BENCH))
from load_datasets import ALL_DATASETS
# `fim` is Borgelt's PyFIM extension module (pip install pyfim). It exposes
# apriori, fpgrowth, eclat (and others) as plain Python callables that invoke
# the same C hot loop as the standalone borgelt_* binaries.
from fim import apriori as pf_apriori, fpgrowth as pf_fpgrowth, eclat as pf_eclat

DATASET = {DATASET!r}
K = {K}
MIN_SUPPORT = {MIN_SUPPORT}
ALGO = {ALGO!r}
OUT_JSON = Path(r"{OUT_JSON}")

df = ALL_DATASETS[DATASET]()
txn_col, item_col = df.columns[0], df.columns[1]
transactions = [tuple(g) for _, g in df.groupby(txn_col, sort=False)[item_col]]

# PyFIM supp semantics: positive value = percent (0..100); negative = absolute
# transaction count. We pass percent to stay consistent with the notebook's
# min_support fraction.
supp_pct = MIN_SUPPORT * 100.0

# zmin=zmax=K pins the output to size-K itemsets exactly (matches Borgelt's
# `-m{{k}} -n{{k}}` flags and fastapriori/efficient-apriori semantics).
kwargs = dict(target='s', supp=supp_pct, zmin=K, zmax=K, report='a')

t0 = time.perf_counter()
if ALGO == 'apriori':
    result = pf_apriori(transactions, **kwargs)
elif ALGO == 'fpgrowth':
    result = pf_fpgrowth(transactions, **kwargs)
elif ALGO == 'eclat':
    result = pf_eclat(transactions, **kwargs)
else:
    raise ValueError(f"unknown pyfim algo {{ALGO!r}}")
inner_wall_s = time.perf_counter() - t0

n_itemsets = int(len(result))
OUT_JSON.write_text(json.dumps({{
    "n_itemsets": n_itemsets,
    "n_rules": n_itemsets,
    "inner_wall_s": float(inner_wall_s),
}}))
"""

MLXTEND_DRIVER = r"""
import json, sys, time
from pathlib import Path
BENCH = Path(r"{BENCH}")
sys.path.insert(0, str(BENCH))
import pandas as pd
from load_datasets import ALL_DATASETS
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori as mlx_apriori, fpgrowth as mlx_fpgrowth

DATASET = {DATASET!r}
K = {K}
MIN_SUPPORT = {MIN_SUPPORT}
ALGO = {ALGO!r}
OUT_JSON = Path(r"{OUT_JSON}")

# One-hot encoding and TransactionEncoder.fit run before the timer — they're
# mlxtend's input-shaping tax, not the algorithm itself. fastapriori pays its
# own equivalent inside find_associations, which is what gets timed there.
df = ALL_DATASETS[DATASET]()
txn_col, item_col = df.columns[0], df.columns[1]
transactions = [list(g) for _, g in df.groupby(txn_col, sort=False)[item_col]]
te = TransactionEncoder()
arr = te.fit(transactions).transform(transactions, sparse=True)
onehot = pd.DataFrame.sparse.from_spmatrix(arr, columns=te.columns_)

t0 = time.perf_counter()
if ALGO == "apriori":
    result = mlx_apriori(onehot, min_support=MIN_SUPPORT, max_len=K, use_colnames=False, low_memory=False)
else:
    result = mlx_fpgrowth(onehot, min_support=MIN_SUPPORT, max_len=K, use_colnames=False)
inner_wall_s = time.perf_counter() - t0

n_itemsets = int(len(result))
OUT_JSON.write_text(json.dumps({{
    "n_itemsets": n_itemsets,
    "n_rules": n_itemsets,
    "inner_wall_s": float(inner_wall_s),
}}))
"""

PYECLAT_DRIVER = r"""
import json, sys, time
from pathlib import Path
BENCH = Path(r"{BENCH}")
sys.path.insert(0, str(BENCH))
import pandas as pd
from load_datasets import ALL_DATASETS
# pyECLAT (pip install pyECLAT): pure-Python Eclat. Slow, but useful as the
# ecosystem's "no-FFI" Python reference alongside efficient-apriori/mlxtend.
from pyECLAT import ECLAT

DATASET = {DATASET!r}
K = {K}
MIN_SUPPORT = {MIN_SUPPORT}
OUT_JSON = Path(r"{OUT_JSON}")

df = ALL_DATASETS[DATASET]()
txn_col, item_col = df.columns[0], df.columns[1]
transactions = [list(g) for _, g in df.groupby(txn_col, sort=False)[item_col]]

# pyECLAT wants a ragged DataFrame, one row per transaction, NaN-padded.
txn_df = pd.DataFrame(transactions)
eclat = ECLAT(data=txn_df, verbose=False)

t0 = time.perf_counter()
# min_combination=max_combination=K pins output to size-K itemsets.
_indices, supports = eclat.fit(
    min_support=MIN_SUPPORT,
    min_combination=K,
    max_combination=K,
    separator=' & ',
    verbose=False,
)
inner_wall_s = time.perf_counter() - t0

n_itemsets = int(len(supports))
OUT_JSON.write_text(json.dumps({{
    "n_itemsets": n_itemsets,
    "n_rules": n_itemsets,
    "inner_wall_s": float(inner_wall_s),
}}))
"""

SPMFPY_DRIVER = r"""
import json, sys, os, tempfile, time
from pathlib import Path
BENCH = Path(r"{BENCH}")
sys.path.insert(0, str(BENCH))
from load_datasets import ALL_DATASETS
# spmf-py (pip install spmf) shells out to spmf.jar internally. Same Java
# core as our direct spmf_* subprocess path; kept as a separate method so
# reviewers can compare the two invocation paths.
from spmf import Spmf
from baseline_runners import df_to_spmf_file, SPMF_JAR

DATASET = {DATASET!r}
K = {K}
MIN_SUPPORT = {MIN_SUPPORT}
ALGO = {ALGO!r}   # 'fpgrowth' or 'eclat'
OUT_JSON = Path(r"{OUT_JSON}")

df = ALL_DATASETS[DATASET]()
spmf_input, _ = df_to_spmf_file(DATASET, df)

algo_name = {{'fpgrowth': 'FPGrowth_itemsets', 'eclat': 'Eclat'}}[ALGO]
# Keep the SPMF output in our own workdir (not /tmp) so leaks — from OOMs,
# timeouts, or SPMF's Java side crashing mid-write — are easier to spot and
# bound by the same quota as the rest of bench_workdir. We always try to
# delete in the finally block, so a clean run leaves nothing behind.
SPMF_TMP_DIR = BENCH / '_bench_workdir' / 'spmf_tmp'
SPMF_TMP_DIR.mkdir(parents=True, exist_ok=True)
out_tmp = tempfile.NamedTemporaryFile(
    prefix='spmfpy_', suffix='.txt', dir=str(SPMF_TMP_DIR), delete=False,
)
out_tmp.close()
out_path = out_tmp.name

try:
    spmf_job = Spmf(
        algo_name,
        input_filename=str(spmf_input),
        output_filename=out_path,
        arguments=[f'{{MIN_SUPPORT * 100:.6f}}%'],
        spmf_bin_location_dir=str(SPMF_JAR.parent),
    )
    t0 = time.perf_counter()
    spmf_job.run()
    inner_wall_s = time.perf_counter() - t0

    # Count size-K itemsets from SPMF output. Line format:
    #   'item1 item2 ... #SUP: count' (items are whitespace-separated ints).
    # Strip any '#TAG: value' annotations (#SUP, #SUPP, #UTIL, ...) and skip
    # any line whose itemset part isn't pure whitespace-separated integers —
    # that filters out blank lines, comments, and algorithm progress markers
    # which would otherwise be miscounted as itemsets.
    import re as _re
    _ANN = _re.compile(r'\s*#[A-Z]+\s*:')
    n = 0
    with open(out_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items_part = _ANN.split(s, maxsplit=1)[0].strip()
            if not items_part:
                continue
            toks = items_part.split()
            if not all(t.isdigit() for t in toks):
                continue
            if len(toks) == K:
                n += 1

    OUT_JSON.write_text(json.dumps({{
        "n_itemsets": int(n),
        "n_rules": int(n),
        "inner_wall_s": float(inner_wall_s),
    }}))
finally:
    try:
        os.unlink(out_path)
    except Exception:
        pass
"""


# -----------------------------------------------------------------------------
# Command builders
# -----------------------------------------------------------------------------

def _borgelt_target_arg(method: str) -> str:
    # Borgelt's -t flag: s = frequent sets, c = association rules, ...
    # We ask for frequent itemsets to match across tools.
    return "-ts"


def _build_borgelt_cmd(method: str, spmf_file: Path, out_file: Path,
                       k: int, min_support: float) -> list[str]:
    bin_name = method.replace("borgelt_", "")
    bin_path = TOOLS_BIN / bin_name
    support_pct = min_support * 100.0
    # Write itemset dump to /dev/null so the disk doesn't fill with GB-scale
    # text output on low-support / large-k configs. We parse the count from the
    # summary Borgelt prints to its default (non-quiet) stderr:
    #   'writing /dev/null ... [213 set(s)] done [0.00s].'
    return [
        str(bin_path),
        _borgelt_target_arg(method),
        f"-s{support_pct:.6f}",
        f"-m{k}",
        f"-n{k}",
        str(spmf_file),
        "/dev/null",
    ]


_BORGELT_COUNT_RE = re.compile(r"\[(\d+)\s*set\(s\)\]")


def _count_borgelt_from_stderr(stderr_text: str) -> int:
    """Parse 'writing ... [N set(s)]' from Borgelt's stdout/stderr."""
    m = _BORGELT_COUNT_RE.search(stderr_text or "")
    return int(m.group(1)) if m else 0


def _build_spmf_cmd(method: str, spmf_file: Path, out_file: Path,
                    k: int, min_support: float, mem_mb: int) -> list[str]:
    algo_name = {
        "spmf_fpgrowth": "FPGrowth_itemsets",
        "spmf_eclat": "Eclat",
    }[method]
    return [
        "java",
        f"-Xmx{mem_mb}m",
        "-XX:+UseG1GC",
        "-jar", str(SPMF_JAR),
        "run", algo_name,
        str(spmf_file),
        str(out_file),
        f"{min_support * 100:.6f}%",
    ]


def _write_driver(method: str, dataset: str, k: int, min_support: float,
                  out_json: Path) -> Path:
    DRIVERS_DIR.mkdir(parents=True, exist_ok=True)
    bench = str(BENCHMARKS_DIR).replace("\\", "/")
    out_json_s = str(out_json).replace("\\", "/")
    if method == "fastapriori_fast":
        src = FASTAPRIORI_DRIVER.format(
            BENCH=bench, DATASET=dataset, K=k, MIN_SUPPORT=min_support,
            ALGO="fast", FAST_VARIANT=None, OUT_JSON=out_json_s,
        )
        name = "driver_fastapriori_fast.py"
    elif method == "fastapriori_fast_apriori":
        src = FASTAPRIORI_DRIVER.format(
            BENCH=bench, DATASET=dataset, K=k, MIN_SUPPORT=min_support,
            ALGO="fast", FAST_VARIANT="apriori", OUT_JSON=out_json_s,
        )
        name = "driver_fastapriori_fast_apriori.py"
    elif method == "fastapriori_fast_eclat":
        src = FASTAPRIORI_DRIVER.format(
            BENCH=bench, DATASET=dataset, K=k, MIN_SUPPORT=min_support,
            ALGO="fast", FAST_VARIANT="eclat", OUT_JSON=out_json_s,
        )
        name = "driver_fastapriori_fast_eclat.py"
    elif method == "fastapriori_fast_1t":
        # Same driver as fastapriori_fast; the thread cap is applied via
        # RAYON_NUM_THREADS=1 in run_method's env merge (see above).
        src = FASTAPRIORI_DRIVER.format(
            BENCH=bench, DATASET=dataset, K=k, MIN_SUPPORT=min_support,
            ALGO="fast", FAST_VARIANT=None, OUT_JSON=out_json_s,
        )
        name = "driver_fastapriori_fast_1t.py"
    elif method == "fastapriori_classic":
        src = FASTAPRIORI_DRIVER.format(
            BENCH=bench, DATASET=dataset, K=k, MIN_SUPPORT=min_support,
            ALGO="classic", FAST_VARIANT=None, OUT_JSON=out_json_s,
        )
        name = "driver_fastapriori_classic.py"
    elif method == "efficient_apriori":
        src = EFFICIENT_APRIORI_DRIVER.format(
            BENCH=bench, DATASET=dataset, K=k, MIN_SUPPORT=min_support,
            OUT_JSON=out_json_s,
        )
        name = "driver_efficient_apriori.py"
    elif method.startswith("pyfim_"):
        algo = method.split("_", 1)[1]   # apriori / fpgrowth / eclat
        src = PYFIM_DRIVER.format(
            BENCH=bench, DATASET=dataset, K=k, MIN_SUPPORT=min_support,
            ALGO=algo, OUT_JSON=out_json_s,
        )
        name = f"driver_pyfim_{algo}.py"
    elif method == "mlxtend_apriori":
        src = MLXTEND_DRIVER.format(
            BENCH=bench, DATASET=dataset, K=k, MIN_SUPPORT=min_support,
            ALGO="apriori", OUT_JSON=out_json_s,
        )
        name = "driver_mlxtend_apriori.py"
    elif method == "mlxtend_fpgrowth":
        src = MLXTEND_DRIVER.format(
            BENCH=bench, DATASET=dataset, K=k, MIN_SUPPORT=min_support,
            ALGO="fpgrowth", OUT_JSON=out_json_s,
        )
        name = "driver_mlxtend_fpgrowth.py"
    elif method == "pyeclat_eclat":
        src = PYECLAT_DRIVER.format(
            BENCH=bench, DATASET=dataset, K=k, MIN_SUPPORT=min_support,
            OUT_JSON=out_json_s,
        )
        name = "driver_pyeclat_eclat.py"
    elif method.startswith("spmf_py_"):
        algo = method.split("spmf_py_", 1)[1]   # fpgrowth / eclat
        src = SPMFPY_DRIVER.format(
            BENCH=bench, DATASET=dataset, K=k, MIN_SUPPORT=min_support,
            ALGO=algo, OUT_JSON=out_json_s,
        )
        name = f"driver_spmf_py_{algo}.py"
    else:
        raise ValueError(f"No Python driver for method={method}")
    path = DRIVERS_DIR / name
    path.write_text(src)
    return path


# -----------------------------------------------------------------------------
# Output parsers (count itemsets produced by each tool)
# -----------------------------------------------------------------------------

def _count_borgelt_itemsets(out_file: Path, k: int | None = None) -> int:
    """Count Borgelt output lines. Each line looks like: 'id1 id2 ... (support)'.
    When k is given, count only lines with exactly k item tokens."""
    try:
        n = 0
        with out_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if k is None:
                    n += 1
                    continue
                # Drop the trailing "(support)" or "(count)" token before counting items.
                if "(" in s:
                    s = s.split("(", 1)[0].strip()
                if len(s.split()) == k:
                    n += 1
        return n
    except FileNotFoundError:
        return 0


# Strip any SPMF trailing annotation like '#SUP: 42', '#SUPP: 42', '#UTIL: 100'.
# SPMF emits at least one '#TAG: value' after the itemset; some algorithms emit
# several (e.g. '#SUP: 42 #UTIL: 100'). Splitting on the first '#<UPPER>:' token
# isolates the itemset tokens regardless of which annotations follow.
_SPMF_ANNOTATION_RE = re.compile(r"\s*#[A-Z]+\s*:")


def _count_spmf_itemsets(out_file: Path, k: int | None = None) -> int:
    """Count SPMF frequent-itemset output lines.

    Line format: ``item1 item2 ... itemN #SUP: count`` (items are whitespace-
    separated integers; SPMF may append extra annotations such as
    ``#UTIL: 100`` which we strip). When ``k`` is given, count only itemsets of
    exactly ``k`` items.

    Skips comment/header/blank lines that don't look like pure-integer itemset
    lines — SPMF writes a run summary at the head of some output files and
    algorithms like ``Eclat`` may print progress markers.
    """
    try:
        n = 0
        with out_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                items_part = _SPMF_ANNOTATION_RE.split(s, maxsplit=1)[0].strip()
                if not items_part:
                    continue
                tokens = items_part.split()
                # Reject lines whose "itemset" part contains non-digit tokens
                # (separators, headers, rule arrows like '==>', etc.). Real
                # itemsets are always whitespace-separated integers because
                # df_to_spmf_file remaps items to dense 1..n ids.
                if not all(t.isdigit() for t in tokens):
                    continue
                if k is None or len(tokens) == k:
                    n += 1
        return n
    except FileNotFoundError:
        return 0


# -----------------------------------------------------------------------------
# Core runner
# -----------------------------------------------------------------------------

@dataclass
class RunResult:
    row: dict[str, Any] = field(default_factory=dict)


def _classify_exit(returncode: int, timed_out: bool, stderr_tail: str) -> str:
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "ok"
    # ulimit -v → process dies with SIGSEGV (139) or allocation error messages;
    # Linux OOM-killer sends SIGKILL (137). Memory-alloc abort → 134 (SIGABRT).
    if returncode in (134, 137, 139):
        return "oom"
    if "MemoryError" in stderr_tail or "OutOfMemoryError" in stderr_tail:
        return "oom"
    if "Cannot allocate memory" in stderr_tail or "std::bad_alloc" in stderr_tail:
        return "oom"
    return "error"


def run_method(
    method: str,
    dataset_name: str,
    df: pd.DataFrame,
    k: int,
    min_support: float,
    timeout_s: int = 1500,
    memout_kb: int = 60 * 1024 * 1024,
    env_snapshot: dict[str, Any] | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Spawn one subprocess for (method, dataset, k, min_support). Returns a
    row dict suitable for append_raw_row()."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    env_snapshot = env_snapshot or {}

    out_dir = WORKDIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    time_file = tempfile.NamedTemporaryFile(prefix="time_", suffix=".log",
                                            dir=WORKDIR, delete=False)
    time_file.close()
    out_file = out_dir / f"{method}_{dataset_name.replace(' ', '_')}_{k}_{min_support}.out"
    out_json = out_dir / f"{method}_{dataset_name.replace(' ', '_')}_{k}_{min_support}.json"

    # Build the inner command
    inner: list[str]
    if method.startswith("borgelt_"):
        spmf_file, _ = df_to_spmf_file(dataset_name, df)
        inner = _build_borgelt_cmd(method, spmf_file, out_file, k, min_support)
    elif method.startswith("spmf_") and not method.startswith("spmf_py_"):
        # Direct java invocation of spmf.jar. (spmf_py_* below is the Python
        # wrapper flavour and falls through to the Python-driver branch.)
        spmf_file, _ = df_to_spmf_file(dataset_name, df)
        mem_mb = min(memout_kb // 1024, 56 * 1024)  # keep some headroom for RSS accounting
        inner = _build_spmf_cmd(method, spmf_file, out_file, k, min_support, mem_mb)
    else:
        driver = _write_driver(method, dataset_name, k, min_support, out_json)
        inner = [sys.executable, "-X", "utf8", "-u", str(driver)]

    # Outer wrapper: /usr/bin/time -v -o timefile <inner>, with ulimit + timeout
    # via a sh -c preamble. We keep the full command as a single shell string
    # so ulimit applies to the entire process tree.
    memout_kb_int = int(memout_kb)
    inner_quoted = " ".join(shlex.quote(x) for x in inner)
    shell_cmd = (
        f"ulimit -v {memout_kb_int}; "
        f"exec /usr/bin/time -v -o {shlex.quote(str(time_file.name))} "
        f"timeout {int(timeout_s)} {inner_quoted}"
    )
    full_cmd = ["/bin/bash", "-c", shell_cmd]

    # Merge any per-run env overrides (e.g. RAYON_NUM_THREADS for the
    # thread-scaling notebook) on top of the parent env.
    #
    # Per-method thread policy:
    #   • fastapriori_fast       — default (all cores). What users pip-install.
    #   • fastapriori_fast_1t    — pinned to RAYON_NUM_THREADS=1. Paper's
    #     "algorithmic parity against serial C" row; listed at k=2 only.
    # All other baselines ignore RAYON_NUM_THREADS, so setting it affects
    # only the fastapriori methods.
    # An explicit RAYON_NUM_THREADS in `extra_env` (thread-scaling notebook)
    # always wins and is never overridden here.
    effective_env: dict[str, str] = {k_: str(v) for k_, v in (extra_env or {}).items()}
    if method == "fastapriori_fast_1t" and "RAYON_NUM_THREADS" not in effective_env:
        effective_env["RAYON_NUM_THREADS"] = "1"

    child_env = None
    if effective_env:
        child_env = os.environ.copy()
        child_env.update(effective_env)

    started = time.monotonic()
    proc = subprocess.run(
        full_cmd, capture_output=True, text=True,
        timeout=timeout_s + 30,  # soft guard; timeout is the hard limit
        env=child_env,
    )
    elapsed_wall_monotonic = time.monotonic() - started

    # Parse /usr/bin/time -v output
    try:
        time_text = Path(time_file.name).read_text()
    except FileNotFoundError:
        time_text = ""
    time_metrics = parse_time_output(time_text)
    if time_text == "" or time_metrics.get("wall_s") != time_metrics.get("wall_s"):  # NaN check
        # time didn't execute (e.g., binary not found before /usr/bin/time ran);
        # fall back to monotonic wall + NaN memory.
        time_metrics["wall_s"] = elapsed_wall_monotonic

    # Clean up time log
    try:
        os.unlink(time_file.name)
    except Exception:
        pass

    # Classify status
    stderr_tail = (proc.stderr or "")[-2000:]
    timed_out_flag = (proc.returncode == 124)
    status = _classify_exit(proc.returncode, timed_out_flag, stderr_tail)

    # Borgelt's apriori / fpgrowth / eclat exit non-zero on several "clean
    # termination, no result" paths and we want all of them treated as ok so
    # the heatmap doesn't get blank cells:
    #   * No item meets support  -> stderr has `[0 item(s)] done [...]` and
    #     never reaches the writing phase. n_itemsets = 0.
    #   * Items frequent but no k-itemset meets support -> stderr has
    #     `[0 set(s)] done`. n_itemsets = 0.
    #   * Normal positive result accidentally exits non-zero on some Borgelt
    #     versions -> stderr has `[N set(s)] done`. n_itemsets = N.
    # All three share a `done [...]` timing footer; harder failures
    # (segfault / OOM / timeout / parse error) never print one.
    if status == "error" and method.startswith("borgelt_"):
        _combined = (proc.stdout or "") + (proc.stderr or "")
        _clean_done = (
            _BORGELT_COUNT_RE.search(_combined) is not None
            or re.search(r"\[\d+\s+item\(s\)\]\s*done", _combined) is not None
            or "done [" in _combined
        )
        # And no hard-failure keyword in the same buffer.
        _hard_fail = any(s in _combined for s in (
            "Segmentation fault", "Cannot allocate", "std::bad_alloc",
            "out of memory", "Aborted",
        ))
        if _clean_done and not _hard_fail:
            status = "ok"

    # Count itemsets produced
    n_itemsets, n_rules = 0, 0
    output_bytes = 0
    if status == "ok":
        if method.startswith("borgelt_"):
            # Borgelt writes to /dev/null now; parse count from its summary
            # which goes to stdout (captured in proc.stdout).
            n_itemsets = _count_borgelt_from_stderr((proc.stdout or "") + (proc.stderr or ""))
            n_rules = n_itemsets
            output_bytes = 0
        elif method.startswith("spmf_") and not method.startswith("spmf_py_"):
            n_itemsets = _count_spmf_itemsets(out_file, k=k)
            n_rules = n_itemsets
            output_bytes = out_file.stat().st_size if out_file.exists() else 0
        else:
            try:
                data = json.loads(out_json.read_text())
                n_itemsets = int(data.get("n_itemsets", 0))
                n_rules = int(data.get("n_rules", 0))
                output_bytes = out_json.stat().st_size if out_json.exists() else 0
                # Prefer the driver's `time.perf_counter()` measurement of just
                # the library call over the subprocess-wall time from
                # /usr/bin/time -v. Subprocess wall includes interpreter
                # startup, dataset load, encoder prep, and post-processing —
                # on Online Retail at low support those add ~90 s of overhead
                # that isn't the algorithm. Borgelt/SPMF direct-binary paths
                # have no such Python overhead, so they keep subprocess wall.
                inner = data.get("inner_wall_s")
                if isinstance(inner, (int, float)) and inner == inner:  # not NaN
                    time_metrics["wall_s"] = float(inner)
            except Exception:
                status = "error"

    # Delete per-run dump immediately; we recorded the count + byte size above.
    # Keeping them would blow up disk usage on SPMF (writes itemsets of all sizes).
    for p in (out_file, out_json):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    # Dataset stats (cached)
    ds_stats = dataset_stats(dataset_name, df)

    # Assemble row
    if method in ("fastapriori_fast",
                  "fastapriori_fast_apriori",
                  "fastapriori_fast_eclat",
                  "fastapriori_fast_1t"):
        algo = "fast"
    elif method == "fastapriori_classic":
        algo = "classic"
    elif method.startswith("spmf_py_"):
        algo = method.split("spmf_py_", 1)[1]   # fpgrowth / eclat
    elif "_" in method:
        algo = method.split("_", 1)[1]
    else:
        algo = ""

    row: dict[str, Any] = {
        "dataset": dataset_name,
        "method": method,
        "algo": algo,
        "k": k,
        "min_support": min_support,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_itemsets": n_itemsets,
        "n_rules": n_rules,
        "output_bytes": output_bytes,
        "status": status,
        "stderr_tail": stderr_tail.replace("\n", " | ")[-500:],
        "cmd": shell_cmd[:500],
    }
    row.update(time_metrics)
    row.update(ds_stats)
    row.update(env_snapshot)

    # Record per-run env overrides inline (e.g. thread count) so scaling
    # notebooks don't need a separate bookkeeping pass.  We log `effective_env`
    # (not the caller's `extra_env`) so internally-added overrides such as
    # the RAYON_NUM_THREADS=1 pin for fastapriori_fast_1t are captured too.
    if effective_env:
        for k_env, v_env in effective_env.items():
            row[f"env_{k_env.lower()}"] = str(v_env)

    # JVM metrics: leave as NaN; SPMF doesn't expose cleanly without extra
    # JVM flags (future work). Adding placeholders keeps schema stable.
    row.setdefault("jvm_heap_used_mb", float("nan"))
    row.setdefault("jvm_heap_committed_mb", float("nan"))
    row.setdefault("jvm_gc_time_ms", float("nan"))

    return row


# -----------------------------------------------------------------------------
# Tiered executor
# -----------------------------------------------------------------------------

def classify_size(wall_s: float) -> str:
    if wall_s < 60:
        return "fast"
    if wall_s < 600:
        return "medium"
    return "heavy"


def measure(
    method: str,
    dataset_name: str,
    df: pd.DataFrame,
    k: int,
    min_support: float,
    timeout_s: int,
    memout_kb: int,
    env_snapshot: dict[str, Any],
    existing_run_ids: set[int] | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run (method, dataset, k, support) N+1 times (N+warm-up) per the tiered
    policy. Uses the first run to classify workload size, then runs as many
    more as the class demands.

    If existing_run_ids is given, skip already-completed run_ids (supports
    resume-from-CSV).
    """
    existing_run_ids = existing_run_ids or set()
    rows: list[dict[str, Any]] = []

    # Probe first (run_id = 0 is always the warm-up)
    if 0 not in existing_run_ids:
        probe = run_method(method, dataset_name, df, k, min_support,
                           timeout_s, memout_kb, env_snapshot, extra_env)
        probe["run_id"] = 0
        probe["warmup"] = True
        rows.append(probe)
        wall = probe.get("wall_s") or 0.0
        status = probe.get("status", "error")
    else:
        # We've seen the probe before; the only way to re-classify without
        # rerunning is to read the recorded wall time from CSV upstream; for
        # simplicity assume medium if unknown. This is a rare edge case.
        wall = 60.0
        status = "ok"

    # If the probe failed, don't repeat.
    if status != "ok":
        return rows

    n_keeps = RUNS_BY_SIZE_CLASS[classify_size(wall)]
    for i in range(1, n_keeps + 1):
        if i in existing_run_ids:
            continue
        r = run_method(method, dataset_name, df, k, min_support,
                       timeout_s, memout_kb, env_snapshot, extra_env)
        r["run_id"] = i
        r["warmup"] = False
        rows.append(r)
        if r.get("status") != "ok":
            # One failure aborts the tier (preserves cost budget)
            break
    return rows


# -----------------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------------

def append_raw_row(row: dict[str, Any], csv_path: Path | None = None) -> None:
    # Resolve default at call time so set_raw_csv_suffix() takes effect even
    # after this function has been imported into the notebook.
    csv_path = csv_path or RAW_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row]).reindex(columns=RAW_COLUMNS)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def load_raw(csv_path: Path | None = None) -> pd.DataFrame:
    csv_path = csv_path or RAW_CSV
    if not csv_path.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.read_csv(csv_path)


def already_done(raw: pd.DataFrame, dataset: str, method: str, k: int,
                 min_support: float) -> set[int]:
    if len(raw) == 0:
        return set()
    mask = (
        (raw["dataset"] == dataset)
        & (raw["method"] == method)
        & (raw["k"] == k)
        & (raw["min_support"].astype(float).round(12) == round(float(min_support), 12))
    )
    return set(raw.loc[mask, "run_id"].dropna().astype(int))


# -----------------------------------------------------------------------------
# Aggregation (raw → summary)
# -----------------------------------------------------------------------------

NUMERIC_METRICS = [
    "wall_s", "user_cpu_s", "sys_cpu_s", "peak_rss_kb", "avg_rss_kb",
    "max_vsize_kb", "major_page_faults", "minor_page_faults",
    "voluntary_ctx_switches", "involuntary_ctx_switches",
    "fs_inputs", "fs_outputs", "n_itemsets", "n_rules",
    "jvm_heap_used_mb",
]


def aggregate(raw: pd.DataFrame, reference_method: str = "fastapriori_fast") -> pd.DataFrame:
    """Drop warm-ups, compute mean/std/cv/median/IQR/min/max/range/n_* for every
    numeric metric. Adds speedup/rss-ratio columns vs a reference method."""
    if len(raw) == 0:
        return pd.DataFrame()

    df = raw.copy()
    df["warmup"] = df["warmup"].fillna(False).astype(bool)
    df = df[~df["warmup"]]

    keys = ["dataset", "method", "algo", "k", "min_support"]
    out_rows: list[dict[str, Any]] = []

    for keyvals, group in df.groupby(keys, dropna=False):
        row: dict[str, Any] = dict(zip(keys, keyvals))
        ok = group[group["status"] == "ok"]
        row["n_runs_total"] = len(group)
        for metric in NUMERIC_METRICS:
            series = ok[metric].astype(float).dropna()
            if len(series) == 0:
                for s in ("mean", "std", "cv", "median", "q25", "q75", "iqr",
                          "min", "max", "range"):
                    row[f"{metric}_{s}"] = float("nan")
                row[f"{metric}_n_runs"] = 0
            else:
                mean = series.mean()
                std = series.std(ddof=1) if len(series) > 1 else 0.0
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                row[f"{metric}_cv"] = (std / mean) if mean else float("nan")
                row[f"{metric}_median"] = series.median()
                row[f"{metric}_q25"] = q25
                row[f"{metric}_q75"] = q75
                row[f"{metric}_iqr"] = q75 - q25
                row[f"{metric}_min"] = series.min()
                row[f"{metric}_max"] = series.max()
                row[f"{metric}_range"] = series.max() - series.min()
                row[f"{metric}_n_runs"] = int(len(series))

            status_counts = group["status"].value_counts().to_dict()
            row[f"{metric}_n_ok"] = int(status_counts.get("ok", 0))
            row[f"{metric}_n_timeout"] = int(status_counts.get("timeout", 0))
            row[f"{metric}_n_oom"] = int(status_counts.get("oom", 0))

        # Copy dataset stats from first row of group
        for col in ("n_transactions", "n_items", "n_rows",
                    "avg_items_per_txn", "median_items_per_txn",
                    "max_items_per_txn", "std_items_per_txn", "density"):
            row[col] = group[col].dropna().iloc[0] if group[col].notna().any() else float("nan")

        row["status_counts_json"] = json.dumps(
            group["status"].value_counts().to_dict()
        )
        out_rows.append(row)

    agg = pd.DataFrame(out_rows)

    # Speedup / rss-ratio vs reference method, per (dataset, k, min_support)
    ref = agg[agg["method"] == reference_method].set_index(
        ["dataset", "k", "min_support"]
    )
    agg["_key"] = list(zip(agg["dataset"], agg["k"], agg["min_support"]))

    def _lookup(key, col):
        if key not in ref.index:
            return float("nan")
        return ref.loc[key, col]

    for key in agg["_key"]:
        pass  # vectorize below

    agg["speedup_vs_fastapriori_fast_mean"] = agg.apply(
        lambda r: _lookup((r["dataset"], r["k"], r["min_support"]), "wall_s_mean") / r["wall_s_mean"]
        if r["wall_s_mean"] and r["wall_s_mean"] == r["wall_s_mean"] else float("nan"),
        axis=1,
    )
    agg["speedup_vs_fastapriori_fast_median"] = agg.apply(
        lambda r: _lookup((r["dataset"], r["k"], r["min_support"]), "wall_s_median") / r["wall_s_median"]
        if r["wall_s_median"] and r["wall_s_median"] == r["wall_s_median"] else float("nan"),
        axis=1,
    )
    agg["rss_ratio_vs_fastapriori_fast_mean"] = agg.apply(
        lambda r: r["peak_rss_kb_mean"] / _lookup((r["dataset"], r["k"], r["min_support"]), "peak_rss_kb_mean")
        if r["peak_rss_kb_mean"] and r["peak_rss_kb_mean"] == r["peak_rss_kb_mean"] else float("nan"),
        axis=1,
    )
    agg["rss_ratio_vs_fastapriori_fast_median"] = agg.apply(
        lambda r: r["peak_rss_kb_median"] / _lookup((r["dataset"], r["k"], r["min_support"]), "peak_rss_kb_median")
        if r["peak_rss_kb_median"] and r["peak_rss_kb_median"] == r["peak_rss_kb_median"] else float("nan"),
        axis=1,
    )

    # Winner per (dataset, k, min_support) = min median wall_s among status=ok
    def _winner(sub):
        ok = sub[sub["wall_s_n_ok"] > 0]
        if len(ok) == 0:
            return "none"
        return ok.loc[ok["wall_s_median"].idxmin(), "method"]

    winners = (
        agg.groupby(["dataset", "k", "min_support"])
        .apply(_winner)
        .rename("winner_overall")
        .reset_index()
    )
    agg = agg.merge(winners, on=["dataset", "k", "min_support"], how="left")

    # Rank wall + rss within each (dataset, k, min_support) group
    agg["rank_wall"] = agg.groupby(["dataset", "k", "min_support"])["wall_s_median"].rank()
    agg["rank_rss"] = agg.groupby(["dataset", "k", "min_support"])["peak_rss_kb_median"].rank()

    agg = agg.drop(columns=["_key"])
    return agg


def save_aggregate(agg: pd.DataFrame, path: Path = AGG_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(path, index=False)


__all__ = [
    "ALL_METHODS", "RAW_COLUMNS", "RUNS_BY_SIZE_CLASS",
    "RAW_CSV", "AGG_CSV", "WORKDIR", "TOOLS_DIR", "TOOLS_BIN", "SPMF_JAR",
    "set_raw_csv_suffix",
    "collect_environment", "detect_tools", "dataset_stats",
    "df_to_spmf_file", "parse_time_output", "classify_size",
    "run_method", "measure",
    "append_raw_row", "load_raw", "already_done",
    "aggregate", "save_aggregate",
]
