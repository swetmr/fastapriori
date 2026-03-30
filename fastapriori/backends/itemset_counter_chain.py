"""Counter+chain backend for k-itemset computation.

Extends the fast Counter+chain pattern (used for k=2 in pandas_backend.py)
to k=3+ by anchoring on frequent (k-1)-sets and using Counter(chain(...))
to discover the k-th item.

Algorithm (for k=3 as example):
1. Extract frequent pairs from the k=2 results
2. Build pair -> set(transaction_ids) mapping (only frequent pairs)
3. For each pair (A,B), Counter(chain(*(trans_dict[t] for t in txn_ids)))
   counts how many of those transactions also contain each other item C
4. Record canonical triplet (A,B,C) with the count
5. De-duplicate: each triplet is found from 3 anchors; first-write-wins

Generalises to k=4 (anchor on triplets), k=5 (anchor on quadruplets), etc.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from itertools import combinations, chain
from multiprocessing import Pool

import pandas as pd


# ---------------------------------------------------------------------------
# Progress helper (shared pattern with pandas_backend / itemsets)
# ---------------------------------------------------------------------------

def _wrap_progress(iterable, total: int | None = None, desc: str | None = None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        print(f"{desc}: processing {total} items (install tqdm for progress bar)...")
        return iterable


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_itemsets(
    trans_dict: dict,
    total_transactions: int,
    k: int,
    frequent_lower: pd.DataFrame | None,
    n_workers: int | None,
    show_progress: bool,
) -> Counter:
    """Compute k-itemset counts using Counter+chain anchored on (k-1)-sets.

    Parameters
    ----------
    trans_dict : dict
        Mapping transaction_id -> set(items).
    total_transactions : int
        Total number of transactions.
    k : int
        Target itemset size (>=2).
    frequent_lower : pd.DataFrame or None
        Results from the (k-1) level. Used to determine which (k-1)-sets
        are frequent (Apriori pruning).
    n_workers : int or None
        Number of parallel workers. None = auto, 1 = serial.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    Counter
        Mapping canonical k-tuple -> co-occurrence count.
    """
    # --- Step 1: extract frequent (k-1)-sets for filtering ---
    freq_lower_sets: set | None = None
    if frequent_lower is not None and k >= 3:
        freq_lower_sets = _extract_frequent_lower_sets(frequent_lower, k)

    # Also extract the set of ALL frequent pairs for full Apriori pruning.
    # When anchoring on (k-1)-set and finding a new item, we must verify that
    # ALL C(k,2) pairs in the resulting k-set are frequent, not just the anchor.
    freq_pair_set: set | None = None
    if freq_lower_sets is not None and k >= 3:
        freq_pair_set = _extract_all_freq_pairs(frequent_lower)

    # --- Step 2: build (k-1)-set -> set(transaction_ids) mapping ---
    # Convert transaction IDs to integers for memory efficiency
    txn_to_int = {txn: i for i, txn in enumerate(trans_dict)}
    int_to_txn = {i: txn for txn, i in txn_to_int.items()}
    int_trans_dict = {txn_to_int[txn]: items for txn, items in trans_dict.items()}

    lower_to_txns = _build_lower_to_txns(
        int_trans_dict, k, freq_lower_sets
    )

    if not lower_to_txns:
        return Counter()

    # --- Step 3: Counter+chain loop (serial or parallel) ---
    effective_workers = _resolve_workers(n_workers, len(lower_to_txns))

    if effective_workers <= 1:
        itemset_counts = _counter_chain_serial(
            lower_to_txns, int_trans_dict, k, freq_pair_set, show_progress
        )
    else:
        itemset_counts = _counter_chain_parallel(
            lower_to_txns, int_trans_dict, k, freq_pair_set,
            effective_workers, show_progress
        )

    return itemset_counts


# ---------------------------------------------------------------------------
# Step 1: Extract frequent (k-1)-sets
# ---------------------------------------------------------------------------

def _extract_all_freq_pairs(frequent_lower: pd.DataFrame) -> set:
    """Extract ALL frequent pairs from any level of results.

    For pair-level input (item_A/item_B columns), extracts pairs directly.
    For higher-level input (antecedent_*/consequent), extracts all C(n,2)
    pairs from each itemset.
    """
    result: set = set()

    if "item_A" in frequent_lower.columns and "item_B" in frequent_lower.columns:
        for a, b in zip(frequent_lower["item_A"], frequent_lower["item_B"]):
            result.add(tuple(sorted([a, b])))
        return result

    ant_cols = [c for c in frequent_lower.columns if c.startswith("antecedent_")]
    if ant_cols and "consequent" in frequent_lower.columns:
        for _, row in frequent_lower[ant_cols + ["consequent"]].iterrows():
            items = [row[c] for c in ant_cols] + [row["consequent"]]
            for pair in combinations(sorted(items), 2):
                result.add(pair)
        return result

    return result


def _extract_frequent_lower_sets(frequent_lower: pd.DataFrame, k: int) -> set:
    """Extract canonical (k-1)-tuples from the lower-level DataFrame."""
    result: set = set()

    # Pair-level columns (output of find_associations)
    if "item_A" in frequent_lower.columns and "item_B" in frequent_lower.columns:
        for a, b in zip(frequent_lower["item_A"], frequent_lower["item_B"]):
            result.add(tuple(sorted([a, b])))
        return result

    # Higher-level: antecedent_* + consequent columns
    ant_cols = [c for c in frequent_lower.columns if c.startswith("antecedent_")]
    if ant_cols and "consequent" in frequent_lower.columns:
        for _, row in frequent_lower[ant_cols + ["consequent"]].iterrows():
            items = tuple(sorted([row[c] for c in ant_cols] + [row["consequent"]]))
            result.add(items)
        return result

    return result


# ---------------------------------------------------------------------------
# Step 2: Build (k-1)-set -> transaction_ids mapping
# ---------------------------------------------------------------------------

def _build_lower_to_txns(
    trans_dict: dict, k: int, freq_lower_sets: set | None
) -> dict:
    """Build mapping from frequent (k-1)-sets to their transaction IDs.

    Only (k-1)-sets that appear in freq_lower_sets are kept (Apriori pruning).
    For k=2, all single items are used (no pruning needed).
    """
    lower_k = k - 1
    lower_to_txns: dict[tuple, list] = defaultdict(list)

    for txn_id, items in trans_dict.items():
        if len(items) < k:
            continue
        sorted_items = sorted(items)
        if lower_k == 1:
            # k=2: anchor on single items
            for item in sorted_items:
                key = (item,)
                if freq_lower_sets is None or key in freq_lower_sets:
                    lower_to_txns[key].append(txn_id)
        else:
            for subset in combinations(sorted_items, lower_k):
                if freq_lower_sets is None or subset in freq_lower_sets:
                    lower_to_txns[subset].append(txn_id)

    return dict(lower_to_txns)


# ---------------------------------------------------------------------------
# Step 3a: Serial Counter+chain
# ---------------------------------------------------------------------------

def _apriori_check_new_item(lower_set, item, freq_pair_set):
    """Check that adding `item` to `lower_set` keeps all pairs frequent.

    The anchor (k-1)-set's own pairs are already known to be frequent,
    so we only check pairs that include the new item.
    """
    for existing in lower_set:
        pair = (existing, item) if existing < item else (item, existing)
        if pair not in freq_pair_set:
            return False
    return True


def _counter_chain_serial(
    lower_to_txns: dict, trans_dict: dict, k: int,
    freq_pair_set: set | None, show_progress: bool
) -> Counter:
    """Serial Counter+chain over (k-1)-sets."""
    itemset_counts: dict = {}

    iterator = lower_to_txns.items()
    if show_progress:
        iterator = _wrap_progress(
            iterator, total=len(lower_to_txns), desc=f"Counter+chain k={k}"
        )

    for lower_set, txn_ids in iterator:
        lower_items = set(lower_set)
        counter = Counter(chain(*(trans_dict[t] for t in txn_ids)))

        for item, count in counter.items():
            if item in lower_items:
                continue
            canonical = tuple(sorted(lower_set + (item,)))
            if canonical in itemset_counts:
                continue
            # Apriori: only check pairs involving the new item
            # (anchor's own pairs are already frequent by construction)
            if freq_pair_set is not None:
                if not _apriori_check_new_item(lower_set, item, freq_pair_set):
                    continue
            itemset_counts[canonical] = count

    return Counter(itemset_counts)


# ---------------------------------------------------------------------------
# Step 3b: Parallel Counter+chain with joblib
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 3b: Parallel Counter+chain with multiprocessing.Pool
# ---------------------------------------------------------------------------

# Module-level shared state for worker processes.
# Pool(initializer=...) populates these once per worker, avoiding
# per-task pickling of large dicts.
_shared: dict = {}


def _init_worker(trans_dict, lower_to_txns, freq_pair_set, k):
    """Initializer called once per worker process."""
    _shared["trans_dict"] = trans_dict
    _shared["lower_to_txns"] = lower_to_txns
    _shared["freq_pair_set"] = freq_pair_set
    _shared["k"] = k


def _worker_fn(chunk_keys):
    """Worker function: process a chunk of (k-1)-set keys using shared data."""
    trans_dict = _shared["trans_dict"]
    lower_to_txns = _shared["lower_to_txns"]
    freq_pair_set = _shared["freq_pair_set"]
    k = _shared["k"]

    result = {}
    for lower_set in chunk_keys:
        txn_ids = lower_to_txns[lower_set]
        lower_items = set(lower_set)
        counter = Counter(chain(*(trans_dict[t] for t in txn_ids)))

        for item, count in counter.items():
            if item in lower_items:
                continue
            canonical = tuple(sorted(lower_set + (item,)))
            if canonical in result:
                continue
            if freq_pair_set is not None:
                if not _apriori_check_new_item(lower_set, item, freq_pair_set):
                    continue
            result[canonical] = count

    return result


def _counter_chain_parallel(
    lower_to_txns: dict, trans_dict: dict, k: int,
    freq_pair_set: set | None, n_workers: int, show_progress: bool
) -> Counter:
    """Parallel Counter+chain using multiprocessing.Pool with initializer.

    Large data (trans_dict, lower_to_txns, freq_pair_set) is sent to each
    worker once at pool creation via the initializer, not per-task.
    Each task only sends a small list of (k-1)-set keys.
    """
    keys = list(lower_to_txns.keys())
    chunk_size = max(1, len(keys) // n_workers)
    chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]

    if show_progress:
        print(f"Counter+chain k={k}: {len(keys)} (k-1)-sets across {len(chunks)} workers")

    with Pool(
        n_workers,
        initializer=_init_worker,
        initargs=(trans_dict, lower_to_txns, freq_pair_set, k),
    ) as pool:
        results = pool.map(_worker_fn, chunks)

    # Merge results (first-write-wins for de-duplication)
    combined: dict = {}
    for partial in results:
        for key, val in partial.items():
            if key not in combined:
                combined[key] = val

    return Counter(combined)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_workers(n_workers: int | None, n_lower_sets: int) -> int:
    """Determine effective number of workers.

    Auto-tuning: only parallelise when workload is large enough to
    amortise the overhead of spawning workers.
    """
    if n_workers is not None:
        return max(1, n_workers)

    # Auto: parallelise only when workload is large enough to amortise
    # the overhead of spawning workers (especially on Windows where
    # ProcessPoolExecutor uses spawn, requiring data pickling).
    if n_lower_sets < 5000:
        return 1

    cpu = os.cpu_count() or 1
    return min(cpu, 4)
