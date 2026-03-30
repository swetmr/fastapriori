"""bin_multi backend for k-itemset computation.

Uses numpy.bincount for vectorized counting (replaces Counter+chain).

Batches many (k-1)-sets into a single bulk numpy operation: one CSR gather +
one bincount on millions of elements per batch. This is ~2x faster than the
Counter+chain approach used in the counter_chain backend.

Threading was tested but provides no benefit: the Python post-processing
(tuple creation, dict lookups, Apriori checks) that follows each bincount
batch is ~80% of total time and GIL-held. Threading is still supported
(n_workers > 1) but defaults to serial.
"""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fastapriori.backends.itemset_counter_chain import (
    _build_lower_to_txns,
    _extract_all_freq_pairs,
    _extract_frequent_lower_sets,
    _wrap_progress,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_itemsets_bin_multi(
    trans_dict: dict,
    total_transactions: int,
    k: int,
    frequent_lower=None,
    n_workers: int | None = None,
    show_progress: bool = False,
) -> Counter:
    """Compute k-itemset counts using batched numpy bincount + threads.

    Parameters
    ----------
    trans_dict : dict
        Mapping transaction_id -> set(items).
    total_transactions : int
        Total number of transactions.
    k : int
        Target itemset size (>=2).
    frequent_lower : pd.DataFrame or None
        Results from the (k-1) level for Apriori pruning.
    n_workers : int or None
        Number of threads. None = auto, 1 = serial.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    Counter
        Mapping canonical k-tuple -> co-occurrence count.
    """
    # --- Extract frequent (k-1)-sets for Apriori pruning ---
    freq_lower_sets: set | None = None
    if frequent_lower is not None and k >= 3:
        freq_lower_sets = _extract_frequent_lower_sets(frequent_lower, k)

    freq_pair_set: set | None = None
    if freq_lower_sets is not None and k >= 3:
        freq_pair_set = _extract_all_freq_pairs(frequent_lower)

    # --- Encode items as integers ---
    all_items: set = set()
    for items in trans_dict.values():
        all_items.update(items)
    item_to_int = {item: i for i, item in enumerate(sorted(all_items))}
    int_to_item = {i: item for item, i in item_to_int.items()}
    n_items = len(item_to_int)

    # Convert transaction IDs to sequential ints
    txn_to_int = {txn: i for i, txn in enumerate(trans_dict)}
    int_trans_dict = {txn_to_int[txn]: items for txn, items in trans_dict.items()}

    # --- Build CSR-like flat structure ---
    n_txns = len(int_trans_dict)
    offsets = np.zeros(n_txns + 1, dtype=np.int64)
    for txn_id in range(n_txns):
        offsets[txn_id + 1] = offsets[txn_id] + len(int_trans_dict[txn_id])

    all_items_flat = np.empty(int(offsets[-1]), dtype=np.int32)
    for txn_id in range(n_txns):
        items = int_trans_dict[txn_id]
        start = int(offsets[txn_id])
        for j, item in enumerate(sorted(items)):
            all_items_flat[start + j] = item_to_int[item]

    # --- Build (k-1)-set -> transaction_ids mapping ---
    lower_to_txns = _build_lower_to_txns(int_trans_dict, k, freq_lower_sets)

    if not lower_to_txns:
        return Counter()

    # Pre-convert txn_id lists to numpy arrays
    lower_to_txns_np = {
        key: np.array(txn_ids, dtype=np.int64)
        for key, txn_ids in lower_to_txns.items()
    }

    # Ordered keys for batch processing
    keys_list = list(lower_to_txns_np.keys())

    # --- Resolve worker count ---
    effective_workers = _resolve_workers(n_workers, len(keys_list))

    # --- Count itemsets ---
    if effective_workers <= 1:
        itemset_counts = _batched_bincount_serial(
            keys_list, lower_to_txns_np, all_items_flat, offsets,
            n_items, int_to_item, k, freq_pair_set, show_progress,
        )
    else:
        itemset_counts = _batched_bincount_threaded(
            keys_list, lower_to_txns_np, all_items_flat, offsets,
            n_items, int_to_item, k, freq_pair_set,
            effective_workers, show_progress,
        )

    return itemset_counts


# ---------------------------------------------------------------------------
# Bulk gather + bincount for a batch of (k-1)-sets
# ---------------------------------------------------------------------------

def _bulk_gather_and_count(
    batch_keys: list,
    lower_to_txns: dict,
    all_items_flat: np.ndarray,
    offsets: np.ndarray,
    n_items: int,
) -> np.ndarray:
    """Gather items for a batch of (k-1)-sets and count via ONE bincount.

    Returns a 2D array (n_batch × n_items) of counts.

    All heavy numpy operations are on large arrays (millions of elements),
    ensuring the GIL is effectively released.
    """
    n_batch = len(batch_keys)

    # 1) Flatten all txn_ids across the batch, tracking group membership
    txn_id_lists = [lower_to_txns[key] for key in batch_keys]
    group_sizes = np.array([len(t) for t in txn_id_lists], dtype=np.int64)
    all_txn_ids = np.concatenate(txn_id_lists)  # large array

    # 2) CSR gather: look up item offsets for each txn
    starts = offsets[all_txn_ids]
    lengths = offsets[all_txn_ids + 1] - starts

    # 3) Build flat gather indices
    total_items = int(lengths.sum())
    if total_items == 0:
        return np.zeros((n_batch, n_items), dtype=np.int64)

    base = np.repeat(starts, lengths)
    cumlen = np.cumsum(lengths)
    within = np.arange(total_items, dtype=np.int64) - np.repeat(
        cumlen - lengths, lengths
    )
    flat_idx = base + within
    gathered_items = all_items_flat[flat_idx]

    # 4) Compute group ID for each gathered item
    #    Each txn belongs to a group (which (k-1)-set), expand to item level
    txn_group_ids = np.repeat(np.arange(n_batch, dtype=np.int64), group_sizes)
    item_group_ids = np.repeat(txn_group_ids, lengths)

    # 5) Single bulk bincount using linear indexing: group * n_items + item
    flat_count_idx = item_group_ids * n_items + gathered_items.astype(np.int64)
    flat_counts = np.bincount(flat_count_idx, minlength=n_batch * n_items)
    count_matrix = flat_counts.reshape(n_batch, n_items)

    return count_matrix


# ---------------------------------------------------------------------------
# Post-processing: extract itemsets from count matrix
# ---------------------------------------------------------------------------

def _apriori_check_new_item(lower_set, item, freq_pair_set):
    """Check that adding `item` to `lower_set` keeps all pairs frequent."""
    for existing in lower_set:
        pair = (existing, item) if existing < item else (item, existing)
        if pair not in freq_pair_set:
            return False
    return True


def _process_count_matrix(
    batch_keys: list,
    count_matrix: np.ndarray,
    int_to_item: dict[int, str],
    k: int,
    freq_pair_set: set | None,
) -> dict:
    """Extract canonical itemsets from a batch count matrix."""
    result: dict = {}

    for i, lower_set in enumerate(batch_keys):
        lower_items = set(lower_set)
        counts = count_matrix[i]
        nonzero_idx = np.flatnonzero(counts)

        for idx in nonzero_idx:
            item = int_to_item[int(idx)]
            if item in lower_items:
                continue
            canonical = tuple(sorted(lower_set + (item,)))
            if canonical in result:
                continue
            if freq_pair_set is not None:
                if not _apriori_check_new_item(lower_set, item, freq_pair_set):
                    continue
            result[canonical] = int(counts[idx])

    return result


# ---------------------------------------------------------------------------
# Serial batched bincount
# ---------------------------------------------------------------------------

_BATCH_SIZE = 500  # (k-1)-sets per batch


def _batched_bincount_serial(
    keys_list: list,
    lower_to_txns: dict,
    all_items_flat: np.ndarray,
    offsets: np.ndarray,
    n_items: int,
    int_to_item: dict[int, str],
    k: int,
    freq_pair_set: set | None,
    show_progress: bool,
) -> Counter:
    """Serial batched bincount."""
    itemset_counts: dict = {}
    n_keys = len(keys_list)
    batches = list(range(0, n_keys, _BATCH_SIZE))

    iterator = batches
    if show_progress:
        iterator = _wrap_progress(
            iterator, total=len(batches),
            desc=f"bin_multi k={k} (batch={_BATCH_SIZE})",
        )

    for batch_start in iterator:
        batch_keys = keys_list[batch_start:batch_start + _BATCH_SIZE]

        count_matrix = _bulk_gather_and_count(
            batch_keys, lower_to_txns, all_items_flat, offsets, n_items,
        )

        partial = _process_count_matrix(
            batch_keys, count_matrix, int_to_item, k, freq_pair_set,
        )

        for key, val in partial.items():
            if key not in itemset_counts:
                itemset_counts[key] = val

    return Counter(itemset_counts)


# ---------------------------------------------------------------------------
# Threaded batched bincount
# ---------------------------------------------------------------------------

def _thread_worker_batched(
    chunk_keys: list,
    lower_to_txns: dict,
    all_items_flat: np.ndarray,
    offsets: np.ndarray,
    n_items: int,
    int_to_item: dict[int, str],
    k: int,
    freq_pair_set: set | None,
) -> dict:
    """Process a chunk of (k-1)-sets using batched bulk bincount in a thread.

    Each batch produces arrays large enough for numpy to release the GIL,
    enabling true parallel execution.
    """
    result: dict = {}

    for batch_start in range(0, len(chunk_keys), _BATCH_SIZE):
        batch_keys = chunk_keys[batch_start:batch_start + _BATCH_SIZE]

        count_matrix = _bulk_gather_and_count(
            batch_keys, lower_to_txns, all_items_flat, offsets, n_items,
        )

        partial = _process_count_matrix(
            batch_keys, count_matrix, int_to_item, k, freq_pair_set,
        )

        for key, val in partial.items():
            if key not in result:
                result[key] = val

    return result


def _batched_bincount_threaded(
    keys_list: list,
    lower_to_txns: dict,
    all_items_flat: np.ndarray,
    offsets: np.ndarray,
    n_items: int,
    int_to_item: dict[int, str],
    k: int,
    freq_pair_set: set | None,
    n_workers: int,
    show_progress: bool,
) -> Counter:
    """Threaded batched bincount using ThreadPoolExecutor.

    Each thread processes a chunk of (k-1)-sets using batched bulk bincount.
    numpy operations on large arrays release the GIL for true parallelism.
    """
    chunk_size = max(1, len(keys_list) // n_workers)
    chunks = [
        keys_list[i:i + chunk_size]
        for i in range(0, len(keys_list), chunk_size)
    ]

    if show_progress:
        print(
            f"bin_multi k={k}: {len(keys_list)} (k-1)-sets across "
            f"{len(chunks)} threads (batch={_BATCH_SIZE})"
        )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                _thread_worker_batched, chunk, lower_to_txns,
                all_items_flat, offsets, n_items, int_to_item,
                k, freq_pair_set,
            )
            for chunk in chunks
        ]
        results = [f.result() for f in futures]

    # Merge results (first-write-wins)
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
    """Determine effective number of threads.

    Auto defaults to 1 (serial) because benchmarks show threading provides
    no benefit — the GIL-held Python post-processing dominates (~80% of time).
    Users can still force n_workers > 1 to test on different platforms.
    """
    if n_workers is not None:
        return max(1, n_workers)
    return 1
