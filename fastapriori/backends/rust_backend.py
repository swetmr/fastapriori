"""Rust backend for fastapriori — PyO3 extension wrapping compiled Rust code."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


try:
    from fastapriori._fastapriori_rs import (
        rust_compute_pairs,
        rust_compute_itemsets,
        rust_compute_pipeline,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def _check_rust():
    if not RUST_AVAILABLE:
        raise ImportError(
            "Rust backend not available. Build with: maturin develop --release"
        )


# ---------------------------------------------------------------------------
# k=2: compute_associations (same signature as pandas/polars backends)
# ---------------------------------------------------------------------------

def compute_associations(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    show_progress: bool,
    trans_dict: dict | None = None,
) -> pd.DataFrame:
    """Compute pairwise associations using the Rust backend."""
    _check_rust()

    clean = df[[transaction_col, item_col]].dropna()

    # Encode items to sequential integers
    unique_items = sorted(clean[item_col].unique())
    item_encoder = {item: i for i, item in enumerate(unique_items)}
    item_decoder = np.array(unique_items)

    txn_codes, _ = pd.factorize(clean[transaction_col])
    txn_ids = txn_codes.astype(np.int64)
    item_ids = clean[item_col].map(item_encoder).to_numpy(dtype=np.int32)
    n_transactions = int(clean[transaction_col].nunique())

    # Call Rust
    result_dict = rust_compute_pairs(
        txn_ids, item_ids, len(unique_items), n_transactions
    )

    # Decode item IDs back to original labels
    result = pd.DataFrame({
        "item_A": item_decoder[result_dict["item_A"]],
        "item_B": item_decoder[result_dict["item_B"]],
        "instances": result_dict["instances"],
        "support": result_dict["support"],
        "confidence": result_dict["confidence"],
        "lift": result_dict["lift"],
        "conviction": result_dict["conviction"],
        "leverage": result_dict["leverage"],
        "cosine": result_dict["cosine"],
        "jaccard": result_dict["jaccard"],
    })
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# k>=3: compute_itemsets_rust (same return type as counter_chain backend)
# ---------------------------------------------------------------------------

def compute_itemsets_rust(
    trans_dict: dict,
    total_transactions: int,
    k: int,
    frequent_lower: pd.DataFrame | None,
    n_workers: int | None,
    show_progress: bool,
    df: pd.DataFrame | None = None,
    transaction_col: str | None = None,
    item_col: str | None = None,
) -> Counter:
    """Compute k-itemset counts using the Rust backend.

    Returns Counter mapping canonical k-tuple -> count (same as
    itemset_counter_chain.compute_itemsets).
    """
    _check_rust()

    # --- Encode items to integers ---
    # Use DataFrame directly if available (avoids expensive trans_dict flatten)
    if df is not None and transaction_col and item_col:
        clean = df[[transaction_col, item_col]].dropna()
        unique_items = sorted(clean[item_col].unique())
        item_encoder = {item: i for i, item in enumerate(unique_items)}
        item_decoder = {i: item for item, i in item_encoder.items()}
        txn_codes, _ = pd.factorize(clean[transaction_col])
        txn_ids = txn_codes.astype(np.int64)
        item_ids = clean[item_col].map(item_encoder).to_numpy(dtype=np.int32)
    else:
        all_items: set = set()
        for items in trans_dict.values():
            all_items.update(items)
        unique_items = sorted(all_items)
        item_encoder = {item: i for i, item in enumerate(unique_items)}
        item_decoder = {i: item for item, i in item_encoder.items()}

        txn_list = []
        item_list = []
        for txn_id, items in trans_dict.items():
            for item in items:
                txn_list.append(txn_id)
                item_list.append(item_encoder[item])
        txn_ids = np.array(txn_list, dtype=np.int64)
        item_ids = np.array(item_list, dtype=np.int32)

    # --- Extract frequent (k-1)-sets as 2D int array ---
    freq_lower_sets = _extract_freq_lower_encoded(frequent_lower, k, item_encoder)

    # Call Rust
    result_dict = rust_compute_itemsets(
        txn_ids, item_ids, freq_lower_sets, k, len(unique_items)
    )

    # Decode integer itemsets back to original labels
    itemsets_arr = result_dict["itemsets"]  # 2D numpy array (n_results, k)
    counts_arr = result_dict["counts"]     # 1D numpy array

    result = Counter()
    for i in range(len(counts_arr)):
        kset = tuple(item_decoder[int(itemsets_arr[i, j])] for j in range(k))
        result[kset] = int(counts_arr[i])

    return result


def _extract_freq_lower_encoded(
    frequent_lower: pd.DataFrame | None,
    k: int,
    item_encoder: dict,
) -> np.ndarray:
    """Extract frequent (k-1)-sets from DataFrame, encode to int32 2D array."""
    lower_k = k - 1

    if frequent_lower is None:
        return np.empty((0, lower_k), dtype=np.int32)

    sets = []

    # Pair-level columns (from k=2 output)
    if "item_A" in frequent_lower.columns and "item_B" in frequent_lower.columns:
        seen = set()
        for a, b in zip(frequent_lower["item_A"], frequent_lower["item_B"]):
            canonical = tuple(sorted([a, b]))
            if canonical not in seen:
                seen.add(canonical)
                encoded = tuple(item_encoder[x] for x in canonical)
                sets.append(encoded)
    else:
        # Higher-level: antecedent_* + consequent columns
        ant_cols = [c for c in frequent_lower.columns if c.startswith("antecedent_")]
        if ant_cols and "consequent" in frequent_lower.columns:
            seen = set()
            for _, row in frequent_lower[ant_cols + ["consequent"]].iterrows():
                items = tuple(sorted(
                    [row[c] for c in ant_cols] + [row["consequent"]]
                ))
                if items not in seen:
                    seen.add(items)
                    encoded = tuple(item_encoder[x] for x in items)
                    sets.append(encoded)

    if not sets:
        return np.empty((0, lower_k), dtype=np.int32)

    return np.array(sets, dtype=np.int32)


# ---------------------------------------------------------------------------
# Full pipeline: k=2 → k=3 → ... → k_max in one Rust call
# ---------------------------------------------------------------------------

def compute_pipeline(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    k: int,
    min_support: float,
    min_confidence: float | None = None,
) -> pd.DataFrame:
    """Full pipeline: encode once, Rust computes k=2..k_max, decode once.

    Returns the same DataFrame schema as core._find_k_itemsets:
    antecedent_1, ..., antecedent_{k-1}, consequent, instances, support,
    confidence, lift.
    """
    _check_rust()

    clean = df[[transaction_col, item_col]].dropna()
    unique_items = sorted(clean[item_col].unique())
    item_encoder = {item: i for i, item in enumerate(unique_items)}
    item_decoder = np.array(unique_items)

    txn_codes, _ = pd.factorize(clean[transaction_col])
    txn_ids = txn_codes.astype(np.int64)
    item_ids = clean[item_col].map(item_encoder).to_numpy(dtype=np.int32)
    n_transactions = int(clean[transaction_col].nunique())

    result_dict = rust_compute_pipeline(
        txn_ids, item_ids, k, len(unique_items), n_transactions, min_support,
    )

    # Unpack results
    itemsets_arr = result_dict["itemsets"]       # 2D int32 (n_results, k)
    counts_arr = result_dict["counts"]           # 1D int64
    lower_arr = result_dict["lower_itemsets"]    # 2D int32 (n_lower, k-1)
    lower_counts_arr = result_dict["lower_counts"]  # 1D int64
    item_counts_arr = result_dict["item_counts"]    # 1D int64 (n_items,)

    # Build output column names
    ant_cols = [f"antecedent_{i}" for i in range(1, k)]
    out_cols = ant_cols + ["consequent", "instances", "support", "confidence", "lift"]

    if len(counts_arr) == 0:
        return pd.DataFrame(columns=out_cols)

    # Build lower_support dict: (k-1)-tuple (decoded) -> support
    lower_support = {}
    for i in range(len(lower_counts_arr)):
        key = tuple(item_decoder[int(lower_arr[i, j])] for j in range(k - 1))
        lower_support[key] = float(lower_counts_arr[i]) / n_transactions

    # Build item_support: item (decoded) -> support
    item_support = {}
    for idx in range(len(item_counts_arr)):
        if item_counts_arr[idx] > 0:
            item_support[item_decoder[idx]] = float(item_counts_arr[idx]) / n_transactions

    # Generate k directional rules per itemset
    records = []
    for i in range(len(counts_arr)):
        itemset = tuple(item_decoder[int(itemsets_arr[i, j])] for j in range(k))
        count = int(counts_arr[i])
        support = count / n_transactions

        for j in range(k):
            consequent = itemset[j]
            antecedents = itemset[:j] + itemset[j + 1:]
            ant_key = tuple(sorted(antecedents))
            ant_sup = lower_support.get(ant_key, 0)
            confidence = support / (ant_sup + 1e-10)
            cons_sup = item_support.get(consequent, 0)
            lift = confidence / (cons_sup + 1e-10)
            records.append(
                (*antecedents, consequent, count, support, confidence, lift)
            )

    result = pd.DataFrame(records, columns=out_cols)

    if min_support is not None and min_support > 0:
        result = result[result["support"] >= min_support]
    if min_confidence is not None:
        result = result[result["confidence"] >= min_confidence]

    result["support"] = np.round(result["support"], 6)
    result["confidence"] = np.round(result["confidence"], 6)
    result["lift"] = np.round(result["lift"], 6)

    return result.reset_index(drop=True)
