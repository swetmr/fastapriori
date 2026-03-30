"""Rust classic Apriori backend — PyO3 port of efficient-apriori algorithm."""

from __future__ import annotations

import numpy as np
import pandas as pd


try:
    from fastapriori._fastapriori_rs import (
        rust_classic_compute_pairs,
        rust_classic_compute_pipeline,
    )
    RUST_CLASSIC_AVAILABLE = True
except ImportError:
    RUST_CLASSIC_AVAILABLE = False


def _check_rust_classic():
    if not RUST_CLASSIC_AVAILABLE:
        raise ImportError(
            "Rust classic backend not available. Build with: maturin develop --release"
        )


# ---------------------------------------------------------------------------
# k=2: compute_associations (same signature as other backends)
# ---------------------------------------------------------------------------

def compute_associations(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    show_progress: bool,
    min_support: float,
    trans_dict: dict | None = None,
) -> pd.DataFrame:
    """Compute pairwise associations using the Rust classic Apriori backend.

    Unlike the fast backend, classic Apriori requires min_support upfront
    for candidate pruning.
    """
    _check_rust_classic()

    clean = df[[transaction_col, item_col]].dropna()

    # Encode items to sequential integers
    unique_items = sorted(clean[item_col].unique())
    item_encoder = {item: i for i, item in enumerate(unique_items)}
    item_decoder = np.array(unique_items)

    txn_codes, _ = pd.factorize(clean[transaction_col])
    txn_ids = txn_codes.astype(np.int64)
    item_ids = clean[item_col].map(item_encoder).to_numpy(dtype=np.int32)
    n_transactions = int(clean[transaction_col].nunique())

    # Call Rust classic
    result_dict = rust_classic_compute_pairs(
        txn_ids, item_ids, len(unique_items), n_transactions, min_support,
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
# Full pipeline: k=2 -> k=3 -> ... -> k_max in one Rust call
# ---------------------------------------------------------------------------

def compute_pipeline(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    k: int,
    min_support: float,
    min_confidence: float | None = None,
) -> pd.DataFrame:
    """Full classic Apriori pipeline: encode once, Rust computes k=1..k_max, decode once.

    Returns the same DataFrame schema as core._find_k_itemsets:
    antecedent_1, ..., antecedent_{k-1}, consequent, instances, support,
    confidence, lift.
    """
    _check_rust_classic()

    clean = df[[transaction_col, item_col]].dropna()
    unique_items = sorted(clean[item_col].unique())
    item_encoder = {item: i for i, item in enumerate(unique_items)}
    item_decoder = np.array(unique_items)

    txn_codes, _ = pd.factorize(clean[transaction_col])
    txn_ids = txn_codes.astype(np.int64)
    item_ids = clean[item_col].map(item_encoder).to_numpy(dtype=np.int32)
    n_transactions = int(clean[transaction_col].nunique())

    result_dict = rust_classic_compute_pipeline(
        txn_ids, item_ids, k, len(unique_items), n_transactions, min_support,
    )

    # Unpack results
    itemsets_arr = result_dict["itemsets"]          # 2D int32 (n_results, k)
    counts_arr = result_dict["counts"]              # 1D int64
    lower_arr = result_dict["lower_itemsets"]       # 2D int32 (n_lower, k-1)
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
