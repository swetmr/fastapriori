"""Rust classic Apriori backend — PyO3 port of efficient-apriori algorithm."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fastapriori.backends.rust_backend import (
    _check_encoder_capacity,
    _sorted_unique_items,
)


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
    unique_items = _sorted_unique_items(clean[item_col])
    _check_encoder_capacity(len(unique_items))
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
    max_items_per_txn: int | None = None,
    item_weights: dict | None = None,
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

    n_unique = len(unique_items)
    if item_weights is not None:
        weights_arr = np.zeros(n_unique, dtype=np.float64)
        for item, weight in item_weights.items():
            if item in item_encoder:
                w = float(weight)
                weights_arr[item_encoder[item]] = 0.0 if np.isnan(w) else w
    else:
        weights_arr = np.zeros(n_unique, dtype=np.float64)
        item_counts_local = clean.groupby(item_col)[transaction_col].nunique()
        for item, count in item_counts_local.items():
            if item in item_encoder:
                weights_arr[item_encoder[item]] = float(count)

    result_dict = rust_classic_compute_pipeline(
        txn_ids, item_ids, k, n_unique, n_transactions, min_support,
        weights_arr, max_items_per_txn,
    )

    # Unpack results
    itemsets_arr = result_dict["itemsets"]          # 2D int32 (n_results, k)
    counts_arr = result_dict["counts"]              # 1D int64
    lower_arr = result_dict["lower_itemsets"]       # 2D int32 (n_lower, k-1)
    lower_counts_arr = result_dict["lower_counts"]  # 1D int64
    item_counts_arr = result_dict["item_counts"]    # 1D int64 (n_items,)

    from fastapriori.backends.rust_backend import decode_pipeline_rules

    return decode_pipeline_rules(
        itemsets_arr, counts_arr, lower_arr, lower_counts_arr,
        item_counts_arr, item_decoder, n_transactions, k,
        min_support, min_confidence,
    )
