"""Core algorithm for fast pairwise co-occurrence analysis."""

from __future__ import annotations

from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd


def _wrap_progress(iterable, total: int | None = None, desc: str | None = None):
    """Wrap an iterable with tqdm if available, else a simple print fallback."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        print(f"{desc}: processing {total} items (install tqdm for progress bar)...")
        return iterable


def find_associations(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    min_support: float | None = None,
    min_confidence: float | None = 0.0,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute pairwise item co-occurrence associations from transactional data.

    For every item A, counts how often each other item B appears in the same
    transactions, then computes support, confidence, and lift for each (A, B) pair.

    Uses a Counter+chain approach over grouped sets — significantly faster than
    traditional candidate-generation Apriori for large datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with at least two columns: one for transaction IDs and one
        for item identifiers. Each row represents one (transaction, item) pair.
    transaction_col : str
        Name of the column containing transaction/group identifiers.
    item_col : str
        Name of the column containing item identifiers.
    min_support : float or None
        Minimum support threshold. Pairs with support below this are removed.
        None (default) disables support filtering.
    min_confidence : float or None
        Minimum confidence threshold. Pairs with confidence below this are
        removed. Default is 0.1 (10%). None disables confidence filtering.
    show_progress : bool
        If True, show a progress bar during the co-occurrence counting step.
        Requires tqdm; falls back to a print message if unavailable.

    Returns
    -------
    pd.DataFrame
        Columns: item_A, item_B, instances, support, confidence, lift

        - item_A: the reference item
        - item_B: the co-occurring item
        - instances: number of transactions containing both A and B
        - support: instances / total_transactions
        - confidence: instances / transactions_containing_A  (i.e. P(B|A))
        - lift: confidence / (transactions_containing_B / total_transactions)

    Currently computes pairs only. Higher-order combinations may be added in
    future versions.
    """
    # --- Step 1: Validate inputs ---
    if transaction_col not in df.columns:
        raise ValueError(f"Column '{transaction_col}' not found in DataFrame")
    if item_col not in df.columns:
        raise ValueError(f"Column '{item_col}' not found in DataFrame")

    df = df.dropna(subset=[transaction_col, item_col])

    # --- Step 2: Build transaction dict (transaction -> set of items) ---
    trans_dict: dict = df.groupby(transaction_col)[item_col].apply(set).to_dict()
    total_transactions = len(trans_dict)

    # --- Step 3: Build item groups (item -> set of transactions, + count) ---
    item_groups = df.groupby(item_col)[transaction_col].apply(set).reset_index()
    item_groups.columns = [item_col, "_transactions"]
    item_groups["_trans_count"] = item_groups["_transactions"].apply(len)

    # Lookup dict for transaction counts per item (needed for lift)
    item_trans_count: dict = dict(
        zip(item_groups[item_col], item_groups["_trans_count"])
    )

    # --- Step 4: Count co-occurrences ---
    if show_progress:
        counters = []
        iterator = _wrap_progress(
            item_groups["_transactions"].items(),
            total=len(item_groups),
            desc="Counting co-occurrences",
        )
        for _idx, transactions in iterator:
            counters.append(
                Counter(chain(*(trans_dict[t] for t in transactions))).items()
            )
        item_groups["_counter"] = counters
    else:
        item_groups["_counter"] = item_groups["_transactions"].apply(
            lambda x: Counter(chain(*(trans_dict[t] for t in x))).items()
        )

    # --- Step 5: Explode into rows ---
    exploded = (
        item_groups[[item_col, "_trans_count", "_counter"]]
        .explode("_counter")
        .reset_index(drop=True)
    )
    exploded["item_B"] = exploded["_counter"].apply(lambda x: x[0])
    exploded["instances"] = exploded["_counter"].apply(lambda x: x[1])

    # --- Step 6: Remove self-associations ---
    result = exploded[exploded[item_col] != exploded["item_B"]].copy()

    # --- Step 7: Compute metrics ---
    result["support"] = np.round(result["instances"] / total_transactions, 4)
    result["confidence"] = np.round(
        result["instances"] / result["_trans_count"], 4
    )
    result["_b_trans_count"] = result["item_B"].map(item_trans_count)
    result["lift"] = np.round(
        result["confidence"] / (result["_b_trans_count"] / total_transactions), 4
    )

    # --- Step 8: Filter ---
    if min_support is not None:
        result = result[result["support"] >= min_support]
    if min_confidence is not None:
        result = result[result["confidence"] >= min_confidence]

    # --- Step 9: Clean up and return ---
    result = result[[item_col, "item_B", "instances", "support", "confidence", "lift"]].copy()
    result.columns = ["item_A", "item_B", "instances", "support", "confidence", "lift"]
    result = result.reset_index(drop=True)
    return result
