"""Pandas backend for fastapriori — Counter+chain approach."""

from __future__ import annotations

from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd


def _wrap_progress(iterable, total: int | None = None, desc: str | None = None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        print(f"{desc}: processing {total} items (install tqdm for progress bar)...")
        return iterable


def compute_associations(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    show_progress: bool,
) -> pd.DataFrame:
    """Compute pairwise associations using pandas Counter+chain approach."""
    df = df.dropna(subset=[transaction_col, item_col])

    # Build transaction dict (transaction -> set of items)
    trans_dict: dict = df.groupby(transaction_col)[item_col].apply(set).to_dict()
    total_transactions = len(trans_dict)

    # Build item groups (item -> set of transactions, + count)
    item_groups = df.groupby(item_col)[transaction_col].apply(set).reset_index()
    item_groups.columns = [item_col, "_transactions"]
    item_groups["_trans_count"] = item_groups["_transactions"].apply(len)

    item_trans_count: dict = dict(
        zip(item_groups[item_col], item_groups["_trans_count"])
    )

    # Count co-occurrences
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

    # Explode into rows
    exploded = (
        item_groups[[item_col, "_trans_count", "_counter"]]
        .explode("_counter")
        .reset_index(drop=True)
    )
    exploded["item_B"] = exploded["_counter"].apply(lambda x: x[0])
    exploded["instances"] = exploded["_counter"].apply(lambda x: x[1])

    # Remove self-associations
    result = exploded[exploded[item_col] != exploded["item_B"]].copy()

    # Compute metrics
    result["support"] = np.round(result["instances"] / total_transactions, 6)
    result["confidence"] = np.round(
        result["instances"] / result["_trans_count"], 6
    )
    result["_b_trans_count"] = result["item_B"].map(item_trans_count)
    support_a = result["_trans_count"] / total_transactions
    support_b = result["_b_trans_count"] / total_transactions
    result["lift"] = np.round(result["confidence"] / support_b, 6)
    result["conviction"] = np.round(
        (1 - support_b) / (1 - result["confidence"] + 1e-10), 4
    )
    result["leverage"] = np.round(
        result["support"] - (support_a * support_b), 6
    )
    result["cosine"] = np.round(
        result["support"] / np.sqrt(support_a * support_b), 4
    )
    result["jaccard"] = np.round(
        result["support"] / (support_a + support_b - result["support"]), 4
    )

    # Clean up
    out_cols = [item_col, "item_B", "instances", "support", "confidence",
                "lift", "conviction", "leverage", "cosine", "jaccard"]
    result = result[out_cols].copy()
    result.columns = ["item_A", "item_B", "instances", "support", "confidence",
                      "lift", "conviction", "leverage", "cosine", "jaccard"]
    return result.reset_index(drop=True)
