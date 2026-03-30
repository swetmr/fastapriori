"""Polars backend for fastapriori — self-join approach."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_associations(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    show_progress: bool,
    trans_dict: dict | None = None,
) -> pd.DataFrame:
    """Compute pairwise associations using Polars self-join approach.

    Accepts a pandas DataFrame (for API compatibility), converts to Polars
    internally, and returns a pandas DataFrame.
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required for the polars backend. "
            "Install it with: pip install polars"
        )

    # Convert pandas -> polars (column-by-column to avoid pyarrow dependency)
    clean = df[[transaction_col, item_col]].dropna()
    df_pl = pl.DataFrame({
        transaction_col: clean[transaction_col].to_numpy(),
        item_col: clean[item_col].to_numpy(),
    }).unique()
    total_transactions = df_pl[transaction_col].n_unique()

    # Item transaction counts (needed for confidence, lift, etc.)
    item_counts = (
        df_pl
        .group_by(item_col)
        .agg(pl.col(transaction_col).n_unique().alias("_trans_count"))
    )
    item_count_map = dict(
        zip(
            item_counts[item_col].to_list(),
            item_counts["_trans_count"].to_list(),
        )
    )

    # Self-join on transaction_col, filter A != B, count pairs
    df_a = df_pl.lazy().select(
        pl.col(transaction_col),
        pl.col(item_col).alias("item_A"),
    )
    df_b = df_pl.lazy().select(
        pl.col(transaction_col),
        pl.col(item_col).alias("item_B"),
    )

    pairs = (
        df_a
        .join(df_b, on=transaction_col)
        .filter(pl.col("item_A") != pl.col("item_B"))
        .group_by(["item_A", "item_B"])
        .agg(pl.len().alias("instances"))
        .collect()
    )

    # Convert to pandas for metric computation (avoid pyarrow dependency)
    if pairs.is_empty():
        return pd.DataFrame(
            columns=["item_A", "item_B", "instances", "support", "confidence",
                     "lift", "conviction", "leverage", "cosine", "jaccard"]
        )

    result = pd.DataFrame({
        "item_A": pairs["item_A"].to_list(),
        "item_B": pairs["item_B"].to_list(),
        "instances": pairs["instances"].to_list(),
    })

    # Compute metrics
    result["_a_count"] = result["item_A"].map(item_count_map)
    result["_b_count"] = result["item_B"].map(item_count_map)

    result["support"] = np.round(result["instances"] / total_transactions, 6)
    result["confidence"] = np.round(result["instances"] / result["_a_count"], 6)

    support_a = result["_a_count"] / total_transactions
    support_b = result["_b_count"] / total_transactions

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
    out_cols = ["item_A", "item_B", "instances", "support", "confidence",
                "lift", "conviction", "leverage", "cosine", "jaccard"]
    result = result[out_cols].copy()
    return result.reset_index(drop=True)
