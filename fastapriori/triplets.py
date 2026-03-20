"""Triplet (3-itemset) co-occurrence analysis — backward-compatible wrapper."""

from __future__ import annotations

import pandas as pd

from fastapriori.itemsets import find_itemsets


def find_triplets(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    min_support: float | None = None,
    min_confidence: float | None = 0.1,
    frequent_pairs: pd.DataFrame | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute 3-itemset co-occurrence associations from transactional data.

    This is a convenience wrapper around ``find_itemsets(k=3)``.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with transaction and item columns.
    transaction_col : str
        Name of the transaction ID column.
    item_col : str
        Name of the item column.
    min_support : float or None
        Minimum support threshold. None disables filtering.
    min_confidence : float or None
        Minimum confidence threshold. Default 0.1 (10%).
    frequent_pairs : pd.DataFrame or None
        Output from find_associations() used to prune the search space.
        Must have columns item_A and item_B. If provided, only triplets
        where all 3 constituent pairs appear in this DataFrame are counted.
    show_progress : bool
        Show progress bar during counting.

    Returns
    -------
    pd.DataFrame
        Columns: antecedent_1, antecedent_2, consequent, instances,
        support, confidence, lift

        Each triplet (A,B,C) produces 3 directional rules:
        A,B -> C  |  A,C -> B  |  B,C -> A
    """
    return find_itemsets(
        df,
        transaction_col=transaction_col,
        item_col=item_col,
        k=3,
        min_support=min_support,
        min_confidence=min_confidence,
        frequent_lower=frequent_pairs,
        show_progress=show_progress,
    )
