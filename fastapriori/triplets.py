"""Triplet (3-itemset) co-occurrence analysis — deprecated wrapper.

Use ``find_associations(k=3)`` from ``fastapriori.core`` instead.
"""

from __future__ import annotations

import warnings

import pandas as pd


def find_triplets(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    min_support: float | None = None,
    min_confidence: float | None = 0.1,
    frequent_pairs: pd.DataFrame | None = None,
    show_progress: bool = False,
    backend: str = "auto",
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Compute 3-itemset co-occurrence associations.

    .. deprecated::
        Use ``find_associations(k=3)`` instead.
    """
    warnings.warn(
        "find_triplets() is deprecated. Use find_associations(k=3) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from fastapriori.core import find_associations

    return find_associations(
        df,
        transaction_col,
        item_col,
        k=3,
        min_support=min_support,
        min_confidence=min_confidence,
        frequent_lower=frequent_pairs,
        show_progress=show_progress,
        backend=backend,
        n_workers=n_workers,
    )
