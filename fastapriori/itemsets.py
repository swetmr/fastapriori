"""Generalized k-itemset co-occurrence analysis — deprecated wrapper.

Use ``find_associations(k=...)`` from ``fastapriori.core`` instead.
"""

from __future__ import annotations

import warnings

import pandas as pd


def find_itemsets(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    k: int = 3,
    min_support: float | None = None,
    min_confidence: float | None = 0.1,
    frequent_lower: pd.DataFrame | None = None,
    show_progress: bool = False,
    backend: str = "auto",
    n_workers: int | None = None,
    algo: str = "fast",
    **kwargs,
) -> pd.DataFrame:
    """Compute k-itemset co-occurrence associations.

    .. deprecated::
        Use ``find_associations(k=...)`` instead.
    """
    warnings.warn(
        "find_itemsets() is deprecated. Use find_associations(k=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from fastapriori.core import find_associations

    return find_associations(
        df,
        transaction_col,
        item_col,
        k=k,
        min_support=min_support,
        min_confidence=min_confidence,
        frequent_lower=frequent_lower,
        show_progress=show_progress,
        backend=backend,
        n_workers=n_workers,
        algo=algo,
        **kwargs,
    )
