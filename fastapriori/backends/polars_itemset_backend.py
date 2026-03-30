"""Polars backend for k-itemset computation — self-join approach.

Extends the k=2 Polars self-join (polars_backend.py) to k=3,4,5 by
chaining k copies of the DataFrame joined on transaction_col and
filtering to canonical ordering (item_1 < item_2 < ... < item_k).

Fully vectorized in Rust via Polars — no Python loops for counting.
"""

from __future__ import annotations

from collections import Counter


def compute_itemsets_polars(
    df,
    transaction_col: str,
    item_col: str,
    k: int,
    total_transactions: int,
) -> Counter:
    """Compute k-itemset counts using Polars k-way self-join.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (pandas; converted to Polars internally).
    transaction_col : str
        Transaction ID column name.
    item_col : str
        Item column name.
    k : int
        Itemset size (2-5).
    total_transactions : int
        Total number of unique transactions.

    Returns
    -------
    Counter
        Mapping canonical k-tuple -> co-occurrence count.
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required for the polars backend. "
            "Install it with: pip install polars"
        )

    # Convert pandas -> polars
    clean = df[[transaction_col, item_col]].dropna()
    df_pl = pl.DataFrame({
        transaction_col: clean[transaction_col].to_numpy(),
        item_col: clean[item_col].to_numpy(),
    }).unique()

    # Build k aliased lazy frames
    item_aliases = [f"item_{i}" for i in range(k)]
    lazy_frames = [
        df_pl.lazy().select(
            pl.col(transaction_col),
            pl.col(item_col).alias(item_aliases[i]),
        )
        for i in range(k)
    ]

    # Chain joins: join all k frames on transaction_col
    joined = lazy_frames[0]
    for i in range(1, k):
        joined = joined.join(lazy_frames[i], on=transaction_col)

    # Filter to canonical ordering: item_0 < item_1 < ... < item_{k-1}
    filters = [
        pl.col(item_aliases[i]) < pl.col(item_aliases[i + 1])
        for i in range(k - 1)
    ]
    for f in filters:
        joined = joined.filter(f)

    # Group by all item columns, count
    result = (
        joined
        .group_by(item_aliases)
        .agg(pl.len().alias("instances"))
        .collect()
    )

    if result.is_empty():
        return Counter()

    # Convert to Counter of canonical tuples
    itemset_counts = Counter()
    # Extract columns as lists for fast iteration
    cols = [result[alias].to_list() for alias in item_aliases]
    instances = result["instances"].to_list()

    for row_idx in range(len(instances)):
        key = tuple(cols[col_idx][row_idx] for col_idx in range(k))
        itemset_counts[key] = instances[row_idx]

    return itemset_counts
