"""Core API for fast pairwise co-occurrence analysis."""

from __future__ import annotations

import pandas as pd


def find_associations(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    min_support: float | None = None,
    min_confidence: float | None = 0.0,
    min_lift: float | None = 0.0,
    min_conviction: float | None = 0.0,
    min_leverage: float | None = None,
    min_cosine: float | None = 0.0,
    min_jaccard: float | None = 0.0,
    show_progress: bool = False,
    backend: str = "auto",
) -> pd.DataFrame:
    """Compute pairwise item co-occurrence associations from transactional data.

    For every item A, counts how often each other item B appears in the same
    transactions, then computes support, confidence, and lift for each (A, B) pair.

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
        Minimum support threshold. Default None (no filtering).
    min_confidence : float or None
        Minimum confidence threshold. Default 0.0.
    min_lift : float or None
        Minimum lift threshold. Default 0.0.
    min_conviction : float or None
        Minimum conviction threshold. Default 0.0.
    min_leverage : float or None
        Minimum leverage threshold. Default None (no filtering, since
        leverage can be negative).
    min_cosine : float or None
        Minimum cosine similarity threshold. Default 0.0.
    min_jaccard : float or None
        Minimum jaccard similarity threshold. Default 0.0.
    show_progress : bool
        If True, show a progress bar during the co-occurrence counting step.
        Requires tqdm; falls back to a print message if unavailable.
        Only supported by the pandas backend.
    backend : str
        Computation backend. "auto" (default) uses polars if installed, else
        falls back to pandas. "pandas" uses Counter+chain approach. "polars"
        uses a self-join approach (requires polars to be installed). Polars
        backend is faster for sparse data (<10 items per transaction) but
        uses more memory due to join expansion.

    Returns
    -------
    pd.DataFrame
        Columns: item_A, item_B, instances, support, confidence, lift,
        conviction, leverage, cosine, jaccard

        - item_A: the reference item
        - item_B: the co-occurring item
        - instances: number of transactions containing both A and B
        - support: instances / total_transactions
        - confidence: instances / transactions_containing_A  (i.e. P(B|A))
        - lift: confidence / support(B)
        - conviction: (1 - support(B)) / (1 - confidence)
        - leverage: support(A,B) - support(A) * support(B)
        - cosine: support(A,B) / sqrt(support(A) * support(B))
        - jaccard: support(A,B) / (support(A) + support(B) - support(A,B))

    Currently computes pairs only. Use find_triplets() for 3-itemsets.
    """
    if transaction_col not in df.columns:
        raise ValueError(f"Column '{transaction_col}' not found in DataFrame")
    if item_col not in df.columns:
        raise ValueError(f"Column '{item_col}' not found in DataFrame")

    if backend == "auto":
        try:
            import polars as _pl  # noqa: F401
            backend = "polars"
        except ImportError:
            backend = "pandas"

    if backend == "polars":
        from fastapriori.backends.polars_backend import compute_associations
    elif backend == "pandas":
        from fastapriori.backends.pandas_backend import compute_associations
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'auto', 'pandas', or 'polars'.")

    result = compute_associations(
        df, transaction_col, item_col, show_progress
    )

    # Apply all metric filters centrally
    filters = {
        "support": min_support,
        "confidence": min_confidence,
        "lift": min_lift,
        "conviction": min_conviction,
        "leverage": min_leverage,
        "cosine": min_cosine,
        "jaccard": min_jaccard,
    }
    for col, threshold in filters.items():
        if threshold is not None:
            result = result[result[col] >= threshold]

    return result.reset_index(drop=True)
