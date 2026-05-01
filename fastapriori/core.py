"""Core API for fast co-occurrence and association analysis (k=2 to k=50)."""

from __future__ import annotations

import logging
import warnings
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger("fastapriori")

_LOWMEM_SENTINEL_STR = "__fastapriori_lowmem_sentinel__"


def find_associations(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    k: int = 2,
    min_support: float | None = None,
    min_confidence: float | None = 0.0,
    min_lift: float | None = 0.0,
    min_conviction: float | None = 0.0,
    min_leverage: float | None = None,
    min_cosine: float | None = 0.0,
    min_jaccard: float | None = 0.0,
    frequent_lower: pd.DataFrame | None = None,
    show_progress: bool = False,
    backend: str = "auto",
    algo: str = "fast",
    n_workers: int | None = None,
    sorted_by: str | None = "support",
    low_memory: bool | str = "auto",
    max_items_per_txn: int | None = None,
    item_weights: dict | None = None,
    verbose: bool = False,
    backend_options: dict | None = None,
) -> pd.DataFrame:
    """Compute item co-occurrence associations from transactional data.

    For k=2, counts how often each pair (A, B) appears in the same
    transactions and computes 7 interestingness metrics.  For k>=3,
    counts k-item combinations and computes support, confidence, and lift
    for each directional rule.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with at least two columns: one for transaction IDs and one
        for item identifiers. Each row represents one (transaction, item) pair.
    transaction_col : str
        Name of the column containing transaction/group identifiers.
    item_col : str
        Name of the column containing item identifiers.
    k : int
        Itemset size (2-50). Default 2 (pairwise associations).
    min_support : float or None
        Minimum support threshold. Default None (no filtering).
        Required when ``algo="classic"`` (Apriori needs it for pruning).
    min_confidence : float or None
        Minimum confidence threshold. Default 0.0.
    min_lift : float or None
        Minimum lift threshold. Default 0.0. Only used when k=2.
    min_conviction : float or None
        Minimum conviction threshold. Default 0.0. Only used when k=2.
    min_leverage : float or None
        Minimum leverage threshold. Default None. Only used when k=2.
    min_cosine : float or None
        Minimum cosine similarity threshold. Default 0.0. Only used when k=2.
    min_jaccard : float or None
        Minimum Jaccard similarity threshold. Default 0.0. Only used when k=2.
    frequent_lower : pd.DataFrame or None
        Results from the (k-1) level, used for Apriori pruning and confidence
        computation when k>=3.  When k=3, this should be the output of
        ``find_associations(k=2)``.  When k=4, this should be the output of
        ``find_associations(k=3)``, and so on.  If None and k>=3, the lower
        levels are auto-computed. Ignored when k=2.
    show_progress : bool
        If True, show a progress bar during counting. Requires tqdm.
    backend : str
        Computation backend.  ``"auto"`` (default) uses the Rust backend if
        the compiled extension is available, otherwise falls back to
        ``"python"``.  ``"python"`` uses polars for k=2 (falling back to
        pandas if polars is not installed) and counter_chain for k>=3.
        Individual backends can still be selected explicitly: ``"rust"``,
        ``"pandas"``, ``"polars"``, ``"counter_chain"``, ``"combinations"``,
        ``"bin_multi"``.  Ignored when ``algo="classic"`` (always uses Rust).
    algo : str
        Algorithm selection.  ``"fast"`` (default) uses the inverted-index
        count-all algorithm.  ``"classic"`` uses a Rust port of the
        efficient-apriori algorithm (multi-pass Apriori with candidate
        generation and pruning; requires ``min_support``).  ``"auto"`` uses
        an ML model to predict which algorithm is faster for the given data
        characteristics (not yet implemented).
    n_workers : int or None
        Number of parallel workers for the counter_chain/bin_multi backends
        (k>=3 only). None auto-selects, 1 forces serial. Ignored when k=2.
    sorted_by : str or None
        Column name to sort results by in descending order.  Default
        ``"support"``.  Pass None to skip sorting.
    low_memory : bool or str
        If True, pre-filters the DataFrame to remove items below
        ``min_support`` before building the inverted index.  This can
        reduce memory usage by 5-10x on large catalogs (e.g. 30GB to
        3GB for Instacart) at the cost of losing "compute all, filter
        later" flexibility.  Requires ``min_support`` to be specified.
        Default ``"auto"`` enables filtering whenever ``min_support``
        is provided, and disables it otherwise.
    max_items_per_txn : int or None
        Maximum number of items to keep per transaction when k>=3.
        Transactions exceeding this limit are capped to their top-N
        items by weight, bounding C(d, k-1) for outlier transactions.
        Default None (no capping).  Note: capping produces approximate
        counts — reported counts are lower bounds (never overcounted),
        but some truly frequent itemsets may be missed if their counts
        drop below ``min_support`` after capping.  Supported for both
        ``algo="fast"`` and ``algo="classic"`` when k>=3; ignored at
        k=2 (pairs are always counted exactly).
    item_weights : dict or None
        Item-to-weight mapping used for ranking when ``max_items_per_txn``
        caps a transaction.  Higher weight = higher priority to keep.
        Default None uses item frequency (transaction count) as weight,
        which preserves the items most likely to form frequent itemsets.
        Accepts any dict-like mapping item labels to numeric values
        (e.g. revenue, volume, or a composite score).
    verbose : bool
        If True, prints dataset features (n_transactions, n_items, d_avg,
        d_max, etc.), the chosen algorithm, and a density warning when
        the estimated combination cost is very high. Useful for
        understanding performance characteristics before long runs.
    backend_options : dict or None
        Reserved for internal tuning. Not a stable API — keys may be
        renamed or removed without deprecation. Currently recognised:
        ``"fast_variant"`` in {``"apriori"``, ``"eclat"``}, applied only
        when ``algo="fast"`` and k>=3.

    Returns
    -------
    pd.DataFrame
        When k=2: columns item_A, item_B, instances, support, confidence,
        lift, conviction, leverage, cosine, jaccard.

        When k>=3: columns antecedent_1, ..., antecedent_{k-1}, consequent,
        instances, support, confidence, lift.  Each k-itemset produces k
        directional rules (one per item as consequent).
    """
    if k < 2 or k > 50:
        raise ValueError(f"k must be between 2 and 50, got {k}")
    if transaction_col not in df.columns:
        raise ValueError(f"Column '{transaction_col}' not found in DataFrame")
    if item_col not in df.columns:
        raise ValueError(f"Column '{item_col}' not found in DataFrame")
    if algo not in ("fast", "classic", "auto"):
        raise ValueError(
            f"algo must be 'fast', 'classic', or 'auto', got '{algo}'"
        )
    if algo == "classic" and min_support is None:
        raise ValueError(
            "algo='classic' requires min_support (Apriori needs it for pruning)"
        )
    if min_support is not None and not (0.0 <= float(min_support) <= 1.0):
        raise ValueError(
            f"min_support must be in [0.0, 1.0], got {min_support}. "
            "Did you mean to pass a percentage as a fraction (e.g. 0.05 for 5%)?"
        )
    if item_weights is not None:
        _bad = [k_ for k_, v_ in item_weights.items()
                if v_ is None or (isinstance(v_, float) and np.isnan(v_))]
        if _bad:
            raise ValueError(
                f"item_weights contains NaN/None for {len(_bad)} item(s) "
                f"(first: {_bad[:3]}). Replace with finite values (e.g. 0)."
            )
    if algo == "auto":
        algo = "fast"

    # backend_options is reserved; "fast_variant" and "impl_variant" are
    # read today.  impl_variant selects the Rust implementation generation
    # for ablation studies (v0_baseline = current code, v1_roaring,
    # v2_memo, v3_simd).  Only applied when backend resolves to rust + algo=fast.
    _IMPL_VARIANTS = (
        "v0_baseline", "v1_roaring", "v2_memo",
        "v3_adaptive", "v5_prefilter", "v6_gating",
    )
    if backend_options is not None:
        _fv = backend_options.get("fast_variant")
        if _fv is not None and _fv not in ("apriori", "eclat"):
            raise ValueError(
                f"backend_options['fast_variant'] must be 'apriori' or "
                f"'eclat', got {_fv!r}"
            )
        _iv = backend_options.get("impl_variant")
        if _iv is not None and _iv not in _IMPL_VARIANTS:
            raise ValueError(
                f"backend_options['impl_variant'] must be one of "
                f"{_IMPL_VARIANTS}, got {_iv!r}"
            )

    # Resolve low_memory="auto": enable when min_support is provided.
    # Skip it when the Rust fast path is available — the pandas
    # groupby(item).nunique() pre-filter costs ~7s on Instacart and is
    # redundant with the int32 downward-closure filter inside
    # count_k_itemsets_internal.  Explicit low_memory=True still wins.
    if low_memory == "auto":
        _rust_fast_path = False
        if backend in ("auto", "rust") and algo == "fast":
            try:
                from fastapriori._fastapriori_rs import rust_compute_pairs  # noqa: F401
                _rust_fast_path = True
            except ImportError:
                pass
        low_memory = (min_support is not None) and not _rust_fast_path

    # Low-memory mode: pre-filter infrequent items before any backend runs
    if low_memory:
        if min_support is None:
            raise ValueError(
                "low_memory=True requires min_support to be specified."
            )
        item_counts = df.groupby(item_col)[transaction_col].nunique()
        n_transactions = df[transaction_col].nunique()
        item_support = item_counts / n_transactions
        frequent_items = item_support[item_support >= min_support].index
        if len(frequent_items) == 0:
            # No items meet min_support — return empty result
            if k == 2:
                return pd.DataFrame(columns=[
                    "item_A", "item_B", "instances", "support",
                    "confidence", "lift", "conviction", "leverage",
                    "cosine", "jaccard",
                ])
            else:
                ant_cols = [f"antecedent_{i}" for i in range(1, k)]
                return pd.DataFrame(columns=ant_cols + [
                    "consequent", "instances", "support",
                    "confidence", "lift",
                ])
        original_rows = len(df)
        original_txns = df[transaction_col].unique()  # numpy, no Python set
        df = df[df[item_col].isin(frequent_items)]
        kept_txns = df[transaction_col].unique()
        # pd.Index.difference uses a hash-based diff; np.setdiff1d falls into
        # an O(n*m) path on object/string arrays (14s on 28K string txn_ids
        # for Online Retail).  Hash diff is ~1000x faster there and within
        # ~40ms of setdiff1d on large int arrays.
        lost_txns = pd.Index(original_txns).difference(kept_txns).to_numpy()
        sentinel = None
        if len(lost_txns) > 0:
            # Preserve n_transactions via one sentinel row per lost transaction.
            # The sentinel is the sole item in its transaction, so it forms no
            # pairs/itemsets — results stay exact.  Tracked here so downstream
            # code can drop it from item_support / lower_support dicts.
            dtype = df[item_col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                sentinel = int(df[item_col].min()) - 1
            elif pd.api.types.is_float_dtype(dtype):
                sentinel = float(df[item_col].min()) - 1.0
            else:
                sentinel = _LOWMEM_SENTINEL_STR
            dummy = pd.DataFrame({
                transaction_col: lost_txns,
                item_col: sentinel,
            })
            df = pd.concat([df, dummy], ignore_index=True)

        if show_progress:
            logger.info(
                "low_memory: kept %d/%d items (%d/%d rows, %d txns preserved via sentinel)",
                len(frequent_items), len(item_counts),
                len(df), original_rows, len(lost_txns),
            )
    else:
        sentinel = None

    # --- Verbose: print dataset features and density warning ---
    if verbose:
        from math import comb
        _grp = df.groupby(transaction_col)[item_col].nunique()
        _n_txn = int(_grp.count())
        _n_items = int(df[item_col].nunique())
        _n_rows = len(df)
        _d_avg = float(_grp.mean())
        _d_max = int(_grp.max())
        _d_median = float(_grp.median())
        _d_std = float(_grp.std()) if _n_txn > 1 else 0.0
        # verbose=True is an explicit opt-in for stdout output; log at INFO
        # (users can silence via logging config) but also print so users who
        # haven't configured logging still see the profile.
        _msgs = [
            f"[fastapriori] Dataset: {_n_txn:,} txns x {_n_items:,} items | {_n_rows:,} rows",
            f"[fastapriori] d_avg={_d_avg:.1f}  d_max={_d_max}  d_median={_d_median:.1f}  d_std={_d_std:.1f}",
            f"[fastapriori] k={k}  min_support={min_support}  algo={algo}",
        ]
        if k >= 3 and _d_max >= k:
            _est_combos = comb(_d_max, k - 1) * _n_txn
            if _est_combos > 1e8:
                _msgs.append(
                    f"[fastapriori] WARNING: C({_d_max},{k-1}) x {_n_txn:,} = "
                    f"{_est_combos:.0e} combinations - may be slow"
                )
        for _m in _msgs:
            logger.info(_m)
            print(_m)

    # --- Classic Apriori path: bypass normal backend resolution ---
    if algo == "classic":
        if k == 2:
            result = _find_classic_pairs(
                df, transaction_col, item_col,
                min_support, min_confidence, min_lift,
                min_conviction, min_leverage, min_cosine, min_jaccard,
                show_progress,
            )
        else:
            result = _find_classic_k_itemsets(
                df, transaction_col, item_col, k,
                min_support, min_confidence,
                max_items_per_txn=max_items_per_txn,
                item_weights=item_weights,
            )

        result = _drop_sentinel_rows(result, sentinel, k)
        result = _apply_sort(result, sorted_by)
        return result

    # Resolve "auto": try Rust, fall back to "python"
    if backend == "auto":
        try:
            from fastapriori._fastapriori_rs import rust_compute_pairs  # noqa: F401
            backend = "rust"
        except ImportError:
            backend = "python"

    # Resolve "python": polars (fallback pandas) for k=2, counter_chain for k>=3
    if backend == "python":
        if k == 2:
            try:
                import polars as _pl  # noqa: F401
                backend = "polars"
            except ImportError:
                backend = "pandas"
        else:
            backend = "counter_chain"

    if k == 2:
        result = _find_pairs(
            df, transaction_col, item_col,
            min_support, min_confidence, min_lift,
            min_conviction, min_leverage, min_cosine, min_jaccard,
            show_progress, backend,
        )
    else:
        result = _find_k_itemsets(
            df, transaction_col, item_col, k,
            min_support, min_confidence,
            frequent_lower, show_progress, backend, n_workers,
            max_items_per_txn=max_items_per_txn,
            item_weights=item_weights,
            backend_options=backend_options,
        )

    result = _drop_sentinel_rows(result, sentinel, k)
    result = _apply_sort(result, sorted_by)
    return result


def _apply_sort(result: pd.DataFrame, sorted_by: str | None) -> pd.DataFrame:
    """Sort by a column if present; warn-and-skip otherwise rather than raise."""
    if sorted_by is None:
        return result
    if sorted_by not in result.columns:
        warnings.warn(
            f"sorted_by='{sorted_by}' not in result columns "
            f"({list(result.columns)}); skipping sort.",
            UserWarning,
            stacklevel=3,
        )
        return result
    return result.sort_values(sorted_by, ascending=False).reset_index(drop=True)


def _drop_sentinel_rows(
    result: pd.DataFrame, sentinel, k: int,
) -> pd.DataFrame:
    """Drop rows where any item column equals the low_memory sentinel label.

    The sentinel is inserted by the low_memory path to preserve n_transactions;
    it should never surface in user-facing output.
    """
    if sentinel is None or result.empty:
        return result
    item_cols: list[str] = []
    if k == 2:
        item_cols = [c for c in ("item_A", "item_B") if c in result.columns]
    else:
        item_cols = [
            c for c in result.columns
            if c.startswith("antecedent_") or c == "consequent"
        ]
    if not item_cols:
        return result
    mask = np.ones(len(result), dtype=bool)
    for c in item_cols:
        mask &= result[c] != sentinel
    return result[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# k=2 path (unchanged from original find_associations)
# ---------------------------------------------------------------------------


def _find_pairs(
    df, transaction_col, item_col,
    min_support, min_confidence, min_lift,
    min_conviction, min_leverage, min_cosine, min_jaccard,
    show_progress, backend, trans_dict=None,
) -> pd.DataFrame:
    """Pairwise (k=2) co-occurrence with 7 metrics."""
    if backend == "polars":
        from fastapriori.backends.polars_backend import compute_associations
    elif backend == "pandas":
        from fastapriori.backends.pandas_backend import compute_associations
    elif backend == "rust":
        from fastapriori.backends.rust_backend import compute_associations
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Use 'auto', 'python', 'pandas', 'polars', or 'rust'."
        )

    # Rust backend applies filters on numeric arrays pre-decode — avoids
    # materialising millions of object-dtype rows only to drop them.
    if backend == "rust":
        result = compute_associations(
            df, transaction_col, item_col, show_progress, trans_dict=trans_dict,
            min_support=min_support, min_confidence=min_confidence,
            min_lift=min_lift, min_conviction=min_conviction,
            min_leverage=min_leverage, min_cosine=min_cosine,
            min_jaccard=min_jaccard,
        )
        return result

    result = compute_associations(
        df, transaction_col, item_col, show_progress, trans_dict=trans_dict,
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


# ---------------------------------------------------------------------------
# k>=3 path (lifted from itemsets.py)
# ---------------------------------------------------------------------------


def _find_k_itemsets(
    df, transaction_col, item_col, k,
    min_support, min_confidence,
    frequent_lower, show_progress, backend, n_workers,
    _trans_dict=None,
    max_items_per_txn=None,
    item_weights=None,
    backend_options=None,
) -> pd.DataFrame:
    """k-itemset (k>=3) co-occurrence with support, confidence, lift."""
    df = df.dropna(subset=[transaction_col, item_col])

    fast_variant = (
        (backend_options or {}).get("fast_variant", "apriori")
    )
    # Default impl_variant: v3_adaptive — adaptive Sparse(Vec)/Dense(Roaring)
    # TID storage with cross-level intersection memoisation. Locally validated
    # to be the best general-purpose variant on small/medium datasets across
    # k=3..5 and supports 0.001..0.0001. Override via
    # backend_options={'impl_variant': 'v0_baseline' | 'v1_roaring' | ...}
    # for ablation studies.
    impl_variant = (
        (backend_options or {}).get("impl_variant", "v3_adaptive")
    )

    # Pipeline short-circuit: Rust handles full k=2→...→k chain in one call.
    # Eclat always takes this path (it rebuilds its own lattice from
    # tid-lists and cannot consume caller-provided frequent_lower, so
    # chaining via _find_k_itemsets would just do duplicate work).
    if backend == "rust" and (
        frequent_lower is None or fast_variant == "eclat"
    ):
        from fastapriori.backends.rust_backend import compute_pipeline
        return compute_pipeline(
            df, transaction_col, item_col, k,
            min_support if min_support is not None else 0.0,
            min_confidence,
            max_items_per_txn=max_items_per_txn,
            item_weights=item_weights,
            fast_variant=fast_variant,
            impl_variant=impl_variant,
        )

    # Build transaction dict ONCE — reuse if passed from a parent call
    if _trans_dict is not None:
        trans_dict = _trans_dict
    else:
        # defaultdict loop is ~5x faster than groupby.apply(set).to_dict()
        _td: dict = defaultdict(set)
        _txns = df[transaction_col].values
        _items = df[item_col].values
        for _i in range(len(_txns)):
            _td[_txns[_i]].add(_items[_i])
        trans_dict = dict(_td)
        del _td, _txns, _items
    total_transactions = len(trans_dict)

    # --- Auto-compute frequent_lower chain when not provided ---
    if frequent_lower is None:
        # Pick a valid k=2 backend: rust stays rust, counter_chain/bin_multi
        # use polars (fallback pandas) since they are k>=3 only backends
        if backend == "rust":
            k2_backend = "rust"
        elif backend in ("polars", "pandas"):
            k2_backend = backend
        else:
            # counter_chain, bin_multi, combinations → use polars or pandas
            try:
                import polars as _pl  # noqa: F401
                k2_backend = "polars"
            except ImportError:
                k2_backend = "pandas"
        frequent_lower = _find_pairs(
            df, transaction_col, item_col,
            min_support, 0.0, 0.0, 0.0, None, 0.0, 0.0,
            show_progress=False, backend=k2_backend, trans_dict=trans_dict,
        )
        for level in range(3, k):
            frequent_lower = _find_k_itemsets(
                df, transaction_col, item_col, k=level,
                min_support=min_support, min_confidence=min_confidence,
                frequent_lower=frequent_lower,
                show_progress=show_progress, backend=backend,
                n_workers=n_workers, _trans_dict=trans_dict,
                backend_options=backend_options,
            )

    # --- Resolve backend ---
    if backend == "auto":
        backend_resolved = "counter_chain"
    else:
        backend_resolved = backend

    # --- Build (k-1)-itemset support lookup for confidence computation ---
    lower_support: dict[tuple, float] = {}
    if frequent_lower is not None:
        lower_support = _extract_lower_support(
            frequent_lower, k, total_transactions
        )

    # --- Count k-itemsets ---
    if backend_resolved == "polars":
        from fastapriori.backends.polars_itemset_backend import (
            compute_itemsets_polars,
        )

        itemset_counts = compute_itemsets_polars(
            df, transaction_col, item_col, k, total_transactions,
        )
    elif backend_resolved == "bin_multi":
        from fastapriori.backends.bin_multi_backend import (
            compute_itemsets_bin_multi,
        )

        itemset_counts = compute_itemsets_bin_multi(
            trans_dict, total_transactions, k,
            frequent_lower, n_workers, show_progress,
        )
    elif backend_resolved == "counter_chain":
        from fastapriori.backends.itemset_counter_chain import compute_itemsets

        itemset_counts = compute_itemsets(
            trans_dict, total_transactions, k,
            frequent_lower, n_workers, show_progress,
            max_items_per_txn=max_items_per_txn,
            item_weights=item_weights,
        )
    elif backend_resolved == "rust":
        from fastapriori.backends.rust_backend import compute_itemsets_rust

        itemset_counts = compute_itemsets_rust(
            trans_dict, total_transactions, k,
            frequent_lower, n_workers, show_progress,
            df=df, transaction_col=transaction_col, item_col=item_col,
            fast_variant=fast_variant,
        )
    else:
        # Original combinations approach
        freq_pair_set: set | None = None
        if frequent_lower is not None and k >= 3:
            freq_pair_set = _extract_freq_pairs(frequent_lower, k)

        itemset_counts = Counter()
        iterator = trans_dict.values()
        if show_progress:
            iterator = _wrap_progress(
                iterator, total=total_transactions,
                desc=f"Counting {k}-itemsets",
            )

        for items in iterator:
            if len(items) < k:
                continue
            sorted_items = sorted(items)
            for itemset in combinations(sorted_items, k):
                if freq_pair_set is not None:
                    if not all(
                        pair in freq_pair_set
                        for pair in combinations(itemset, 2)
                    ):
                        continue
                itemset_counts[itemset] += 1

    # --- Build output columns ---
    ant_cols = [f"antecedent_{i}" for i in range(1, k)]
    out_cols = ant_cols + [
        "consequent", "instances", "support", "confidence", "lift",
    ]

    if not itemset_counts:
        return pd.DataFrame(columns=out_cols)

    # --- Build (k-1)-itemset support from counting if not available ---
    if not lower_support:
        lower_support = _count_lower_support(trans_dict, k, total_transactions)

    # --- Item-level support (needed for lift) ---
    item_counts = df.groupby(item_col)[transaction_col].nunique()
    item_support = (item_counts / total_transactions).to_dict()

    # --- Generate k directional rules per itemset ---
    records = []
    for itemset, count in itemset_counts.items():
        support = count / total_transactions
        for i in range(k):
            consequent = itemset[i]
            antecedents = itemset[:i] + itemset[i + 1:]
            ant_key = tuple(sorted(antecedents))
            ant_sup = lower_support.get(ant_key, 0)
            confidence = support / (ant_sup + 1e-10)
            cons_sup = item_support.get(consequent, 0)
            lift = confidence / (cons_sup + 1e-10)
            records.append(
                (*antecedents, consequent, count, support, confidence, lift)
            )

    result = pd.DataFrame(records, columns=out_cols)
    result["support"] = np.round(result["support"], 6)
    result["confidence"] = np.round(result["confidence"], 6)
    result["lift"] = np.round(result["lift"], 6)

    # Filter
    if min_support is not None:
        result = result[result["support"] >= min_support]
    if min_confidence is not None:
        result = result[result["confidence"] >= min_confidence]

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Classic Apriori paths
# ---------------------------------------------------------------------------


def _find_classic_pairs(
    df, transaction_col, item_col,
    min_support, min_confidence, min_lift,
    min_conviction, min_leverage, min_cosine, min_jaccard,
    show_progress,
) -> pd.DataFrame:
    """Pairwise (k=2) associations via classic Apriori (Rust port of efficient-apriori)."""
    from fastapriori.backends.rust_classic_backend import compute_associations

    result = compute_associations(
        df, transaction_col, item_col, show_progress, min_support=min_support,
    )

    # Apply metric filters
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


def _find_classic_k_itemsets(
    df, transaction_col, item_col, k,
    min_support, min_confidence,
    max_items_per_txn=None,
    item_weights=None,
) -> pd.DataFrame:
    """k-itemset (k>=3) via classic Apriori pipeline (Rust port of efficient-apriori)."""
    from fastapriori.backends.rust_classic_backend import compute_pipeline

    return compute_pipeline(
        df, transaction_col, item_col, k,
        min_support if min_support is not None else 0.0,
        min_confidence,
        max_items_per_txn=max_items_per_txn,
        item_weights=item_weights,
    )


# ---------------------------------------------------------------------------
# Helpers (moved from itemsets.py)
# ---------------------------------------------------------------------------


def _wrap_progress(iterable, total: int | None = None, desc: str | None = None):
    """Wrap an iterable with tqdm if available, else a simple print fallback."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        print(
            f"{desc}: processing {total} items "
            "(install tqdm for progress bar)..."
        )
        return iterable


def _extract_freq_pairs(frequent_lower: pd.DataFrame, k: int) -> set:
    """Extract the set of frequent pairs from the lower-level DataFrame."""
    freq_pairs: set = set()

    # Try pair-level columns first (output of find_associations k=2)
    if "item_A" in frequent_lower.columns and "item_B" in frequent_lower.columns:
        for a, b in zip(
            frequent_lower["item_A"].to_numpy(),
            frequent_lower["item_B"].to_numpy(),
        ):
            freq_pairs.add(tuple(sorted([a, b])))
        return freq_pairs

    # Otherwise, reconstruct itemsets from antecedent_* + consequent columns
    ant_cols = [c for c in frequent_lower.columns if c.startswith("antecedent_")]
    if ant_cols and "consequent" in frequent_lower.columns:
        cols = ant_cols + ["consequent"]
        arr = frequent_lower[cols].to_numpy()
        for row in arr:
            items = tuple(sorted(row.tolist()))
            for pair in combinations(items, 2):
                freq_pairs.add(pair)
        return freq_pairs

    return freq_pairs


def _extract_lower_support(
    frequent_lower: pd.DataFrame, k: int, total_transactions: int
) -> dict[tuple, float]:
    """Build a dict of (k-1)-itemset -> support from the lower-level DataFrame."""
    support_dict: dict[tuple, float] = {}

    # Pair-level (from find_associations k=2 output)
    if "item_A" in frequent_lower.columns and "item_B" in frequent_lower.columns:
        a_arr = frequent_lower["item_A"].to_numpy()
        b_arr = frequent_lower["item_B"].to_numpy()
        if "support" in frequent_lower.columns:
            s_arr = frequent_lower["support"].to_numpy()
            for a, b, s in zip(a_arr, b_arr, s_arr):
                key = tuple(sorted([a, b]))
                if key not in support_dict:
                    support_dict[key] = s
        elif "instances" in frequent_lower.columns:
            inst_arr = frequent_lower["instances"].to_numpy()
            for a, b, inst in zip(a_arr, b_arr, inst_arr):
                key = tuple(sorted([a, b]))
                if key not in support_dict:
                    support_dict[key] = inst / total_transactions
        return support_dict

    # Higher-level (antecedent_* + consequent)
    ant_cols = [c for c in frequent_lower.columns if c.startswith("antecedent_")]
    if ant_cols and "consequent" in frequent_lower.columns:
        item_cols = ant_cols + ["consequent"]
        items_arr = frequent_lower[item_cols].to_numpy()
        has_support = "support" in frequent_lower.columns
        has_instances = "instances" in frequent_lower.columns
        metric_arr = None
        if has_support:
            metric_arr = frequent_lower["support"].to_numpy()
        elif has_instances:
            metric_arr = frequent_lower["instances"].to_numpy()

        for i, row in enumerate(items_arr):
            items = tuple(sorted(row.tolist()))
            if items in support_dict:
                continue
            if has_support:
                support_dict[items] = metric_arr[i]
            elif has_instances:
                support_dict[items] = metric_arr[i] / total_transactions

    return support_dict


def _count_lower_support(
    trans_dict: dict, k: int, total_transactions: int
) -> dict[tuple, float]:
    """Count (k-1)-itemset support directly from transactions.

    Used as fallback when frequent_lower is not provided.
    """
    lower_k = k - 1
    counts: Counter = Counter()
    for items in trans_dict.values():
        if len(items) < lower_k:
            continue
        for subset in combinations(sorted(items), lower_k):
            counts[subset] += 1
    return {key: count / total_transactions for key, count in counts.items()}
