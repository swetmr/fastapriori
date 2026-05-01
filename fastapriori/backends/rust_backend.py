"""Rust backend for fastapriori — PyO3 extension wrapping compiled Rust code."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

# i32 encoder cap: item IDs are encoded as int32 before crossing into Rust.
_MAX_ITEMS = 2**31 - 1


try:
    from fastapriori._fastapriori_rs import (
        rust_compute_pairs,
        rust_compute_itemsets,
        rust_compute_pipeline,
        rust_eclat_pipeline,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


# Optional ablation variants — present only after the corresponding Rust
# changes land.  Resolved lazily via getattr() so a stale build that lacks
# v1/v2/v3 still imports cleanly.
def _resolve_pipeline_variant(name: str):
    """Return the Rust pipeline function for the given impl_variant.

    Falls back to a clear NotImplementedError when a variant has not yet
    been built — the local-test notebook treats this as 'skip cell'.
    """
    if name == "v0_baseline":
        return rust_compute_pipeline
    try:
        import fastapriori._fastapriori_rs as _rs
    except ImportError as exc:
        raise ImportError(
            "Rust backend not available. Build with: maturin develop --release"
        ) from exc
    fn = getattr(_rs, f"rust_compute_pipeline_{name}", None)
    if fn is None:
        raise NotImplementedError(
            f"impl_variant={name!r} not available in this Rust build. "
            f"Re-run `maturin develop --release` after landing the "
            f"corresponding phase, or fall back to v0_baseline."
        )
    return fn


def _check_rust():
    if not RUST_AVAILABLE:
        raise ImportError(
            "Rust backend not available. Build with: maturin develop --release"
        )


def _sorted_unique_items(series: pd.Series) -> list:
    """Sort unique item labels, translating unordered-mixed-type errors."""
    try:
        return sorted(series.unique())
    except TypeError as e:
        raise TypeError(
            "item column contains values that cannot be compared (e.g. mixed "
            "str/int or objects without __lt__). Convert the column to a "
            "single comparable dtype before calling find_associations()."
        ) from e


def _check_encoder_capacity(n_unique: int) -> None:
    if n_unique > _MAX_ITEMS:
        raise ValueError(
            f"item cardinality {n_unique} exceeds i32 encoder limit "
            f"({_MAX_ITEMS}). fastapriori encodes items as int32 before "
            "dispatch; split the data or use a smaller catalog."
        )


# ---------------------------------------------------------------------------
# k=2: compute_associations (same signature as pandas/polars backends)
# ---------------------------------------------------------------------------

def compute_associations(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    show_progress: bool,
    trans_dict: dict | None = None,
    min_support: float | None = None,
    min_confidence: float | None = None,
    min_lift: float | None = None,
    min_conviction: float | None = None,
    min_leverage: float | None = None,
    min_cosine: float | None = None,
    min_jaccard: float | None = None,
) -> pd.DataFrame:
    """Compute pairwise associations using the Rust backend.

    When filter thresholds are provided, they are applied on the numeric
    numpy arrays returned by Rust *before* decoding item IDs back to
    original labels and constructing the DataFrame.  This avoids
    materialising an object-dtype DataFrame of millions of rows when the
    item labels are strings — the 7M-row Online Retail case drops from
    ~4s to <1s because only the surviving rows pay the string-decode cost.
    """
    _check_rust()

    clean = df[[transaction_col, item_col]].dropna()

    # Encode items to sequential integers
    unique_items = _sorted_unique_items(clean[item_col])
    _check_encoder_capacity(len(unique_items))
    item_encoder = {item: i for i, item in enumerate(unique_items)}
    item_decoder = np.array(unique_items)

    # Skip pd.factorize when the txn column is already integer — Rust does
    # its own HashMap<i64, u32> factorize inside build_inverted_index.
    # Also skip .nunique(): Rust returns n_transactions in the result dict
    # and takes min_support directly (fraction), computing min_count itself.
    txn_col = clean[transaction_col]
    if pd.api.types.is_integer_dtype(txn_col):
        txn_ids = txn_col.to_numpy(dtype=np.int64, copy=False)
    else:
        txn_codes, _ = pd.factorize(txn_col)
        txn_ids = txn_codes.astype(np.int64)
    item_ids = clean[item_col].map(item_encoder).to_numpy(dtype=np.int32)

    min_support_arg = 0.0 if min_support is None or min_support <= 0 else float(min_support)

    # Call Rust
    result_dict = rust_compute_pairs(
        txn_ids, item_ids, len(unique_items), min_support_arg,
    )

    # Build filter mask on numpy arrays (fast — no object cols involved).
    # NaN comparisons evaluate to False, so NaN rows are dropped here
    # exactly as `df[df[col] >= threshold]` would drop them downstream.
    a_codes = result_dict["item_A"]
    b_codes = result_dict["item_B"]
    instances = result_dict["instances"]
    support = result_dict["support"]
    confidence = result_dict["confidence"]
    lift = result_dict["lift"]
    conviction = result_dict["conviction"]
    leverage = result_dict["leverage"]
    cosine = result_dict["cosine"]
    jaccard = result_dict["jaccard"]

    filters = [
        (min_support, support),
        (min_confidence, confidence),
        (min_lift, lift),
        (min_conviction, conviction),
        (min_leverage, leverage),
        (min_cosine, cosine),
        (min_jaccard, jaccard),
    ]
    mask = None
    for threshold, arr in filters:
        if threshold is None:
            continue
        sub = arr >= threshold
        mask = sub if mask is None else (mask & sub)

    if mask is not None and not mask.all():
        a_codes = a_codes[mask]
        b_codes = b_codes[mask]
        instances = instances[mask]
        support = support[mask]
        confidence = confidence[mask]
        lift = lift[mask]
        conviction = conviction[mask]
        leverage = leverage[mask]
        cosine = cosine[mask]
        jaccard = jaccard[mask]

    # Decode item IDs back to original labels — only for survivors
    result = pd.DataFrame({
        "item_A": item_decoder[a_codes],
        "item_B": item_decoder[b_codes],
        "instances": instances,
        "support": support,
        "confidence": confidence,
        "lift": lift,
        "conviction": conviction,
        "leverage": leverage,
        "cosine": cosine,
        "jaccard": jaccard,
    })
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# k>=3: compute_itemsets_rust (same return type as counter_chain backend)
# ---------------------------------------------------------------------------

def compute_itemsets_rust(
    trans_dict: dict,
    total_transactions: int,
    k: int,
    frequent_lower: pd.DataFrame | None,
    n_workers: int | None,
    show_progress: bool,
    df: pd.DataFrame | None = None,
    transaction_col: str | None = None,
    item_col: str | None = None,
    fast_variant: str = "apriori",
) -> Counter:
    """Compute k-itemset counts using the Rust backend.

    Returns Counter mapping canonical k-tuple -> count (same as
    itemset_counter_chain.compute_itemsets).

    When ``fast_variant="eclat"``, the caller-provided ``frequent_lower``
    is ignored — Eclat rebuilds the full k=2..k recursion internally from
    tid-lists (its natural API).  The chained-caller pattern in
    ``core._find_k_itemsets`` detects eclat and routes through
    ``compute_pipeline`` directly, so this path normally only fires when
    a user explicitly forces the k>=3 intermediate entry point.
    """
    _check_rust()
    if fast_variant == "eclat":
        # Route eclat through compute_pipeline — it runs the vertical
        # recursion from scratch and does not consume frequent_lower.
        if df is None or transaction_col is None or item_col is None:
            raise ValueError(
                "fast_variant='eclat' requires the original DataFrame "
                "(df, transaction_col, item_col) to rebuild tid-lists; "
                "pass them via compute_itemsets_rust(df=..., "
                "transaction_col=..., item_col=...)."
            )
        rules_df = compute_pipeline(
            df=df, transaction_col=transaction_col, item_col=item_col,
            k=k, min_support=0.0, min_confidence=None,
            fast_variant="eclat",
        )
        # Collapse rules back to Counter<canonical k-tuple, count> so this
        # function's return shape matches the counter_chain backend.
        ant_cols = [f"antecedent_{i}" for i in range(1, k)]
        cols = ant_cols + ["consequent"]
        counter: Counter = Counter()
        if len(rules_df) == 0:
            return counter
        items = rules_df[cols].to_numpy()
        counts = rules_df["instances"].to_numpy()
        for i in range(len(counts)):
            key = tuple(sorted(items[i].tolist()))
            counter[key] = int(counts[i])
        return counter

    # --- Encode items to integers ---
    # Use DataFrame directly if available (avoids expensive trans_dict flatten)
    if df is not None and transaction_col and item_col:
        clean = df[[transaction_col, item_col]].dropna()
        unique_items = _sorted_unique_items(clean[item_col])
        _check_encoder_capacity(len(unique_items))
        item_encoder = {item: i for i, item in enumerate(unique_items)}
        item_decoder = {i: item for item, i in item_encoder.items()}
        # Skip pd.factorize for integer txn cols — Rust factorises internally.
        txn_col_series = clean[transaction_col]
        if pd.api.types.is_integer_dtype(txn_col_series):
            txn_ids = txn_col_series.to_numpy(dtype=np.int64, copy=False)
        else:
            txn_codes, _ = pd.factorize(txn_col_series)
            txn_ids = txn_codes.astype(np.int64)
        item_ids = clean[item_col].map(item_encoder).to_numpy(dtype=np.int32)
    else:
        all_items: set = set()
        for items in trans_dict.values():
            all_items.update(items)
        try:
            unique_items = sorted(all_items)
        except TypeError as e:
            raise TypeError(
                "transaction dict contains items that cannot be compared "
                "(mixed str/int, or objects without __lt__). Use a single "
                "comparable dtype."
            ) from e
        _check_encoder_capacity(len(unique_items))
        item_encoder = {item: i for i, item in enumerate(unique_items)}
        item_decoder = {i: item for item, i in item_encoder.items()}

        txn_list = []
        item_list = []
        for txn_id, items in trans_dict.items():
            for item in items:
                txn_list.append(txn_id)
                item_list.append(item_encoder[item])
        txn_ids = np.array(txn_list, dtype=np.int64)
        item_ids = np.array(item_list, dtype=np.int32)

    # --- Extract frequent (k-1)-sets as 2D int array ---
    freq_lower_sets = _extract_freq_lower_encoded(frequent_lower, k, item_encoder)

    # Weights are only consumed inside max_items_per_txn capping, which is
    # always None here (this entry point doesn't plumb it through).  Skip
    # the groupby and pass an empty placeholder — Rust uses .get() with an
    # unwrap_or default so an empty slice is safe.
    n_unique = len(unique_items)
    weights_arr = np.empty(0, dtype=np.float64)

    # Call Rust
    result_dict = rust_compute_itemsets(
        txn_ids, item_ids, freq_lower_sets, k, len(unique_items),
        weights_arr, None,
    )

    # Decode integer itemsets back to original labels
    itemsets_arr = result_dict["itemsets"]  # 2D numpy array (n_results, k)
    counts_arr = result_dict["counts"]     # 1D numpy array

    result = Counter()
    for i in range(len(counts_arr)):
        kset = tuple(item_decoder[int(itemsets_arr[i, j])] for j in range(k))
        result[kset] = int(counts_arr[i])

    return result


def _extract_freq_lower_encoded(
    frequent_lower: pd.DataFrame | None,
    k: int,
    item_encoder: dict,
) -> np.ndarray:
    """Extract frequent (k-1)-sets from DataFrame, encode to int32 2D array."""
    lower_k = k - 1

    if frequent_lower is None:
        return np.empty((0, lower_k), dtype=np.int32)

    sets = []

    # Pair-level columns (from k=2 output)
    if "item_A" in frequent_lower.columns and "item_B" in frequent_lower.columns:
        seen = set()
        for a, b in zip(frequent_lower["item_A"], frequent_lower["item_B"]):
            canonical = tuple(sorted([a, b]))
            if canonical not in seen:
                seen.add(canonical)
                encoded = tuple(item_encoder[x] for x in canonical)
                sets.append(encoded)
    else:
        # Higher-level: antecedent_* + consequent columns
        ant_cols = [c for c in frequent_lower.columns if c.startswith("antecedent_")]
        if ant_cols and "consequent" in frequent_lower.columns:
            seen = set()
            arr = frequent_lower[ant_cols + ["consequent"]].to_numpy()
            for row in arr:
                items = tuple(sorted(row.tolist()))
                if items not in seen:
                    seen.add(items)
                    encoded = tuple(item_encoder[x] for x in items)
                    sets.append(encoded)

    if not sets:
        return np.empty((0, lower_k), dtype=np.int32)

    return np.array(sets, dtype=np.int32)


# ---------------------------------------------------------------------------
# Shared vectorized decode: itemset arrays → rules DataFrame
# ---------------------------------------------------------------------------


def decode_pipeline_rules(
    itemsets_arr: np.ndarray,
    counts_arr: np.ndarray,
    lower_arr: np.ndarray,
    lower_counts_arr: np.ndarray,
    item_counts_arr: np.ndarray,
    item_decoder: np.ndarray,
    n_transactions: int,
    k: int,
    min_support: float | None = None,
    min_confidence: float | None = None,
) -> pd.DataFrame:
    """Vectorized decode of Rust pipeline output to directional rules.

    Works entirely in encoded int32 space for lookups, decodes only
    when building the final DataFrame.  ~10x faster than the equivalent
    Python loop for large output (>100K itemsets).
    """
    ant_cols = [f"antecedent_{i}" for i in range(1, k)]
    out_cols = ant_cols + ["consequent", "instances", "support",
                           "confidence", "lift"]

    n_results = len(counts_arr)
    if n_results == 0:
        return pd.DataFrame(columns=out_cols)

    # 1. Expand each itemset to k rules (one per consequent position)
    ant_col_indices = np.array(
        [[c for c in range(k) if c != j] for j in range(k)], dtype=np.intp,
    )  # shape (k, k-1)

    expanded = np.repeat(itemsets_arr, k, axis=0)       # (n_rules, k)
    rule_j = np.tile(np.arange(k), n_results)            # consequent column
    n_rules = len(expanded)

    # 2. Extract antecedents (encoded) and consequent (encoded)
    ant_indices = ant_col_indices[rule_j]                 # (n_rules, k-1)
    antecedents_enc = expanded[np.arange(n_rules)[:, None], ant_indices]
    antecedents_sorted = np.sort(antecedents_enc, axis=1)  # for merge keys
    consequent_enc = expanded[np.arange(n_rules), rule_j]

    # 3. Lower support lookup via pandas merge (encoded int32 keys)
    merge_cols = [f"_c{j}" for j in range(k - 1)]
    df_rules = pd.DataFrame(antecedents_sorted, columns=merge_cols)
    df_lower = pd.DataFrame(lower_arr, columns=merge_cols)
    df_lower["_ant_sup"] = lower_counts_arr.astype(np.float64) / n_transactions
    df_rules = df_rules.merge(df_lower, on=merge_cols, how="left")
    ant_sup = df_rules["_ant_sup"].fillna(0).values

    # 4. Item support — direct array indexing (encoded item ID → count)
    cons_sup = item_counts_arr[consequent_enc].astype(np.float64) / n_transactions

    # 5. Metrics (vectorized)
    instances = np.repeat(counts_arr, k)
    support = instances.astype(np.float64) / n_transactions
    confidence = support / (ant_sup + 1e-10)
    lift = confidence / (cons_sup + 1e-10)

    # 6. Decode to original labels and build DataFrame
    decoded_ant = item_decoder[antecedents_enc]
    decoded_con = item_decoder[consequent_enc]

    data = {}
    for j in range(k - 1):
        data[ant_cols[j]] = decoded_ant[:, j]
    data["consequent"] = decoded_con
    data["instances"] = instances
    data["support"] = np.round(support, 6)
    data["confidence"] = np.round(confidence, 6)
    data["lift"] = np.round(lift, 6)

    result = pd.DataFrame(data)

    if min_support is not None and min_support > 0:
        result = result[result["support"] >= min_support]
    if min_confidence is not None:
        result = result[result["confidence"] >= min_confidence]

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full pipeline: k=2 → k=3 → ... → k_max in one Rust call
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
    fast_variant: str = "apriori",
    impl_variant: str = "v3_adaptive",
) -> pd.DataFrame:
    """Full pipeline: encode once, Rust computes k=2..k_max, decode once.

    Returns the same DataFrame schema as core._find_k_itemsets:
    antecedent_1, ..., antecedent_{k-1}, consequent, instances, support,
    confidence, lift.
    """
    _check_rust()
    if fast_variant == "eclat" and max_items_per_txn is not None:
        raise ValueError(
            "fast_variant='eclat' does not support max_items_per_txn; "
            "transaction capping is an Apriori-path approximation knob "
            "with no direct Eclat analogue. Use "
            "fast_variant='apriori' if you need capping."
        )

    clean = df[[transaction_col, item_col]].dropna()
    unique_items = _sorted_unique_items(clean[item_col])
    _check_encoder_capacity(len(unique_items))
    item_encoder = {item: i for i, item in enumerate(unique_items)}
    item_decoder = np.array(unique_items)

    # Skip pd.factorize when the txn column is already integer — Rust does
    # its own HashMap<i64, u32> factorize inside build_inverted_index, so
    # pd.factorize would just be a redundant O(n) Python-side pass.
    # Also skip .nunique(): Rust returns n_transactions in the result dict.
    txn_col = clean[transaction_col]
    if pd.api.types.is_integer_dtype(txn_col):
        txn_ids = txn_col.to_numpy(dtype=np.int64, copy=False)
    else:
        txn_codes, _ = pd.factorize(txn_col)
        txn_ids = txn_codes.astype(np.int64)
    item_ids = clean[item_col].map(item_encoder).to_numpy(dtype=np.int32)

    n_unique = len(unique_items)

    if fast_variant == "eclat":
        # Eclat ignores item_weights / max_items_per_txn (rejected above).
        result_dict = rust_eclat_pipeline(
            txn_ids, item_ids, k, n_unique, min_support,
        )
    else:
        # Build item weights array (indexed by encoded item ID).  NaN weights
        # are coerced to 0 because Rust's truncation sort uses partial_cmp,
        # which maps NaN→Equal and makes top-N capping non-deterministic.
        # Weights are only consumed inside the max_items_per_txn capping
        # branch in src/itemsets.rs — skip the groupby when capping is off.
        if max_items_per_txn is None and item_weights is None:
            weights_arr = np.empty(0, dtype=np.float64)
        elif item_weights is not None:
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

        _pipeline_fn = _resolve_pipeline_variant(impl_variant)
        result_dict = _pipeline_fn(
            txn_ids, item_ids, k, n_unique, min_support,
            weights_arr, max_items_per_txn,
        )

    # Unpack results
    itemsets_arr = result_dict["itemsets"]       # 2D int32 (n_results, k)
    counts_arr = result_dict["counts"]           # 1D int64
    lower_arr = result_dict["lower_itemsets"]    # 2D int32 (n_lower, k-1)
    lower_counts_arr = result_dict["lower_counts"]  # 1D int64
    item_counts_arr = result_dict["item_counts"]    # 1D int64 (n_items,)
    n_transactions = int(result_dict["n_transactions"])

    return decode_pipeline_rules(
        itemsets_arr, counts_arr, lower_arr, lower_counts_arr,
        item_counts_arr, item_decoder, n_transactions, k,
        min_support, min_confidence,
    )
