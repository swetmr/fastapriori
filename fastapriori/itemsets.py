"""Generalized k-itemset co-occurrence analysis (k=2,3,4,5)."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

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


def find_itemsets(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    k: int = 3,
    min_support: float | None = None,
    min_confidence: float | None = 0.1,
    frequent_lower: pd.DataFrame | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute k-itemset co-occurrence associations from transactional data.

    For every transaction, generates all k-item combinations and counts their
    global frequency, then computes support, confidence, and lift for each
    directional rule.

    Uses Apriori pruning when ``frequent_lower`` is provided: only itemsets
    whose **all** C(k,2) pairs appear in the frequent-pairs set are counted.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with transaction and item columns.
    transaction_col : str
        Name of the transaction ID column.
    item_col : str
        Name of the item column.
    k : int
        Itemset size (2-5). Default 3.
    min_support : float or None
        Minimum support threshold. None disables filtering.
    min_confidence : float or None
        Minimum confidence threshold. Default 0.1 (10%).
    frequent_lower : pd.DataFrame or None
        Results from the (k-1) level, used for two purposes:

        1. **Apriori pruning** — for k>=3, pair-level pruning is extracted
           from the k=2 results (columns item_A, item_B) or from the
           antecedent/consequent columns. Only itemsets whose all C(k,2) pairs
           are frequent are counted.
        2. **Confidence computation** — support of the (k-1)-itemset antecedent
           is looked up from this DataFrame. For k=3 this means pair support;
           for k=4 this means triplet support; etc.

        When ``k=3``, this should be the output of ``find_associations()``
        (pair-level). When ``k=4``, this should be the output of
        ``find_itemsets(k=3)``, and so on.
    show_progress : bool
        Show progress bar during counting.

    Returns
    -------
    pd.DataFrame
        Columns: antecedent_1, ..., antecedent_{k-1}, consequent,
        instances, support, confidence, lift

        Each k-itemset produces k directional rules (one per item as
        consequent).
    """
    if k < 2 or k > 5:
        raise ValueError(f"k must be between 2 and 5, got {k}")
    if transaction_col not in df.columns:
        raise ValueError(f"Column '{transaction_col}' not found in DataFrame")
    if item_col not in df.columns:
        raise ValueError(f"Column '{item_col}' not found in DataFrame")

    df = df.dropna(subset=[transaction_col, item_col])

    # Build transaction dict: transaction_id -> set(items)
    trans_dict: dict = df.groupby(transaction_col)[item_col].apply(set).to_dict()
    total_transactions = len(trans_dict)

    # --- Build frequent pair set for Apriori pruning (used for k >= 3) ---
    freq_pair_set: set | None = None
    if frequent_lower is not None and k >= 3:
        freq_pair_set = _extract_freq_pairs(frequent_lower, k)

    # --- Build (k-1)-itemset support lookup for confidence computation ---
    lower_support: dict[tuple, float] = {}
    if frequent_lower is not None:
        lower_support = _extract_lower_support(frequent_lower, k, total_transactions)

    # --- Count k-itemsets across all transactions ---
    itemset_counts: Counter = Counter()
    iterator = trans_dict.values()
    if show_progress:
        iterator = _wrap_progress(
            iterator, total=total_transactions, desc=f"Counting {k}-itemsets"
        )

    for items in iterator:
        if len(items) < k:
            continue
        sorted_items = sorted(items)
        for itemset in combinations(sorted_items, k):
            # Apriori pruning: check all C(k,2) pairs are frequent
            if freq_pair_set is not None:
                if not all(pair in freq_pair_set for pair in combinations(itemset, 2)):
                    continue
            itemset_counts[itemset] += 1

    # --- Build output columns ---
    ant_cols = [f"antecedent_{i}" for i in range(1, k)]
    out_cols = ant_cols + ["consequent", "instances", "support", "confidence", "lift"]

    if not itemset_counts:
        return pd.DataFrame(columns=out_cols)

    # --- Build (k-1)-itemset support from counting if not available ---
    # For confidence we need support of the antecedent (k-1)-itemset.
    # If frequent_lower was not provided, we compute (k-1)-itemset counts
    # directly from transactions.
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
            records.append((*antecedents, consequent, count, support, confidence, lift))

    result = pd.DataFrame(records, columns=out_cols)
    result["support"] = np.round(result["support"], 6)
    result["confidence"] = np.round(result["confidence"], 6)
    result["lift"] = np.round(result["lift"], 6)

    # Filter
    if min_support is not None:
        result = result[result["support"] >= min_support]
    if min_confidence is not None:
        result = result[result["confidence"] >= min_confidence]

    result = result.reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_freq_pairs(frequent_lower: pd.DataFrame, k: int) -> set:
    """Extract the set of frequent pairs from the lower-level DataFrame.

    For k=3: frequent_lower is pair-level (has item_A, item_B).
    For k=4: frequent_lower is triplet-level — we need to extract all
             constituent pairs from the antecedent/consequent columns, but
             it's simpler and more reliable to re-derive from the itemsets.
    For k>=4 we walk down to pair level by extracting all C(k-1, 2) pairs
    from each (k-1)-itemset in the DataFrame.
    """
    freq_pairs: set = set()

    # Try pair-level columns first (output of find_associations)
    if "item_A" in frequent_lower.columns and "item_B" in frequent_lower.columns:
        for _, row in frequent_lower[["item_A", "item_B"]].iterrows():
            freq_pairs.add(tuple(sorted([row["item_A"], row["item_B"]])))
        return freq_pairs

    # Otherwise, reconstruct itemsets from antecedent_* + consequent columns
    ant_cols = [c for c in frequent_lower.columns if c.startswith("antecedent_")]
    if ant_cols and "consequent" in frequent_lower.columns:
        for _, row in frequent_lower[ant_cols + ["consequent"]].iterrows():
            items = tuple(sorted([row[c] for c in ant_cols] + [row["consequent"]]))
            # Extract all pairs from this itemset
            for pair in combinations(items, 2):
                freq_pairs.add(pair)
        return freq_pairs

    return freq_pairs


def _extract_lower_support(
    frequent_lower: pd.DataFrame, k: int, total_transactions: int
) -> dict[tuple, float]:
    """Build a dict of (k-1)-itemset -> support from the lower-level DataFrame.

    For k=3: lower is pair-level with item_A/item_B or antecedent_1/consequent.
    For k=4+: lower has antecedent_1..antecedent_{k-2}, consequent.

    Each (k-1)-itemset appears multiple times (once per directional rule).
    We use the first occurrence's support since they share the same itemset count.
    """
    support_dict: dict[tuple, float] = {}

    # Pair-level (from find_associations output)
    if "item_A" in frequent_lower.columns and "item_B" in frequent_lower.columns:
        if "support" in frequent_lower.columns:
            for _, row in frequent_lower[["item_A", "item_B", "support"]].iterrows():
                key = tuple(sorted([row["item_A"], row["item_B"]]))
                if key not in support_dict:
                    support_dict[key] = row["support"]
        elif "instances" in frequent_lower.columns:
            for _, row in frequent_lower[["item_A", "item_B", "instances"]].iterrows():
                key = tuple(sorted([row["item_A"], row["item_B"]]))
                if key not in support_dict:
                    support_dict[key] = row["instances"] / total_transactions
        return support_dict

    # Higher-level (antecedent_* + consequent)
    ant_cols = [c for c in frequent_lower.columns if c.startswith("antecedent_")]
    if ant_cols and "consequent" in frequent_lower.columns:
        need_cols = ant_cols + ["consequent"]
        has_support = "support" in frequent_lower.columns
        has_instances = "instances" in frequent_lower.columns
        if has_support:
            need_cols.append("support")
        elif has_instances:
            need_cols.append("instances")

        for _, row in frequent_lower[need_cols].iterrows():
            items = tuple(sorted([row[c] for c in ant_cols] + [row["consequent"]]))
            if items not in support_dict:
                if has_support:
                    support_dict[items] = row["support"]
                elif has_instances:
                    support_dict[items] = row["instances"] / total_transactions

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
