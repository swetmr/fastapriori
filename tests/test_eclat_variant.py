"""Correctness oracle for backend_options={'fast_variant': 'eclat'}.

Asserts the Eclat vertical-recursion path returns the same set of itemsets
(and same counts) as the Apriori reference path on a deterministic small
dataset and a moderate synthetic dataset.  Both paths go through
``find_associations`` and end up in the Rust pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastapriori import find_associations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _small_df():
    """8 transactions, 6 items, enough density for k=3, k=4 to have survivors."""
    rows = [
        (1, "A"), (1, "B"), (1, "C"), (1, "D"),
        (2, "A"), (2, "B"), (2, "C"),
        (3, "A"), (3, "B"), (3, "D"), (3, "E"),
        (4, "B"), (4, "C"), (4, "D"), (4, "E"),
        (5, "A"), (5, "C"), (5, "D"), (5, "E"), (5, "F"),
        (6, "A"), (6, "B"), (6, "E"), (6, "F"),
        (7, "A"), (7, "B"), (7, "C"), (7, "D"), (7, "E"),
        (8, "B"), (8, "C"), (8, "D"),
    ]
    return pd.DataFrame(rows, columns=["txn", "item"])


def _synthetic_df(n_txn: int = 200, n_items: int = 30, avg_d: int = 8, seed: int = 0):
    """Random transactions with controllable shape. Integer items so np.sort works."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_txn):
        d = max(2, int(rng.poisson(avg_d)))
        d = min(d, n_items)
        items = rng.choice(n_items, size=d, replace=False)
        for it in items:
            rows.append((t, int(it)))
    return pd.DataFrame(rows, columns=["txn", "item"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _itemset_counter(df: pd.DataFrame, k: int) -> dict:
    """Collapse k-itemsets (rules-form output) into {frozenset(items): count}.
    Each k-itemset appears k times in rules output (one per consequent);
    all k appearances must share the same `instances` count."""
    if k == 2:
        a = df["item_A"].to_numpy()
        b = df["item_B"].to_numpy()
        out: dict = {}
        for i in range(len(df)):
            key = frozenset((a[i], b[i]))
            out[key] = int(df["instances"].iat[i])
        return out
    cols = [f"antecedent_{i}" for i in range(1, k)] + ["consequent"]
    items = df[cols].to_numpy()
    counts = df["instances"].to_numpy()
    out = {}
    for i in range(len(counts)):
        key = frozenset(items[i].tolist())
        out[key] = int(counts[i])
    return out


# ---------------------------------------------------------------------------
# Correctness: eclat and apriori produce identical itemsets + counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [3, 4])
def test_eclat_matches_apriori_small(k):
    df = _small_df()
    apriori = find_associations(
        df, "txn", "item", k=k, min_support=0.2,
        backend_options={"fast_variant": "apriori"},
    )
    eclat = find_associations(
        df, "txn", "item", k=k, min_support=0.2,
        backend_options={"fast_variant": "eclat"},
    )
    assert _itemset_counter(apriori, k) == _itemset_counter(eclat, k)


@pytest.mark.parametrize("k", [3, 4])
def test_eclat_matches_apriori_synthetic(k):
    df = _synthetic_df(n_txn=300, n_items=25, avg_d=6, seed=42)
    apriori = find_associations(
        df, "txn", "item", k=k, min_support=0.05,
        backend_options={"fast_variant": "apriori"},
    )
    eclat = find_associations(
        df, "txn", "item", k=k, min_support=0.05,
        backend_options={"fast_variant": "eclat"},
    )
    assert _itemset_counter(apriori, k) == _itemset_counter(eclat, k)


# ---------------------------------------------------------------------------
# Guardrails: unsupported combinations raise clearly
# ---------------------------------------------------------------------------

def test_eclat_rejects_max_items_per_txn():
    df = _small_df()
    with pytest.raises(ValueError, match="max_items_per_txn"):
        find_associations(
            df, "txn", "item", k=3, min_support=0.1,
            max_items_per_txn=5,
            backend_options={"fast_variant": "eclat"},
        )


def test_invalid_fast_variant_rejected():
    df = _small_df()
    with pytest.raises(ValueError, match="fast_variant"):
        find_associations(
            df, "txn", "item", k=3, min_support=0.1,
            backend_options={"fast_variant": "fpgrowth"},
        )


def test_eclat_ignores_frequent_lower():
    """Chained caller passes k=2 result as frequent_lower for k=3; Eclat must
    ignore it (rebuilds internally) and still produce the same answer."""
    df = _small_df()
    k2 = find_associations(
        df, "txn", "item", k=2, min_support=0.2,
        backend_options={"fast_variant": "eclat"},
    )
    with_chain = find_associations(
        df, "txn", "item", k=3, min_support=0.2,
        frequent_lower=k2,
        backend_options={"fast_variant": "eclat"},
    )
    standalone = find_associations(
        df, "txn", "item", k=3, min_support=0.2,
        backend_options={"fast_variant": "eclat"},
    )
    assert _itemset_counter(with_chain, 3) == _itemset_counter(standalone, 3)
