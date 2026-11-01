"""Correctness oracle for the opt-in 0.2.0 kernels.

* backend_options={'pair_variant': ...} at k=2 must return exactly the same
  rows (and metric values) as the default v0 kernel.
* backend_options={'fast_variant': 'eclat_seeded'} at k>=3 must return the
  same itemsets and counts as the default anchor-and-extend path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastapriori import find_associations

rs = pytest.importorskip("fastapriori._fastapriori_rs")
if not hasattr(rs, "rust_compute_pairs_v2"):
    pytest.skip("Rust build predates the 0.2.0 variants", allow_module_level=True)

PAIR_VARIANTS = ["suffix", "filtered", "suffix_filtered"]
METRICS = ["support", "confidence", "lift", "conviction", "leverage", "cosine", "jaccard"]


def _synthetic_df(n_txn=400, n_items=40, avg_d=7, seed=0, str_items=False):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_txn):
        d = min(max(1, int(rng.poisson(avg_d))), n_items)
        # Skewed item popularity so some items fall below min_count.
        p = rng.dirichlet(np.full(n_items, 0.3))
        for it in rng.choice(n_items, size=d, replace=False, p=p):
            rows.append((t, f"i{it:03d}" if str_items else int(it)))
    return pd.DataFrame(rows, columns=["txn", "item"])


def _canon_pairs(df):
    return df.sort_values(["item_A", "item_B"]).reset_index(drop=True)


@pytest.mark.parametrize("variant", PAIR_VARIANTS)
@pytest.mark.parametrize("min_support", [None, 0.01, 0.05])
@pytest.mark.parametrize("str_items", [False, True])
def test_pair_variants_match_v0(variant, min_support, str_items):
    df = _synthetic_df(str_items=str_items)
    kw = dict(transaction_col="txn", item_col="item", k=2,
              min_support=min_support, sorted_by=None, backend="rust")
    ref = _canon_pairs(find_associations(df, **kw))
    got = _canon_pairs(find_associations(df, backend_options={"pair_variant": variant}, **kw))
    assert len(ref) > 0
    pd.testing.assert_frame_equal(ref[["item_A", "item_B", "instances"]],
                                  got[["item_A", "item_B", "instances"]])
    for m in METRICS:
        np.testing.assert_allclose(ref[m].to_numpy(), got[m].to_numpy(),
                                   rtol=1e-12, equal_nan=True)


def _itemset_counts(rules: pd.DataFrame, k: int) -> dict:
    cols = [f"antecedent_{i}" for i in range(1, k)] + ["consequent"]
    out = {}
    for row, inst in zip(rules[cols].to_numpy(), rules["instances"].to_numpy()):
        key = frozenset(row.tolist())
        assert out.setdefault(key, int(inst)) == int(inst)
    return out


@pytest.mark.parametrize("k", [3, 4, 5])
@pytest.mark.parametrize("min_support", [0.005, 0.02])
def test_eclat_seeded_matches_apriori(k, min_support):
    df = _synthetic_df(n_txn=500, n_items=15, avg_d=9, seed=k)
    kw = dict(transaction_col="txn", item_col="item", k=k,
              min_support=min_support, sorted_by=None, backend="rust")
    ref = _itemset_counts(find_associations(df, **kw), k)
    got = _itemset_counts(
        find_associations(df, backend_options={"fast_variant": "eclat_seeded"}, **kw), k)
    plain = _itemset_counts(
        find_associations(df, backend_options={"fast_variant": "eclat"}, **kw), k)
    assert len(ref) > 0
    assert got == ref
    assert plain == ref


def test_eclat_seeded_k2_kernel_direct():
    """k_max=2 short-circuit in the seeded kernel returns exact pair counts."""
    df = _synthetic_df(n_txn=300, n_items=20, seed=7)
    txn = df["txn"].to_numpy(np.int64)
    items = df["item"].to_numpy(np.int32)
    out = rs.rust_eclat_pipeline_seeded(txn, items, 2, 20, 0.02)
    got = {(int(a), int(b)): int(c) for (a, b), c in zip(out["itemsets"], out["counts"])}
    ref = find_associations(df, transaction_col="txn", item_col="item", k=2,
                            min_support=0.02, sorted_by=None, backend="rust")
    ref = ref[ref.item_A < ref.item_B]
    assert got == {(int(a), int(b)): int(c)
                   for a, b, c in ref[["item_A", "item_B", "instances"]].to_numpy()}
