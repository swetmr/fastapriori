"""Regression tests: exact minimum count, unrounded k>=3 filtering, deterministic row order,
the Rust row-capacity guard, NaN-only transactions, exact antecedent counts, and metric
thresholds on exact counts and final rules only."""
import numpy as np
import pandas as pd
import pytest

from fastapriori import find_associations
from fastapriori.backends import rust_backend


def _exactly_seven_of_hundred():
    """100 transactions; items 0,1,2 co-occur in exactly 7 of them (support 0.07)."""
    rows = []
    for t in range(100):
        if t < 7:
            rows += [(t, 0), (t, 1), (t, 2)]
        else:
            rows.append((t, 10 + t))      # singleton filler, all distinct
    return pd.DataFrame(rows, columns=["txn", "item"])


@pytest.mark.parametrize("algo", ["fast", "classic"])
def test_pairs_at_exact_threshold_are_kept(algo):
    df = _exactly_seven_of_hundred()
    out = find_associations(df, "txn", "item", k=2, min_support=0.07, algo=algo, sorted_by=None)
    pairs = {frozenset((a, b)) for a, b in zip(out.item_A, out.item_B)}
    assert pairs == {frozenset(p) for p in [(0, 1), (0, 2), (1, 2)]}


@pytest.mark.parametrize("bo", [None, {"pair_variant": "suffix_filtered"}])
def test_pair_variants_at_exact_threshold(bo):
    df = _exactly_seven_of_hundred()
    out = find_associations(df, "txn", "item", k=2, min_support=0.07, sorted_by=None,
                            backend_options=bo)
    assert len(out) == 6          # 3 pairs, both directions


@pytest.mark.parametrize("bo", [None, {"fast_variant": "eclat"}, {"fast_variant": "eclat_seeded"}])
def test_triples_at_exact_threshold(bo):
    df = _exactly_seven_of_hundred()
    out = find_associations(df, "txn", "item", k=3, min_support=0.07, sorted_by=None,
                            backend_options=bo)
    assert len(out) == 3 and (out.instances == 7).all()


def _two_of_seven():
    # 7 transactions, triple in 2 of them: support 0.285714285...; threshold below has > 6 decimals
    rows = [(t, i) for t in range(2) for i in (0, 1, 2)] + [(t, 100 + t) for t in range(2, 7)]
    return pd.DataFrame(rows, columns=["txn", "item"])


@pytest.mark.parametrize("backend", ["auto", "rust", "python", "counter_chain", "pandas"])
def test_k3_filter_uses_unrounded_support(backend):
    out = find_associations(_two_of_seven(), "txn", "item", k=3, min_support=0.2857142,
                            sorted_by=None, backend=backend)
    assert len(out) == 3


@pytest.mark.parametrize("backend,algo", [("rust", "fast"), ("pandas", "fast"), ("auto", "classic")])
def test_k2_filter_uses_unrounded_support(backend, algo):
    out = find_associations(_two_of_seven(), "txn", "item", k=2, min_support=0.2857142,
                            sorted_by=None, backend=backend, algo=algo)
    assert len(out) == 6


def test_sorted_output_is_deterministic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"txn": rng.integers(0, 400, 6000), "item": rng.integers(0, 40, 6000)})
    a = find_associations(df, "txn", "item", k=3, min_support=0.01)
    b = find_associations(df, "txn", "item", k=3, min_support=0.01)
    pd.testing.assert_frame_equal(a, b)
    cols = ["antecedent_1", "antecedent_2", "consequent"]
    # ties on support are ordered by the item columns
    for _, g in a.groupby("support", sort=False):
        assert g[cols].apply(tuple, axis=1).is_monotonic_increasing


def test_row_capacity_guard():
    with pytest.raises(ValueError, match="Rust index limit"):
        rust_backend._check_row_capacity(2**32)
    rust_backend._check_row_capacity(2**32 - 1)


@pytest.mark.parametrize("backend", ["rust", "python", "counter_chain"])
@pytest.mark.parametrize("bo", [None, {"fast_variant": "eclat_seeded"}])
def test_k3_confidence_exactly_one_kept(backend, bo):
    # items 0,1,2 always together in 3 of 10 transactions: every rule has confidence exactly 1.0
    rows = [(t, i) for t in range(3) for i in (0, 1, 2)] + [(t, 50 + t) for t in range(3, 10)]
    df = pd.DataFrame(rows, columns=["txn", "item"])
    if backend != "rust" and bo is not None:
        pytest.skip("backend_options apply to the Rust backend only")
    out = find_associations(df, "txn", "item", k=3, min_support=0.1, min_confidence=1.0,
                            sorted_by=None, backend=backend, backend_options=bo)
    assert len(out) == 3 and (out.confidence == 1.0).all()


@pytest.mark.parametrize("backend", ["pandas", "polars", "rust"])
def test_nan_only_transaction_not_in_support_denominator(backend):
    """99 real transactions + 1 with only a NaN item; the pair sits exactly at min_support."""
    if backend == "polars":
        pytest.importorskip("polars")
    rows = [(t, f"filler{t}") for t in range(99)]
    rows += [(t, it) for t in range(7) for it in ("A", "B")]
    rows.append((999, np.nan))
    df = pd.DataFrame(rows, columns=["txn", "item"])
    r = find_associations(df, transaction_col="txn", item_col="item", min_support=7 / 99,
                          backend=backend, sorted_by=None)
    assert len(r) == 2


def test_lower_support_uses_exact_counts():
    """Above ~1M transactions a 6-decimal support no longer recovers the count."""
    from fastapriori.core import _extract_lower_support

    n, count = 3_000_000, 1_234_567
    pairs = pd.DataFrame({"item_A": ["a"], "item_B": ["b"], "instances": [count],
                          "support": [round(count / n, 6)]})
    triples = pd.DataFrame({"antecedent_1": ["a"], "antecedent_2": ["b"], "consequent": ["c"],
                            "instances": [count], "support": [round(count / n, 6)]})
    assert round(pairs["support"].iloc[0] * n) != count    # the case being guarded
    assert round(_extract_lower_support(pairs, 3, n)[("a", "b")] * n) == count
    assert round(_extract_lower_support(triples, 4, n)[("a", "b", "c")] * n) == count


def _baskets(*baskets):
    return pd.DataFrame([(t, it) for t, b in enumerate(baskets) for it in b], columns=["txn", "item"])


@pytest.mark.parametrize("backend", ["rust", "counter_chain", "pandas"])
def test_confidence_applies_to_final_rules_only(backend):
    """The triples have confidence 0.5, the four 4-itemset rules have confidence 1.0."""
    df = _baskets("ABCD", "AB", "AC", "AD", "BC", "BD", "CD")
    r = find_associations(df, transaction_col="txn", item_col="item", k=4, min_support=0.1,
                          min_confidence=0.75, backend=backend, sorted_by=None)
    assert len(r) == 4 and (r.confidence == 1.0).all()


@pytest.mark.parametrize("backend", ["rust", "pandas", "polars"])
def test_pair_confidence_threshold_on_exact_counts(backend):
    """A -> B has confidence exactly 1/3; the rounded column holds 0.333333."""
    if backend == "polars":
        pytest.importorskip("polars")
    df = _baskets("AB", "A", "A", "C")
    r = find_associations(df, transaction_col="txn", item_col="item", min_confidence=1 / 3,
                          backend=backend, sorted_by=None)
    assert ((r.item_A == "A") & (r.item_B == "B")).sum() == 1


def test_exact_metric_filter_matches_rounded_columns():
    """Away from a threshold the exact filter keeps what the rounded columns would keep."""
    df = _baskets("ABC", "ABD", "AC", "BCD", "ABCD", "D")
    base = find_associations(df, transaction_col="txn", item_col="item", backend="pandas", sorted_by=None)
    for col, t in [("confidence", 0.55), ("lift", 0.9), ("leverage", 0.0), ("cosine", 0.6), ("jaccard", 0.4)]:
        r = find_associations(df, transaction_col="txn", item_col="item", backend="pandas", sorted_by=None,
                              **{f"min_{col}": t})
        assert len(r) == (base[col] >= t).sum(), col
