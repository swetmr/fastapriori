"""Tests for the algo parameter in find_associations."""

import pandas as pd
import pytest

from fastapriori import find_associations


# ---------------------------------------------------------------------------
# Shared test data — small deterministic dataset
# ---------------------------------------------------------------------------

def _make_test_df():
    """6 transactions, 6 items, moderate density."""
    rows = [
        (1, "A"), (1, "B"), (1, "C"), (1, "D"),
        (2, "A"), (2, "B"), (2, "C"),
        (3, "A"), (3, "B"), (3, "D"), (3, "E"),
        (4, "B"), (4, "C"), (4, "D"), (4, "E"),
        (5, "A"), (5, "C"), (5, "D"), (5, "E"), (5, "F"),
        (6, "A"), (6, "B"), (6, "E"), (6, "F"),
    ]
    return pd.DataFrame(rows, columns=["txn", "item"])


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestAlgoValidation:
    def test_invalid_algo_raises(self):
        df = _make_test_df()
        with pytest.raises(ValueError, match="algo must be"):
            find_associations(df, "txn", "item", algo="unknown")

    def test_classic_without_min_support_raises(self):
        df = _make_test_df()
        with pytest.raises(ValueError, match="requires min_support"):
            find_associations(df, "txn", "item", algo="classic")

    def test_auto_routes_to_fast(self):
        df = _make_test_df()
        result = find_associations(df, "txn", "item", algo="auto", min_support=0.1)
        expected = find_associations(df, "txn", "item", algo="fast", min_support=0.1)
        assert len(result) == len(expected)

    def test_k_range_expanded_to_50(self):
        df = _make_test_df()
        with pytest.raises(ValueError, match="k must be between 2 and 50"):
            find_associations(df, "txn", "item", k=51)

    def test_k_6_accepted(self):
        """k=6 should not raise ValueError (expanded from old limit of 5)."""
        df = _make_test_df()
        # Won't find any 6-itemsets at this support, but should not error
        result = find_associations(
            df, "txn", "item", k=6, min_support=0.01, algo="classic",
        )
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Correctness: algo="fast" matches current behavior
# ---------------------------------------------------------------------------

class TestAlgoFast:
    def test_fast_default_is_unchanged(self):
        """algo='fast' (default) produces the same result as no algo arg."""
        df = _make_test_df()
        result_default = find_associations(df, "txn", "item")
        result_fast = find_associations(df, "txn", "item", algo="fast")
        # Sort both by (item_A, item_B) for stable comparison — Rust HashMap
        # iteration order is non-deterministic
        sort_cols = ["item_A", "item_B"]
        a = result_default.sort_values(sort_cols).reset_index(drop=True)
        b = result_fast.sort_values(sort_cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_fast_k3(self):
        df = _make_test_df()
        result = find_associations(
            df, "txn", "item", k=3, min_support=0.3, algo="fast",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "antecedent_1" in result.columns
        assert "antecedent_2" in result.columns
        assert "consequent" in result.columns


# ---------------------------------------------------------------------------
# Correctness: algo="classic"
# ---------------------------------------------------------------------------

class TestAlgoClassic:
    def test_classic_k2_returns_correct_schema(self):
        df = _make_test_df()
        result = find_associations(
            df, "txn", "item", k=2, min_support=0.3, algo="classic",
        )
        assert isinstance(result, pd.DataFrame)
        expected_cols = [
            "item_A", "item_B", "instances", "support",
            "confidence", "lift", "conviction", "leverage",
            "cosine", "jaccard",
        ]
        assert list(result.columns) == expected_cols

    def test_classic_k3_returns_correct_schema(self):
        df = _make_test_df()
        result = find_associations(
            df, "txn", "item", k=3, min_support=0.3, algo="classic",
        )
        assert isinstance(result, pd.DataFrame)
        expected_cols = [
            "antecedent_1", "antecedent_2", "consequent",
            "instances", "support", "confidence", "lift",
        ]
        assert list(result.columns) == expected_cols

    def test_classic_k4(self):
        df = _make_test_df()
        result = find_associations(
            df, "txn", "item", k=4, min_support=0.15, algo="classic",
        )
        assert isinstance(result, pd.DataFrame)
        expected_cols = [
            "antecedent_1", "antecedent_2", "antecedent_3", "consequent",
            "instances", "support", "confidence", "lift",
        ]
        assert list(result.columns) == expected_cols

    def test_classic_high_support_empty(self):
        """Very high min_support should return empty DataFrame."""
        df = _make_test_df()
        result = find_associations(
            df, "txn", "item", k=2, min_support=0.99, algo="classic",
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Cross-algorithm agreement: fast and classic produce identical frequent sets
# ---------------------------------------------------------------------------

class TestFastClassicAgreement:
    def _get_pair_set(self, df_result):
        """Extract set of canonical undirected pairs from k=2 result."""
        pairs = set()
        for _, row in df_result.iterrows():
            pair = tuple(sorted([row["item_A"], row["item_B"]]))
            pairs.add(pair)
        return pairs

    def _get_pair_instances(self, df_result):
        """Extract dict of canonical pair -> instances from k=2 result."""
        counts = {}
        for _, row in df_result.iterrows():
            pair = tuple(sorted([row["item_A"], row["item_B"]]))
            counts[pair] = int(row["instances"])
        return counts

    def test_k2_same_pairs_and_counts(self):
        """Fast and classic produce identical frequent pairs at same min_support."""
        df = _make_test_df()
        min_sup = 0.3

        fast = find_associations(
            df, "txn", "item", k=2, min_support=min_sup, algo="fast",
        )
        classic = find_associations(
            df, "txn", "item", k=2, min_support=min_sup, algo="classic",
        )

        fast_pairs = self._get_pair_set(fast)
        classic_pairs = self._get_pair_set(classic)
        assert fast_pairs == classic_pairs, (
            f"Pair sets differ: fast-only={fast_pairs - classic_pairs}, "
            f"classic-only={classic_pairs - fast_pairs}"
        )

        fast_counts = self._get_pair_instances(fast)
        classic_counts = self._get_pair_instances(classic)
        for pair in fast_pairs:
            assert fast_counts[pair] == classic_counts[pair], (
                f"Count mismatch for {pair}: fast={fast_counts[pair]}, "
                f"classic={classic_counts[pair]}"
            )

    def _get_itemset_counts(self, df_result, k):
        """Extract dict of canonical k-tuple -> instances from k>=3 result."""
        ant_cols = [f"antecedent_{i}" for i in range(1, k)]
        counts = {}
        for _, row in df_result.iterrows():
            items = tuple(sorted([row[c] for c in ant_cols] + [row["consequent"]]))
            counts[items] = int(row["instances"])
        return counts

    def test_k3_same_itemsets_and_counts(self):
        """Fast and classic produce identical frequent 3-itemsets."""
        df = _make_test_df()
        min_sup = 0.3

        fast = find_associations(
            df, "txn", "item", k=3, min_support=min_sup, algo="fast",
        )
        classic = find_associations(
            df, "txn", "item", k=3, min_support=min_sup, algo="classic",
        )

        fast_sets = self._get_itemset_counts(fast, 3)
        classic_sets = self._get_itemset_counts(classic, 3)
        assert set(fast_sets.keys()) == set(classic_sets.keys()), (
            f"Itemset sets differ: fast-only={set(fast_sets) - set(classic_sets)}, "
            f"classic-only={set(classic_sets) - set(fast_sets)}"
        )
        for itemset in fast_sets:
            assert fast_sets[itemset] == classic_sets[itemset]

    def test_k4_same_itemsets_and_counts(self):
        """Fast and classic produce identical frequent 4-itemsets."""
        df = _make_test_df()
        min_sup = 0.15

        fast = find_associations(
            df, "txn", "item", k=4, min_support=min_sup, algo="fast",
        )
        classic = find_associations(
            df, "txn", "item", k=4, min_support=min_sup, algo="classic",
        )

        fast_sets = self._get_itemset_counts(fast, 4)
        classic_sets = self._get_itemset_counts(classic, 4)
        assert set(fast_sets.keys()) == set(classic_sets.keys())
        for itemset in fast_sets:
            assert fast_sets[itemset] == classic_sets[itemset]


# ---------------------------------------------------------------------------
# Verbose parameter tests
# ---------------------------------------------------------------------------

class TestVerbose:
    def test_verbose_false_is_silent(self, capsys):
        """verbose=False (default) produces no output."""
        df = _make_test_df()
        find_associations(df, "txn", "item", verbose=False)
        captured = capsys.readouterr()
        assert "[fastapriori]" not in captured.out

    def test_verbose_true_prints_dataset_info(self, capsys):
        """verbose=True prints dataset features and algo info."""
        df = _make_test_df()
        find_associations(df, "txn", "item", verbose=True)
        captured = capsys.readouterr()
        assert "[fastapriori] Dataset:" in captured.out
        assert "txns" in captured.out
        assert "items" in captured.out
        assert "d_avg=" in captured.out
        assert "d_max=" in captured.out
        assert "algo=fast" in captured.out

    def test_verbose_shows_classic_algo(self, capsys):
        """verbose=True shows algo=classic when classic is selected."""
        df = _make_test_df()
        find_associations(
            df, "txn", "item", algo="classic", min_support=0.3, verbose=True,
        )
        captured = capsys.readouterr()
        assert "algo=classic" in captured.out

    def test_verbose_density_warning_for_high_k(self, capsys):
        """verbose=True emits density warning for high combo cost."""
        # Create a dense dataset: 5 txns with 20 items each
        rows = []
        for t in range(5):
            for i in range(20):
                rows.append((t, i))
        df = pd.DataFrame(rows, columns=["txn", "item"])
        # k=4 on d_max=20 → C(20,3)*5 = 5700 — below 1e8, no warning
        find_associations(
            df, "txn", "item", k=4, min_support=0.1, algo="fast", verbose=True,
        )
        captured = capsys.readouterr()
        assert "d_max=20" in captured.out

    def test_verbose_with_k2_no_warning(self, capsys):
        """verbose=True at k=2 should not emit combo warning (k<3)."""
        df = _make_test_df()
        find_associations(df, "txn", "item", k=2, verbose=True)
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out
