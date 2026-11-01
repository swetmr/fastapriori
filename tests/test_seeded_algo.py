"""algo="seeded" and exact decoding of integer labels."""

import numpy as np
import pandas as pd
import pytest

from fastapriori import find_associations


def _rules(frame):
    cols = [c for c in frame.columns if c.startswith("antecedent") or c == "consequent"]
    return sorted(map(tuple, frame[cols + ["instances"]].astype(str).to_numpy()))


def _df():
    rng = np.random.default_rng(3)
    t = rng.integers(0, 300, 4000)
    i = (rng.zipf(1.6, 4000) % 40).astype(int)
    return pd.DataFrame({"t": t, "i": i}).drop_duplicates()


@pytest.mark.parametrize("k", [3, 4, 5])
def test_seeded_matches_fast(k):
    df = _df()
    kw = dict(transaction_col="t", item_col="i", k=k, min_support=0.01, sorted_by=None)
    assert _rules(find_associations(df, algo="seeded", **kw)) == _rules(find_associations(df, algo="fast", **kw))


def test_seeded_is_fast_at_k2():
    df = _df()
    kw = dict(transaction_col="t", item_col="i", k=2, min_support=0.01)
    pd.testing.assert_frame_equal(find_associations(df, algo="seeded", **kw), find_associations(df, algo="fast", **kw))


def test_seeded_rejects_capping_and_python_backends():
    df = _df()
    kw = dict(transaction_col="t", item_col="i", k=3, min_support=0.01)
    with pytest.raises(ValueError, match="max_items_per_txn"):
        find_associations(df, algo="seeded", max_items_per_txn=10, **kw)
    with pytest.raises(ValueError, match="Rust backend"):
        find_associations(df, algo="seeded", backend="pandas", **kw)
    with pytest.raises(ValueError, match="conflicts"):
        find_associations(df, algo="seeded", backend_options={"fast_variant": "apriori"}, **kw)


@pytest.mark.parametrize("algo", ["fast", "seeded", "classic"])
@pytest.mark.parametrize("k", [2, 3])
def test_huge_integer_labels_stay_distinct(algo, k):
    a, b, c = -1, 2**63, 2**63 + 1
    df = pd.DataFrame({"t": [1, 1, 1, 2, 2, 2], "i": pd.Series([a, b, c, a, b, c], dtype=object)})
    r = find_associations(df, transaction_col="t", item_col="i", k=k, min_support=0.5, algo=algo)
    cols = ["item_A", "item_B"] if k == 2 else ["antecedent_1", "antecedent_2", "consequent"]
    assert set(r[cols].to_numpy().ravel().tolist()) == {a, b, c}


def test_ordinary_labels_keep_their_dtype():
    df = pd.DataFrame({"t": [1, 1, 2, 2], "i": [10, 20, 10, 20]})
    r = find_associations(df, transaction_col="t", item_col="i", min_support=0.5)
    assert pd.api.types.is_integer_dtype(r["item_A"])


@pytest.mark.parametrize("k", [3, 4])
def test_default_is_seeded_above_pairs(k, monkeypatch):
    df = pd.DataFrame({"t": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4], "i": list("abcabdabca")})
    kw = dict(transaction_col="t", item_col="i", k=k, min_support=0.25)
    calls = []
    import fastapriori.core as core
    real = core._seeded_usable
    monkeypatch.setattr(core, "_seeded_usable", lambda *a: calls.append(real(*a)) or calls[-1])
    pd.testing.assert_frame_equal(find_associations(df, **kw), find_associations(df, algo="fast", **kw))
    assert calls == [True]


def test_auto_honours_caller_frequent_lower():
    df = _df()
    kw = dict(transaction_col="t", item_col="i", min_support=0.01)
    pairs = find_associations(df, k=2, min_confidence=0.5, **kw)
    pd.testing.assert_frame_equal(find_associations(df, k=3, frequent_lower=pairs, **kw),
                                  find_associations(df, k=3, frequent_lower=pairs, algo="fast", **kw))


def test_seeded_warns_when_it_ignores_frequent_lower():
    df = _df()
    kw = dict(transaction_col="t", item_col="i", min_support=0.01)
    pairs = find_associations(df, k=2, **kw)
    with pytest.warns(UserWarning, match="ignores frequent_lower"):
        got = find_associations(df, k=3, frequent_lower=pairs, algo="seeded", **kw)
    pd.testing.assert_frame_equal(got, find_associations(df, k=3, algo="seeded", **kw))


@pytest.mark.parametrize("algo", ["auto", "seeded"])
def test_seeded_warns_about_profiling(algo, monkeypatch, tmp_path):
    monkeypatch.setenv("FASTAPRIORI_PROFILE", str(tmp_path / "p.json"))
    with pytest.warns(RuntimeWarning, match="anchor path only"):
        find_associations(_df(), transaction_col="t", item_col="i", k=3, min_support=0.01, algo=algo)


def test_auto_stays_fast_when_seeded_is_unavailable():
    df = pd.DataFrame({"t": [1, 1, 1, 2, 2, 2], "i": list("abcabd")})
    kw = dict(transaction_col="t", item_col="i", k=3, min_support=0.5)
    pd.testing.assert_frame_equal(find_associations(df, max_items_per_txn=2, **kw), find_associations(df, algo="fast", max_items_per_txn=2, **kw))
    pd.testing.assert_frame_equal(find_associations(df, backend="pandas", **kw), find_associations(df, algo="fast", backend="pandas", **kw))
    assert find_associations(df, transaction_col="t", item_col="i", k=3, algo="auto").equals(find_associations(df, transaction_col="t", item_col="i", k=3, algo="fast"))
