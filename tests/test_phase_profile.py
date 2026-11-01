"""Phase profiling (0.2.1). With the Cargo feature `phaseprof`, FASTAPRIORI_PROFILE=<file> makes the
anchor k>=3 Rust path (algo="fast") write one JSON object with its phase times and work counters. Default builds
(no feature) warn instead. Without the variable nothing is written and results are unchanged."""
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from fastapriori import find_associations


def _random_baskets(seed=0, n_txn=2000, n_items=40, per_txn=6):
    rng = np.random.default_rng(seed)
    rows = [(t, int(i)) for t in range(n_txn)
            for i in rng.choice(n_items, size=per_txn, replace=False)]
    rows += [(t, i) for t in range(0, n_txn, 40) for i in (100, 101, 102, 103, 104)]   # planted 5-set
    return pd.DataFrame(rows, columns=["txn", "item"])


def _mine(df, k):
    out = find_associations(df, "txn", "item", k=k, min_support=0.002, min_confidence=0.0,
                            backend="rust", algo="fast", sorted_by=None)
    cols = [f"antecedent_{i}" for i in range(1, k)] + ["consequent"]
    return out.sort_values(cols).reset_index(drop=True)


@pytest.fixture(scope="module")
def profiling_build(tmp_path_factory):
    """True if the installed extension was built with the `phaseprof` feature."""
    path = tmp_path_factory.mktemp("probe") / "profile.json"
    os.environ["FASTAPRIORI_PROFILE"] = str(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _mine(_random_baskets(n_txn=200), 3)
    finally:
        del os.environ["FASTAPRIORI_PROFILE"]
    return path.exists()


@pytest.mark.parametrize("k", [3, 4])
def test_profile_written_and_results_unchanged(tmp_path, monkeypatch, k, profiling_build):
    if not profiling_build:
        pytest.skip("extension built without the `phaseprof` feature")
    df = _random_baskets()
    monkeypatch.delenv("FASTAPRIORI_PROFILE", raising=False)
    plain = _mine(df, k)

    path = tmp_path / "profile.json"
    monkeypatch.setenv("FASTAPRIORI_PROFILE", str(path))
    profiled = _mine(df, k)
    pd.testing.assert_frame_equal(plain, profiled)

    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["k"] == k and rec["n_transactions"] == 2000 and rec["n_rules"] == len(profiled)
    for key in ("encode_s", "rust_call_s", "decode_s", "total_s", "index_s", "pairs_s",
                "pair_tids_s", "core_s", "output_rust_s"):
        assert rec[key] >= 0.0, key
    assert rec["total_s"] >= rec["rust_call_s"] >= rec["core_s"]
    assert rec["core_s"] >= rec["index_s"] + rec["pairs_s"] + rec["pair_tids_s"]
    assert [lv["k"] for lv in rec["levels"]] == list(range(3, k + 1))
    # the last level's frequent itemsets are the mined k-itemsets; each gives k rules
    assert rec["levels"][-1]["n_frequent"] * k == len(profiled)
    assert rec["threads"] >= 1


def test_default_build_warns(tmp_path, monkeypatch, profiling_build):
    if profiling_build:
        pytest.skip("extension built with the `phaseprof` feature")
    monkeypatch.setenv("FASTAPRIORI_PROFILE", str(tmp_path / "profile.json"))
    with pytest.warns(RuntimeWarning, match="phaseprof"):
        _mine(_random_baskets(n_txn=200), 3)
    assert not (tmp_path / "profile.json").exists()


def test_seeded_default_says_profiling_needs_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTAPRIORI_PROFILE", str(tmp_path / "profile.json"))
    with pytest.warns(RuntimeWarning, match="algo='fast'"):
        find_associations(_random_baskets(n_txn=200), "txn", "item", k=3, min_support=0.002)
    assert not (tmp_path / "profile.json").exists()


def test_no_profile_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("FASTAPRIORI_PROFILE", raising=False)
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _mine(_random_baskets(), 3)
    assert list(tmp_path.iterdir()) == []
