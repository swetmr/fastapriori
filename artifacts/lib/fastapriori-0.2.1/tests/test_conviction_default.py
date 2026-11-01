"""Default thresholds must not drop rules whose conviction is undefined.

Conviction is NaN when the consequent occurs in every transaction
(0/0). With the old default ``min_conviction=0.0`` those rules were
silently dropped, because ``NaN >= 0.0`` is False.
"""

import numpy as np
import pandas as pd
import pytest

from fastapriori import find_associations

try:
    import fastapriori._fastapriori_rs  # noqa: F401
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

try:
    import polars  # noqa: F401
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


@pytest.fixture
def df_universal_item():
    """'U' is in every transaction; 'a', 'b', 'c' are not."""
    txns = {
        1: ["U", "a", "b"],
        2: ["U", "a"],
        3: ["U", "b", "c"],
        4: ["U", "a", "c"],
        5: ["U"],
    }
    return pd.DataFrame(
        [(t, i) for t, items in txns.items() for i in items],
        columns=["txn", "item"],
    )


def _configs():
    out = [("fast", "pandas")]
    if HAS_POLARS:
        out.append(("fast", "polars"))
    if HAS_RUST:
        out.append(("fast", "rust"))
        out.append(("classic", "rust"))
    return out


@pytest.mark.parametrize("algo,backend", _configs())
def test_default_keeps_rules_into_universal_item(df_universal_item, algo, backend):
    kwargs = {"min_support": 0.0} if algo == "classic" else {}
    res = find_associations(
        df_universal_item, "txn", "item", k=2,
        algo=algo, backend=backend, **kwargs,
    )
    pairs = set(zip(res["item_A"], res["item_B"]))
    # 3 unordered pairs with U plus a-b, a-c, b-c: 6 pairs, 12 directed rules.
    assert len(res) == 12
    for x in ["a", "b", "c"]:
        assert (x, "U") in pairs and ("U", x) in pairs
    into_u = res[res["item_B"] == "U"]
    assert into_u["conviction"].isna().all()
    assert (into_u["confidence"] == 1.0).all()


@pytest.mark.parametrize("algo,backend", _configs())
def test_explicit_conviction_threshold_drops_undefined(df_universal_item, algo, backend):
    """An explicit threshold keeps the documented semantics: NaN fails it."""
    kwargs = {"min_support": 0.0} if algo == "classic" else {}
    res = find_associations(
        df_universal_item, "txn", "item", k=2,
        algo=algo, backend=backend, min_conviction=0.0, **kwargs,
    )
    assert not (res["item_B"] == "U").any()
    assert len(res) == 9
    assert np.isfinite(res["conviction"]).sum() + np.isinf(res["conviction"]).sum() == 9


def _k3_backends():
    out = ["counter_chain", "pandas"]
    if HAS_POLARS:
        out.append("polars")
    if HAS_RUST:
        out.append("rust")
    return out


@pytest.mark.parametrize("backend", _k3_backends())
def test_k3_auto_lower_level_keeps_pair_of_universal_items(backend):
    """With two universal items U, V both directions of {U, V} have NaN
    conviction; the auto-computed k=2 level must not drop that pair."""
    txns = {
        1: ["U", "V", "a", "b"],
        2: ["U", "V", "a"],
        3: ["U", "V", "b", "c"],
        4: ["U", "V", "a", "c"],
        5: ["U", "V"],
    }
    df = pd.DataFrame(
        [(t, i) for t, items in txns.items() for i in items],
        columns=["txn", "item"],
    )
    res = find_associations(df, "txn", "item", k=3, min_support=0.0, backend=backend)
    itemsets = {
        frozenset(r) for r in
        res[["antecedent_1", "antecedent_2", "consequent"]].itertuples(index=False)
    }
    for x in ["a", "b", "c"]:
        assert frozenset({"U", "V", x}) in itemsets
    # {U,V,x}, {U,x,y} and {V,x,y} for x,y in a,b,c ({a,b,c} never occurs):
    # 9 itemsets x 3 directed rules.
    assert len(res) == 27
    assert (res["confidence"] <= 1.0 + 1e-9).all()
