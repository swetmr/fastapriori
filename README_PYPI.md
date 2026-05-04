# fastapriori

[![PyPI](https://img.shields.io/pypi/v/fastapriori.svg)](https://pypi.org/project/fastapriori/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20021781.svg)](https://doi.org/10.5281/zenodo.20021781)

Fast frequent itemset mining (with full association-rule metrics at k=2) — even at very low support thresholds.

A compiled Rust engine with an inverted-index architecture. Counts pair co-occurrences exhaustively at k=2 (constant runtime in `min_support`) and uses anchor-and-extend with Apriori pruning at k>=3. Across eight real-world datasets (9.8K to 3.2M transactions, up to 49K items), `algo="fast"` wins **100% of k=2 configurations vs `efficient-apriori`** (median 61x, up to 969x) and **100% vs the like-for-like compiled Apriori baseline** (median 3.9x, up to 92.8x).

→ See the [GitHub repository](https://github.com/swetmr/fastapriori) for benchmarks, the decision rule, performance plots, and the algorithm description.

## Installation

```bash
pip install fastapriori
```

Pre-built wheels are shipped for Linux x86_64/aarch64, macOS arm64, and Windows x64 across Python 3.9–3.13. Other platforms build from source and need the [Rust toolchain](https://rustup.rs).

## Quick Start

```python
import pandas as pd
from fastapriori import find_associations

# Transactional data: one row per (transaction, item) pair
df = pd.DataFrame({
    "txn_id": [1, 1, 1, 2, 2, 3, 3, 3],
    "item":   ["A", "B", "C", "A", "B", "B", "C", "D"],
})

# k=2 (default) — pairwise associations with seven metrics
pairs = find_associations(
    df,
    transaction_col="txn_id",
    item_col="item",
    min_support=0.01,
    min_confidence=0.1,
)

# k=3 — triplet associations
triplets = find_associations(
    df,
    transaction_col="txn_id",
    item_col="item",
    k=3,
    min_support=0.01,
)
```

### k=2 output columns
`item_A`, `item_B`, `instances`, `support`, `confidence`, `lift`, `conviction`, `leverage`, `cosine`, `jaccard`.

### k>=3 output columns
`antecedent_1` … `antecedent_{k-1}`, `consequent`, `instances`, `support`, `confidence`, `lift`.

## API Reference

### `find_associations()`

```python
find_associations(
    df,
    transaction_col,
    item_col,
    k=2,                     # itemset size (2 to 50)
    min_support=None,        # minimum support (float or None)
    min_confidence=0.0,      # minimum P(B|A)
    min_lift=0.0,            # minimum lift (k=2 only)
    min_conviction=0.0,      # minimum conviction (k=2 only)
    min_leverage=None,       # minimum leverage (k=2 only)
    min_cosine=0.0,          # minimum cosine similarity (k=2 only)
    min_jaccard=0.0,         # minimum Jaccard similarity (k=2 only)
    max_items_per_txn=None,  # cap outlier transactions (k>=3)
    item_weights=None,       # dict for custom ranking used by max_items_per_txn
    low_memory="auto",       # pre-filter infrequent items to reduce memory
    show_progress=False,     # tqdm progress bar
    backend="auto",          # "auto", "rust", "python", "polars", "pandas"
    algo="fast",             # "fast" (default), "classic", or "auto"
    sorted_by="support",     # sort column (or None to skip)
    verbose=False,           # print dataset stats and density warnings
)
```

**`algo`**:
- `"fast"` (default) — inverted-index count-all. Constant runtime at k=2; wins 91–97% of real-world configurations vs the compiled Apriori control across k=3..9.
- `"classic"` — Rust port of Apriori (join + prune + short-circuit). Requires `min_support`. Useful for dense, correlated data at k>=4.
- `"auto"` — routes to `"fast"`.

**`backend`**:
- `"auto"` (default) — Rust if the compiled extension is available, otherwise Python.
- `"python"` — polars for k=2 (falling back to pandas), counter_chain for k>=3.

### Helper functions

```python
from fastapriori import (
    get_top_associations,   # top-N items associated with a given item
    filter_associations,    # filter results to associations involving specific items
    to_heatmap,             # pivot results into an item x item matrix
    plot_heatmap,           # matplotlib heatmap (requires matplotlib)
    to_graph,               # networkx.DiGraph (requires networkx)
)
```

## Practical Workflow: Run Once, Filter Many Times

Because `fast` k=2 runtime is constant in `min_support`, you can compute the full co-occurrence table once with `min_support=None` and then filter interactively — no re-computation:

```python
full = find_associations(df, "txn_id", "item")
strong      = full[(full["lift"] > 2) & (full["confidence"] > 0.3)]
substitutes = full[full["lift"] < 1]
rare        = full[full["instances"] == 1]   # anomaly / error detection
```

## Memory and Limitations

- The pair counter scales as O(m²) in the unique-item count. Instacart (~50k items) peaks at ~1.5 GB; Chainstore (~46k items) at ~0.94 GB. Use `low_memory=True` (with a `min_support`) to pre-filter infrequent items for an additional 5–10x reduction on large catalogs.
- At k>=3 with dense baskets, `max_items_per_txn` caps the C(d_max, k-1) blow-up at the cost of lower-bound counts.
- Single-machine only — no distributed (Spark / Dask) version.

## License

MIT
