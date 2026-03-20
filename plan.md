# Plan: Extend fastapriori — Polars Backend, Triplets, and Utilities

## Context
The user wants to explore three extensions to the fastapriori package:
1. A Polars-based implementation for potential speedups
2. Triplet (3-itemset) support beyond current pairwise
3. Additional utility functions and metrics

---

## 1. Polars Implementation

### Approach: Self-Join (Pure Polars, no Counter+chain)

The Counter+chain step can't be expressed natively in Polars (no Python objects in columns). Instead, use a **self-join** that is fully vectorized:

```python
# Join df to itself on transaction_col, filter item_A != item_B, groupby+count
pairs = (
    df.lazy()
    .join(df.lazy(), on=transaction_col, suffix="_B")
    .filter(pl.col(item_col) != pl.col(f"{item_col}_B"))
    .group_by([item_col, f"{item_col}_B"])
    .agg(pl.len().alias("instances"))
    .collect()
)
```

This replaces the entire groupby -> Counter+chain -> explode pipeline with one vectorized operation.

### Tradeoffs
| | Pandas (current) | Polars self-join |
|---|---|---|
| **Speed** | Baseline | 2-4x faster (sparse data), similar for dense |
| **Memory** | Moderate (stores sets/counters) | Higher (self-join expansion: items_per_txn^2 rows per txn) |
| **Best for** | Dense transactions (>20 items/txn) | Sparse transactions (<10 items/txn) |
| **Parallelism** | Single-threaded | Multi-threaded by default |

### Implementation Plan
- Add `fastapriori/backends/polars_backend.py` with same API signature
- Add `fastapriori/backends/pandas_backend.py` (extract current logic)
- `find_associations()` gets a `backend="pandas"` parameter (default pandas for backward compat)
- Auto-detect: if polars installed and `backend="auto"`, pick based on data shape
- Polars as optional dependency: `polars>=0.20` in `[project.optional-dependencies.polars]`

### Files to Create/Modify
- `fastapriori/backends/__init__.py` — NEW
- `fastapriori/backends/pandas_backend.py` — NEW (extract from current core.py)
- `fastapriori/backends/polars_backend.py` — NEW
- `fastapriori/core.py` — MODIFY (dispatch to backend)
- `pyproject.toml` — MODIFY (add polars optional dep)

---

## 2. Triplet Extension

### Algorithm
For each transaction, generate all 3-item combinations, then count globally:

```python
from itertools import combinations

# For each transaction's item set, generate all triplets
triplet_counts = Counter()
for txn_items in trans_dict.values():
    for triplet in combinations(sorted(txn_items), 3):
        triplet_counts[triplet] += 1
```

### Pre-filtering (critical for performance)
With 500 items, C(500,3) = ~20M possible triplets. Must pre-filter:
1. **Item-level**: Only consider items that individually meet min_support
2. **Pair-level**: Only consider triplets where all 3 constituent pairs are frequent (Apriori property)

```python
# After computing frequent pairs:
frequent_pairs = set of (A,B) passing thresholds
# Only keep triplet (A,B,C) if (A,B), (A,C), (B,C) all frequent
```

This typically reduces the search space from millions to thousands.

### Metrics for triplets
- `support(A,B,C)` = instances / total_transactions
- `confidence(A,B -> C)` = support(A,B,C) / support(A,B)
- `lift(A,B -> C)` = confidence / support(C)

### API
```python
def find_triplets(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    min_support: float | None = None,
    min_confidence: float | None = 0.1,
    frequent_pairs: pd.DataFrame | None = None,  # from find_associations()
    show_progress: bool = False,
) -> pd.DataFrame:
```

Returns DataFrame with columns: `item_A, item_B, item_C, instances, support, confidence, lift`

### Files
- `fastapriori/triplets.py` — NEW
- `fastapriori/__init__.py` — MODIFY (export `find_triplets`)

---

## 3. Additional Metrics

Add to existing `find_associations()` output. These are all O(1) per row, computed in Step 7:

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **conviction** | `(1 - support_B) / (1 - confidence)` | Departure from independence (directional) |
| **leverage** | `support_AB - support_A * support_B` | Difference from expected co-occurrence |
| **cosine** | `support_AB / sqrt(support_A * support_B)` | Geometric similarity |
| **jaccard** | `support_AB / (support_A + support_B - support_AB)` | Set overlap |

### Implementation
Add after existing lift computation in `core.py`:

```python
support_a = result["_trans_count"] / total_transactions
support_b = result["_b_trans_count"] / total_transactions

result["conviction"] = np.round((1 - support_b) / (1 - result["confidence"] + 1e-10), 4)
result["leverage"] = np.round(result["support"] - (support_a * support_b), 6)
result["cosine"] = np.round(result["support"] / np.sqrt(support_a * support_b), 4)
result["jaccard"] = np.round(result["support"] / (support_a + support_b - result["support"]), 4)
```

### Files
- `fastapriori/core.py` — MODIFY (add 4 metric columns)

---

## 4. Utility Functions

### A. `get_top_associations(result_df, item, metric="lift", n=10)`
Returns top N associated items for a given item (both as antecedent and consequent).

### B. `filter_associations(result_df, items, role="any")`
Filter results by item(s). `role`: "antecedent", "consequent", or "any".

### C. `to_graph(result_df, metric="lift", min_value=1.0)`
Export as NetworkX directed graph (edges weighted by metric). NetworkX as optional dep.

### D. `to_heatmap(result_df, metric="lift")`
Return pivot table (item_A x item_B) for heatmap visualization.

### Files
- `fastapriori/utils.py` — NEW (all utility functions)
- `fastapriori/__init__.py` — MODIFY (export utilities)
- `pyproject.toml` — MODIFY (add `networkx` to optional deps under `[graph]`)

---

## Implementation Order

1. **Additional metrics** in `core.py` (conviction, leverage, cosine, jaccard) — small change, high value
2. **Utility functions** in `utils.py` — standalone, no changes to core algorithm
3. **Triplets** in `triplets.py` — new module, independent of pairs
4. **Polars backend** — largest change, requires refactoring core.py into backends

---

## Updated File Tree

```
fastapriori/
├── __init__.py          # MODIFY - export new functions
├── core.py              # MODIFY - add metrics, backend dispatch
├── triplets.py          # NEW - find_triplets()
├── utils.py             # NEW - top_associations, filter, graph, heatmap
└── backends/
    ├── __init__.py      # NEW
    ├── pandas_backend.py  # NEW - extracted from core.py
    └── polars_backend.py  # NEW - self-join approach
```

## Verification

1. Run existing test notebook — all sanity checks still pass with new metric columns
2. Test triplets on skewed dataset — verify support/confidence values are valid
3. Benchmark Polars vs Pandas backends on faker.csv
4. Test utility functions: top-N, filter, graph export, heatmap pivot
