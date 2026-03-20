# fastapriori

Fast pairwise (k=2) co-occurrence and association analysis for transactional data.

Uses a polars backend to compute all item-pair metrics **5-10x faster** than traditional Apriori implementations on large datasets (>100k rows) and low support condition (<.01).

## Installation

```bash
pip install fastapriori
```

For best performance, install with polars:

```bash
pip install fastapriori polars
```

If polars is not installed, fastapriori falls back to a pandas Counter+chain backend automatically.

## Quick Start

```python
import pandas as pd
from fastapriori import find_associations

# Example: transactional data with one row per (transaction, item) pair
df = pd.DataFrame({
    "txn_id": [1, 1, 1, 2, 2, 3, 3, 3],
    "item":   ["A", "B", "C", "A", "B", "B", "C", "D"],
})

result = find_associations(
    df,
    transaction_col="txn_id",
    item_col="item",
    min_support=0.01,
    min_confidence=0.1,
)
```

Returns a DataFrame with columns:

| Column | Description |
|--------|-------------|
| `item_A` | Reference item |
| `item_B` | Co-occurring item |
| `instances` | Transactions containing both A and B |
| `support` | instances / total_transactions |
| `confidence` | P(B\|A) = instances / transactions_containing_A |
| `lift` | confidence / support(B) |
| `conviction` | (1 - support(B)) / (1 - confidence) |
| `leverage` | support(A,B) - support(A) * support(B) |
| `cosine` | support(A,B) / sqrt(support(A) * support(B)) |
| `jaccard` | support(A,B) / (support(A) + support(B) - support(A,B)) |

## Metric Interpretation Guide

| Metric | Range | What it tells you | When to use |
|--------|-------|-------------------|-------------|
| **support** | 0 – 1 | How frequently the pair appears across all transactions. Higher = more common pair. | Filter out noise — set a floor to focus on pairs that occur often enough to matter. |
| **confidence** | 0 – 1 | P(B\|A) — given item A was purchased, how likely is B? Directional: confidence(A→B) ≠ confidence(B→A). | Product recommendations — "customers who bought A also bought B" requires high confidence(A→B). |
| **lift** | 0 – ∞ | How much more likely A and B co-occur than if they were independent. **lift = 1** means no association; **> 1** means positive; **< 1** means negative (substitutes). | Best general-purpose metric for finding interesting associations. Filter with `lift > 1` to find items that genuinely attract each other. |
| **conviction** | 0.5 – ∞ | How much the rule A→B would be wrong if A and B were independent. Higher = stronger directional dependency. **conviction = 1** means independence; **∞** means the rule is always correct. | Preferred over confidence when you need to distinguish strong directional rules — less sensitive to support(B) than lift. |
| **leverage** | -0.25 – 0.25 | Difference between observed support and expected support under independence. **0** means independence; positive means co-occurrence; negative means avoidance. | Useful when you want an absolute (not relative) measure of association strength. Comparable across pairs with similar support levels. |
| **cosine** | 0 – 1 | Symmetric similarity between A and B, normalized by their geometric mean frequency. Not inflated by rare items like lift can be. | When you need a symmetric, bounded alternative to lift — good for clustering or similarity matrices. |
| **jaccard** | 0 – 1 | Overlap coefficient — what fraction of transactions containing A or B contain both? Stricter than cosine. | When you want to measure how "interchangeable" two items are — high jaccard means they almost always appear together. |

**Quick decision guide:**
- **Bundling / cross-sell** → filter by `lift > 1` and `confidence > 0.1`
- **Substitute detection** → look for `lift < 1` (items that suppress each other)
- **Symmetric similarity** → use `cosine` or `jaccard`
- **Directional rules** (A implies B) → use `confidence` or `conviction`

## API Reference

### `find_associations()`

```python
find_associations(
    df,
    transaction_col,
    item_col,
    min_support=None,       # minimum pair support (float or None)
    min_confidence=0.0,     # minimum P(B|A)
    min_lift=0.0,           # minimum lift
    min_conviction=0.0,     # minimum conviction
    min_leverage=None,      # minimum leverage (None = no filter, can be negative)
    min_cosine=0.0,         # minimum cosine similarity
    min_jaccard=0.0,        # minimum Jaccard similarity
    show_progress=False,    # tqdm progress bar (pandas backend only)
    backend="auto",         # "auto", "polars", or "pandas"
)
```

**Backend choice:**
- `"auto"` (default) — uses polars if installed, otherwise pandas
- `"polars"` — self-join approach, fastest for sparse data (<10 items per transaction)
- `"pandas"` — Counter+chain approach, lower memory usage

### `get_top_associations()`

```python
from fastapriori import get_top_associations

# Top 10 items associated with "widget-A" by lift
top = get_top_associations(result, item="widget-A", metric="lift", n=10, role="any")
```

`role` controls direction: `"antecedent"` (item as item_A), `"consequent"` (item as item_B), or `"any"` (both).

### `filter_associations()`

```python
from fastapriori import filter_associations

# All associations involving one or more items
filtered = filter_associations(result, items=["widget-A", "widget-B"], role="any")
```

### `to_heatmap()`

```python
from fastapriori import to_heatmap

pivot = to_heatmap(result, metric="lift")  # returns a pivot table (item_A x item_B)
```

### `plot_heatmap()`

```python
from fastapriori import plot_heatmap

fig = plot_heatmap(result, metric="lift", cmap="RdYlGn", annot=True)
```

Requires `matplotlib`.

### `to_graph()`

```python
from fastapriori import to_graph

G = to_graph(result, metric="lift", min_value=1.5)  # returns networkx.DiGraph
```

Requires `networkx` (`pip install fastapriori[graph]`).

## Performance

Benchmarked on the online-retail dataset, pairwise (k=2) associations — sweeping `min_support` and `min_confidence` side by side:

![Benchmark: min_support and min_confidence sweep](benchmarks/retail_support_confidence_sweep.png)

- **Left** — varying `min_support` (log scale 0.0001–0.01, confidence=0.0): execution time
- **Right** — varying `min_confidence` (0.05–0.50, support=0.001): execution time

The polars backend is consistently **5-10x faster** than efficient-apriori for k=2, with both backends producing matching rule counts across all threshold values.

## When to Use Something Else

fastapriori is currently optimized for **pairwise (k=2) associations only**. If you need higher-order itemsets (k=3, k=4, ...), use efficient-apriori.

## License

MIT
