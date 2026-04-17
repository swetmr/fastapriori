# fastapriori

Fast association rule mining — even at very low support thresholds.

Built on a compiled **Rust engine** with an inverted-index architecture. fastapriori counts pair co-occurrences exhaustively at k=2 and uses an anchor-and-extend strategy with Apriori pruning at k>=3. Across eight real-world datasets and 10 synthetic ones, it wins **100% of k=2 configurations** and **89% at k=3**, with real-world speedups up to **12x at k=3** and **13.2x at k=5** over a like-for-like compiled Apriori baseline (and 352x over the standard Python `efficient-apriori`).

## Installation

```bash
pip install fastapriori
```

The Rust extension is included in the wheel. To build from source, you need the Rust toolchain ([rustup.rs](https://rustup.rs)):

```bash
pip install -e .
```

## Quick Start

```python
import pandas as pd
from fastapriori import find_associations

# Transactional data: one row per (transaction, item) pair
df = pd.DataFrame({
    "txn_id": [1, 1, 1, 2, 2, 3, 3, 3],
    "item":   ["A", "B", "C", "A", "B", "B", "C", "D"],
})

# k=2 (default): pairwise associations with 7 metrics
pairs = find_associations(
    df,
    transaction_col="txn_id",
    item_col="item",
    min_support=0.01,
    min_confidence=0.1,
)

# k=3: triplet associations
triplets = find_associations(
    df,
    transaction_col="txn_id",
    item_col="item",
    k=3,
    min_support=0.01,
)
```

### k=2 Output Columns

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

### k>=3 Output Columns

| Column | Description |
|--------|-------------|
| `antecedent_1` .. `antecedent_{k-1}` | Items in the antecedent |
| `consequent` | Predicted item |
| `instances` | Transactions containing all k items |
| `support` | instances / total_transactions |
| `confidence` | P(consequent \| antecedents) |
| `lift` | confidence / support(consequent) |

## Why Low Support Matters

Classical Apriori relies on a high `min_support` to prune candidates. But the most valuable discoveries — long-tail specialty products, seasonal items, rare industrial part pairings, B2B bundles with 50,000-part catalogs — live exactly where support is low. On a 3.2M-transaction Instacart dataset at `s = 0.0001`, Apriori spends minutes pruning candidates that fastapriori never has to generate.

**fastapriori sidesteps this entirely at k=2**: compute every pair in a single pass, then filter on any metric afterward. No support threshold required.

## Algorithm

fastapriori uses a **"count everything, filter later"** architecture:

- **k=2**: Builds an inverted index (item -> transaction set), then for each item counts ALL co-occurring items in a single pass using a flat array buffer. Runtime is **constant with respect to min_support** — the threshold is applied as a post-hoc filter on precomputed counts.

- **k>=3**: Extends the same inverted index with **anchor-and-extend**: each frequent (k-1)-set serves as an anchor, and candidate k-th items are counted by scanning only the anchor's transactions. Apriori's downward-closure property prunes items that cannot participate in frequent k-sets.

This "best of both worlds" design applies brute-force counting where pruning cannot help (k=2) and principled pruning where it genuinely reduces work (k>=3).

### Three Algorithms

| Algorithm | Description | When to use |
|-----------|-------------|-------------|
| `algo="fast"` (default) | Inverted-index count-all | Best in the overwhelming majority of cases, especially at low support |
| `algo="classic"` | Rust port of Apriori (join + prune + short-circuit) | Dense, correlated data where k>=4 *and* the transaction-size tail is heavy |
| `algo="auto"` | Routes to `"fast"` | Safe default |

## Performance

Benchmarked on eight real-world datasets at k=2, sweeping `min_support` from 0.0001 to 0.01:

![Benchmark: fastapriori vs efficient-apriori (k=2)](benchmarks/k2_all_real_datasets_grid.png)

**fastapriori's runtime is flat at k=2** across all support levels (it computes everything regardless), while `efficient-apriori` slows by 1–2 orders of magnitude as support drops. The `classic` line (compiled Rust Apriori) tracks `fast` at high support but diverges sharply at low support — the same crossover pattern Apriori has always shown, just without the Python tax.

**The risk profile is asymmetric.** When `fast` is suboptimal (very high support on dense data), the penalty is milliseconds. When classic Apriori is suboptimal (low support), the penalty is 5–53 minutes. Choosing `fast` by default costs you nothing in the worst case and saves everything in the common case.

## Choosing an Algorithm

The default (`algo="fast"`) is correct for almost every workflow. If you are working on dense, highly correlated data at high k and want to double-check, use this lightweight heuristic (computed from two one-pass dataset statistics — no benchmarking required):

```python
d_per_txn = df.groupby(transaction_col)[item_col].nunique()
d_avg, d_max, d_std = d_per_txn.mean(), d_per_txn.max(), d_per_txn.std()

tau  = d_max / d_avg        # tail ratio — how extreme your biggest basket is
d_cv = d_std  / d_avg       # coefficient of variation of transaction size

if tau < 25 and d_cv < 1.25:
    algo = "fast"           # safe default across every dataset tested
else:
    # Optional: run both at k=2 and k=3 on target min_support, measure
    # A_obs = speedup_at_k3 / speedup_at_k2.  If A_obs >= 1.3, commit to
    # fast for every higher k.  Otherwise benchmark both at your target k.
    ...
```

Three empirical findings back this up:
1. **k=2**: `fast` wins **100%** of datasets (real + synthetic).
2. **k=3**: `fast` wins **89%** of datasets with speedups up to 12x.
3. **k=4–9**: `fast` wins 56–72% of datasets with speedups up to 13.2x.

If the heuristic above flags your data as risky *and* you have k>=4 *and* you have tight runtime SLAs, benchmark both at your target `k` and pick the winner. In every other case, use `fast`.

## Handling Dense / Outlier Transactions: `max_items_per_txn`

When a few transactions are huge — think Online Retail with `d_max = 539`, or Kosarak with `d_max = 2,498` — the per-transaction enumeration cost at k>=3 is dominated by those outliers (you enumerate C(d, k-1) items per transaction). fastapriori supports an opt-in cap that truncates each transaction to its top-N items by weight:

```python
result = find_associations(
    df, "txn_id", "item",
    k=4,
    min_support=0.001,
    max_items_per_txn=50,      # cap outliers: 539 -> 50 items for the largest basket
    item_weights=None,          # default: rank by global item frequency (keeps frequent items)
)
```

- **What it does.** Transactions with more than N items are reduced to their top-N by weight. Others are left untouched.
- **Counts are lower bounds.** A capped itemset's count is never higher than the true count; some genuinely frequent itemsets may be missed if their count drops below `min_support` after capping. Use it when you *prefer a fast, conservative answer* to a slow, exact one.
- **Custom weights.** Pass an `item_weights={item: score}` dict to prioritize high-revenue / high-margin / business-critical items regardless of frequency.
- **Only applies at k>=3.** Pairs are always counted exactly. Supported by both `algo="fast"` and `algo="classic"`.
- **When it pays off.** Reported impact from the paper: on Online Retail at k=4 with cap=50, per-anchor enumeration cost drops by ~1,300x; on Kosarak, ~133,000x.

## Low-Memory Mode

The pair counter scales as O(m^2) in the unique-item count. Instacart (~50k items) uses ~30 GB at `s = 0.0001`. `low_memory=True` pre-filters items below `min_support` before building the index — typically a 5–10x memory reduction on large catalogs:

```python
find_associations(df, "txn_id", "item", min_support=0.001, low_memory=True)
```

Requires `min_support`. Results are exact because the dropped items could not have met the threshold in the first place.

## Verbose Mode

Use `verbose=True` to inspect dataset characteristics before a long run:

```python
find_associations(df, "txn_id", "item", k=4, min_support=0.001, verbose=True)
```

```
[fastapriori] Dataset: 28,816 txns x 4,632 items | 525,476 rows
[fastapriori] d_avg=18.2  d_max=539  d_median=11.0  d_std=16.3
[fastapriori] k=4  min_support=0.001  algo=fast
[fastapriori] WARNING: C(539, 3) x 28,816 = 752M combinations -- may be slow
```

A warning fires when `C(d_max, k-1) * n_transactions` exceeds 10^8 — a good cue to consider `max_items_per_txn`.

## Metric Interpretation Guide

| Metric | Range | What it tells you | When to use |
|--------|-------|-------------------|-------------|
| **support** | 0 — 1 | How frequently the pair appears across all transactions. | Filter out noise — set a floor to focus on pairs that occur often enough to matter. |
| **confidence** | 0 — 1 | P(B\|A) — given item A was purchased, how likely is B? Directional. | Product recommendations — "customers who bought A also bought B". |
| **lift** | 0 — inf | How much more likely A and B co-occur than if independent. >1 = positive association, <1 = substitutes. | Best general-purpose metric. Filter with `lift > 1`. |
| **conviction** | 0.5 — inf | How much the rule A->B would be wrong if A and B were independent. inf = always correct. | Preferred over confidence for strong directional rules. |
| **leverage** | -0.25 — 0.25 | Absolute deviation from independence. 0 = independent. | When you want absolute (not relative) association strength. |
| **cosine** | 0 — 1 | Symmetric similarity, not inflated by rare items like lift can be. | Clustering, similarity matrices. |
| **jaccard** | 0 — 1 | Overlap coefficient — fraction of transactions with A or B that contain both. | How "interchangeable" two items are. |

**Quick decision guide:**
- **Bundling / cross-sell** -> filter by `lift > 1` and `confidence > 0.1`
- **Substitute detection** -> look for `lift < 1`
- **Symmetric similarity** -> use `cosine` or `jaccard`
- **Directional rules** (A implies B) -> use `confidence` or `conviction`

## Practical Workflow: Run Once, Filter Many Times

Because `fast` k=2 runtime is constant in `min_support`, you can compute the full co-occurrence table once with `min_support=None` and then filter interactively. Each threshold change is a pandas filter, not a recompute:

```python
full = find_associations(df, "txn_id", "item")           # computes everything
strong = full[(full["lift"] > 2) & (full["confidence"] > 0.3)]
rare   = full[full["instances"] == 1]                    # anomaly / error detection
substitutes = full[full["lift"] < 1]
```

`min_support=None` also surfaces pairs that *never* co-occur (zero support) and cold-start items with few transactions — both often filtered away by traditional Apriori before you ever see them.

## API Reference

### `find_associations()`

```python
find_associations(
    df,
    transaction_col,
    item_col,
    k=2,                     # itemset size (2-50)
    min_support=None,        # minimum pair support (float or None)
    min_confidence=0.0,      # minimum P(B|A)
    min_lift=0.0,            # minimum lift (k=2 only)
    min_conviction=0.0,      # minimum conviction (k=2 only)
    min_leverage=None,       # minimum leverage (k=2 only)
    min_cosine=0.0,          # minimum cosine similarity (k=2 only)
    min_jaccard=0.0,         # minimum Jaccard similarity (k=2 only)
    max_items_per_txn=None,  # cap outlier transactions (k>=3, fast + classic)
    item_weights=None,       # dict for custom ranking used by max_items_per_txn
    low_memory="auto",       # pre-filter infrequent items to reduce memory
    show_progress=False,     # tqdm progress bar
    backend="auto",          # "auto", "rust", "python", "polars", "pandas"
    algo="fast",             # "fast", "classic", or "auto"
    sorted_by="support",     # sort column (or None to skip)
    verbose=False,           # print dataset stats and density warnings
)
```

**Algorithm choice:**
- `"fast"` (default) — inverted-index count-all. Constant runtime at k=2, wins 89% at k=3.
- `"classic"` — Rust port of Apriori with join+prune+short-circuit. Requires `min_support`. Potentially useful for dense, correlated data at k>=4 where the transaction-size tail violates the `tau < 25 and d_cv < 1.25` heuristic above.
- `"auto"` — routes to `"fast"` (the safe default in all but edge cases).

**Backend choice:**
- `"auto"` (default) — uses Rust if the compiled extension is available, otherwise falls back to Python.
- `"python"` — polars for k=2 (falling back to pandas), counter_chain for k>=3.
- Individual backends: `"rust"`, `"pandas"`, `"polars"`, `"counter_chain"`.

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

## Limitations

- **Memory at low s, large m**: the pair counter scales O(m^2); on Instacart (~50k items) expect ~30 GB at `s = 10^-4`. Use `low_memory=True` when you have a `min_support`.
- **Dense data at high k**: the combinatorial C(d_max, k-1) term is fundamental. `max_items_per_txn` bounds it at the cost of lower-bound counts.
- **Single-machine**: no distributed (Spark / Dask) version is provided.

## License

MIT
