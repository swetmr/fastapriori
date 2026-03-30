"""Utility functions for working with fastapriori association results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def describe_dataset(
    df: pd.DataFrame,
    transaction_col: str,
    item_col: str,
    k: int = 2,
    min_support: float | None = None,
) -> pd.DataFrame:
    """Analyze dataset properties and recommend fastapriori vs efficient-apriori.

    Computes key dataset statistics -- transaction count, unique items,
    items-per-transaction distribution, item frequency distribution (skewness),
    and density -- then prints a structured profile with an algorithm
    recommendation for the given k level.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with transaction and item columns.
    transaction_col : str
        Name of the transaction ID column.
    item_col : str
        Name of the item column.
    k : int
        Association level (2 = pairs, 3 = triplets). Affects recommendation.
    min_support : float or None
        Planned min_support threshold. Affects recommendation since
        efficient-apriori benefits greatly from higher thresholds.

    Returns
    -------
    pd.DataFrame
        Item frequency distribution table (item, count, support, cumulative%).
    """
    # --- Basic counts ---
    n_rows = len(df)
    txn_groups = df.groupby(transaction_col)[item_col]
    n_txn = txn_groups.ngroups
    items_per_txn = txn_groups.count()

    item_counts = df[item_col].value_counts()
    n_items = len(item_counts)

    avg_iptn = items_per_txn.mean()
    median_iptn = items_per_txn.median()
    std_iptn = items_per_txn.std()
    max_iptn = items_per_txn.max()
    min_iptn = items_per_txn.min()

    # --- Item frequency distribution ---
    item_supports = item_counts / n_txn
    freq_skewness = float(item_counts.skew())
    freq_kurtosis = float(item_counts.kurtosis())

    # Top-heavy ratio: fraction of total item appearances from top 10% of items
    top_10pct_n = max(1, n_items // 10)
    top_10pct_share = item_counts.nlargest(top_10pct_n).sum() / item_counts.sum()

    # Gini coefficient for item frequency inequality
    sorted_counts = np.sort(item_counts.values).astype(float)
    n = len(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts) - (n + 1) * np.sum(sorted_counts)) / (n * np.sum(sorted_counts)) if np.sum(sorted_counts) > 0 else 0.0

    # Distribution shape classification
    if gini < 0.15:
        dist_shape = "uniform"
    elif gini < 0.35:
        dist_shape = "mildly skewed"
    elif gini < 0.55:
        dist_shape = "moderately skewed (Zipf-like)"
    else:
        dist_shape = "highly skewed (power-law)"

    # --- Density metrics ---
    density = avg_iptn / n_items  # fraction of items per transaction
    if k == 2:
        avg_pairs_per_txn = avg_iptn * (avg_iptn - 1) / 2
        theoretical_pairs = n_items * (n_items - 1) / 2
        pair_density = avg_pairs_per_txn / theoretical_pairs if theoretical_pairs > 0 else 0
    else:
        from math import comb
        avg_ksets_per_txn = comb(int(round(avg_iptn)), k) if avg_iptn >= k else 0
        theoretical_ksets = comb(n_items, k) if n_items >= k else 0
        pair_density = avg_ksets_per_txn / theoretical_ksets if theoretical_ksets > 0 else 0

    # --- Effective min_support analysis ---
    min_item_support = float(item_supports.min())
    max_item_support = float(item_supports.max())
    median_item_support = float(item_supports.median())

    if min_support is not None:
        items_surviving = int((item_supports >= min_support).sum())
        pct_surviving = 100.0 * items_surviving / n_items
    else:
        items_surviving = n_items
        pct_surviving = 100.0

    # --- Print profile ---
    print("=" * 64)
    print("  DATASET PROFILE")
    print("=" * 64)
    print(f"  Rows (txn, item pairs)   : {n_rows:>12,}")
    print(f"  Transactions             : {n_txn:>12,}")
    print(f"  Unique items             : {n_items:>12,}")
    print()
    print("  Items per transaction:")
    print(f"    mean / median          : {avg_iptn:>8.1f} / {median_iptn:.1f}")
    print(f"    std                    : {std_iptn:>8.1f}")
    print(f"    range                  : [{min_iptn}, {max_iptn}]")
    print()
    print("  Item frequency distribution:")
    print(f"    skewness               : {freq_skewness:>8.2f}")
    print(f"    kurtosis               : {freq_kurtosis:>8.2f}")
    print(f"    Gini coefficient       : {gini:>8.3f}")
    print(f"    top 10% items share    : {top_10pct_share:>8.1%}")
    print(f"    shape                  : {dist_shape}")
    print()
    print(f"    min item support       : {min_item_support:.6f}")
    print(f"    median item support    : {median_item_support:.6f}")
    print(f"    max item support       : {max_item_support:.6f}")
    print()
    print("  Density:")
    print(f"    items/txn / n_items    : {density:.6f}")
    if k == 2:
        print(f"    avg pairs per txn      : {avg_pairs_per_txn:>8.1f}")
    else:
        print(f"    avg {k}-sets per txn     : {avg_ksets_per_txn:>8.1f}")
    print(f"    k={k} set density         : {pair_density:.2e}")

    # --- min_support analysis ---
    if min_support is not None:
        print()
        print(f"  min_support = {min_support}:")
        print(f"    items surviving        : {items_surviving:>6} / {n_items} ({pct_surviving:.1f}%)")
        if pct_surviving < 5:
            print("    WARNING: <5% of items survive -- efficient-apriori will")
            print("    prune aggressively, making it appear fast but finding nothing.")
        elif pct_surviving < 50:
            print("    NOTE: Many items pruned -- favors efficient-apriori (less work).")

    # --- Recommendation ---
    print()
    print("-" * 64)
    print("  RECOMMENDATION (k={})".format(k))
    print("-" * 64)

    reasons_fa = []
    reasons_ea = []

    # Factor 1: min_support effect
    if min_support is not None:
        if min_support < 0.005:
            reasons_fa.append(f"Low min_support ({min_support}) -- EA cannot prune, "
                              f"FA constant-time")
        elif min_support > 0.05:
            reasons_ea.append(f"High min_support ({min_support}) -- EA prunes heavily, "
                              f"reducing candidates")

    # Factor 2: Dataset size
    if n_txn > 100_000:
        reasons_fa.append(f"Large dataset ({n_txn:,} txn) -- FA scales linearly, "
                          f"EA candidate explosion risk")
    elif n_txn < 10_000:
        reasons_ea.append(f"Small dataset ({n_txn:,} txn) -- EA overhead minimal")

    # Factor 3: Item count
    if n_items > 5_000:
        reasons_fa.append(f"Many items ({n_items:,}) -- EA candidate generation "
                          f"O(items^k) is expensive")
    elif n_items < 200:
        reasons_ea.append(f"Few items ({n_items}) -- EA candidate space small")

    # Factor 4: Density
    if avg_iptn > 15:
        reasons_fa.append(f"Dense transactions ({avg_iptn:.0f} items/txn) -- "
                          f"many co-occurrences favor Counter+chain")
    elif avg_iptn < 5 and n_items < 500:
        reasons_ea.append(f"Sparse transactions ({avg_iptn:.0f} items/txn, "
                          f"{n_items} items) -- few candidates for EA")

    # Factor 5: Distribution shape
    if gini > 0.4:
        reasons_ea.append(f"Skewed distribution (Gini={gini:.2f}) -- "
                          f"hot items create prunable candidates in EA")
    elif gini < 0.15:
        reasons_fa.append(f"Uniform distribution (Gini={gini:.2f}) -- "
                          f"EA cannot exploit frequency variance for pruning")

    # Factor 6: k level
    if k >= 3:
        if min_support is not None and min_support > 0.01:
            reasons_ea.append(f"k={k} with moderate support -- EA prunes at "
                              f"each level, compounding savings")
        else:
            reasons_fa.append(f"k={k} with low/no support threshold -- "
                              f"EA must enumerate all candidates")

    # Verdict
    fa_score = len(reasons_fa)
    ea_score = len(reasons_ea)

    if fa_score > ea_score:
        verdict = "fastapriori (Counter+chain)"
        marker = ">>>"
    elif ea_score > fa_score:
        verdict = "efficient-apriori (Apriori)"
        marker = ">>>"
    else:
        verdict = "Either -- profile is mixed"
        marker = "---"

    print(f"\n  {marker} Use: {verdict}\n")

    if reasons_fa:
        print("  Factors favoring fastapriori:")
        for r in reasons_fa:
            print(f"    + {r}")
    if reasons_ea:
        print("  Factors favoring efficient-apriori:")
        for r in reasons_ea:
            print(f"    + {r}")

    if not reasons_fa and not reasons_ea:
        print("  No strong signal either way -- benchmark on a sample to decide.")

    print("=" * 64)

    # --- Return item frequency table ---
    freq_df = pd.DataFrame({
        "item": item_counts.index,
        "count": item_counts.values,
        "support": item_supports.values,
    })
    freq_df = freq_df.sort_values("count", ascending=False).reset_index(drop=True)
    freq_df["cumulative_pct"] = 100.0 * freq_df["count"].cumsum() / freq_df["count"].sum()
    return freq_df


def generate_synthetic_dataset(
    n_transactions: int = 1_416_769,
    n_items: int = 48_849,
    avg_items_per_txn: float = 7.8,
    items_per_txn_std: float = 14.0,
    item_freq_exponent: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic transactional dataset with realistic properties.

    Produces a DataFrame mimicking the statistical fingerprint of a real
    parts-bundling dataset: heavy-tailed items-per-transaction distribution
    (negative binomial) and skewed item frequencies (Zipf-like).

    Default parameters match a ~11M-row industrial parts dataset with
    ~1.4M transactions and ~49K unique items.

    Parameters
    ----------
    n_transactions : int
        Number of unique transactions.
    n_items : int
        Number of unique items (labeled 1 to n_items).
    avg_items_per_txn : float
        Target mean items per transaction.
    items_per_txn_std : float
        Target standard deviation of items per transaction. When std > sqrt(mean),
        a negative binomial is used to produce a heavy right tail; otherwise
        Poisson is used.
    item_freq_exponent : float
        Zipf exponent for item selection probabilities. Higher values make the
        distribution more skewed (0 = uniform, 1 = classic Zipf).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns ``["txn_id", "item"]`` where both are integers. Each row
        is one (transaction, item) pair.
    """
    rng = np.random.default_rng(seed)
    mu = avg_items_per_txn
    sigma = items_per_txn_std

    # --- Items per transaction ---
    if sigma**2 > mu:
        # Negative binomial: parameterize from mean and variance.
        # NB gives 0-based counts; we add 1 so every txn has >= 1 item.
        # Target mu_nb = mu - 1 to compensate for the +1 shift.
        mu_nb = max(mu - 1, 0.1)
        variance_nb = sigma**2  # variance is ~unchanged by +1 shift
        r = mu_nb**2 / (variance_nb - mu_nb)
        p = r / (r + mu_nb)
        sizes = rng.negative_binomial(r, p, size=n_transactions) + 1
    else:
        sizes = rng.poisson(mu, size=n_transactions)

    # Clip to [1, n_items]
    sizes = np.clip(sizes, 1, n_items)

    # --- Item selection probabilities (Zipf-like) ---
    ranks = np.arange(1, n_items + 1, dtype=np.float64)
    probs = 1.0 / ranks**item_freq_exponent
    probs /= probs.sum()

    # --- Build rows (vectorized) ---
    # Strategy: sample all items at once WITH replacement using Zipf weights,
    # then deduplicate within each transaction. For typical k << n_items,
    # collision rate is negligible (~k^2 / 2*n_items).
    total_rows = int(sizes.sum())
    all_items = rng.choice(n_items, size=total_rows, replace=True, p=probs) + 1
    txn_ids = np.repeat(np.arange(n_transactions, dtype=np.int64), sizes)

    df = pd.DataFrame({"txn_id": txn_ids, "item": all_items})
    # Drop within-transaction duplicates (rare for k << n_items)
    df = df.drop_duplicates(subset=["txn_id", "item"])

    return df.reset_index(drop=True)


def get_top_associations(
    result_df: pd.DataFrame,
    item: str,
    metric: str = "lift",
    n: int = 10,
    role: str = "any",
) -> pd.DataFrame:
    """Get top N associated items for a given item.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output from find_associations().
    item : str
        The item to look up.
    metric : str
        Column to sort by (e.g. "lift", "confidence", "support").
    n : int
        Number of top associations to return.
    role : str
        "antecedent" (item as item_A), "consequent" (item as item_B),
        or "any" (both directions).

    Returns
    -------
    pd.DataFrame
        Top N associations sorted by metric descending.
    """
    if role == "antecedent":
        subset = result_df[result_df["item_A"] == item]
    elif role == "consequent":
        subset = result_df[result_df["item_B"] == item]
    else:
        subset = result_df[
            (result_df["item_A"] == item) | (result_df["item_B"] == item)
        ]
    return subset.nlargest(n, metric).reset_index(drop=True)


def filter_associations(
    result_df: pd.DataFrame,
    items: str | list[str],
    role: str = "any",
) -> pd.DataFrame:
    """Filter association results by item(s).

    Parameters
    ----------
    result_df : pd.DataFrame
        Output from find_associations().
    items : str or list of str
        Item(s) to filter by.
    role : str
        "antecedent" (items as item_A), "consequent" (items as item_B),
        or "any" (either direction).

    Returns
    -------
    pd.DataFrame
        Filtered associations.
    """
    if isinstance(items, str):
        items = [items]
    items_set = set(items)

    if role == "antecedent":
        mask = result_df["item_A"].isin(items_set)
    elif role == "consequent":
        mask = result_df["item_B"].isin(items_set)
    else:
        mask = result_df["item_A"].isin(items_set) | result_df["item_B"].isin(items_set)
    return result_df[mask].reset_index(drop=True)


def to_heatmap(
    result_df: pd.DataFrame,
    metric: str = "lift",
) -> pd.DataFrame:
    """Create a pivot table (item_A x item_B) for heatmap visualization.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output from find_associations().
    metric : str
        Column to use as cell values.

    Returns
    -------
    pd.DataFrame
        Pivot table with item_A as rows, item_B as columns.
    """
    return result_df.pivot_table(
        index="item_A", columns="item_B", values=metric, aggfunc="first"
    ).fillna(0)


def plot_heatmap(
    result_df: pd.DataFrame,
    metric: str = "lift",
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
    annot: bool = True,
    fmt: str = ".2f",
):
    """Plot a heatmap with a color bar using matplotlib.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output from find_associations().
    metric : str
        Column to use as cell values.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float or None
        Color scale bounds. Auto-detected if None.
    figsize : tuple or None
        Figure size (width, height). Auto-scaled if None.
    title : str or None
        Plot title. Defaults to the metric name.
    annot : bool
        Annotate cells with numeric values.
    fmt : str
        Format string for annotations.

    Returns
    -------
    matplotlib.figure.Figure
        The heatmap figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_heatmap(). "
            "Install it with: pip install matplotlib"
        )

    pivot = to_heatmap(result_df, metric=metric)

    n_rows, n_cols = pivot.shape
    if figsize is None:
        figsize = (max(6, n_cols * 0.9 + 2), max(4, n_rows * 0.7 + 1.5))
    if title is None:
        title = f"Heatmap — {metric}"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("item_B")
    ax.set_ylabel("item_A")
    ax.set_title(title)

    # Color bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric)

    # Annotate cells
    if annot:
        for i in range(n_rows):
            for j in range(n_cols):
                val = pivot.values[i, j]
                color = "white" if abs(val - (im.norm.vmin or 0)) < 0.3 * ((im.norm.vmax or 1) - (im.norm.vmin or 0)) else "black"
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        color=color, fontsize=8)

    fig.tight_layout()
    return fig


def to_graph(
    result_df: pd.DataFrame,
    metric: str = "lift",
    min_value: float = 1.0,
):
    """Export associations as a NetworkX directed graph.

    Edges are weighted by the chosen metric. Only edges where
    metric >= min_value are included.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output from find_associations().
    metric : str
        Column to use as edge weight.
    min_value : float
        Minimum metric value to include an edge.

    Returns
    -------
    networkx.DiGraph
        Directed graph with weighted edges.

    Raises
    ------
    ImportError
        If networkx is not installed.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for to_graph(). "
            "Install it with: pip install networkx"
        )

    filtered = result_df[result_df[metric] >= min_value]
    G = nx.DiGraph()
    for _, row in filtered.iterrows():
        G.add_edge(
            row["item_A"],
            row["item_B"],
            weight=row[metric],
            instances=row["instances"],
            support=row["support"],
            confidence=row["confidence"],
            lift=row["lift"],
        )
    return G
