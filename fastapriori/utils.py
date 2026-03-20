"""Utility functions for working with fastapriori association results."""

from __future__ import annotations

import pandas as pd


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
