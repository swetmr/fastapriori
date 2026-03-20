"""fastapriori - Fast pairwise co-occurrence and association analysis."""

from fastapriori.core import find_associations
from fastapriori.itemsets import find_itemsets
from fastapriori.triplets import find_triplets
from fastapriori.utils import (
    filter_associations,
    get_top_associations,
    plot_heatmap,
    to_graph,
    to_heatmap,
)

__version__ = "0.1.0"
__all__ = [
    "find_associations",
    "find_itemsets",
    "find_triplets",
    "get_top_associations",
    "filter_associations",
    "plot_heatmap",
    "to_graph",
    "to_heatmap",
]
