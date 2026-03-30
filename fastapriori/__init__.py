"""fastapriori - Fast pairwise co-occurrence and association analysis."""

import warnings as _warnings

try:
    from fastapriori._fastapriori_rs import rust_compute_pairs as _  # noqa: F401
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    _warnings.warn(
        "Rust extension not found — falling back to Python backends. "
        "For best performance, install the Rust toolchain "
        "(https://rustup.rs) and reinstall: pip install -e .",
        stacklevel=1,
    )

from fastapriori.core import find_associations
from fastapriori.itemsets import find_itemsets
from fastapriori.triplets import find_triplets
from fastapriori.utils import (
    describe_dataset,
    filter_associations,
    generate_synthetic_dataset,
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
    "describe_dataset",
    "generate_synthetic_dataset",
    "get_top_associations",
    "filter_associations",
    "plot_heatmap",
    "to_graph",
    "to_heatmap",
]
