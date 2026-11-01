"""fastapriori - Fast pairwise co-occurrence and association analysis."""

import logging as _logging
import os as _os
import warnings as _warnings

try:
    from fastapriori._fastapriori_rs import rust_compute_pairs as _  # noqa: F401
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    # Set FASTAPRIORI_QUIET=1 to silence this warning in CI / notebook use
    # where the Python fallback is an intentional choice.
    if not _os.environ.get("FASTAPRIORI_QUIET"):
        _warnings.warn(
            "Rust extension not found — falling back to Python backends. "
            "For best performance, install the Rust toolchain "
            "(https://rustup.rs) and reinstall: pip install -e . "
            "Set FASTAPRIORI_QUIET=1 to silence.",
            stacklevel=1,
        )
    else:
        _logging.getLogger("fastapriori").debug(
            "Rust extension not found; using Python backends (quiet mode)."
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

from importlib.metadata import PackageNotFoundError as _PNF, version as _pkg_version

try:
    __version__ = _pkg_version("fastapriori")
except _PNF:
    __version__ = "0.0.0+unknown"
del _PNF, _pkg_version
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
