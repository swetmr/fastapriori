# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development

The package is a Python extension with a compiled Rust core (PyO3). The Rust module is exposed as `fastapriori._fastapriori_rs` and is mandatory for the `rust` / `classic` backends; pure-Python backends (`pandas`, `polars`, `counter_chain`, `bin_multi`) are used when the extension is missing.

- **Build & install (dev)**: `maturin develop --release`  — compiles the Rust crate in-place and installs the Python package. Despite `pyproject.toml` listing `setuptools-rust` under `[build-system]`, the project is built with **maturin** in CI (`PyO3/maturin-action@v1`) and locally; the `[tool.maturin]` section is authoritative. `pip install -e .` works but produces a debug build.
- **Build release wheel**: `maturin build --release`
- **Run tests**: `pytest tests/` (or `pytest tests/test_algo_parameter.py::TestAlgoValidation::test_auto_routes_to_fast` for a single test). There is no pytest config file; tests assume the Rust extension is built.
- **Rust-only checks**: `cargo check` / `cargo build --release`. The crate is `crate-type = ["cdylib"]` only, so `cargo test` has no Rust tests to run — all tests are Python-side.

## Architecture

The project implements association-rule mining with a "count everything, filter later" strategy. The important structural points:

### Two algorithms, one entry point
`find_associations()` (`fastapriori/core.py`) dispatches on the `algo` parameter:
- `algo="fast"` (default) → inverted-index count-all. At **k=2** runtime is constant in `min_support` (threshold is post-hoc). At **k>=3** it uses anchor-and-extend with Apriori downward-closure pruning.
- `algo="classic"` → Rust port of efficient-apriori (join + prune + short-circuit). **Requires `min_support`**. Only useful for dense, correlated data at k>=4.
- `algo="auto"` currently routes to `"fast"` (the ML-based router mentioned in the docstring is not implemented).

### Backend resolution (fast path only)
`backend="auto"` → tries `rust`, else `python`. `backend="python"` → `polars` (or `pandas` fallback) at k=2, `counter_chain` at k>=3. The Rust path at k>=3 **short-circuits the entire k=2→…→k_max chain into a single `compute_pipeline` call** (see `_find_k_itemsets` when `frequent_lower is None`), avoiding Python round-trips between levels. If `frequent_lower` is supplied, Python orchestrates level-by-level.

### Rust crate layout (`src/`)
- `lib.rs` — PyO3 module registration. Exposes 5 functions: `rust_compute_pairs`, `rust_compute_itemsets`, `rust_compute_pipeline`, `rust_classic_compute_pairs`, `rust_classic_compute_pipeline`.
- `common.rs` — shared inverted-index builder (`build_inverted_index`) and encoders.
- `pairs.rs` — k=2 count-all using a flat `counts: Vec<u32>` buffer per anchor item (the inner loop resets only touched slots, giving O(pair-work) not O(m²)).
- `itemsets.rs` — k>=3 anchor-and-extend with Apriori pruning.
- `pipeline.rs` — one-shot k=2→k_max chain that reuses the inverted index across levels.
- `classic.rs` — Apriori port (candidate generation, join+prune, short-circuit for sparse data).

### Python package layout (`fastapriori/`)
- `core.py` — `find_associations()` is the single canonical entry point. `itemsets.py` / `triplets.py` are deprecated wrappers that delegate to it.
- `backends/` — one module per backend: `rust_backend.py`, `rust_classic_backend.py`, `polars_backend.py`, `pandas_backend.py`, `polars_itemset_backend.py`, `bin_multi_backend.py`, `itemset_counter_chain.py`. Each exposes `compute_associations` (k=2) and/or `compute_itemsets` / `compute_pipeline` (k>=3) with matching signatures.
- `utils.py` — post-processing helpers (`get_top_associations`, `filter_associations`, `to_heatmap`, `plot_heatmap`, `to_graph`, `describe_dataset`, `generate_synthetic_dataset`).

### Non-obvious semantics to preserve when editing

- **`low_memory="auto"`** enables pre-filtering whenever `min_support` is set. When rows are dropped, transactions that lose *all* their items are preserved via a sentinel item (`_find_associations` in `core.py`) so `n_transactions` — and therefore every support denominator — stays exact.
- **`max_items_per_txn`** produces **lower-bound** counts by design. Changes to the capping logic must not produce over-counts. Pairs (k=2) are always counted exactly; capping only applies at k>=3.
- **Item encoding**: Rust backends factorize items to sequential `i32` IDs and transactions to `i64` on the Python side, then decode back via a numpy array of the originals. The Rust code assumes dense 0..n_items IDs.
- **k=2 output is canonical undirected** (`item_A < item_B`); k>=3 output is **k directional rules per itemset** (one row per choice of consequent), which is why row counts scale with k.

## Dependencies & Packaging

- Runtime deps: `pandas>=1.5`, `numpy>=1.23` only. `polars`, `tqdm`, `networkx`, `matplotlib` are optional extras.
- Rust deps (`Cargo.toml`): `pyo3 0.23`, `numpy 0.23`, `rayon 1.10`. The PyO3 / numpy versions must move together.
- Release to PyPI is automated via `.github/workflows/release.yml` on GitHub Release publish (builds wheels for Linux x86_64/aarch64, macOS x86_64/arm64, Windows x64, Python 3.9–3.13, plus sdist).
