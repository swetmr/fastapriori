# Changelog

## 0.2.1 — 2026-10-30

### Added
- `algo="seeded"`: scatter-seeded Eclat at k>=3 as a supported option, replacing the
  internal `backend_options={"fast_variant": "eclat_seeded"}` route (which still works). It
  returns the same rules as `algo="fast"`, needs the Rust backend and does not support
  `max_items_per_txn`. At k=2 it is `algo="fast"`.
- The default is now `algo="auto"`, which takes `algo="seeded"` at k>=3 and `algo="fast"` at k=2, or
  whenever the seeded path is unavailable (no Rust engine, a non-Rust `backend`, `max_items_per_txn`
  or `backend_options` set). Rules are unchanged. Pass `algo="fast"` to keep anchor-and-extend at k>=3.
- Opt-in phase profiling for the default k≥3 Rust path (`rust_compute_pipeline_v3_adaptive`),
  behind the Cargo feature `phaseprof`. In a build with the feature, `FASTAPRIORI_PROFILE=<file>`
  makes each call write one JSON object with Python-side times (encoding, Rust call, decoding),
  Rust phase times (inverted index, pair counting, pair TID-lists, per level projection and
  anchor-extend, output) and per-level work counters. See the README. This is the instrumentation
  used for the phase-profiling experiment of the accompanying paper.
- A default build warns when `FASTAPRIORI_PROFILE` is set on the default k≥3 path.

### Fixed
- Rust-backed paths: integer item labels no longer collapse when NumPy would store them as
  float (for example `-1` together with `2**63`); such labels now keep their exact values.
- Python backends: `min_support` no longer counts transactions whose rows are all NaN in the
  denominator (the pandas and polars k=2 paths and the classic k=2 path). The Rust `algo="fast"`
  path was not affected.
- Python k≥3 path: antecedent counts come from `instances` when available, not from `support`
  rounded to 6 decimals, which could be off by one above about 1M transactions.
- Python k>=4 path: `min_confidence` was also applied to the intermediate levels, which dropped
  valid rules because confidence is not anti-monotone. It now applies to the final rules only.
  The Rust `algo="fast"` path was not affected.
- Python k=2 paths and the classic k=2 path: metric thresholds are tested on exact counts, not on
  the rounded output columns, so a pair sitting on a threshold such as 1/3 is kept.

### Changed
- `benchmarks/` is no longer included in the source distribution.
- `requires-python` is now `>=3.9,<3.14`: PyO3 0.23 does not build on CPython 3.14 and no wheels
  exist for it.

### Unchanged
- Default builds (including the PyPI wheels) compile the 0.2.0 Rust code unchanged (the fixes
  above are Python only): the extension
  built from 0.2.1 without the feature is byte-identical to the one built from 0.2.0 (same
  toolchain and path). The Rust crate version stays 0.2.0 for this reason.
- Results are identical with and without the feature. The instrumented build can be a few percent
  slower even with profiling off (measured: up to 3.5% at one thread), a code-layout effect.

## 0.2.0 — 2026-10-30

### Added
- `backend_options={"pair_variant": ...}` for k=2 on the Rust backend with `algo="fast"`:
  `"suffix"` (count only the sorted B>A suffix), `"filtered"` (skip items below the
  minimum count before counting), `"suffix_filtered"` (both). Default `"v0"` is the
  0.1.x kernel. All variants return the same pairs and counts.
- `backend_options={"fast_variant": "eclat_seeded"}` for k≥3: Eclat whose first level comes
  from suffix-scatter pair counting (frequency-ordered items, count-only last level).
  Returns the same itemsets as the default anchor-and-extend path.
- Rust functions `rust_compute_pairs_v2` and `rust_eclat_pipeline_seeded` backing the above.

### Changed
- `min_conviction` now defaults to `None` (was `0.0`). Conviction is NaN when the consequent
  occurs in every transaction, so any threshold silently dropped those rules.
- k≥3: the pair level used to seed higher levels is filtered on `min_support` only. Before,
  metric thresholds of 0.0 were applied internally and dropped pairs of two items that occur in
  every transaction.

### Fixed
- Minimum count: every Rust kernel computed `ceil(min_support * n_transactions)` in floating
  point, which overshoots by one when the exact product is an integer (e.g. `0.07 * 100`,
  `1e-5 * 300_000`), silently dropping itemsets whose support equals `min_support`. All kernels now
  use the smallest count `c` with `c / n >= min_support`, the same rule as the Python backends.
  (None of the configurations in the accompanying benchmark hit this boundary.)
- Rounding before filtering: `support`/`confidence` were rounded to 6 decimals *before* the
  `min_support`/`min_confidence` filter (Rust k≥3, all Python backends, classic k=2), dropping
  itemsets for thresholds with more than 6 decimals. Support is now tested on exact counts and
  k≥3 confidence on unrounded values; output values are still rounded. k≥3 confidence and lift
  are now computed from exact counts without the former `+1e-10` in the denominator, so a
  rule with confidence exactly 1.0 passes `min_confidence=1.0`.
- Sorted output (`sorted_by=...`) is now deterministic: ties are broken by the item columns with a
  stable sort. Before, tied rows could come out in a different order from run to run.
- Inputs with more than 2³²−1 (transaction, item) rows now raise `ValueError` instead of
  overflowing the Rust index's 32-bit offsets.

### Unchanged
- The default code paths (`algo="fast"` and `algo="classic"` without `backend_options`) run the
  same Rust kernels as 0.1.0.post1 apart from the minimum-count fix above.
