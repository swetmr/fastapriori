# Changelog

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
