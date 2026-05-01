use std::collections::{HashMap, HashSet};

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::common::{self, InvertedIndex, TidList};
use roaring::RoaringBitmap;

/// Lazy iterator over C(n, r) combinations of a sorted slice.
/// Yields one `Vec<u32>` at a time — no bulk allocation.
struct CombinationIter<'a> {
    items: &'a [u32],
    indices: Vec<usize>,
    r: usize,
    first: bool,
    done: bool,
}

impl<'a> CombinationIter<'a> {
    fn new(items: &'a [u32], r: usize) -> Self {
        if r > items.len() {
            return Self { items, indices: Vec::new(), r, first: true, done: true };
        }
        Self {
            items,
            indices: (0..r).collect(),
            r,
            first: true,
            done: r == 0,
        }
    }
}

impl<'a> Iterator for CombinationIter<'a> {
    type Item = Vec<u32>;

    fn next(&mut self) -> Option<Vec<u32>> {
        if self.done {
            return None;
        }

        if self.first {
            self.first = false;
            return Some(self.indices.iter().map(|&i| self.items[i]).collect());
        }

        // Find rightmost index that can be incremented
        let n = self.items.len();
        let r = self.r;
        let mut i = r;
        loop {
            if i == 0 {
                self.done = true;
                return None;
            }
            i -= 1;
            if self.indices[i] != i + n - r {
                break;
            }
        }
        self.indices[i] += 1;
        for j in (i + 1)..r {
            self.indices[j] = self.indices[j - 1] + 1;
        }
        Some(self.indices.iter().map(|&i| self.items[i]).collect())
    }
}

/// Check that adding `new_item` to `anchor` keeps all pairs frequent.
fn apriori_check(anchor: &[u32], new_item: u32, freq_pairs: &HashSet<(u32, u32)>) -> bool {
    for &existing in anchor {
        let pair = if existing < new_item {
            (existing, new_item)
        } else {
            (new_item, existing)
        };
        if !freq_pairs.contains(&pair) {
            return false;
        }
    }
    true
}

/// Process a chunk of anchor (k-1)-sets: count co-occurring items, apply Apriori, dedup.
fn process_anchors(
    anchors: &[Vec<u32>],
    lower_to_txns: &HashMap<Vec<u32>, Vec<u32>>,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    _k: usize,
    has_freq_pairs: bool,
) -> HashMap<Vec<u32>, u32> {
    let mut result: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut counts = vec![0u64; n_items];
    // Dirty-list sparse scatter: record touched indices so we reset O(|dirty|)
    // rather than O(n_items) per anchor. Mirror of src/pairs.rs.
    let mut dirty: Vec<u32> = Vec::with_capacity(256);

    for anchor in anchors {
        if let Some(txn_codes) = lower_to_txns.get(anchor) {
            for &txn in txn_codes {
                if let Some(items) = txn_to_items.get(&txn) {
                    for &item in items {
                        let idx = item as usize;
                        if counts[idx] == 0 {
                            dirty.push(item);
                        }
                        counts[idx] += 1;
                    }
                }
            }

            for &item in &dirty {
                let idx = item as usize;
                let count = counts[idx];
                counts[idx] = 0;

                // Anchors are small (k-1 ≤ ~5) and sorted — linear scan beats
                // a per-anchor HashSet build + probe.
                if anchor.contains(&item) {
                    continue;
                }

                let mut kset = anchor.clone();
                kset.push(item);
                kset.sort_unstable();

                if result.contains_key(&kset) {
                    continue;
                }

                if has_freq_pairs && !apriori_check(anchor, item, freq_pairs) {
                    continue;
                }

                result.insert(kset, count as u32);
            }
            dirty.clear();
        }
    }

    result
}

/// Run anchor-and-extend counting with optional Rayon parallelism.
fn run_anchor_extend(
    anchor_keys: &[Vec<u32>],
    lower_to_txns: &HashMap<Vec<u32>, Vec<u32>>,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    k: usize,
) -> HashMap<Vec<u32>, u32> {
    let has_freq_pairs = !freq_pairs.is_empty();

    if anchor_keys.len() > 500 {
        let chunk_size = std::cmp::max(1, anchor_keys.len() / rayon::current_num_threads());
        let chunks: Vec<&[Vec<u32>]> = anchor_keys.chunks(chunk_size).collect();

        let partial_results: Vec<HashMap<Vec<u32>, u32>> = chunks
            .par_iter()
            .map(|chunk| {
                process_anchors(
                    chunk,
                    lower_to_txns,
                    txn_to_items,
                    freq_pairs,
                    n_items,
                    k,
                    has_freq_pairs,
                )
            })
            .collect();

        // First-write-wins merge. Correctness invariant: each canonical k-set
        // S has a unique (anchor, extra_item) decomposition in
        // `process_anchors` — the anchor equals S.without(max(S)). Each
        // anchor is produced by exactly one chunk, so `S`'s count is
        // written exactly once. `or_insert` is therefore equivalent to a
        // collision-free merge; summing would double-count.
        let mut merged: HashMap<Vec<u32>, u32> = HashMap::new();
        for partial in partial_results {
            for (key, val) in partial {
                merged.entry(key).or_insert(val);
            }
        }
        merged
    } else {
        process_anchors(
            anchor_keys,
            lower_to_txns,
            txn_to_items,
            freq_pairs,
            n_items,
            k,
            has_freq_pairs,
        )
    }
}

/// Internal function: count k-itemsets from the CSR inverted index.
/// Used by pipeline and by rust_compute_itemsets.
///
/// `item_weights`: per-item weight indexed by item_id (used for capping).
/// `max_items_per_txn`: if Some(n), cap each transaction to its top-n items
///   by weight after frequency filtering.  Reduces C(d, k-1) for outlier
///   transactions.
pub fn count_k_itemsets_internal(
    idx: &InvertedIndex,
    freq_lower_sets: &HashSet<Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    k: usize,
    n_items: usize,
    item_weights: &[f64],
    max_items_per_txn: Option<usize>,
) -> HashMap<Vec<u32>, u32> {
    let lower_k = k - 1;

    // Opt 1: items that appear in any frequent (k-1)-set. Items outside this
    // set cannot contribute to any frequent k-set (downward closure).
    let freq_items: HashSet<u32> = freq_lower_sets.iter()
        .flat_map(|s| s.iter().copied())
        .collect();

    // Opt 2: pre-filter each txn's items to the frequent subset. Keyed by
    // dense u32 txn code — avoids the i64 HashMap lookup that dominated
    // the old path.
    let mut txn_to_items_filtered: HashMap<u32, Vec<u32>> =
        HashMap::with_capacity(idx.n_txns as usize);
    for txn_code in 0..idx.n_txns {
        let items = idx.txn_items(txn_code);
        let filtered: Vec<u32> = items.iter()
            .copied()
            .filter(|item| freq_items.contains(item))
            .collect();
        if filtered.len() >= k {
            txn_to_items_filtered.insert(txn_code, filtered);
        }
    }

    // Opt 2b: cap per-transaction items to avoid combinatorial explosion.
    if let Some(max_items) = max_items_per_txn {
        for items in txn_to_items_filtered.values_mut() {
            if items.len() > max_items {
                items.sort_by(|a, b| {
                    let wa = item_weights.get(*a as usize).copied().unwrap_or(0.0);
                    let wb = item_weights.get(*b as usize).copied().unwrap_or(0.0);
                    wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
                });
                items.truncate(max_items);
                items.sort_unstable();
            }
        }
    }

    // Opt 3: build lower_to_txns (Rayon for large n_txns).
    let lower_to_txns: HashMap<Vec<u32>, Vec<u32>> = if txn_to_items_filtered.len() > 10_000 {
        let txn_vec: Vec<(&u32, &Vec<u32>)> = txn_to_items_filtered.iter().collect();
        let partial_maps: Vec<HashMap<Vec<u32>, Vec<u32>>> = txn_vec
            .par_chunks(std::cmp::max(1, txn_vec.len() / rayon::current_num_threads()))
            .map(|chunk| {
                let mut local: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
                for &(&txn_code, items) in chunk {
                    for combo in CombinationIter::new(items, lower_k) {
                        if freq_lower_sets.contains(&combo) {
                            local.entry(combo).or_default().push(txn_code);
                        }
                    }
                }
                local
            })
            .collect();

        let mut merged: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
        for partial in partial_maps {
            for (key, mut txns) in partial {
                merged.entry(key).or_insert_with(Vec::new).append(&mut txns);
            }
        }
        merged
    } else {
        let mut lower_to_txns: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
        for (&txn_code, items) in &txn_to_items_filtered {
            for combo in CombinationIter::new(items, lower_k) {
                if freq_lower_sets.contains(&combo) {
                    lower_to_txns.entry(combo).or_default().push(txn_code);
                }
            }
        }
        lower_to_txns
    };

    let anchor_keys: Vec<Vec<u32>> = lower_to_txns.keys().cloned().collect();

    run_anchor_extend(
        &anchor_keys,
        &lower_to_txns,
        &txn_to_items_filtered,
        freq_pairs,
        n_items,
        k,
    )
}

/// Compute k-itemset (k>=3) counts using anchor-and-extend with Rayon parallelism.
///
/// Accepts integer-encoded arrays + frequent (k-1)-sets.
/// Returns dict with "itemsets" (2D array) and "counts" (1D array).
#[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, freq_lower, k, n_items, item_weights, max_items_per_txn=None))]
pub fn rust_compute_itemsets<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    freq_lower: PyReadonlyArray2<'py, i32>,
    k: usize,
    n_items: u32,
    item_weights: PyReadonlyArray1<'py, f64>,
    max_items_per_txn: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;
    let weights_slice = item_weights.as_slice()?;

    let lower_k = k - 1;

    // Parse frequent (k-1)-sets from 2D array
    let freq_lower_shape = freq_lower.shape();
    let n_freq = freq_lower_shape[0];
    let freq_lower_raw = freq_lower.as_slice()?;

    let mut freq_lower_sets: HashSet<Vec<u32>> = HashSet::with_capacity(n_freq);
    for i in 0..n_freq {
        let start = i * lower_k;
        let set: Vec<u32> = freq_lower_raw[start..start + lower_k]
            .iter()
            .map(|&x| x as u32)
            .collect();
        freq_lower_sets.insert(set);
    }

    // Extract all frequent pairs for Apriori pruning
    let mut freq_pairs: HashSet<(u32, u32)> = HashSet::new();
    for lower_set in &freq_lower_sets {
        for i in 0..lower_set.len() {
            for j in (i + 1)..lower_set.len() {
                let a = lower_set[i];
                let b = lower_set[j];
                let pair = if a < b { (a, b) } else { (b, a) };
                freq_pairs.insert(pair);
            }
        }
    }

    let combined = py.allow_threads(|| {
        let idx = common::build_inverted_index(txn_slice, item_slice, n_items);
        count_k_itemsets_internal(
            &idx,
            &freq_lower_sets,
            &freq_pairs,
            k,
            n_items as usize,
            weights_slice,
            max_items_per_txn,
        )
    });

    // Build output arrays — collect into Vec to ensure consistent ordering
    let results: Vec<(Vec<u32>, u32)> = combined.into_iter().collect();

    let itemsets_vecs: Vec<Vec<i32>> = results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let out_counts: Vec<i64> = results.iter().map(|(_, count)| *count as i64).collect();

    let dict = PyDict::new(py);
    let itemsets_array = PyArray2::from_vec2(py, &itemsets_vecs)?;
    dict.set_item("itemsets", itemsets_array)?;
    dict.set_item("counts", PyArray1::from_vec(py, out_counts))?;

    Ok(dict)
}

// ============================================================================
// v1_roaring — phase 1 ablation: identical algorithm to count_k_itemsets_internal,
// but `lower_to_txns` values are stored as RoaringBitmap instead of Vec<u32>.
// ============================================================================

/// Same body as `process_anchors` but takes a per-anchor RoaringBitmap.
/// The Counter+chain inner loop is verbatim — only the source of `txn_codes`
/// changes from `&[u32]` to `bitmap.iter()`.
fn process_anchors_roaring(
    anchors: &[Vec<u32>],
    lower_to_txns: &HashMap<Vec<u32>, RoaringBitmap>,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    _k: usize,
    has_freq_pairs: bool,
) -> HashMap<Vec<u32>, u32> {
    let mut result: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut counts = vec![0u64; n_items];
    let mut dirty: Vec<u32> = Vec::with_capacity(256);

    for anchor in anchors {
        if let Some(tids) = lower_to_txns.get(anchor) {
            for txn in tids.iter() {
                if let Some(items) = txn_to_items.get(&txn) {
                    for &item in items {
                        let idx = item as usize;
                        if counts[idx] == 0 {
                            dirty.push(item);
                        }
                        counts[idx] += 1;
                    }
                }
            }

            for &item in &dirty {
                let idx = item as usize;
                let count = counts[idx];
                counts[idx] = 0;

                if anchor.contains(&item) {
                    continue;
                }

                let mut kset = anchor.clone();
                kset.push(item);
                kset.sort_unstable();

                if result.contains_key(&kset) {
                    continue;
                }

                if has_freq_pairs && !apriori_check(anchor, item, freq_pairs) {
                    continue;
                }

                result.insert(kset, count as u32);
            }
            dirty.clear();
        }
    }

    result
}

fn run_anchor_extend_roaring(
    anchor_keys: &[Vec<u32>],
    lower_to_txns: &HashMap<Vec<u32>, RoaringBitmap>,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    k: usize,
) -> HashMap<Vec<u32>, u32> {
    let has_freq_pairs = !freq_pairs.is_empty();

    if anchor_keys.len() > 500 {
        let chunk_size = std::cmp::max(1, anchor_keys.len() / rayon::current_num_threads());
        let chunks: Vec<&[Vec<u32>]> = anchor_keys.chunks(chunk_size).collect();

        let partial_results: Vec<HashMap<Vec<u32>, u32>> = chunks
            .par_iter()
            .map(|chunk| {
                process_anchors_roaring(
                    chunk,
                    lower_to_txns,
                    txn_to_items,
                    freq_pairs,
                    n_items,
                    k,
                    has_freq_pairs,
                )
            })
            .collect();

        // First-write-wins merge — same correctness invariant as the
        // non-roaring run_anchor_extend: each canonical k-set has a
        // unique (anchor, extra_item) decomposition produced by exactly
        // one chunk.
        let mut merged: HashMap<Vec<u32>, u32> = HashMap::new();
        for partial in partial_results {
            for (key, count) in partial {
                merged.entry(key).or_insert(count);
            }
        }
        merged
    } else {
        process_anchors_roaring(
            anchor_keys,
            lower_to_txns,
            txn_to_items,
            freq_pairs,
            n_items,
            k,
            has_freq_pairs,
        )
    }
}

/// Phase-1 variant of `count_k_itemsets_internal`: identical algorithm
/// (BFS-vertical Counter+chain with combo-enumerated `lower_to_txns` rebuild
/// each level), but `lower_to_txns` values are RoaringBitmaps instead of
/// `Vec<u32>`. There is **no cross-level memoisation** — that's phase 2.
///
/// Algorithmic identity: preserved (only the storage container differs).
pub fn count_k_itemsets_v1_roaring(
    idx: &InvertedIndex,
    freq_lower_sets: &HashSet<Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    k: usize,
    n_items: usize,
    item_weights: &[f64],
    max_items_per_txn: Option<usize>,
) -> HashMap<Vec<u32>, u32> {
    let lower_k = k - 1;

    let freq_items: HashSet<u32> = freq_lower_sets.iter()
        .flat_map(|s| s.iter().copied())
        .collect();

    let mut txn_to_items_filtered: HashMap<u32, Vec<u32>> =
        HashMap::with_capacity(idx.n_txns as usize);
    for txn_code in 0..idx.n_txns {
        let items = idx.txn_items(txn_code);
        let filtered: Vec<u32> = items.iter()
            .copied()
            .filter(|item| freq_items.contains(item))
            .collect();
        if filtered.len() >= k {
            txn_to_items_filtered.insert(txn_code, filtered);
        }
    }

    if let Some(max_items) = max_items_per_txn {
        for items in txn_to_items_filtered.values_mut() {
            if items.len() > max_items {
                items.sort_by(|a, b| {
                    let wa = item_weights.get(*a as usize).copied().unwrap_or(0.0);
                    let wb = item_weights.get(*b as usize).copied().unwrap_or(0.0);
                    wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
                });
                items.truncate(max_items);
                items.sort_unstable();
            }
        }
    }

    // Build lower_to_txns with RoaringBitmap values.
    let lower_to_txns: HashMap<Vec<u32>, RoaringBitmap> = if txn_to_items_filtered.len() > 10_000 {
        let txn_vec: Vec<(&u32, &Vec<u32>)> = txn_to_items_filtered.iter().collect();
        let partial_maps: Vec<HashMap<Vec<u32>, RoaringBitmap>> = txn_vec
            .par_chunks(std::cmp::max(1, txn_vec.len() / rayon::current_num_threads()))
            .map(|chunk| {
                let mut local: HashMap<Vec<u32>, RoaringBitmap> = HashMap::new();
                for &(&txn_code, items) in chunk {
                    for combo in CombinationIter::new(items, lower_k) {
                        if freq_lower_sets.contains(&combo) {
                            local.entry(combo).or_default().insert(txn_code);
                        }
                    }
                }
                local
            })
            .collect();

        let mut merged: HashMap<Vec<u32>, RoaringBitmap> = HashMap::new();
        for partial in partial_maps {
            for (key, txns) in partial {
                *merged.entry(key).or_default() |= txns;
            }
        }
        merged
    } else {
        let mut lower_to_txns: HashMap<Vec<u32>, RoaringBitmap> = HashMap::new();
        for (&txn_code, items) in &txn_to_items_filtered {
            for combo in CombinationIter::new(items, lower_k) {
                if freq_lower_sets.contains(&combo) {
                    lower_to_txns.entry(combo).or_default().insert(txn_code);
                }
            }
        }
        lower_to_txns
    };

    let anchor_keys: Vec<Vec<u32>> = lower_to_txns.keys().cloned().collect();

    run_anchor_extend_roaring(
        &anchor_keys,
        &lower_to_txns,
        &txn_to_items_filtered,
        freq_pairs,
        n_items,
        k,
    )
}

// ============================================================================
// v2_memo — phase 2 ablation: cross-level intersection memoisation.
//
// Identical Counter+chain inner loop as v1_roaring (BFS-vertical, sparse-scatter
// dirty-list, Apriori downward-closure check). Difference: instead of rebuilding
// `lower_to_txns` each level via combo enumeration over `txn_to_items_filtered`,
// we carry `lower_to_txns_{k-1}` from the previous BFS level and build
// `lower_to_txns_k` by intersecting each emitted survivor's parent TID-list
// with the extending item's TID-list.
//
// Algorithmic identity (preserved):
//   - BFS structure: level-by-level, one full L_k before L_{k+1}.
//   - Counter-discovery: extensions surfaced by per-anchor sparse-scatter
//     Counter, NOT by enumerating items past the prefix (eclat's pattern).
//   - Apriori downward-closure check: `apriori_check` retained.
// ============================================================================

use crate::common::ItemTxnsRoaring;

fn process_anchors_v2_memo(
    anchors: &[Vec<u32>],
    lower_to_txns_prev: &HashMap<Vec<u32>, RoaringBitmap>,
    item_roar: &ItemTxnsRoaring,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    min_count: u32,
    _k: usize,
    has_freq_pairs: bool,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, RoaringBitmap>) {
    let mut result: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut new_lower: HashMap<Vec<u32>, RoaringBitmap> = HashMap::new();
    let mut counts = vec![0u64; n_items];
    let mut dirty: Vec<u32> = Vec::with_capacity(256);

    for anchor in anchors {
        if let Some(parent_tids) = lower_to_txns_prev.get(anchor) {
            // Counter+chain inner loop — verbatim from process_anchors_roaring.
            for txn in parent_tids.iter() {
                if let Some(items) = txn_to_items.get(&txn) {
                    for &item in items {
                        let idx = item as usize;
                        if counts[idx] == 0 {
                            dirty.push(item);
                        }
                        counts[idx] += 1;
                    }
                }
            }

            for &item in &dirty {
                let idx = item as usize;
                let count = counts[idx];
                counts[idx] = 0;

                if anchor.contains(&item) {
                    continue;
                }
                // Early support filter: drop subthreshold extensions BEFORE
                // computing the (expensive) intersection. v0/v1 return all
                // counts and the caller filters; v2 filters here so we don't
                // pay the intersection for sets that won't survive anyway.
                if (count as u32) < min_count {
                    continue;
                }

                let mut kset = anchor.clone();
                kset.push(item);
                kset.sort_unstable();

                if result.contains_key(&kset) {
                    continue;
                }

                if has_freq_pairs && !apriori_check(anchor, item, freq_pairs) {
                    continue;
                }

                // Memoised level-k+1 TID-list = parent ∩ item_bitmap.
                // Both operands are RoaringBitmap; the BitAnd impl returns
                // a fresh RoaringBitmap with auto-selected container layout.
                let new_tids = parent_tids & item_roar.get(item);
                result.insert(kset.clone(), count as u32);
                new_lower.insert(kset, new_tids);
            }
            dirty.clear();
        }
    }

    (result, new_lower)
}

fn run_anchor_extend_v2_memo(
    anchor_keys: &[Vec<u32>],
    lower_to_txns_prev: &HashMap<Vec<u32>, RoaringBitmap>,
    item_roar: &ItemTxnsRoaring,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    min_count: u32,
    k: usize,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, RoaringBitmap>) {
    let has_freq_pairs = !freq_pairs.is_empty();

    if anchor_keys.len() > 500 {
        let chunk_size = std::cmp::max(1, anchor_keys.len() / rayon::current_num_threads());
        let chunks: Vec<&[Vec<u32>]> = anchor_keys.chunks(chunk_size).collect();

        let partial_results: Vec<(HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, RoaringBitmap>)> =
            chunks
                .par_iter()
                .map(|chunk| {
                    process_anchors_v2_memo(
                        chunk,
                        lower_to_txns_prev,
                        item_roar,
                        txn_to_items,
                        freq_pairs,
                        n_items,
                        min_count,
                        k,
                        has_freq_pairs,
                    )
                })
                .collect();

        let mut merged_counts: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut merged_lower: HashMap<Vec<u32>, RoaringBitmap> = HashMap::new();
        for (partial_counts, partial_lower) in partial_results {
            for (key, count) in partial_counts {
                merged_counts.entry(key).or_insert(count);
            }
            for (key, tids) in partial_lower {
                merged_lower.entry(key).or_insert(tids);
            }
        }
        (merged_counts, merged_lower)
    } else {
        process_anchors_v2_memo(
            anchor_keys,
            lower_to_txns_prev,
            item_roar,
            txn_to_items,
            freq_pairs,
            n_items,
            min_count,
            k,
            has_freq_pairs,
        )
    }
}

/// Phase-2 entry point. Takes the previous level's TID-list map as input;
/// returns this level's counts AND the next level's TID-list map (so the
/// caller carries it across the BFS level loop).
pub fn count_k_itemsets_v2_memo(
    idx: &InvertedIndex,
    lower_to_txns_prev: &HashMap<Vec<u32>, RoaringBitmap>,
    item_roar: &ItemTxnsRoaring,
    freq_pairs: &HashSet<(u32, u32)>,
    k: usize,
    n_items: usize,
    min_count: u32,
    item_weights: &[f64],
    max_items_per_txn: Option<usize>,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, RoaringBitmap>) {
    let lower_k = k - 1;
    let _ = lower_k;  // documentation marker: anchors are (k-1)-itemsets

    // Frequent items = items appearing in some L_{k-1} key.
    let freq_items: HashSet<u32> = lower_to_txns_prev.keys()
        .flat_map(|s| s.iter().copied())
        .collect();

    // Per-txn item filtering (downward-closure pre-prune) — same as v0/v1.
    let mut txn_to_items_filtered: HashMap<u32, Vec<u32>> =
        HashMap::with_capacity(idx.n_txns as usize);
    for txn_code in 0..idx.n_txns {
        let items = idx.txn_items(txn_code);
        let filtered: Vec<u32> = items.iter()
            .copied()
            .filter(|item| freq_items.contains(item))
            .collect();
        if filtered.len() >= k {
            txn_to_items_filtered.insert(txn_code, filtered);
        }
    }

    if let Some(max_items) = max_items_per_txn {
        for items in txn_to_items_filtered.values_mut() {
            if items.len() > max_items {
                items.sort_by(|a, b| {
                    let wa = item_weights.get(*a as usize).copied().unwrap_or(0.0);
                    let wb = item_weights.get(*b as usize).copied().unwrap_or(0.0);
                    wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
                });
                items.truncate(max_items);
                items.sort_unstable();
            }
        }
    }

    let anchor_keys: Vec<Vec<u32>> = lower_to_txns_prev.keys().cloned().collect();

    run_anchor_extend_v2_memo(
        &anchor_keys,
        lower_to_txns_prev,
        item_roar,
        &txn_to_items_filtered,
        freq_pairs,
        n_items,
        min_count,
        k,
    )
}

/// Build the initial `lower_to_txns_2` map for the BFS k=2→k=3 transition.
/// Each frequent pair {a,b}'s TID-list is item_txns(a) ∩ item_txns(b).
/// Built in parallel via Rayon since pairs are independent.
pub fn build_pair_tid_map(
    freq_pairs: &HashSet<(u32, u32)>,
    item_roar: &ItemTxnsRoaring,
) -> HashMap<Vec<u32>, RoaringBitmap> {
    let pairs_vec: Vec<&(u32, u32)> = freq_pairs.iter().collect();
    let entries: Vec<(Vec<u32>, RoaringBitmap)> = pairs_vec
        .par_iter()
        .map(|&&(a, b)| {
            let bm = item_roar.get(a) & item_roar.get(b);
            // Canonical key form: sorted Vec<u32> (matches lower_sets keys
            // that the BFS level loop produces from filtered counts).
            let key = if a < b { vec![a, b] } else { vec![b, a] };
            (key, bm)
        })
        .collect();
    let mut map: HashMap<Vec<u32>, RoaringBitmap> = HashMap::with_capacity(entries.len());
    for (k, v) in entries {
        map.insert(k, v);
    }
    map
}

// ============================================================================
// v3_adaptive — phase D ablation: same algorithm as v2_memo, but TID-lists
// use the adaptive `TidList` enum (Sparse Vec<u32> for short, Dense
// RoaringBitmap for long). Fixes the small-list iteration / memory regression
// that v1 / v2 paid by always using RoaringBitmap.
//
// Algorithmic identity (preserved): identical to v2_memo. Only the storage
// container changes.
// ============================================================================

fn process_anchors_v3_adaptive(
    anchors: &[Vec<u32>],
    lower_to_txns_prev: &HashMap<Vec<u32>, TidList>,
    item_roar: &common::ItemTxnsRoaring,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    min_count: u32,
    _k: usize,
    has_freq_pairs: bool,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>) {
    let mut result: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut new_lower: HashMap<Vec<u32>, TidList> = HashMap::new();
    let mut counts = vec![0u64; n_items];
    let mut dirty: Vec<u32> = Vec::with_capacity(256);

    for anchor in anchors {
        if let Some(parent_tids) = lower_to_txns_prev.get(anchor) {
            // Counter+chain inner loop — TidIter dispatches on enum variant
            // without the Box<dyn> tax.
            for txn in parent_tids.iter() {
                if let Some(items) = txn_to_items.get(&txn) {
                    for &item in items {
                        let idx = item as usize;
                        if counts[idx] == 0 {
                            dirty.push(item);
                        }
                        counts[idx] += 1;
                    }
                }
            }

            for &item in &dirty {
                let idx = item as usize;
                let count = counts[idx];
                counts[idx] = 0;

                if anchor.contains(&item) {
                    continue;
                }
                if (count as u32) < min_count {
                    continue;
                }

                let mut kset = anchor.clone();
                kset.push(item);
                kset.sort_unstable();

                if result.contains_key(&kset) {
                    continue;
                }

                if has_freq_pairs && !apriori_check(anchor, item, freq_pairs) {
                    continue;
                }

                // Adaptive intersection: dispatches on Sparse vs Dense
                // parent. Result is auto-classified by size.
                let new_tids = parent_tids.intersect_with_bitmap(item_roar.get(item));
                result.insert(kset.clone(), count as u32);
                new_lower.insert(kset, new_tids);
            }
            dirty.clear();
        }
    }

    (result, new_lower)
}

fn run_anchor_extend_v3_adaptive(
    anchor_keys: &[Vec<u32>],
    lower_to_txns_prev: &HashMap<Vec<u32>, TidList>,
    item_roar: &common::ItemTxnsRoaring,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    min_count: u32,
    k: usize,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>) {
    let has_freq_pairs = !freq_pairs.is_empty();

    if anchor_keys.len() > 500 {
        let chunk_size = std::cmp::max(1, anchor_keys.len() / rayon::current_num_threads());
        let chunks: Vec<&[Vec<u32>]> = anchor_keys.chunks(chunk_size).collect();

        let partial_results: Vec<(HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>)> =
            chunks
                .par_iter()
                .map(|chunk| {
                    process_anchors_v3_adaptive(
                        chunk,
                        lower_to_txns_prev,
                        item_roar,
                        txn_to_items,
                        freq_pairs,
                        n_items,
                        min_count,
                        k,
                        has_freq_pairs,
                    )
                })
                .collect();

        let mut merged_counts: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut merged_lower: HashMap<Vec<u32>, TidList> = HashMap::new();
        for (partial_counts, partial_lower) in partial_results {
            for (key, count) in partial_counts {
                merged_counts.entry(key).or_insert(count);
            }
            for (key, tids) in partial_lower {
                merged_lower.entry(key).or_insert(tids);
            }
        }
        (merged_counts, merged_lower)
    } else {
        process_anchors_v3_adaptive(
            anchor_keys,
            lower_to_txns_prev,
            item_roar,
            txn_to_items,
            freq_pairs,
            n_items,
            min_count,
            k,
            has_freq_pairs,
        )
    }
}

pub fn count_k_itemsets_v3_adaptive(
    idx: &InvertedIndex,
    lower_to_txns_prev: &HashMap<Vec<u32>, TidList>,
    item_roar: &common::ItemTxnsRoaring,
    freq_pairs: &HashSet<(u32, u32)>,
    k: usize,
    n_items: usize,
    min_count: u32,
    item_weights: &[f64],
    max_items_per_txn: Option<usize>,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>) {
    let freq_items: HashSet<u32> = lower_to_txns_prev.keys()
        .flat_map(|s| s.iter().copied())
        .collect();

    let mut txn_to_items_filtered: HashMap<u32, Vec<u32>> =
        HashMap::with_capacity(idx.n_txns as usize);
    for txn_code in 0..idx.n_txns {
        let items = idx.txn_items(txn_code);
        let filtered: Vec<u32> = items.iter()
            .copied()
            .filter(|item| freq_items.contains(item))
            .collect();
        if filtered.len() >= k {
            txn_to_items_filtered.insert(txn_code, filtered);
        }
    }

    if let Some(max_items) = max_items_per_txn {
        for items in txn_to_items_filtered.values_mut() {
            if items.len() > max_items {
                items.sort_by(|a, b| {
                    let wa = item_weights.get(*a as usize).copied().unwrap_or(0.0);
                    let wb = item_weights.get(*b as usize).copied().unwrap_or(0.0);
                    wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
                });
                items.truncate(max_items);
                items.sort_unstable();
            }
        }
    }

    let anchor_keys: Vec<Vec<u32>> = lower_to_txns_prev.keys().cloned().collect();

    run_anchor_extend_v3_adaptive(
        &anchor_keys,
        lower_to_txns_prev,
        item_roar,
        &txn_to_items_filtered,
        freq_pairs,
        n_items,
        min_count,
        k,
    )
}

/// Adaptive variant of `build_pair_tid_map`. Each pair's TID-list is
/// computed as `item_roar(a) & item_roar(b)` then auto-classified into
/// Sparse / Dense based on size.
pub fn build_pair_tid_map_adaptive(
    freq_pairs: &HashSet<(u32, u32)>,
    item_roar: &common::ItemTxnsRoaring,
) -> HashMap<Vec<u32>, TidList> {
    let pairs_vec: Vec<&(u32, u32)> = freq_pairs.iter().collect();
    let entries: Vec<(Vec<u32>, TidList)> = pairs_vec
        .par_iter()
        .map(|&&(a, b)| {
            let bm = item_roar.get(a) & item_roar.get(b);
            let key = if a < b { vec![a, b] } else { vec![b, a] };
            (key, TidList::from_roaring(bm))
        })
        .collect();
    let mut map: HashMap<Vec<u32>, TidList> = HashMap::with_capacity(entries.len());
    for (k, v) in entries {
        map.insert(k, v);
    }
    map
}

// ============================================================================
// v6_gating — per-anchor candidate set built from pair_neighbors.
//
// Idea: for each item i, precompute pair_neighbors[i] = sorted list of items j
// such that (i, j) ∈ freq_pairs. Per anchor, the candidate-extension set is
// the intersection of pair_neighbors[a] for each a in anchor. The inner loop
// then gates each (txn, item) pair on a single HashSet membership test
// against the candidate set, replacing both anchor.contains() and the
// per-pair apriori_check.
//
// Difference from v5: per-pair apriori cost is paid ONCE per anchor (during
// candidate-set construction) instead of once per (txn, item) pair.
// ============================================================================

/// Build per-item neighbour lists from freq_pairs. neighbors[i] is the
/// sorted Vec<u32> of items j such that (i, j) — order-independent —
/// appears in freq_pairs. By construction i ∉ neighbors[i].
pub fn build_pair_neighbors(
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: u32,
) -> Vec<Vec<u32>> {
    let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); n_items as usize];
    for &(a, b) in freq_pairs {
        neighbors[a as usize].push(b);
        neighbors[b as usize].push(a);
    }
    for v in &mut neighbors {
        v.sort_unstable();
        // freq_pairs is a HashSet of unordered pairs (a, b) with a != b;
        // build never inserts duplicates, so no dedup needed.
    }
    neighbors
}

/// Two-pointer intersection of two sorted u32 slices.
#[inline]
fn intersect_sorted_u32(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Build the candidate-extension set for a given anchor by intersecting
/// pair_neighbors entries for every item in the anchor. Anchor items are
/// excluded by construction (i ∉ pair_neighbors[i]).
fn build_anchor_candidates(anchor: &[u32], pair_neighbors: &[Vec<u32>]) -> Vec<u32> {
    if anchor.is_empty() {
        return Vec::new();
    }
    let mut cands = pair_neighbors[anchor[0] as usize].clone();
    for &i in &anchor[1..] {
        if cands.is_empty() {
            return cands;
        }
        cands = intersect_sorted_u32(&cands, &pair_neighbors[i as usize]);
    }
    cands
}

fn process_anchors_v6_gating(
    anchors: &[Vec<u32>],
    lower_to_txns_prev: &HashMap<Vec<u32>, TidList>,
    item_roar: &common::ItemTxnsRoaring,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    pair_neighbors: &[Vec<u32>],
    n_items: usize,
    min_count: u32,
    _k: usize,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>) {
    let mut result: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut new_lower: HashMap<Vec<u32>, TidList> = HashMap::new();
    let mut counts = vec![0u64; n_items];
    let mut dirty: Vec<u32> = Vec::with_capacity(256);

    for anchor in anchors {
        // Precompute the candidate-extension set for this anchor.
        let cands_vec = build_anchor_candidates(anchor, pair_neighbors);
        if cands_vec.is_empty() {
            continue;
        }
        // HashSet for O(1) membership test in the hot path. Built once per anchor.
        let candidates: HashSet<u32> = cands_vec.into_iter().collect();

        if let Some(parent_tids) = lower_to_txns_prev.get(anchor) {
            for txn in parent_tids.iter() {
                if let Some(items) = txn_to_items.get(&txn) {
                    for &item in items {
                        if !candidates.contains(&item) {
                            continue;
                        }
                        let idx = item as usize;
                        if counts[idx] == 0 {
                            dirty.push(item);
                        }
                        counts[idx] += 1;
                    }
                }
            }

            // No need for anchor.contains() or apriori_check here — both
            // are subsumed by the candidate-set membership test above.
            for &item in &dirty {
                let idx = item as usize;
                let count = counts[idx];
                counts[idx] = 0;

                if (count as u32) < min_count {
                    continue;
                }

                let mut kset = anchor.clone();
                kset.push(item);
                kset.sort_unstable();

                if result.contains_key(&kset) {
                    continue;
                }

                let new_tids = parent_tids.intersect_with_bitmap(item_roar.get(item));
                result.insert(kset.clone(), count as u32);
                new_lower.insert(kset, new_tids);
            }
            dirty.clear();
        }
    }

    (result, new_lower)
}

fn run_anchor_extend_v6_gating(
    anchor_keys: &[Vec<u32>],
    lower_to_txns_prev: &HashMap<Vec<u32>, TidList>,
    item_roar: &common::ItemTxnsRoaring,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    pair_neighbors: &[Vec<u32>],
    n_items: usize,
    min_count: u32,
    k: usize,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>) {
    if anchor_keys.len() > 500 {
        let chunk_size = std::cmp::max(1, anchor_keys.len() / rayon::current_num_threads());
        let chunks: Vec<&[Vec<u32>]> = anchor_keys.chunks(chunk_size).collect();

        let partial_results: Vec<(HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>)> =
            chunks
                .par_iter()
                .map(|chunk| {
                    process_anchors_v6_gating(
                        chunk,
                        lower_to_txns_prev,
                        item_roar,
                        txn_to_items,
                        pair_neighbors,
                        n_items,
                        min_count,
                        k,
                    )
                })
                .collect();

        let mut merged_counts: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut merged_lower: HashMap<Vec<u32>, TidList> = HashMap::new();
        for (partial_counts, partial_lower) in partial_results {
            for (key, count) in partial_counts {
                merged_counts.entry(key).or_insert(count);
            }
            for (key, tids) in partial_lower {
                merged_lower.entry(key).or_insert(tids);
            }
        }
        (merged_counts, merged_lower)
    } else {
        process_anchors_v6_gating(
            anchor_keys,
            lower_to_txns_prev,
            item_roar,
            txn_to_items,
            pair_neighbors,
            n_items,
            min_count,
            k,
        )
    }
}

pub fn count_k_itemsets_v6_gating(
    idx: &InvertedIndex,
    lower_to_txns_prev: &HashMap<Vec<u32>, TidList>,
    item_roar: &common::ItemTxnsRoaring,
    pair_neighbors: &[Vec<u32>],
    k: usize,
    n_items: usize,
    min_count: u32,
    item_weights: &[f64],
    max_items_per_txn: Option<usize>,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>) {
    let freq_items: HashSet<u32> = lower_to_txns_prev.keys()
        .flat_map(|s| s.iter().copied())
        .collect();

    let mut txn_to_items_filtered: HashMap<u32, Vec<u32>> =
        HashMap::with_capacity(idx.n_txns as usize);
    for txn_code in 0..idx.n_txns {
        let items = idx.txn_items(txn_code);
        let filtered: Vec<u32> = items.iter()
            .copied()
            .filter(|item| freq_items.contains(item))
            .collect();
        if filtered.len() >= k {
            txn_to_items_filtered.insert(txn_code, filtered);
        }
    }

    if let Some(max_items) = max_items_per_txn {
        for items in txn_to_items_filtered.values_mut() {
            if items.len() > max_items {
                items.sort_by(|a, b| {
                    let wa = item_weights.get(*a as usize).copied().unwrap_or(0.0);
                    let wb = item_weights.get(*b as usize).copied().unwrap_or(0.0);
                    wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
                });
                items.truncate(max_items);
                items.sort_unstable();
            }
        }
    }

    let anchor_keys: Vec<Vec<u32>> = lower_to_txns_prev.keys().cloned().collect();

    run_anchor_extend_v6_gating(
        &anchor_keys,
        lower_to_txns_prev,
        item_roar,
        &txn_to_items_filtered,
        pair_neighbors,
        n_items,
        min_count,
        k,
    )
}

// ============================================================================
// v5_prefilter — pre-counter apriori downward-closure check.
//
// Same as v3_adaptive, but the apriori_check (and the anchor-membership test)
// is applied BEFORE the counter increment instead of after. This skips the
// counter scatter, dirty-list push, and dirty-list iteration step for any
// item that can't form a frequent k-itemset with the anchor.
//
// Expected wins on wide-sparse data: pre-filter rejects most candidates,
// making the dirty list short and the per-dirty-item intersection step
// (the actual hot path at k>=3) only runs for true survivors.
// Expected regressions on narrow-dense data: most candidates pass apriori,
// so the pre-check is wasted overhead.
// ============================================================================

#[inline]
fn apriori_pre_pair_check(anchor: &[u32], new_item: u32, freq_pairs: &HashSet<(u32, u32)>) -> bool {
    // Identical body to apriori_check; inlined for the hot pre-counter path
    // so the compiler is more likely to keep it tight inside the inner loop.
    for &existing in anchor {
        let pair = if existing < new_item {
            (existing, new_item)
        } else {
            (new_item, existing)
        };
        if !freq_pairs.contains(&pair) {
            return false;
        }
    }
    true
}

fn process_anchors_v5_prefilter(
    anchors: &[Vec<u32>],
    lower_to_txns_prev: &HashMap<Vec<u32>, TidList>,
    item_roar: &common::ItemTxnsRoaring,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    min_count: u32,
    _k: usize,
    has_freq_pairs: bool,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>) {
    let mut result: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut new_lower: HashMap<Vec<u32>, TidList> = HashMap::new();
    let mut counts = vec![0u64; n_items];
    let mut dirty: Vec<u32> = Vec::with_capacity(256);

    for anchor in anchors {
        if let Some(parent_tids) = lower_to_txns_prev.get(anchor) {
            for txn in parent_tids.iter() {
                if let Some(items) = txn_to_items.get(&txn) {
                    for &item in items {
                        // PRE-counter filter: skip items that can't form a
                        // frequent k-itemset with this anchor.
                        if anchor.contains(&item) {
                            continue;
                        }
                        if has_freq_pairs && !apriori_pre_pair_check(anchor, item, freq_pairs) {
                            continue;
                        }
                        let idx = item as usize;
                        if counts[idx] == 0 {
                            dirty.push(item);
                        }
                        counts[idx] += 1;
                    }
                }
            }

            // Dirty iteration — apriori_check is no longer needed (done pre-counter).
            for &item in &dirty {
                let idx = item as usize;
                let count = counts[idx];
                counts[idx] = 0;

                if (count as u32) < min_count {
                    continue;
                }

                let mut kset = anchor.clone();
                kset.push(item);
                kset.sort_unstable();

                if result.contains_key(&kset) {
                    continue;
                }

                let new_tids = parent_tids.intersect_with_bitmap(item_roar.get(item));
                result.insert(kset.clone(), count as u32);
                new_lower.insert(kset, new_tids);
            }
            dirty.clear();
        }
    }

    (result, new_lower)
}

fn run_anchor_extend_v5_prefilter(
    anchor_keys: &[Vec<u32>],
    lower_to_txns_prev: &HashMap<Vec<u32>, TidList>,
    item_roar: &common::ItemTxnsRoaring,
    txn_to_items: &HashMap<u32, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    min_count: u32,
    k: usize,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>) {
    let has_freq_pairs = !freq_pairs.is_empty();

    if anchor_keys.len() > 500 {
        let chunk_size = std::cmp::max(1, anchor_keys.len() / rayon::current_num_threads());
        let chunks: Vec<&[Vec<u32>]> = anchor_keys.chunks(chunk_size).collect();

        let partial_results: Vec<(HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>)> =
            chunks
                .par_iter()
                .map(|chunk| {
                    process_anchors_v5_prefilter(
                        chunk,
                        lower_to_txns_prev,
                        item_roar,
                        txn_to_items,
                        freq_pairs,
                        n_items,
                        min_count,
                        k,
                        has_freq_pairs,
                    )
                })
                .collect();

        let mut merged_counts: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut merged_lower: HashMap<Vec<u32>, TidList> = HashMap::new();
        for (partial_counts, partial_lower) in partial_results {
            for (key, count) in partial_counts {
                merged_counts.entry(key).or_insert(count);
            }
            for (key, tids) in partial_lower {
                merged_lower.entry(key).or_insert(tids);
            }
        }
        (merged_counts, merged_lower)
    } else {
        process_anchors_v5_prefilter(
            anchor_keys,
            lower_to_txns_prev,
            item_roar,
            txn_to_items,
            freq_pairs,
            n_items,
            min_count,
            k,
            has_freq_pairs,
        )
    }
}

pub fn count_k_itemsets_v5_prefilter(
    idx: &InvertedIndex,
    lower_to_txns_prev: &HashMap<Vec<u32>, TidList>,
    item_roar: &common::ItemTxnsRoaring,
    freq_pairs: &HashSet<(u32, u32)>,
    k: usize,
    n_items: usize,
    min_count: u32,
    item_weights: &[f64],
    max_items_per_txn: Option<usize>,
) -> (HashMap<Vec<u32>, u32>, HashMap<Vec<u32>, TidList>) {
    let freq_items: HashSet<u32> = lower_to_txns_prev.keys()
        .flat_map(|s| s.iter().copied())
        .collect();

    let mut txn_to_items_filtered: HashMap<u32, Vec<u32>> =
        HashMap::with_capacity(idx.n_txns as usize);
    for txn_code in 0..idx.n_txns {
        let items = idx.txn_items(txn_code);
        let filtered: Vec<u32> = items.iter()
            .copied()
            .filter(|item| freq_items.contains(item))
            .collect();
        if filtered.len() >= k {
            txn_to_items_filtered.insert(txn_code, filtered);
        }
    }

    if let Some(max_items) = max_items_per_txn {
        for items in txn_to_items_filtered.values_mut() {
            if items.len() > max_items {
                items.sort_by(|a, b| {
                    let wa = item_weights.get(*a as usize).copied().unwrap_or(0.0);
                    let wb = item_weights.get(*b as usize).copied().unwrap_or(0.0);
                    wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
                });
                items.truncate(max_items);
                items.sort_unstable();
            }
        }
    }

    let anchor_keys: Vec<Vec<u32>> = lower_to_txns_prev.keys().cloned().collect();

    run_anchor_extend_v5_prefilter(
        &anchor_keys,
        lower_to_txns_prev,
        item_roar,
        &txn_to_items_filtered,
        freq_pairs,
        n_items,
        min_count,
        k,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combination_iter_pairs() {
        let items = vec![10, 20, 30, 40];
        let result: Vec<Vec<u32>> = CombinationIter::new(&items, 2).collect();
        assert_eq!(result, vec![
            vec![10, 20], vec![10, 30], vec![10, 40],
            vec![20, 30], vec![20, 40], vec![30, 40],
        ]);
    }

    #[test]
    fn test_combination_iter_triplets() {
        let items = vec![1, 2, 3, 4, 5];
        let result: Vec<Vec<u32>> = CombinationIter::new(&items, 3).collect();
        assert_eq!(result.len(), 10);
        assert_eq!(result[0], vec![1, 2, 3]);
        assert_eq!(result[9], vec![3, 4, 5]);
    }

    #[test]
    fn test_combination_iter_r_equals_n() {
        let items = vec![1, 2, 3];
        let result: Vec<Vec<u32>> = CombinationIter::new(&items, 3).collect();
        assert_eq!(result, vec![vec![1, 2, 3]]);
    }

    #[test]
    fn test_combination_iter_r_greater_than_n() {
        let items = vec![1, 2];
        let result: Vec<Vec<u32>> = CombinationIter::new(&items, 5).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_combination_iter_r_zero() {
        let items = vec![1, 2, 3];
        let result: Vec<Vec<u32>> = CombinationIter::new(&items, 0).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_combination_iter_single_element() {
        let items = vec![42];
        let result: Vec<Vec<u32>> = CombinationIter::new(&items, 1).collect();
        assert_eq!(result, vec![vec![42]]);
    }

    #[test]
    fn test_combination_iter_count_c18_3() {
        let items: Vec<u32> = (0..18).collect();
        let count = CombinationIter::new(&items, 3).count();
        assert_eq!(count, 816);
    }

    #[test]
    fn test_combination_iter_count_c18_4() {
        let items: Vec<u32> = (0..18).collect();
        let count = CombinationIter::new(&items, 4).count();
        assert_eq!(count, 3060);
    }
}
