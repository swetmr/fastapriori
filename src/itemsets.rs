use std::collections::{HashMap, HashSet};

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::common;

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
    lower_to_txns: &HashMap<Vec<u32>, Vec<i64>>,
    txn_to_items: &HashMap<i64, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    _k: usize,
    has_freq_pairs: bool,
) -> HashMap<Vec<u32>, u32> {
    let mut result: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut counts = vec![0u32; n_items];

    for anchor in anchors {
        let anchor_set: HashSet<u32> = anchor.iter().copied().collect();

        if let Some(txn_ids) = lower_to_txns.get(anchor) {
            // Count all items across anchor's transactions
            for &txn in txn_ids {
                if let Some(items) = txn_to_items.get(&txn) {
                    for &item in items {
                        counts[item as usize] += 1;
                    }
                }
            }

            // Extract candidates
            for item_idx in 0..n_items {
                let count = counts[item_idx];
                counts[item_idx] = 0; // reset

                if count == 0 {
                    continue;
                }
                let item = item_idx as u32;
                if anchor_set.contains(&item) {
                    continue;
                }

                // Build canonical k-set
                let mut kset = anchor.clone();
                kset.push(item);
                kset.sort_unstable();

                // First-write-wins dedup
                if result.contains_key(&kset) {
                    continue;
                }

                // Apriori pruning
                if has_freq_pairs && !apriori_check(anchor, item, freq_pairs) {
                    continue;
                }

                result.insert(kset, count);
            }
        }
    }

    result
}

/// Run anchor-and-extend counting with optional Rayon parallelism.
fn run_anchor_extend(
    anchor_keys: &[Vec<u32>],
    lower_to_txns: &HashMap<Vec<u32>, Vec<i64>>,
    txn_to_items: &HashMap<i64, Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    n_items: usize,
    k: usize,
) -> HashMap<Vec<u32>, u32> {
    let has_freq_pairs = !freq_pairs.is_empty();

    if anchor_keys.len() > 500 {
        // Parallel: partition anchors across threads
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

        // Merge with first-write-wins
        let mut merged: HashMap<Vec<u32>, u32> = HashMap::new();
        for partial in partial_results {
            for (key, val) in partial {
                merged.entry(key).or_insert(val);
            }
        }
        merged
    } else {
        // Serial
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

/// Internal function: count k-itemsets from in-memory data structures.
/// Used by pipeline and by rust_compute_itemsets.
pub fn count_k_itemsets_internal(
    txn_to_items: &HashMap<i64, Vec<u32>>,
    freq_lower_sets: &HashSet<Vec<u32>>,
    freq_pairs: &HashSet<(u32, u32)>,
    k: usize,
    n_items: usize,
) -> HashMap<Vec<u32>, u32> {
    let lower_k = k - 1;

    // Opt 1: Build set of items appearing in ANY frequent (k-1)-set.
    // Items outside this set cannot contribute to any frequent k-set
    // (downward closure), so filtering them out is exact.
    let freq_items: HashSet<u32> = freq_lower_sets.iter()
        .flat_map(|s| s.iter().copied())
        .collect();

    // Opt 2: Pre-filter txn_to_items to only frequent items.
    // Reused for both lower_to_txns building and anchor-extend counting.
    let txn_to_items_filtered: HashMap<i64, Vec<u32>> = txn_to_items.iter()
        .filter_map(|(&txn_id, items)| {
            let filtered: Vec<u32> = items.iter()
                .filter(|&&item| freq_items.contains(&item))
                .copied()
                .collect();
            if filtered.len() >= k {
                Some((txn_id, filtered))
            } else {
                None
            }
        })
        .collect();

    // Opt 3: Build lower_to_txns with Rayon parallelism for large datasets.
    let lower_to_txns = if txn_to_items_filtered.len() > 10_000 {
        // Parallel: each thread builds a partial map, then merge
        let txn_vec: Vec<(&i64, &Vec<u32>)> = txn_to_items_filtered.iter().collect();
        let partial_maps: Vec<HashMap<Vec<u32>, Vec<i64>>> = txn_vec
            .par_chunks(std::cmp::max(1, txn_vec.len() / rayon::current_num_threads()))
            .map(|chunk| {
                let mut local: HashMap<Vec<u32>, Vec<i64>> = HashMap::new();
                for &(&txn_id, items) in chunk {
                    for combo in CombinationIter::new(items, lower_k) {
                        if freq_lower_sets.contains(&combo) {
                            local.entry(combo).or_default().push(txn_id);
                        }
                    }
                }
                local
            })
            .collect();

        let mut merged: HashMap<Vec<u32>, Vec<i64>> = HashMap::new();
        for partial in partial_maps {
            for (key, mut txns) in partial {
                merged.entry(key).or_insert_with(Vec::new).append(&mut txns);
            }
        }
        merged
    } else {
        // Serial for small datasets (avoid Rayon overhead)
        let mut lower_to_txns: HashMap<Vec<u32>, Vec<i64>> = HashMap::new();
        for (&txn_id, items) in &txn_to_items_filtered {
            for combo in CombinationIter::new(items, lower_k) {
                if freq_lower_sets.contains(&combo) {
                    lower_to_txns.entry(combo).or_default().push(txn_id);
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
pub fn rust_compute_itemsets<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    freq_lower: PyReadonlyArray2<'py, i32>,
    k: usize,
    n_items: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;

    let (txn_to_items, _item_to_txns, _item_counts) =
        common::build_inverted_index(txn_slice, item_slice);

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

    // Use internal function
    let combined = count_k_itemsets_internal(
        &txn_to_items,
        &freq_lower_sets,
        &freq_pairs,
        k,
        n_items as usize,
    );

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
        assert_eq!(result.len(), 10); // C(5,3) = 10
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
        // The problematic case: C(18,3) = 816
        let items: Vec<u32> = (0..18).collect();
        let count = CombinationIter::new(&items, 3).count();
        assert_eq!(count, 816);
    }

    #[test]
    fn test_combination_iter_count_c18_4() {
        // k=5 case: C(18,4) = 3060
        let items: Vec<u32> = (0..18).collect();
        let count = CombinationIter::new(&items, 4).count();
        assert_eq!(count, 3060);
    }
}
