//! Classic Apriori algorithm — Rust port of efficient-apriori
//! (https://github.com/tommyod/Efficient-Apriori).
//!
//! Same algorithm, same optimizations (inverted index for support counting,
//! short-circuit intersection, prefix-based join, downward-closure pruning).
//! Compiled Rust instead of Python for a fair like-to-like comparison that
//! isolates the algorithmic difference from the language difference.

use std::collections::{HashMap, HashSet};

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::common;

// ─────────────────────────────────────────────────────────────────────────────
// TransactionIndex — equivalent of EA's TransactionManager
// ─────────────────────────────────────────────────────────────────────────────

struct TransactionIndex {
    /// item → sorted transaction IDs that contain it
    item_to_txns: HashMap<u32, Vec<i64>>,
    /// item → count of transactions containing it (== item_to_txns[item].len())
    item_counts: HashMap<u32, u32>,
    n_transactions: u32,
}

impl TransactionIndex {
    fn build(txn_ids: &[i64], item_ids: &[i32]) -> Self {
        Self::build_with_cap(txn_ids, item_ids, None, None)
    }

    /// Build the inverted index with optional per-transaction capping.
    ///
    /// When `max_items_per_txn` and `item_weights` are both provided, each
    /// transaction with more than N distinct items is truncated to its top-N
    /// items by weight (descending).  Matches the semantics of the fast
    /// backend's `max_items_per_txn` optimization: counts derived from a
    /// capped index are lower bounds on the true counts.
    fn build_with_cap(
        txn_ids: &[i64],
        item_ids: &[i32],
        item_weights: Option<&[f64]>,
        max_items_per_txn: Option<usize>,
    ) -> Self {
        // Step 1: group (deduped) items by transaction
        let mut txn_to_items: HashMap<i64, Vec<u32>> = HashMap::new();
        for i in 0..txn_ids.len() {
            let txn = txn_ids[i];
            let item = item_ids[i] as u32;
            txn_to_items.entry(txn).or_default().push(item);
        }
        for items in txn_to_items.values_mut() {
            items.sort_unstable();
            items.dedup();
        }

        // Step 2: optional top-N cap by weight (matches fast backend Opt 2b)
        if let (Some(max_items), Some(weights)) = (max_items_per_txn, item_weights) {
            for items in txn_to_items.values_mut() {
                if items.len() > max_items {
                    items.sort_by(|a, b| {
                        let wa = weights.get(*a as usize).copied().unwrap_or(0.0);
                        let wb = weights.get(*b as usize).copied().unwrap_or(0.0);
                        wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    items.truncate(max_items);
                    items.sort_unstable(); // restore ascending order
                }
            }
        }

        // Step 3: invert → item_to_txns
        let n_transactions = txn_to_items.len() as u32;
        let mut item_to_txns: HashMap<u32, Vec<i64>> = HashMap::new();
        for (&txn, items) in &txn_to_items {
            for &item in items {
                item_to_txns.entry(item).or_default().push(txn);
            }
        }
        for txns in item_to_txns.values_mut() {
            txns.sort_unstable();
            // items were already deduped per-txn, so no dup entries per item
        }

        let item_counts: HashMap<u32, u32> = item_to_txns
            .iter()
            .map(|(&item, txns)| (item, txns.len() as u32))
            .collect();

        Self {
            item_to_txns,
            item_counts,
            n_transactions,
        }
    }

    /// Short-circuit support counting — port of EA's transaction_indices_sc().
    ///
    /// Sorts items by frequency (ascending = rarest first for tightest pruning),
    /// intersects transaction sets, bails early if count drops below min_count.
    /// Returns Some(count) if itemset meets threshold, None otherwise.
    fn count_support_sc(&self, itemset: &[u32], min_count: u32) -> Option<u32> {
        if itemset.is_empty() {
            return None;
        }

        // Sort items by frequency (rarest first) for best pruning
        let mut items_by_freq: Vec<u32> = itemset.to_vec();
        items_by_freq.sort_by_key(|&item| {
            self.item_counts.get(&item).copied().unwrap_or(0)
        });

        // Start with transaction set of rarest item
        let first_item = items_by_freq[0];
        let mut current_txns: Vec<i64> = match self.item_to_txns.get(&first_item) {
            Some(txns) => txns.clone(),
            None => return None,
        };

        if (current_txns.len() as u32) < min_count {
            return None; // Short-circuit
        }

        // Intersect with remaining items' transaction sets
        for &item in &items_by_freq[1..] {
            let item_txns = match self.item_to_txns.get(&item) {
                Some(txns) => txns,
                None => return None,
            };

            // Sorted intersection
            current_txns = sorted_intersect(&current_txns, item_txns);

            if (current_txns.len() as u32) < min_count {
                return None; // Short-circuit
            }
        }

        Some(current_txns.len() as u32)
    }
}

/// Sorted intersection of two sorted slices.
fn sorted_intersect(a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut result = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Candidate generation — port of EA's join_step + prune_step
// ─────────────────────────────────────────────────────────────────────────────

/// Prefix-based join step — port of EA's join_step().
///
/// Given sorted frequent (k-1)-itemsets, generate k-candidates by joining
/// pairs that share the first (k-2) items.
fn join_step(freq_itemsets: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut candidates = Vec::new();
    let mut i = 0;

    while i < freq_itemsets.len() {
        // Split into prefix and last item
        let itemset = &freq_itemsets[i];
        let k_minus_1 = itemset.len();
        if k_minus_1 == 0 {
            i += 1;
            continue;
        }
        let prefix = &itemset[..k_minus_1 - 1];
        let last_item = itemset[k_minus_1 - 1];

        // Collect all items sharing the same prefix
        let mut tail_items = vec![last_item];
        let mut skip = 1;

        for j in (i + 1)..freq_itemsets.len() {
            let candidate = &freq_itemsets[j];
            let cand_prefix = &candidate[..k_minus_1 - 1];
            if cand_prefix == prefix {
                tail_items.push(candidate[k_minus_1 - 1]);
                skip += 1;
            } else {
                break;
            }
        }

        // Generate 2-combinations of tail items → k-candidates
        for a_idx in 0..tail_items.len() {
            for b_idx in (a_idx + 1)..tail_items.len() {
                let mut candidate = prefix.to_vec();
                candidate.push(tail_items[a_idx]);
                candidate.push(tail_items[b_idx]);
                candidates.push(candidate);
            }
        }

        i += skip;
    }

    candidates
}

/// Downward-closure prune step — port of EA's prune_step().
///
/// Remove candidates where any (k-1)-subset is not in the frequent set.
/// Matches EA's implementation: checks subsets formed by removing each item
/// at positions 0..len-2 (the last two items came from the join step and
/// their subsets are guaranteed frequent by the join property).
fn prune_step(freq_lower: &HashSet<Vec<u32>>, candidates: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    candidates
        .into_iter()
        .filter(|candidate| {
            let k = candidate.len();
            // Check (k-1)-subsets by removing item at each position except the last two
            // (same as EA: for i in range(len(candidate) - 2))
            for i in 0..(k.saturating_sub(2)) {
                let mut subset: Vec<u32> = Vec::with_capacity(k - 1);
                for (j, &item) in candidate.iter().enumerate() {
                    if j != i {
                        subset.push(item);
                    }
                }
                if !freq_lower.contains(&subset) {
                    return false;
                }
            }
            true
        })
        .collect()
}

/// Candidate generation: join + prune — port of EA's apriori_gen().
fn apriori_gen(freq_itemsets: &[Vec<u32>], freq_lower_set: &HashSet<Vec<u32>>) -> Vec<Vec<u32>> {
    let joined = join_step(freq_itemsets);
    prune_step(freq_lower_set, joined)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Apriori loop — port of EA's itemsets_from_transactions()
// ─────────────────────────────────────────────────────────────────────────────

/// Count candidates in parallel using Rayon when candidate count > threshold.
fn count_candidates_parallel(
    index: &TransactionIndex,
    candidates: &[Vec<u32>],
    min_count: u32,
) -> HashMap<Vec<u32>, u32> {
    if candidates.len() > 500 {
        // Parallel
        candidates
            .par_iter()
            .filter_map(|candidate| {
                index
                    .count_support_sc(candidate, min_count)
                    .map(|count| (candidate.clone(), count))
            })
            .collect()
    } else {
        // Serial
        candidates
            .iter()
            .filter_map(|candidate| {
                index
                    .count_support_sc(candidate, min_count)
                    .map(|count| (candidate.clone(), count))
            })
            .collect()
    }
}

/// Run the full Apriori algorithm from k=1 to k_max.
///
/// Returns frequent itemsets at each level: level -> (itemset -> count).
fn apriori_itemsets(
    index: &mut TransactionIndex,
    min_support: f64,
    k_max: usize,
) -> HashMap<usize, HashMap<Vec<u32>, u32>> {
    let n_txn = index.n_transactions as f64;
    let min_count = if min_support > 0.0 {
        (min_support * n_txn).ceil() as u32
    } else {
        1
    };

    let mut all_levels: HashMap<usize, HashMap<Vec<u32>, u32>> = HashMap::new();

    // Level 1: frequent 1-itemsets
    let freq_1: HashMap<Vec<u32>, u32> = index
        .item_counts
        .iter()
        .filter(|(_, &count)| count >= min_count)
        .map(|(&item, &count)| (vec![item], count))
        .collect();

    if freq_1.is_empty() {
        return all_levels;
    }

    // Opt 4: Remove infrequent items from the index to reduce memory
    // and speed up subsequent intersection operations.
    let freq_item_set: HashSet<u32> = freq_1.keys().map(|v| v[0]).collect();
    index.item_to_txns.retain(|item, _| freq_item_set.contains(item));
    index.item_counts.retain(|item, _| freq_item_set.contains(item));

    all_levels.insert(1, freq_1);

    // Levels 2..k_max
    for k in 2..=k_max {
        let prev_level = match all_levels.get(&(k - 1)) {
            Some(level) if !level.is_empty() => level,
            _ => break,
        };

        // Sort itemsets lexicographically for join step
        let mut sorted_itemsets: Vec<Vec<u32>> = prev_level.keys().cloned().collect();
        sorted_itemsets.sort();

        let freq_lower_set: HashSet<Vec<u32>> = prev_level.keys().cloned().collect();

        // Generate candidates
        let candidates = apriori_gen(&sorted_itemsets, &freq_lower_set);

        if candidates.is_empty() {
            break;
        }

        // Count candidates (with short-circuit + optional parallelism)
        let freq_k = count_candidates_parallel(index, &candidates, min_count);

        if freq_k.is_empty() {
            break;
        }

        all_levels.insert(k, freq_k);
    }

    all_levels
}

// ─────────────────────────────────────────────────────────────────────────────
// PyO3 exports
// ─────────────────────────────────────────────────────────────────────────────

/// Classic Apriori k=2 pair computation with 7 metrics.
///
/// Returns the same 10-column schema as rust_compute_pairs, but only
/// frequent pairs (support >= min_support). Classic Apriori requires
/// min_support upfront for candidate pruning.
#[pyfunction]
pub fn rust_classic_compute_pairs<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    n_items: u32,
    n_transactions: u32,
    min_support: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;

    // Build the transaction index (EA's TransactionManager)
    let mut index = TransactionIndex::build(txn_slice, item_slice);

    // Also build inverted index for metric computation
    let (_, _, item_counts_map) =
        common::build_inverted_index(txn_slice, item_slice);

    let n_txn_f64 = n_transactions as f64;

    // Pre-compute item supports
    let mut item_support = vec![0.0_f64; n_items as usize];
    for (&item, &count) in &item_counts_map {
        item_support[item as usize] = count as f64 / n_txn_f64;
    }

    // Run classic Apriori up to k=2
    let levels = apriori_itemsets(&mut index, min_support, 2);

    // Extract level 2 (frequent pairs)
    let freq_pairs = match levels.get(&2) {
        Some(pairs) => pairs,
        None => {
            // No frequent pairs — return empty arrays
            let dict = PyDict::new(py);
            dict.set_item("item_A", PyArray1::<i32>::zeros(py, 0, false))?;
            dict.set_item("item_B", PyArray1::<i32>::zeros(py, 0, false))?;
            dict.set_item("instances", PyArray1::<i64>::zeros(py, 0, false))?;
            dict.set_item("support", PyArray1::<f64>::zeros(py, 0, false))?;
            dict.set_item("confidence", PyArray1::<f64>::zeros(py, 0, false))?;
            dict.set_item("lift", PyArray1::<f64>::zeros(py, 0, false))?;
            dict.set_item("conviction", PyArray1::<f64>::zeros(py, 0, false))?;
            dict.set_item("leverage", PyArray1::<f64>::zeros(py, 0, false))?;
            dict.set_item("cosine", PyArray1::<f64>::zeros(py, 0, false))?;
            dict.set_item("jaccard", PyArray1::<f64>::zeros(py, 0, false))?;
            return Ok(dict);
        }
    };

    // Emit directed pairs (A->B and B->A) with metrics, matching fast path output
    let n_pairs = freq_pairs.len() * 2;
    let mut out_item_a = Vec::with_capacity(n_pairs);
    let mut out_item_b = Vec::with_capacity(n_pairs);
    let mut out_instances = Vec::with_capacity(n_pairs);
    let mut out_support = Vec::with_capacity(n_pairs);
    let mut out_confidence = Vec::with_capacity(n_pairs);
    let mut out_lift = Vec::with_capacity(n_pairs);
    let mut out_conviction = Vec::with_capacity(n_pairs);
    let mut out_leverage = Vec::with_capacity(n_pairs);
    let mut out_cosine = Vec::with_capacity(n_pairs);
    let mut out_jaccard = Vec::with_capacity(n_pairs);

    for (pair, &co_count) in freq_pairs {
        let item_a = pair[0];
        let item_b = pair[1];
        let count_a = item_counts_map.get(&item_a).copied().unwrap_or(1);
        let count_b = item_counts_map.get(&item_b).copied().unwrap_or(1);

        let instances = co_count as f64;
        let sup_a = item_support[item_a as usize];
        let sup_b = item_support[item_b as usize];
        let support = instances / n_txn_f64;
        let cosine = support / (sup_a * sup_b).sqrt();
        let jaccard = support / (sup_a + sup_b - support);
        let leverage = support - sup_a * sup_b;

        // Direction A -> B
        let conf_ab = instances / count_a as f64;
        let lift_ab = conf_ab / (sup_b + 1e-10);
        let conviction_ab = (1.0 - sup_b) / (1.0 - conf_ab + 1e-10);

        out_item_a.push(item_a as i32);
        out_item_b.push(item_b as i32);
        out_instances.push(co_count as i64);
        out_support.push(support);
        out_confidence.push(conf_ab);
        out_lift.push(lift_ab);
        out_conviction.push(conviction_ab);
        out_leverage.push(leverage);
        out_cosine.push(cosine);
        out_jaccard.push(jaccard);

        // Direction B -> A
        let conf_ba = instances / count_b as f64;
        let lift_ba = conf_ba / (sup_a + 1e-10);
        let conviction_ba = (1.0 - sup_a) / (1.0 - conf_ba + 1e-10);

        out_item_a.push(item_b as i32);
        out_item_b.push(item_a as i32);
        out_instances.push(co_count as i64);
        out_support.push(support);
        out_confidence.push(conf_ba);
        out_lift.push(lift_ba);
        out_conviction.push(conviction_ba);
        out_leverage.push(leverage);
        out_cosine.push(cosine);
        out_jaccard.push(jaccard);
    }

    let dict = PyDict::new(py);
    dict.set_item("item_A", PyArray1::from_vec(py, out_item_a))?;
    dict.set_item("item_B", PyArray1::from_vec(py, out_item_b))?;
    dict.set_item("instances", PyArray1::from_vec(py, out_instances))?;
    dict.set_item("support", PyArray1::from_vec(py, out_support))?;
    dict.set_item("confidence", PyArray1::from_vec(py, out_confidence))?;
    dict.set_item("lift", PyArray1::from_vec(py, out_lift))?;
    dict.set_item("conviction", PyArray1::from_vec(py, out_conviction))?;
    dict.set_item("leverage", PyArray1::from_vec(py, out_leverage))?;
    dict.set_item("cosine", PyArray1::from_vec(py, out_cosine))?;
    dict.set_item("jaccard", PyArray1::from_vec(py, out_jaccard))?;

    Ok(dict)
}

/// Classic Apriori full pipeline k=1..k_max.
///
/// Returns the same schema as rust_compute_pipeline:
/// - "itemsets": 2D int32 array (n_results, k_max)
/// - "counts": 1D int64 array (n_results,)
/// - "lower_itemsets": 2D int32 array (n_lower, k_max-1)
/// - "lower_counts": 1D int64 array (n_lower,)
/// - "item_counts": 1D int64 array (n_items,) — per-item transaction counts
#[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, k_max, n_items, _n_transactions, min_support, item_weights, max_items_per_txn=None))]
pub fn rust_classic_compute_pipeline<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    k_max: usize,
    n_items: u32,
    _n_transactions: u32,
    min_support: f64,
    item_weights: PyReadonlyArray1<'py, f64>,
    max_items_per_txn: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;
    let weights_slice = item_weights.as_slice()?;

    let mut index = TransactionIndex::build_with_cap(
        txn_slice,
        item_slice,
        Some(weights_slice),
        max_items_per_txn,
    );

    // Run classic Apriori
    let levels = apriori_itemsets(&mut index, min_support, k_max);

    // Extract k_max level
    let empty_map: HashMap<Vec<u32>, u32> = HashMap::new();
    let final_level = levels.get(&k_max).unwrap_or(&empty_map);

    let results: Vec<(&Vec<u32>, &u32)> = final_level.iter().collect();
    let itemsets_vecs: Vec<Vec<i32>> = results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let counts_vec: Vec<i64> = results.iter().map(|(_, &count)| count as i64).collect();

    // Extract (k_max - 1) level
    let lower_level = if k_max >= 2 {
        levels.get(&(k_max - 1)).unwrap_or(&empty_map)
    } else {
        &empty_map
    };

    let lower_results: Vec<(&Vec<u32>, &u32)> = lower_level.iter().collect();
    let lower_itemsets_vecs: Vec<Vec<i32>> = lower_results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let lower_counts_vec: Vec<i64> = lower_results
        .iter()
        .map(|(_, &count)| count as i64)
        .collect();

    // Per-item counts
    let mut item_counts_arr = vec![0i64; n_items as usize];
    for (&item, &count) in &index.item_counts {
        if (item as usize) < item_counts_arr.len() {
            item_counts_arr[item as usize] = count as i64;
        }
    }

    // Build output dict
    let dict = PyDict::new(py);
    dict.set_item("itemsets", PyArray2::from_vec2(py, &itemsets_vecs)?)?;
    dict.set_item("counts", PyArray1::from_vec(py, counts_vec))?;
    dict.set_item(
        "lower_itemsets",
        PyArray2::from_vec2(py, &lower_itemsets_vecs)?,
    )?;
    dict.set_item(
        "lower_counts",
        PyArray1::from_vec(py, lower_counts_vec),
    )?;
    dict.set_item(
        "item_counts",
        PyArray1::from_vec(py, item_counts_arr),
    )?;

    Ok(dict)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_index() -> TransactionIndex {
        // 5 transactions, 5 items:
        // T0: {0, 1, 2}
        // T1: {0, 1, 3}
        // T2: {0, 1, 2, 3}
        // T3: {2, 3, 4}
        // T4: {0, 2, 4}
        let txn_ids: Vec<i64> = vec![
            0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4,
        ];
        let item_ids: Vec<i32> = vec![
            0, 1, 2, 0, 1, 3, 0, 1, 2, 3, 2, 3, 4, 0, 2, 4,
        ];
        TransactionIndex::build(&txn_ids, &item_ids)
    }

    #[test]
    fn test_sorted_intersect() {
        assert_eq!(
            sorted_intersect(&[1, 3, 5, 7], &[2, 3, 5, 6]),
            vec![3, 5]
        );
        assert_eq!(sorted_intersect(&[1, 2, 3], &[4, 5, 6]), Vec::<i64>::new());
        assert_eq!(sorted_intersect(&[1, 2, 3], &[1, 2, 3]), vec![1, 2, 3]);
    }

    #[test]
    fn test_short_circuit_counting() {
        let index = make_test_index();

        // Item 0 appears in T0,T1,T2,T4 → count=4
        assert_eq!(index.count_support_sc(&[0], 1), Some(4));

        // Pair {0,1} appears in T0,T1,T2 → count=3
        assert_eq!(index.count_support_sc(&[0, 1], 1), Some(3));

        // Pair {0,1} with min_count=4 should short-circuit
        assert_eq!(index.count_support_sc(&[0, 1], 4), None);

        // Triplet {0,1,2} appears in T0,T2 → count=2
        assert_eq!(index.count_support_sc(&[0, 1, 2], 1), Some(2));

        // Triplet {0,1,3} appears in T1,T2 → count=2
        assert_eq!(index.count_support_sc(&[0, 1, 3], 1), Some(2));
    }

    #[test]
    fn test_join_step() {
        // k=2 → k=3: join pairs sharing first item
        let itemsets = vec![
            vec![0u32, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 2],
            vec![1, 3],
            vec![2, 3],
        ];
        let mut result = join_step(&itemsets);
        result.sort();
        // Expected: all 3-combinations of {0,1,2,3}
        assert_eq!(
            result,
            vec![
                vec![0, 1, 2],
                vec![0, 1, 3],
                vec![0, 2, 3],
                vec![1, 2, 3],
            ]
        );
    }

    #[test]
    fn test_join_step_k1_to_k2() {
        // k=1 → k=2: join singletons
        let itemsets = vec![vec![0u32], vec![1], vec![2], vec![3]];
        let mut result = join_step(&itemsets);
        result.sort();
        assert_eq!(
            result,
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![1, 2],
                vec![1, 3],
                vec![2, 3],
            ]
        );
    }

    #[test]
    fn test_prune_step() {
        // Suppose only {0,1}, {0,2}, {1,2} are frequent (not {0,3}, {1,3}, {2,3})
        let freq_lower: HashSet<Vec<u32>> = [vec![0, 1], vec![0, 2], vec![1, 2]]
            .into_iter()
            .collect();

        let candidates = vec![
            vec![0u32, 1, 2], // all subsets frequent → keep
            vec![0, 1, 3],    // subset {0,3} not frequent → prune
            vec![0, 2, 3],    // subset {0,3} not frequent → prune
            vec![1, 2, 3],    // subset {1,3} not frequent → prune
        ];

        let result = prune_step(&freq_lower, candidates);
        assert_eq!(result, vec![vec![0u32, 1, 2]]);
    }

    #[test]
    fn test_apriori_full_k2() {
        let mut index = make_test_index();
        // min_support = 0.4 → min_count = ceil(0.4 * 5) = 2
        let levels = apriori_itemsets(&mut index, 0.4, 2);

        // Level 1: all items appear in >= 2 transactions
        assert!(levels.contains_key(&1));
        let l1 = &levels[&1];
        // Item 0: 4 txns, Item 1: 3 txns, Item 2: 4 txns, Item 3: 3 txns, Item 4: 2 txns
        assert_eq!(l1.len(), 5);
        assert_eq!(l1[&vec![0]], 4);
        assert_eq!(l1[&vec![4]], 2);

        // Level 2: pairs with count >= 2
        assert!(levels.contains_key(&2));
        let l2 = &levels[&2];
        // {0,1}=3, {0,2}=3, {0,3}=2, {1,2}=2, {1,3}=2, {2,3}=2, {2,4}=2
        // {0,4}=1, {3,4}=1, {1,4}=0 → not frequent
        assert_eq!(l2.len(), 7);
        assert_eq!(l2[&vec![0, 1]], 3);
    }

    #[test]
    fn test_apriori_full_k3() {
        let mut index = make_test_index();
        // min_support = 0.4 → min_count = 2
        let levels = apriori_itemsets(&mut index, 0.4, 3);

        assert!(levels.contains_key(&3));
        let l3 = &levels[&3];
        // {0,1,2}=2 (T0,T2), {0,1,3}=2 (T1,T2), {0,2,3}=1 (T2 only) → pruned
        // {2,3,4}=1 → pruned
        assert!(l3.contains_key(&vec![0, 1, 2]));
        assert_eq!(l3[&vec![0, 1, 2]], 2);
        assert!(l3.contains_key(&vec![0, 1, 3]));
        assert_eq!(l3[&vec![0, 1, 3]], 2);
    }

    #[test]
    fn test_classic_matches_fast_on_pairs() {
        // Build a small dataset and verify that classic and fast produce
        // the same frequent pair set at the same min_support.
        let txn_ids: Vec<i64> = vec![
            0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4,
            5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
        ];
        let item_ids: Vec<i32> = vec![
            0, 1, 2, 0, 1, 3, 0, 1, 2, 3, 2, 3, 4, 0, 2, 4,
            1, 4, 0, 3, 4, 1, 2, 3, 0, 1, 4, 2, 3, 4,
        ];
        let n_items = 5u32;
        let n_transactions = 10u32;
        let min_support = 0.3;

        // Classic path
        let mut index = TransactionIndex::build(&txn_ids, &item_ids);
        let levels = apriori_itemsets(&mut index, min_support, 2);
        let classic_pairs: HashMap<Vec<u32>, u32> =
            levels.get(&2).cloned().unwrap_or_default();

        // Fast path (using the pairs module)
        let (txn_to_items, item_to_txns, _) =
            common::build_inverted_index(&txn_ids, &item_ids);
        let (fast_pair_counts, _) = crate::pairs::count_frequent_pairs(
            &txn_to_items,
            &item_to_txns,
            n_items as usize,
            n_transactions,
            min_support,
        );
        let fast_pairs: HashMap<Vec<u32>, u32> = fast_pair_counts
            .into_iter()
            .map(|((a, b), count)| (vec![a, b], count))
            .collect();

        // They should produce the same set of frequent pairs with the same counts
        assert_eq!(
            classic_pairs.len(),
            fast_pairs.len(),
            "Different number of frequent pairs: classic={} fast={}",
            classic_pairs.len(),
            fast_pairs.len()
        );

        for (pair, &classic_count) in &classic_pairs {
            let fast_count = fast_pairs
                .get(pair)
                .unwrap_or_else(|| panic!("Classic has pair {:?} but fast doesn't", pair));
            assert_eq!(
                classic_count, *fast_count,
                "Count mismatch for pair {:?}: classic={} fast={}",
                pair, classic_count, fast_count
            );
        }
    }
}
