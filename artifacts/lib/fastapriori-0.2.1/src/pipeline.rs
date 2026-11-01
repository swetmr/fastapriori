//! Full pipeline: k=2 → k=3 → ... → k_max in a single Rust call.
//! Eliminates Python round-trips between levels.

use std::collections::{HashMap, HashSet};

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::common;
use crate::common::{ItemTxnsRoaring, TidList};
use crate::itemsets;
use crate::pairs;
use roaring::RoaringBitmap;

/// Compute the full k=2 → ... → k_max association pipeline in Rust.
///
/// Returns a dict with:
/// - "itemsets": 2D int32 array (n_results, k_max) — the k_max-level itemsets
/// - "counts": 1D int64 array (n_results,) — their counts
/// - "lower_itemsets": 2D int32 array (n_lower, k_max-1) — (k_max-1)-level freq sets
/// - "lower_counts": 1D int64 array (n_lower,) — their counts
/// - "item_counts": 1D int64 array (n_items,) — per-item transaction counts
#[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, k_max, n_items, min_support, item_weights, max_items_per_txn=None))]
pub fn rust_compute_pipeline<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    k_max: usize,
    n_items: u32,
    min_support: f64,
    item_weights: PyReadonlyArray1<'py, f64>,
    max_items_per_txn: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;
    let weights_slice = item_weights.as_slice()?;

    // Release the GIL for the entire k=2 → k_max computation — no Python
    // objects are touched inside. Mirrors src/pairs.rs.
    let (final_counts, lower_counts, item_counts_vec, n_transactions) = py.allow_threads(|| {
        // Step 1: Build CSR inverted index ONCE (factorises txn_ids to dense u32).
        let idx = common::build_inverted_index(txn_slice, item_slice, n_items);
        let n_transactions = idx.n_txns;

        let n_txn_f64 = n_transactions as f64;
        let min_count = if min_support > 0.0 {
            common::min_count_for(min_support, n_txn_f64) as u32
        } else {
            1
        };

        // Step 2: k=2 pair counting (filtered by min_support)
        let (pair_counts, freq_pairs) = pairs::count_frequent_pairs(
            &idx,
            n_items as usize,
            n_transactions,
            min_support,
        );

        // Convert pair_counts to generic Vec<u32> format
        let mut lower_counts: HashMap<Vec<u32>, u32> = pair_counts
            .iter()
            .map(|(&(a, b), &c)| (vec![a, b], c))
            .collect();
        drop(pair_counts);
        let mut lower_sets: HashSet<Vec<u32>> = lower_counts.keys().cloned().collect();

        // Step 3+: k=3, k=4, ... k_max
        let mut final_counts: HashMap<Vec<u32>, u32> = HashMap::new();

        for k in 3..=k_max {
            let kset_counts = itemsets::count_k_itemsets_internal(
                &idx,
                &lower_sets,
                &freq_pairs,
                k,
                n_items as usize,
                weights_slice,
                max_items_per_txn,
            );

            // Filter by min_support
            let filtered: HashMap<Vec<u32>, u32> = kset_counts
                .into_iter()
                .filter(|(_, count)| *count >= min_count)
                .collect();

            if k < k_max {
                // This level becomes the lower level for the next iteration
                lower_sets = filtered.keys().cloned().collect();
                lower_counts = filtered;
            } else {
                // This is the output level; lower_counts stays from previous level
                final_counts = filtered;
            }
        }

        // Clone item_counts out of the CSR struct before it's dropped.
        let item_counts_vec = idx.item_counts.clone();
        (final_counts, lower_counts, item_counts_vec, n_transactions)
    });

    // Build output dict
    let dict = PyDict::new(py);

    // k_max level: itemsets + counts
    let results: Vec<(&Vec<u32>, &u32)> = final_counts.iter().collect();
    let itemsets_vecs: Vec<Vec<i32>> = results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let counts_vec: Vec<i64> = results.iter().map(|(_, &count)| count as i64).collect();

    dict.set_item("itemsets", PyArray2::from_vec2(py, &itemsets_vecs)?)?;
    dict.set_item("counts", PyArray1::from_vec(py, counts_vec))?;

    // (k_max-1) level: lower_itemsets + lower_counts
    let lower_results: Vec<(&Vec<u32>, &u32)> = lower_counts.iter().collect();
    let lower_itemsets_vecs: Vec<Vec<i32>> = lower_results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let lower_counts_vec: Vec<i64> = lower_results
        .iter()
        .map(|(_, &count)| count as i64)
        .collect();

    dict.set_item(
        "lower_itemsets",
        PyArray2::from_vec2(py, &lower_itemsets_vecs)?,
    )?;
    dict.set_item(
        "lower_counts",
        PyArray1::from_vec(py, lower_counts_vec),
    )?;

    // Item-level counts (indexed by item_id). idx.item_counts is already a
    // dense Vec<u32> of length n_items — just widen to i64 for Python.
    let item_counts_arr: Vec<i64> = item_counts_vec.into_iter().map(|c| c as i64).collect();
    dict.set_item(
        "item_counts",
        PyArray1::from_vec(py, item_counts_arr),
    )?;

    // Return the factorised txn count so the caller can skip its own
    // `.nunique()` pass over the raw column.
    dict.set_item("n_transactions", n_transactions as i64)?;

    Ok(dict)
}

/// Phase-1 ablation variant. Same level loop and output schema as
/// `rust_compute_pipeline`, but the per-level itemset counter calls the
/// roaring-bitmap variant `count_k_itemsets_v1_roaring`. Used by the
/// local-test notebook for A/B comparison; selected via
/// `backend_options={'impl_variant': 'v1_roaring'}` from Python.
#[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, k_max, n_items, min_support, item_weights, max_items_per_txn=None))]
pub fn rust_compute_pipeline_v1_roaring<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    k_max: usize,
    n_items: u32,
    min_support: f64,
    item_weights: PyReadonlyArray1<'py, f64>,
    max_items_per_txn: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;
    let weights_slice = item_weights.as_slice()?;

    let (final_counts, lower_counts, item_counts_vec, n_transactions) = py.allow_threads(|| {
        let idx = common::build_inverted_index(txn_slice, item_slice, n_items);
        let n_transactions = idx.n_txns;

        let n_txn_f64 = n_transactions as f64;
        let min_count = if min_support > 0.0 {
            common::min_count_for(min_support, n_txn_f64) as u32
        } else {
            1
        };

        // k=2 path is unchanged from baseline (pairs.rs is intentionally
        // not modified — its scatter-counter is already optimal for k=2).
        let (pair_counts, freq_pairs) = pairs::count_frequent_pairs(
            &idx,
            n_items as usize,
            n_transactions,
            min_support,
        );

        let mut lower_counts: HashMap<Vec<u32>, u32> = pair_counts
            .iter()
            .map(|(&(a, b), &c)| (vec![a, b], c))
            .collect();
        drop(pair_counts);
        let mut lower_sets: HashSet<Vec<u32>> = lower_counts.keys().cloned().collect();

        let mut final_counts: HashMap<Vec<u32>, u32> = HashMap::new();

        for k in 3..=k_max {
            let kset_counts = itemsets::count_k_itemsets_v1_roaring(
                &idx,
                &lower_sets,
                &freq_pairs,
                k,
                n_items as usize,
                weights_slice,
                max_items_per_txn,
            );

            let filtered: HashMap<Vec<u32>, u32> = kset_counts
                .into_iter()
                .filter(|(_, count)| *count >= min_count)
                .collect();

            if k < k_max {
                lower_sets = filtered.keys().cloned().collect();
                lower_counts = filtered;
            } else {
                final_counts = filtered;
            }
        }

        let item_counts_vec = idx.item_counts.clone();
        (final_counts, lower_counts, item_counts_vec, n_transactions)
    });

    // Output dict — identical schema to rust_compute_pipeline.
    let dict = PyDict::new(py);

    let results: Vec<(&Vec<u32>, &u32)> = final_counts.iter().collect();
    let itemsets_vecs: Vec<Vec<i32>> = results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let counts_vec: Vec<i64> = results.iter().map(|(_, &count)| count as i64).collect();

    dict.set_item("itemsets", PyArray2::from_vec2(py, &itemsets_vecs)?)?;
    dict.set_item("counts", PyArray1::from_vec(py, counts_vec))?;

    let lower_results: Vec<(&Vec<u32>, &u32)> = lower_counts.iter().collect();
    let lower_itemsets_vecs: Vec<Vec<i32>> = lower_results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let lower_counts_vec: Vec<i64> = lower_results
        .iter()
        .map(|(_, &count)| count as i64)
        .collect();

    dict.set_item(
        "lower_itemsets",
        PyArray2::from_vec2(py, &lower_itemsets_vecs)?,
    )?;
    dict.set_item(
        "lower_counts",
        PyArray1::from_vec(py, lower_counts_vec),
    )?;

    let item_counts_arr: Vec<i64> = item_counts_vec.into_iter().map(|c| c as i64).collect();
    dict.set_item(
        "item_counts",
        PyArray1::from_vec(py, item_counts_arr),
    )?;

    dict.set_item("n_transactions", n_transactions as i64)?;

    Ok(dict)
}

/// Phase-2 ablation variant. Same level loop and output schema as
/// `rust_compute_pipeline`, but the per-level itemset counter uses
/// `count_k_itemsets_v2_memo` and carries `lower_to_txns: HashMap<Vec<u32>,
/// RoaringBitmap>` across BFS levels.
#[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, k_max, n_items, min_support, item_weights, max_items_per_txn=None))]
pub fn rust_compute_pipeline_v2_memo<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    k_max: usize,
    n_items: u32,
    min_support: f64,
    item_weights: PyReadonlyArray1<'py, f64>,
    max_items_per_txn: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;
    let weights_slice = item_weights.as_slice()?;

    let (final_counts, lower_counts, item_counts_vec, n_transactions) = py.allow_threads(|| {
        let idx = common::build_inverted_index(txn_slice, item_slice, n_items);
        let n_transactions = idx.n_txns;

        let n_txn_f64 = n_transactions as f64;
        let min_count = if min_support > 0.0 {
            common::min_count_for(min_support, n_txn_f64) as u32
        } else {
            1
        };

        let (pair_counts, freq_pairs) = pairs::count_frequent_pairs(
            &idx,
            n_items as usize,
            n_transactions,
            min_support,
        );

        let mut lower_counts: HashMap<Vec<u32>, u32> = pair_counts
            .iter()
            .map(|(&(a, b), &c)| (vec![a, b], c))
            .collect();
        drop(pair_counts);

        let mut final_counts: HashMap<Vec<u32>, u32> = HashMap::new();

        if k_max >= 3 {
            let item_roar = ItemTxnsRoaring::from_index(&idx);
            let mut lower_to_txns: HashMap<Vec<u32>, RoaringBitmap> =
                itemsets::build_pair_tid_map(&freq_pairs, &item_roar);

            for k in 3..=k_max {
                let (kset_counts, new_lower) = itemsets::count_k_itemsets_v2_memo(
                    &idx,
                    &lower_to_txns,
                    &item_roar,
                    &freq_pairs,
                    k,
                    n_items as usize,
                    min_count,
                    weights_slice,
                    max_items_per_txn,
                );

                if k < k_max {
                    lower_counts = kset_counts;
                    lower_to_txns = new_lower;
                } else {
                    final_counts = kset_counts;
                    drop(new_lower);
                }
            }
            drop(lower_to_txns);
        }

        let item_counts_vec = idx.item_counts.clone();
        (final_counts, lower_counts, item_counts_vec, n_transactions)
    });

    let dict = PyDict::new(py);

    let results: Vec<(&Vec<u32>, &u32)> = final_counts.iter().collect();
    let itemsets_vecs: Vec<Vec<i32>> = results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let counts_vec: Vec<i64> = results.iter().map(|(_, &count)| count as i64).collect();

    dict.set_item("itemsets", PyArray2::from_vec2(py, &itemsets_vecs)?)?;
    dict.set_item("counts", PyArray1::from_vec(py, counts_vec))?;

    let lower_results: Vec<(&Vec<u32>, &u32)> = lower_counts.iter().collect();
    let lower_itemsets_vecs: Vec<Vec<i32>> = lower_results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let lower_counts_vec: Vec<i64> = lower_results
        .iter()
        .map(|(_, &count)| count as i64)
        .collect();

    dict.set_item(
        "lower_itemsets",
        PyArray2::from_vec2(py, &lower_itemsets_vecs)?,
    )?;
    dict.set_item(
        "lower_counts",
        PyArray1::from_vec(py, lower_counts_vec),
    )?;

    let item_counts_arr: Vec<i64> = item_counts_vec.into_iter().map(|c| c as i64).collect();
    dict.set_item(
        "item_counts",
        PyArray1::from_vec(py, item_counts_arr),
    )?;

    dict.set_item("n_transactions", n_transactions as i64)?;

    Ok(dict)
}

/// Phase-D ablation variant: same memoised BFS as `rust_compute_pipeline_v2_memo`
/// but using the adaptive `TidList` enum (Sparse Vec<u32> / Dense
/// RoaringBitmap) for level TID-lists. Fixes the small-list iteration
/// regression v1/v2 paid by always using RoaringBitmap.
#[cfg(not(feature = "phaseprof"))] #[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, k_max, n_items, min_support, item_weights, max_items_per_txn=None))]
pub fn rust_compute_pipeline_v3_adaptive<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    k_max: usize,
    n_items: u32,
    min_support: f64,
    item_weights: PyReadonlyArray1<'py, f64>,
    max_items_per_txn: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;
    let weights_slice = item_weights.as_slice()?;

    let (final_counts, lower_counts, item_counts_vec, n_transactions) = py.allow_threads(|| {
        let idx = common::build_inverted_index(txn_slice, item_slice, n_items);
        let n_transactions = idx.n_txns;

        let n_txn_f64 = n_transactions as f64;
        let min_count = if min_support > 0.0 {
            common::min_count_for(min_support, n_txn_f64) as u32
        } else {
            1
        };

        let (pair_counts, freq_pairs) = pairs::count_frequent_pairs(
            &idx,
            n_items as usize,
            n_transactions,
            min_support,
        );

        let mut lower_counts: HashMap<Vec<u32>, u32> = pair_counts
            .iter()
            .map(|(&(a, b), &c)| (vec![a, b], c))
            .collect();
        drop(pair_counts);

        let mut final_counts: HashMap<Vec<u32>, u32> = HashMap::new();

        if k_max >= 3 {
            let item_roar = ItemTxnsRoaring::from_index(&idx);
            let mut lower_to_txns: HashMap<Vec<u32>, TidList> =
                itemsets::build_pair_tid_map_adaptive(&freq_pairs, &item_roar);

            for k in 3..=k_max {
                let (kset_counts, new_lower) = itemsets::count_k_itemsets_v3_adaptive(
                    &idx,
                    &lower_to_txns,
                    &item_roar,
                    &freq_pairs,
                    k,
                    n_items as usize,
                    min_count,
                    weights_slice,
                    max_items_per_txn,
                );

                if k < k_max {
                    lower_counts = kset_counts;
                    lower_to_txns = new_lower;
                } else {
                    final_counts = kset_counts;
                    drop(new_lower);
                }
            }
            drop(lower_to_txns);
        }

        let item_counts_vec = idx.item_counts.clone();
        (final_counts, lower_counts, item_counts_vec, n_transactions)
    });

    let dict = PyDict::new(py);

    let results: Vec<(&Vec<u32>, &u32)> = final_counts.iter().collect();
    let itemsets_vecs: Vec<Vec<i32>> = results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let counts_vec: Vec<i64> = results.iter().map(|(_, &count)| count as i64).collect();

    dict.set_item("itemsets", PyArray2::from_vec2(py, &itemsets_vecs)?)?;
    dict.set_item("counts", PyArray1::from_vec(py, counts_vec))?;

    let lower_results: Vec<(&Vec<u32>, &u32)> = lower_counts.iter().collect();
    let lower_itemsets_vecs: Vec<Vec<i32>> = lower_results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let lower_counts_vec: Vec<i64> = lower_results
        .iter()
        .map(|(_, &count)| count as i64)
        .collect();

    dict.set_item(
        "lower_itemsets",
        PyArray2::from_vec2(py, &lower_itemsets_vecs)?,
    )?;
    dict.set_item(
        "lower_counts",
        PyArray1::from_vec(py, lower_counts_vec),
    )?;

    let item_counts_arr: Vec<i64> = item_counts_vec.into_iter().map(|c| c as i64).collect();
    dict.set_item(
        "item_counts",
        PyArray1::from_vec(py, item_counts_arr),
    )?;

    dict.set_item("n_transactions", n_transactions as i64)?;

    Ok(dict)
}

/// v5_prefilter — apriori downward-closure check applied PRE-counter
/// instead of POST-counter. Otherwise identical to v3_adaptive (uses the
/// same adaptive Sparse/Dense TidList storage and roaring-rs as the dense
/// container). Expected wins on wide-sparse data; possible regression on
/// narrow-dense.
#[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, k_max, n_items, min_support, item_weights, max_items_per_txn=None))]
pub fn rust_compute_pipeline_v5_prefilter<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    k_max: usize,
    n_items: u32,
    min_support: f64,
    item_weights: PyReadonlyArray1<'py, f64>,
    max_items_per_txn: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;
    let weights_slice = item_weights.as_slice()?;

    let (final_counts, lower_counts, item_counts_vec, n_transactions) = py.allow_threads(|| {
        let idx = common::build_inverted_index(txn_slice, item_slice, n_items);
        let n_transactions = idx.n_txns;

        let n_txn_f64 = n_transactions as f64;
        let min_count = if min_support > 0.0 {
            common::min_count_for(min_support, n_txn_f64) as u32
        } else {
            1
        };

        let (pair_counts, freq_pairs) = pairs::count_frequent_pairs(
            &idx,
            n_items as usize,
            n_transactions,
            min_support,
        );

        let mut lower_counts: HashMap<Vec<u32>, u32> = pair_counts
            .iter()
            .map(|(&(a, b), &c)| (vec![a, b], c))
            .collect();
        drop(pair_counts);

        let mut final_counts: HashMap<Vec<u32>, u32> = HashMap::new();

        if k_max >= 3 {
            let item_roar = ItemTxnsRoaring::from_index(&idx);
            let mut lower_to_txns: HashMap<Vec<u32>, TidList> =
                itemsets::build_pair_tid_map_adaptive(&freq_pairs, &item_roar);

            for k in 3..=k_max {
                let (kset_counts, new_lower) = itemsets::count_k_itemsets_v5_prefilter(
                    &idx,
                    &lower_to_txns,
                    &item_roar,
                    &freq_pairs,
                    k,
                    n_items as usize,
                    min_count,
                    weights_slice,
                    max_items_per_txn,
                );

                if k < k_max {
                    lower_counts = kset_counts;
                    lower_to_txns = new_lower;
                } else {
                    final_counts = kset_counts;
                    drop(new_lower);
                }
            }
            drop(lower_to_txns);
        }

        let item_counts_vec = idx.item_counts.clone();
        (final_counts, lower_counts, item_counts_vec, n_transactions)
    });

    let dict = PyDict::new(py);

    let results: Vec<(&Vec<u32>, &u32)> = final_counts.iter().collect();
    let itemsets_vecs: Vec<Vec<i32>> = results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let counts_vec: Vec<i64> = results.iter().map(|(_, &count)| count as i64).collect();

    dict.set_item("itemsets", PyArray2::from_vec2(py, &itemsets_vecs)?)?;
    dict.set_item("counts", PyArray1::from_vec(py, counts_vec))?;

    let lower_results: Vec<(&Vec<u32>, &u32)> = lower_counts.iter().collect();
    let lower_itemsets_vecs: Vec<Vec<i32>> = lower_results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let lower_counts_vec: Vec<i64> = lower_results
        .iter()
        .map(|(_, &count)| count as i64)
        .collect();

    dict.set_item(
        "lower_itemsets",
        PyArray2::from_vec2(py, &lower_itemsets_vecs)?,
    )?;
    dict.set_item(
        "lower_counts",
        PyArray1::from_vec(py, lower_counts_vec),
    )?;

    let item_counts_arr: Vec<i64> = item_counts_vec.into_iter().map(|c| c as i64).collect();
    dict.set_item(
        "item_counts",
        PyArray1::from_vec(py, item_counts_arr),
    )?;

    dict.set_item("n_transactions", n_transactions as i64)?;

    Ok(dict)
}

/// v6_gating — per-anchor candidate set built from pre-computed
/// pair_neighbors. Replaces both anchor.contains() and apriori_check() in
/// the inner loop with a single HashSet membership test, amortizing the
/// pair-check work to once-per-anchor instead of once-per-(txn, item) pair.
/// Otherwise identical to v3_adaptive (uses the same TidList Sparse/Dense
/// adaptive storage).
#[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, k_max, n_items, min_support, item_weights, max_items_per_txn=None))]
pub fn rust_compute_pipeline_v6_gating<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    k_max: usize,
    n_items: u32,
    min_support: f64,
    item_weights: PyReadonlyArray1<'py, f64>,
    max_items_per_txn: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;
    let weights_slice = item_weights.as_slice()?;

    let (final_counts, lower_counts, item_counts_vec, n_transactions) = py.allow_threads(|| {
        let idx = common::build_inverted_index(txn_slice, item_slice, n_items);
        let n_transactions = idx.n_txns;

        let n_txn_f64 = n_transactions as f64;
        let min_count = if min_support > 0.0 {
            common::min_count_for(min_support, n_txn_f64) as u32
        } else {
            1
        };

        let (pair_counts, freq_pairs) = pairs::count_frequent_pairs(
            &idx,
            n_items as usize,
            n_transactions,
            min_support,
        );

        let mut lower_counts: HashMap<Vec<u32>, u32> = pair_counts
            .iter()
            .map(|(&(a, b), &c)| (vec![a, b], c))
            .collect();
        drop(pair_counts);

        let mut final_counts: HashMap<Vec<u32>, u32> = HashMap::new();

        if k_max >= 3 {
            let item_roar = ItemTxnsRoaring::from_index(&idx);
            // Build pair_neighbors ONCE per pipeline (reused across all levels k>=3).
            let pair_neighbors = itemsets::build_pair_neighbors(&freq_pairs, n_items);
            let mut lower_to_txns: HashMap<Vec<u32>, TidList> =
                itemsets::build_pair_tid_map_adaptive(&freq_pairs, &item_roar);

            for k in 3..=k_max {
                let (kset_counts, new_lower) = itemsets::count_k_itemsets_v6_gating(
                    &idx,
                    &lower_to_txns,
                    &item_roar,
                    &pair_neighbors,
                    k,
                    n_items as usize,
                    min_count,
                    weights_slice,
                    max_items_per_txn,
                );

                if k < k_max {
                    lower_counts = kset_counts;
                    lower_to_txns = new_lower;
                } else {
                    final_counts = kset_counts;
                    drop(new_lower);
                }
            }
            drop(lower_to_txns);
        }

        let item_counts_vec = idx.item_counts.clone();
        (final_counts, lower_counts, item_counts_vec, n_transactions)
    });

    let dict = PyDict::new(py);

    let results: Vec<(&Vec<u32>, &u32)> = final_counts.iter().collect();
    let itemsets_vecs: Vec<Vec<i32>> = results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let counts_vec: Vec<i64> = results.iter().map(|(_, &count)| count as i64).collect();

    dict.set_item("itemsets", PyArray2::from_vec2(py, &itemsets_vecs)?)?;
    dict.set_item("counts", PyArray1::from_vec(py, counts_vec))?;

    let lower_results: Vec<(&Vec<u32>, &u32)> = lower_counts.iter().collect();
    let lower_itemsets_vecs: Vec<Vec<i32>> = lower_results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let lower_counts_vec: Vec<i64> = lower_results
        .iter()
        .map(|(_, &count)| count as i64)
        .collect();

    dict.set_item(
        "lower_itemsets",
        PyArray2::from_vec2(py, &lower_itemsets_vecs)?,
    )?;
    dict.set_item(
        "lower_counts",
        PyArray1::from_vec(py, lower_counts_vec),
    )?;

    let item_counts_arr: Vec<i64> = item_counts_vec.into_iter().map(|c| c as i64).collect();
    dict.set_item(
        "item_counts",
        PyArray1::from_vec(py, item_counts_arr),
    )?;

    dict.set_item("n_transactions", n_transactions as i64)?;

    Ok(dict)
}

// ---------------------------------------------------------------------------
// Phase profiling (fastapriori 0.2.1). Compiled only with `--features phaseprof`;
// default builds use the 0.2.0 function above unchanged. The instrumented copy below
// takes coarse phase timers and, when FASTAPRIORI_PROFILE is set, returns a "profile"
// entry with per-level work counters.
// ---------------------------------------------------------------------------
#[cfg(feature = "phaseprof")]
use std::time::Instant;
#[cfg(feature = "phaseprof")]
use pyo3::types::PyList;

#[cfg(feature = "phaseprof")]
#[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, k_max, n_items, min_support, item_weights, max_items_per_txn=None))]
pub fn rust_compute_pipeline_v3_adaptive<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    k_max: usize,
    n_items: u32,
    min_support: f64,
    item_weights: PyReadonlyArray1<'py, f64>,
    max_items_per_txn: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;
    let weights_slice = item_weights.as_slice()?;

    // [phaseprof] coarse phase timers. They are always taken (about ten clock reads per call);
    // the per-level work counters and the "profile" entry are produced only when the
    // FASTAPRIORI_PROFILE environment variable is set.
    let profile = std::env::var_os("FASTAPRIORI_PROFILE").is_some();
    let t_call = Instant::now();
    let (final_counts, lower_counts, item_counts_vec, n_transactions, ph) = py.allow_threads(|| {
        let mut ph = PhaseTimes::default();
        let t = Instant::now();
        let idx = common::build_inverted_index(txn_slice, item_slice, n_items);
        ph.index_s = t.elapsed().as_secs_f64();
        let n_transactions = idx.n_txns;

        let n_txn_f64 = n_transactions as f64;
        let min_count = if min_support > 0.0 {
            common::min_count_for(min_support, n_txn_f64) as u32
        } else {
            1
        };

        let t = Instant::now();
        let (pair_counts, freq_pairs) = pairs::count_frequent_pairs(
            &idx,
            n_items as usize,
            n_transactions,
            min_support,
        );

        let mut lower_counts: HashMap<Vec<u32>, u32> = pair_counts
            .iter()
            .map(|(&(a, b), &c)| (vec![a, b], c))
            .collect();
        drop(pair_counts);
        ph.pairs_s = t.elapsed().as_secs_f64();
        ph.n_freq_pairs = freq_pairs.len();

        let mut final_counts: HashMap<Vec<u32>, u32> = HashMap::new();

        if k_max >= 3 {
            let t = Instant::now();
            let item_roar = ItemTxnsRoaring::from_index(&idx);
            let mut lower_to_txns: HashMap<Vec<u32>, TidList> =
                itemsets::build_pair_tid_map_adaptive(&freq_pairs, &item_roar);
            ph.pair_tids_s = t.elapsed().as_secs_f64();

            for k in 3..=k_max {
                let n_anchors = lower_to_txns.len();
                let anchor_tids: u64 = if profile {
                    lower_to_txns.values().map(|tids| tids.len()).sum()
                } else {
                    0
                };
                let t = Instant::now();
                let (kset_counts, new_lower, project_s, n_projected) =
                    itemsets::count_k_itemsets_v3_adaptive(
                        &idx,
                        &lower_to_txns,
                        &item_roar,
                        &freq_pairs,
                        k,
                        n_items as usize,
                        min_count,
                        weights_slice,
                        max_items_per_txn,
                    );
                let level_s = t.elapsed().as_secs_f64();
                ph.levels.push(LevelTimes {
                    k,
                    project_s,
                    extend_s: level_s - project_s,
                    n_anchors,
                    anchor_tids,
                    n_projected,
                    n_frequent: kset_counts.len(),
                });

                if k < k_max {
                    lower_counts = kset_counts;
                    lower_to_txns = new_lower;
                } else {
                    final_counts = kset_counts;
                    drop(new_lower);
                }
            }
            drop(lower_to_txns);
        }

        let item_counts_vec = idx.item_counts.clone();
        (final_counts, lower_counts, item_counts_vec, n_transactions, ph)
    });
    let core_s = t_call.elapsed().as_secs_f64();

    let t_out = Instant::now();
    let dict = PyDict::new(py);

    let results: Vec<(&Vec<u32>, &u32)> = final_counts.iter().collect();
    let itemsets_vecs: Vec<Vec<i32>> = results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let counts_vec: Vec<i64> = results.iter().map(|(_, &count)| count as i64).collect();

    dict.set_item("itemsets", PyArray2::from_vec2(py, &itemsets_vecs)?)?;
    dict.set_item("counts", PyArray1::from_vec(py, counts_vec))?;

    let lower_results: Vec<(&Vec<u32>, &u32)> = lower_counts.iter().collect();
    let lower_itemsets_vecs: Vec<Vec<i32>> = lower_results
        .iter()
        .map(|(kset, _)| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let lower_counts_vec: Vec<i64> = lower_results
        .iter()
        .map(|(_, &count)| count as i64)
        .collect();

    dict.set_item(
        "lower_itemsets",
        PyArray2::from_vec2(py, &lower_itemsets_vecs)?,
    )?;
    dict.set_item(
        "lower_counts",
        PyArray1::from_vec(py, lower_counts_vec),
    )?;

    let item_counts_arr: Vec<i64> = item_counts_vec.into_iter().map(|c| c as i64).collect();
    dict.set_item(
        "item_counts",
        PyArray1::from_vec(py, item_counts_arr),
    )?;

    dict.set_item("n_transactions", n_transactions as i64)?;

    if profile {
        let output_rust_s = t_out.elapsed().as_secs_f64();
        let p = PyDict::new(py);
        p.set_item("threads", rayon::current_num_threads())?;
        p.set_item("index_s", ph.index_s)?;
        p.set_item("pairs_s", ph.pairs_s)?;
        p.set_item("pair_tids_s", ph.pair_tids_s)?;
        p.set_item("core_s", core_s)?;
        p.set_item("output_rust_s", output_rust_s)?;
        p.set_item("n_freq_pairs", ph.n_freq_pairs)?;
        p.set_item("n_final", final_counts.len())?;
        p.set_item("n_lower", lower_counts.len())?;
        let levels = PyList::empty(py);
        for lv in &ph.levels {
            let d = PyDict::new(py);
            d.set_item("k", lv.k)?;
            d.set_item("project_s", lv.project_s)?;
            d.set_item("extend_s", lv.extend_s)?;
            d.set_item("n_anchors", lv.n_anchors)?;
            d.set_item("anchor_tids", lv.anchor_tids)?;
            d.set_item("n_projected", lv.n_projected)?;
            d.set_item("n_frequent", lv.n_frequent)?;
            levels.append(d)?;
        }
        p.set_item("levels", levels)?;
        dict.set_item("profile", p)?;
    }

    Ok(dict)
}

/// [phaseprof] Per-call phase times of `rust_compute_pipeline_v3_adaptive`.
#[cfg(feature = "phaseprof")]
#[derive(Default)]
struct PhaseTimes {
    index_s: f64,
    pairs_s: f64,
    pair_tids_s: f64,
    n_freq_pairs: usize,
    levels: Vec<LevelTimes>,
}

/// [phaseprof] One extension level: projection (filtering transactions to frequent items),
/// extension (anchor-and-extend counting, intersections and merge) and work counters.
/// `anchor_tids` is the summed TID-list length of the anchors, i.e. the transactions scanned.
#[cfg(feature = "phaseprof")]
#[derive(Default)]
struct LevelTimes {
    k: usize,
    project_s: f64,
    extend_s: f64,
    n_anchors: usize,
    anchor_tids: u64,
    n_projected: usize,
    n_frequent: usize,
}
