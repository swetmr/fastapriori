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
            (min_support * n_txn_f64).ceil() as u32
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
            (min_support * n_txn_f64).ceil() as u32
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
            (min_support * n_txn_f64).ceil() as u32
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

    let (final_counts, lower_counts, item_counts_vec, n_transactions) = py.allow_threads(|| {
        let idx = common::build_inverted_index(txn_slice, item_slice, n_items);
        let n_transactions = idx.n_txns;

        let n_txn_f64 = n_transactions as f64;
        let min_count = if min_support > 0.0 {
            (min_support * n_txn_f64).ceil() as u32
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
            (min_support * n_txn_f64).ceil() as u32
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
            (min_support * n_txn_f64).ceil() as u32
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
