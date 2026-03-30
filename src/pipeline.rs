//! Full pipeline: k=2 → k=3 → ... → k_max in a single Rust call.
//! Eliminates Python round-trips between levels.

use std::collections::{HashMap, HashSet};

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::common;
use crate::itemsets;
use crate::pairs;

/// Compute the full k=2 → ... → k_max association pipeline in Rust.
///
/// Returns a dict with:
/// - "itemsets": 2D int32 array (n_results, k_max) — the k_max-level itemsets
/// - "counts": 1D int64 array (n_results,) — their counts
/// - "lower_itemsets": 2D int32 array (n_lower, k_max-1) — (k_max-1)-level freq sets
/// - "lower_counts": 1D int64 array (n_lower,) — their counts
/// - "item_counts": 1D int64 array (n_items,) — per-item transaction counts
#[pyfunction]
pub fn rust_compute_pipeline<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    k_max: usize,
    n_items: u32,
    n_transactions: u32,
    min_support: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;

    // Step 1: Build inverted index ONCE
    let (txn_to_items, item_to_txns, item_counts) =
        common::build_inverted_index(txn_slice, item_slice);

    let n_txn_f64 = n_transactions as f64;
    let min_count = if min_support > 0.0 {
        (min_support * n_txn_f64).ceil() as u32
    } else {
        1
    };

    // Step 2: k=2 pair counting (filtered by min_support)
    let (pair_counts, freq_pairs) = pairs::count_frequent_pairs(
        &txn_to_items,
        &item_to_txns,
        n_items as usize,
        n_transactions,
        min_support,
    );

    // Convert pair_counts to generic Vec<u32> format
    let mut lower_counts: HashMap<Vec<u32>, u32> = pair_counts
        .iter()
        .map(|(&(a, b), &c)| (vec![a, b], c))
        .collect();
    let mut lower_sets: HashSet<Vec<u32>> = lower_counts.keys().cloned().collect();

    // Step 3+: k=3, k=4, ... k_max
    let mut final_counts: HashMap<Vec<u32>, u32> = HashMap::new();

    for k in 3..=k_max {
        let kset_counts = itemsets::count_k_itemsets_internal(
            &txn_to_items,
            &lower_sets,
            &freq_pairs,
            k,
            n_items as usize,
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

    // Item-level counts (indexed by item_id)
    let mut item_counts_arr = vec![0i64; n_items as usize];
    for (&item, &count) in &item_counts {
        item_counts_arr[item as usize] = count as i64;
    }
    dict.set_item(
        "item_counts",
        PyArray1::from_vec(py, item_counts_arr),
    )?;

    Ok(dict)
}
