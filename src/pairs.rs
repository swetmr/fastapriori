use std::collections::{HashMap, HashSet};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::common;

/// Count frequent undirected pairs. Returns (freq_pair_counts, freq_pairs_set).
/// Only pairs with support >= min_support are returned.
/// Pairs are canonical: (a, b) where a < b.
pub fn count_frequent_pairs(
    txn_to_items: &HashMap<i64, Vec<u32>>,
    item_to_txns: &HashMap<u32, Vec<i64>>,
    n_items: usize,
    n_transactions: u32,
    min_support: f64,
) -> (HashMap<(u32, u32), u32>, HashSet<(u32, u32)>) {
    let n_txn_f64 = n_transactions as f64;
    let min_count = if min_support > 0.0 {
        (min_support * n_txn_f64).ceil() as u32
    } else {
        1 // at least 1 occurrence
    };

    let mut pair_counts: HashMap<(u32, u32), u32> = HashMap::new();
    let mut freq_pairs: HashSet<(u32, u32)> = HashSet::new();
    let mut counts = vec![0u32; n_items];

    for (&item_a, txns_a) in item_to_txns {
        // Count co-occurring items across all transactions of item_a
        for &txn in txns_a {
            if let Some(items) = txn_to_items.get(&txn) {
                for &item in items {
                    counts[item as usize] += 1;
                }
            }
        }

        // Extract canonical pairs (a < b only)
        for item_b_idx in 0..n_items {
            let co_count = counts[item_b_idx];
            counts[item_b_idx] = 0; // reset

            if co_count == 0 {
                continue;
            }
            let item_b = item_b_idx as u32;
            if item_b <= item_a {
                continue; // canonical: only a < b
            }

            if co_count >= min_count {
                let pair = (item_a, item_b);
                pair_counts.insert(pair, co_count);
                freq_pairs.insert(pair);
            }
        }
    }

    (pair_counts, freq_pairs)
}

/// Compute all pairwise (k=2) associations with 7 metrics.
///
/// Accepts integer-encoded arrays. Returns a Python dict of numpy arrays.
#[pyfunction]
pub fn rust_compute_pairs<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    n_items: u32,
    n_transactions: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;

    let (txn_to_items, item_to_txns, item_counts) =
        common::build_inverted_index(txn_slice, item_slice);

    let n_txn_f64 = n_transactions as f64;

    // Pre-compute item supports
    let mut item_support = vec![0.0_f64; n_items as usize];
    for (&item, &count) in &item_counts {
        item_support[item as usize] = count as f64 / n_txn_f64;
    }

    // Output buffers
    let mut out_item_a: Vec<i32> = Vec::new();
    let mut out_item_b: Vec<i32> = Vec::new();
    let mut out_instances: Vec<i64> = Vec::new();
    let mut out_support: Vec<f64> = Vec::new();
    let mut out_confidence: Vec<f64> = Vec::new();
    let mut out_lift: Vec<f64> = Vec::new();
    let mut out_conviction: Vec<f64> = Vec::new();
    let mut out_leverage: Vec<f64> = Vec::new();
    let mut out_cosine: Vec<f64> = Vec::new();
    let mut out_jaccard: Vec<f64> = Vec::new();

    // Reusable count buffer (flat array, size n_items)
    let mut counts = vec![0u32; n_items as usize];

    // For each item A, count co-occurrences with all other items
    for (&item_a, txns_a) in &item_to_txns {
        let count_a = item_counts[&item_a];
        let sup_a = item_support[item_a as usize];

        // Count co-occurring items across all transactions of A
        for &txn in txns_a {
            if let Some(items) = txn_to_items.get(&txn) {
                for &item in items {
                    counts[item as usize] += 1;
                }
            }
        }

        // Extract pairs and compute metrics
        for item_b in 0..n_items {
            let co_count = counts[item_b as usize];
            counts[item_b as usize] = 0; // reset for next iteration

            if co_count == 0 || item_b == item_a {
                continue;
            }

            let instances = co_count as f64;
            let sup_b = item_support[item_b as usize];
            let support = instances / n_txn_f64;
            let confidence = instances / count_a as f64;
            let lift = confidence / (sup_b + 1e-10);
            let conviction = (1.0 - sup_b) / (1.0 - confidence + 1e-10);
            let leverage = support - sup_a * sup_b;
            let cosine = support / (sup_a * sup_b).sqrt();
            let jaccard = support / (sup_a + sup_b - support);

            out_item_a.push(item_a as i32);
            out_item_b.push(item_b as i32);
            out_instances.push(co_count as i64);
            out_support.push(support);
            out_confidence.push(confidence);
            out_lift.push(lift);
            out_conviction.push(conviction);
            out_leverage.push(leverage);
            out_cosine.push(cosine);
            out_jaccard.push(jaccard);
        }
    }

    // Build return dict of numpy arrays
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
