//! Eclat vertical tid-list recursion for k>=3.
//!
//! Produces the same output-dict schema as `pipeline::rust_compute_pipeline`
//! (keys: itemsets, counts, lower_itemsets, lower_counts, item_counts) so the
//! Python decoder in `decode_pipeline_rules` is reused unchanged.
//!
//! Algorithm: textbook Eclat (Zaki 1997) with prefix-based equivalence
//! classes. The top-level loop over frequent singletons is Rayon-parallel;
//! each task owns an independent prefix subtree and emits to thread-local
//! buffers that are concatenated at the end. `common::build_inverted_index`
//! already returns sorted-deduped tid-lists, so no vertical-representation
//! build is needed here.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::common;

/// Linear-merge intersection of two sorted-ascending i64 slices.
#[inline]
fn intersect_sorted(a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (ai, bj) = (a[i], b[j]);
        if ai == bj {
            out.push(ai);
            i += 1;
            j += 1;
        } else if ai < bj {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

#[derive(Default)]
struct EclatOut {
    // k_max-level itemsets + their counts
    final_itemsets: Vec<Vec<u32>>,
    final_counts: Vec<u32>,
    // (k_max-1)-level itemsets + their counts
    lower_itemsets: Vec<Vec<u32>>,
    lower_counts: Vec<u32>,
}

impl EclatOut {
    fn merge(mut self, other: EclatOut) -> EclatOut {
        self.final_itemsets.extend(other.final_itemsets);
        self.final_counts.extend(other.final_counts);
        self.lower_itemsets.extend(other.lower_itemsets);
        self.lower_counts.extend(other.lower_counts);
        self
    }
}

/// Extend a prefix by each candidate extension; recurse on survivors.
///
/// `prefix` has length `depth`. Each element of `extensions` is `(item, tids)`
/// where `tids = tid(prefix ∪ {item})` — already intersected at this level
/// and already >= `min_count`. Produces `(depth+1)`-itemsets.
fn extend(
    prefix: &mut Vec<u32>,
    extensions: &[(u32, Vec<i64>)],
    depth: usize,
    k_max: usize,
    min_count: u32,
    out: &mut EclatOut,
) {
    for (i, (item_i, tid_i)) in extensions.iter().enumerate() {
        let new_level = depth + 1;
        let count = tid_i.len() as u32;

        prefix.push(*item_i);

        if new_level == k_max {
            out.final_itemsets.push(prefix.clone());
            out.final_counts.push(count);
            prefix.pop();
            continue;
        }

        if new_level + 1 == k_max {
            // new_level == k_max - 1: this is a (k-1)-itemset
            out.lower_itemsets.push(prefix.clone());
            out.lower_counts.push(count);
        }

        // Build next-level extensions: intersect tid_i with each later tid_j.
        let mut next_ext: Vec<(u32, Vec<i64>)> = Vec::new();
        for j in (i + 1)..extensions.len() {
            let (item_j, tid_j) = &extensions[j];
            let inter = intersect_sorted(tid_i, tid_j);
            if (inter.len() as u32) >= min_count {
                next_ext.push((*item_j, inter));
            }
        }

        if !next_ext.is_empty() {
            extend(prefix, &next_ext, new_level, k_max, min_count, out);
        }

        prefix.pop();
    }
}

/// Eclat pipeline: k=2 through k_max in one recursive pass.
///
/// Returns a Python dict shaped identically to
/// `pipeline::rust_compute_pipeline` so the Python-side decoder works
/// unchanged. Does not accept `max_items_per_txn` / `item_weights` — those
/// are Apriori-path concerns and are rejected by the Python dispatcher
/// before reaching this function.
#[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, k_max, n_items, min_support))]
pub fn rust_eclat_pipeline<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    k_max: usize,
    n_items: u32,
    min_support: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if k_max < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rust_eclat_pipeline requires k_max >= 2",
        ));
    }

    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;

    // Step 1: CSR inverted index. `idx.item_txns(item)` is the vertical
    // tid-list (dense u32 txn codes), sorted ascending + deduped.
    let idx = common::build_inverted_index(txn_slice, item_slice, n_items);
    let n_transactions = idx.n_txns;

    let n_txn_f64 = n_transactions as f64;
    let min_count: u32 = if min_support > 0.0 {
        (min_support * n_txn_f64).ceil() as u32
    } else {
        1
    };

    // Step 2: frequent singletons, sorted ascending by item id. Each tid-list
    // is widened u32 → i64 so existing `intersect_sorted` / `extend` logic is
    // untouched; tid values are equality labels only (no round-trip needed).
    let mut frequent: Vec<(u32, Vec<i64>)> = (0..n_items)
        .filter_map(|item| {
            let tids = idx.item_txns(item);
            if (tids.len() as u32) >= min_count {
                Some((item, tids.iter().map(|&c| c as i64).collect()))
            } else {
                None
            }
        })
        .collect();
    frequent.sort_by_key(|(item, _)| *item);
    let n_frequent = frequent.len();
    let frequent_ref = &frequent;

    // Step 3: parallelise the top-level singleton loop.  Each task owns its
    // own prefix subtree (no shared mutable state); thread-local EclatOut
    // buffers are concatenated at the end.
    let out: EclatOut = py.allow_threads(|| {
        (0..n_frequent)
            .into_par_iter()
            .fold(EclatOut::default, |mut local, i| {
                let (item_i, tid_i) = &frequent_ref[i];
                // First-level extensions: (item_j, tid(item_i) ∩ tid(item_j))
                // where item_j > item_i (lex canonical).
                let mut extensions: Vec<(u32, Vec<i64>)> = Vec::new();
                for j in (i + 1)..n_frequent {
                    let (item_j, tid_j) = &frequent_ref[j];
                    let inter = intersect_sorted(tid_i, tid_j);
                    if (inter.len() as u32) >= min_count {
                        extensions.push((*item_j, inter));
                    }
                }

                if !extensions.is_empty() {
                    let mut prefix = vec![*item_i];
                    extend(
                        &mut prefix,
                        &extensions,
                        1,
                        k_max,
                        min_count,
                        &mut local,
                    );
                }
                local
            })
            .reduce(EclatOut::default, EclatOut::merge)
    });

    // Step 4: build output dict, shaped identically to rust_compute_pipeline.
    let dict = PyDict::new(py);

    // Final (k_max) level
    let itemsets_vecs: Vec<Vec<i32>> = out
        .final_itemsets
        .iter()
        .map(|kset| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let counts_vec: Vec<i64> = out.final_counts.iter().map(|&c| c as i64).collect();

    dict.set_item("itemsets", PyArray2::from_vec2(py, &itemsets_vecs)?)?;
    dict.set_item("counts", PyArray1::from_vec(py, counts_vec))?;

    // (k_max - 1) level
    let lower_vecs: Vec<Vec<i32>> = out
        .lower_itemsets
        .iter()
        .map(|kset| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let lower_counts_vec: Vec<i64> =
        out.lower_counts.iter().map(|&c| c as i64).collect();

    dict.set_item("lower_itemsets", PyArray2::from_vec2(py, &lower_vecs)?)?;
    dict.set_item("lower_counts", PyArray1::from_vec(py, lower_counts_vec))?;

    // Per-item transaction counts (indexed by item id). idx.item_counts is
    // already a dense Vec<u32>; just widen to i64 for Python.
    let item_counts_arr: Vec<i64> =
        idx.item_counts.iter().map(|&c| c as i64).collect();
    dict.set_item("item_counts", PyArray1::from_vec(py, item_counts_arr))?;

    // Return n_transactions so the Python caller can skip `.nunique()`.
    dict.set_item("n_transactions", n_transactions as i64)?;

    Ok(dict)
}
