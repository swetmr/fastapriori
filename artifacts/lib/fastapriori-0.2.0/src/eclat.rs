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
        common::min_count_for(min_support, n_txn_f64) as u32
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

// ---------------------------------------------------------------------------
// Scatter-seeded Eclat (0.2.0, opt-in).
//
// `rust_eclat_pipeline` builds its first level by intersecting tid(i) with
// tid(j) for every j > i among the frequent singletons: C(|F1|, 2)
// candidate intersections, which is exactly the operator the pairwise cost
// model says loses at low support. Here each top-level task first runs a
// suffix-only sparse scatter over tid(i) to find its frequent partners
// j > i, then intersects only those |F2| pairs to obtain the tid-lists the
// depth-first recursion needs. Levels >= 3 are unchanged Eclat.
// ---------------------------------------------------------------------------

/// Size ratio above which intersection gallops through the longer list.
const GALLOP_RATIO: usize = 32;

/// Intersection of two sorted u32 slices into `out`.
#[inline]
fn intersect_u32(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if small.is_empty() {
        return;
    }
    if large.len() / small.len() >= GALLOP_RATIO {
        let mut lo = 0usize;
        for &x in small {
            lo += large[lo..].partition_point(|&y| y < x);
            if lo == large.len() {
                break;
            }
            if large[lo] == x {
                out.push(x);
                lo += 1;
            }
        }
    } else {
        let (mut i, mut j) = (0usize, 0usize);
        while i < small.len() && j < large.len() {
            let (x, y) = (small[i], large[j]);
            if x == y {
                out.push(x);
                i += 1;
                j += 1;
            } else if x < y {
                i += 1;
            } else {
                j += 1;
            }
        }
    }
}

/// |a ∩ b| without materialising it. Stops early once the remaining
/// elements of the shorter list cannot lift the count to `min_count`.
#[inline]
fn intersect_count_u32(a: &[u32], b: &[u32], min_count: u32) -> u32 {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if (small.len() as u32) < min_count {
        return 0;
    }
    let mut count = 0u32;
    if large.len() / small.len().max(1) >= GALLOP_RATIO {
        let mut lo = 0usize;
        for (pos, &x) in small.iter().enumerate() {
            if count + ((small.len() - pos) as u32) < min_count {
                return 0;
            }
            lo += large[lo..].partition_point(|&y| y < x);
            if lo == large.len() {
                break;
            }
            if large[lo] == x {
                count += 1;
                lo += 1;
            }
        }
    } else {
        let (mut i, mut j) = (0usize, 0usize);
        while i < small.len() && j < large.len() {
            if count + ((small.len() - i) as u32) < min_count {
                return 0;
            }
            let (x, y) = (small[i], large[j]);
            if x == y {
                count += 1;
                i += 1;
                j += 1;
            } else if x < y {
                i += 1;
            } else {
                j += 1;
            }
        }
    }
    count
}

/// `extend` over u32 tid-lists. When the children of a node are the final
/// level, only their counts are needed, so they are intersected count-only.
fn extend_u32(
    prefix: &mut Vec<u32>,
    extensions: &[(u32, Vec<u32>)],
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
            out.lower_itemsets.push(prefix.clone());
            out.lower_counts.push(count);
            for (item_j, tid_j) in &extensions[(i + 1)..] {
                let c = intersect_count_u32(tid_i, tid_j, min_count);
                if c >= min_count {
                    let mut s = prefix.clone();
                    s.push(*item_j);
                    out.final_itemsets.push(s);
                    out.final_counts.push(c);
                }
            }
            prefix.pop();
            continue;
        }

        let mut next_ext: Vec<(u32, Vec<u32>)> = Vec::new();
        for (item_j, tid_j) in &extensions[(i + 1)..] {
            let mut inter = Vec::new();
            intersect_u32(tid_i, tid_j, &mut inter);
            if (inter.len() as u32) >= min_count {
                next_ext.push((*item_j, inter));
            }
        }
        if !next_ext.is_empty() {
            extend_u32(prefix, &next_ext, new_level, k_max, min_count, out);
        }
        prefix.pop();
    }
}

fn eclat_out_to_dict<'py>(
    py: Python<'py>,
    out: EclatOut,
    item_counts: &[u32],
    n_transactions: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let itemsets_vecs: Vec<Vec<i32>> = out
        .final_itemsets
        .iter()
        .map(|kset| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let counts_vec: Vec<i64> = out.final_counts.iter().map(|&c| c as i64).collect();
    dict.set_item("itemsets", PyArray2::from_vec2(py, &itemsets_vecs)?)?;
    dict.set_item("counts", PyArray1::from_vec(py, counts_vec))?;

    let lower_vecs: Vec<Vec<i32>> = out
        .lower_itemsets
        .iter()
        .map(|kset| kset.iter().map(|&x| x as i32).collect())
        .collect();
    let lower_counts_vec: Vec<i64> =
        out.lower_counts.iter().map(|&c| c as i64).collect();
    dict.set_item("lower_itemsets", PyArray2::from_vec2(py, &lower_vecs)?)?;
    dict.set_item("lower_counts", PyArray1::from_vec(py, lower_counts_vec))?;

    let item_counts_arr: Vec<i64> = item_counts.iter().map(|&c| c as i64).collect();
    dict.set_item("item_counts", PyArray1::from_vec(py, item_counts_arr))?;
    dict.set_item("n_transactions", n_transactions as i64)?;
    Ok(dict)
}

/// Eclat with a sparse-scatter first level. Same output dict as
/// `rust_eclat_pipeline`.
#[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, k_max, n_items, min_support))]
pub fn rust_eclat_pipeline_seeded<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    k_max: usize,
    n_items: u32,
    min_support: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if k_max < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rust_eclat_pipeline_seeded requires k_max >= 2",
        ));
    }

    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;
    let n_items_usize = n_items as usize;

    // Recode items by ascending frequency (ties by id), the Borgelt/Zaki
    // Eclat order: prefix classes are then rooted at rare items with short
    // tid-lists instead of at whichever popular item has a small label.
    // Row counts are only an ordering heuristic; correctness does not
    // depend on them.
    let mut row_counts = vec![0u64; n_items_usize];
    for &it in item_slice {
        row_counts[it as usize] += 1;
    }
    let mut order: Vec<u32> = (0..n_items).collect(); // order[rank] = original id
    order.sort_unstable_by_key(|&i| (row_counts[i as usize], i));
    let mut rank_of = vec![0i32; n_items_usize];
    for (r, &orig) in order.iter().enumerate() {
        rank_of[orig as usize] = r as i32;
    }
    let recoded: Vec<i32> = item_slice.iter().map(|&it| rank_of[it as usize]).collect();
    let idx = common::build_inverted_index(txn_slice, &recoded, n_items);
    drop(recoded);
    let n_transactions = idx.n_txns;
    let min_count: u32 = if min_support > 0.0 {
        common::min_count_for(min_support, n_transactions as f64) as u32
    } else {
        1
    };

    let frequent_item: Vec<bool> = (0..n_items_usize)
        .map(|i| idx.item_counts[i] >= min_count.max(1))
        .collect();
    let frequent: Vec<u32> = (0..n_items).filter(|&i| frequent_item[i as usize]).collect();

    let idx_ref = &idx;
    let frequent_item_ref = &frequent_item;
    let frequent_ref = &frequent;

    struct SeedScratch {
        out: EclatOut,
        counts: Vec<u32>,
        dirty: Vec<u32>,
    }

    let out: EclatOut = py.allow_threads(|| {
        frequent_ref
            .par_iter()
            .fold(
                || SeedScratch {
                    out: EclatOut::default(),
                    counts: vec![0u32; n_items_usize],
                    dirty: Vec::with_capacity(256),
                },
                |mut s, &item_i| {
                    let tids_i = idx_ref.item_txns(item_i);
                    // Suffix-only scatter: co-occurrence counts for j > i.
                    for &txn_code in tids_i {
                        let items = idx_ref.txn_items(txn_code);
                        let start = items.partition_point(|&x| x <= item_i);
                        for &j in &items[start..] {
                            let ju = j as usize;
                            if !frequent_item_ref[ju] {
                                continue;
                            }
                            if s.counts[ju] == 0 {
                                s.dirty.push(j);
                            }
                            s.counts[ju] += 1;
                        }
                    }
                    let mut partners: Vec<(u32, u32)> = Vec::new();
                    for &j in &s.dirty {
                        let c = s.counts[j as usize];
                        s.counts[j as usize] = 0;
                        if c >= min_count {
                            partners.push((j, c));
                        }
                    }
                    s.dirty.clear();
                    if partners.is_empty() {
                        return s;
                    }
                    partners.sort_unstable_by_key(|&(j, _)| j);

                    if k_max == 2 {
                        // Counts are exact; no tid-lists needed.
                        for (j, c) in partners {
                            s.out.final_itemsets.push(vec![item_i, j]);
                            s.out.final_counts.push(c);
                        }
                        return s;
                    }

                    // tid(i) ∩ tid(j) straight from the CSR slices (no copies).
                    let extensions: Vec<(u32, Vec<u32>)> = partners
                        .iter()
                        .map(|&(j, c)| {
                            let mut tids = Vec::with_capacity(c as usize);
                            intersect_u32(tids_i, idx_ref.item_txns(j), &mut tids);
                            (j, tids)
                        })
                        .collect();
                    let mut prefix = vec![item_i];
                    extend_u32(&mut prefix, &extensions, 1, k_max, min_count, &mut s.out);
                    s
                },
            )
            .map(|s| s.out)
            .reduce(EclatOut::default, EclatOut::merge)
    });

    // Back to original ids; the decoder expects each itemset sorted ascending.
    let decode = |sets: &mut Vec<Vec<u32>>| {
        for s in sets.iter_mut() {
            for x in s.iter_mut() {
                *x = order[*x as usize];
            }
            s.sort_unstable();
        }
    };
    let mut out = out;
    decode(&mut out.final_itemsets);
    decode(&mut out.lower_itemsets);
    let mut item_counts = vec![0u32; n_items_usize];
    for (r, &c) in idx.item_counts.iter().enumerate() {
        item_counts[order[r] as usize] = c;
    }
    eclat_out_to_dict(py, out, &item_counts, n_transactions)
}
