use std::collections::{HashMap, HashSet};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::common::{self, InvertedIndex};

/// Count frequent undirected pairs. Returns (freq_pair_counts, freq_pairs_set).
/// Only pairs with support >= min_support are returned.
/// Pairs are canonical: (a, b) where a < b.
///
/// Parallel scatter using Rayon fold/reduce with per-thread scratch. Each
/// canonical pair (a, b) with a < b is emitted by exactly one outer
/// iteration (item_a == a), so map reduction uses `or_insert` — no
/// summation, no double-count.
pub fn count_frequent_pairs(
    idx: &InvertedIndex,
    n_items: usize,
    n_transactions: u32,
    min_support: f64,
) -> (HashMap<(u32, u32), u32>, HashSet<(u32, u32)>) {
    let n_txn_f64 = n_transactions as f64;
    let min_count = if min_support > 0.0 {
        (min_support * n_txn_f64).ceil() as u64
    } else {
        1
    };

    // Only items with per-item count >= min_count can seed a frequent pair
    // on the larger end of the inequality (item_a == a, looking for b > a).
    // Filter once here — Instacart has ~50K items but ~3K frequent, so this
    // is a 16× reduction in outer-loop work.
    let items_vec: Vec<u32> = (0..idx.n_items)
        .filter(|&item| idx.item_counts[item as usize] as u64 >= min_count)
        .collect();

    struct LocalScratch {
        counts: Vec<u64>,
        dirty: Vec<u32>,
        out: HashMap<(u32, u32), u32>,
    }

    let pair_counts: HashMap<(u32, u32), u32> = items_vec
        .par_iter()
        .fold(
            || LocalScratch {
                counts: vec![0u64; n_items],
                dirty: Vec::with_capacity(256),
                out: HashMap::new(),
            },
            |mut scratch, &item_a| {
                let txns_a = idx.item_txns(item_a);
                for &txn_code in txns_a {
                    let items = idx.txn_items(txn_code);
                    for &item in items {
                        let idx_ = item as usize;
                        if scratch.counts[idx_] == 0 {
                            scratch.dirty.push(item);
                        }
                        scratch.counts[idx_] += 1;
                    }
                }

                for &item_b in &scratch.dirty {
                    let co_count = scratch.counts[item_b as usize];
                    scratch.counts[item_b as usize] = 0;

                    if item_b <= item_a {
                        continue;
                    }
                    if co_count >= min_count {
                        scratch.out.insert((item_a, item_b), co_count as u32);
                    }
                }
                scratch.dirty.clear();

                scratch
            },
        )
        .map(|scratch| scratch.out)
        .reduce(HashMap::new, |mut a, b| {
            a.reserve(b.len());
            for (k, v) in b {
                a.entry(k).or_insert(v);
            }
            a
        });

    let freq_pairs: HashSet<(u32, u32)> = pair_counts.keys().copied().collect();
    (pair_counts, freq_pairs)
}

/// Per-thread output buffers. Kept in a struct so the Rayon fold state stays
/// readable — 10 parallel Vec<T>s in a naked tuple is illegible.
#[derive(Default)]
struct PairOut {
    a: Vec<i32>,
    b: Vec<i32>,
    inst: Vec<i64>,
    sup: Vec<f64>,
    conf: Vec<f64>,
    lift: Vec<f64>,
    conv: Vec<f64>,
    lev: Vec<f64>,
    cos: Vec<f64>,
    jac: Vec<f64>,
}

impl PairOut {
    fn merge(mut self, other: PairOut) -> PairOut {
        self.a.extend(other.a);
        self.b.extend(other.b);
        self.inst.extend(other.inst);
        self.sup.extend(other.sup);
        self.conf.extend(other.conf);
        self.lift.extend(other.lift);
        self.conv.extend(other.conv);
        self.lev.extend(other.lev);
        self.cos.extend(other.cos);
        self.jac.extend(other.jac);
        self
    }
}

/// Per-thread scratch state. `counts` is reused across outer iterations within
/// the same thread — each iteration resets only the entries listed in `dirty`.
struct Scratch {
    out: PairOut,
    counts: Vec<u64>,
    dirty: Vec<u32>,
}

/// Compute all pairwise (k=2) associations with 7 metrics.
///
/// Accepts integer-encoded arrays. Returns a Python dict of numpy arrays.
///
/// `min_support` acts as an early cutoff on co-occurrence counts. When
/// positive, pairs with `co_count < ceil(min_support * n_transactions)` are
/// skipped before metrics are computed. `0.0` disables the filter.
///
/// `n_transactions` is derived from the CSR build (dense txn factorise),
/// so the caller does not pre-compute `.nunique()`.
#[pyfunction]
#[pyo3(signature = (txn_ids, item_ids, n_items, min_support=0.0))]
pub fn rust_compute_pairs<'py>(
    py: Python<'py>,
    txn_ids: PyReadonlyArray1<'py, i64>,
    item_ids: PyReadonlyArray1<'py, i32>,
    n_items: u32,
    min_support: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let txn_slice = txn_ids.as_slice()?;
    let item_slice = item_ids.as_slice()?;

    let (out, n_transactions): (PairOut, u32) = py.allow_threads(|| {
        let idx = common::build_inverted_index(txn_slice, item_slice, n_items);
        let n_transactions = idx.n_txns;
        let min_count: u64 = if min_support > 0.0 {
            (min_support * n_transactions as f64).ceil() as u64
        } else {
            0
        };

        let n_txn_f64 = n_transactions as f64;
        let mut item_support = vec![0.0_f64; n_items as usize];
        for i in 0..n_items as usize {
            item_support[i] = idx.item_counts[i] as f64 / n_txn_f64;
        }

        // Parallel outer loop: one item_a per iteration. Each item's
        // Rayon work cell needs only the idx slices — no locking.
        let n_items_usize = n_items as usize;
        let idx_ref = &idx;
        let item_support_ref = &item_support;

        let out: PairOut = (0..n_items)
            .into_par_iter()
            .fold(
                || Scratch {
                    out: PairOut::default(),
                    counts: vec![0u64; n_items_usize],
                    dirty: Vec::with_capacity(256),
                },
                |mut scratch, item_a| {
                    let count_a = idx_ref.item_counts[item_a as usize];
                    if count_a == 0 {
                        return scratch;
                    }
                    let sup_a = item_support_ref[item_a as usize];
                    let txns_a = idx_ref.item_txns(item_a);

                    for &txn_code in txns_a {
                        let items = idx_ref.txn_items(txn_code);
                        for &item in items {
                            let i = item as usize;
                            if scratch.counts[i] == 0 {
                                scratch.dirty.push(item);
                            }
                            scratch.counts[i] += 1;
                        }
                    }

                    for &item_b in &scratch.dirty {
                        let co_count = scratch.counts[item_b as usize];
                        scratch.counts[item_b as usize] = 0;

                        if item_b == item_a {
                            continue;
                        }
                        if co_count < min_count {
                            continue;
                        }

                        let instances = co_count as f64;
                        let sup_b = item_support_ref[item_b as usize];
                        let support = instances / n_txn_f64;
                        let confidence = instances / count_a as f64;
                        let lift = if sup_b > 0.0 {
                            confidence / sup_b
                        } else {
                            f64::INFINITY
                        };
                        let conviction = if sup_b >= 1.0 {
                            f64::NAN
                        } else if confidence >= 1.0 {
                            f64::INFINITY
                        } else {
                            (1.0 - sup_b) / (1.0 - confidence)
                        };
                        let leverage = support - sup_a * sup_b;
                        let cosine = support / (sup_a * sup_b).sqrt();
                        let jaccard = support / (sup_a + sup_b - support);

                        scratch.out.a.push(item_a as i32);
                        scratch.out.b.push(item_b as i32);
                        scratch.out.inst.push(co_count as i64);
                        scratch.out.sup.push(support);
                        scratch.out.conf.push(confidence);
                        scratch.out.lift.push(lift);
                        scratch.out.conv.push(conviction);
                        scratch.out.lev.push(leverage);
                        scratch.out.cos.push(cosine);
                        scratch.out.jac.push(jaccard);
                    }
                    scratch.dirty.clear();

                    scratch
                },
            )
            .map(|scratch| scratch.out)
            .reduce(PairOut::default, PairOut::merge);
        (out, n_transactions)
    });

    // Build return dict of numpy arrays
    let dict = PyDict::new(py);
    dict.set_item("item_A", PyArray1::from_vec(py, out.a))?;
    dict.set_item("item_B", PyArray1::from_vec(py, out.b))?;
    dict.set_item("instances", PyArray1::from_vec(py, out.inst))?;
    dict.set_item("support", PyArray1::from_vec(py, out.sup))?;
    dict.set_item("confidence", PyArray1::from_vec(py, out.conf))?;
    dict.set_item("lift", PyArray1::from_vec(py, out.lift))?;
    dict.set_item("conviction", PyArray1::from_vec(py, out.conv))?;
    dict.set_item("leverage", PyArray1::from_vec(py, out.lev))?;
    dict.set_item("cosine", PyArray1::from_vec(py, out.cos))?;
    dict.set_item("jaccard", PyArray1::from_vec(py, out.jac))?;

    // Expose the factorised txn count so the caller can skip `.nunique()`.
    dict.set_item("n_transactions", n_transactions as i64)?;

    Ok(dict)
}
