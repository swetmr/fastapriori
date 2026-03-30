use std::collections::HashMap;

/// Build inverted index from parallel arrays of (txn_id, item_id).
/// Returns (txn_to_items, item_to_txns, item_counts).
pub fn build_inverted_index(
    txn_ids: &[i64],
    item_ids: &[i32],
) -> (
    HashMap<i64, Vec<u32>>,
    HashMap<u32, Vec<i64>>,
    HashMap<u32, u32>,
) {
    let n = txn_ids.len();
    let mut txn_to_items: HashMap<i64, Vec<u32>> = HashMap::new();
    let mut item_to_txns: HashMap<u32, Vec<i64>> = HashMap::new();

    for i in 0..n {
        let txn = txn_ids[i];
        let item = item_ids[i] as u32;
        txn_to_items.entry(txn).or_default().push(item);
        item_to_txns.entry(item).or_default().push(txn);
    }

    // Deduplicate (in case of duplicate rows)
    for items in txn_to_items.values_mut() {
        items.sort_unstable();
        items.dedup();
    }
    for txns in item_to_txns.values_mut() {
        txns.sort_unstable();
        txns.dedup();
    }

    let item_counts: HashMap<u32, u32> = item_to_txns
        .iter()
        .map(|(&item, txns)| (item, txns.len() as u32))
        .collect();

    (txn_to_items, item_to_txns, item_counts)
}
