"""Agrawal & Srikant (1994) synthetic transaction data generator.

Extracted from runtime_prediction.ipynb for reuse by benchmark scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_agrawal_dataset(
    D: int,
    T: int,
    I: int,
    L: int,
    N: int,
    correlation: float = 0.5,
    corruption_mean: float = 0.5,
    corruption_std: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic transaction data using Agrawal & Srikant (1994) model.

    Parameters
    ----------
    D : int   — Number of transactions
    T : int   — Average transaction size (Poisson mean)
    I : int   — Average size of maximal potentially large itemsets
    L : int   — Number of maximal potentially large itemsets
    N : int   — Number of distinct items (item universe)
    correlation : float — Probability of picking items from previous itemset (0-1)
    corruption_mean : float — Mean of item drop probability (Normal dist)
    corruption_std  : float — Std of item drop probability
    seed : int — Random seed

    Returns
    -------
    pd.DataFrame with columns ['txn_id', 'item'] in long format.
    """
    rng = np.random.default_rng(seed)

    # Step 1: Generate L maximal potentially large itemsets
    itemsets = []
    for l_idx in range(L):
        size = max(1, rng.poisson(I))
        if l_idx == 0:
            items = rng.choice(N, size=min(size, N), replace=False)
        else:
            prev = itemsets[l_idx - 1]
            items = set()
            for _ in range(size):
                if rng.random() < correlation and len(prev) > 0:
                    items.add(rng.choice(prev))
                else:
                    items.add(rng.integers(0, N))
            items = np.array(sorted(items))
        itemsets.append(items)

    # Step 2: Assign exponential weights (Zipf-like skew)
    weights = rng.exponential(1.0, size=L)
    weights /= weights.sum()

    # Step 3: Generate D transactions
    txn_sizes = np.maximum(1, rng.poisson(T, size=D))

    all_txn_ids = []
    all_items = []

    for d in range(D):
        target = txn_sizes[d]
        txn_items: set = set()
        attempts = 0
        while len(txn_items) < target and attempts < target * 3:
            idx = rng.choice(L, p=weights)
            itemset = itemsets[idx]
            drop_prob = np.clip(
                rng.normal(corruption_mean, corruption_std), 0.0, 1.0
            )
            mask = rng.random(len(itemset)) > drop_prob
            surviving = itemset[mask]
            txn_items.update(surviving.tolist())
            attempts += 1
        txn_items_list = list(txn_items)[:target]
        for item in txn_items_list:
            all_txn_ids.append(d)
            all_items.append(item)

    return pd.DataFrame({"txn_id": all_txn_ids, "item": all_items})


def estimate_agrawal_params(
    n_transactions: int,
    n_items: int,
    d_avg: float,
    d_median: float | None = None,
    d_max: float | None = None,
    d_std: float | None = None,
) -> tuple[int, int]:
    """Estimate Agrawal model parameters I and L from observable dataset properties.

    Assumptions (calibrated against synthetic baseline T=10, I=4, L=2000, N=1000):
    - I ~ d_avg / 2.5  (after ~50% corruption, each pattern contributes ~I/2 items)
    - L ~ 2 * n_items   (baseline ratio L/N = 2)

    Returns (I_est, L_est).
    """
    I_est = int(max(2, round(d_avg / 2.5)))
    L_est = int(max(10, round(2 * n_items)))
    return I_est, L_est


def dataset_features(
    df: pd.DataFrame,
    txn_col: str = "txn_id",
    item_col: str = "item",
) -> dict:
    """Extract numeric features from a transaction dataset.

    Includes estimated Agrawal parameters I and L.
    """
    grp = df.groupby(txn_col)[item_col].nunique()
    n_transactions = df[txn_col].nunique()
    n_items = df[item_col].nunique()
    d_avg = grp.mean()
    d_median = grp.median()
    d_max = grp.max()
    d_std = grp.std()

    I_est, L_est = estimate_agrawal_params(
        n_transactions=n_transactions,
        n_items=n_items,
        d_avg=d_avg,
        d_median=d_median,
        d_max=d_max,
        d_std=d_std,
    )

    return {
        "n_transactions": n_transactions,
        "n_items": n_items,
        "n_rows": len(df),
        "d_avg": d_avg,
        "d_median": d_median,
        "d_max": d_max,
        "d_std": d_std,
        "I_est": I_est,
        "L_est": L_est,
    }
