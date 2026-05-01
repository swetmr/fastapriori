"""Load benchmark datasets from SPMF format into pandas DataFrames.

Each SPMF file has one transaction per line, with space-separated item IDs.
We convert to (txn_id, item) long-form DataFrames matching fastapriori's input format.

Datasets (downloaded from SPMF repository):
  - BMS-WebView-1:  59K txns,    497 items,   150K rows  (clickstream, very sparse)
  - BMS-WebView-2:  77K txns,  3,340 items,   358K rows  (clickstream, sparse)
  - Retail:         88K txns, 16,470 items,   909K rows  (Belgian retail, moderate)
  - Kosarak:       990K txns, 41,270 items, 8.0M rows    (Hungarian clickstream, large)
  - T10I4D100K:    100K txns,    870 items, 1.0M rows    (IBM synthetic benchmark)
  - Chainstore:   1.1M txns, 46,086 items, 8.0M rows    (US chain store, large)
"""

import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def load_spmf(filename: str) -> pd.DataFrame:
    """Load an SPMF-format file into a (txn_id, item) DataFrame."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    txn_ids = []
    items = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for txn_id, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            for item in parts:
                txn_ids.append(txn_id)
                items.append(int(item))

    return pd.DataFrame({"txn_id": txn_ids, "item": items})


def load_bms1() -> pd.DataFrame:
    """BMS-WebView-1: clickstream, 59K txns, 497 items."""
    return load_spmf("BMS1.txt")


def load_bms2() -> pd.DataFrame:
    """BMS-WebView-2: clickstream, 77K txns, 3340 items."""
    return load_spmf("BMS2.txt")


def load_retail() -> pd.DataFrame:
    """Belgian Retail: 88K txns, 16K items."""
    return load_spmf("retail.txt")


def load_kosarak() -> pd.DataFrame:
    """Kosarak: Hungarian clickstream, 990K txns, 41K items."""
    return load_spmf("kosarak.dat")


def load_t10i4d100k() -> pd.DataFrame:
    """T10I4D100K: IBM synthetic benchmark, 100K txns, 870 items."""
    return load_spmf("T10I4D100K.txt")


def load_chainstore() -> pd.DataFrame:
    """Chainstore: US chain store, 1.1M txns, 46K items."""
    return load_spmf("chainstore.txt")


def load_online_retail() -> pd.DataFrame:
    """Online Retail: UCI e-commerce, 25K txns, 4K items."""
    path = Path(__file__).parent / "online-retail.csv"
    if not path.exists():
        raise FileNotFoundError(f"Online Retail dataset not found: {path}")
    df = pd.read_csv(path)
    # Detect column names (may vary)
    txn_col = [c for c in df.columns if "invoice" in c.lower() or "txn" in c.lower()][0]
    item_col = [c for c in df.columns if "stock" in c.lower() or "item" in c.lower()][0]
    return df[[txn_col, item_col]].rename(columns={txn_col: "txn_id", item_col: "item"}).dropna()


def load_groceries() -> pd.DataFrame:
    """Groceries: classic arules dataset, 9835 txns, 169 items."""
    path = DATA_DIR / "groceries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Groceries dataset not found: {path}")
    txn_ids = []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for txn_id, line in enumerate(f):
            for item in line.strip().split(","):
                item = item.strip()
                if item:
                    txn_ids.append(txn_id)
                    items.append(item)
    return pd.DataFrame({"txn_id": txn_ids, "item": items})


def load_instacart() -> pd.DataFrame:
    """Instacart: 3.4M orders, 50K products, 32M rows.

    Requires Kaggle download:
        kaggle datasets download -d psparks/instacart-market-basket-analysis
    Then extract order_products__prior.csv into benchmarks/data/
    """
    import tarfile

    csv_path = DATA_DIR / "instacart_orders.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    # Try tar.gz if downloaded manually
    tar_path = DATA_DIR / "instacart.tar.gz"
    if tar_path.exists():
        print("Extracting Instacart data (first time only)...")
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if "order_products__prior" in member.name:
                    f = tar.extractfile(member)
                    if f is not None:
                        raw = pd.read_csv(f)
                        df = raw[["order_id", "product_id"]].rename(
                            columns={"order_id": "txn_id", "product_id": "item"}
                        )
                        df.to_csv(csv_path, index=False)
                        print(f"Saved {len(df):,} rows to {csv_path}")
                        return df

    # Try extracted CSV from Kaggle (flat or in instacart/ subfolder)
    prior_path = DATA_DIR / "order_products__prior.csv"
    if not prior_path.exists():
        prior_path = DATA_DIR / "instacart" / "order_products__prior.csv"
    if prior_path.exists():
        raw = pd.read_csv(prior_path)
        df = raw[["order_id", "product_id"]].rename(
            columns={"order_id": "txn_id", "product_id": "item"}
        )
        df.to_csv(csv_path, index=False)
        return df

    raise FileNotFoundError(
        "Instacart dataset not found. Download from Kaggle:\n"
        "  kaggle datasets download -d psparks/instacart-market-basket-analysis\n"
        "Then extract order_products__prior.csv to benchmarks/data/"
    )


ALL_DATASETS = {
    "Groceries": load_groceries,
    "BMS-WebView-1": load_bms1,
    "BMS-WebView-2": load_bms2,
    "T10I4D100K": load_t10i4d100k,
    "Retail (Belgian)": load_retail,
    "Online Retail": load_online_retail,
    "Kosarak": load_kosarak,
    "Chainstore": load_chainstore,
    "Instacart": load_instacart,
}


# -----------------------------------------------------------------------------
# Synthetic Agrawal-style datasets, stored as parquet next to this file.
#
# Drivers (e.g. driver_fastapriori_fast.py) load datasets via
# `ALL_DATASETS[DATASET]()` in a *spawned subprocess*, so the registry has to
# live here, not in the notebook's session. Each loader searches a few
# plausible directories so the same code works whether the parquets live in
# benchmarks/, benchmarks/data/, or the project root.
# -----------------------------------------------------------------------------
#
# Naming convention: dotted Agrawal parameter tag T<T>.I<I>.D<D>.N<N>.L<L>
# (matching the filename minus the df_ prefix and .parquet suffix). The two
# T15 datasets carry a `.low` / `.high` correlation suffix. These names flow
# through the raw / agg CSVs and into heatmap axis labels, so they must stay
# stable across runs.
_SYNTH_PARQUET_FILES = {
    "T3.I2.D100k.N50k.L50k":      "df_T3_I2_D100k_N50k_L50k.parquet",
    "T5.I2.D25k.N1k.L1k":         "df_T5_I2_D25k_N1k_L1k.parquet",
    "T10.I4.D100k.N25k.L25k":     "df_T10_I4_D100k_N25k_L25k.parquet",
    "T12.I4.D5k.N1k.L1k":         "df_T12_I4_D5k_N1k_L1k.parquet",
    "T15.I8.D50k.N1k.L1k.low":    "df_low_T15_I8_D50k_N1k_L1k.parquet",
    "T15.I8.D50k.N1k.L1k.high":   "df_high_T15_I8_D50k_N1k_L1k.parquet",
    "T20.I4.D1000k.N50k.L50k":    "df_T20_I4_D1000k_N50k_L50k.parquet",
    "T25.I8.D5000k.N40k.L40k":    "df_T25_I8_D5000k_N40k_L40k.parquet",
    "T30.I8.D50k.N5k.L5k":        "df_T30_I8_D50k_N5k_L5k.parquet",
    "T30.I8.D2000k.N10k.L10k":    "df_T30_I8_D2000k_N10k_L10k.parquet",
    "T50.I12.D10k.N3k.L3k":       "df_T50_I12_D10k_N3k_L3k.parquet",
}

_BENCH_DIR = Path(__file__).parent  # benchmarks/

def _resolve_synth_parquet(filename: str) -> Path:
    for d in (_BENCH_DIR, _BENCH_DIR / "data", _BENCH_DIR.parent / "benchmarks",
              _BENCH_DIR.parent / "benchmarks" / "data", _BENCH_DIR.parent):
        if not d.exists():
            continue
        p = d / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Synthetic parquet {filename!r} not found in any of: "
        f"{_BENCH_DIR}, {_BENCH_DIR/'data'}, {_BENCH_DIR.parent}"
    )

def _make_synth_loader(filename: str):
    def _load() -> pd.DataFrame:
        df = pd.read_parquet(_resolve_synth_parquet(filename))
        if {"txn_id", "item"} <= set(df.columns):
            df = df[["txn_id", "item"]].copy()
        return df
    _load.__name__ = f"load_{filename.replace('.parquet','')}"
    return _load

for _name, _fn in _SYNTH_PARQUET_FILES.items():
    ALL_DATASETS[_name] = _make_synth_loader(_fn)


if __name__ == "__main__":
    for name, loader in ALL_DATASETS.items():
        try:
            df = loader()
            n_txn = df["txn_id"].nunique()
            n_items = df["item"].nunique()
            avg = len(df) / n_txn
            print(f"{name:20s} | {len(df):>10,} rows | {n_txn:>10,} txns | {n_items:>6,} items | avg {avg:.1f}/txn")
        except Exception as e:
            print(f"{name:20s} | ERROR: {e}")
