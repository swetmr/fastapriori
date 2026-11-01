# Data sources

All inputs used by the paper except Instacart are in this folder in the archive at https://doi.org/10.5281/zenodo.23855457. `env/fetch_real_data.sh` rebuilds the eight
real datasets from their public sources. Each rebuilt file is **byte-identical** to the shipped one and
is verified against `SHA256SUMS`. Cite the primary sources, not this artifact.

## Real datasets (`data/real/`)

| Dataset (paper name) | File | Public source (exact file) | Preprocessing to the shipped file | Transactions / items |
|---|---|---|---|---:|
| Groceries | `groceries.csv` | https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv (the `Groceries` data of the R package `arules`) | none | 9,835 / 169 |
| BMS-WebView-1 | `BMS1.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/BMS1_itemset_mining.txt | none | 59,602 / 497 |
| BMS-WebView-2 | `BMS2.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/BMS2_itemset_mining.txt | none | 77,512 / 3,340 |
| Retail (Belgian) | `retail.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/retail.txt | none | 88,162 / 16,470 |
| Kosarak | `kosarak.dat` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/kosarak.dat | none | 990,002 / 41,270 |
| Chainstore | `chainstore.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/chainstoreFIM.txt | none (renamed) | 1,112,949 / 46,086 |
| Online Retail | `online-retail.csv` | UCI ML Repository, **Online Retail II** (dataset 502): https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip → `online_retail_II.xlsx` | sheet **"Year 2009-2010"** only (1 Dec 2009 – 9 Dec 2010, 525,461 rows), `Country` column dropped, exported as CSV (dates `%m/%d/%y %I:%M %p`, prices without trailing `.0`) | 28,816 / 4,632 |
| Instacart | `instacart_orders.csv` | Kaggle, [*Instacart Market Basket Analysis*](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) (`psparks/instacart-market-basket-analysis`) → `order_products__prior.csv` (SHA-256 `27b59dc9cf0d485f55e64e0bc9c5d6fb077ae167f8dfeab9246a03b5b0186359`) | columns `order_id,product_id` renamed `txn_id,item`, written with CRLF line endings | 3,214,874 / 49,677 |

How the harness reads them (`bench/load_datasets.py`):
- SPMF files: one transaction per line, space-separated integer items.
- Groceries: comma-separated item names.
- Online Retail: `Invoice` is the transaction and `StockCode` the item. Rows with missing values are
  dropped. Cancellations and repeated (invoice, item) rows are kept; the cost statistics drop duplicates.
- Instacart: the prior orders only.

Wrong or outdated sources seen elsewhere, so don't use these:
- UCI dataset 352, "Online Retail" (2010–2011, columns `InvoiceNo`/`UnitPrice`), is a different release.
- `spmf/datasets/<name>` URLs now return an HTTP 300 page.
- SPMF's `BMS1.txt` / `BMS2.txt` / `chainstore.txt` are sequence- or utility-format files, not the
  itemset files used here.

## Licences and citations

| Dataset | Terms | Cite |
|---|---|---|
| Groceries | Shipped with the R package `arules` | M. Hahsler, K. Hornik, T. Reutterer, "Implications of probabilistic data modeling for mining association rules", 2006 |
| BMS-WebView-1/2 | KDD Cup 2000 data, distributed for research use via SPMF | Z. Zheng, R. Kohavi, L. Mason, "Real world performance of association rule algorithms", KDD 2001; P. Fournier-Viger et al., SPMF |
| Retail | Distributed for research use via FIMI/SPMF | T. Brijs et al., "Using association rules for product assortment decisions: a case study", KDD 1999 |
| Kosarak | Anonymised click-stream (Hungarian news portal), contributed to FIMI by F. Bodon; via SPMF | FIMI repository; SPMF |
| Chainstore | From NU-MineBench, via SPMF | J. Pisharath et al., "NU-MineBench 2.0", Northwestern University tech. report, 2005; SPMF |
| Online Retail II | CC BY 4.0 (UCI) | D. Chen, "Online Retail II", UCI Machine Learning Repository, 2019, https://doi.org/10.24432/C5CG6D |
| Instacart | Instacart's dataset terms (via Kaggle; login required). **Check them before re-hosting**; this file is not shipped: download it from Kaggle and run `env/fetch_real_data.sh` | "The Instacart Online Grocery Shopping Dataset 2017" |

## Synthetic datasets (`data/synthetic/`)

These are the parquet files written in March–April 2026 by `bench/agrawal_generator.py`. Their
parameters and seeds are documented in `notebooks/01_synthetic_data.ipynb`, which regenerates the eight
smaller ones identically. The files are shipped because regenerating T25.D5M took about 10 hours.

## FIMI dense datasets (`data/generated/`)

These are used only by the optional notebook 14. Sources: chess, connect, mushroom (`mushrooms.txt`),
pumsb and accidents from `https://www.philippe-fournier-viger.com/spmf/publicdatasets/<name>.txt`, with
the FIMI repository as a fallback (`bench/exp_data.py::fetch_fimi`). Raw files and parquets are shipped
and checksummed.
