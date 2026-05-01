# Benchmark datasets

The benchmark notebook (`benchmarks/rigorous_baseline_benchmark_py2_1t.ipynb`) and `benchmarks/load_datasets.py` expect the raw files below to live in `benchmarks/data/`. None of these datasets are redistributed in this repo — please download from the canonical source listed and place each file at the indicated path.

| Dataset | Source | Download | Place at |
|---|---|---|---|
| Groceries | R `arules` package | https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv | `benchmarks/data/groceries.csv` |
| BMS-WebView-1 | SPMF (KDD Cup 2000) | https://www.philippe-fournier-viger.com/spmf/datasets/BMS1.txt | `benchmarks/data/BMS1.txt` |
| BMS-WebView-2 | SPMF (KDD Cup 2000) | https://www.philippe-fournier-viger.com/spmf/datasets/BMS2.txt | `benchmarks/data/BMS2.txt` |
| T10I4D100K | SPMF (IBM synthetic) | https://www.philippe-fournier-viger.com/spmf/datasets/T10I4D100K.txt | `benchmarks/data/T10I4D100K.txt` |
| Retail (Belgian) | SPMF (Brijs et al., 1999) | https://www.philippe-fournier-viger.com/spmf/datasets/retail.txt | `benchmarks/data/retail.txt` |
| Online Retail | UCI ML Repository | https://archive.ics.uci.edu/dataset/352/online+retail (export as `online-retail.csv`) | `benchmarks/online-retail.csv` |
| Kosarak | SPMF (Hungarian news portal) | https://www.philippe-fournier-viger.com/spmf/datasets/kosarak.dat | `benchmarks/data/kosarak.dat` |
| Chainstore | SPMF (NU-MineBench) | https://www.philippe-fournier-viger.com/spmf/datasets/chainstore.txt | `benchmarks/data/chainstore.txt` |
| Instacart | Kaggle | https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis (extract `order_products__prior.csv`) | `benchmarks/data/order_products__prior.csv` |

## File formats

- **SPMF format** (BMS1, BMS2, T10I4D100K, retail, kosarak, chainstore): one transaction per line, space-separated integer item IDs.
- **Groceries**: one transaction per line, comma-separated item names.
- **Online Retail**: standard CSV with `InvoiceNo` and `StockCode` columns.
- **Instacart**: standard CSV with `order_id` and `product_id` columns.

## Approximate sizes

| Dataset | Disk | Transactions | Items |
|---|---:|---:|---:|
| Groceries | 0.5 MB | 9,835 | 169 |
| BMS-WebView-1 | 1 MB | 59,602 | 497 |
| BMS-WebView-2 | 2 MB | 77,512 | 3,340 |
| T10I4D100K | 4 MB | 100,000 | 870 |
| Retail (Belgian) | 4 MB | 88,162 | 16,470 |
| Online Retail | 25 MB | 28,816 | 4,632 |
| Kosarak | 33 MB | 990,002 | 41,270 |
| Chainstore | 47 MB | 1,112,949 | 46,086 |
| Instacart | 470 MB | 3,214,874 | 49,677 |

## Citation

If you publish results using these datasets, please cite the original sources. Most of the SPMF datasets ship with a citation note in the SPMF dataset page; the Online Retail and Instacart pages list their preferred citations directly.
