"""One measured run, executed in a fresh process by exp_runners.run_once.

Usage: python exp_driver.py <spec.json>

The spec names the method kind, the input file and the parameters. Data are
loaded BEFORE the timer, as in bench/baseline_runners.py; the timer covers only the
library call (the HybridFIM call includes encoding, index build and output
decode). Writes {"n_itemsets", "inner_wall_s", ...} to spec["out_json"].

kind = "hybcli" is the memory-boundary mode: numpy only, no pandas, the Rust
extension loaded straight from its file so the fastapriori package (and
pandas) are never imported. With "load_only" it exits after loading the
arrays, giving the interpreter + input baseline.
"""
import json
import sys
import time


def _load_ext(path):
    import importlib.machinery
    import importlib.util
    loader = importlib.machinery.ExtensionFileLoader("_fastapriori_rs", path)
    spec = importlib.util.spec_from_file_location("_fastapriori_rs", path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def main():
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    kind, k, s = spec["kind"], int(spec["k"]), float(spec["min_support"])
    res = {}

    if kind == "hybcli":
        import numpy as np
        z = np.load(spec["npz"])
        txn, items, n_items = z["txn_ids"], z["item_ids"], int(z["n_items"])
        if not spec.get("load_only"):
            rs = _load_ext(spec["ext_path"])
            fn = spec["fn"]
            if fn.startswith("pairs") and k != 2:
                raise ValueError(f"{fn} is a k=2 kernel, got k={k}")
            if fn == "anchor" and k < 3:
                raise ValueError("anchor-and-extend is the k>=3 kernel; use pairs_* at k=2")
            t0 = time.perf_counter()
            if fn == "pairs_v0":
                out = rs.rust_compute_pairs(txn, items, n_items, s)
                n = len(out["item_A"]) // 2
            elif fn == "pairs_v2":
                out = rs.rust_compute_pairs_v2(txn, items, n_items, s, suffix=True,
                                               filter_infrequent=True)
                n = len(out["item_A"]) // 2
            elif fn == "anchor":
                out = rs.rust_compute_pipeline_v3_adaptive(
                    txn, items, k, n_items, s, np.empty(0, dtype=np.float64), None)
                n = len(out["counts"])
            elif fn == "seeded":
                out = rs.rust_eclat_pipeline_seeded(txn, items, k, n_items, s)
                n = len(out["counts"])
            else:
                raise ValueError(fn)
            res["inner_wall_s"] = time.perf_counter() - t0
            res["n_itemsets"] = int(n)
        json.dump(res, open(spec["out_json"], "w"))
        return

    sys.path.insert(0, spec["bench_dir"])
    import pandas as pd
    df = pd.read_parquet(spec["parquet"])
    txn_col, item_col = df.columns[0], df.columns[1]

    if kind in ("hyb", "classic"):
        from fastapriori import find_associations
        t0 = time.perf_counter()
        r = find_associations(
            df, transaction_col=txn_col, item_col=item_col, k=k,
            min_support=s, min_confidence=0.0, backend="auto",
            algo="classic" if kind == "classic" else "fast",
            sorted_by=None, low_memory=False,
            backend_options=spec.get("backend_options"),
        )
        res["inner_wall_s"] = time.perf_counter() - t0
        # Count distinct itemsets, not rules/k: the default metric filters
        # (min_conviction=0.0) drop rules whose consequent is in every
        # transaction (conviction undefined), so rule rows per itemset vary.
        cols = (["item_A", "item_B"] if k == 2
                else [f"antecedent_{i}" for i in range(1, k)] + ["consequent"])
        import numpy as np
        arr = np.sort(r[cols].to_numpy(), axis=1)   # copy: to_numpy may be read-only
        res["n_itemsets"] = int(len(pd.DataFrame(arr).drop_duplicates()))
    elif kind == "pyfim":
        from fim import apriori, eclat, fpgrowth
        fn = {"apriori": apriori, "eclat": eclat, "fpgrowth": fpgrowth}[spec["algo"]]
        transactions = [tuple(g) for _, g in df.groupby(txn_col, sort=False)[item_col]]
        t0 = time.perf_counter()
        r = fn(transactions, target="s", supp=s * 100.0, zmin=k, zmax=k, report="a")
        res["inner_wall_s"] = time.perf_counter() - t0
        res["n_itemsets"] = int(len(r))
    else:
        raise ValueError(kind)
    json.dump(res, open(spec["out_json"], "w"))


if __name__ == "__main__":
    main()
