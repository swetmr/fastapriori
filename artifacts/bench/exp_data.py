"""Datasets for the October 2026 experiments (notebooks 10-19).

* `register_parquet` / `load_df` -- a name -> parquet registry layered on top
  of `bench/load_datasets.ALL_DATASETS`, persisted in
  `data/generated/registry.json` so that spawned driver processes resolve the
  same names.
* `fetch_fimi` -- dense FIMI benchmark datasets (chess, connect, mushroom,
  pumsb, accidents) from the SPMF mirror, FIMI repository as fallback.
* `quest` -- vectorised generator for the Agrawal-Srikant (Quest) model
  used by `bench/agrawal_generator.py`. Same generative model, not the
  same random stream: sweep datasets are internally consistent with each
  other, not bit-identical to the shipped parquets in data/synthetic/.
* `inject_tail` -- raises d_max at fixed D and fixed sum_t |T_t| (so d_avg is
  fixed) by widening a few transactions and thinning the rest.
* `cost_stats` -- every statistic the cost model needs (|F1|, S2, rho, P1).

All loaders return a DataFrame with columns ["txn_id", "item"].
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# [artifact patch] harness modules are all in bench/; FIMI and generated sweep datasets
# (and their registry) live in <bundle>/data/generated (override: FA_GENERATED_DIR).
import os
HARNESS_DIR = Path(__file__).resolve().parent
BENCH_DIR = HARNESS_DIR
DATA_DIR = Path(os.environ.get("FA_GENERATED_DIR", HARNESS_DIR.parent / "data" / "generated"))
REGISTRY = DATA_DIR / "registry.json"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from load_datasets import ALL_DATASETS  # noqa: E402

# Short names used in the paper -> registry names in load_datasets.
PAPER_NAMES = {
    "Groceries": "Groceries", "BMS-WebView-1": "BMS-WebView-1",
    "BMS-WebView-2": "BMS-WebView-2", "Retail": "Retail (Belgian)",
    "Online Retail": "Online Retail", "Kosarak": "Kosarak",
    "Chainstore": "Chainstore", "Instacart": "Instacart",
    "T3.D100k": "T3.I2.D100k.N50k.L50k", "T5.D25k": "T5.I2.D25k.N1k.L1k",
    "T10.D100k": "T10.I4.D100k.N25k.L25k", "T12.D5k": "T12.I4.D5k.N1k.L1k",
    "T15.D50k.h": "T15.I8.D50k.N1k.L1k.high", "T15.D50k.l": "T15.I8.D50k.N1k.L1k.low",
    "T20.D1M": "T20.I4.D1000k.N50k.L50k", "T25.D5M": "T25.I8.D5000k.N40k.L40k",
    "T30.D2M": "T30.I8.D2000k.N10k.L10k", "T30.D50k": "T30.I8.D50k.N5k.L5k",
}
CORE_DATASETS = list(PAPER_NAMES.values())
SMALL_FIRST = [  # rough size order, for running cheap datasets first
    "Groceries", "T12.I4.D5k.N1k.L1k", "T5.I2.D25k.N1k.L1k", "BMS-WebView-1",
    "T3.I2.D100k.N50k.L50k", "BMS-WebView-2", "Online Retail",
    "T15.I8.D50k.N1k.L1k.high", "T15.I8.D50k.N1k.L1k.low", "Retail (Belgian)",
    "T10.I4.D100k.N25k.L25k", "T30.I8.D50k.N5k.L5k", "Kosarak", "Chainstore",
    "T20.I4.D1000k.N50k.L50k", "Instacart", "T30.I8.D2000k.N10k.L10k",
    "T25.I8.D5000k.N40k.L40k",
]


# ----------------------------------------------------------------- registry --
def _read_registry() -> dict[str, str]:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {}


def register_parquet(name: str, df: pd.DataFrame, meta: dict | None = None) -> Path:
    """Store `df` (txn_id, item) as parquet under data/ and register `name`."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    path = DATA_DIR / f"{slug}.parquet"
    df[["txn_id", "item"]].to_parquet(path, index=False)
    reg = _read_registry()
    reg[name] = path.name
    REGISTRY.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    if meta is not None:
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return path


def is_registered(name: str) -> bool:
    return name in _read_registry()


def load_df(name: str) -> pd.DataFrame:
    """Load any of the 18 core datasets or a registered generated dataset as (txn_id, item)."""
    reg = _read_registry()
    if name in reg:
        return pd.read_parquet(DATA_DIR / reg[name])
    if name in ALL_DATASETS:
        df = ALL_DATASETS[name]()
        df = df.iloc[:, :2].copy()
        df.columns = ["txn_id", "item"]
        return df
    raise KeyError(f"unknown dataset {name!r}; registered: {sorted(reg)}")


# --------------------------------------------------------------------- FIMI --
FIMI_DENSE = {  # paper name -> file stem on the mirrors
    "chess": "chess", "connect": "connect", "mushroom": "mushrooms",
    "pumsb": "pumsb", "accidents": "accidents",
}
_MIRRORS = [
    "https://www.philippe-fournier-viger.com/spmf/publicdatasets/{stem}.txt",
    "http://fimi.uantwerpen.be/data/{fimi}.dat",
]


def fetch_fimi(name: str, timeout: int = 60) -> pd.DataFrame:
    """Download (once) and register a dense FIMI dataset; returns the frame.

    The SPMF server answers 403 to urllib's default User-Agent, so a browser
    style one is sent. Offline: place the file at data/raw/<name>.txt.
    """
    if is_registered(name):
        return load_df(name)
    stem = FIMI_DENSE[name]
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / f"{name}.txt"
    if not raw.exists():
        last = None
        for tpl in _MIRRORS:
            url = tpl.format(stem=stem, fimi=name)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) fastapriori-artifacts"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    raw.write_bytes(r.read())
                break
            except Exception as exc:  # try next mirror
                last = exc
        else:
            raise RuntimeError(f"could not download {name}: {last}")
    txn, items = [], []
    with raw.open() as f:
        for t, line in enumerate(f):
            toks = line.split()
            if not toks or not toks[0].lstrip("-").isdigit():
                continue
            for tok in dict.fromkeys(toks):  # dedupe, keep order
                txn.append(t)
                items.append(int(tok))
    df = pd.DataFrame({"txn_id": np.asarray(txn, np.int64), "item": np.asarray(items, np.int64)})
    register_parquet(name, df, meta={"source": "FIMI dense", "file": raw.name})
    return df


# -------------------------------------------------------------------- Quest --
def quest(D: int, T: float, I: float, L: int, N: int, correlation: float = 0.5,
          corruption_mean: float = 0.5, corruption_std: float = 0.1,
          seed: int = 42) -> pd.DataFrame:
    """Vectorised Agrawal-Srikant generator (model of agrawal_generator.py).

    Patterns: L itemsets of size Poisson(I), each sharing items with its
    predecessor with probability `correlation`; exponential pattern weights.
    Transactions: size Poisson(T) (>=1), filled with corrupted pattern
    instances (each item dropped with a per-instance probability drawn from
    N(corruption_mean, corruption_std)); the last instance is truncated to
    the target size and duplicates are removed, so realised sizes can be a
    little below target. Realised statistics are reported by `cost_stats`.
    """
    rng = np.random.default_rng(seed)
    pats: list[np.ndarray] = []
    for l_idx in range(L):
        size = max(1, rng.poisson(I))
        if l_idx == 0:
            items = rng.choice(N, size=min(size, N), replace=False)
        else:
            prev = pats[-1]
            take_prev = rng.random(size) < correlation
            n_prev = int(take_prev.sum())
            from_prev = rng.choice(prev, size=n_prev) if len(prev) and n_prev else np.empty(0, np.int64)
            fresh = rng.integers(0, N, size=size - n_prev)
            items = np.unique(np.concatenate([from_prev, fresh]))
        pats.append(np.asarray(items, dtype=np.int64))
    pat_len = np.array([len(p) for p in pats])
    pat_off = np.concatenate([[0], np.cumsum(pat_len)])
    pat_items = np.concatenate(pats)
    cumw = np.cumsum(rng.exponential(1.0, size=L))
    cumw /= cumw[-1]

    target = np.maximum(1, rng.poisson(T, size=D))
    # Instances: surviving length per instance, drawn in bulk.
    exp_surv = max(1e-9, pat_len.mean() * (1 - corruption_mean))
    n_inst = int(target.sum() / exp_surv * 1.3) + 1000
    out_txn, out_item = [], []
    t0 = 0
    while t0 < D:
        inst = np.searchsorted(cumw, rng.random(n_inst))
        drop = np.clip(rng.normal(corruption_mean, corruption_std, size=n_inst), 0.0, 1.0)
        lens = pat_len[inst]
        flat_inst = np.repeat(np.arange(n_inst), lens)
        flat_pos = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        flat_item = pat_items[pat_off[inst][flat_inst] + flat_pos]
        keep = rng.random(len(flat_item)) > drop[flat_inst]
        flat_inst, flat_item = flat_inst[keep], flat_item[keep]
        surv = np.bincount(flat_inst, minlength=n_inst)
        # Greedy assignment of whole instances to transactions (Python loop
        # over integers only; the item work above is vectorised).
        surv_l = surv.tolist()
        owner = np.full(n_inst, -1, dtype=np.int64)
        pos, t = 0, t0
        tgt = target.tolist()
        while t < D and pos < n_inst:
            acc, attempts, cap = 0, 0, 3 * tgt[t]
            start = pos
            while acc < tgt[t] and attempts < cap and pos < n_inst:
                acc += surv_l[pos]
                pos += 1
                attempts += 1
            if acc < tgt[t] and attempts < cap:  # ran out of instances
                pos = start
                break
            owner[start:pos] = t
            t += 1
        sel = owner[flat_inst] >= 0
        out_txn.append(owner[flat_inst][sel])
        out_item.append(flat_item[sel])
        t0 = t
    txn = np.concatenate(out_txn)
    item = np.concatenate(out_item)
    df = pd.DataFrame({"txn_id": txn, "item": item}).drop_duplicates()
    # Truncate each transaction to its target size (first-come order).
    rank = df.groupby("txn_id", sort=False).cumcount().to_numpy()
    df = df[rank < target[df["txn_id"].to_numpy()]]
    return df.reset_index(drop=True)


def inject_tail(df: pd.DataFrame, d_max: int, n_wide: int = 100,
                seed: int = 7) -> pd.DataFrame:
    """Widen `n_wide` transactions to widths geomspace(d_max/4, d_max) by
    merging in items of other random transactions, then delete the same
    number of rows uniformly from the remaining transactions (never emptying
    one). D and sum_t |T_t| (hence d_avg) are unchanged; d_max and d_CV rise.
    """
    rng = np.random.default_rng(seed)
    df = df[["txn_id", "item"]].drop_duplicates().reset_index(drop=True)
    groups = df.groupby("txn_id")["item"].apply(np.asarray)
    tids = groups.index.to_numpy()
    n_items = df["item"].nunique()
    widths = np.unique(np.round(np.geomspace(max(2, d_max / 4), d_max, n_wide)).astype(int))
    widths = np.resize(widths, n_wide)
    widths[-1] = d_max
    wide = rng.choice(tids, size=n_wide, replace=False)
    wide_set = set(wide.tolist())
    new_rows = []
    for t, w in zip(wide.tolist(), widths.tolist()):
        cur = dict.fromkeys(groups[t].tolist())
        w = min(w, n_items)
        while len(cur) < w:
            donor = groups[tids[rng.integers(len(tids))]]
            for x in donor.tolist():
                cur.setdefault(x, None)
                if len(cur) >= w:
                    break
        new_rows.append(pd.DataFrame({"txn_id": t, "item": list(cur)}))
    rest = df[~df["txn_id"].isin(wide_set)]
    added = sum(len(r) for r in new_rows) - int(df["txn_id"].isin(wide_set).sum())
    rank = rest.groupby("txn_id", sort=False).cumcount().to_numpy()
    eligible = np.flatnonzero(rank >= 1)  # keep the first item of every txn
    if added > len(eligible):
        raise ValueError("tail too large for the base dataset; lower d_max or n_wide")
    drop = rng.choice(eligible, size=max(0, added), replace=False)
    rest = rest.drop(rest.index[drop])
    out = pd.concat([rest] + new_rows, ignore_index=True)
    return out.sort_values("txn_id", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- statistics --
def cost_stats(df: pd.DataFrame, supports=(1e-4, 5e-4, 1e-3, 5e-3, 1e-2)) -> list[dict]:
    """Cost-model statistics per support: |F1|, S2 (raw and after singleton
    filtering), rho = (|F1|^2/2)/S2, and the P1 ratio (|F1|-1)/(d_avg(1+cv^2))."""
    d = df[["txn_id", "item"]].drop_duplicates()
    sizes = d.groupby("txn_id").size()
    D = int(len(sizes))
    item_cnt = d.groupby("item").size()
    davg, sd, dmax = float(sizes.mean()), float(sizes.std(ddof=0)), int(sizes.max())
    cv = sd / davg if davg else float("nan")
    s2_raw = float((sizes.astype(np.float64) ** 2).sum())
    base = dict(D=D, N=int(len(item_cnt)), rows=int(len(d)), d_avg=davg,
                d_max=dmax, d_cv=cv, S2_raw=s2_raw)
    out = []
    for s in supports:
        mu = int(np.ceil(s * D)) if s > 0 else 1
        freq = item_cnt.index[item_cnt.to_numpy() >= mu]
        F1 = int(len(freq))
        fsizes = d[d["item"].isin(freq)].groupby("txn_id").size().astype(np.float64)
        s2_f = float((fsizes ** 2).sum())
        fdavg = float(fsizes.sum() / D)
        fcv = float(fsizes.reindex(sizes.index, fill_value=0).std(ddof=0) / fdavg) if fdavg else float("nan")
        out.append(dict(base, min_support=s, min_count=mu, F1=F1, S2_filtered=s2_f,
                        rho_raw=(F1 ** 2 / 2) / s2_raw if s2_raw else float("nan"),
                        rho_filtered=(F1 ** 2 / 2) / s2_f if s2_f else float("nan"),
                        P1_raw=(F1 - 1) / (davg * (1 + cv ** 2)) if davg else float("nan"),
                        P1_filtered=(F1 - 1) / (fdavg * (1 + fcv ** 2)) if fdavg else float("nan")))
    return out
