"""Profiling-OFF overhead check (VM): production 0.2.0 vs the phaseprof build without
FASTAPRIORI_PROFILE, 1 thread, on the six cells notebook 19 flagged. Paired, alternating order."""
import os, sys, json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

BUNDLE = Path("/home/user/artifacts")
PROF_SITE = BUNDLE / "_work" / "phaseprof_site"
sys.path.insert(0, str(BUNDLE / "bench"))
os.environ.pop("PYTHONPATH", None)
os.environ.pop("FASTAPRIORI_PROFILE", None)
import pandas as pd
import exp_runners as pr

OUT = pr.RESULTS / "phaseprof_off_check.csv"
CELLS = [("Chainstore", 3), ("Chainstore", 4), ("Chainstore", 5), ("Instacart", 3),
         ("T25.I8.D5000k.N40k.L40k", 4), ("T30.I8.D2000k.N10k.L10k", 4)]
PAIRS, S = 4, 1e-4
ENVS = {"prod_0.2.0": None, "phaseprof_off": str(PROF_SITE)}

def child_version(pp):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if pp:
        env["PYTHONPATH"] = pp
    return subprocess.run([sys.executable, "-c", "import fastapriori, importlib.metadata as m; "
                           "print(m.version('fastapriori'), fastapriori.__file__)"],
                          capture_output=True, text=True, env=env).stdout.strip()

for lab, pp in ENVS.items():
    v = child_version(pp)
    print(lab, "->", v, flush=True)
    assert ("phaseprof" in v) == (lab == "phaseprof_off"), v

def run(lab, ds, k):
    pp = ENVS[lab]
    if pp:
        os.environ["PYTHONPATH"] = pp
    try:
        r = pr.run_once("hyb_1t", ds, k, S, timeout_s=1500)
    finally:
        os.environ.pop("PYTHONPATH", None)
    return r

rows, warmed = [], set()
t0 = time.time()
for ds, k in CELLS:
    if ds not in warmed:                       # page cache + allocator warm-up, discarded
        run("prod_0.2.0", ds, k); warmed.add(ds)
    for p in range(PAIRS):
        order = list(ENVS) if p % 2 == 0 else list(ENVS)[::-1]
        for lab in order:
            r = run(lab, ds, k)
            r.update(experiment="phaseprof_off_check", dataset=ds, method=lab, k=k, min_support=S,
                     run_id=p, warmup=False, host=os.uname().nodename,
                     timestamp_utc=datetime.now(timezone.utc).isoformat())
            rows.append(r)
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"{time.time()-t0:7.0f}s {ds} k={k} pair {p} {lab}: {r['status']} wall={r['wall_s']}", flush=True)

d = pd.DataFrame(rows)
d = d[d.status == "ok"]
w = d.pivot_table(index=["dataset", "k", "run_id"], columns="method", values="wall_s")
w["ratio"] = w["phaseprof_off"] / w["prod_0.2.0"]
g = w.groupby(level=["dataset", "k"])
summ = pd.DataFrame({"prod_median_s": g["prod_0.2.0"].median(), "off_median_s": g["phaseprof_off"].median(),
                     "paired_ratio_median": g["ratio"].median(), "ratio_min": g["ratio"].min(),
                     "ratio_max": g["ratio"].max()})
pd.set_option("display.width", 200)
print(summ.round(4).to_string())
summ.to_csv(pr.RESULTS / "phaseprof_off_check_summary.csv")
print("CHECK DONE", flush=True)
