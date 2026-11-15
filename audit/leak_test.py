"""Data-level look-ahead test. Rebuild panels from RAW data truncated at CUT and compare with the full build
on dates <= CUT. Any difference = the cleaning/alignment pipeline used information from after CUT."""
import sys, shutil, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/claude-0/audit")
from oafbt import config as C
from oafbt.data import store

FULL_RAW = Path("/tmp/claude-0/audit/data/raw")
CUT = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2015-12-31")
WORK = Path("/tmp/claude-0/leak"); (WORK / "raw").mkdir(parents=True, exist_ok=True)
for f in FULL_RAW.glob("yahoo_*.parquet"):
    d = pd.read_parquet(f); d = d[pd.to_datetime(d["date"]) <= CUT]; d.to_parquet(WORK / "raw" / f.name)
mem = pd.read_csv(FULL_RAW / "sp500_membership.csv", parse_dates=["date"]); mem[mem.date <= CUT].to_csv(WORK / "raw" / "sp500_membership.csv", index=False)
shutil.copy(FULL_RAW / "kf.pkl", WORK / "raw" / "kf.pkl")

def build_into(raw, vault, dev):
    C.RAW = Path(raw); store.VAULT = Path(vault); store.DEV_DIR = Path(dev)
    store.build()
build_into(WORK / "raw", WORK / "vault", WORK / "dev")
res = {}
for k in ["ret", "adjclose", "xret", "printed", "rf", "sp500", "tri"]:
    a = pd.read_parquet(store.VAULT.parent.parent / "audit" / "data" / "vault" / f"{k}.parquet") if False else pd.read_parquet(Path("/tmp/claude-0/audit/data/vault") / f"{k}.parquet")
    b = pd.read_parquet(WORK / "vault" / f"{k}.parquet")
    a = a.loc[:CUT]; b = b.loc[:CUT]
    cols = [c for c in b.columns if c in a.columns]
    a, b = a.reindex(index=b.index, columns=cols), b[cols]
    if k in ("printed", "sp500"):
        diff = (a.fillna(False).astype(bool) != b.fillna(False).astype(bool))
    elif k in ("adjclose", "tri"):
        # levels: compare scale-free (returns of the level), since Yahoo's adj factor is a constant across a truncation
        diff = ((a.pct_change(fill_method=None) - b.pct_change(fill_method=None)).abs() > 1e-9) | (a.notna() != b.notna())
    else:
        diff = ((a - b).abs() > 1e-9) | (a.notna() != b.notna())
    n = int(diff.to_numpy().sum()); tot = int(b.notna().to_numpy().sum()) if k not in ("printed", "sp500") else diff.size
    worst = diff.sum().sort_values(ascending=False).head(8)
    res[k] = (n, tot, worst)
    print(f"{k:9s}: {n:8d} differing cells of {tot:10d}  ({100*n/max(tot,1):.4f}%)   cols affected: {int((diff.sum()>0).sum())}/{len(cols)}")
    if n:
        dd = diff.sum(axis=1); dd = dd[dd > 0]
        print("          most-affected tickers:", dict(worst[worst > 0].head(6)))
        print("          date range of diffs :", dd.index.min().date(), "->", dd.index.max().date(), " | in last 10 sessions before cut:", int(dd[dd.index > dd.index.max() - pd.Timedelta(days=15)].sum()))
