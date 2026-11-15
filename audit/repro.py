"""Run the author's FROZEN strategy code through the author's engine on freshly downloaded data."""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/claude-0/audit")
from oafbt import config as C
from oafbt.data.store import DataStore
from oafbt.engine import runner
from oafbt.engine.metrics import perf_stats
from oafbt.strategies.registry import load
names = sys.argv[1:]
ds = DataStore(holdout=True)
author = pd.read_csv("/tmp/claude-0/audit/results/holdout/holdout_metrics.csv").set_index("strategy")
rows = []
for n in names:
    kw = {"fomc_day": False} if n == "s10_turn_of_month" else {}
    st = load(n, **kw)
    out = runner.run(st, ds=ds, holdout=True, save=False, log=False, tag="repro")
    net = out["res"].net; bench = runner.benchmark(ds, st.universe)
    d = perf_stats(net, bench=bench, start=C.EVAL_START[st.universe], end=C.DEV_END, first_trade=out["res"].info.get("first_trade"))
    h = perf_stats(net, bench=bench, start=C.HOLDOUT_START)
    rows.append(dict(strategy=n, SR_dev_mine=round(d["sharpe"], 2), SR_dev_author=round(author.loc[n, "SR_dev"], 2),
                     SR_hold_mine=round(h["sharpe"], 2), SR_hold_author=round(author.loc[n, "SR_holdout"], 2),
                     DD_hold_mine=round(h["max_dd"], 3), DD_hold_author=round(author.loc[n, "maxDD_holdout"], 3),
                     vol_hold=round(h["ann_vol"], 3)))
    print(rows[-1], flush=True)
    net.to_frame("net").to_parquet(f"/home/user/OAFMiniProject/audit/repro_{n}.parquet")
pd.DataFrame(rows).to_csv("/home/user/OAFMiniProject/audit/repro_%s.csv" % ("_".join(n[:3] for n in names)), index=False)
