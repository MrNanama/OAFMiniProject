import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/claude-0/audit")
from oafbt import config as C
from oafbt.data.store import DataStore
from oafbt.engine import runner
from oafbt.engine.metrics import perf_stats
from oafbt.strategies.registry import load
ds = DataStore(holdout=True)
rows = []
for n in sys.argv[1:]:
    kw = {"fomc_day": False} if n == "s10_turn_of_month" else {}
    st = load(n, **kw)
    for label, cm, lag in [("base", 1, None), ("cost x2", 2, None), ("cost x3", 3, None), ("cost x5", 5, None), ("exec +1 day", 1, st.lag + 1)]:
        if label == "exec +1 day" and getattr(st, "calendar_timed", False):
            continue          # calendar strategies are scheduled; a one-day delay is a different strategy (author's reasoning)
        out = runner.run(st, ds=ds, holdout=True, save=False, log=False, tag="stress", cost_mult=cm, lag=lag)
        net = out["res"].net; bench = runner.benchmark(ds, st.universe)
        d = perf_stats(net, bench=bench, start=C.EVAL_START[st.universe], end=C.DEV_END)
        h = perf_stats(net, bench=bench, start=C.HOLDOUT_START)
        rows.append(dict(strategy=n[:22], case=label, SR_dev=round(d["sharpe"], 2), SR_hold=round(h["sharpe"], 2), ret_hold=round(h["ann_ret"], 3), DD_hold=round(h["max_dd"], 3)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("/home/user/OAFMiniProject/audit/stress_results.csv", index=False)
