"""Run one strategy on the DEV period with the standard battery:
   dev metrics (logged as a trial), 2x-cost and lag+1 sensitivity, common-window (2004-2020) metrics,
   and the look-ahead truncation check.

   python scripts/run_strategy.py s01_gp_trend [--note "..."] [--no-check] [--param key=value ...]
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from oafbt import config as C
from oafbt.data.store import DataStore
from oafbt.engine import runner
from oafbt.engine.checks import lookahead_check
from oafbt.engine.metrics import perf_stats
from oafbt.strategies.registry import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--param", nargs="*", default=[])
    a = ap.parse_args()
    params = {}
    for kv in a.param:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:
            pass
        params[k] = v
    t0 = time.time()
    strat = load(a.name, **params)
    ds = DataStore()
    out = runner.run(strat, ds=ds, note=a.note)
    s = out["stats"]
    res = out["res"]
    bench = runner.benchmark(ds, strat.universe)
    common = perf_stats(res.net, bench=bench, turnover=res.turnover, cost=res.cost, start="2004-01-01", end=C.DEV_END)
    c2 = runner.run(strat, ds=ds, cost_mult=2.0, save=False, log=False)["stats"]
    l2 = runner.run(strat, ds=ds, lag=strat.lag + 1, save=False, log=False)["stats"]
    summ = {k: s.get(k) for k in ["strategy", "version", "start", "end", "n_years", "sharpe", "ann_ret", "ann_vol",
                                  "max_dd", "calmar", "sortino", "skew", "turnover_yr", "cost_drag_yr",
                                  "avg_gross_lev", "beta_spy", "corr_spy", "t_hac", "psr0", "dsr", "n_trials",
                                  "sr_subperiods", "sr_ci_lo", "sr_ci_hi", "pos_years", "hit_monthly", "n_assets"]}
    summ["sharpe_2004_2020"] = common.get("sharpe")
    summ["sharpe_cost2x"] = c2.get("sharpe")
    summ["sharpe_lag+1"] = l2.get("sharpe")
    if not a.no_check:
        lc = lookahead_check(strat)
        summ["lookahead_passed"] = lc["passed"]
        summ["lookahead_detail"] = {k: v for k, v in lc.items() if k != "passed"}
    summ["params"] = strat.params
    summ["runtime_s"] = round(time.time() - t0, 1)
    d = runner.RUNS / strat.name / strat.version
    (d / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    f = lambda k, pct=False: (f"{summ[k]:.1%}" if pct else f"{summ[k]:.2f}") if isinstance(summ.get(k), (int, float)) and summ.get(k) == summ.get(k) else "nan"
    print(f"{summ['strategy']} {summ['version']} [{summ['start']}..{summ['end']}, {summ['n_years']}y] "
          f"SR={f('sharpe')} (04-20: {f('sharpe_2004_2020')}, 2xcost: {f('sharpe_cost2x')}, lag+1: {f('sharpe_lag+1')}) "
          f"ret={f('ann_ret', True)} vol={f('ann_vol', True)} maxDD={f('max_dd', True)} calmar={f('calmar')} "
          f"sortino={f('sortino')} skew={f('skew')} turn={f('turnover_yr')}x cost={f('cost_drag_yr', True)} "
          f"lev={f('avg_gross_lev')} beta={f('beta_spy')} t={f('t_hac')} dsr={f('dsr')} sub={summ['sr_subperiods']} "
          f"CI=[{f('sr_ci_lo')},{f('sr_ci_hi')}] posyrs={summ['pos_years']} lookahead={summ.get('lookahead_passed')} "
          f"({summ['runtime_s']}s)")


if __name__ == "__main__":
    main()
