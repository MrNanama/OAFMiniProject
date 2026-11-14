"""Extract every number/series used in the s15 pitch deck (post-freeze reporting) -> presentation/deck_data.json"""
import json

import numpy as np
import pandas as pd

from oafbt import config as C
from oafbt.data.store import DataStore
from oafbt.engine.metrics import perf_stats, drawdown
from oafbt.strategies.s10_turn_of_month import month_positions

ds = DataStore(holdout=True)
out = {}
d = pd.read_parquet(C.RESULTS / "runs/s15_rebalancing_flows/v3_holdout/daily.parquet")
net = d["net"]
first = net.ne(0).idxmax()
net = net.loc[first:]
wealth = (1 + net).cumprod()
m = wealth.resample("ME").last()
out["cum"] = {"dates": [x.strftime("%Y-%m") for x in m.index], "values": [round(float(v) * 100 - 100, 2) for v in m]}
yr = net.resample("YE").apply(lambda s: (1 + s).prod() - 1)
out["annual"] = {"years": [int(x.year) for x in yr.index], "ret": [round(float(v) * 100, 2) for v in yr]}
for lab, a, b in [("dev", None, C.DEV_END), ("holdout", C.HOLDOUT_START, None)]:
    s = perf_stats(net, start=a, end=b)
    out[f"stats_{lab}"] = {k: (round(float(s[k]), 4) if isinstance(s[k], (int, float, np.floating)) else s[k])
                          for k in ["sharpe", "ann_ret", "ann_vol", "max_dd", "t_hac", "pos_years", "hit_monthly",
                                    "sortino", "calmar", "start", "end", "skew", "worst_month", "best_month"]}
out["stats_dev"]["turnover_yr"] = round(float(d["turnover"].loc[:C.DEV_END].mean() * 252), 1)
out["stats_dev"]["cost_yr"] = round(float(d["cost"].loc[first:C.DEV_END].mean() * 252) * 100, 2)

# event study: MTD SPY-TLT log spread at day -4 -> spread over days -2..-1 and +1..+2, by quintile
x = ds.xret(["SPY", "TLT"]).dropna()
cal = x.index
fe, fs = month_positions(cal)
lr = np.log1p(x["SPY"]) - np.log1p(x["TLT"])
rows = []
per = cal.to_period("M")
months = sorted(set(per))
for i, mth in enumerate(months[:-1]):
    idx = cal[per == mth]
    f = fe.reindex(idx)
    if (f == -4).sum() != 1 or idx[0] == cal[0]:
        continue
    t4 = idx[f == -4][0]
    rp = lr.loc[idx[0]:t4].sum()
    late = lr.loc[idx[f.isin([-2, -1])]].sum()
    nxt = cal[per == months[i + 1]]
    early = lr.loc[nxt[:2]].sum() if len(nxt) >= 2 else np.nan
    rows.append({"month": mth.to_timestamp(), "rp": rp, "late": late, "early": early})
ev = pd.DataFrame(rows).dropna()
for lab, a, b in [("dev", "2004-01-01", C.DEV_END), ("holdout", C.HOLDOUT_START, "2026-12-31")]:
    e = ev[(ev.month >= a) & (ev.month <= b)].copy()
    e["q"] = pd.qcut(e.rp, 5, labels=False) + 1
    g = e.groupby("q")[["late", "early", "rp"]].mean() * 1e4
    out[f"event_{lab}"] = {"q": [int(q) for q in g.index], "late_bp": [round(float(v), 1) for v in g.late],
                           "early_bp": [round(float(v), 1) for v in g.early], "mtd_bp": [round(float(v), 0) for v in g.rp],
                           "n": int(len(e)), "corr_late": round(float(e.rp.corr(e.late)), 2),
                           "corr_early": round(float(e.rp.corr(e.early)), 2)}

h = pd.read_csv(C.RESULTS / "holdout/holdout_metrics.csv")
frozen = json.loads(C.FROZEN_FILE.read_text())
sr0420 = {}
for n in h.strategy:
    sr0420[n] = json.load(open(C.RESULTS / f"runs/{n}/{frozen['versions'][n]}/summary.json"))["sharpe_2004_2020"]
out["book"] = [{"strategy": r.strategy, "sr_dev": round(float(sr0420[r.strategy]), 2), "sr_holdout": round(float(r.SR_holdout), 2)}
               for r in h.itertuples()]
sens = pd.read_csv(C.RESULTS / "sensitivity.csv")
out["sens"] = sens[sens.strategy == "s15_rebalancing_flows"][["override", "SR_04_20"]].to_dict(orient="records")
s15sum = json.load(open(C.RESULTS / "runs/s15_rebalancing_flows/v3/summary.json"))
out["robust"] = {"sr_cost2x": round(s15sum["sharpe_cost2x"], 2), "sr_lag1": round(s15sum["sharpe_lag+1"], 2),
                 "ci": [round(s15sum["sr_ci_lo"], 2), round(s15sum["sr_ci_hi"], 2)], "sub": s15sum["sr_subperiods"],
                 "trials": s15sum["n_trials"], "dsr": round(s15sum["dsr"], 2), "beta": round(s15sum["beta_spy"], 3)}
# correlations with SPY and the other 9 strategies (weekly)
rets = {}
for n in h.strategy:
    rr = pd.read_parquet(C.RESULTS / f"runs/{n}/{frozen['versions'][n]}_holdout/daily.parquet")["net"]
    rets[n] = rr
R = pd.DataFrame(rets)
spy = ds.xret(["SPY"])["SPY"]
for lab, a, b in [("dev", "2005-01-01", C.DEV_END), ("holdout", C.HOLDOUT_START, None)]:
    W = R.loc[a:b].resample("W").sum()
    W["SPY"] = spy.loc[a:b].resample("W").sum()
    c = W.corr()["s15_rebalancing_flows"].drop("s15_rebalancing_flows")
    out[f"corr_{lab}"] = {k: round(float(v), 2) for k, v in c.items()}
comb = pd.read_parquet(C.RESULTS / "holdout/combined_daily.parquet")["combined"]
for lab, a, b in [("dev", "2005-01-01", C.DEV_END), ("holdout", C.HOLDOUT_START, None)]:
    s = perf_stats(comb, start=a, end=b)
    out[f"combo_{lab}"] = {"sharpe": round(s["sharpe"], 2), "ann_ret": round(s["ann_ret"], 4), "max_dd": round(s["max_dd"], 4)}
json.dump(out, open(C.ROOT / "presentation/deck_data.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ("cum", "annual")}, indent=1, default=str)[:4000])
