import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from oaf.gap import load, signals, events, run
from oaf.backtest import metrics, boot_sharpe, psr, deflated_sr
O, C = load(); S = signals(O, C)
res = {}
for kind in ["all", "fade", "go", "down"]:
    E = events(S, kind); res[kind] = run(E, S["cc"], hold=5)
rows = {}
for kind, r in res.items():
    for nm, (a, b) in {"2006-26": ("2006", "2026"), "2006-15": ("2006", "2015"), "2016-26": ("2016", "2026"), "2024-26": ("2024", "2026")}.items():
        x = r["net"].loc[a:b]; g = r["gross"].loc[a:b]; m = metrics(x); mg = metrics(g)
        rows[(kind, nm)] = dict(net_sharpe=m["sharpe"], gross_sharpe=mg["sharpe"], net_ann=m["ann_ret"], vol=m["ann_vol"], maxdd=m["maxdd"],
                                  gross_exp=r["gross_exp"].loc[a:b].mean(), turn_yr=r["turnover"].loc[a:b].mean() * 252)
T = pd.DataFrame(rows).T.round(3); print(T.to_string()); T.to_csv("results/gap_metrics.csv")
for c in (0, 2.5, 5, 10):
    x = run(events(S, "all"), S["cc"], hold=5, cost_bps=c)["net"].loc["2006":]; print(f"all-gaps hold5 cost {c}bps/side: Sharpe {metrics(x)['sharpe']:.2f}")
for h in (2, 10, 20):
    x = run(events(S, "all"), S["cc"], hold=h)["net"].loc["2006":]; print(f"all-gaps hold{h} (5bps): Sharpe {metrics(x)['sharpe']:.2f}")
x = res["all"]["net"].loc["2006":]; print("all-gaps CI", boot_sharpe(x).round(2), "PSR", round(psr(x), 3), "DSR(N=10)", round(deflated_sr(x, 10), 3))
pd.DataFrame({k: v["net"] for k, v in res.items()}).to_pickle("results/gap_pnl.pkl")
