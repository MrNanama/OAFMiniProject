import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from oaf.corr import build, strategy
from oaf.backtest import metrics, boot_sharpe, psr, deflated_sr
d = build(); d.to_pickle("results/corr_d.pkl")
R, W = {}, {}
for k in ["bh", "voltarget", "mv_const", "corr"]:
    R[k], W[k] = strategy(d, k)
P = pd.DataFrame(R).dropna(); W = pd.DataFrame(W).reindex(P.index)
P.to_pickle("results/corr_pnl.pkl"); W.to_pickle("results/corr_w.pkl")
def row(x): 
    m = metrics(x); return dict(sharpe=m["sharpe"], ann_ret=m["ann_ret"], vol=m["ann_vol"], maxdd=m["maxdd"])
tab = {}
for nm, (a, b) in {"ALL 2000-22": ("2000", "2022"), "2000-10": ("2000", "2010"), "2011-18": ("2011", "2018"), "2019-22 (rf~0)": ("2019", "2022")}.items():
    for k in P: tab[(nm, k)] = row(P[k].loc[a:b])
T = pd.DataFrame(tab).T.round(3); print(T.to_string())
T.to_csv("results/corr_metrics.csv")
print("\navg |w| / turnover per yr:"); print(pd.DataFrame({"mean_w": W.mean(), "turn/yr": W.diff().abs().mean() * 252}).round(2))
diff = (P["corr"] - P["mv_const"]); print("\ncorr minus mv_const: ann", round(diff.mean()*252,4), "t-ish", round(diff.mean()/diff.std()*np.sqrt(len(diff)),2))
print("Sharpe 95% CI (block bootstrap) corr:", boot_sharpe(P["corr"]).round(2), " voltarget:", boot_sharpe(P["voltarget"]).round(2), " mv_const:", boot_sharpe(P["mv_const"]).round(2))
print("diff-series Sharpe CI (corr-mv_const):", boot_sharpe(diff).round(2))
