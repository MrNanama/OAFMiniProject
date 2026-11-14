import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from oaf.kf import industries
from oaf.leadlag import weekly, walk_forward, backtest
from oaf.backtest import boot_sharpe
I = industries(); rel, Rw = weekly(I)
def sh(x): return x.mean() / x.std() * np.sqrt(52)
res = {}
for nm in ("own", "net"):
    pr = walk_forward(rel, nm); res[nm], W = backtest(pr, Rw, rel); pr.to_pickle(f"results/ll_pred_{nm}.pkl")
rng = np.random.default_rng(0); null = []
for i in range(10):
    sft = int(rng.integers(104, rel.shape[0] - 104)); pr = walk_forward(rel, "net", shift=sft); b, _ = backtest(pr, Rw, rel); null.append(sh(b["net"]))
rows = {}
for nm in res:
    for per, (a, b) in {"1947-2026": ("1947", "2026"), "1947-69": ("1947", "1969"), "1970-89": ("1970", "1989"), "1990-2009": ("1990", "2009"), "2010-26": ("2010", "2026")}.items():
        x = res[nm].loc[a:b]; rows[(nm, per)] = dict(gross_sh=sh(x.gross), net_sh=sh(x.net), net_ann=x.net.mean() * 52, vol=x.net.std() * np.sqrt(52), turn_yr=x.turn.mean() * 52)
T = pd.DataFrame(rows).T.round(3); print(T.to_string()); T.to_csv("results/ll_metrics.csv")
print("placebo (time-shifted predictors) net Sharpes:", np.round(null, 2), "mean %.2f max %.2f" % (np.mean(null), np.max(null)))
x = res["net"]["net"]; print("net CI", boot_sharpe(x.rename(None), n=1000, block=8).round(2))
for c in (0, 2, 5, 10, 20):
    b, _ = backtest(pd.read_pickle("results/ll_pred_net.pkl"), Rw, rel, cost_bps=c); print(f"cost {c} bps/side: net Sharpe 1947-2026 {sh(b.net):.2f}, 1990-2026 {sh(b.net.loc['1990':]):.2f}")
pd.DataFrame({k: v["net"] for k, v in res.items()}).to_pickle("results/ll_pnl.pkl")
