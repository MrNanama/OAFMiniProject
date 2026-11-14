import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from oaf.crypto import load, universe, run
C, V = load(); U = universe(C, V)
def sh(x): x = x.dropna(); return x.mean() / x.std() * np.sqrt(365) if len(x) > 100 else np.nan
rows = {}; P = {}
for kind in ("mom", "rev"):
    for k in ((7, 14, 30, 60, 90) if kind == "mom" else (1, 3, 7)):
        net, gross, turn = run(C, U, k, kind); P[f"{kind}{k}"] = net
        s = lambda x, a, b: sh(x.loc[a:b])
        rows[f"{kind}{k}"] = {"gross": sh(gross), "net": sh(net), "net 18-21": s(net, "2018", "2021"), "net 22-26": s(net, "2022", "2026"), "ann net%": net.mean() * 365 * 100, "turn/yr": turn.mean() * 365}
T = pd.DataFrame(rows).T.round(2); print(T.to_string()); T.to_csv("results/crypto_metrics.csv")
print("\navg universe size:", round(U.sum(axis=1).mean(), 1), " BTC buy&hold Sharpe:", round(sh(C['BTC-USD'].pct_change().loc['2018':]), 2))
pd.DataFrame(P).to_pickle("results/crypto_pnl.pkl")
