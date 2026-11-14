import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from oaf.crypto import load, universe
C, V = load(); U = universe(C, V); r = C.pct_change(); lr = np.log(C); reb = pd.Series(np.arange(len(C)) % 7 == 0, C.index)
def weights(k):
    sig = (lr - lr.shift(k)).where(U); rk = sig.rank(axis=1, method="first"); n = sig.notna().sum(axis=1)
    L = (rk.gt(n - 5, axis=0) & sig.notna()).astype(float) / 5; S = (rk.le(5, axis=0) & sig.notna()).astype(float) / 5
    ok = n >= 10; return L.where(ok).where(reb, np.nan).ffill(), S.where(ok).where(reb, np.nan).ffill()
L14, S14 = weights(14); L30, S30 = weights(30); L = (L14 + L30) / 2; S = (S14 + S30) / 2
lp = (L.shift(1) * r).sum(axis=1, min_count=1); sp = -(S.shift(1) * r).sum(axis=1, min_count=1); net = lp + sp
btc = r["BTC-USD"]; mkt = r.where(U).mean(axis=1)
print("PnL by year (% of capital, gross of cost): long leg / short leg / total")
for y in range(2018, 2027): print(y, round(lp.loc[str(y)].sum() * 100, 1), round(sp.loc[str(y)].sum() * 100, 1), round(net.loc[str(y)].sum() * 100, 1), "| mkt ret", round(mkt.loc[str(y)].sum() * 100, 0))
x = net.loc["2026"]; print("\n2026 monthly total %:", (x.groupby(x.index.month).sum() * 100).round(1).to_dict())
print("2026 monthly mkt-avg %:", (mkt.loc['2026'].groupby(mkt.loc['2026'].index.month).sum() * 100).round(1).to_dict())
ct = ((L + 0) .shift(1) * r - (S.shift(1) * r)).loc["2026"].sum().sort_values(); print("\n2026 worst contributors (%):", (ct.head(6) * 100).round(1).to_dict(), "\nbest:", (ct.tail(3) * 100).round(1).to_dict())
# is it a momentum crash? correlate daily PnL with market rebound after drawdown
dd = (1 + mkt.fillna(0)).cumprod(); dd = dd / dd.rolling(90).max() - 1
reb_days = (dd.shift(1) < -0.25) & (mkt > 0.03)
print("\navg PnL on market-rebound days (after >25% DD, mkt>+3%):", round(net[reb_days].mean() * 100, 2), "% vs all days", round(net.mean() * 100, 3), "%; n =", int(reb_days.sum()))
print("worst 5 days overall:", (net.nsmallest(5) * 100).round(1).to_dict())
