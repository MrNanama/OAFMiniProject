import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, itertools
from oaf.crypto import load, universe, run
C, V = load()
def sh(x): x = x.dropna(); return x.mean() / x.std() * np.sqrt(365)
# 1) sensitivity grid
res = []
for top, k, q, reb in itertools.product((20, 30, 40), (14, 21, 30), (3, 5, 8), (3, 7, 14)):
    U = universe(C, V, top=top); net, g, t = run(C, U, k, "mom", reb=reb, q=q); res.append(dict(top=top, k=k, q=q, reb=reb, net=sh(net), h1=sh(net.loc[:"2021"]), h2=sh(net.loc["2022":])))
G = pd.DataFrame(res); print("GRID (81 combos) net Sharpe: median %.2f  mean %.2f  min %.2f  max %.2f  frac>0: %.2f  frac both halves>0: %.2f" % (G.net.median(), G.net.mean(), G.net.min(), G.net.max(), (G.net > 0).mean(), ((G.h1 > 0) & (G.h2 > 0)).mean()))
print(G.groupby("k").net.median().round(2).to_dict(), G.groupby("q").net.median().round(2).to_dict(), G.groupby("reb").net.median().round(2).to_dict(), G.groupby("top").net.median().round(2).to_dict())
G.to_csv("results/crypto_grid.csv", index=False)
# 2) random-signal placebo (same universe, same q/reb as base case): gross Sharpe dist of random long/short
U = universe(C, V); r = C.pct_change(); rng = np.random.default_rng(1); null = []
n = U.sum(axis=1)
for i in range(300):
    sig = pd.DataFrame(rng.random(C.shape), C.index, C.columns).where(U); rk = sig.rank(axis=1, method="first"); nn = sig.notna().sum(axis=1)
    raw = (rk.gt(nn - 5, axis=0) & sig.notna()).astype(float) / 5 - (rk.le(5, axis=0) & sig.notna()).astype(float) / 5
    raw = raw.where(nn >= 10); W = raw.where(pd.Series(np.arange(len(C)) % 7 == 0, C.index), np.nan).ffill()
    null.append(sh((W.shift(1) * r).sum(axis=1, min_count=1)))
net, gross, turn = run(C, U, 14, "mom"); print("PLACEBO random L/S gross Sharpe: mean %.2f sd %.2f p95 %.2f p99 %.2f | mom14 gross %.2f -> p-value %.3f" % (np.mean(null), np.std(null), np.percentile(null, 95), np.percentile(null, 99), sh(gross), (np.array(null) >= sh(gross)).mean()))
# 3) leave-top-contributors-out + concentration
lr = np.log(C); sig = ((lr - lr.shift(14))).where(U); rk = sig.rank(axis=1, method="first"); nn = sig.notna().sum(axis=1)
raw = (rk.gt(nn - 5, axis=0) & sig.notna()).astype(float) / 5 - (rk.le(5, axis=0) & sig.notna()).astype(float) / 5; raw = raw.where(nn >= 10)
W = raw.where(pd.Series(np.arange(len(C)) % 7 == 0, C.index), np.nan).ffill(); contrib = (W.shift(1) * r).sum().sort_values(ascending=False)
print("top contributors (sum daily pnl, %):", (contrib.head(6) * 100).round(0).to_dict()); print("share of total pnl from top5 coins: %.0f%%" % (contrib.head(5).sum() / contrib.sum() * 100))
drop = list(contrib.head(5).index); C2 = C.drop(columns=drop); V2 = V.drop(columns=drop); n2, g2, t2 = run(C2, universe(C2, V2), 14, "mom")
print("net Sharpe without top-5 contributors: %.2f (base %.2f)" % (sh(n2), sh(net)))
# 4) market exposure
btc = r["BTC-USD"].reindex(net.index); ok = net.notna() & btc.notna(); b = np.cov(net[ok], btc[ok])[0, 1] / btc[ok].var(); print("beta to BTC %.2f  corr %.2f" % (b, net[ok].corr(btc[ok])))
print("year-by-year net Sharpe:", {y: round(sh(net.loc[str(y)]), 2) for y in range(2018, 2027)})
