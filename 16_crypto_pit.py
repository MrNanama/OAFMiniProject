import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, itertools
from oaf.binance import load
from oaf.crypto import run
C, Q = load(); C = C.loc["2018-01-01":]; Q = Q.loc[C.index]
def uni(top=30, win=30, min_hist=90):
    dv = Q.rolling(win, min_periods=15).mean().shift(1); ok = C.rolling(min_hist).count().shift(1) >= min_hist
    return dv.where(ok).rank(axis=1, ascending=False) <= top
def sh(x): x = x.dropna(); return x.mean() / x.std() * np.sqrt(365) if len(x) > 100 else np.nan
U = uni(); print("avg universe size", round(U.sum(axis=1).mean(), 1), "| share of universe-days that are later-delisted coins: %.1f%%" % (100 * (U & C.apply(lambda s: s.last_valid_index() < C.index[-30]).values[None, :]).sum().sum() / U.sum().sum()))
rows = {}; P = {}
for kind in ("mom", "rev"):
    for k in ((7, 14, 21, 30, 60, 90) if kind == "mom" else (1, 3, 7)):
        net, gross, turn = run(C, U, k, kind); P[f"{kind}{k}"] = net
        rows[f"{kind}{k}"] = {"gross": sh(gross), "net": sh(net), "18-21": sh(net.loc[:"2021"]), "22-26": sh(net.loc["2022":]), "ann net%": net.mean() * 365 * 100, "turn/yr": turn.mean() * 365}
T = pd.DataFrame(rows).T.round(2); print(T.to_string()); T.to_csv("results/crypto_pit_metrics.csv")
res = []
for top, k, q, reb in itertools.product((20, 30, 40), (14, 21, 30), (3, 5, 8), (3, 7, 14)):
    net, g, t = run(C, uni(top=top), k, "mom", reb=reb, q=q); res.append(dict(top=top, k=k, q=q, reb=reb, net=sh(net), h1=sh(net.loc[:"2021"]), h2=sh(net.loc["2022":])))
G = pd.DataFrame(res); print("\nGRID (81): net Sharpe median %.2f mean %.2f min %.2f max %.2f | frac>0 %.2f | both halves>0 %.2f" % (G.net.median(), G.net.mean(), G.net.min(), G.net.max(), (G.net > 0).mean(), ((G.h1 > 0) & (G.h2 > 0)).mean()))
print(G.groupby("k").net.median().round(2).to_dict(), G.groupby("q").net.median().round(2).to_dict(), G.groupby("reb").net.median().round(2).to_dict())
net14 = P["mom14"]; btc = C["BTC"].pct_change().reindex(net14.index); ok = net14.notna() & btc.notna()
print("beta to BTC %.2f" % (np.cov(net14[ok], btc[ok])[0, 1] / btc[ok].var()), "| year net Sharpe mom14:", {y: round(sh(net14.loc[str(y)]), 2) for y in range(2018, 2027)})
G.to_csv("results/crypto_pit_grid.csv", index=False); pd.DataFrame(P).to_pickle("results/crypto_pit_pnl.pkl")
