import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
d = pd.read_pickle("data/vol.pkl"); sv = d["SVXY"].dropna().pct_change().dropna()
rf = (d["^IRX"].reindex(sv.index).ffill() / 100 / 252); ts = (d["^VIX"] / d["^VIX3M"]).reindex(sv.index)
spy = d["^SP500TR"].pct_change().reindex(sv.index)
print("SVXY", sv.index[0].date(), sv.index[-1].date(), "worst days:", sv.nsmallest(4).round(3).to_dict())
def sh(x): x = x.dropna(); return x.mean() / x.std() * np.sqrt(252)
def mdd(x): e = (1 + x).cumprod(); return (e / e.cummax() - 1).min()
def run(sig, size, cost=2.0):
    w = (sig.shift(0) * size).fillna(0); pnl = w.shift(1) * (sv - rf) + 0; c = w.diff().abs().fillna(0) * cost / 1e4
    return (pnl - c).iloc[1:]
cases = {"always short vol, 100%": (pd.Series(1.0, sv.index), 1.0), "always short vol, 25%": (pd.Series(1.0, sv.index), 0.25),
         "contango (VIX/VIX3M<0.95), 100%": ((ts < 0.95).astype(float), 1.0), "contango<0.95, 25%": ((ts < 0.95).astype(float), 0.25),
         "contango<0.90, 25%": ((ts < 0.90).astype(float), 0.25), "contango<0.95 & VIX<25, 25%": (((ts < 0.95) & (d['^VIX'].reindex(sv.index) < 25)).astype(float), 0.25)}
rows = {}; out = {}
for k, (s, z) in cases.items():
    x = run(s, z); out[k] = x
    rows[k] = {"Sh all": sh(x), "Sh 11-17": sh(x.loc[:"2017"]), "Sh 18-26": sh(x.loc["2018":]), "Sh ex-Feb18": sh(x.drop(x.loc["2018-02-01":"2018-02-28"].index)), "maxDD": mdd(x), "worst day": x.min(), "vol%": x.std() * np.sqrt(252) * 100}
T = pd.DataFrame(rows).T.round(2); print(T.to_string()); T.to_csv("results/shortvol_metrics.csv")
print("\ncorr with SPX (best case 25% contango):", round(out["contango<0.95, 25%"].corr(spy.reindex(out['contango<0.95, 25%'].index)), 2))
print("SPX B&H Sharpe same period:", round(sh(spy - rf), 2))
pd.DataFrame(out).to_pickle("results/shortvol_pnl.pkl")
