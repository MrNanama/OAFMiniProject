import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from oaf.premia import hold_monthly, sleeve_pnl, turn_of_month
def sh(x): x = x.dropna(); return x.mean() / x.std() * np.sqrt(252)
sect = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
d = yf.download(sect + ["SPY", "BIL"], start="1998-12-01", auto_adjust=True, progress=False)["Close"]
R = d[sect].pct_change(); spy = d["SPY"].pct_change(); print("sector ETFs from", R.dropna().index[0].date(), "to", R.index[-1].date())
# sector momentum 12-1, long top3 / short bottom3, monthly, cost per side
L = np.log1p(R); sig = L.rolling(252).sum() - L.rolling(21).sum(); rk = sig.rank(axis=1, method="first"); n = sig.notna().sum(axis=1)
raw = (rk.gt(n - 3, axis=0) & sig.notna()).astype(float) / 3 - (rk.le(3, axis=0) & sig.notna()).astype(float) / 3; raw = raw.where(n >= 9)
W = hold_monthly(raw, R.index); out = {}
for c in (0, 2, 5, 10):
    p, t = sleeve_pnl(W, R.fillna(0), c); out[f"sector_mom {c}bp"] = p.loc["2000":]
print("sector_mom turnover x/yr: %.1f" % (t.loc["2000":].mean() * 252))
# turn-of-month on SPY (long SPY over TOM window, cash otherwise) vs buy-hold, 1bp/side futures-like and 5bp
Fm = pd.DataFrame({"Mkt-RF": spy}); Wt, Rt = turn_of_month(Fm)
for c in (1, 5): p, t = sleeve_pnl(Wt, Rt, c); out[f"SPY TOM {c}bp"] = p.loc["2000":]
out["SPY buy&hold"] = spy.loc["2000":]
O = pd.DataFrame(out)
eras = {"2000-26": ("2000", "2026"), "2000-09": ("2000", "2009"), "2010-19": ("2010", "2019"), "2020-26": ("2020", "2026")}
print(pd.DataFrame({k: {e: sh(O[k].loc[a:b]) for e, (a, b) in eras.items()} for k in O}).T.round(2).to_string())
print("ann vol%:", (O.std() * np.sqrt(252) * 100).round(1).to_dict())
O.to_pickle("results/impl_pnl.pkl")
