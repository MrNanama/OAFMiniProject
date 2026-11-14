import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from oaf.crypto import load, universe, run
from oaf.backtest import boot_sharpe
C, V = load(); U = universe(C, V)
def sh(x): x = x.dropna(); return x.mean() / x.std() * np.sqrt(365)
def mdd(x): e = (1 + x.fillna(0)).cumprod(); return (e / e.cummax() - 1).min()
cands = {k: run(C, U, k, "mom")[0] for k in (7, 14, 30, 60)}
# walk-forward lookback selection: each Jan 1 pick best trailing-730d net Sharpe among candidates
sel = pd.Series(np.nan, C.index); choice = {}
for y in range(2020, 2027):
    t0 = pd.Timestamp(f"{y}-01-01"); best = max(cands, key=lambda k: sh(cands[k].loc[t0 - pd.Timedelta(days=730):t0 - pd.Timedelta(days=1)])); choice[y] = best
    m = (C.index >= t0) & (C.index < pd.Timestamp(f"{y+1}-01-01")); sel[m] = cands[best][m]
print("selected lookback by year:", choice); S = sel.dropna()
print("walk-forward-selected: net Sharpe %.2f  ann %.1f%%  vol %.1f%%  maxDD %.0f%%  (2020-26)" % (sh(S), S.mean() * 365 * 100, S.std() * np.sqrt(365) * 100, mdd(S) * 100))
print("fixed mom14 same window: %.2f | fixed mom30: %.2f | BTC B&H: %.2f" % (sh(cands[14].loc[S.index]), sh(cands[30].loc[S.index]), sh(C['BTC-USD'].pct_change().loc[S.index])))
# blended equal-weight of 14d and 30d (no selection), with funding cost on shorts and vol target
b = (cands[14] + cands[30]) / 2
for fund in (0.0, 0.05, 0.10):
    bb = b - fund / 365 * 0.5      # short leg = 0.5 of capital notional; funding paid on it
    print(f"blend 14/30, short funding {fund*100:.0f}%/yr: Sharpe {sh(bb):.2f}  (2018-21 {sh(bb.loc[:'2021']):.2f}, 2022-26 {sh(bb.loc['2022':]):.2f})")
vol = b.rolling(60).std().shift(1) * np.sqrt(365); w = (0.20 / vol).clip(upper=3); vt = (b * w).dropna()
print("vol-targeted 20%% blend: ann %.1f%% vol %.1f%% Sharpe %.2f maxDD %.0f%% avg lev %.2f" % (vt.mean() * 365 * 100, vt.std() * np.sqrt(365) * 100, sh(vt), mdd(vt) * 100, w.mean()))
print("blend bootstrap CI:", boot_sharpe(b.dropna().rename(None), n=500, block=21).round(2) * np.sqrt(365 / 252))
pd.DataFrame({"blend": b, "wf_selected": sel, "vt20": vt}).to_pickle("results/crypto_final_pnl.pkl")
