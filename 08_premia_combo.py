import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from oaf.backtest import boot_sharpe, psr, deflated_sr
P = pd.read_pickle("results/premia_pnl.pkl"); mkt = P["mkt_bh"]
def scale(x, tv=0.10, win=252, cap=4.0):
    s = x.rolling(win, min_periods=126).std().shift(1) * np.sqrt(252); return (x * (tv / s).clip(upper=cap))
def sh(x): x = x.dropna(); return x.mean() / x.std() * np.sqrt(252)
def combo(names): return pd.concat([scale(P[n]) for n in names], axis=1).mean(axis=1).where(lambda s: s.index >= "1928")
C = {"slow3 (fm+im+tom)": combo(["factor_mom", "ind_mom", "tom"]), "slow2 (fm+im, mkt-neutral-ish)": combo(["factor_mom", "ind_mom"]), "slow4 (+vol_mgd)": combo(["factor_mom", "ind_mom", "tom", "vol_mgd_mkt"])}
eras = {"all": ("1928", "2026"), "1928-62": ("1928", "1962"), "1963-89": ("1963", "1989"), "1990-2009": ("1990", "2009"), "2010-26": ("2010", "2026"), "2020-26": ("2020", "2026")}
print("Sharpe by era"); print(pd.DataFrame({k: {e: sh(v.loc[a:b]) for e, (a, b) in eras.items()} for k, v in C.items()}).T.round(2).to_string())
print("\nMarket B&H Sharpe:", {e: round(sh(mkt.loc[a:b]), 2) for e, (a, b) in eras.items()})
main = C["slow3 (fm+im+tom)"]
print("\nslow3 stats: ann ret %.1f%% vol %.1f%%" % (main.mean() * 252 * 100, main.std() * np.sqrt(252) * 100), " maxDD %.0f%%" % (((1 + main.fillna(0)).cumprod() / (1 + main.fillna(0)).cumprod().cummax() - 1).min() * 100))
for e in ("all", "1990-2009", "2010-26"):
    a, b = eras[e]; y = main.loc[a:b].dropna(); x = mkt.reindex(y.index)
    r = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    print(f"  {e:10s} beta {r.params.iloc[1]:+.2f}  alpha {r.params.iloc[0]*252*100:+.1f}%/yr  t={r.tvalues.iloc[0]:.1f}")
print("\nblock-bootstrap Sharpe CI slow3 (all / 1990+ / 2010+):", boot_sharpe(main.dropna(), n=500, block=21).round(2), boot_sharpe(main.loc["1990":].dropna(), n=500, block=21).round(2), boot_sharpe(main.loc["2010":].dropna(), n=500, block=21).round(2))
print("PSR(0) 1990+: %.3f   DSR(N=8 sleeve/params tried) 1990+: %.3f" % (psr(main.loc["1990":].dropna()), deflated_sr(main.loc["1990":].dropna(), 8)))
for k, v in C.items(): v.to_pickle("results/combo_" + k.split()[0] + ".pkl")
# extra cost stress on factor_mom/ind_mom only (factor notional trades): 5x costs
