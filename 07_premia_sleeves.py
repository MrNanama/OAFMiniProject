import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from oaf.kf import industries, factors8
from oaf.premia import *
F = factors8(); I = industries()
fcols = ["SMB", "HML", "RMW", "CMA", "UMD"]
S = {}; T = {}
for nm, (W, R) in {"factor_mom": factor_momentum(F, fcols), "ind_mom": industry_mom(I), "tom": turn_of_month(F), "vol_mgd_mkt": vol_managed(F), "mkt_bh": (pd.DataFrame({"MKT": 1.0}, index=F.index), F[["Mkt-RF"]].rename(columns={"Mkt-RF": "MKT"}))}.items():
    c = 5.0 if nm == "ind_mom" else (10.0 if nm == "factor_mom" else 1.0)
    S[nm], T[nm] = sleeve_pnl(W, R, c)
P = pd.DataFrame(S).dropna(how="all"); P.to_pickle("results/premia_pnl.pkl")
def sh(x): x = x.dropna(); return x.mean() / x.std() * np.sqrt(252) if len(x) > 250 and x.std() > 0 else np.nan
eras = {"all": ("1927", "2026"), "1927-62": ("1927", "1962"), "1963-89": ("1963", "1989"), "1990-2009": ("1990", "2009"), "2010-26": ("2010", "2026")}
tab = pd.DataFrame({k: {e: sh(P[k].loc[a:b]) for e, (a, b) in eras.items()} for k in P}).T.round(2); print("NET Sharpe by era"); print(tab.to_string())
print("\nann turnover (x/yr):", {k: round(T[k].mean() * 252, 1) for k in T})
print("ann vol %:", (P.std() * np.sqrt(252) * 100).round(1).to_dict())
print("correlations 1990+:"); print(P.loc["1990":].corr().round(2).to_string())
