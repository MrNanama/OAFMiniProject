import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from oaf.binance import load
from oaf.crypto import weights, pnl, beta_neutral, state_scale, vol_target
C, Q = load(); C = C.loc["2018-01-01":]; Q = Q.loc[C.index]; r = C.pct_change(); lr = np.log(C)
dv = Q.rolling(30, min_periods=15).mean().shift(1); U = (dv.where(C.rolling(90).count().shift(1) >= 90).rank(axis=1, ascending=False) <= 30)
def sh(x): x = x.dropna(); return x.mean() / x.std() * np.sqrt(365)
def mdd(x): e = (1 + x.fillna(0)).cumprod(); return (e / e.cummax() - 1).min()
def worst_month(x): return x.resample("ME").sum().min()
def stats(x): return dict(Sharpe=sh(x), s18_21=sh(x.loc[:"2021"]), s22_26=sh(x.loc["2022":]), s25_26=sh(x.loc["2025":]), ann=x.mean() * 365 * 100, vol=x.std() * np.sqrt(365) * 100, maxDD=mdd(x) * 100, worstM=worst_month(x) * 100)
# ---- Step 2: momentum-crash fixes on mom14 (pre-specified: beta-neutral legs; bear-state x0.5 if EW-market DD<-25%; vol target 20%)
WL, WS = weights(lr - lr.shift(14), U); out = {}
out["base (gross 2x)"] = pnl(WL, WS, r)
bL, bS = beta_neutral(WL, WS, r, U); out["+beta-neutral"] = pnl(bL, bS, r)
sL, sS = state_scale(WL, WS, r, U); out["+bear-state x0.5"] = pnl(sL, sS, r)
vL, vS, lev = vol_target(WL, WS, r); out["+vol-target 20%"] = pnl(vL, vS, r)
aL, aS = beta_neutral(WL, WS, r, U); aL, aS = state_scale(aL, aS, r, U); aL, aS, lev2 = vol_target(aL, aS, r); out["all three"] = pnl(aL, aS, r)
T = pd.DataFrame({k: stats(v) for k, v in out.items()}).T.round(2); print("STEP 2: crash mitigation, mom14 PIT (10bp, 10% short funding)"); print(T.to_string()); T.to_csv("results/crypto_crash_fix.csv")
# ---- Step 3: orthogonal signals (pre-specified literature signals), all rank-based within universe
m = r.where(U).mean(axis=1); btc = r["BTC"]
beta60 = r.rolling(60, min_periods=30).cov(btc).div(btc.rolling(60, min_periods=30).var(), axis=0)
resid = r.sub(beta60.shift(1).mul(btc, axis=0)); sig = {
 "mom14": lr - lr.shift(14),
 "BTC-resid mom14": np.log1p(resid.fillna(0)).rolling(14).sum(),
 "small-size (low $vol)": -np.log(dv),
 "low idio vol (60d)": -resid.rolling(60, min_periods=30).std(),
 "volume shock (7d/90d)": Q.rolling(7).mean() / Q.rolling(90, min_periods=45).mean()}
res = {}; R = {}
for k, s in sig.items():
    a, b = weights(s, U); R[k] = pnl(a, b, r); res[k] = stats(R[k])
rk = lambda s: s.where(U).rank(axis=1, pct=True)
comp = rk(sig["mom14"]) * 0.5 + rk(sig["BTC-resid mom14"]) * 0.5; a, b = weights(comp, U); R["mom14+resid (50/50)"] = pnl(a, b, r); res["mom14+resid (50/50)"] = stats(R["mom14+resid (50/50)"])
comp2 = rk(sig["mom14"]) * 0.5 + rk(-sig["volume shock (7d/90d)"]) * 0.5; a, b = weights(comp2, U); R["mom14 + low-vol-shock"] = pnl(a, b, r); res["mom14 + low-vol-shock"] = stats(R["mom14 + low-vol-shock"])
T3 = pd.DataFrame(res).T.round(2); print("\nSTEP 3: signal scan (all 10bp, 10% funding, unscaled)"); print(T3.to_string()); T3.to_csv("results/crypto_signals.csv")
print("\ncorr of daily PnL with mom14:", {k: round(R[k].corr(R["mom14"]), 2) for k in R if k != "mom14"})
pd.DataFrame({**out, **{"sig_" + k: v for k, v in R.items()}}).to_pickle("results/crypto_step23_pnl.pkl")
