"""Is the sparse-model edge real or an artefact? Survivorship-neutral P&L, sub-periods, legs, cost curve, plots."""
import numpy as np, pandas as pd, copy, importlib.util, sys, io, contextlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from oaf.data import load, features
from oaf.backtest import run, metrics, boot_sharpe
px, mk = load(); F = features(px, mk); r, sig = F["r"], F["sig_e"]
HOLD, FT = 5, "2004-01-01"; ew, mkt = r.mean(axis=1), F["m"]
ev = pd.read_pickle("results/ev_2.0.pkl").dropna(subset=["pred", "y"])
def weights(df, col):
    P = df.pivot_table(index="date", columns="stock", values=col, aggfunc="sum").reindex(index=r.index, columns=r.columns).fillna(0.0)
    P = P.div(sig.fillna(sig.mean().mean())); return sum(P.shift(k).fillna(0.0) for k in range(HOLD)) / HOLD
W = weights(ev, "pred")
base = run(W, F, cost_bps=5.0)
# (1) survivorship-neutral: strip every stock's own full-sample mean return from r (and keep the SPX hedge)
F2 = dict(F); F2["r"] = r - r.mean()
sn = run(W, F2, cost_bps=5.0)
# (2) legs
Wl, Ws_ = W.clip(lower=0), W.clip(upper=0)
leg = {"long leg only (unhedged stock P&L)": (base["W"].clip(lower=0).shift(1) * r).sum(axis=1),
       "short leg only (unhedged stock P&L)": (base["W"].clip(upper=0).shift(1) * r).sum(axis=1)}
# (3) by direction of the triggering move
dn = run(weights(ev[ev.z < 0], "pred"), F, cost_bps=5.0)["net"]; up = run(weights(ev[ev.z > 0], "pred"), F, cost_bps=5.0)["net"]
rows = {}
for nm, s in [("model (5bps)", base["net"]), ("model, survivorship-neutral (5bps)", sn["net"]), ("only after DOWN moves", dn), ("only after UP moves", up)]:
    rows[nm] = metrics(s, start=FT, ew=ew, mkt=mkt)
for per, (a, b) in {"2004-2012": ("2004", "2012"), "2013-2022": ("2013", "2022")}.items():
    s = base["net"][(base["net"].index >= a) & (base["net"].index <= b + "-12-31")]; rows[f"model {per}"] = metrics(s, ew=ew, mkt=mkt)
s = base["net"]; 
for yr0, yr1, nm in [("2008-09-01", "2009-06-30", "GFC 2008-09"), ("2020-02-15", "2020-06-30", "COVID 2020")]:
    x = s[(s.index >= yr0) & (s.index <= yr1)]; rows[nm] = dict(ann_ret=(1 + x).prod() - 1, ann_vol=np.nan, sharpe=np.nan, maxdd=((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min())
T = pd.DataFrame(rows).T; pd.set_option("display.width", 220); print(T[["ann_ret", "ann_vol", "sharpe", "maxdd", "beta_EW", "beta_SPX"]].round(3).to_string())
T.to_csv("results/diagnostics.csv")
# cost curve
cc = {c: metrics(run(W, F, cost_bps=c)["net"], start=FT)["sharpe"] for c in [0, 2, 5, 8, 10, 15, 20]}
print("\nSharpe vs one-way cost (bps):", {k: round(v, 2) for k, v in cc.items()})
# attribution of P&L by year
yr = base["net"][base["net"].index >= FT].groupby(lambda d: d.year).sum(); print("\nannual net return by year (%):", (yr * 100).round(1).to_dict())
# time-out of positive years
print("positive years: %d/%d" % ((yr > 0).sum(), len(yr)))
# leg decomposition
print("\nleg P&L (ann, % of unhedged stock P&L)"); 
for k, v in leg.items(): print("  %-40s %6.1f%%" % (k, v[v.index >= FT].mean() * 252 * 100))
# plots
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
pn = pd.read_pickle("results/model_pnl.pkl")
(1 + base["net"][base["net"].index >= FT]).cumprod().plot(ax=ax[0], label="Regime event model (net 5bps)", lw=2, color="#1f77b4")
(1 + sn["net"][sn["net"].index >= FT]).cumprod().plot(ax=ax[0], label="…survivorship-neutral", lw=1.5, color="#2ca02c", ls="--")
(1 + ew[ew.index >= FT]).cumprod().plot(ax=ax[0], label="EW basket of the 20 (survivors)", color="#aaa", lw=1)
ax[0].set_yscale("log"); ax[0].set_title("Walk-forward OOS equity (log)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].plot(list(cc.keys()), list(cc.values()), "o-", color="#1f77b4"); ax[1].axhline(0, color="k", lw=.8); ax[1].set_title("Sharpe vs one-way cost (bps)"); ax[1].grid(alpha=.3)
ax[2].bar((yr * 100).index, (yr * 100).values, color=np.where(yr > 0, "#1f77b4", "#d62728")); ax[2].set_title("Net return by year (%)"); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/diagnostics.png", dpi=130)
