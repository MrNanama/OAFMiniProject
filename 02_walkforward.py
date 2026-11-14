"""Walk-forward test of the regime-conditioned liquidity/news hypothesis + baselines + placebo."""
import numpy as np, pandas as pd, sys
from sklearn.linear_model import Ridge
from oaf.data import load, features
from oaf.backtest import run, metrics, boot_sharpe, psr, deflated_sr
px, mk = load(); F = features(px, mk)
r, e, z, sig, stress, mz = F["r"], F["e"], F["z_e"], F["sig_e"], F["stress"], F["mz"]
HOLD, FIRST_TEST = 5, "2004-01-01"
ew, mkt = r.mean(axis=1), F["m"]
fwd = e.rolling(HOLD).sum().shift(-HOLD)

def feats(zv, S, M):
    zc = np.clip(zv, -6, 6); dn, up = np.minimum(zc, 0), np.maximum(zc, 0)
    s, sh = S - .5, np.minimum(np.abs(M), 4) / 2
    return np.column_stack([dn, up, dn*s, up*s, dn*sh, up*sh, dn*s*sh, up*s*sh])
names = ["dn", "up", "dn×S", "up×S", "dn×Mk", "up×Mk", "dn×S×Mk", "up×S×Mk"]

def table(thr):
    """long table of stock-days with |z|>=thr (thr=0 -> dense)"""
    Z = z.stack(); Z = Z[Z.abs() >= thr].rename("z").reset_index(); Z.columns = ["date", "stock", "z"]
    Z["stress"] = Z.date.map(stress); Z["mz"] = Z.date.map(mz)
    Z["sig"] = sig.stack().reindex(pd.MultiIndex.from_frame(Z[["date", "stock"]])).values
    Z["fwd"] = fwd.stack().reindex(pd.MultiIndex.from_frame(Z[["date", "stock"]])).values
    Z = Z.dropna(subset=["stress", "sig"]).reset_index(drop=True)
    Z["y"] = Z.fwd / (Z.sig * np.sqrt(HOLD)); return Z

def walk_forward(ev, X, alpha, perm=None):
    pred = np.full(len(ev), np.nan); coefs = []
    for yr in range(int(FIRST_TEST[:4]), 2023):
        cut = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=10)
        tr = ((ev.date < cut) & ev.y.notna()).values; te = ((ev.date >= f"{yr}-01-01") & (ev.date < f"{yr+1}-01-01")).values
        ytr = ev.y[tr] - ev[tr].groupby("stock").y.transform("mean")      # train-only de-drift
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m_ = Ridge(alpha=alpha, fit_intercept=False).fit((X[tr] - mu) / sd, ytr)
        pred[te] = m_.predict((X[te] - mu) / sd); coefs.append(m_.coef_)
    return pred, np.array(coefs)

def weights(ev, col):
    P = ev.pivot_table(index="date", columns="stock", values=col, aggfunc="sum").reindex(index=r.index, columns=r.columns).fillna(0.0)
    P = P.div(sig.reindex(index=r.index).fillna(sig.mean().mean()))          # vol-scaled conviction
    return sum(P.shift(k).fillna(0.0) for k in range(HOLD)) / HOLD

def evaluate(W, cost=5.0): 
    o = run(W, F, cost_bps=cost); return o, metrics(o["net"], start=FIRST_TEST, ew=ew, mkt=mkt, turn=o["turnover"])

out = {}
for label, thr, alpha in [("sparse |z|>=2", 2.0, 300.0), ("dense (all stock-days)", 0.0, 3000.0)]:
    ev = table(thr); X = feats(ev.z.values, ev.stress.values, ev.mz.values)
    ev["pred"], C = walk_forward(ev, X, alpha)
    o_ = ev.dropna(subset=["pred", "y"]); dm = o_.y - o_.groupby("stock").y.transform("mean")
    print(f"\n[{label}] events={len(ev)} OOS={len(o_)}  IC(pred,y)={np.corrcoef(o_.pred, dm)[0,1]:.4f}  hit={(np.sign(o_.pred)==np.sign(dm)).mean():.3f}")
    print("   mean ridge coefs:", dict(zip(names, C.mean(0).round(3))))
    o_ = o_.assign(pred=o_.pred)
    for c in [0, 5, 10]:
        o, mt = evaluate(weights(o_, "pred"), c); out[(f"MODEL {label}", c)] = mt
        if c == 5: out_pnl = o["net"]; out[("_pnl", label)] = o["net"]; ev.to_pickle(f"results/ev_{thr}.pkl")
# --- baselines through the same engine
rank_neutral = lambda S: S.sub(S.mean(axis=1), axis=0).pipe(lambda s: s.div(s.abs().sum(axis=1), axis=0)).fillna(0)
B = {"B1 1-day x-sec reversal (daily)": rank_neutral(-r.fillna(0)),
     "B2 12-1 x-sec momentum (monthly)": rank_neutral(px.pct_change(231).shift(21)).iloc[::21].reindex(r.index).ffill()}
ev2 = table(2.0); ev2 = ev2[ev2.date >= FIRST_TEST]
B["B3 extreme-move reversal, no regime"] = weights(ev2.assign(p=-np.sign(ev2.z)), "p")
B["B4 extreme-move continuation, no regime"] = weights(ev2.assign(p=np.sign(ev2.z)), "p")
for nm, W in B.items():
    for c in [0, 5, 10]: out[(nm, c)] = evaluate(W, c)[1]
T = pd.DataFrame({k: v for k, v in out.items() if k[0] != "_pnl"}).T
T.index.names = ["strategy", "cost_bps"]; pd.set_option("display.width", 250)
print("\n=== OOS 2004-2022, 10%% ex-ante vol target, market-hedged, costs per side ==="); print(T.round(3).to_string()); T.to_csv("results/walkforward_metrics.csv")
for lab in ["sparse |z|>=2", "dense (all stock-days)"]:
    s = out[("_pnl", lab)]; s = s[s.index >= FIRST_TEST]
    print(f"{lab:24s} @5bps  Sharpe CI95={np.round(boot_sharpe(s),2)}  PSR(>0)={psr(s):.3f}  DSR(N=20)={deflated_sr(s,20):.3f}")
pd.DataFrame({k[1]: v for k, v in out.items() if k[0] == "_pnl"}).to_pickle("results/model_pnl.pkl")

# --- placebo: permute (stress, mkt-shock) labels across dates (keeps stock shocks, destroys regime info) --
if "--placebo" in sys.argv:
    rng = np.random.default_rng(1); ev = pd.read_pickle("results/ev_2.0.pkl"); sh = []
    ev = ev.drop(columns="pred", errors="ignore")
    dates = ev.date.unique()
    for i in range(60):
        perm = dict(zip(dates, rng.permutation(dates))); d2 = ev.date.map(perm)
        S2 = d2.map(stress).values; M2 = d2.map(mz).values
        ok = ~np.isnan(S2); e2 = ev[ok].reset_index(drop=True)
        X2 = feats(e2.z.values, S2[ok], M2[ok]); e2["pred"], _ = walk_forward(e2, X2, 300.0)
        e2 = e2.dropna(subset=["pred", "y"]); o, mt = evaluate(weights(e2, "pred"), 5.0); sh.append(mt["sharpe"])
    real = out[("MODEL sparse |z|>=2", 5)]["sharpe"]
    print(f"\nPLACEBO (60 perms) Sharpe @5bps: mean={np.mean(sh):.2f} sd={np.std(sh):.2f}  real={real:.2f}  p(placebo>=real)={(np.array(sh)>=real).mean():.3f}")
