import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

def load():
    o = pd.read_pickle("data/crypto.pkl"); return o["close"], o["vol"]

def universe(C, V, top=30, win=30, min_hist=90):
    dv = (C * V).rolling(win).mean().shift(1); ok = C.rolling(min_hist).count().shift(1) >= min_hist
    rk = dv.where(ok).rank(axis=1, ascending=False); return rk <= top

def run(C, U, k, kind, reb=7, q=5, cost_bps=10.0):
    r = C.pct_change(); lr = np.log(C)
    sig = (lr - lr.shift(k)) * (1 if kind == "mom" else -1)
    sig = sig.where(U)
    n = sig.notna().sum(axis=1); rk = sig.rank(axis=1, method="first")
    raw = (rk.gt(n - q, axis=0) & sig.notna()).astype(float) / q - (rk.le(q, axis=0) & sig.notna()).astype(float) / q
    raw = raw.where(n >= 2 * q)
    reb_mask = pd.Series(np.arange(len(C)) % reb == 0, C.index)
    W = raw.where(reb_mask, np.nan).ffill()
    gross = (W.shift(1) * r).sum(axis=1, min_count=1)
    turn = W.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turn * cost_bps / 1e4, gross, turn

def weights(sig, U, q=5, reb=7):
    sig = sig.where(U); n = sig.notna().sum(axis=1); rk = sig.rank(axis=1, method="first")
    L = (rk.gt(n - q, axis=0) & sig.notna()).astype(float) / q; S = (rk.le(q, axis=0) & sig.notna()).astype(float) / q
    ok = n >= 2 * q; m = pd.Series(np.arange(len(sig)) % reb == 0, sig.index)
    return L.where(ok).where(m, np.nan).ffill().fillna(0.0), S.where(ok).where(m, np.nan).ffill().fillna(0.0)

def pnl(WL, WS, r, cost_bps=10.0, fund=0.10):
    W = WL - WS; gross = (W.shift(1) * r.fillna(0)).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(0.0); short_notional = WS.shift(1).sum(axis=1)
    return gross - turn * cost_bps / 1e4 - short_notional * fund / 365

def beta_neutral(WL, WS, r, U, win=60):
    m = r.where(U).mean(axis=1); beta = r.rolling(win, min_periods=30).cov(m).div(m.rolling(win, min_periods=30).var(), axis=0).shift(0).clip(0.2, 3.0).fillna(1.0)
    bL = (WL * beta).sum(axis=1).replace(0, np.nan); bS = (WS * beta).sum(axis=1).replace(0, np.nan)
    L2 = WL.div(bL, axis=0).fillna(0); S2 = WS.div(bS, axis=0).fillna(0); g = (L2.sum(axis=1) + S2.sum(axis=1)).replace(0, np.nan)
    return L2.mul(2 / g, axis=0).fillna(0), S2.mul(2 / g, axis=0).fillna(0)

def state_scale(WL, WS, r, U, dd_thr=-0.25, win=90, factor=0.5):
    idx = (1 + r.where(U).mean(axis=1).fillna(0)).cumprod(); dd = idx / idx.rolling(win, min_periods=30).max() - 1
    s = pd.Series(np.where(dd < dd_thr, factor, 1.0), r.index); return WL.mul(s, axis=0), WS.mul(s, axis=0)

def vol_target(WL, WS, r, tv=0.20, win=60, cap=3.0):
    base = pnl(WL, WS, r, cost_bps=0.0, fund=0.0); lev = (tv / (base.rolling(win, min_periods=30).std() * np.sqrt(365))).clip(upper=cap).fillna(0.0)
    return WL.mul(lev, axis=0), WS.mul(lev, axis=0), lev
