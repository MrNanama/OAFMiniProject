"""Iteration 5: fixed-rule slow premia sleeves. No fitted parameters. Signals use info through close t; weights earn t+1.."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

def month_ends(idx): return pd.Series(idx, idx).groupby(idx.to_period("M")).transform("max") == pd.Series(idx, idx)

def hold_monthly(raw, idx):
    """raw: weights computed daily (info thru t). Take value at month-end, hold for next month, return daily weights W[t] used for return of day t+1."""
    me = month_ends(idx).values
    W = raw.where(pd.Series(me, idx), np.nan).ffill()
    return W

def sleeve_pnl(W, R, cost_bps=0.0):
    pnl = (W.shift(1) * R).sum(axis=1, min_count=1)
    turn = W.diff().abs().sum(axis=1).fillna(0.0)
    return pnl - turn * cost_bps / 1e4, turn

def factor_momentum(F, cols, lb=252, vw=63, tvol=0.10):
    R = F[cols]; s = np.sign(np.log1p(R.fillna(0)).rolling(lb, min_periods=lb).sum())
    vol = R.rolling(vw).std() * np.sqrt(252)
    raw = (s * tvol / vol / len(cols)).where(R.notna() | True)
    raw = raw.where(R.rolling(lb, min_periods=lb).count() >= lb)
    return hold_monthly(raw, R.index), R

def industry_mom(I, lb=252, skip=21, q=10, tvol=None):
    L = np.log1p(I.fillna(0)); sig = L.rolling(lb).sum() - L.rolling(skip).sum()
    ok = I.rolling(lb, min_periods=lb).count() >= lb
    sig = sig.where(ok)
    rk = sig.rank(axis=1, method="first"); n = sig.notna().sum(axis=1)
    raw = (rk.gt(n - q, axis=0) & sig.notna()).astype(float) / q - (rk.le(q, axis=0) & sig.notna()).astype(float) / q
    raw = raw.where(n >= 3 * q)
    return hold_monthly(raw, I.index), I

def turn_of_month(F, after=3, before=1):
    idx = F.index; per = idx.to_period("M")
    pos = pd.Series(np.arange(len(idx)), idx).groupby(per).rank(method="first").values      # day-of-month index 1..
    cnt = pd.Series(1, idx).groupby(per).transform("sum").values
    inw = (pos <= after) | (pos > cnt - before)                                              # return of day t earned if in window
    W = pd.Series(inw.astype(float), idx).shift(-1)                                         # weight set at close t-1 earns day t -> W[t-1]=inw[t]
    return W.rename("MKT").to_frame(), F[["Mkt-RF"]].rename(columns={"Mkt-RF": "MKT"})

def vol_managed(F, win=21, cap=3.0, norm_win=1260):
    r = F["Mkt-RF"]; rv = r.rolling(win).var().shift(0) * 252
    raw = 1 / rv; base = raw.rolling(norm_win, min_periods=504).mean().shift(1)
    w = (raw / base).clip(upper=cap)
    return hold_monthly(w.to_frame("MKT"), F.index), r.to_frame("MKT")
