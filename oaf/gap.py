"""Iteration 3: overnight-gap drift. Signals use info through close t; positions earn from t+1."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

def load(path="data/ohlcv.pkl", min_cov=0.9):
    o = pd.read_pickle(path); C = o["close"]; C = C.loc[:, C.notna().mean() > min_cov]
    O = o["open"][C.columns]; return O, C

def signals(O, C, vol_win=60):
    on = O / C.shift() - 1; idr = C / O - 1; cc = C / C.shift() - 1
    vol = cc.rolling(vol_win).std().shift(1)
    zo = on.sub(on.median(axis=1), axis=0) / vol; zi = idr.sub(idr.median(axis=1), axis=0) / vol
    return dict(on=on, idr=idr, cc=cc, zo=zo, zi=zi)

def events(S, kind, thr=2.0, confirm=0.5):
    zo, zi = S["zo"], S["zi"]; sg = np.sign(zo); ev = zo.abs() > thr
    if kind == "all": m = ev
    elif kind == "fade": m = ev & (np.sign(zi) == -sg) & (zi.abs() > confirm)
    elif kind == "go": m = ev & (np.sign(zi) == sg) & (zi.abs() > confirm)
    elif kind == "down": m = ev & (sg < 0)
    return (sg * m).fillna(0.0)

def run(E, cc, hold=5, w=0.01, cost_bps=5.0, hedge_cost_bps=1.0, shift_sign=1.0):
    """Hold each event hold days (close t -> close t+hold), signed continuation, equal weight w of capital each, hedged with EW market."""
    W = sum(E.shift(k).fillna(0.0) for k in range(hold)) * w * shift_sign
    mkt = cc.mean(axis=1)
    H = -W.sum(axis=1)                                       # market hedge notional (beta~1)
    gross = (W.shift(1) * cc).sum(axis=1) + H.shift(1) * mkt
    turn = W.diff().abs().sum(axis=1).fillna(0.0); hturn = H.diff().abs().fillna(0.0)
    cost = turn * cost_bps / 1e4 + hturn * hedge_cost_bps / 1e4
    net = gross - cost
    return dict(net=net, gross=gross, cost=cost, gross_exp=W.abs().sum(axis=1), turnover=turn)
