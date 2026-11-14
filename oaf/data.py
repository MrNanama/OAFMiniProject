"""Data + feature construction. All features at row t use information up to and including close t;
rolling estimators are shifted so they never include day t (no look-ahead)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import skfolio.datasets as d

def load():
    px = d.load_sp500_dataset()          # 20 large-cap US stocks, adjusted daily closes 1990-2022
    mk = d.load_sp500_index().iloc[:, 0]  # S&P 500 index
    px, mk = px.align(mk, join="inner", axis=0)
    return px, mk

def features(px, mk, beta_win=252, vol_win=60, stress_win=21, rank_win=756):
    r = px.pct_change(); m = mk.pct_change()
    r, m = r.iloc[1:], m.iloc[1:]
    # trailing beta (uses t-1 and earlier)
    cov = r.rolling(beta_win).cov(m); var = m.rolling(beta_win).var()
    beta = cov.div(var, axis=0).shift(1)
    e = r - beta.mul(m, axis=0)                              # residual (market-neutral) return
    sig_e = e.rolling(vol_win).std().shift(1)                # idiosyncratic vol, known at t-1
    sig_r = r.rolling(vol_win).std().shift(1)
    z_e = e / sig_e                                          # idiosyncratic shock size
    z_r = r / sig_r                                          # total shock size
    mvol = m.rolling(stress_win).std().shift(1) * np.sqrt(252)
    # stress = trailing percentile rank of market vol (information at t-1)
    stress = mvol.rolling(rank_win, min_periods=252).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    mz = m / m.rolling(vol_win).std().shift(1)               # market move today in sigma
    return dict(r=r, m=m, beta=beta, e=e, sig_e=sig_e, z_e=z_e, z_r=z_r, mvol=mvol, stress=stress, mz=mz)
