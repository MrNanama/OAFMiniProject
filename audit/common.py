"""Independent test harness (does NOT import the author's engine). Raw Yahoo adjclose -> total returns."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
RAW = "/tmp/claude-0/audit/data/raw/"

def load(group):
    d = pd.read_parquet(RAW + f"yahoo_{group}.parquet")
    return d

def wide(group, field="adjclose"):
    d = load(group)
    return d.pivot_table(index="date", columns="ticker", values=field, aggfunc="last").sort_index()

def nyse_cal():
    adj = wide("etf")
    return adj["SPY"].dropna().index

def rets(tickers, group="etf"):
    cal = nyse_cal()
    p = wide(group)[tickers].reindex(cal)
    return p.pct_change(fill_method=None)      # NaN on holes; no ffill of returns

def rf_daily():
    cal = nyse_cal()
    irx = wide("index", "close")["^IRX"].reindex(cal).ffill()
    return (irx / 100 / 252).shift(1).fillna(0.0)    # lagged one session

def sharpe(x, ann=252):
    x = x.dropna()
    return np.nan if x.std() == 0 or len(x) < 30 else x.mean() / x.std() * np.sqrt(ann)

def stats(x, label=""):
    x = x.dropna()
    x = x[x.index >= x[x != 0].index[0]] if (x != 0).any() else x
    eq = (1 + x).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    n = len(x)
    t = x.mean() / (x.std() / np.sqrt(n)) if x.std() > 0 else np.nan
    return dict(label=label, SR=round(sharpe(x), 2), ann_ret=round(x.mean() * 252, 4), vol=round(x.std() * np.sqrt(252), 4),
                maxDD=round(dd, 3), t=round(t, 2), yrs=round(n / 252, 1))

def run_positions(pos, r, cost_bps, borrow_bps=0.0):
    """pos: target position (fraction of NAV) HELD OVER day t's return = decided at close t-1 and traded at close t-1.
    P&L_t = sum pos_t * r_t ; cost on |pos_t - drifted pos_{t-1}| approx |pos_t - pos_{t-1}|."""
    pos = pos.reindex(r.index).fillna(0.0)
    gross = (pos * r.fillna(0.0)).sum(axis=1)
    turn = pos.diff().abs().fillna(pos.abs())
    c = np.asarray(cost_bps) / 1e4
    cost = (turn * c).sum(axis=1)
    bor = (pos.clip(upper=0).abs() * (np.asarray(borrow_bps) / 1e4 / 252)).sum(axis=1)
    return gross - cost - bor, gross, turn.sum(axis=1)
