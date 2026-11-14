import warnings; warnings.filterwarnings("ignore")
import pandas as pd, yfinance as yf
from .universe import TICKERS

def fetch(start="2005-01-01", out="data/ohlcv.pkl", batch=80):
    u = list(dict.fromkeys(TICKERS)); frames = []
    for i in range(0, len(u), batch):
        d = yf.download(u[i:i+batch], start=start, auto_adjust=False, progress=False, threads=True)
        frames.append(d)
    d = pd.concat(frames, axis=1)
    d = d.loc[:, ~d.columns.duplicated()]
    f = d["Adj Close"] / d["Close"]                           # split+dividend adjustment factor
    o = {"open": d["Open"] * f, "high": d["High"] * f, "low": d["Low"] * f, "close": d["Adj Close"], "vol": d["Volume"], "rawclose": d["Close"]}
    o = {k: v.dropna(axis=1, how="all") for k, v in o.items()}
    pd.to_pickle(o, out); return o, u
if __name__ == "__main__":
    o, u = fetch(); c = o["close"]
    print("requested", len(u), "returned", c.shape[1], "range", c.index[0].date(), c.index[-1].date())
    print("history start dist:", c.apply(lambda s: s.first_valid_index().year).value_counts().sort_index().to_dict())
    print("last obs before today (delisted proxy):", int((c.apply(lambda s: s.last_valid_index()) < c.index[-5]).sum()))
