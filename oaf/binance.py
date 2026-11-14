"""Point-in-time crypto universe from Binance public data (data.binance.vision): includes delisted pairs."""
import io, re, zipfile, os, time, requests, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"; CDN = "https://data.binance.vision"
EXCL = re.compile(r"(UP|DOWN|BULL|BEAR)USDT$")
STABLE = {"USDC", "TUSD", "BUSD", "FDUSD", "USDP", "DAI", "EUR", "GBP", "AUD", "USDS", "UST", "USDSB", "SUSD", "PAX", "AEUR", "XUSD", "USD1", "EURI", "BFUSD", "RLUSD", "PAXG", "WBTC", "WBETH"}
ses = requests.Session()

def _get(url, params=None, tries=6):
    for a in range(tries):
        try:
            r = ses.get(url, params=params, timeout=60)
            if r.status_code in (200, 404): return r
        except Exception: pass
        time.sleep(2 * (a + 1))
    raise RuntimeError("failed " + url)

def _list(prefix, delim="/"):
    out, marker = [], ""
    while True:
        r = _get(S3, params=dict(delimiter=delim, prefix=prefix, marker=marker, **{"max-keys": 1000})); r.raise_for_status()
        txt = r.text; keys = re.findall(r"<Key>([^<]+)</Key>", txt) + re.findall(r"<Prefix>([^<]+)</Prefix>", txt.split("</Prefix>", 1)[1] if "</Prefix>" in txt else txt)
        out += keys; nm = re.search(r"<NextMarker>([^<]+)</NextMarker>", txt)
        if "<IsTruncated>true" in txt and nm: marker = nm.group(1)
        elif "<IsTruncated>true" in txt and keys: marker = keys[-1]
        else: break
    return list(dict.fromkeys(out))

def symbols():
    ps = _list("data/spot/monthly/klines/"); sy = [p.rstrip("/").split("/")[-1] for p in ps if p.startswith("data/spot/monthly/klines/") and p.count("/") >= 4]
    sy = [s for s in sy if s.endswith("USDT") and not EXCL.search(s) and s[:-4] not in STABLE]; return sorted(set(sy))

def _read(content):
    z = zipfile.ZipFile(io.BytesIO(content)); df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 1, 2, 3, 4, 5, 7], names=["t", "o", "h", "l", "c", "v", "qv"])
    if df.t.dtype == object or df.iloc[0, 0] == "open_time": df = df[pd.to_numeric(df.t, errors="coerce").notna()].astype(float)
    t = df.t.astype("int64"); t = t // 1000 if t.iloc[0] > 1e14 else t
    df.index = pd.to_datetime(t, unit="ms").dt.normalize(); return df.drop(columns="t")

def fetch_symbol(sym, retries=3):
    os.makedirs("data/binance", exist_ok=True); f = f"data/binance/{sym}.pkl"
    if os.path.exists(f): return pd.read_pickle(f)
    d = _fetch_symbol(sym, retries)
    if d is not None: d.to_pickle(f)
    return d

def _fetch_symbol(sym, retries=3):
    files = [k for k in _list(f"data/spot/monthly/klines/{sym}/1d/") if k.endswith(".zip")]
    months = {re.search(r"(\d{4}-\d{2})\.zip", k).group(1) for k in files}
    dfs = []
    def get(k):
        for a in range(retries):
            try:
                r = _get(f"{CDN}/{k}")
                if r.status_code == 200: return _read(r.content)
            except Exception: pass
        return None
    for k in files: dfs.append(get(k))
    last = max(months) if months else None                       # daily files for months after the last monthly file
    if last:
        dk = [k for k in _list(f"data/spot/daily/klines/{sym}/1d/") if k.endswith(".zip") and re.search(r"(\d{4}-\d{2})-\d{2}\.zip", k).group(1) > last]
        for k in dk: dfs.append(get(k))
    dfs = [d for d in dfs if d is not None]
    return pd.concat(dfs).sort_index().loc[lambda x: ~x.index.duplicated()] if dfs else None

def fetch_all(out="data/binance.pkl", workers=6):
    sy = symbols(); print("symbols:", len(sy), flush=True); res = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, (s, d) in enumerate(zip(sy, ex.map(fetch_symbol, sy))):
            if d is not None and len(d) > 30: res[s[:-4]] = d
            if i % 50 == 0: print(i, s, flush=True)
    C = pd.DataFrame({k: v.c for k, v in res.items()}); QV = pd.DataFrame({k: v.qv for k, v in res.items()}); O = pd.DataFrame({k: v.o for k, v in res.items()})
    pd.to_pickle(dict(close=C, qvol=QV, open=O), out); return C, QV
if __name__ == "__main__":
    C, QV = fetch_all(); print(C.shape, C.index[0].date(), C.index[-1].date())
    last = C.apply(lambda s: s.last_valid_index()); print("coins whose data ends >30d before end (delisted):", int((last < C.index[-30]).sum()), "of", C.shape[1])

def load(path="data/binance.pkl"):
    """Clean loader: drops out-of-range timestamps (KLAY->KAIA migration rows), returns close, quote-volume (USDT)."""
    o = pd.read_pickle(path); lo, hi = pd.Timestamp("2017-07-01"), pd.Timestamp("2026-12-31")
    C = o["close"]; C = C[(C.index >= lo) & (C.index <= hi)]; Q = o["qvol"].reindex(C.index)
    C.index = pd.DatetimeIndex(C.index).astype("datetime64[ns]"); Q.index = C.index
    return C, Q
