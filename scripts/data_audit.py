"""Data audit on the DEV panels (never touches the holdout). Writes results/data_audit.md"""
import numpy as np, pandas as pd
from oafbt.data.store import DataStore
from oafbt.data import universe as U
from oafbt import config as C

ds = DataStore()
px, ret, g = ds.px(), ds.ret(), ds.groups()
lines = ["# Data audit (dev panels, <= %s)\n" % C.DEV_END]

# 1. coverage per group
lines.append("## Coverage\n| group | n tickers | first dates (median) |\n|---|---|---|")
for grp in ["etf", "mf", "fut", "fx", "index", "crypto", "stocks"]:
    cols = [c for c in px.columns if g.get(c) == grp]
    f = px[cols].apply(lambda s: s.first_valid_index())
    lines.append(f"| {grp} | {len(cols)} | {pd.Series(f).dropna().median().date() if len(cols) else ''} |")

# 2. non-positive raw adjclose (before cleaning) -- from raw parquet
raw = pd.concat([pd.read_parquet(C.RAW / f"yahoo_{x}.parquet") for x in ["etf", "mf", "fut", "fx", "index", "crypto", "stocks"]])
bad = raw[raw.adjclose <= 0]
lines.append(f"\n## Non-positive adjusted closes in raw Yahoo data: {len(bad)} rows, tickers: {bad.ticker.value_counts().head(15).to_dict()}")

# 3. futures vs ETF: roll-gap diagnosis
pairs = [("ES=F", "SPY"), ("NQ=F", "QQQ"), ("ZN=F", "IEF"), ("ZB=F", "TLT"), ("GC=F", "GLD"), ("SI=F", "SLV"),
         ("CL=F", "USO"), ("NG=F", "UNG"), ("6E=F", "FXE"), ("6J=F", "FXY"), ("HG=F", "CPER"), ("ZC=F", "DBA")]
lines.append("\n## Futures (Yahoo continuous) vs ETF, daily returns, common dates\n| fut | etf | corr | ann.mean fut | ann.mean etf | ann.vol fut | ann.vol etf | #days |fut-etf|>3sd |\n|---|---|---|---|---|---|---|---|")
for f, e in pairs:
    if f in ret and e in ret:
        a = pd.concat([ret[f], ret[e]], axis=1).dropna()
        a = a[a.index >= "2007-01-01"]
        d = a[f] - a[e]
        lines.append(f"| {f} | {e} | {a.corr().iloc[0,1]:.3f} | {a[f].mean()*252:.1%} | {a[e].mean()*252:.1%} | {a[f].std()*np.sqrt(252):.1%} | {a[e].std()*np.sqrt(252):.1%} | {int((d.abs() > 3*d.std()).sum())} |")

# 4. stale prices: share of zero returns in last 5y of dev per group
lines.append("\n## Share of exactly-zero daily returns (2016-2020)\n| group | median | worst 5 |\n|---|---|---|")
r5 = ret.loc["2016":"2020"]
for grp in ["etf", "fut", "fx", "index", "stocks"]:
    cols = [c for c in r5.columns if g.get(c) == grp]
    z = ((r5[cols] == 0).sum() / r5[cols].notna().sum()).dropna()
    lines.append(f"| {grp} | {z.median():.2%} | {z.sort_values().tail(5).round(3).to_dict()} |")

# 5. extreme returns
lines.append("\n## Largest absolute daily returns (after cleaning)")
for grp in ["etf", "fut", "fx", "stocks"]:
    cols = [c for c in ret.columns if g.get(c) == grp]
    s = ret[cols].stack().dropna()
    top = s.abs().sort_values().tail(8)
    lines.append(f"- {grp}: " + ", ".join(f"{t}@{d.date()}={s.loc[(d, t)]:+.1%}" for d, t in top.index))

# 6. S&P membership coverage
M = ds.sp500()
cov = M.sum(axis=1)
lines.append(f"\n## S&P 500 point-in-time members WITH Yahoo data: " + ", ".join(f"{y}: {int(cov.loc[str(y)].mean())}" for y in [1996, 2000, 2004, 2008, 2012, 2016, 2020]))

# 7. rf
rf = ds.rf
lines.append(f"\n## Risk-free (annualised mean by decade): " + ", ".join(f"{d}s: {rf[(rf.index.year//10*10)==d].mean()*252:.2%}" for d in [1990, 2000, 2010]))
# 8. mutual-fund proxies vs their ETFs
lines.append("\n## Mutual-fund proxies vs ETFs (daily corr, ann. mean diff, common dates)")
for mf, etf in U.MF_PROXY.items():
    if etf and mf in ret and etf in ret:
        a = pd.concat([ret[mf], ret[etf]], axis=1).dropna()
        if len(a) > 250:
            lines.append(f"- {mf}->{etf}: corr {a.corr().iloc[0,1]:.3f}, mean diff {(a[mf]-a[etf]).mean()*252:+.2%}/yr, n={len(a)}, mf starts {px[mf].first_valid_index().date()}")
# 9. vol ETPs
for t in ["VXX", "VIXY", "SVXY", "VIXM", "VXZ"]:
    s_ = px[t].dropna()
    if len(s_):
        lines.append(f"- {t}: {s_.index[0].date()} -> {s_.index[-1].date()}, n={len(s_)}, ann.mean ret {ret[t].mean()*252:+.1%}")
open(C.RESULTS / "data_audit.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
