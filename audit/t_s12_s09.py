import numpy as np, pandas as pd
from common import *
pd.set_option("display.width",200)
# ---------------- s12 crypto trend (independent) ----------------
cr = wide("crypto").sort_index()
print("crypto data:", {c:(cr[c].first_valid_index().date(), cr[c].last_valid_index().date()) for c in cr})
r = cr.pct_change(fill_method=None)
r = r.loc["2014-10-01":]
def trend_sig(px, lbs=(21,63,126,252)):
    return sum((px/px.shift(L)-1>0).astype(float).where(px.shift(L).notna()) for L in lbs)/len(lbs)
def s12(assets, vol_target=None, lag=2, cost=15.0, long_flat_rule="any"):
    px = cr[assets].loc["2014-09-01":]; ret = px.pct_change(fill_method=None)
    sig = trend_sig(px)
    vol = ret.ewm(halflife=42, min_periods=30).std()*np.sqrt(365)
    raw = (sig/vol)
    raw = raw.where(raw.notna())
    w = raw.div(raw.abs().sum(axis=1).replace(0,np.nan), axis=0).fillna(0.0)   # unit gross, fully invested if any signal
    # weekly rebalance (Sunday), decisions at close, executed next close, earns from the following day (lag=2 days)
    wk = w.copy(); mask = wk.index.dayofweek!=6; wk[mask]=np.nan; wk=wk.ffill().fillna(0.0)
    if vol_target:
        pv = (wk.shift(1)*ret).sum(axis=1).rolling(90,min_periods=30).std()*np.sqrt(365)   # realized vol of the unscaled book, lagged
        sc = (vol_target/pv.shift(1)).clip(upper=1.5).fillna(0.0); wk = wk.mul(sc,axis=0)
    pos = wk.shift(lag-1)         # pos_t held over return t ; decision at close t-lag
    return pos.reindex(ret.index).fillna(0.0), ret
res=[]
for assets in (["BTC-USD"],["BTC-USD","ETH-USD"]):
    pos, ret = s12(assets)
    net,gross,turn = run_positions(pos, ret.fillna(0.0), [15.0]*len(assets), [0]*len(assets))
    bh = ret[assets[0]].fillna(0)
    for nm,(a,b) in {"2015-2019":("2015-10-01","2019-12-31"),"2020-2026":("2020-01-01","2026-11-13"),"2021-2026 (author holdout)":("2021-01-01","2026-11-13"),"full":("2015-10-01","2026-11-13")}.items():
        d=stats(net[a:b], f"trend {'+'.join(assets)} {nm}"); d["SR"]=round(sharpe(net[a:b],365),2); d["BH_BTC_SR"]=round(sharpe(bh[a:b],365),2)
        d["BH_maxDD"]=round(((1+bh[a:b]).cumprod()/(1+bh[a:b]).cumprod().cummax()-1).min(),2)
        d["corr_BTC"]=round(net[a:b].corr(bh[a:b]),2); d["avg_expo"]=round(pos[a:b].abs().sum(axis=1).mean(),2); res.append(d)
print(pd.DataFrame(res).to_string(index=False))
# yearly
pos, ret = s12(["BTC-USD","ETH-USD"]); net,_,_ = run_positions(pos, ret.fillna(0.0), [15,15],[0,0])
print("yearly net (unlevered long/flat):", (1+net).groupby(net.index.year).prod().sub(1).round(2).to_dict())
print("yearly BTC buy&hold:", (1+ret['BTC-USD'].fillna(0)).groupby(ret.index.year).prod().sub(1).round(2).to_dict())

# ---------------- s09 short VIXY ----------------
etf = wide("etf"); idx = wide("index","close"); cal = nyse_cal()
print("\nVIXY first/last", etf["VIXY"].first_valid_index().date(), etf["VIXY"].last_valid_index().date())
vix, vix3m = idx["^VIX"].reindex(cal), idx["^VIX3M"].reindex(cal)
spy = etf["SPY"].reindex(cal).pct_change()
vixy = etf["VIXY"].reindex(cal).pct_change(fill_method=None)
rf = rf_daily()
rv = spy.rolling(21).std()*np.sqrt(252)*100
on = ((vix/vix3m)<0.95) & (vix>rv) & vixy.notna()
vol = np.sqrt((vixy**2).ewm(halflife=42,min_periods=20).mean())*np.sqrt(252)
def s09(ratio=0.95, borrow=300.0, cost=10.0, lag=2, volt=0.10, cap=1.0):
    on = ((vix/vix3m)<ratio) & (vix>rv) & vixy.notna()
    w = (-(volt/vol).clip(upper=cap)).where(on, 0.0)
    pos = pd.DataFrame({"VIXY":w}).shift(lag-1).fillna(0.0)
    net,gross,turn = run_positions(pos, pd.DataFrame({"VIXY":vixy-rf}).fillna(0.0), [cost],[borrow])
    return net, pos
net,pos = s09()
rows=[]
for nm,(a,b) in {"2011-2020 (dev)":("2011-03-01","2020-12-31"),"2021-2026 (holdout)":("2021-01-01","2026-11-13"),"2011-2016":("2011-03-01","2016-12-31"),"2017-2020":("2017-01-01","2020-12-31"),"full":("2011-03-01","2026-11-13")}.items():
    d=stats(net[a:b],nm); d["avg_expo"]=round(pos[a:b].abs().mean().iloc[0],3); rows.append(d)
print(pd.DataFrame(rows).to_string(index=False))
print("yearly:", (1+net).groupby(net.index.year).prod().sub(1).round(3).to_dict())
print("worst 5 days:\n", net.nsmallest(5).round(4).to_string())
print("VIXY biggest 1-day rallies:", vixy.nlargest(5).round(3).to_dict())
for b in [300,1000,2000,4000]:
    n,_=s09(borrow=b); print(f"borrow {b} bps/yr: SR dev {sharpe(n['2011-03-01':'2020-12-31']):.2f}  holdout {sharpe(n['2021-01-01':]):.2f}")
for lg in [2,3]:
    n,_=s09(lag=lg); print(f"extra lag {lg}: SR dev {sharpe(n['2011-03-01':'2020-12-31']):.2f} holdout {sharpe(n['2021-01-01':]):.2f}")
# always short VIXY (no timing) for comparison
alw = pd.DataFrame({"VIXY": (-(0.10/vol).clip(upper=1.0)).where(vixy.notna(),0.0)}).shift(1).fillna(0.0)
na,_,_ = run_positions(alw, pd.DataFrame({"VIXY":vixy-rf}).fillna(0.0),[10],[300])
print("always-short VIXY: SR dev %.2f holdout %.2f maxDD %.2f"%(sharpe(na['2011-03-01':'2020-12-31']),sharpe(na['2021-01-01':]),stats(na['2011-03-01':])['maxDD']))
