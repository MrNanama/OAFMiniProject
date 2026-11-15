import numpy as np, pandas as pd
from common import *
pd.set_option("display.width",220)
cal = nyse_cal(); rf = rf_daily()
etf = wide("etf"); fut = wide("fut")
# ================= s11 commodity carry (independent) =================
COMMOD = {"USO":"CL=F","UNG":"NG=F","GLD":"GC=F","SLV":"SI=F","CPER":"HG=F","UGA":"RB=F","PPLT":"PL=F","PALL":"PA=F"}
E = etf[list(COMMOD)].reindex(cal).pct_change(fill_method=None).sub(rf,axis=0)
F = fut[list(COMMOD.values())].reindex(cal).pct_change(fill_method=None); F.columns=list(COMMOD)
gap = (E-F)
carry = gap.rolling(252,min_periods=200).mean()*252
vol = np.sqrt((E**2).ewm(halflife=63,min_periods=40).mean())*np.sqrt(252)
mends = pd.Series(cal,index=cal).groupby(cal.to_period("M")).max().values
def s11(borrow=50.0, cost=4.0, lag=2, borrow_map=None, drop=()):
    cols=[c for c in COMMOD if c not in drop]
    W = pd.DataFrame(0.0,index=cal,columns=cols)
    for d in mends:
        c = carry.loc[d,cols].where(E.loc[:d,cols].notna().iloc[-1] & vol.loc[d,cols].notna()).dropna()
        if len(c)<3: continue
        r = c.rank(); r = r - r.mean()
        w = r/vol.loc[d,c.index]
        lg, sh = w.clip(lower=0), w.clip(upper=0)
        if lg.sum()==0 or sh.sum()==0: continue
        w = (lg/lg.sum() + sh/(-sh.sum()))/2
        W.loc[d,w.index]=w.values
    # rows only at month ends, hold between
    rows = W.index.isin(mends); Wh = W.copy(); Wh.loc[~rows] = np.nan; Wh = Wh.ffill().fillna(0.0)
    pos = Wh.shift(lag-1).fillna(0.0)    # decision close d, executes close d+1, earns from d+2
    bm = np.array([ (borrow_map or {}).get(c,borrow) for c in cols],float)
    net,gross,turn = run_positions(pos, E[cols].fillna(0.0), [cost]*len(cols), bm)
    return net,gross,pos
def rep(net,label,spans):
    return pd.DataFrame([stats(net[a:b],f"{label} {n}") for n,(a,b) in spans.items()])
SP = {"train 2010-16":("2010-01-01","2016-12-31"),"test 2017-26":("2017-01-01","2026-11-13"),"dev 09-20":("2009-06-01","2020-12-31"),"holdout 21-26":("2021-01-01","2026-11-13")}
net,gross,pos = s11()
sc = 0.10/net["2010-01-01":"2016-12-31"].std()/np.sqrt(252)
print("scale to 10%% vol in train: %.2fx (mean gross after scale %.2f)"%(sc,(pos.abs().sum(axis=1)*sc).mean()))
print(rep(net*sc,"s11 base",SP).to_string(index=False))
print("gross SR (no cost):", {k:round(sharpe(gross[a:b]),2) for k,(a,b) in SP.items()})
# who earns it?  Per-asset contribution
contrib = (pos*E[pos.columns].fillna(0)).sum()
print("per-asset gross P&L (sum of daily, unscaled):\n", contrib.round(3).to_string())
# borrow realism: hard-to-borrow ETFs
for b in [50,300,800,1500]:
    bm={"UNG":b,"USO":b,"UGA":b}
    n,_,_=s11(borrow_map=bm)
    print(f"borrow on UNG/USO/UGA {b} bps: SR train {sharpe(n['2010-01-01':'2016-12-31']):.2f} test {sharpe(n['2017-01-01':]):.2f} holdout {sharpe(n['2021-01-01':]):.2f}")
n,_,_=s11(drop=("UNG",)); print("drop UNG: test SR %.2f holdout %.2f"%(sharpe(n['2017-01-01':]),sharpe(n['2021-01-01':])))
n,_,_=s11(drop=("UNG","UGA")); print("drop UNG,UGA: train %.2f test SR %.2f holdout %.2f"%(sharpe(n['2010-01-01':'2016-12-31']),sharpe(n['2017-01-01':]),sharpe(n['2021-01-01':])))
n,_,_=s11(cost=10.0); print("cost 10bps: test %.2f holdout %.2f"%(sharpe(n['2017-01-01':]),sharpe(n['2021-01-01':])))
print("yearly:", (1+net*sc).groupby(net.index.year).prod().sub(1).round(3).loc[2009:].to_dict())

# ================= s10 month-turn flows (independent, no FOMC leg) =================
R = etf[["SPY","TLT","IEF"]].reindex(cal).pct_change(fill_method=None).sub(rf,axis=0)
s = pd.Series(np.arange(len(cal)),index=cal); per=cal.to_period("M")
fe = s.groupby(per).transform(lambda x:x-x.max()-1); fs = s.groupby(per).transform(lambda x:x-x.min()+1)
# the final (incomplete) month has no true "last days"; drop its fe labels
fe[per==per[-1]] = np.nan
print("\n=== s10 event study: mean excess return per day, in window vs all other days ===")
def ev(asset, mask, a, b):
    x=R[asset][a:b].dropna(); m=mask.reindex(x.index).fillna(False)
    d=x[m]; o=x[~m]
    diff=d.mean()-o.mean(); se=np.sqrt(d.var()/len(d)+o.var()/len(o))
    return dict(asset=asset,n_days=len(d),mean_bps=round(d.mean()*1e4,1),other_bps=round(o.mean()*1e4,1),diff_t=round(diff/se,2))
bondm = fe.isin([-2,-1]); eqm = fs.isin([1,2,3])
rows=[]
for nm,(a,b) in {"train 04-13":("2004-01-01","2013-12-31"),"test 14-26":("2014-01-01","2026-11-13"),"holdout 21-26":("2021-01-01","2026-11-13")}.items():
    for asset,mask,lab in [("TLT",bondm,"bond days -2,-1"),("IEF",bondm,"bond days -2,-1"),("SPY",eqm,"equity days +1..+3")]:
        d=ev(asset,mask,a,b); d["window"]=lab; d["period"]=nm; rows.append(d)
print(pd.DataFrame(rows)[["period","asset","window","n_days","mean_bps","other_bps","diff_t"]].to_string(index=False))
# P&L of the (no-FOMC) strategy, inverse-vol weights, decided ahead; costs 2bps/side
vol2 = np.sqrt((R**2).ewm(halflife=63,min_periods=40).mean())*np.sqrt(252)
pos = pd.DataFrame(0.0,index=cal,columns=["SPY","TLT","IEF"])
pos["TLT"] = np.where(bondm,0.5/vol2["TLT"].shift(1),0.0); pos["IEF"]=np.where(bondm,0.5/vol2["IEF"].shift(1),0.0); pos["SPY"]=np.where(eqm,1.0/vol2["SPY"].shift(1),0.0)
pos=pos.fillna(0.0)
net,gross,turn = run_positions(pos,R.fillna(0.0),[2,2,2],[0,0,0])
sc=0.10/net["2004-06-01":"2013-12-31"].std()/np.sqrt(252)
SP2={"train 04-13":("2004-06-01","2013-12-31"),"test 14-26":("2014-01-01","2026-11-13"),"holdout 21-26":("2021-01-01","2026-11-13")}
print(rep(net*sc,"s10(no FOMC)",SP2).to_string(index=False))
# placebo: shift windows by k sessions
for k in [-7,-4,0,5,9,13]:
    bm_=fe.isin([-2+k,-1+k]); em_=fs.isin([1+k,2+k,3+k])
    p=pd.DataFrame(0.0,index=cal,columns=["SPY","TLT","IEF"])
    p["TLT"]=np.where(bm_,0.5/vol2["TLT"].shift(1),0.0);p["IEF"]=np.where(bm_,0.5/vol2["IEF"].shift(1),0.0);p["SPY"]=np.where(em_,1.0/vol2["SPY"].shift(1),0.0)
    n,_,_=run_positions(p.fillna(0.0),R.fillna(0.0),[2,2,2],[0,0,0])
    print(f"  placebo shift {k:+d}: SR train {sharpe(n['2004-06-01':'2013-12-31']):.2f}  test {sharpe(n['2014-01-01':]):.2f}")
