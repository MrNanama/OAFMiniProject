import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from common import wide, sharpe, stats, run_positions
etf=wide("etf"); mf=wide("mf"); idx=wide("index","close"); fut=wide("fut")
P=pd.concat([etf, mf.drop(columns=[c for c in mf.columns if c in etf.columns],errors="ignore")],axis=1)
irx=idx["^IRX"]
def prep(a,b,start=None,end=None,src=None):
    S=src if src is not None else P
    px=S[[a,b]].dropna()
    if start: px=px[start:]
    if end: px=px[:end]
    r=px.pct_change(fill_method=None).iloc[1:]
    rf=(irx.reindex(r.index).ffill()/100/252).shift(1).fillna(0)
    return r.sub(rf,axis=0)
def build(X, sig_day=-4, late=(-2,-1), early=(1,2), z_months=36, z_cap=2.0, z_min=0.5, zperm=None):
    cal=X.index; per=cal.to_period("M"); s=pd.Series(np.arange(len(cal)),index=cal)
    fe=s.groupby(per).transform(lambda x:x-x.max()-1); fs=s.groupby(per).transform(lambda x:x-x.min()+1)
    fe[per==per[-1]]=np.nan
    a,b=X.columns
    lr=np.log1p(X[a])-np.log1p(X[b]); mtd=lr.groupby(per).cumsum()
    meas=mtd.where(fe==sig_day).dropna(); sd=meas.rolling(z_months,min_periods=24).std().shift(1)
    z=(meas/sd).clip(-z_cap,z_cap); z=z.where(z.abs()>=z_min,0.0)
    if zperm is not None: z=pd.Series(zperm.permutation(z.values),index=z.index)
    zprev=z.reindex(cal).ffill(limit=40).shift(1)
    sgn=pd.Series(0.0,index=cal); sgn[fe.isin(late).values]=-1.0; sgn[fs.isin(early).values]=1.0
    vol=(np.sqrt((X**2).ewm(halflife=63,min_periods=20).mean())*np.sqrt(252)).shift(1)
    e=sgn*zprev
    return pd.DataFrame({a:e/vol[a],b:-e/vol[b]},index=cal).replace([np.inf,-np.inf],np.nan).fillna(0.0)
def evaluate(a,b,start=None,end=None,cost=2.0,nperm=150,src=None,label=None):
    X=prep(a,b,start,end,src)
    if len(X)<1500: return None
    pos=build(X); net0,_,_=run_positions(pos,X,[cost,cost],[30,30]); 
    v=net0.std()*np.sqrt(252); sc=0.10/v if v>0 else 0
    net,_,_=run_positions(pos*sc,X,[cost,cost],[30,30])
    rng=np.random.default_rng(0); null=[]
    for _ in range(nperm):
        p2=build(X,zperm=rng); n2,_,_=run_positions(p2*sc,X,[cost,cost],[30,30]); null.append(sharpe(n2))
    sr=sharpe(net); p=float(np.mean(np.array(null)>=sr))
    h=len(net)//2
    return dict(pair=label or f"{a}/{b}",span=f"{X.index[0].year}-{X.index[-1].year}",SR=round(sr,2),SR_1st_half=round(sharpe(net.iloc[:h]),2),SR_2nd_half=round(sharpe(net.iloc[h:]),2),perm_p=round(p,3),null_mean=round(np.mean(null),2),maxDD=stats(net)["maxDD"])
if __name__=='__main__':
    rows=[]
    print("=== A. Years the author's strategy was never tuned on: mutual-fund proxies, pre-ETF era ===")
    for a,b,s,e in [("VFINX","VUSTX","1987-01-01","2003-12-31"),("VFINX","VFITX","1992-01-01","2003-12-31"),("VFINX","VUSTX","1987-01-01","2026-11-13")]:
        r=evaluate(a,b,s,e,cost=2.0); 
        if r: rows.append(r); print(r,flush=True)
    print("\n=== B. Other stock/bond pairs, same rule, no tuning (ETF era) ===")
    for a,b in [("SPY","IEF"),("SPY","AGG"),("SPY","LQD"),("QQQ","TLT"),("IWM","TLT"),("EFA","TLT"),("EEM","TLT"),("VGK","TLT"),("EWJ","TLT"),("DIA","TLT"),("SPY","TLH")]:
        r=evaluate(a,b,"2004-06-01",None,cost=3.0 if a in("EWJ","EEM","VGK","IWM") else 2.0)
        if r: rows.append(r); print(r,flush=True)
    print("\n=== C. Negative controls (should show NOTHING if the rebalancing story is right) ===")
    for a,b in [("SPY","GLD"),("SPY","USO"),("GLD","TLT"),("GLD","SLV"),("XLE","XLU"),("SPY","UUP")]:
        r=evaluate(a,b,"2007-01-01",None,cost=4.0)
        if r: rows.append(r); print(r,flush=True)
    pd.DataFrame(rows).to_csv("s15_markets.csv",index=False)
