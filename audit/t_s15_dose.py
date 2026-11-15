import numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings("ignore")
import t_s15_markets as M
from common import run_positions
X=M.prep("SPY","TLT","2004-06-01")
pos=M.build(X); n0,_,_=run_positions(pos,X,[2,2],[0,30]); sc=0.10/(n0.std()*np.sqrt(252))
net,_,_=run_positions(pos*sc,X,[2,2],[0,30])
cal=X.index; per=cal.to_period("M"); s=pd.Series(np.arange(len(cal)),index=cal)
fs=s.groupby(per).transform(lambda x:x-x.min()+1); fe=s.groupby(per).transform(lambda x:x-x.max()-1)
sm=pd.Series([(p-1) if f<=2 else p for p,f in zip(per,fs)],index=cal)
qe=sm.map(lambda p:p.month in (3,6,9,12))
for nm,mask in [("quarter-end signal months",qe),("other months",~qe)]:
    mo=net[mask].groupby(sm[mask]).sum()
    print(f"{nm:28s}: months {len(mo):3d}  mean P&L/month {mo.mean()*1e4:6.1f} bps  t={mo.mean()/(mo.std()/np.sqrt(len(mo))):.2f}  hit {100*(mo>0).mean():.0f}%")
a,b=X.columns; lr=np.log1p(X[a])-np.log1p(X[b]); mtd=lr.groupby(per).cumsum()
meas=mtd.where(fe==-4).dropna(); z=(meas/meas.rolling(36,min_periods=24).std().shift(1)).abs()
mo=net.groupby(sm).sum(); mo.index=mo.index.astype(str); z.index=z.index.to_period("M").astype(str)
d=pd.concat([mo.rename("pnl"),z.rename("absz")],axis=1).dropna(); d=d[d.absz>=0.5]
d["bucket"]=pd.qcut(d.absz,3,labels=["small |z|","mid |z|","large |z|"])
print("\nDose-response (months with |z|>=0.5):")
print(d.groupby("bucket",observed=True).pnl.agg(n="size",mean_bps=lambda v:v.mean()*1e4,t=lambda v:v.mean()/(v.std()/np.sqrt(len(v))),hit=lambda v:100*(v>0).mean()).round(1).to_string())
