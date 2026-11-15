import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings("ignore")
import t_s15_markets as M
from common import run_positions, sharpe, stats
F=pd.read_parquet("long_data.parquet")
def bond_ret(y, mat):          # y in decimal; reprice a par bond (coupon = prior yield) one day older at the new yield
    c=y.shift(1); T=mat-1/252
    def price(c,yy,T):
        n=2*T; a=(1-(1+yy/2)**(-n))/(yy/2)
        return c/2*a*100/1 + 100*(1+yy/2)**(-n)   # per 100 face, coupon c/2 semiannual
    p=price(c,y,T)/100.0
    return (p-1)+c/252
def make(start="1962-01-01", div=0.035, mat=10, ycol="DGS10", keep_cal=None):
    d=F[["GSPC",ycol,"DTB3"]].copy().dropna()
    d=d[start:] if start else d
    y=d[ycol]/100; rf=(d["DTB3"]/100/252)
    # drop days where yield unchanged mechanically? keep. Returns between consecutive common dates (multi-day over gaps)
    sp=d["GSPC"].pct_change()+div/252
    br=bond_ret(y,mat)
    X=pd.DataFrame({"SPX":sp-rf.shift(1),"B10":br-rf.shift(1)}).iloc[2:].dropna()
    return X
def run(X,cost=2.0,nperm=150,label=""):
    pos=M.build(X); n0,_,_=run_positions(pos,X,[cost,cost],[30,30]); sc=0.10/(n0.std()*np.sqrt(252))
    net,gross,_=run_positions(pos*sc,X,[cost,cost],[30,30])
    rng=np.random.default_rng(1); null=[]
    for _ in range(nperm):
        n2,_,_=run_positions(M.build(X,zperm=rng)*sc,X,[cost,cost],[30,30]); null.append(sharpe(n2))
    sr=sharpe(net); return net,sc,sr,float(np.mean(np.array(null)>=sr)),np.mean(null)
if __name__=="__main__":
    # ---- 1. validate proxy vs REAL ETFs on 2004-2026
    real=M.prep("SPY","IEF","2004-06-01"); syn=make("2004-06-01")
    j=real.index.intersection(syn.index)
    print("PROXY CHECK 2004-26: daily corr bond proxy vs IEF %.3f | stock proxy vs SPY %.3f"%(real.loc[j,"IEF"].corr(syn.loc[j,"B10"]),real.loc[j,"SPY"].corr(syn.loc[j,"SPX"])))
    print("  annual vol IEF %.1f%% vs proxy %.1f%% ; ann. excess return IEF %.1f%% vs proxy %.1f%%"%(real.IEF.std()*np.sqrt(252)*100,syn.B10.std()*np.sqrt(252)*100,real.IEF.mean()*252*100,syn.B10.mean()*252*100))
    nr,_,sr_r,pr,_=run(real,2.0); ns,_,sr_s,ps,_=run(syn,2.0)
    print("  Strategy 2004-26: REAL SPY/IEF Sharpe %.2f (p=%.3f)  |  PROXY SPX/B10 Sharpe %.2f (p=%.3f)  | daily P&L corr %.2f"%(sr_r,pr,sr_s,ps,nr.corr(ns)))
    # ---- 2. full history, by era
    X=make("1962-01-01")
    print("\nFULL proxy history %s..%s, %d days"%(X.index[0].date(),X.index[-1].date(),len(X)))
    eras={"1964-1978":("1964-01-01","1978-12-31"),"1979-1986":("1979-01-01","1986-12-31"),"1987-1999":("1987-01-01","1999-12-31"),"2000-2003":("2000-01-01","2003-12-31"),"2004-2026":("2004-01-01","2026-11-13")}
    rows=[]
    for cost in [2.0,10.0]:
        net,sc,sr,p,nm=run(X,cost)
        print(f"\n--- cost {cost:.0f} bps one-way: full-sample SR {sr:.2f}  perm p={p:.3f} (null mean {nm:.2f}) ---")
        for k,(a,b) in eras.items():
            z=net[a:b]; 
            # era-specific permutation p
            Xe=X[a:b]; 
            rows.append((cost,k,round(sharpe(z),2),round(z.mean()*252,3),stats(z)["maxDD"],round(z.mean()/(z.std()/np.sqrt(len(z))),2)))
            print(f"  {k}: SR {sharpe(z):5.2f}  ann.ret {z.mean()*252*100:5.1f}%  maxDD {stats(z)['maxDD']:.2f}  t={z.mean()/(z.std()/np.sqrt(len(z))):.2f}")
    # era permutations with fixed full-sample z shuffles restricted to era
    print("\nPermutation p-values per era (cost 2 bps; shuffle month signals within era):")
    for k,(a,b) in eras.items():
        Xe=X[a:b]; 
        if len(Xe)<1500: continue
        _,_,sr,p,nm=run(Xe,2.0,nperm=150); print(f"  {k}: SR {sr:.2f}  p={p:.3f}  null mean {nm:.2f}")
