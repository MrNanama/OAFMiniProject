import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0,'.')
import t_s15 as T
from common import sharpe, stats
def run(train_end, tests):
    TR=("2004-06-01",train_end); T.TR=TR
    grid=list(itertools.product([-6,-4,-3],[(-2,-1),(-3,-2,-1),(-1,)],[(1,2),(1,),(1,2,3)],[24,36,60],[0.0,0.5,1.0]))
    res=[]
    for sd_,late,early,zm,zmin in grid:
        p=dict(sig_day=sd_,late=late,early=early,z_months=zm,z_min=zmin)
        pos=T.build(**p); sc=T.calib(pos,*TR); net,_,_=T.pnl(pos,sc)
        r=dict(sig_day=sd_,late=str(late),early=str(early),zm=zm,zmin=zmin,SR_train=sharpe(net[TR[0]:TR[1]]))
        for k,(a,b) in tests.items(): r[k]=sharpe(net[a:b])
        res.append(r)
    G=pd.DataFrame(res); ks=list(tests)
    print(f"\n=== fit on 2004-06..{train_end} ({len(G)} configs) ===")
    top=G.sort_values("SR_train",ascending=False).head(5)
    print("Top-5 by TRAIN Sharpe:\n",top.round(2).to_string(index=False))
    print("\nBest-by-train config, test SRs:",{k:round(float(top.iloc[0][k]),2) for k in ks})
    print("Top-5 avg test SR:",{k:round(float(top[k].mean()),2) for k in ks})
    print("All-config median test SR:",{k:round(float(G[k].median()),2) for k in ks}, "| share >0:",{k:f"{100*(G[k]>0).mean():.0f}%" for k in ks})
    print("corr(train SR, test SR):",{k:round(float(G[['SR_train',k]].corr().iloc[0,1]),2) for k in ks})
    # author's default params under same fit window
    pos=T.build(); sc=T.calib(pos,*TR); net,_,_=T.pnl(pos,sc)
    print("Author's default params: train %.2f"%sharpe(net[TR[0]:TR[1]]),{k:round(sharpe(net[a:b]),2) for k,(a,b) in tests.items()})
    return G
run("2019-12-31",{"test 2020-26 (incl. covid)":("2020-01-01","2026-11-13"),"test 2021-26 (ex covid crash)":("2021-01-01","2026-11-13"),"test 2020 only":("2020-01-01","2020-12-31")})
run("2018-12-31",{"test 2019-26":("2019-01-01","2026-11-13"),"test 2021-26":("2021-01-01","2026-11-13")})
