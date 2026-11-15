import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
V="/tmp/claude-0/audit/data/vault/"; RAW="/tmp/claude-0/audit/data/raw/"
ret=pd.read_parquet(V+"ret.parquet"); grp=pd.read_parquet(V+"groups.parquet")["group"]
rows=[]
for g in ["etf","mf","fut","crypto","stocks"]:
    d=pd.read_parquet(RAW+f"yahoo_{g}.parquet"); w=d.pivot_table(index="date",columns="ticker",values="adjclose",aggfunc="last").sort_index().reindex(ret.index)
    raw=(w/w.ffill().shift(1)-1).where(w.notna())
    cols=[c for c in raw.columns if c in ret.columns and grp.get(c)==g]
    R=raw[cols]; C_=ret[cols]
    m=((C_-R).abs()>1e-6) | (C_.isna() & R.notna())
    n=int(m.to_numpy().sum()); tot=int(R.notna().to_numpy().sum())
    # direction: among modified cells, were raw returns mostly big negatives removed (flattering) or big positives removed?
    rv=R.where(m).to_numpy().ravel(); rv=rv[~np.isnan(rv)]; cv=C_.where(m).to_numpy().ravel(); cv=cv[~np.isnan(cv)]
    n_neg=int((rv<-0.05).sum()); n_pos=int((rv>0.05).sum())
    rows.append(dict(group=g,modified_cells=n,of_total=tot,pct=round(100*n/tot,4),raw_big_neg_removed=n_neg,raw_big_pos_removed=n_pos,sum_raw=round(np.nansum(rv),2),sum_clean=round(np.nansum(C_.where(m).to_numpy()),2)))
    if g!="stocks":
        s=pd.DataFrame({"raw":R.where(m).stack(),"clean":C_.where(m).stack()}); print(f"\n[{g}] modified cells:\n",s.round(3).to_string())
print(pd.DataFrame(rows).to_string(index=False))
