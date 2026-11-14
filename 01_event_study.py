"""Event study: what happens AFTER a large idiosyncratic move, conditioned on market stress / market-wide-ness?"""
import numpy as np, pandas as pd
from oaf.data import load, features
px, mk = load(); F = features(px, mk)
e, z, stress, mz = F["e"], F["z_e"], F["stress"], F["mz"]
H = [1, 2, 5, 10]
# forward residual return over next h days (t+1..t+h), signed so that + = continuation, - = reversal
e_dm = e - e.mean()   # remove each stock's unconditional residual drift (survivorship) 
fwd = {h: e_dm.rolling(h).sum().shift(-h) for h in H}
rows = []
THR = 2.0
big = (z.abs() >= THR)
for dt in z.index[big.any(axis=1)]:
    for s in z.columns[big.loc[dt]]:
        if np.isnan(stress.loc[dt, s] if False else stress.loc[dt]): continue
        rec = dict(date=dt, stock=s, z=z.loc[dt, s], sign=np.sign(z.loc[dt, s]), stress=stress.loc[dt],
                   mz=mz.loc[dt], mkt_move=abs(mz.loc[dt]),
                   share=abs(e.loc[dt, s]) / (abs(F["r"].loc[dt, s]) + 1e-12))
        for h in H: rec[f"f{h}"] = rec["sign"] * fwd[h].loc[dt, s] * 1e4
        rows.append(rec)
ev = pd.DataFrame(rows).dropna(subset=["f1"]); ev.to_pickle("results/events.pkl")
print(f"events: {len(ev)}  dates: {ev.date.nunique()}  ({ev.date.min().date()} to {ev.date.max().date()})")

def summ(df, label):
    out = {"n": len(df)}
    for h in H:
        g = df.groupby("date")[f"f{h}"].mean()      # cluster by date -> honest t-stat
        w = df.groupby("date")[f"f{h}"].size()
        out[f"f{h}_bps"] = df[f"f{h}"].mean()
        out[f"f{h}_t"] = g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 2 else np.nan
    return pd.Series(out, name=label)
tab = [summ(ev, "ALL |z|>=2")]
tab += [summ(ev[ev.sign > 0], "up moves"), summ(ev[ev.sign < 0], "down moves")]
tab += [summ(ev[ev.stress < .33], "calm mkt (stress<33%)"), summ(ev[(ev.stress >= .33) & (ev.stress < .67)], "mid mkt"), summ(ev[ev.stress >= .67], "stressed mkt (>67%)")]
tab += [summ(ev[ev.mkt_move < 1], "market flat (|m|<1σ)"), summ(ev[ev.mkt_move >= 2], "market-wide shock (|m|>=2σ)")]
tab += [summ(ev[(ev.stress < .33) & (ev.mkt_move < 1)], "calm & mkt flat  [IDIO NEWS]"),
        summ(ev[(ev.stress >= .67) & (ev.mkt_move >= 1.5) ], "stressed & mkt shock [LIQUIDITY]")]
tab += [summ(ev[(ev.stress >= .67) & (ev.mkt_move >= 1.5) & (ev.sign < 0)], "  liquidity & DOWN move"),
        summ(ev[(ev.stress >= .67) & (ev.mkt_move >= 1.5) & (ev.sign > 0)], "  liquidity & UP move")]
T = pd.DataFrame(tab).round(2); print(T.to_string()); T.to_csv("results/event_study.csv")
ev["half"]=np.where(ev.date<"2007-01-01","1991-2006","2007-2022")
print("\nStability across halves, [stress>=67% & |mkt|>=1.5σ] (LIQUIDITY) by direction, 5d:")
L=ev[(ev.stress>=.67)&(ev.mkt_move>=1.5)]
for h_,g in L.groupby("half"):
    for sgn,nm in [(-1,"down"),(1,"up")]:
        x=g[g.sign==sgn]; c=x.groupby("date").f5.mean(); print(f"  {h_} {nm:4s} n={len(x):4d} f5={x.f5.mean():7.1f}bps t={c.mean()/(c.std()/np.sqrt(len(c))):5.2f}")
print("Stability, [calm & mkt flat] IDIO NEWS by direction, 5d:")
L=ev[(ev.stress<.33)&(ev.mkt_move<1)]
for h_,g in L.groupby("half"):
    for sgn,nm in [(-1,"down"),(1,"up")]:
        x=g[g.sign==sgn]; c=x.groupby("date").f5.mean(); print(f"  {h_} {nm:4s} n={len(x):4d} f5={x.f5.mean():7.1f}bps t={c.mean()/(c.std()/np.sqrt(len(c))):5.2f}")
print("\n(+ bps = continuation, - bps = reversal. t-stats clustered by date.)")
