import itertools, numpy as np, pandas as pd
from common import *
cal = nyse_cal()
R = rets(["SPY","TLT"]); rf = rf_daily()
X = R.sub(rf, axis=0)
lr = np.log1p(R["SPY"]) - np.log1p(R["TLT"])
per = cal.to_period("M")
s = pd.Series(np.arange(len(cal)), index=cal)
fe = s.groupby(per).transform(lambda x: x - x.max() - 1)      # -1 = last session of month
fs = s.groupby(per).transform(lambda x: x - x.min() + 1)      # +1 = first session
# drop the incomplete final month's end labels (no look at future): last month end unknown
last = cal[-1]; 
mtd = lr.groupby(per).cumsum()
vol = np.sqrt((X**2).ewm(halflife=63, min_periods=20).mean()) * np.sqrt(252)
vol = vol.shift(1)    # known at start of the day (decided previous close)

def build(sig_day=-4, late=(-2,-1), early=(1,2), z_months=36, z_cap=2.0, z_min=0.5, placebo_shift=0, zperm=None):
    # z measured at the close of month-day `sig_day` (relative to month end); the position starts after that close
    m = fe == sig_day
    meas = mtd.where(m).dropna()
    sd = meas.rolling(z_months, min_periods=24).std().shift(1)
    z = (meas / sd).clip(-z_cap, z_cap)
    z = z.where(z.abs() >= z_min, 0.0)
    if zperm is not None:     # permutation null: shuffle z across months
        z = pd.Series(zperm.permutation(z.values), index=z.index)
    # assign each day the z of the month whose signal day is the most recent one at/before the PREVIOUS close
    zday = z.reindex(cal).ffill(limit=40)
    zprev = zday.shift(1)                       # information known at previous close
    sgn = pd.Series(0.0, index=cal)
    L = fe.isin([d + placebo_shift for d in late]) 
    E = fs.isin([d + placebo_shift for d in early])
    # early leg uses the z of the month that just ended: at prev close (day -1 or +1) zday still holds that month
    sgn[L.values] = -1.0
    sgn[E.values] = 1.0
    expo = sgn * zprev
    # NB: for the early leg zprev would flip to NEW month's z only at its day sig_day, far later. ok.
    pos = pd.DataFrame({"SPY": expo / vol["SPY"], "TLT": -expo / vol["TLT"]}, index=cal)
    return pos.replace([np.inf,-np.inf],np.nan).fillna(0.0)

COST = [2.0, 2.0]; BORR = [0, 30.0]
def pnl(pos, scale=1.0):
    net, gross, turn = run_positions(pos*scale, X, COST, BORR)
    return net, gross, turn

def calib(pos, start, end):
    net, _, _ = pnl(pos)
    v = net[start:end].std() * np.sqrt(252)
    return 0.10 / v if v > 0 else 0.0

TR = ("2004-06-01","2013-12-31"); TE = ("2014-01-01","2026-11-13")
DEV = ("2004-06-01","2020-12-31"); HO = ("2021-01-01","2026-11-13")

def evaluate(p, label):
    pos = build(**p); sc = calib(pos, *TR)
    net, gross, turn = pnl(pos, sc)
    out = []
    for nm, w in [("train 04-13", TR), ("TEST 14-26", TE), ("dev 04-20", DEV), ("holdout 21-26", HO)]:
        d = stats(net[w[0]:w[1]], f"{label} {nm}"); d["gross_SR"] = round(sharpe(gross[w[0]:w[1]]),2); out.append(d)
    return out, net

if __name__ == "__main__":
    base = dict()   # author's default
    rows, net0 = evaluate(base, "author-params")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nYearly returns (author params, scaled in-train):")
    print((net0.groupby(net0.index.year).apply(lambda x: (1+x).prod()-1)).round(3).to_dict())

    # --- refit in training window only: grid
    grid = list(itertools.product([-6,-4,-3], [(-2,-1),(-3,-2,-1),(-1,)], [(1,2),(1,),(1,2,3)], [24,36,60], [0.0,0.5,1.0]))
    res = []
    for sd_, late, early, zm, zmin in grid:
        p = dict(sig_day=sd_, late=late, early=early, z_months=zm, z_min=zmin)
        pos = build(**p); sc = calib(pos, *TR)
        net,_,_ = pnl(pos, sc)
        res.append(dict(**{k:str(v) for k,v in p.items()}, SR_train=sharpe(net[TR[0]:TR[1]]), SR_test=sharpe(net[TE[0]:TE[1]])))
    G = pd.DataFrame(res)
    print("\nGrid:", len(G), "configs")
    best = G.sort_values("SR_train", ascending=False).head(5)
    print("Top-5 by TRAIN Sharpe (with their TEST Sharpe):\n", best.to_string(index=False))
    print("\nmedian grid SR train %.2f test %.2f ; share of configs with SR_test>0: %.0f%%; corr(train,test)=%.2f" %
          (G.SR_train.median(), G.SR_test.median(), 100*(G.SR_test>0).mean(), G[["SR_train","SR_test"]].corr().iloc[0,1]))

    # --- placebo windows: same z, shifted calendar windows
    print("\nPlacebo: apply the same rule to windows shifted away from month end (shift = days)")
    for sh in [-12,-8,-5,0,5,8,12]:
        pos = build(placebo_shift=sh); sc = 1.0
        net,_,_ = pnl(pos, calib(pos,*TR))
        print(f"  shift {sh:+d}: SR train {sharpe(net[TR[0]:TR[1]]):.2f}  test {sharpe(net[TE[0]:TE[1]]):.2f}  gross-all {sharpe(net['2004-06-01':]):.2f}")

    # --- permutation: shuffle z across months, 500 draws; compare to real SR in test
    rng = np.random.default_rng(0)
    real_pos = build(); sc = calib(real_pos,*TR); real = sharpe(pnl(real_pos,sc)[0][TE[0]:TE[1]])
    real_all = sharpe(pnl(real_pos,sc)[0]["2004-06-01":])
    null=[]; null_all=[]
    for i in range(300):
        pos = build(zperm=rng); net,_,_ = pnl(pos, sc)
        null.append(sharpe(net[TE[0]:TE[1]])); null_all.append(sharpe(net["2004-06-01":]))
    print(f"\nPermutation (shuffle month signals): real TEST SR {real:.2f}; null mean {np.mean(null):.2f} sd {np.std(null):.2f}; p={np.mean(np.array(null)>=real):.3f}")
    print(f"                          real FULL SR {real_all:.2f}; null mean {np.mean(null_all):.2f} sd {np.std(null_all):.2f}; p={np.mean(np.array(null_all)>=real_all):.3f}")
