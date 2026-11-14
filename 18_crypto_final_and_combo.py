import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, itertools, os
from oaf.binance import load
from oaf.crypto import weights, pnl, state_scale, vol_target
from oaf.backtest import boot_sharpe
C, Q = load(); C = C.loc["2018-01-01":]; Q = Q.loc[C.index]; r = C.pct_change(); lr = np.log(C)
dv = Q.rolling(30, min_periods=15).mean().shift(1); U = (dv.where(C.rolling(90).count().shift(1) >= 90).rank(axis=1, ascending=False) <= 30)
def sh(x): x = x.dropna(); return x.mean() / x.std() * np.sqrt(365)
WL, WS = weights(lr - lr.shift(14), U); base = pnl(WL, WS, r)
# (a) sensitivity of the bear-state rule
rows = []
for thr, win, f in itertools.product((-0.15, -0.25, -0.35), (60, 90, 180), (0.0, 0.5)):
    a, b = state_scale(WL, WS, r, U, dd_thr=thr, win=win, factor=f); x = pnl(a, b, r); rows.append(dict(thr=thr, win=win, factor=f, Sharpe=sh(x), h1=sh(x.loc[:"2021"]), h2=sh(x.loc["2022":])))
G = pd.DataFrame(rows); print("bear-state rule grid (18 combos): Sharpe median %.2f min %.2f max %.2f | all > base (%.2f): %s | both halves beat base: %.2f" % (G.Sharpe.median(), G.Sharpe.min(), G.Sharpe.max(), sh(base), bool((G.Sharpe > sh(base)).all()), ((G.h1 > sh(base.loc[:'2021'])) & (G.h2 > sh(base.loc['2022':]))).mean()))
# (b) placebo: random exposure cuts with same frequency/persistence as bear-state
idx = (1 + r.where(U).mean(axis=1).fillna(0)).cumprod(); dd = idx / idx.rolling(90, min_periods=30).max() - 1; flag = (dd < -0.25).astype(int); frac = flag.mean()
rng = np.random.default_rng(0); gain = []; blocks = (flag.diff().fillna(0) != 0).sum() / 2
for i in range(300):
    st = np.zeros(len(flag)); t = 0
    while t < len(flag):                                   # alternate random-length on/off spells with same mean spell length
        on = rng.random() < frac; L = max(1, int(rng.exponential(len(flag) * frac / max(blocks, 1) if on else len(flag) * (1 - frac) / max(blocks, 1)))); st[t:t+L] = on; t += L
    s = pd.Series(np.where(st == 1, 0.5, 1.0), r.index); x = pnl(WL.mul(s, axis=0), WS.mul(s, axis=0), r); gain.append(sh(x) - sh(base))
a, b = state_scale(WL, WS, r, U); real = sh(pnl(a, b, r)) - sh(base)
print("bear-state real gain in Sharpe %+.2f | placebo random cuts: mean %+.2f sd %.2f p95 %+.2f | p-value %.3f | flagged %.0f%% of days" % (real, np.mean(gain), np.std(gain), np.percentile(gain, 95), (np.array(gain) >= real).mean(), frac * 100))
# (c) final book = mom14 + bear-state + vol-target 20%
a, b = state_scale(WL, WS, r, U); a, b, lev = vol_target(a, b, r); fin = pnl(a, b, r)
print("FINAL crypto book: Sharpe %.2f ann %.1f%% vol %.1f%% | by year:" % (sh(fin), fin.mean() * 365 * 100, fin.std() * np.sqrt(365) * 100), {y: round(sh(fin.loc[str(y)]), 2) for y in range(2018, 2027)})
print("block-bootstrap Sharpe CI (scaled to 365d):", (boot_sharpe(fin.dropna().rename(None), n=800, block=21) * np.sqrt(365 / 252)).round(2))
# ---- Step 4: combine with slow premia core (iteration 5), weekly, risk-parity 50/50 by vol
f = "results/combo_slow3.pkl"
if os.path.exists(f):
    slow = pd.read_pickle(f); wk = lambda x: (1 + x.fillna(0)).resample("W-FRI").prod() - 1
    cw, sw = wk(fin.loc["2018":]), wk(slow.loc["2018":]); j = pd.concat([cw, sw], axis=1, keys=["crypto", "slow3"]).dropna().loc[:"2026-09-25"]
    s52 = lambda x: x.mean() / x.std() * np.sqrt(52)
    print("\nweekly overlap", j.index[0].date(), j.index[-1].date(), "corr %.2f" % j.corr().iloc[0, 1])
    sc = j / j.std(); comb = sc.mean(axis=1) * j.std().mean()
    print("Sharpe: crypto %.2f | slow3 %.2f | 50/50 risk-parity %.2f" % (s52(j.crypto), s52(j.slow3), s52(comb)), "| 2022+: crypto %.2f slow3 %.2f combo %.2f" % (s52(j.crypto.loc['2022':]), s52(j.slow3.loc['2022':]), s52(comb.loc['2022':])))
    pd.DataFrame({"crypto": j.crypto, "slow3": j.slow3, "combo": comb}).to_pickle("results/combo_crypto_slow.pkl")
else: print("slow3 results not found on this branch")
fin.to_pickle("results/crypto_final_book.pkl")
