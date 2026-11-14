import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from oaf.vrp import prep, strategy
from oaf.backtest import boot_sharpe
X = prep(); R = {}; W = {}
spec = {"bh": ("bh", (), "vix"), "vt_rv": ("voltarget", (), "rv"), "vt_vix": ("voltarget", (), "vix"), "mv_const_vix": ("mv_const", (), "vix"),
        "tilt_vrpr+rv (vix sizing)": ("reg", ("vrpr", "lrv"), "vix"), "tilt_vixlevel (vix sizing)": ("reg", ("lvix",), "vix")}
for k, (kind, feats, src) in spec.items(): R[k], W[k] = strategy(X, kind, feats or ("vrpr",), sig_src=src)
P = pd.DataFrame(R).dropna(); W = pd.DataFrame(W).reindex(P.index); P.to_pickle("results/vrp_pnl.pkl")
def sh(x): return x.mean() / x.std() * np.sqrt(252)
def mdd(x): e = (1 + x).cumprod(); return (e / e.cummax() - 1).min()
eras = {"1998-2026": ("1998", "2026"), "1998-2007": ("1998", "2007"), "2008-2016": ("2008", "2016"), "2017-2026": ("2017", "2026"), "2020-26": ("2020", "2026")}
T = pd.DataFrame({k: {e: sh(P[k].loc[a:b]) for e, (a, b) in eras.items()} for k in P}).T.round(2); T["maxDD"] = [round(mdd(P[k]), 2) for k in P]; T["vol%"] = (P.std() * np.sqrt(252) * 100).round(1).values
print(T.to_string()); T.to_csv("results/vrp_metrics.csv")
print("\navg exposure / turnover per yr:"); print(pd.DataFrame({"mean_w": W.mean(), "turn": W.diff().abs().mean() * 252}).round(2).T.to_string())
for a, b in (("vt_vix", "vt_rv"), ("tilt_vrpr+rv (vix sizing)", "vt_vix"), ("tilt_vrpr+rv (vix sizing)", "mv_const_vix")):
    d = P[a] - P[b]; print(f"{a} - {b}: ann {d.mean()*252*100:+.2f}%  t={d.mean()/d.std()*np.sqrt(len(d)):.2f}  diffSharpe CI {boot_sharpe(d, n=500, block=21).round(2)}")
