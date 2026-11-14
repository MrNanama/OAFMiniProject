"""Run a Strategy through the engine, compute metrics, save artefacts, log the trial."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config as C
from ..data.store import DataStore
from . import ledger
from .backtest import backtest
from .metrics import perf_stats, drawdown

RUNS = C.RESULTS / "runs"


def benchmark(ds: DataStore, universe: str) -> pd.Series:
    if universe == "kf":
        return ds.kf("F-F_Research_Data_Factors_daily")["Mkt-RF"]
    return ds.xret(["SPY"])["SPY"]


def run(strategy, ds: DataStore | None = None, holdout: bool = False, save: bool = True, note: str = "",
        cost_mult: float = 1.0, lag: int | None = None, eval_start=None, eval_end=None, log: bool = True,
        keep_held: bool = False, tag: str | None = None) -> dict:
    ds = ds or DataStore(holdout=holdout)
    W = strategy.target_weights(ds)
    W = W.astype(float)
    used = (W.fillna(0.0) != 0).any()
    W = W.loc[:, used[used].index]
    X = strategy.asset_returns(ds, list(W.columns))
    meta = strategy.asset_meta(ds, list(W.columns))
    lev_cap = strategy.lev_cap or C.LEV_CAP[strategy.universe]
    res = backtest(W, X, cost_bps=meta["cost_bps"] * cost_mult, borrow_bps_yr=meta["borrow_bps"],
                   funded=meta["funded"], lag=strategy.lag if lag is None else lag,
                   vol_target=strategy.vol_target, vol_window=strategy.vol_window,
                   vol_halflife=strategy.vol_halflife, lev_cap=lev_cap, keep_held=keep_held)
    if holdout:
        start, end = eval_start or C.HOLDOUT_START, eval_end
    else:
        start, end = eval_start or C.EVAL_START[strategy.universe], eval_end or C.DEV_END
    tag = tag or ("holdout" if holdout else "dev")
    n_prev = ledger.n_trials(strategy.name, tag="dev")
    bench = benchmark(ds, strategy.universe)
    stats = perf_stats(res.net, bench=bench, turnover=res.turnover, cost=res.cost, gross_lev=res.gross_lev,
                       n_trials=max(n_prev, 1), start=start, end=end)
    stats.update({"strategy": strategy.name, "version": strategy.version, "universe": strategy.universe,
                  "cost_mult": cost_mult, "lag": res.info["lag"], "n_assets": int(W.shape[1]),
                  "params": strategy.params})
    if save:
        d = RUNS / strategy.name / (strategy.version + ("" if tag == "dev" else f"_{tag}"))
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps(stats, indent=1, default=str))
        pd.DataFrame({"net": res.net, "gross": res.gross, "cost": res.cost, "fees": res.fees,
                      "turnover": res.turnover, "gross_lev": res.gross_lev, "net_lev": res.net_lev,
                      "n_pos": res.n_pos}).to_parquet(d / "daily.parquet")
        tearsheet(res.net, bench, stats, d / "tearsheet.png", start, end)
    if log and cost_mult == 1.0 and (lag is None or lag == strategy.lag):
        ledger.log_trial(strategy.name, strategy.version, strategy.params, stats, note=note, tag=tag)
    return {"stats": stats, "res": res, "weights": W}


def tearsheet(net: pd.Series, bench: pd.Series, stats: dict, path, start=None, end=None):
    x = net.copy()
    if start is not None:
        x = x[x.index >= pd.Timestamp(start)]
    if end is not None:
        x = x[x.index <= pd.Timestamp(end)]
    nz = np.flatnonzero(x.to_numpy() != 0)
    if nz.size == 0:
        return
    x = x.iloc[nz[0]:]
    b = bench.reindex(x.index).fillna(0.0)
    b = b * (x.std() / b.std()) if b.std() > 0 else b
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
    (1 + x).cumprod().plot(ax=ax[0, 0], logy=True, label="strategy (net, excess)", color="#1f4e79")
    (1 + b).cumprod().plot(ax=ax[0, 0], logy=True, label="SPY / Mkt (vol-matched)", color="#aaaaaa")
    ax[0, 0].legend(loc="upper left")
    ax[0, 0].set_title(f"{stats.get('strategy')} {stats.get('version')}  SR={stats.get('sharpe', np.nan):.2f}  "
                       f"ann={stats.get('ann_ret', np.nan):.1%}  vol={stats.get('ann_vol', np.nan):.1%}")
    dd = drawdown(x)
    dd.plot(ax=ax[0, 1], color="#c0392b")
    ax[0, 1].fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.3)
    ax[0, 1].set_title(f"drawdown (max {stats.get('max_dd', np.nan):.1%})")
    rs = x.rolling(252).mean() / x.rolling(252).std() * np.sqrt(252)
    rs.plot(ax=ax[1, 0], color="#2e86c1")
    ax[1, 0].axhline(0, color="k", lw=0.6)
    ax[1, 0].set_title("rolling 1y Sharpe")
    y = x.resample("YE").apply(lambda s: (1 + s).prod() - 1)
    ax[1, 1].bar(y.index.year, y.values, color=np.where(y.values > 0, "#27ae60", "#c0392b"))
    ax[1, 1].set_title("calendar-year returns")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
