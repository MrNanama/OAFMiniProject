"""Path-dependent daily portfolio backtester.

Timing convention (no look-ahead)
---------------------------------
* A strategy returns TARGET weights (fraction of NAV) indexed by DECISION date d. A row may only use
  information available at the close of d.
* Rows that are entirely NaN mean "no rebalance" (positions drift with prices). Within a rebalance row,
  NaN means 0.
* The target decided at d is EXECUTED at the close of d + lag trading days (default lag=1, i.e. a full day of
  slippage-free delay; lag=0 = trade at the same close the signal was computed on, which is optimistic).
* Holdings executed at the close of t earn the return of day t+1 onwards.

Accounting (everything in EXCESS-of-cash terms)
------------------------------------------------
gross_t = sum_i h_{t-1,i} * xret_{t,i}          xret = asset return minus rf for funded assets (ETFs, stocks),
                                                 raw return for unfunded ones (futures, FX, long-short factors)
fees_t  = borrow fee on short notional + leverage spread on funded long notional above 1x (daily accrual)
cost_t  = sum_i cost_bps_i * |trade_{t,i}|       trade = new target - drifted holdings
net_t   = gross_t - fees_t - cost_t

Volatility targeting (optional, default on)
-------------------------------------------
At each decision date d the target vector w_d is scaled so that its EX-ANTE volatility, estimated by applying
w_d to the trailing `vol_window` days of asset returns (exponentially weighted, information <= d), equals
`vol_target`; gross leverage is capped at `lev_cap`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TD = 252


@dataclass
class BTResult:
    net: pd.Series
    gross: pd.Series
    cost: pd.Series
    fees: pd.Series
    turnover: pd.Series
    gross_lev: pd.Series
    net_lev: pd.Series
    n_pos: pd.Series
    scale: pd.Series
    held: pd.DataFrame | None = None
    info: dict = field(default_factory=dict)


def exante_vol(target: pd.DataFrame, xret: pd.DataFrame, window: int = 126, halflife: float = 42,
               min_obs: int = 40) -> pd.Series:
    """Annualised ex-ante vol of each rebalance row of `target`, by historical simulation over the trailing
    window ending AT the decision date (inclusive). Returns NaN where history is too short."""
    R = xret.reindex(columns=target.columns).fillna(0.0).to_numpy()
    avail = xret.reindex(columns=target.columns).notna().to_numpy()
    pos = {d: i for i, d in enumerate(xret.index)}
    lam = 0.5 ** (1.0 / halflife)
    out = pd.Series(np.nan, index=target.index)
    W = target.to_numpy()
    for k, d in enumerate(target.index):
        w = W[k]
        if np.all(np.isnan(w)):
            continue
        w = np.nan_to_num(w)
        if not np.any(w):
            out.iloc[k] = 0.0
            continue
        i = pos.get(d)
        if i is None:
            continue
        lo = max(0, i - window + 1)
        Rw = R[lo:i + 1]
        if Rw.shape[0] < min_obs:
            continue
        # only count days on which the held assets actually had data
        live = avail[lo:i + 1][:, w != 0].any(axis=1)
        p = Rw @ w
        p = p[live]
        if p.size < min_obs:
            continue
        ew = lam ** np.arange(p.size)[::-1]
        out.iloc[k] = np.sqrt(np.sum(ew * p * p) / np.sum(ew) * TD)
    return out


def backtest(target: pd.DataFrame, xret: pd.DataFrame, cost_bps: pd.Series | float = 0.0,
             borrow_bps_yr: pd.Series | float = 0.0, funded: pd.Series | bool = True, lag: int = 1,
             vol_target: float | None = 0.10, vol_window: int = 126, vol_halflife: float = 42,
             lev_cap: float = 3.0, lever_spread_bps_yr: float = 50.0, drift: bool = True,
             keep_held: bool = False, max_scale: float = 10.0) -> BTResult:
    """See module docstring. `xret` defines the calendar; `target` is reindexed to it (decision dates that are
    not trading days are moved to the previous trading day's close is NOT done - they are dropped with a warning)."""
    cols = list(target.columns)
    xret = xret.reindex(columns=cols)
    cal = xret.index
    tgt = target.reindex(cal)
    dropped = target.index.difference(cal)
    if len(dropped):
        # move each off-calendar decision to the NEXT trading day (conservative: decided later)
        nxt = cal.searchsorted(dropped)
        for d, j in zip(dropped, nxt):
            if j < len(cal) and tgt.iloc[j].isna().all():
                tgt.iloc[j] = target.loc[d].values
    is_reb = tgt.notna().any(axis=1)

    # ---- volatility targeting at decision dates
    scale = pd.Series(np.nan, index=cal)
    if vol_target is not None:
        reb = tgt[is_reb].fillna(0.0)
        v = exante_vol(reb, xret, window=vol_window, halflife=vol_halflife)
        gross = reb.abs().sum(axis=1)
        s = (vol_target / v).clip(upper=max_scale)
        s = np.minimum(s, lev_cap / gross.replace(0, np.nan))
        s = s.fillna(0.0)                       # not enough history -> stay flat
        scale.loc[reb.index] = s
        tgt.loc[is_reb] = reb.mul(s, axis=0)
    else:
        reb = tgt[is_reb].fillna(0.0)
        gross = reb.abs().sum(axis=1)
        s = np.minimum(1.0, lev_cap / gross.replace(0, np.nan)).fillna(1.0)
        scale.loc[reb.index] = s
        tgt.loc[is_reb] = reb.mul(s, axis=0)

    # ---- execution lag
    exe = tgt.shift(lag)
    exe_reb = exe.notna().any(axis=1).to_numpy()
    E = exe.fillna(0.0).to_numpy()

    X = xret.to_numpy()
    avail = ~np.isnan(X)
    X0 = np.nan_to_num(X)
    # last valid date per asset (to force liquidation of positions in assets whose data ends)
    last_valid = np.full(len(cols), -1)
    for j in range(len(cols)):
        idx = np.flatnonzero(avail[:, j])
        if idx.size:
            last_valid[j] = idx[-1]

    def as_vec(x, default):
        if isinstance(x, pd.Series):
            return x.reindex(cols).fillna(default).to_numpy(dtype=float)
        return np.full(len(cols), float(x))

    c = as_vec(cost_bps, 0.0) / 1e4
    b = as_vec(borrow_bps_yr, 0.0) / 1e4 / TD
    f = as_vec(funded, 1.0).astype(bool) if not isinstance(funded, bool) else np.full(len(cols), funded)
    ls = lever_spread_bps_yr / 1e4 / TD

    T, N = X.shape
    h = np.zeros(N)
    out = np.zeros((T, 7))  # gross, fees, cost, turnover, glev, nlev, npos
    H = np.zeros((T, N)) if keep_held else None
    for t in range(T):
        r = X0[t]
        g = float(h @ r)
        fee = float(b @ np.maximum(-h, 0.0)) + ls * max(float(np.maximum(h, 0.0)[f].sum()) - 1.0, 0.0)
        if drift:
            denom = 1.0 + g
            hp = h * (1.0 + r) / denom if denom > 1e-6 else h.copy()
        else:
            hp = h.copy()
        trade_cost = 0.0
        turn = 0.0
        # forced liquidation of assets whose history has ended
        dead = (last_valid < t) & (hp != 0)
        if dead.any():
            turn += float(np.abs(hp[dead]).sum())
            trade_cost += float((c[dead] * np.abs(hp[dead])).sum())
            hp[dead] = 0.0
        if exe_reb[t]:
            new = E[t].copy()
            # cannot trade assets without a price today: keep drifted position
            nt = ~avail[t]
            new[nt] = hp[nt]
            tr = new - hp
            turn += float(np.abs(tr).sum())
            trade_cost += float((c * np.abs(tr)).sum())
            h = new
        else:
            h = hp
        out[t] = (g, fee, trade_cost, turn, np.abs(h).sum(), h.sum(), np.count_nonzero(h))
        if keep_held:
            H[t] = h
    o = pd.DataFrame(out, index=cal, columns=["gross", "fees", "cost", "turnover", "glev", "nlev", "npos"])
    net = o["gross"] - o["fees"] - o["cost"]
    return BTResult(net=net, gross=o["gross"], cost=o["cost"], fees=o["fees"], turnover=o["turnover"],
                    gross_lev=o["glev"], net_lev=o["nlev"], n_pos=o["npos"], scale=scale,
                    held=pd.DataFrame(H, index=cal, columns=cols) if keep_held else None,
                    info={"lag": lag, "vol_target": vol_target, "lev_cap": lev_cap})
