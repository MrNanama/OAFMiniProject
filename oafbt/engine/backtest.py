"""Path-dependent daily portfolio backtester.

Timing convention (no look-ahead)
---------------------------------
* A strategy returns TARGET weights indexed by DECISION date d. A row may only use information available at the
  close of d. Rows that are entirely NaN mean "no rebalance" (positions drift with prices); within a rebalance row,
  NaN means 0. A row of zeros means "go flat".
* The target decided at d is EXECUTED at the close of d + lag trading days. lag >= 1 is enforced: several asset
  classes do not print at the NYSE close, so same-close execution (lag 0) would be a look-ahead.
* Holdings executed at the close of t earn the return of day t+1 onwards. An asset without a print on its
  execution day keeps its old position and the order stays PENDING until the asset next prints (unless a newer
  decision replaces it).
* Decision dates that are not trading days map to the next session (the latest decision wins).

Accounting (excess-of-cash terms)
---------------------------------
gross_t = sum_i h_{t-1,i} * xret_{t,i}           xret = return minus rf for funded assets (ETFs, stocks), raw
                                                  return for unfunded ones (long-short factors)
fees_t  = borrow fee on short notional + leverage spread on funded long notional above 1x
          + holding cost on |position| (implementation drag of paper portfolios), daily accrual
cost_t  = sum_i cost_bps_i * |trade_{t,i}|        trade = new target - drifted holdings
net_t   = gross_t - fees_t - cost_t
Drift uses total returns (xret + rf for funded assets; NAV earns rf). If NAV is wiped out (1 + return <= 0) the
book is closed and the run flagged as ruined.

Sizing
------
Every rebalance row is first normalised to unit gross (so the arbitrary scale of a strategy's weights is
irrelevant), then:
  vol_mode="row"      (default) scaled so its EX-ANTE volatility (historical simulation of today's weights over the
                      trailing window, exponentially weighted, using only days on which >= 95% of the book's gross
                      weight had prices) equals vol_target;
  vol_mode="strategy" scaled by vol_target / trailing realised vol of the UNSCALED strategy (252d, lagged), which
                      preserves the strategy's own risk timing (e.g. half exposure in bad regimes);
  vol_mode=None       no vol scaling (unit gross).
Gross leverage is capped at lev_cap at rebalance (drift can move it between rebalances).
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


def _align_targets(target: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """map decision rows to sessions: off-calendar dates go to the next session; latest decision wins"""
    t = target.copy()
    t = t[t.index <= cal[-1]] if len(cal) else t
    pos = cal.searchsorted(t.index)
    t.index = cal[np.minimum(pos, len(cal) - 1)]
    t = t[t.notna().any(axis=1)]
    t = t.groupby(level=0).last() if t.index.has_duplicates else t   # last = latest decision
    return t.reindex(cal)


def exante_vol(target: pd.DataFrame, xret: pd.DataFrame, window: int = 126, halflife: float = 42,
               min_obs: int = 40, coverage: float = 0.95) -> pd.Series:
    """Annualised ex-ante vol of each rebalance row of `target` by historical simulation over the trailing window
    ending AT the decision date (inclusive). Only days on which >= `coverage` of the row's gross weight had prices
    are used (missing returns are not treated as zero risk). NaN if fewer than min_obs such days."""
    cols = list(target.columns)
    R = xret.reindex(columns=cols).to_numpy()
    avail = ~np.isnan(R)
    R0 = np.nan_to_num(R)
    pos = {d: i for i, d in enumerate(xret.index)}
    lam = 0.5 ** (1.0 / halflife)
    out = pd.Series(np.nan, index=target.index)
    W = target.to_numpy()
    for k, d in enumerate(target.index):
        w = W[k]
        if np.all(np.isnan(w)):
            continue
        w = np.nan_to_num(w)
        gw = np.abs(w)
        if gw.sum() == 0:
            out.iloc[k] = 0.0
            continue
        i = pos.get(d)
        if i is None:
            continue
        lo = max(0, i - window + 1)
        cov = (avail[lo:i + 1] * gw).sum(axis=1) / gw.sum()
        live = cov >= coverage
        if live.sum() < min_obs:
            continue
        p = (R0[lo:i + 1] @ w)[live]
        ew = lam ** np.arange(p.size)[::-1]
        out.iloc[k] = np.sqrt(np.sum(ew * p * p) / np.sum(ew) * TD)
    return out


def backtest(target: pd.DataFrame, xret: pd.DataFrame, cost_bps: pd.Series | float = 0.0,
             borrow_bps_yr: pd.Series | float = 0.0, funded: pd.Series | bool = True, lag: int = 1,
             hold_bps_yr: pd.Series | float = 0.0, rf: pd.Series | None = None,
             vol_target: float | None = 0.10, vol_mode: str | None = "row", vol_window: int = 126,
             vol_halflife: float = 42, lev_cap: float = 3.0, lever_spread_bps_yr: float = 50.0,
             drift: bool = True, keep_held: bool = False, fee_mult: float = 1.0) -> BTResult:
    if lag < 1:
        raise ValueError("lag must be >= 1 (same-close execution is a look-ahead for non-NYSE-close assets)")
    cols = list(target.columns)
    xret = xret.reindex(columns=cols)
    cal = xret.index
    tgt = _align_targets(target.astype(float), cal)
    is_reb = tgt.notna().any(axis=1)
    reb = tgt[is_reb].fillna(0.0)
    gross = reb.abs().sum(axis=1)

    scale = pd.Series(np.nan, index=cal)
    mode = vol_mode if vol_target is not None else None
    if mode == "row":
        # each row normalised to unit gross, then scaled to the ex-ante vol target (raw scale is irrelevant)
        reb = reb.div(gross.replace(0, np.nan), axis=0).fillna(0.0)
        v = exante_vol(reb, xret, window=vol_window, halflife=vol_halflife)
        s = (vol_target / v.replace(0, np.nan)).clip(upper=lev_cap).fillna(0.0)
        s[gross == 0] = 0.0
    elif mode == "strategy":
        # raw weights keep the strategy's own exposure timing; one scalar per date from trailing realised vol
        unit = backtest(reb.reindex(cal), xret, lag=lag, rf=rf, funded=funded, vol_target=None, lev_cap=1e9,
                        drift=drift)
        live = unit.gross_lev.shift(1) > 0
        rv = unit.gross.where(live).rolling(252, min_periods=63).std().shift(1) * np.sqrt(TD)
        s = (vol_target / rv.reindex(reb.index).replace(0, np.nan)).fillna(0.0)
        s = np.minimum(s, lev_cap / gross.replace(0, np.nan)).fillna(0.0)
    else:
        s = np.minimum(1.0, lev_cap / gross.replace(0, np.nan)).fillna(1.0)
    scale.loc[reb.index] = s
    tgt.loc[is_reb] = reb.mul(s, axis=0)

    exe = tgt.shift(lag)
    exe_reb = exe.notna().any(axis=1).to_numpy()
    E = exe.fillna(0.0).to_numpy()

    X = xret.to_numpy()
    avail = ~np.isnan(X)
    X0 = np.nan_to_num(X)
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
    b = as_vec(borrow_bps_yr, 0.0) / 1e4 / TD * fee_mult
    hc = as_vec(hold_bps_yr, 0.0) / 1e4 / TD * fee_mult
    f = as_vec(funded, 1.0).astype(bool) if not isinstance(funded, bool) else np.full(len(cols), funded)
    ls = lever_spread_bps_yr / 1e4 / TD * fee_mult
    rfv = rf.reindex(cal).fillna(0.0).to_numpy() if rf is not None else np.zeros(len(cal))

    T, N = X.shape
    h = np.zeros(N)
    pending = np.full(N, np.nan)
    out = np.zeros((T, 7))
    H = np.zeros((T, N)) if keep_held else None
    ruined = None
    for t in range(T):
        r = X0[t]
        g = float(h @ r)
        fee = float(b @ np.maximum(-h, 0.0)) + ls * max(float(np.maximum(h, 0.0)[f].sum()) - 1.0, 0.0) \
            + float(hc @ np.abs(h))
        if drift:
            rtot = r + rfv[t] * f
            denom = 1.0 + g + rfv[t]
            if denom <= 1e-9:
                ruined = cal[t] if ruined is None else ruined
                hp = np.zeros(N)
            else:
                hp = h * (1.0 + rtot) / denom
        else:
            hp = h.copy()
        turn = 0.0
        tc = 0.0
        dead = (last_valid < t) & (hp != 0)
        if dead.any():
            turn += float(np.abs(hp[dead]).sum())
            tc += float((c[dead] * np.abs(hp[dead])).sum())
            hp[dead] = 0.0
        new = hp.copy()
        if exe_reb[t]:
            pending[:] = E[t]
        todo = ~np.isnan(pending) & avail[t]
        if todo.any():
            new[todo] = pending[todo]
            pending[todo] = np.nan
        tr = new - hp
        if np.any(tr):
            turn += float(np.abs(tr).sum())
            tc += float((c * np.abs(tr)).sum())
        h = new if ruined is None else np.zeros(N)
        out[t] = (g, fee, tc, turn, np.abs(h).sum(), h.sum(), np.count_nonzero(h))
        if keep_held:
            H[t] = h
    o = pd.DataFrame(out, index=cal, columns=["gross", "fees", "cost", "turnover", "glev", "nlev", "npos"])
    net = o["gross"] - o["fees"] - o["cost"]
    first_trade = o.index[np.argmax(o["glev"].to_numpy() > 0)] if (o["glev"] > 0).any() else None
    return BTResult(net=net, gross=o["gross"], cost=o["cost"], fees=o["fees"], turnover=o["turnover"],
                    gross_lev=o["glev"], net_lev=o["nlev"], n_pos=o["npos"], scale=scale,
                    held=pd.DataFrame(H, index=cal, columns=cols) if keep_held else None,
                    info={"lag": lag, "vol_target": vol_target, "vol_mode": mode, "lev_cap": lev_cap,
                          "ruined": ruined, "first_trade": first_trade})
