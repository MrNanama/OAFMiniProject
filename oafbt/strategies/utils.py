"""Shared helpers for strategies. Every helper uses only information at or before each row's date."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ schedules
def period_ends(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """last trading day of each period in `index` (freq 'W', 'M', 'Q'). The final period is kept only if its last
    date is a TRUE period end according to the published NYSE calendar (otherwise truncating the data would create
    a phantom rebalance on the last available day - which the look-ahead check would rightly flag)."""
    if len(index) == 0:
        return pd.DatetimeIndex([])
    s = pd.Series(index, index=index)
    ends = s.groupby(index.to_period(freq)).max()
    last = pd.Timestamp(ends.iloc[-1])
    nxt = next_session(last)
    if nxt is not None and nxt.to_period(freq) == last.to_period(freq):
        ends = ends.iloc[:-1]
    return pd.DatetimeIndex(ends.values)


def keep_rows(W: pd.DataFrame, dates) -> pd.DataFrame:
    """NaN-out all rows except `dates` (=> engine rebalances only on those dates, positions drift otherwise)"""
    out = W.copy()
    mask = ~out.index.isin(pd.DatetimeIndex(dates))
    out.loc[mask] = np.nan
    return out


# ------------------------------------------------------------------ cross-sectional transforms
def xs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """cross-sectional rank scaled to [-0.5, 0.5] per row (NaN ignored)"""
    r = df.rank(axis=1, pct=True)
    n = df.notna().sum(axis=1)
    return (r - 0.5 - 0.5 / n.values[:, None]).where(df.notna()) if len(df) else r


def xs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    m = df.mean(axis=1)
    s = df.std(axis=1)
    return df.sub(m, axis=0).div(s.replace(0, np.nan), axis=0).clip(-clip, clip)


def long_short(signal: pd.DataFrame, q: float = 0.2, eligible: pd.DataFrame | None = None,
               weight: str = "equal") -> pd.DataFrame:
    """dollar-neutral portfolio: +1 spread over the top-q names, -1 over the bottom-q names (each leg sums to 1)"""
    s = signal.where(eligible) if eligible is not None else signal
    pct = s.rank(axis=1, pct=True)
    top = (pct > 1 - q)
    bot = (pct <= q)
    if weight == "equal":
        lw = top.astype(float).div(top.sum(axis=1).replace(0, np.nan), axis=0)
        sw = bot.astype(float).div(bot.sum(axis=1).replace(0, np.nan), axis=0)
    else:  # signal-proportional within legs
        z = xs_zscore(s)
        lw = z.where(top, 0).clip(lower=0)
        lw = lw.div(lw.sum(axis=1).replace(0, np.nan), axis=0)
        sw = (-z).where(bot, 0).clip(lower=0)
        sw = sw.div(sw.sum(axis=1).replace(0, np.nan), axis=0)
    return (lw.fillna(0) - sw.fillna(0)).where(s.notna().any(axis=1), np.nan, axis=0)


def rank_weights(signal: pd.DataFrame, eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    """dollar-neutral rank weights (sum |w| = 2 per row: 1 long, 1 short)"""
    s = signal.where(eligible) if eligible is not None else signal
    r = xs_rank(s)
    g = r.abs().sum(axis=1).replace(0, np.nan)
    return r.div(g, axis=0) * 2


# ------------------------------------------------------------------ time-series helpers
def ewm_vol(ret: pd.DataFrame, halflife: float = 42, min_periods: int = 20, annualise: bool = True):
    v = np.sqrt((ret ** 2).ewm(halflife=halflife, min_periods=min_periods).mean())
    return v * np.sqrt(252) if annualise else v


def rolling_beta(ret: pd.DataFrame, mkt: pd.Series, window: int = 252, min_periods: int = 126) -> pd.DataFrame:
    cov = ret.rolling(window, min_periods=min_periods).cov(mkt)
    var = mkt.rolling(window, min_periods=min_periods).var()
    return cov.div(var, axis=0)


def cum_ret(ret: pd.DataFrame, window: int, skip: int = 0, min_frac: float = 0.8) -> pd.DataFrame:
    """compounded return over (t-window-skip, t-skip] using log sums; NaN if < min_frac of days observed"""
    lr = np.log1p(ret)
    s = lr.rolling(window, min_periods=int(window * min_frac)).sum()
    return np.expm1(s.shift(skip)) if skip else np.expm1(s)


# ------------------------------------------------------------------ Ken French assets
def kf_rf(ds) -> pd.Series:
    """daily risk-free on the KF calendar: ^IRX-based ds.rf where available (1985+), KF daily RF before
    (the KF daily file rounds RF to 0.01%/day, so it is only a fallback)"""
    kf = ds.kf("F-F_Research_Data_Factors_daily")["RF"]
    return ds.rf.reindex(kf.index).combine_first(kf)


def kf_factor_returns(ds, specs: dict) -> pd.DataFrame:
    """specs: {asset_name: (table, column)} for long-short factors (already excess). Asset names must start with
    'KF:F:' and end with the factor code used for the implementation-drag table (e.g. 'KF:F:US:Mom').
    All series are put on the US KF calendar: returns of non-US sessions (foreign trading on US holidays) are
    compounded into the next US session, so annualisation with 252 days/yr stays correct."""
    us = ds.kf("F-F_Research_Data_Factors_daily").index
    cols = {}
    for name, (table, col) in specs.items():
        r = ds.kf(table)[col]
        if not r.index.isin(us).all():
            nxt = us[np.minimum(us.searchsorted(r.index), len(us) - 1)]
            ok = r.index <= us[-1]
            lr = np.log1p(r[ok]).groupby(nxt[ok]).sum(min_count=1)
            r = np.expm1(lr)
        cols[name] = r.reindex(us)
    return pd.DataFrame(cols)


def kf_portfolio_returns(ds, table: str, prefix: str | None = None) -> pd.DataFrame:
    """value/equal-weighted portfolios (total returns) converted to EXCESS returns over the KF 1M bill."""
    p = ds.kf(table)
    rf = kf_rf(ds).reindex(p.index)
    x = p.sub(rf, axis=0)
    pre = prefix or f"KF:P:{table.split('_')[0]}"
    x.columns = [f"{pre}:{c}" for c in x.columns]
    return x


# ------------------------------------------------------------------ published trading calendar
_XNYS = None


def nyse_sessions() -> pd.DatetimeIndex:
    """all NYSE sessions 1985-2030 from the exchange's published schedule (cached)"""
    global _XNYS
    if _XNYS is None:
        import exchange_calendars as xc
        cal = xc.get_calendar("XNYS", start="1985-01-01", end="2030-12-31")
        ses = pd.DatetimeIndex(cal.sessions)
        _XNYS = ses.tz_localize(None) if ses.tz is not None else ses
    return _XNYS


def next_session(d) -> pd.Timestamp | None:
    ses = nyse_sessions()
    i = ses.searchsorted(pd.Timestamp(d), side="right")
    return ses[i] if i < len(ses) else None


def trading_calendar(ds, ahead: int = 40) -> pd.DatetimeIndex:
    """ds.calendar extended by the next `ahead` NYSE sessions. The exchange publishes its holiday schedule in
    advance, so a strategy may know which future dates are trading days (e.g. 'tomorrow is the first trading day
    of the month') without any market data from those days. Use this instead of shifting the data calendar."""
    cal = ds.calendar
    ses = nyse_sessions()
    i = ses.searchsorted(cal[-1], side="right")
    return cal.append(ses[i:i + ahead])


def scheduled_calendar(ds, ahead: int = 40) -> pd.DatetimeIndex:
    """trading_calendar() plus the exchange's AD-HOC closures (9/11, Hurricane Sandy, state funerals ...), which
    were not known in advance. Calendar positions such as 'day -2 of the month' must be computed on THIS schedule;
    a decision that falls on a closed day is executed at the next session by the engine."""
    import exchange_calendars as xc
    cal = trading_calendar(ds, ahead)
    ah = pd.DatetimeIndex(xc.get_calendar("XNYS", start="1985-01-01", end="2030-12-31").adhoc_holidays)
    ah = ah.tz_localize(None) if ah.tz is not None else ah
    ah = ah[(ah >= cal[0]) & (ah <= cal[-1])]
    return cal.union(ah)
