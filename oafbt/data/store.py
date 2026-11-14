"""Processed data panels + the HOLDOUT LOCK.

`python -m oafbt.data.store build` turns data/raw/* into aligned, cleaned panels:

    data/processed/dev/*.parquet    everything truncated at config.DEV_END   <- research code reads ONLY these
    data/vault/*.parquet            full history (incl. 2021+ holdout)        <- readable only if ./FROZEN exists

All price panels share one master calendar (NYSE trading days, from ^GSPC/SPY). Assets on other calendars
(FX, crypto, foreign indices) are sampled as-of each NYSE date (last print at or before it, max 5 days stale).
Returns are simple daily total returns from adjusted closes. Excess returns subtract the daily risk-free rate
(Ken French 1M T-bill, spliced with ^IRX) for FUNDED assets only (ETFs, stocks, mutual funds, indices).
"""
from __future__ import annotations

import sys
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config as C
from . import universe as U

warnings.filterwarnings("ignore")

DEV_DIR = C.DATA / "processed" / "dev"
VAULT = C.DATA / "vault"
FIELDS = ["adjclose", "close", "open", "high", "low", "volume"]
# Yahoo quotes these FX pairs as USD/XXX; we invert them so every FX series = USD value of one unit of foreign ccy
FX_INVERT = {"JPY=X", "CAD=X", "CHF=X", "SEK=X", "NOK=X", "MXN=X", "ZAR=X", "BRL=X", "INR=X", "KRW=X", "SGD=X",
             "TRY=X", "PLN=X", "HUF=X", "CZK=X", "ILS=X", "CNY=X", "HKD=X", "IDR=X", "THB=X", "TWD=X", "CLP=X",
             "COP=X", "PHP=X"}


# --------------------------------------------------------------------------------------------- build step
def _wide(df: pd.DataFrame, field: str) -> pd.DataFrame:
    w = df.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
    return w.sort_index()


def _despike_multi(px: pd.DataFrame, thresh: float, max_k: int = 3) -> tuple[pd.DataFrame, int]:
    """remove price spikes lasting 1..max_k days that revert to the pre-spike level (bad prints)"""
    lp = np.log(px)
    n_total = 0
    for k in range(1, max_k + 1):
        jump = lp - lp.shift(1)                       # jump into the spike at t
        back = lp.shift(-k) - lp.shift(1)             # level k days later vs pre-spike level
        inside = pd.DataFrame(True, index=lp.index, columns=lp.columns)
        for j in range(1, k):                          # intermediate days stay near the spike level
            inside &= ((lp.shift(-j) - lp).abs() < 0.5 * jump.abs())
        bad = (jump.abs() > np.log1p(thresh)) & (back.abs() < 0.25 * jump.abs()) & inside
        if not bad.to_numpy().any():
            continue
        mask = bad.copy()
        for j in range(1, k):
            mask |= bad.shift(j, fill_value=False)
        n_total += int(bad.to_numpy().sum())
        lp = lp.mask(mask)
    return np.exp(lp), n_total


def _despike(px: pd.DataFrame, thresh: float) -> tuple[pd.DataFrame, int]:
    """remove single-day price spikes that fully reverse the next day (bad prints)"""
    r1 = px / px.shift(1) - 1
    r2 = px.shift(-1) / px - 1
    bad = (r1.abs() > thresh) & (r2.abs() > thresh * 0.7) & (np.sign(r1) != np.sign(r2)) & \
          (((1 + r1) * (1 + r2) - 1).abs() < 0.3 * r1.abs())
    n = int(bad.sum().sum())
    return px.mask(bad), n


def build():
    raw = C.RAW
    parts = []
    for g in ["etf", "mf", "fut", "fx", "index", "crypto", "stocks"]:
        f = raw / f"yahoo_{g}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            d["group"] = g
            parts.append(d)
    y = pd.concat(parts, ignore_index=True)
    y = y.drop_duplicates(subset=["date", "ticker"], keep="last")
    groups = y.groupby("ticker")["group"].first()

    # master calendar = NYSE days (dates on which ^GSPC or SPY printed), weekdays only
    cal = pd.DatetimeIndex(sorted(set(y.loc[y.ticker.isin(["^GSPC", "SPY"]), "date"])))
    cal = cal[cal.day_of_week < 5]
    print("calendar", cal[0].date(), cal[-1].date(), len(cal))

    panels = {}
    report = {}
    for fld in FIELDS:
        w = _wide(y, fld)
        # FX inversion (prices only)
        inv = [c for c in w.columns if c in FX_INVERT]
        if fld in ("adjclose", "close", "open"):
            w[inv] = 1.0 / w[inv]
        elif fld in ("high", "low") and inv:
            hi, lo = w[inv].copy(), None
        panels[fld] = w
    # fix high/low for inverted FX
    inv = [c for c in panels["high"].columns if c in FX_INVERT]
    hi, lo = panels["high"][inv].copy(), panels["low"][inv].copy()
    panels["high"][inv] = 1.0 / lo
    panels["low"][inv] = 1.0 / hi

    # crude sanity: non-positive prices -> NaN (e.g. CL=F on 2020-04-20)
    for fld in ("adjclose", "close", "open", "high", "low"):
        p = panels[fld]
        nonpos = (p <= 0).sum().sum()
        report[f"nonpos_{fld}"] = int(nonpos)
        panels[fld] = p.where(p > 0)

    # align to master calendar, as-of sampling with a 5-day staleness limit
    out = {}
    for fld, p in panels.items():
        p = p.sort_index()
        full_idx = p.index.union(cal)
        if fld == "volume":
            q = p.reindex(cal)               # no forward fill for volume
        else:
            q = p.reindex(full_idx).ffill(limit=5).reindex(cal)
            # but never fill BEFORE an asset's first print or after its last print
            first = p.apply(lambda s: s.first_valid_index())
            last = p.apply(lambda s: s.last_valid_index())
            for c in q.columns:
                if first[c] is None:
                    continue
                q.loc[(q.index < first[c]) | (q.index > last[c] + pd.Timedelta(days=7)), c] = np.nan
        out[fld] = q.astype("float64")
    # ---- cleaning on the aligned calendar
    adj = out["adjclose"]
    th = {"etf": 0.25, "mf": 0.20, "fut": 0.30, "fx": 0.12, "index": 0.30, "crypto": 0.50, "stocks": 0.40}
    for grp, t in th.items():
        cols = [c for c in adj.columns if groups.get(c) == grp and c not in U.ETF_VOL and c not in U.LEVEL_INDICES]
        if cols:
            adj[cols], n = _despike_multi(adj[cols], t)
            report[f"despiked_{grp}"] = n
    ret = adj.pct_change(fill_method=None)
    # hard caps: moves that are not credible for these instruments -> treat as bad data (return NaN)
    caps = {"etf": (0.30, -0.30), "mf": (0.25, -0.25), "fx": (0.25, -0.25), "stocks": (1.00, -0.85),
            "fut": (0.40, -0.40), "index": (0.40, -0.40)}
    for grp, (hi, lo) in caps.items():
        cols = [c for c in ret.columns if groups.get(c) == grp and c not in U.ETF_VOL and c not in U.LEVEL_INDICES]
        bad = (ret[cols] > hi) | (ret[cols] < lo)
        report[f"capped_{grp}"] = int(bad.sum().sum())
        ret[cols] = ret[cols].mask(bad)
    out["adjclose"] = adj
    out["ret"] = ret
    # total-return index rebuilt from CLEANED returns (use this, not adjclose, for price-based signals)
    tri = (1 + ret.fillna(0.0)).cumprod()
    out["tri"] = tri.where(adj.notna())
    # stale-price flag: share of exactly-zero returns over the trailing 63 days (info <= t)
    zero = (ret == 0).astype(float).where(ret.notna())
    out["stale63"] = zero.rolling(63, min_periods=20).mean()

    # ---- extended long-history series: ETF spliced with its index-fund proxy before inception
    ext = {}
    for mf, etf in U.MF_PROXY.items():
        if etf and etf in ret and mf in ret and etf in U.EXTEND:
            r_e, r_m = ret[etf], ret[mf]
            first = r_e.first_valid_index()
            ext[etf + "_X"] = r_m.where(r_m.index < first).combine_first(r_e)
    if "GLD" in ret and "GC=F" in ret:
        first = ret["GLD"].first_valid_index()
        ext["GLD_X"] = ret["GC=F"].where(ret.index < first).combine_first(ret["GLD"])
    for k, v in ext.items():
        ret[k] = v
        groups[k] = "ext"
        out["tri"][k] = (1 + v.fillna(0.0)).cumprod().where(v.notna() | v.shift(-1).notna())
    report["extended"] = {k: str(v.first_valid_index().date()) for k, v in ext.items()}
    out["ret"] = ret

    # risk-free: Ken French daily RF, spliced with ^IRX (13w T-bill, % p.a.) where KF is missing
    kf = pd.read_pickle(raw / "kf.pkl")
    rf_kf = None
    for (f, t), v in kf.items():
        if f == "F-F_Research_Data_Factors_daily":
            rf_kf = v["RF"]
    irx = out["close"]["^IRX"] / 100.0 / C.TRADING_DAYS
    rf = rf_kf.reindex(cal) if rf_kf is not None else pd.Series(np.nan, index=cal)
    rf = rf.fillna(irx).ffill().fillna(0.0)
    out["rf"] = rf.to_frame("rf")

    # excess returns: subtract rf for funded assets
    funded = [c for c in ret.columns if not U.is_excess_return(c)]
    xret = ret.copy()
    xret[funded] = ret[funded].sub(rf, axis=0)
    out["xret"] = xret

    # group map + S&P 500 point-in-time membership
    out["groups"] = groups.to_frame("group")
    mem = pd.read_csv(raw / "sp500_membership.csv", parse_dates=["date"]).sort_values("date")
    stocks = [c for c in adj.columns if groups.get(c) == "stocks"]
    M = pd.DataFrame(False, index=pd.DatetimeIndex(mem["date"]), columns=stocks)
    for i, row in enumerate(mem.itertuples()):
        tick = [t.replace(".", "-") for t in row.tickers.split(",")]
        tick = [t for t in tick if t in M.columns]
        M.iloc[i, M.columns.get_indexer(tick)] = True
    M = M[~M.index.duplicated(keep="last")]
    M = M.reindex(M.index.union(cal)).ffill().reindex(cal).fillna(False).astype(bool)
    out["sp500"] = M

    # Ken French tables (own calendars)
    kfo = {}
    for (f, t), v in kf.items():
        key = f + ("|EW" if "Equal Weighted" in t else "|VW" if "Value Weighted" in t else "")
        if "Number of Firms" in t or "Average Firm Size" in t or "Sum of BE" in t:
            continue
        kfo[key] = v
    VAULT.mkdir(parents=True, exist_ok=True)
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(VAULT / f"{k}.parquet")
        v.loc[:C.DEV_END].to_parquet(DEV_DIR / f"{k}.parquet") if isinstance(v.index, pd.DatetimeIndex) \
            else v.to_parquet(DEV_DIR / f"{k}.parquet")
    # scheduled FOMC announcement dates (published >= 1 year ahead; dev copy keeps dates up to DEV_END + 90 days)
    if (raw / "fomc.csv").exists():
        fomc = pd.read_csv(raw / "fomc.csv", parse_dates=["announce"])[["announce"]]
        fomc.to_parquet(VAULT / "fomc.parquet")
        fomc[fomc.announce <= pd.Timestamp(C.DEV_END) + pd.Timedelta(days=90)].to_parquet(DEV_DIR / "fomc.parquet")
    pd.to_pickle(kfo, VAULT / "kf.pkl")
    pd.to_pickle({k: v.loc[:C.DEV_END] for k, v in kfo.items()}, DEV_DIR / "kf.pkl")
    print("report", report)
    print("panels", {k: v.shape for k, v in out.items()})
    print("kf tables", len(kfo))


# --------------------------------------------------------------------------------------------- access
class HoldoutLocked(PermissionError):
    pass


class DataStore:
    """Read-only access to processed panels. Default = dev data only (<= DEV_END)."""

    def __init__(self, holdout: bool = False, end=None):
        """end: optional earlier truncation date (used by the look-ahead test harness)"""
        if holdout and not C.FROZEN_FILE.exists():
            raise HoldoutLocked("Holdout data is locked until the strategies are frozen (FROZEN file + git tag).")
        self.holdout = holdout
        self.dir = VAULT if holdout else DEV_DIR
        self.end = None if holdout else pd.Timestamp(C.DEV_END)
        if end is not None:
            e = pd.Timestamp(end)
            self.end = e if self.end is None else min(e, self.end)

    @lru_cache(maxsize=None)
    def _load(self, name: str) -> pd.DataFrame:
        df = pd.read_parquet(self.dir / f"{name}.parquet")
        if self.end is not None and isinstance(df.index, pd.DatetimeIndex):
            df = df.loc[:self.end]
        return df

    # panels ---------------------------------------------------------------------------------
    def panel(self, field: str, tickers=None) -> pd.DataFrame:
        df = self._load(field)
        return df if tickers is None else df.reindex(columns=list(tickers))

    def px(self, tickers=None):
        return self.panel("adjclose", tickers)

    def ret(self, tickers=None):
        return self.panel("ret", tickers)

    def xret(self, tickers=None):
        return self.panel("xret", tickers)

    @property
    def rf(self) -> pd.Series:
        return self._load("rf")["rf"]

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self._load("rf").index

    def groups(self) -> pd.Series:
        return self._load("groups")["group"]

    def tickers(self, group: str) -> list:
        g = self.groups()
        return sorted(g[g == group].index)

    def sp500(self) -> pd.DataFrame:
        """point-in-time S&P 500 membership (True = member that day) for tickers with Yahoo data"""
        return self._load("sp500")

    def fomc_dates(self) -> pd.DatetimeIndex:
        """scheduled FOMC announcement dates. The schedule is published well in advance, so dates shortly after
        the data end are legitimately known (dev copy: up to DEV_END + 90 days)."""
        f = pd.read_parquet(self.dir / "fomc.parquet")["announce"]
        if self.end is not None:
            f = f[f <= self.end + pd.Timedelta(days=90)]
        return pd.DatetimeIndex(f)

    def tri(self, tickers=None) -> pd.DataFrame:
        """total-return index rebuilt from cleaned returns (preferred for price-based signals)"""
        return self.panel("tri", tickers)

    def stock_universe(self, min_hist: int = 252, max_stale: float = 0.30,
                       min_dollar_vol: float = 1e6) -> pd.DataFrame:
        """Boolean (dates x stocks) eligibility at each date using only info <= t:
        point-in-time S&P 500 member, priced today, >= min_hist days of history, share of zero-return days
        over 63d <= max_stale (pre-2001 fractional ticks make 10-25% normal), 63d median dollar volume
        >= min_dollar_vol. NOTE: no price filter on purpose - Yahoo 'close' is split-adjusted, so a price
        floor would use FUTURE splits (excluding later winners) = look-ahead bias."""
        M = self.sp500()
        cols = list(M.columns)
        ret = self.ret(cols)
        hist = ret.notna().cumsum()
        close = self.panel("close", cols)
        dv = (close * self.panel("volume", cols)).rolling(63, min_periods=20).median()
        stale = self.panel("stale63", cols)
        ok = M & ret.notna() & (hist >= min_hist) & (stale <= max_stale) & (dv >= min_dollar_vol)
        return ok.fillna(False).astype(bool)

    def kf(self, name: str) -> pd.DataFrame:
        d = self._kf()
        if name not in d:
            raise KeyError(f"{name} not in {sorted(d)}")
        return d[name]

    @lru_cache(maxsize=1)
    def _kf(self) -> dict:
        d = pd.read_pickle(self.dir / "kf.pkl")
        if self.end is not None:
            d = {k: v.loc[:self.end] for k, v in d.items()}
        return d

    def kf_names(self) -> list:
        return sorted(self._kf())

    # metadata -------------------------------------------------------------------------------
    def meta(self, tickers) -> pd.DataFrame:
        rows = []
        for t in tickers:
            cls = U.asset_class(t)
            hold = C.HOLD_BPS_YR.get(cls, 0.0)
            if cls == "kf_factor":       # factor-specific drag, e.g. KF:F:Mom
                hold = C.KF_FACTOR_DRAG_BPS_YR.get(t.split(":")[-1], hold)
            rows.append({"ticker": t, "cls": cls, "cost_bps": C.COST_BPS[cls], "borrow_bps": C.BORROW_BPS_YR[cls],
                         "hold_bps": hold, "funded": not U.is_excess_return(t)})
        return pd.DataFrame(rows).set_index("ticker")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
