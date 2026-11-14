"""Processed data panels + the HOLDOUT LOCK.

`python -m oafbt.data.store build` turns data/raw/* into aligned, cleaned panels:

    data/processed/dev/*.parquet    everything truncated at config.DEV_END   <- research code reads ONLY these
    data/vault/*.parquet            full history (incl. 2021+ holdout)        <- readable only if ./FROZEN exists

Conventions
-----------
* One master calendar: NYSE sessions (from ^GSPC/SPY prints).
* NYSE-listed instruments (ETFs, stocks, mutual funds) are NOT forward-filled: a day without a print has NaN price
  and NaN return; the return on the next print is measured from the LAST VALID price, so moves across data holes
  are booked (not lost). `printed` marks real prints (the engine only trades on printed days).
* Assets on other calendars (FX, crypto, futures, foreign indices) are sampled as-of each NYSE date (last print
  <= D, max 5 days stale). They are SIGNAL-ONLY (not tradable): spot FX lacks carry, Yahoo continuous futures are
  not roll-adjusted, indices are price-only/local currency.
* Timestamps: Yahoo spot FX bars carry the previous session's price from 2010-08-04 on; before that they are
  same-day. We delay pre-2010-08-04 FX by one session so FX values are always ~1 session stale (never ahead of the
  NYSE close). Crypto daily closes are at 00:00 UTC (after the NYSE close) -> delayed by one session as well.
* Cleaning is deliberately conservative (it never deletes plausible moves):
  - corporate-action repair: a jump in the adjclose/close ratio (> 4%) that is not matched by a close drop on the
    same day is moved to the nearby day where the close dropped by that amount, or removed if no such day exists
    (double-counted spin-off adjustments, misdated fund distributions);
  - bad prints: a price spike that reverts to (almost exactly) the prior level within 1-3 days is masked;
    thresholds are large for stocks (|jump| > 50%, reversal within 15% of the jump);
  - FX: volatility-relative spike filter (> 8 robust sigmas with reversal);
  - no blanket return caps (FNMA -90% in 2008 is real and stays).
* Risk-free: ^IRX (13-week bill, % p.a.) / 252, lagged one session (rate fixed at the start of the day);
  Ken French daily RF only where ^IRX is missing (it is rounded to 0.01%/day).
* Stock universe: point-in-time S&P 500 membership (fja05680) with renamed members re-linked to their current
  Yahoo ticker; spells whose Yahoo history starts > 60 days after the spell start (ticker reuse) and a short list of
  verified reused tickers are excluded. The DEV panels contain only stocks that were members on or before DEV_END,
  so future index additions cannot leak into research through the column set. Survivorship remains: Yahoo has no
  history for most delisted members (bias measured at ~+1.7%/yr for an equal-weight long-only book).
"""
from __future__ import annotations

import contextlib
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
PRICE_FIELDS = ["adjclose", "close", "open", "high", "low"]
FX_INVERT = {"JPY=X", "CAD=X", "CHF=X", "SEK=X", "NOK=X", "MXN=X", "ZAR=X", "BRL=X", "INR=X", "KRW=X", "SGD=X",
             "TRY=X", "PLN=X", "HUF=X", "CZK=X", "ILS=X", "CNY=X", "HKD=X", "IDR=X", "THB=X", "TWD=X", "CLP=X",
             "COP=X", "PHP=X"}
FX_SWITCH = pd.Timestamp("2010-08-04")
GROUP_PRIORITY = ["etf", "mf", "fut", "fx", "index", "crypto", "stocks"]
TRADABLE_GROUPS = {"etf", "ext", "stocks", "crypto"}
NYSE_GROUPS = {"etf", "mf", "stocks"}
# S&P members renamed after they left/while in the index: membership ticker -> current Yahoo ticker
RENAMES = {"FB": "META", "ANTM": "ELV", "BLL": "BALL", "BK": "BNY", "ABC": "COR", "NLOK": "GEN", "RE": "EG",
           "FLT": "CPAY", "LB": "BBWI", "PEAK": "DOC", "MMC": "MRSH"}
# Yahoo series that belong to a different issuer than the S&P member that used the ticker (review findings)
REUSE_EXCLUDE = {"MCIC", "EP", "CPWR", "DIGI", "COMS", "USBC", "BBT"}
REUSE_VALID_FROM = {"JCI": "2016-09-06", "DD": "2019-06-03", "IR": "2020-03-02", "AN": "1999-04-01"}

_GLOBAL_END = None      # process-wide truncation used by the look-ahead harness (see global_end())


@contextlib.contextmanager
def global_end(end):
    """Every DataStore created inside this context is truncated at `end` (even ones a strategy builds itself)."""
    global _GLOBAL_END
    prev = _GLOBAL_END
    _GLOBAL_END = pd.Timestamp(end) if end is not None else None
    try:
        yield
    finally:
        _GLOBAL_END = prev


# --------------------------------------------------------------------------------------------- build helpers
def _wide(df: pd.DataFrame, field: str) -> pd.DataFrame:
    return df.pivot_table(index="date", columns="ticker", values=field, aggfunc="last").sort_index()


def _ret_from_last_valid(p: pd.DataFrame) -> pd.DataFrame:
    """simple return vs the last valid price; NaN on days without a price"""
    return (p / p.ffill().shift(1) - 1).where(p.notna())


def _despike(p: pd.DataFrame, thresh: float, back_frac: float = 0.15, max_k: int = 3,
             volume: pd.DataFrame | None = None, vol_mult: float = 2.0) -> tuple[pd.DataFrame, int]:
    """mask prices of 1..max_k-day spikes that revert to within back_frac of the jump (bad prints). If volume is
    given, a spike that comes with a volume surge (> vol_mult x trailing 63d median) is GENUINE and never masked
    (e.g. AIG 2008-09-17: -45% then a full rebound on record volume)."""
    lp = np.log(p)
    surge = None
    if volume is not None:
        v = volume.reindex(index=p.index, columns=p.columns)
        med = v.rolling(63, min_periods=20).median().shift(1)
        surge = (v > vol_mult * med).fillna(False)
    n_total = 0
    for k in range(1, max_k + 1):
        prev = lp.ffill().shift(1)
        jump = lp - prev
        back = lp.shift(-k) - prev
        inside = pd.DataFrame(True, index=lp.index, columns=lp.columns)
        for j in range(1, k):
            inside &= ((lp.shift(-j) - lp).abs() < 0.5 * jump.abs())
        bad = (jump.abs() > np.log1p(thresh)) & (back.abs() < back_frac * jump.abs()) & inside
        if surge is not None:
            bad &= ~surge
        if not bad.to_numpy().any():
            continue
        mask = bad.copy()
        for j in range(1, k):
            mask |= bad.shift(j, fill_value=False)
        n_total += int(bad.to_numpy().sum())
        lp = lp.mask(mask)
    return np.exp(lp), n_total


def _despike_volrel(p: pd.DataFrame, z: float = 8.0, floor: float = 0.03) -> tuple[pd.DataFrame, int]:
    """FX: mask spikes larger than z robust sigmas (and > floor) that revert within 3 days"""
    lp = np.log(p)
    r = lp - lp.ffill().shift(1)
    mad = r.abs().rolling(60, min_periods=20).median().shift(1) * 1.4826
    thr = np.maximum(z * mad, floor)
    n_total = 0
    for k in range(1, 4):
        prev = lp.ffill().shift(1)
        jump = lp - prev
        back = lp.shift(-k) - prev
        bad = (jump.abs() > thr) & (back.abs() < 0.3 * jump.abs())
        if bad.to_numpy().any():
            mask = bad.copy()
            for j in range(1, k):
                mask |= bad.shift(j, fill_value=False)
            n_total += int(bad.to_numpy().sum())
            lp = lp.mask(mask)
    return np.exp(lp), n_total


def _repair_corporate_actions(ret: pd.DataFrame, adj: pd.DataFrame, close: pd.DataFrame,
                              j_thresh: float = 0.04, window: int = 10) -> tuple[pd.DataFrame, dict]:
    """Fix days where Yahoo's adjustment factor jumps without a matching close drop (double-counted spin-off
    adjustments, misdated distributions). Returns repaired returns and counts."""
    rc = _ret_from_last_valid(close)
    ra = ret
    J = (1 + ra) / (1 + rc) - 1
    cand = (J.abs() > j_thresh) & ra.notna() & rc.notna()
    phantom = cand & ((rc + J).abs() > 0.5 * J.abs())
    out = ra.copy()
    moved = removed = 0
    R = out.to_numpy(copy=True)
    RC = rc.to_numpy()
    JJ = J.to_numpy()
    rows, cols = np.nonzero(phantom.to_numpy())
    for i, j in zip(rows, cols):
        jt = JJ[i, j]
        lo, hi = max(0, i - window), min(len(R) - 1, i + window)
        best, bestd = None, None
        for s in range(lo, hi + 1):
            if s == i or np.isnan(RC[s, j]):
                continue
            # a close drop of about the distribution size on day s
            if abs(RC[s, j] + jt / (1 + jt)) < 0.3 * abs(jt) and abs(RC[s, j]) > 0.5 * abs(jt) / (1 + abs(jt)):
                d = abs(s - i)
                if bestd is None or d < bestd:
                    best, bestd = s, d
        R[i, j] = RC[i, j]                                   # remove the phantom jump on day i
        if best is not None:
            R[best, j] = (1 + RC[best, j]) * (1 + jt) - 1    # book the distribution on the day the close dropped
            moved += 1
        else:
            removed += 1
    out = pd.DataFrame(R, index=ra.index, columns=ra.columns)
    return out, {"ca_moved": moved, "ca_removed": removed}


def _assign_groups(y: pd.DataFrame) -> pd.Series:
    pr = {g: i for i, g in enumerate(GROUP_PRIORITY)}
    g = y[["ticker", "group"]].drop_duplicates()
    g["p"] = g["group"].map(pr)
    return g.sort_values("p").drop_duplicates("ticker").set_index("ticker")["group"]


def _membership(raw, stocks_with_data: set, cal: pd.DatetimeIndex, first_print: pd.Series) -> pd.DataFrame:
    mem = pd.read_csv(raw / "sp500_membership.csv", parse_dates=["date"]).sort_values("date")
    rows = []
    for row in mem.itertuples():
        tick = {RENAMES.get(t.replace(".", "-"), t.replace(".", "-")) for t in row.tickers.split(",")}
        rows.append((row.date, tick))
    cols = sorted(set().union(*[t for _, t in rows]) & stocks_with_data)
    cols = [c for c in cols if c not in REUSE_EXCLUDE]
    M = pd.DataFrame(False, index=pd.DatetimeIndex([d for d, _ in rows]), columns=cols)
    for i, (d, tick) in enumerate(rows):
        idx = M.columns.get_indexer([t for t in tick if t in M.columns])
        M.iloc[i, idx[idx >= 0]] = True
    M = M[~M.index.duplicated(keep="last")]
    M = M.reindex(M.index.union(cal)).ffill().reindex(cal).fillna(False).astype(bool)
    # spells whose Yahoo history starts > 60 days after the spell start are another issuer's history
    for c in M.columns:
        s = M[c]
        if not s.any():
            continue
        spell = (s != s.shift(fill_value=False)).cumsum()
        fp = first_print.get(c)
        for sid, grp in s.groupby(spell):
            if not grp.iloc[0]:
                continue
            start = grp.index[0]
            if fp is None or pd.isna(fp) or fp > start + pd.Timedelta(days=60):
                M.loc[grp.index, c] = False
        if c in REUSE_VALID_FROM:
            M.loc[M.index < pd.Timestamp(REUSE_VALID_FROM[c]), c] = False
    return M


# --------------------------------------------------------------------------------------------- build
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
    groups = _assign_groups(y)
    y["prio"] = y["group"].map({g: i for i, g in enumerate(GROUP_PRIORITY)})
    y = y.sort_values("prio").drop_duplicates(subset=["date", "ticker"], keep="first")   # ETF beats stock collision
    y = y[y["group"] == y["ticker"].map(groups)]
    report = {}

    cal = pd.DatetimeIndex(sorted(set(y.loc[y.ticker.isin(["^GSPC", "SPY"]), "date"])))
    cal = cal[cal.day_of_week < 5]
    print("calendar", cal[0].date(), cal[-1].date(), len(cal))

    W = {f: _wide(y, f) for f in PRICE_FIELDS + ["volume"]}
    # FX inversion: every FX series = USD value of one unit of foreign currency
    inv = [c for c in W["adjclose"].columns if c in FX_INVERT]
    for f in ["adjclose", "close", "open"]:
        W[f][inv] = 1.0 / W[f][inv]
    hi, lo = W["high"][inv].copy(), W["low"][inv].copy()
    W["high"][inv], W["low"][inv] = 1.0 / lo, 1.0 / hi
    for f in PRICE_FIELDS:
        report[f"nonpos_{f}"] = int((W[f] <= 0).sum().sum())
        W[f] = W[f].where(W[f] > 0)

    cols = list(W["adjclose"].columns)
    nyse = [c for c in cols if groups.get(c) in NYSE_GROUPS]
    other = [c for c in cols if c not in nyse]
    first_print = W["adjclose"].apply(lambda s: s.first_valid_index())

    out = {}
    printed = W["adjclose"][nyse].reindex(cal).notna()
    for f in PRICE_FIELDS + ["volume"]:
        a = W[f][nyse].reindex(cal)                                        # NYSE assets: no fill
        b = W[f][other]
        if f != "volume":
            full = b.index.union(cal)
            b = b.reindex(full).ffill(limit=5).reindex(cal)                # other calendars: as-of, <= 5 days
            last = W[f][other].apply(lambda s: s.last_valid_index())
            for c in other:
                if last[c] is not None:
                    b.loc[b.index > last[c] + pd.Timedelta(days=7), c] = np.nan
        else:
            b = b.reindex(cal)
        out[f] = pd.concat([a, b], axis=1)[cols]
    printed = pd.concat([printed, out["adjclose"][other].notna()], axis=1)[cols]

    # timestamp fixes: delay pre-switch FX and all crypto by one session (never ahead of the NYSE close)
    fx = [c for c in cols if groups.get(c) == "fx"]
    cr = [c for c in cols if groups.get(c) == "crypto"]
    for f in PRICE_FIELDS:
        pre = out[f].index < FX_SWITCH
        shifted = out[f][fx].shift(1)
        out[f].loc[pre, fx] = shifted.loc[pre]
        out[f][cr] = out[f][cr].shift(1)

    adj = out["adjclose"]
    # bad-print removal on prices (conservative)
    th = {"etf": 0.25, "mf": 0.20, "fut": 0.30, "stocks": 0.50, "crypto": 0.50}
    for grp, t in th.items():
        cc = [c for c in cols if groups.get(c) == grp and c not in U.ETF_VOL]
        if cc:
            adj[cc], n = _despike(adj[cc], t, volume=out["volume"][cc] if grp in ("stocks", "etf") else None)
            report[f"despiked_{grp}"] = n
    adj[fx], report["despiked_fx"] = _despike_volrel(adj[fx])

    ret = _ret_from_last_valid(adj)
    # corporate-action repair for stocks, ETFs and funds
    ca = [c for c in cols if groups.get(c) in NYSE_GROUPS]
    close_ca = out["close"][ca].where(adj[ca].notna())
    ret[ca], rep = _repair_corporate_actions(ret[ca], adj[ca], close_ca)
    report.update(rep)
    for c in U.LEVEL_INDICES:                      # levels are never returns
        if c in ret:
            ret[c] = np.nan
    out["adjclose"] = adj
    out["printed"] = printed & adj.notna()

    # extended long-history twins: ETF spliced with its index-fund (or gold future) proxy before inception
    ext = {}
    for mf, etf in U.MF_PROXY.items():
        if etf and etf in ret and mf in ret and etf in U.EXTEND:
            first = ret[etf].first_valid_index()
            ext[etf + "_X"] = ret[mf].where(ret.index < first).combine_first(ret[etf])
    if "GLD" in ret and "GC=F" in ret:
        first = ret["GLD"].first_valid_index()
        ext["GLD_X"] = ret["GC=F"].where(ret.index < first).combine_first(ret["GLD"])
    for k, v in ext.items():
        ret[k] = v
        groups[k] = "ext"
        out["printed"][k] = v.notna()
    report["extended"] = {k: str(v.first_valid_index().date()) for k, v in ext.items()}
    out["ret"] = ret
    tri = (1 + ret.fillna(0.0)).cumprod()
    started = ret.notna().cumsum() > 0
    out["tri"] = tri.where(started)
    zero = (ret == 0).astype(float).where(ret.notna())
    out["stale63"] = zero.rolling(63, min_periods=20).mean()

    # risk-free: ^IRX (lagged one session), KF daily RF fallback
    kf = pd.read_pickle(raw / "kf.pkl")
    rf_kf = None
    for (f, t), v in kf.items():
        if f == "F-F_Research_Data_Factors_daily":
            rf_kf = v["RF"]
    irx = (out["close"]["^IRX"] / 100.0 / C.TRADING_DAYS).shift(1)
    rf = irx.reindex(cal)
    if rf_kf is not None:
        rf = rf.fillna(rf_kf.reindex(cal))
    out["rf"] = rf.ffill().fillna(0.0).to_frame("rf")

    funded = [c for c in ret.columns if not U.is_excess_return(c)]
    xret = ret.copy()
    xret[funded] = ret[funded].sub(out["rf"]["rf"], axis=0)
    out["xret"] = xret

    stocks_with_data = {c for c in cols if groups.get(c) == "stocks"}
    M = _membership(raw, stocks_with_data, cal, first_print)
    out["sp500"] = M
    out["groups"] = groups.to_frame("group")
    first_member = M.idxmax().where(M.any())
    report["stock_cols_total"] = len(stocks_with_data)
    report["stock_cols_member_by_dev_end"] = int((first_member <= pd.Timestamp(C.DEV_END)).sum())

    kfo = {}
    for (f, t), v in kf.items():
        key = f + ("|EW" if "Equal Weighted" in t else "|VW" if "Value Weighted" in t else "")
        if "Number of Firms" in t or "Average Firm Size" in t or "Sum of BE" in t:
            continue
        kfo[key] = v

    VAULT.mkdir(parents=True, exist_ok=True)
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    # dev: drop stocks that were never members on or before DEV_END (and their membership columns)
    never_dev = set(first_member[~(first_member <= pd.Timestamp(C.DEV_END))].index) | \
        (stocks_with_data - set(M.columns))
    for k, v in out.items():
        v.to_parquet(VAULT / f"{k}.parquet")
        if isinstance(v.index, pd.DatetimeIndex):
            vd = v.loc[:C.DEV_END]
            vd = vd[[c for c in vd.columns if c not in never_dev]]
        else:
            vd = v.loc[[i for i in v.index if i not in never_dev]]
        vd.to_parquet(DEV_DIR / f"{k}.parquet")
    if (raw / "fomc.csv").exists():
        fomc = pd.read_csv(raw / "fomc.csv", parse_dates=["announce"])[["announce"]]
        fomc.to_parquet(VAULT / "fomc.parquet")
        fomc[fomc.announce <= pd.Timestamp(C.DEV_END) + pd.Timedelta(days=90)].to_parquet(DEV_DIR / "fomc.parquet")
    if (raw / "treasury_auctions.csv").exists():
        au = pd.read_csv(raw / "treasury_auctions.csv", parse_dates=["auction", "announce"])
        au = au[["auction", "announce", "orig_term", "term", "type", "reopening"]].dropna(subset=["auction"])
        au.to_parquet(VAULT / "treasury_auctions.parquet")
        au[au.announce <= pd.Timestamp(C.DEV_END)].to_parquet(DEV_DIR / "treasury_auctions.parquet")
    pd.to_pickle(kfo, VAULT / "kf.pkl")
    pd.to_pickle({k: v.loc[:C.DEV_END] for k, v in kfo.items()}, DEV_DIR / "kf.pkl")
    report["dev_stock_cols"] = len(stocks_with_data - never_dev)
    print("report", report)
    print("panels", {k: v.shape for k, v in out.items()})


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
        for e in (end, _GLOBAL_END):
            if e is not None:
                e = pd.Timestamp(e)
                self.end = e if self.end is None else min(e, self.end)

    @lru_cache(maxsize=None)
    def _load(self, name: str) -> pd.DataFrame:
        df = pd.read_parquet(self.dir / f"{name}.parquet")
        if self.end is not None and isinstance(df.index, pd.DatetimeIndex):
            df = df.loc[:self.end]
            if name not in ("rf", "groups") and df.shape[1] > 1:
                hidden = self._future_stocks()
                if hidden:
                    df = df[[c for c in df.columns if c not in hidden]]
        return df

    @lru_cache(maxsize=1)
    def _future_stocks(self) -> frozenset:
        """stocks whose first S&P membership is after self.end (their presence would leak future additions)"""
        M = pd.read_parquet(self.dir / "sp500.parquet")
        first = M.idxmax().where(M.any())
        if self.end is None:
            return frozenset()
        return frozenset(first[~(first <= self.end)].index)

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

    def tri(self, tickers=None) -> pd.DataFrame:
        """total-return index rebuilt from cleaned returns (preferred for price-based signals)"""
        return self.panel("tri", tickers)

    def printed(self, tickers=None) -> pd.DataFrame:
        return self.panel("printed", tickers)

    @property
    def rf(self) -> pd.Series:
        return self._load("rf")["rf"]

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self._load("rf").index

    def groups(self) -> pd.Series:
        g = self._load("groups")["group"]
        hidden = self._future_stocks()
        return g[[i for i in g.index if i not in hidden]]

    def tickers(self, group: str) -> list:
        g = self.groups()
        return sorted(g[g == group].index)

    def sp500(self) -> pd.DataFrame:
        """point-in-time S&P 500 membership (True = member that day) for tickers with Yahoo data"""
        return self._load("sp500")

    @lru_cache(maxsize=8)
    def stock_universe(self, min_hist: int = 252, max_stale: float = 0.30,
                       min_dollar_vol: float = 1e6) -> pd.DataFrame:
        """Boolean (dates x stocks) eligibility at each date using only info <= t:
        point-in-time S&P 500 member, priced today, >= min_hist days of history, share of zero-return days
        over 63d <= max_stale (pre-2001 fractional ticks make 10-25% normal), 63d median dollar volume
        >= min_dollar_vol. NOTE: no price filter on purpose - Yahoo 'close' is split-adjusted, so a price
        floor would use FUTURE splits (excluding later winners) = look-ahead bias. For the same reason, compare
        volume ACROSS stocks only as dollar volume (share volume is split-adjusted with future splits)."""
        M = self.sp500()
        cols = list(M.columns)
        ret = self.ret(cols)
        hist = ret.notna().cumsum()
        close = self.panel("close", cols)
        dv = (close * self.panel("volume", cols)).rolling(63, min_periods=20).median()
        stale = self.panel("stale63", cols)
        ok = M & ret.notna() & (hist >= min_hist) & (stale <= max_stale) & (dv >= min_dollar_vol)
        return ok.fillna(False).astype(bool)

    def fomc_dates(self) -> pd.DatetimeIndex:
        """scheduled FOMC announcement dates. The schedule is published well in advance, so dates shortly after
        the data end are legitimately known (dev copy: up to DEV_END + 90 days). Note: the list is the realised
        meeting list; cancellations (e.g. 2020-03-18) were announced before our 2-day decision lead."""
        f = pd.read_parquet(self.dir / "fomc.parquet")["announce"]
        if self.end is not None:
            f = f[f <= self.end + pd.Timedelta(days=90)]
        return pd.DatetimeIndex(f)

    def treasury_auctions(self) -> pd.DataFrame:
        """Treasury note/bond auctions (TreasuryDirect) that had been ANNOUNCED by the data end: an auction is
        known only from its announcement date (point-in-time)."""
        a = pd.read_parquet(self.dir / "treasury_auctions.parquet")
        if self.end is not None:
            a = a[a.announce <= self.end]
        return a.reset_index(drop=True)

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
        g = self._load("groups")["group"]
        rows = []
        for t in tickers:
            cls = U.asset_class(t)
            hold = C.HOLD_BPS_YR.get(cls, 0.0)
            if cls == "kf_factor":       # factor-specific drag, e.g. KF:F:US:Mom
                hold = C.KF_FACTOR_DRAG_BPS_YR.get(t.split(":")[-1], hold)
            tradable = t.startswith("KF:") or g.get(t) in TRADABLE_GROUPS
            rows.append({"ticker": t, "cls": cls, "cost_bps": C.COST_BPS[cls], "borrow_bps": C.BORROW_BPS_YR[cls],
                         "hold_bps": hold, "funded": not U.is_excess_return(t), "tradable": bool(tradable)})
        return pd.DataFrame(rows).set_index("ticker")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
