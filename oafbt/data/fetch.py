"""Download raw market data into data/raw/ (gitignored).

  python -m oafbt.data.fetch            # everything
  python -m oafbt.data.fetch yahoo      # only Yahoo groups
  python -m oafbt.data.fetch kf         # only Ken French
  python -m oafbt.data.fetch sp500      # only S&P 500 membership + member prices

Sources: Yahoo Finance (via yfinance), Kenneth French data library, fja05680/sp500 (point-in-time S&P 500 membership).
"""
import io
import sys
import time
import zipfile
import warnings

import numpy as np
import pandas as pd
import requests

from ..config import RAW
from . import universe as U

warnings.filterwarnings("ignore")
UA = {"User-Agent": "Mozilla/5.0 (research backtester)"}


# ----------------------------------------------------------------------------------------------- Yahoo
def _yf():
    import yfinance as yf
    cache = RAW / "yf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(cache))
    except Exception:
        pass
    return yf


def _download(tickers, start="1985-01-01", batch=60, retries=3):
    """returns long-format frame: date, ticker, open, high, low, close, adjclose, volume"""
    yf = _yf()
    out = []
    todo = list(dict.fromkeys(tickers))
    for attempt in range(retries):
        failed = []
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            try:
                d = yf.download(chunk, start=start, auto_adjust=False, actions=False, progress=False,
                                threads=True, group_by="column")
            except Exception as e:  # network hiccup: retry whole chunk later
                print("  chunk error", e)
                failed += chunk
                continue
            if d is None or d.empty:
                failed += chunk
                continue
            if not isinstance(d.columns, pd.MultiIndex):  # single ticker
                d.columns = pd.MultiIndex.from_product([d.columns, chunk])
            for t in chunk:
                try:
                    sub = pd.DataFrame({
                        "open": d[("Open", t)], "high": d[("High", t)], "low": d[("Low", t)],
                        "close": d[("Close", t)], "adjclose": d[("Adj Close", t)], "volume": d[("Volume", t)]})
                except KeyError:
                    failed.append(t)
                    continue
                sub = sub.dropna(subset=["close", "adjclose"])
                if len(sub) == 0:
                    failed.append(t)
                    continue
                sub["ticker"] = t
                out.append(sub)
            time.sleep(0.5)
        print(f"  attempt {attempt}: ok={len(todo) - len(failed)} failed={len(failed)}")
        if not failed:
            break
        todo = failed
        time.sleep(3)
    if not out:
        return pd.DataFrame(), todo
    df = pd.concat(out)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df, (failed if out else todo)


def fetch_yahoo_groups():
    groups = {
        "etf": U.ETF_ALL,
        "mf": list(U.MF_PROXY),
        "fut": U.FUT_FIN + U.FUT_CMDTY,
        "fx": U.FX,
        "index": U.INDEX,
        "crypto": U.CRYPTO,
    }
    RAW.mkdir(parents=True, exist_ok=True)
    for g, tick in groups.items():
        print(f"[yahoo:{g}] {len(tick)} tickers")
        df, failed = _download(tick, start="1985-01-01" if g != "crypto" else "2014-01-01")
        df.to_parquet(RAW / f"yahoo_{g}.parquet")
        print(f"  saved {df['ticker'].nunique() if len(df) else 0} tickers, failed: {failed}")


# ----------------------------------------------------------------------------------------------- S&P 500
SP500_BASE = "https://raw.githubusercontent.com/fja05680/sp500/master/"


def fetch_sp500():
    RAW.mkdir(parents=True, exist_ok=True)
    hist = requests.get(SP500_BASE + "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv",
                        headers=UA, timeout=60)
    hist.raise_for_status()
    (RAW / "sp500_membership.csv").write_bytes(hist.content)
    mem = pd.read_csv(io.BytesIO(hist.content))
    tick = sorted({t for row in mem["tickers"] for t in row.split(",")})
    ytick = [t.replace(".", "-") for t in tick]
    print(f"[sp500] {len(tick)} unique historical members since {mem['date'].iloc[0]}")
    df, failed = _download(ytick, start="1990-01-01", batch=80)
    df.to_parquet(RAW / "yahoo_stocks.parquet")
    pd.Series(failed, name="ticker").to_csv(RAW / "sp500_missing_on_yahoo.csv", index=False)
    print(f"  saved {df['ticker'].nunique()} tickers with data; {len(failed)} not on Yahoo (delisted/renamed)")


# ----------------------------------------------------------------------------------------------- Ken French
KF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
KF_FILES = [
    "F-F_Research_Data_Factors_daily", "F-F_Research_Data_5_Factors_2x3_daily", "F-F_Momentum_Factor_daily",
    "F-F_ST_Reversal_Factor_daily", "F-F_LT_Reversal_Factor_daily", "49_Industry_Portfolios_daily",
    "12_Industry_Portfolios_daily", "25_Portfolios_5x5_Daily", "10_Portfolios_Prior_12_2_Daily",
    "25_Portfolios_ME_Prior_12_2_Daily", "Developed_5_Factors_Daily", "Developed_Mom_Factor_Daily",
    "Europe_5_Factors_Daily", "Europe_Mom_Factor_Daily", "Japan_5_Factors_Daily", "Japan_Mom_Factor_Daily",
    "Asia_Pacific_ex_Japan_5_Factors_Daily", "Asia_Pacific_ex_Japan_Mom_Factor_Daily",
    "North_America_5_Factors_Daily", "North_America_Mom_Factor_Daily",
    "Portfolios_Formed_on_BE-ME_Daily", "Portfolios_Formed_on_ME_Daily",
]


def parse_kf_csv(text: str) -> dict:
    """Ken French CSVs hold several tables (e.g. value- and equal-weighted). Returns {title: DataFrame} with
    returns in decimal (not percent) and a DatetimeIndex. Only daily (8-digit date) tables are kept."""
    lines = text.splitlines()
    tables, title, header, rows = {}, "main", None, []

    def flush():
        if header is not None and rows:
            df = pd.DataFrame(rows, columns=["date"] + header)
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
            df = df.replace([-99.99, -999], np.nan) / 100.0
            tables[title] = df

    for ln in lines:
        s = ln.strip()
        parts = [p.strip() for p in s.split(",")]
        if len(parts) > 1 and parts[0] == "" and header is None:
            header = parts[1:]
            continue
        if len(parts) > 1 and len(parts[0]) == 8 and parts[0].isdigit():
            if header is None:
                header = [f"c{i}" for i in range(len(parts) - 1)]
            rows.append(parts[:len(header) + 1] if header else parts)
            continue
        # anything else ends a table
        if rows:
            flush()
            rows, header = [], None
        if s and not s.startswith(",") and len(parts) <= 2:
            title = s
        if len(parts) > 1 and parts[0] == "":
            header = parts[1:]
    if rows:
        flush()
    return tables


def fetch_kf():
    out = {}
    for f in KF_FILES:
        try:
            r = requests.get(KF_URL.format(f), headers=UA, timeout=120)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            txt = z.read(z.namelist()[0]).decode("latin1")
            tabs = parse_kf_csv(txt)
            for k, v in tabs.items():
                out[(f, k)] = v
            print(f"[kf] {f}: tables={[(k[:40], v.shape, str(v.index[0].date()), str(v.index[-1].date())) for k, v in tabs.items()]}")
        except Exception as e:
            print(f"[kf] {f}: FAILED {e}")
    pd.to_pickle(out, RAW / "kf.pkl")


# ----------------------------------------------------------------------------------------------- Treasury auctions
def fetch_auctions(first_year=1998, last_year=2026):
    """Treasury note/bond auctions with announcement dates (TreasuryDirect securities search API)."""
    rows = []
    for t in ["Note", "Bond"]:
        for y in range(first_year, last_year + 1):
            url = (f"https://www.treasurydirect.gov/TA_WS/securities/search?format=json&type={t}"
                   f"&startDate={y}-01-01&endDate={y}-12-31")
            try:
                d = requests.get(url, headers=UA, timeout=60).json()
            except Exception as e:
                print("auctions", t, y, "failed", e)
                continue
            if not isinstance(d, list):
                continue
            for x in d:
                rows.append({"type": t, "term": x.get("securityTerm"), "orig_term": x.get("originalSecurityTerm"),
                             "auction": x.get("competitiveAuctionCloseDate") or x.get("auctionDate"),
                             "announce": x.get("announcementDate"), "reopening": x.get("reopening"),
                             "cusip": x.get("cusip")})
            time.sleep(0.2)
    df = pd.DataFrame(rows).drop_duplicates(subset=["cusip", "auction"])
    df["auction"] = pd.to_datetime(df["auction"])
    df["announce"] = pd.to_datetime(df["announce"])
    df = df.dropna(subset=["auction"]).sort_values("auction")
    df.to_csv(RAW / "treasury_auctions.csv", index=False)
    print("[auctions]", len(df), df.auction.min().date(), "->", df.auction.max().date())


# ----------------------------------------------------------------------------------------------- FOMC schedule
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def _month_num(tok: str):
    tok = tok.strip()[:3].title()
    ab = [m[:3] for m in MONTHS]
    return ab.index(tok) + 1 if tok in ab else None


def fetch_fomc(first_year=1994, last_hist_year=2020):
    """Scheduled FOMC meetings (announcement date = last day of the meeting). Historical pages per year up to
    `last_hist_year`, current calendar page for later years. Conference calls / unscheduled actions are excluded.
    The schedule is published a year or more in advance, so using it is not look-ahead."""
    import re
    rows = []
    for y in range(first_year, last_hist_year + 1):
        url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
        try:
            html = requests.get(url, headers=UA, timeout=60).text
        except Exception as e:
            print("fomc", y, "failed", e)
            continue
        M = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
        pat = (r"(" + M + r")(?:/(" + M + r"))?\s+(\d{1,2})(?:\s*-\s*(?:(" + M + r")\s+)?(\d{1,2}))?"
               r"\s+(Meeting|Conference Call)")
        for m in re.finditer(pat, html):
            if m.group(6) != "Meeting":
                continue
            mon_end = _month_num(m.group(4) or m.group(2) or m.group(1))
            d_end = int(m.group(5) or m.group(3))
            rows.append({"year": y, "announce": pd.Timestamp(y, mon_end, d_end), "src": "historical"})
    html = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=UA, timeout=60).text
    for block in re.split(r'<div class="panel panel-default">', html)[1:]:
        ym = re.search(r"(\d{4}) FOMC Meetings", block)
        if not ym:
            continue
        y = int(ym.group(1))
        if y <= last_hist_year:
            continue
        for mm in re.finditer(r'fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<', block, re.S):
            mon = _month_num(mm.group(1).strip().split("/")[-1])
            days = re.findall(r"\d{1,2}", mm.group(2))
            if mon and days:
                rows.append({"year": y, "announce": pd.Timestamp(y, mon, int(days[-1])), "src": "calendar"})
    df = pd.DataFrame(rows).drop_duplicates("announce").sort_values("announce").reset_index(drop=True)
    # sanity: one announcement per meeting (keep the later of two dates < 5 days apart); modern statements are
    # released mid-week, so drop post-2000 Friday/weekend artefacts of the page parser
    gap_next = df["announce"].shift(-1) - df["announce"]
    df = df[~(gap_next < pd.Timedelta(days=5))]
    df = df[~((df["announce"].dt.year > 2000) & (df["announce"].dt.dayofweek >= 4))]
    df.to_csv(RAW / "fomc.csv", index=False)
    print("[fomc]", len(df), "scheduled meetings", df.announce.min().date(), "->", df.announce.max().date(),
          df.groupby("year").size().to_dict())
    return df


if __name__ == "__main__":
    what = sys.argv[1:] or ["yahoo", "kf", "sp500", "fomc", "auctions"]
    if "auctions" in what:
        fetch_auctions()
    if "fomc" in what:
        fetch_fomc()
    if "kf" in what:
        fetch_kf()
    if "yahoo" in what:
        fetch_yahoo_groups()
    if "sp500" in what:
        fetch_sp500()
