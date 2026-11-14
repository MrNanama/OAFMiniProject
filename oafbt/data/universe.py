"""Ticker universes (Yahoo Finance symbols) and per-asset metadata.

Groups
------
ETF_*      exchange-traded funds (total-return via Adj Close)
MF_PROXY   long-history mutual funds, used only to extend ETF histories back in time
FUT_*      Yahoo continuous front-month futures (NOT back-adjusted; see data audit)
FX         spot FX (price return only; no carry)
INDEX      non-tradable indices used as signals (VIX, yields, foreign indices)
CRYPTO     BTC, ETH
Stocks come from the point-in-time S&P 500 membership file (see fetch.py).
"""

ETF_CORE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "TIP", "GLD", "VNQ", "DIA", "MDY"]
ETF_SECTOR = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
ETF_STYLE = ["IJR", "IVE", "IVW", "RSP", "IWD", "IWF", "IWN", "IWO"]
ETF_COUNTRY = ["VGK", "EWJ", "EWG", "EWU", "EWC", "EWA", "EWH", "EWS", "EWT", "EWY", "EWZ", "EWW", "EWQ",
               "EWI", "EWP", "EWL", "EWD", "EWN", "FXI", "EZA", "ILF", "EPP", "EWK", "EWO", "EIS", "THD", "TUR", "INDA"]
ETF_BOND = ["IEI", "EMB", "MUB", "BWX", "MBB", "JNK", "BND", "TLH", "SHV", "CWB", "PFF"]
ETF_CMDTY = ["SLV", "DBC", "USO", "UNG", "DBA", "DBB", "DBE", "DBP", "GSG", "DBO", "UGA", "CPER", "PALL", "PPLT"]
ETF_REIT = ["IYR", "RWX"]
ETF_FX = ["UUP", "UDN", "FXE", "FXY", "FXB", "FXA", "FXC", "FXF"]
ETF_VOL = ["VIXY", "SVXY", "VXX", "VIXM", "VXZ"]

# long-history mutual funds (Vanguard etc.) used to extend asset-class histories before ETF inception
MF_PROXY = {"VFINX": "SPY", "NAESX": "IWM", "VUSTX": "TLT", "VFITX": "IEF", "VFISX": "SHY", "VWESX": "LQD",
            "VWEHX": "HYG", "VFIIX": "MBB", "VGPMX": "GLD", "VGSIX": "VNQ", "VEIEX": "EEM", "VEURX": "VGK",
            "VPACX": "EPP", "VTMGX": "EFA", "VBMFX": "AGG", "VIPSX": "TIP", "PCRIX": "DBC", "FSAGX": None,
            "VGTSX": None, "VWIGX": None}

# ETFs that get a spliced long-history twin "<ETF>_X" (index mutual fund before ETF inception)
EXTEND = {"SPY", "IWM", "TLT", "IEF", "EFA", "EEM", "VNQ", "DBC", "VGK", "TIP", "AGG"}

FUT_FIN = ["ES=F", "NQ=F", "YM=F", "RTY=F", "ZN=F", "ZB=F", "ZF=F", "ZT=F", "6E=F", "6J=F", "6B=F", "6A=F",
           "6C=F", "6S=F", "6N=F", "6M=F", "DX=F", "NKD=F"]
FUT_CMDTY = ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F", "CL=F", "BZ=F", "NG=F", "HO=F", "RB=F", "ZC=F", "ZS=F",
             "ZW=F", "ZL=F", "ZM=F", "KE=F", "KC=F", "SB=F", "CC=F", "CT=F", "OJ=F", "LE=F", "HE=F", "GF=F"]
FX = ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "NZDUSD=X", "CAD=X", "CHF=X", "SEK=X", "NOK=X", "MXN=X",
      "ZAR=X", "BRL=X", "INR=X", "KRW=X", "SGD=X", "TRY=X", "PLN=X", "HUF=X", "CZK=X", "ILS=X", "CNY=X",
      "HKD=X", "IDR=X", "THB=X", "TWD=X", "CLP=X", "COP=X", "PHP=X"]
INDEX = ["^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT", "^VIX", "^VIX3M", "^VIX9D", "^VVIX", "^SKEW", "^VXN", "^OVX",
         "^GVZ", "^EVZ", "^MOVE", "^TNX", "^IRX", "^FVX", "^TYX", "^N225", "^FTSE", "^GDAXI", "^FCHI", "^HSI",
         "^STOXX50E", "^AXJO", "^GSPTSE", "^BVSP", "^MXX", "^KS11", "^TWII", "DX-Y.NYB", "^SOX", "^SPGSCI"]
CRYPTO = ["BTC-USD", "ETH-USD"]
# indices that are LEVELS (volatility indices, yields): never de-spiked or capped, never traded
LEVEL_INDICES = ["^VIX", "^VIX3M", "^VIX9D", "^VVIX", "^SKEW", "^VXN", "^OVX", "^GVZ", "^EVZ", "^MOVE",
                 "^TNX", "^IRX", "^FVX", "^TYX"]

ETF_ALL = ETF_CORE + ETF_SECTOR + ETF_STYLE + ETF_COUNTRY + ETF_BOND + ETF_CMDTY + ETF_REIT + ETF_FX + ETF_VOL


def asset_class(ticker: str) -> str:
    """cost/borrow category used by the engine"""
    if ticker.endswith("_X"):
        return asset_class(ticker[:-2])
    if ticker in ETF_CORE:
        return "etf_core"
    if ticker in ETF_VOL:
        return "etf_vol"
    if ticker in ETF_ALL:
        return "etf"
    if ticker in MF_PROXY:
        return "mutual_fund"
    if ticker in FUT_FIN:
        return "fut_fin"
    if ticker in FUT_CMDTY:
        return "fut_cmdty"
    if ticker in FX:
        return "fx"
    if ticker in CRYPTO:
        return "crypto"
    if ticker.startswith("^") or ticker == "DX-Y.NYB":
        return "index"
    if ticker.startswith("KF:"):
        return "kf_factor" if ticker.startswith("KF:F:") else "kf_port"
    return "stock"


def is_excess_return(ticker: str) -> bool:
    """futures, FX-spot and long-short factor returns are already excess of cash (no funding needed)"""
    return asset_class(ticker) in ("fut_fin", "fut_cmdty", "fx", "kf_factor")
