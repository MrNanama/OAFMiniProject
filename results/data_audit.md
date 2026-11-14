# Data audit (dev panels, <= 2020-12-31)

## Coverage
| group | n tickers | first dates (median) |
|---|---|---|
| etf | 100 | 2003-04-22 |
| mf | 20 | 1988-09-13 |
| fut | 41 | 2000-09-13 |
| fx | 28 | 2003-12-01 |
| index | 33 | 1990-01-02 |
| crypto | 2 | 2016-04-13 |
| stocks | 759 | 1993-05-15 |

## Non-positive adjusted closes in raw Yahoo data: 740 rows, tickers: {'DEC': 732, '^IRX': 7, 'CL=F': 1}

## Futures (Yahoo continuous) vs ETF, daily returns, common dates
| fut | etf | corr | ann.mean fut | ann.mean etf | ann.vol fut | ann.vol etf | #days |fut-etf|>3sd |
|---|---|---|---|---|---|---|---|
| ES=F | SPY | 0.976 | 9.1% | 11.1% | 21.0% | 20.6% | 51 |
| NQ=F | QQQ | 0.967 | 16.7% | 17.5% | 22.3% | 22.1% | 64 |
| ZN=F | IEF | 0.900 | 1.9% | 5.4% | 5.9% | 6.7% | 57 |
| ZB=F | TLT | 0.878 | 3.7% | 8.3% | 10.9% | 14.9% | 36 |
| GC=F | GLD | 0.869 | 9.5% | 9.1% | 18.3% | 18.1% | 54 |
| SI=F | SLV | 0.887 | 10.6% | 9.7% | 33.0% | 31.6% | 55 |
| CL=F | USO | 0.853 | 12.5% | -8.7% | 43.6% | 36.8% | 39 |
| NG=F | UNG | 0.881 | 4.5% | -28.5% | 50.1% | 43.0% | 51 |
| 6E=F | FXE | 0.960 | -0.1% | -0.1% | 9.3% | 9.4% | 75 |
| 6J=F | FXY | 0.952 | 1.6% | 1.2% | 10.0% | 10.2% | 65 |
| HG=F | CPER | 0.784 | 1.6% | 0.2% | 20.1% | 21.5% | 41 |
| ZC=F | DBA | 0.649 | 6.4% | -1.2% | 29.6% | 17.6% | 41 |

## Share of exactly-zero daily returns (2016-2020)
| group | median | worst 5 |
|---|---|---|
| etf | 1.20% | {'UDN': 0.051, 'CPER': 0.053, 'SHY': 0.1, 'VXZ': 0.245, 'SHV': 0.28} |
| fut | 1.03% | {'KE=F': 0.02, 'ZF=F': 0.021, 'ZC=F': 0.025, 'ZN=F': 0.029, 'ZT=F': 0.063} |
| fx | 0.48% | {'CLP=X': 0.014, 'IDR=X': 0.017, 'HKD=X': 0.021, 'THB=X': 0.021, 'CNY=X': 0.024} |
| index | 1.67% | {'^TWII': 0.054, '^KS11': 0.056, '^N225': 0.059, '^SKEW': 0.06, '^IRX': 0.115} |
| stocks | 0.56% | {'AMCR': 0.585, 'SW': 0.647, 'EP': 0.663, 'FERG': 0.684, 'DIGI': 0.74} |

## Largest absolute daily returns (after cleaning)
- etf: SVXY@2018-02-05=-32.0%, VXX@2018-02-05=+33.5%, VIXY@2020-06-11=+33.6%, VXX@2020-06-11=+33.7%, VIXY@2018-02-05=+34.2%, VXX@2020-03-16=+37.1%, VIXY@2020-03-16=+39.1%, SVXY@2018-02-06=-83.0%
- fut: SB=F@2000-05-01=+26.6%, HE=F@2018-04-16=+26.7%, NG=F@2006-09-28=+28.4%, NG=F@2009-09-29=+30.7%, RB=F@2020-03-23=-32.0%, PL=F@2000-04-25=+37.1%, CL=F@2020-04-22=+37.7%, NG=F@2003-02-24=+38.2%
- fx: KRW=X@2008-10-30=+10.6%, IDR=X@2001-07-26=+10.8%, ILS=X@2011-03-01=+11.0%, BRL=X@2008-10-09=+11.7%, ZAR=X@2008-10-15=-14.4%, TRY=X@2018-08-13=-18.6%, CHF=X@2015-01-16=+19.3%, IDR=X@2013-11-29=+22.2%
- stocks: CPWR@2011-12-23=+100.0%, EP@2003-03-27=+100.0%, MCIC@2016-10-07=+100.0%, USBC@2009-04-17=+100.0%, AN@1995-05-22=+100.0%, EP@2018-12-21=+100.0%, CPWR@2011-01-13=+100.0%, CPWR@2012-10-17=+100.0%

## S&P 500 point-in-time members WITH Yahoo data: 1996: 269, 2000: 295, 2004: 315, 2008: 340, 2012: 367, 2016: 407, 2020: 448

## Risk-free (annualised mean by decade): 1990s: 4.70%, 2000s: 2.72%, 2010s: 0.50%

## Mutual-fund proxies vs ETFs (daily corr, ann. mean diff, common dates)
- VFINX->SPY: corr 0.982, mean diff -0.02%/yr, n=7032, mf starts 1985-01-02
- NAESX->IWM: corr 0.981, mean diff +0.50%/yr, n=5182, mf starts 1985-01-02
- VUSTX->TLT: corr 0.978, mean diff -0.45%/yr, n=4639, mf starts 1986-05-19
- VFITX->IEF: corr 0.939, mean diff -0.97%/yr, n=4639, mf starts 1991-10-28
- VFISX->SHY: corr 0.777, mean diff +0.34%/yr, n=4639, mf starts 1991-10-28
- VWESX->LQD: corr 0.578, mean diff +1.64%/yr, n=4639, mf starts 1985-01-02
- VWEHX->HYG: corr 0.491, mean diff +0.19%/yr, n=3457, mf starts 1985-01-02
- VFIIX->MBB: corr 0.748, mean diff +0.30%/yr, n=3474, mf starts 1985-01-02
- VGPMX->GLD: corr 0.472, mean diff -3.95%/yr, n=4057, mf starts 1985-01-02
- VGSIX->VNQ: corr 0.994, mean diff +0.15%/yr, n=4093, mf starts 1996-05-13
- VEIEX->EEM: corr 0.935, mean diff -1.69%/yr, n=4461, mf starts 1994-05-04
- VEURX->VGK: corr 0.976, mean diff -0.42%/yr, n=3981, mf starts 1990-06-18
- VPACX->EPP: corr 0.835, mean diff -3.78%/yr, n=4828, mf starts 1990-06-18
- VTMGX->EFA: corr 0.965, mean diff +0.02%/yr, n=4867, mf starts 1999-08-17
- VBMFX->AGG: corr 0.689, mean diff +0.01%/yr, n=4345, mf starts 1986-12-11
- VIPSX->TIP: corr 0.933, mean diff -0.21%/yr, n=4297, mf starts 2000-06-29
- PCRIX->DBC: corr 0.849, mean diff -0.04%/yr, n=3752, mf starts 2002-07-01
- VXX: 2018-01-25 -> 2020-12-31, n=740, ann.mean ret +14.7%
- VIXY: 2011-01-04 -> 2020-12-31, n=2516, ann.mean ret -40.6%
- SVXY: 2011-10-04 -> 2020-12-31, n=2327, ann.mean ret +34.9%
- VIXM: 2011-01-04 -> 2020-12-31, n=2516, ann.mean ret -16.1%
- VXZ: 2018-01-25 -> 2020-12-31, n=740, ann.mean ret +28.6%

## Survivorship & placebo checks (stock universe = point-in-time S&P 500 members with Yahoo data)
| portfolio | 2000-2020 ann. excess / SR | 2004-2020 ann. excess / SR |
|---|---|---|
| equal-weight eligible stocks (monthly, 5 bps) | 11.5% / 0.54 | 12.0% / 0.56 |
| SPY | 6.7% / 0.34 | 9.7% / 0.51 |
| RSP (real equal-weight S&P 500 ETF, incl. later-delisted names; 2003+) | - | 10.3% / 0.49 |

=> survivorship bias of our stock universe is roughly **+1.7%/yr** for a long-only equal-weight book (2004-2020).
Long-short stock strategies can be biased in either direction (missing delisted names are mostly acquisitions at a premium
and failures); treat stock long-short Sharpe differences below ~0.2 as noise.

Random monthly quintile long-short placebo (eligible stocks, 10% vol): Sharpe -0.51, -2.8%/yr (= costs and fees at
~39x turnover), i.e. the engine produces no spurious edge.
