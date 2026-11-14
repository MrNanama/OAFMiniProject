"""Performance statistics for daily EXCESS returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TD = 252


def _clean(x: pd.Series) -> pd.Series:
    return x.dropna().astype(float)


def sharpe(x) -> float:
    x = _clean(pd.Series(x))
    s = x.std()
    return float(x.mean() / s * np.sqrt(TD)) if s > 0 else np.nan


def sortino(x) -> float:
    x = _clean(pd.Series(x))
    dd = np.sqrt((np.minimum(x, 0) ** 2).mean())
    return float(x.mean() / dd * np.sqrt(TD)) if dd > 0 else np.nan


def drawdown(x: pd.Series) -> pd.Series:
    """drawdown of compounded wealth, starting from initial capital 1.0 (a first-day loss counts)"""
    w = (1 + _clean(x)).cumprod()
    peak = np.maximum(w.cummax(), 1.0)
    return (w / peak - 1).clip(lower=-1.0)


def max_dd_duration(x: pd.Series) -> int:
    """longest stretch (trading days) below a previous equity high"""
    dd = drawdown(x).to_numpy()
    best = cur = 0
    for v in dd:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return int(best)


def cagr(x: pd.Series) -> float:
    x = _clean(x)
    if len(x) == 0:
        return np.nan
    w = float((1 + x).prod())
    return w ** (TD / len(x)) - 1 if w > 0 else -1.0


def hac_tstat(x: pd.Series, lags: int = 10) -> float:
    """Newey-West t-stat of the mean"""
    x = _clean(x).to_numpy()
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def psr(x: pd.Series, sr_bench_annual: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012): P(true SR > benchmark)"""
    x = _clean(x)
    T = len(x)
    if T < 30 or x.std() == 0:
        return np.nan
    sr = x.mean() / x.std()
    sr_b = sr_bench_annual / np.sqrt(TD)
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_b) * np.sqrt(T - 1) / den))


def expected_max_sr(n_trials: int, sr_std_annual: float) -> float:
    """E[max] of n_trials iid N(0, sr_std) Sharpe estimates (annualised)"""
    if n_trials <= 1:
        return 0.0
    g = np.euler_gamma
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sr_std_annual * ((1 - g) * z1 + g * z2))


def dsr(x: pd.Series, n_trials: int, sr_std_annual: float | None = None) -> float:
    """Deflated Sharpe Ratio: PSR against the expected max Sharpe of n_trials zero-skill strategies.
    sr_std_annual defaults to the sampling std of an annualised SR estimate, sqrt(252/T)."""
    x = _clean(x)
    T = len(x)
    if sr_std_annual is None:
        sr_std_annual = np.sqrt(TD / max(T, 1))
    return psr(x, expected_max_sr(n_trials, sr_std_annual))


def boot_sharpe_ci(x: pd.Series, n: int = 1000, block: int = 21, seed: int = 0, q=(2.5, 97.5)):
    """circular block bootstrap CI for the annualised Sharpe"""
    v = _clean(x).to_numpy()
    T = len(v)
    if T < 2 * block:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T
    s = v[idx.reshape(n, -1)[:, :T]]
    sr = s.mean(axis=1) / s.std(axis=1) * np.sqrt(TD)
    return tuple(float(np.percentile(sr, qq)) for qq in q)


def subperiod_sharpes(x: pd.Series, k: int = 3) -> list:
    x = _clean(x)
    parts = np.array_split(np.arange(len(x)), k)
    return [round(sharpe(x.iloc[p]), 3) for p in parts]


def perf_stats(net: pd.Series, bench: pd.Series | None = None, turnover: pd.Series | None = None,
               cost: pd.Series | None = None, gross_lev: pd.Series | None = None, n_trials: int = 1,
               start=None, end=None, first_trade=None, fees: pd.Series | None = None,
               n_trials_program: int | None = None, sr_std_program: float | None = None) -> dict:
    """Stats on daily excess returns over [start, end]. Only the engine warm-up (before `first_trade`, the first
    day with a position) is trimmed; flat periods the strategy chose are kept."""
    x = net.copy()
    if first_trade is not None:
        st = pd.Timestamp(first_trade) + pd.Timedelta(days=1)
        start = st if start is None else max(pd.Timestamp(start), st)
    if start is not None:
        x = x[x.index >= pd.Timestamp(start)]
    if end is not None:
        x = x[x.index <= pd.Timestamp(end)]
    if len(x) < 20 or x.std() == 0:
        return {"sharpe": np.nan, "n_years": round(len(x) / TD, 2)}
    yrs = len(x) / TD
    dd = drawdown(x)
    mdd = float(dd.min())
    c = cagr(x)
    m = x.resample("ME").apply(lambda s: (1 + s).prod() - 1)
    y = x.resample("YE").apply(lambda s: (1 + s).prod() - 1)
    out = {
        "start": str(x.index[0].date()), "end": str(x.index[-1].date()), "n_years": round(yrs, 2),
        "ann_ret": float(x.mean() * TD), "cagr": c, "ann_vol": float(x.std() * np.sqrt(TD)),
        "sharpe": sharpe(x), "sortino": sortino(x), "max_dd": mdd,
        "calmar": float(c / abs(mdd)) if mdd < 0 else np.nan, "max_dd_days": max_dd_duration(x),
        "skew": float(stats.skew(x)), "kurt": float(stats.kurtosis(x)),
        "hit_daily": float((x > 0).mean()), "hit_monthly": float((m > 0).mean()),
        "worst_day": float(x.min()), "worst_month": float(m.min()), "best_month": float(m.max()),
        "pos_years": f"{int((y > 0).sum())}/{len(y)}",
        "t_hac": hac_tstat(x), "psr0": psr(x, 0.0), "dsr": dsr(x, max(n_trials, 1)), "n_trials": int(n_trials),
        "sr_subperiods": subperiod_sharpes(x, 3),
    }
    lo, hi = boot_sharpe_ci(x)
    out["sr_ci_lo"], out["sr_ci_hi"] = lo, hi
    if turnover is not None:
        out["turnover_yr"] = float(turnover.reindex(x.index).mean() * TD)
    if cost is not None:
        out["cost_drag_yr"] = float(cost.reindex(x.index).mean() * TD)
    if fees is not None:
        out["fee_drag_yr"] = float(fees.reindex(x.index).mean() * TD)
        out["total_drag_yr"] = out.get("cost_drag_yr", 0.0) + out["fee_drag_yr"]
    if n_trials_program:
        out["n_trials_program"] = int(n_trials_program)
        out["dsr_program"] = dsr(x, n_trials_program, sr_std_program)
    if gross_lev is not None:
        out["avg_gross_lev"] = float(gross_lev.reindex(x.index).mean())
    if bench is not None:
        b = bench.reindex(x.index).fillna(0.0)
        if b.std() > 0:
            beta = float(np.cov(x, b)[0, 1] / b.var())
            out["beta_spy"] = beta
            out["corr_spy"] = float(np.corrcoef(x, b)[0, 1])
            out["alpha_t_hac"] = hac_tstat(x - beta * b)
    return out


def fmt_table(rows: dict, cols=("sharpe", "ann_ret", "ann_vol", "max_dd", "calmar", "sortino", "turnover_yr",
                                "cost_drag_yr", "beta_spy", "dsr", "sr_subperiods", "n_years")) -> str:
    """markdown table: {name: stats}"""
    head = "| strategy | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    lines = []
    for k, s in rows.items():
        vals = []
        for c in cols:
            v = s.get(c, np.nan)
            if isinstance(v, float):
                vals.append(f"{v:.1%}" if c in ("ann_ret", "ann_vol", "max_dd", "cost_drag_yr", "cagr") else f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append(f"| {k} | " + " | ".join(vals) + " |")
    return head + "\n".join(lines) + "\n"
