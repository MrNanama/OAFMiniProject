"""s04 - Abnormal-volume (attention) premium on S&P 500 stocks: the high-volume return premium at a monthly horizon.

Inspired by the Volume Filters deck (volume tells us *who* is trading). The deck conditioned 1-day momentum/reversal
trades after 2-sigma moves in 50 stocks on that day's normalised volume (no costs, 1-day holding, P&L from one stock).
This strategy uses volume differently: as a slow-moving *visibility* signal harvested over a month across the whole
point-in-time S&P 500, net of costs.

Thesis (Gervais, Kaniel & Mingelgrin 2001, "The High-Volume Return Premium"; Kaniel, Ozoguz & Starks 2012; Merton 1987
investor recognition; Miller 1977): a stock whose trading activity is unusually high relative to its own history becomes
more visible. New investors become aware of it, its investor base and demand widen, and with short-sale constraints the
price rises over the following weeks. Stocks with unusually LOW activity lose visibility. Holding the same-window
return fixed (so the effect is not momentum or reversal), high-abnormal-volume stocks outperform low-abnormal-volume
stocks over the next month and beyond (GKM: the premium does not reverse over 100 days).

Pipeline (a weight row dated d uses only data dated <= d):
1. Abnormal volume (split-adjusted Yahoo share volume, zero prints = missing):
       AV = log( mean V over the last n=21 days / mean V over the 252 days before that window ).
   A ratio to the stock's own history is unaffected by future split adjustments and by the stock's size.
2. Neutralisation, cross-sectional OLS on each formation date over the eligible stocks (ds.stock_universe()):
       AV = a + b1 z(r_n) + b2 |z(r_n)| + b3 z(log DV_base) + b4 z(beta_252) + b5 z(vol_21) + e
   * r_n: same-window log return, plus its absolute value. Volume rises with big moves in either direction (V-shape;
     GKM sort within return deciles for this reason), and the 1-month return is itself a known predictor (reversal).
   * DV_base: mean dollar volume (close x volume) over the baseline window, a market-cap/liquidity proxy.
   * beta_252 vs SPY: the AV-beta cross-sectional correlation is ~0 on average but swings with sd 0.15 over time
     (market-wide volume surges hit high-beta stocks first). Uncontrolled, it would be a time-varying market bet.
   * vol_21: 1-month realised volatility. AV correlates +0.24 with it, and high 1-month volatility predicts LOW returns
     (Ang, Hodrick, Xing & Zhang 2006), which would mask the volume effect.
   Controls are z-scored and clipped at +-3; AV is winsorised at +-3 sd. The signal is the residual e.
3. Portfolio: on the last trading day of every week (published NYSE calendar), a tranche goes long the top quintile and
   short the bottom quintile of e, equal weight, each leg summing to 1 (dollar neutral). The book is the average of
   the last 4 weekly tranches, so each tranche is held 4 weeks (~1 month) and timing luck is diversified. At every
   weekly rebalance, names no longer eligible are dropped and the tranche's leg is re-normalised.
4. The engine executes at the next close (lag 1), charges 5 bps one-way plus 30 bps/yr borrow, and scales to 10% ex-ante
   vol under the 2x gross cap (a dollar-neutral quintile book at gross 2 usually sits at the cap).

Development path (dev data only; results/runs/s04_volume_attention/v1/notes.md):
* run 1 = these defaults (a priori spec): SR -0.31 (2004-20 -0.27), gross ~0, 29x turnover, 1.5%/yr cost.
* run 2 = GKM-faithful weekly shock (av_window=5, base_window=50): SR -0.52 (2004-20 -0.62), gross ~0, 51x turnover.
* Diagnostics: no high-minus-low premium among S&P 500 stocks 2000-2020 at any horizon up to 6 months. Both tails beat
  the middle deciles: extreme volume shocks get a 1-month pop (top decile +0.18%/month vs EW universe, t 3.2), and quiet
  stocks a slow, persistent premium (Merton neglect, ~+0.1%/month). A quintile long-short therefore nets to zero gross.
* v1 keeps the run-1 configuration: the a priori spec, and the less negative variant at half the turnover. It is NOT
  recommended for capital. It is the record of a clean negative test; see the notes for redesign ideas.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .base import Strategy
from .utils import period_ends


# ------------------------------------------------------------------ private helpers
def _rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    """trailing w-row sum per column (rows < w sum what exists); `a` must be NaN-free"""
    c = np.cumsum(a, axis=0)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def _rolling_beta(R: pd.DataFrame, m: pd.Series, window: int = 252, min_periods: int = 126) -> pd.DataFrame:
    """trailing OLS beta of each column of R on m over `window` rows, using only rows where both are observed.
    Same result as utils.rolling_beta (pairwise-complete moments) but vectorised with cumulative sums (~50x faster)."""
    X = R.to_numpy(dtype=float)
    mm = m.reindex(R.index).to_numpy(dtype=float)
    valid = ~np.isnan(X) & ~np.isnan(mm)[:, None]
    x = np.where(valid, X, 0.0)
    y = np.where(valid, np.nan_to_num(mm)[:, None], 0.0)
    n = _rolling_sum(valid.astype(float), window)
    sx, sy = _rolling_sum(x, window), _rolling_sum(y, window)
    sxy, syy = _rolling_sum(x * y, window), _rolling_sum(y * y, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        var = syy / n - (sy / n) ** 2
        b = cov / var
    b[n < min_periods] = np.nan
    return pd.DataFrame(b, index=R.index, columns=R.columns)


def _z(x: np.ndarray, clip: float) -> np.ndarray:
    s = x.std()
    return np.clip((x - x.mean()) / s, -clip, clip) if s > 0 else np.zeros_like(x)


def _pct_rank(x: np.ndarray) -> np.ndarray:
    """ranks scaled to (0, 1]; ties broken by order (continuous data, ties are rare)"""
    r = np.empty(x.size)
    r[np.argsort(x, kind="mergesort")] = np.arange(1, x.size + 1)
    return r / x.size


# ------------------------------------------------------------------ strategy
class VolumeAttention(Strategy):
    name = "s04_volume_attention"
    version = "v1"
    universe = "stocks"
    thesis = ("High-volume return premium (Gervais, Kaniel & Mingelgrin 2001): an unusual rise in a stock's trading "
              "activity relative to its own history raises its visibility (Merton 1987 investor recognition), widening "
              "its investor base and demand, so it outperforms over the following month; unusually quiet stocks lose "
              "visibility. Measured as abnormal volume neutralised against the same-window return, |return|, size, beta "
              "and 1-month volatility, so it is not momentum/reversal, size, or the low-volatility effect.")
    inspired_by = ("Volume Filters deck (volume as information about who trades) - redone as a cross-sectional, "
                   "monthly-horizon attention premium on the point-in-time S&P 500 instead of 1-day volume-filtered "
                   "momentum/reversal trades in 50 stocks without costs")

    def __init__(self, **params):
        super().__init__(**params)
        self._cache = {}
        self.diag = {}

    def default_params(self):
        return {"av_window": 21,          # recent window for abnormal volume (days)
                "base_window": 252,       # baseline window, immediately before the recent window
                "controls": ["ret", "absret", "logdv", "beta", "vol"],
                "vol_window": 21,         # realised-vol control (Ang et al. 2006 use 1 month)
                "beta_window": 252,
                "winsor": 3.0,            # z clip for controls and AV
                "freq": "W",              # formation on the last trading day of each week
                "n_tranches": 4,          # book = average of the last 4 weekly tranches (each held 4 weeks)
                "q": 0.2,                 # quintiles
                "min_names": 100}         # minimum eligible stocks with complete inputs to form a tranche

    # -------------------------------------------------------------- signal
    def signal(self, ds):
        """residual abnormal volume on formation dates -> (cal, tick, U, fdates (row positions), E, AV)"""
        p = self.params
        n, nb = int(p["av_window"]), int(p["base_window"])
        tick = ds.tickers("stocks")
        U = ds.stock_universe().reindex(columns=tick).fillna(False).astype(bool)
        cal = U.index
        V = ds.panel("volume", tick).reindex(cal)
        V = V.where(V > 0)                                         # zero prints = missing data
        C = ds.panel("close", tick).reindex(cal)
        R = ds.ret(tick).reindex(cal)

        rec = V.rolling(n, min_periods=max(2, int(np.ceil(0.7 * n)))).mean()
        base = V.shift(n).rolling(nb, min_periods=int(0.8 * nb)).mean()
        AV = np.log(rec / base)
        ctl = {}
        cset = set(p["controls"])
        if cset & {"ret", "absret"}:
            ctl["ret"] = np.log1p(R).rolling(n, min_periods=max(2, int(np.ceil(0.7 * n)))).sum()
        if "logdv" in cset:
            ctl["logdv"] = np.log((C * V).shift(n).rolling(nb, min_periods=int(0.8 * nb)).mean())
        if "beta" in cset:
            spy = ds.ret(["SPY"])["SPY"].reindex(cal)
            ctl["beta"] = _rolling_beta(R, spy, int(p["beta_window"]), int(p["beta_window"]) // 2)
        if "vol" in cset:
            vw = int(p["vol_window"])
            ctl["vol"] = R.rolling(vw, min_periods=int(0.7 * vw)).std()

        fd = period_ends(cal, p["freq"])
        rows = np.flatnonzero(cal.isin(fd))
        Uv, AVv = U.to_numpy(), AV.to_numpy()
        Cv = {k: v.to_numpy() for k, v in ctl.items()}
        T, N = AVv.shape
        E = np.full((T, N), np.nan)
        clip = float(p["winsor"])
        r2 = np.full(T, np.nan)
        for i in rows:
            ok = Uv[i] & np.isfinite(AVv[i])
            for v in Cv.values():
                ok &= np.isfinite(v[i])
            idx = np.flatnonzero(ok)
            if idx.size < int(p["min_names"]):
                continue
            y = AVv[i, idx]
            mu, sd = y.mean(), y.std()
            y = np.clip(y, mu - clip * sd, mu + clip * sd)
            cols = [np.ones(idx.size)]
            for c in p["controls"]:
                if c == "ret":
                    cols.append(_z(Cv["ret"][i, idx], clip))
                elif c == "absret":
                    cols.append(np.abs(_z(Cv["ret"][i, idx], clip)))
                else:
                    cols.append(_z(Cv[c][i, idx], clip))
            X = np.column_stack(cols)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            e = y - X @ coef
            E[i, idx] = e
            r2[i] = 1.0 - e.var() / y.var() if y.var() > 0 else np.nan
        self.diag["r2"] = pd.Series(r2, index=cal).dropna()
        return cal, tick, Uv, rows, E, AV

    # -------------------------------------------------------------- book
    def target_weights(self, ds) -> pd.DataFrame:
        key = (str(ds.dir), str(ds.end), json.dumps(self.params, sort_keys=True, default=str))
        if key in self._cache:
            return self._cache[key].copy()
        p = self.params
        cal, tick, Uv, rows, E, AV = self.signal(ds)
        T, N = E.shape
        q, ntr = float(p["q"]), int(p["n_tranches"])
        legs = []                                                  # (row, long idx, short idx) per formation date
        for i in rows:
            idx = np.flatnonzero(np.isfinite(E[i]))
            if idx.size == 0:
                legs.append((i, None, None))
                continue
            pr = _pct_rank(E[i, idx])
            legs.append((i, idx[pr > 1 - q], idx[pr <= q]))
        Wt = np.full((T, N), np.nan)
        started = False
        for k, (i, _, _) in enumerate(legs):
            w = np.zeros(N)
            m = 0
            for j in range(ntr):
                if k - j < 0:
                    break
                _, lo, sh = legs[k - j]
                if lo is None:
                    continue
                lo, sh = lo[Uv[i, lo]], sh[Uv[i, sh]]              # still eligible today
                if lo.size == 0 or sh.size == 0:
                    continue
                w[lo] += 1.0 / lo.size
                w[sh] -= 1.0 / sh.size
                m += 1
            if m == 0:
                if started:
                    Wt[i] = 0.0
                continue
            Wt[i] = w / m
            started = True
        out = pd.DataFrame(Wt, index=cal, columns=tick)
        self.diag.update({"E": pd.DataFrame(E, index=cal, columns=tick).iloc[rows], "AV": AV.iloc[rows],
                          "legs": [(cal[i], (None if lo is None else [tick[x] for x in lo]),
                                    (None if sh is None else [tick[x] for x in sh])) for i, lo, sh in legs]})
        self._cache = {key: out}
        return out.copy()


STRATEGY = VolumeAttention
