"""s01 - Gaussian-process Bayesian trend across asset classes (universe "multiasset").

Idea type: Bayesian Quadrature deck (GP machinery, robust ensembling over a trend parameter) + momentum. NEW:
  1. the Gaussian process is fitted to each asset's PRICE PATH (its vol-normalised daily increments), not to the
     strategy-parameter space. The trend is the GP posterior DRIFT over the coming holding week, scored by its own
     posterior uncertainty, z = E[drift | data] / sd[drift | data];
  2. trend horizons (RBF lengthscales of 1/3/6/12 months) and drift amplitudes are combined by BAYESIAN EVIDENCE -
     each model's marginal likelihood on the asset's own trailing year (Bayesian model averaging) - instead of by
     in-sample backtest Sharpe (the deck's weighting, which adds selection noise; OOS Sharpe ~0.1 on single futures);
  3. diversified across 35 ETFs in four risk groups instead of one future at a time.

Model, per asset and weekly decision date t (information <= close of t only)
  r_s  = log total-return-index change minus log(1 + T-bill)          (excess log return; ds.tri + ds.rf)
  e_s  = clip(r_s / sigma_{s-1}, +-5)    sigma = EWMA vol (halflife 21d) known at s-1;  s = t-251 .. t
  e_s  = mu(s) + eps_s,   eps_s ~ N(0, 1),   mu ~ GP(0, sm^2 exp(-(s - s')^2 / (2 l^2))),   sm = sr / sqrt(252)
Equivalently the vol-normalised log price path y = cumsum(e) is a GP whose kernel is an integrated-RBF trend plus a
Brownian (random-walk) noise term; the "slope" of its smooth part is mu. Models: lengthscale l in {21, 63, 126, 252}
days x drift amplitude sr in {0.25, 0.5, 1, 2, 4} (prior sd of the drift in annualised Sharpe units; log-uniform grid,
so the amplitude is integrated out rather than hand-set) = 20 models. On the fixed 252-day grid with fixed
hyper-parameters, the posterior mean of the average drift over the holding week (sessions t+2..t+6: one-day execution
lag, weekly hold) is a FIXED LINEAR FILTER m_k = h_k'e of the window, its posterior variance v_k is a constant, and the
log evidence is a fixed quadratic form  -0.5 e'(K_k + I)^-1 e - 0.5 log|K_k + I| + const.  All of it is precomputed
once (one eigen-decomposition of the 252x252 unit RBF matrix per lengthscale; amplitudes only rescale eigenvalues).
  BMA weights   w_k ∝ p(e | l_k, sr_k)^temper   (uniform prior on the grid; temper = 1 is plain BMA, 0 = uniform
                ensemble). The weight of lengthscale l is sum_sr w = p(e | l) with the amplitude marginalised.
  posterior     drift | e ~ sum_k w_k N(m_k, v_k)  ->  M = sum w m,  V = sum w (v + m^2) - M^2,  z = M / sqrt(V)
Position (target): clip(z_i / 2, -1, 1) / max(vol_i, 3%)  (risk parity on EWMA vol, halflife 42d; the floor stops
  1-3y Treasuries from absorbing the leverage budget), divided by the number of live assets in its risk group
  (equity / fixed income / commodity / FX: equal risk units per group, as in MOP 2012 / HOP 2017 - otherwise 16
  correlated equity ETFs carry 55% of the risk). Long and short.
Trading: decisions on the last NYSE session of each week (published exchange calendar); each week the book closes
  half of the gap to the target (partial adjustment, Garleanu & Pedersen 2013) - the run-1 version with full weekly
  rebalancing turned over 63x/yr and costs ate the gross edge. The engine scales the book to 10% ex-ante vol.

Why GP on the increments and not RBF + white noise on the price LEVELS: prices are close to random walks; a smooth
RBF fit leaves strongly autocorrelated random-walk residuals that a white-noise term cannot explain, so the evidence
always favours the shortest lengthscale and BMA collapses onto it (synthetic random walks: l=10 wins 100% of windows
by 40-280 nats). Putting the Brownian part into the kernel (= white noise on the increments) keeps the evidence
informative (differences of a few nats), so no ad-hoc tempering is needed. With a single fixed amplitude the
evidence barely separates the models (z correlates 0.995-0.997 with the uniform ensemble); integrating the amplitude
lets the evidence shrink weak/choppy trends and keep strong persistent ones (corr ~0.96 with uniform).
Dev result (2004-2020): SR 0.31; the uniform-weight ablation scored 0.36 (difference within noise) - the evidence
weighting itself adds no measurable value; see results/runs/s01_gp_trend/v1/notes.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .base import Strategy
from .utils import ewm_vol, next_session

ASSET_CLASSES = {
    "equity": ["SPY_X", "IWM_X", "QQQ", "EFA_X", "VGK_X", "EWJ", "EWG", "EWU", "EWC", "EWA",
               "EEM_X", "EWZ", "EWY", "EWT", "FXI", "VNQ_X"],
    "rates": ["TLT_X", "IEF_X", "TIP_X", "SHY"],
    "credit": ["LQD", "HYG"],
    "commodity": ["GLD_X", "DBC_X", "SLV", "USO", "DBA", "DBB"],
    "fx": ["UUP", "FXE", "FXY", "FXB", "FXA", "FXC", "FXF"],
}
ASSETS = [a for v in ASSET_CLASSES.values() for a in v]
CLASS_OF = {a: k for k, v in ASSET_CLASSES.items() for a in v}
# risk-budget groups for the optional class budget: the four classic trend-following classes (MOP 2012, HOP 2017)
BUDGET_OF = {a: ("fixed_income" if k in ("rates", "credit") else k) for a, k in CLASS_OF.items()}


# --------------------------------------------------------------------------------------------- GP algebra
def gp_drift_models(n: int, lengthscales, prior_srs, horizon=(2, 6)) -> list:
    """Fixed-grid GP drift models on n vol-normalised returns (unit noise variance), one per (lengthscale, amplitude).

    K = s2 * Kt with Kt the unit RBF matrix (eigen-decomposed once per lengthscale: Kt = U diag(lam) U'), so with
    Y = U'e (computed once per lengthscale and window) every model is cheap:
      posterior mean of the average drift over sessions t+horizon[0]..t+horizon[1]   m = Y @ g     (fixed filter)
      posterior variance of that drift                                              v              (constant)
      log evidence  = -0.5 * (Y**2) @ d - 0.5 * logdet - n/2 log(2 pi)       d = 1 / (s2 lam + 1)
    Returns a list of dicts (ell, sr, U, g, d, v, logdet) - U shared by models with the same lengthscale.
    """
    t = np.arange(n, dtype=float)
    tgt = (n - 1) + np.arange(horizon[0], horizon[1] + 1, dtype=float)
    out = []
    for ell in lengthscales:
        Kt = np.exp(-0.5 * (t[:, None] - t[None, :]) ** 2 / ell ** 2)
        lam, U = np.linalg.eigh(Kt)
        lam = np.clip(lam, 0.0, None)
        kb = np.exp(-0.5 * (tgt[:, None] - t[None, :]) ** 2 / ell ** 2).mean(axis=0)
        kff = np.exp(-0.5 * (tgt[:, None] - tgt[None, :]) ** 2 / ell ** 2).mean()
        ub = U.T @ kb
        for sr in prior_srs:
            s2 = sr ** 2 / 252.0
            d = 1.0 / (s2 * lam + 1.0)
            out.append({"ell": ell, "sr": sr, "U": U, "g": d * s2 * ub, "d": d,
                        "v": float(s2 * kff - s2 ** 2 * np.sum(ub ** 2 * d)),
                        "logdet": float(np.log(s2 * lam + 1.0).sum())})
    return out


# --------------------------------------------------------------------------------------------- schedule
def week_ends(cal: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading session of each (Mon-Sun) week. A session is a week end if the NEXT session is in a later week:
    for every date but the last this is the next data date; for the last data date it is the next session of the
    exchange's published calendar (no market data needed) - so a truncated history gives the same schedule."""
    if len(cal) == 0:
        return cal
    nxt_last = next_session(cal[-1])
    nxt = cal[1:].append(pd.DatetimeIndex([nxt_last if nxt_last is not None else cal[-1] + pd.Timedelta(days=7)]))
    wk = cal.to_period("W")
    return cal[np.asarray(pd.DatetimeIndex(nxt).to_period("W") != wk)]


# --------------------------------------------------------------------------------------------- strategy
class GPTrend(Strategy):
    name = "s01_gp_trend"
    version = "v1"
    universe = "multiasset"
    thesis = ("Asset-class trends persist (time-series momentum). A GP on each asset's vol-normalised price path gives "
              "the posterior drift and its uncertainty; Bayesian evidence (not in-sample Sharpe) weights trend horizons "
              "and drift amplitudes per asset and date, shrinking weak/choppy trends and keeping persistent ones.")
    inspired_by = "Bayesian Quadrature deck (GP machinery / robust ensembling) + momentum type"

    def default_params(self):
        return {
            "assets": ASSETS,
            "window": 252,                    # trailing trading days in each GP window (asset enters after ~1y)
            "lengthscales": [21, 63, 126, 252],   # RBF lengthscales: 1/3/6/12-month trends (MOP 2012, HOP 2017)
            "prior_sr": [0.25, 0.5, 1.0, 2.0, 4.0],   # grid (log-uniform prior) for the drift sd in annualised
                                              # Sharpe units, integrated out by BMA (sm = sr / sqrt(252))
            "horizon": [2, 6],                # forecast the mean drift over sessions t+2..t+6 (lag-1 execution, 1w hold)
            "temper": 1.0,                    # BMA weights ∝ evidence^temper (1 = Bayes, 0 = uniform ensemble)
            "norm_halflife": 21,              # EWMA vol used to normalise daily returns (lagged one day)
            "winsor": 5.0,                    # clip normalised returns at +-5 sd (Gaussian likelihood robustness)
            "min_frac": 0.95,                 # need >= 95% valid days in the window
            "size_halflife": 42,              # EWMA vol for risk-parity sizing (MOP 2012 / engine convention)
            "z_scale": 2.0,                   # position = clip(z / z_scale, -1, 1) / vol
            "vol_floor": 0.03,                # annualised floor on the sizing vol (SHY ~1-2% vol otherwise takes
                                              # ~25% of gross notional)
            "class_budget": True,             # divide by #active assets in its group (equity / fixed income /
                                              # commodity / FX) -> equal risk units per group
            "adjust": 0.5,                    # fraction of the gap to the target closed each week (1 = full rebalance)
        }

    # ---------------------------------------------------------------- data
    def _excess_log_returns(self, ds) -> pd.DataFrame:
        tri = ds.tri(self.params["assets"])
        rf = ds.rf.reindex(tri.index).fillna(0.0)
        lr = np.log(tri).diff()
        return lr.sub(np.log1p(rf), axis=0)

    # ---------------------------------------------------------------- signal
    def signals(self, ds) -> dict:
        """z-scores and diagnostics at the weekly decision dates (rows) x assets (columns)."""
        p = self.params
        n = int(p["window"])
        ells = list(p["lengthscales"])
        R = self._excess_log_returns(ds)
        cal = R.index
        sig = ewm_vol(R, halflife=p["norm_halflife"], min_periods=20, annualise=False)
        E = (R / sig.shift(1)).clip(-p["winsor"], p["winsor"])
        avail = ds.tri(p["assets"]).notna()
        srs = list(p["prior_sr"]) if isinstance(p["prior_sr"], (list, tuple)) else [p["prior_sr"]]
        models = gp_drift_models(n, ells, srs, tuple(p["horizon"]))
        nm = len(models)
        v = np.array([c["v"] for c in models])                               # (nm,) posterior variances
        logdet = np.array([c["logdet"] for c in models])
        log_prior = np.full(nm, -np.log(nm))                                 # uniform over lengthscale x amplitude
        groups = [[k for k, c in enumerate(models) if c["ell"] == ell] for ell in ells]
        Us = {ell: models[g[0]]["U"] for ell, g in zip(ells, groups)}

        reb = week_ends(cal)
        pos = cal.get_indexer(reb)
        shape = (len(reb), len(p["assets"]))
        Z, M, S, MAXW = (np.full(shape, np.nan) for _ in range(4))
        WL = np.full(shape + (len(ells),), np.nan)                           # BMA weight per lengthscale
        for j, a in enumerate(p["assets"]):
            e = E[a].to_numpy()
            ok = ~np.isnan(e)
            e0 = np.where(ok, e, 0.0)
            if len(e0) < n:
                continue
            cnt = np.convolve(ok.astype(float), np.ones(n))[:len(e0)]      # trailing count of valid days
            av = avail[a].to_numpy()
            sel = np.flatnonzero((pos >= n - 1) & (cnt[np.maximum(pos, 0)] >= p["min_frac"] * n - 1e-9)
                                 & av[np.maximum(pos, 0)])
            if sel.size == 0:
                continue
            win = sliding_window_view(e0, n)[pos[sel] - n + 1]               # (k, n) windows ending at decision dates
            m = np.empty((len(sel), nm))
            q = np.empty((len(sel), nm))
            for ell, g in zip(ells, groups):
                Y = win @ Us[ell]                                            # (k, n) eigen-coordinates
                Y2 = Y * Y
                for k in g:
                    m[:, k] = Y @ models[k]["g"]
                    q[:, k] = Y2 @ models[k]["d"]
            loge = -0.5 * q - 0.5 * logdet[None, :]
            a_ = p["temper"] * loge + log_prior[None, :]
            a_ -= a_.max(axis=1, keepdims=True)
            w = np.exp(a_)
            w /= w.sum(axis=1, keepdims=True)
            mix_m = (w * m).sum(axis=1)
            mix_v = (w * (v[None, :] + m ** 2)).sum(axis=1) - mix_m ** 2
            Z[sel, j] = mix_m / np.sqrt(mix_v)
            M[sel, j] = mix_m
            S[sel, j] = np.sqrt(mix_v)
            wl = np.column_stack([w[:, g].sum(axis=1) for g in groups])
            MAXW[sel, j] = wl.max(axis=1)
            WL[sel, j, :] = wl
        df = lambda X: pd.DataFrame(X, index=reb, columns=p["assets"])
        return {"z": df(Z), "mean": df(M), "sd": df(S), "max_w": df(MAXW), "w_by_ell": WL, "ells": ells,
                "dates": reb}

    # ---------------------------------------------------------------- weights
    def target_weights(self, ds) -> pd.DataFrame:
        p = self.params
        s = self.signals(ds)
        z = s["z"]
        R = self._excess_log_returns(ds)
        vol = ewm_vol(R, halflife=p["size_halflife"], min_periods=20).reindex(z.index)
        if p["vol_floor"]:
            vol = vol.clip(lower=p["vol_floor"])
        W = (z / p["z_scale"]).clip(-1.0, 1.0) / vol
        if p["class_budget"]:
            grp = pd.Series({a: BUDGET_OF[a] for a in W.columns})
            n_act = W.notna().T.groupby(grp).transform("sum").T          # active assets in the same budget group
            W = W / n_act.where(n_act > 0)
        W = W.where(np.isfinite(W))
        live = W.notna().any(axis=1)
        W = W[live].fillna(0.0)                         # rebalance rows: assets without a signal are flat
        k = float(p["adjust"])
        if k < 1.0:
            # partial adjustment toward the target (Garleanu & Pedersen 2013): close a fraction k of the gap at each
            # weekly decision; recursion over past decisions only, so it is causal
            A = W.to_numpy().copy()
            for i in range(1, len(A)):
                A[i] = A[i - 1] + k * (A[i] - A[i - 1])
            W = pd.DataFrame(A, index=W.index, columns=W.columns)
        out = pd.DataFrame(np.nan, index=R.index, columns=W.columns)
        out.loc[W.index] = W.to_numpy()
        return out


STRATEGY = GPTrend
