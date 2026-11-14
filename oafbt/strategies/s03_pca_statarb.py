"""s03 - Eigenportfolio residual statistical arbitrage on S&P 500 stocks (Avellaneda & Lee 2010).

Inspired by the RL Pairs Trading deck (relative value / mean reversion). That deck traded 4 hand-picked pairs with a
DDQN agent (no breadth, pairs chosen in-sample, overfitting). Here every stock is paired with its own factor-mimicking
portfolio built from PCA eigenportfolios, and the rule is a simple Ornstein-Uhlenbeck s-score with hysteresis bands,
applied to the whole point-in-time S&P 500 cross-section.

Pipeline (a weight row dated d uses only data dated <= d):
1. PCA on the first trading day of each week (a schedule known in real time - no future calendar): correlation
   matrix of the trailing 252 daily returns of the stocks eligible that day (standardised returns); top k=15
   eigenvectors v_j; eigenportfolio weights Q_ij = v_ij / sigma_i.
2. On each decision day d: eigenportfolio returns F_t = sum_i Q_ij R_it over the trailing 60 days (weights of the
   latest PCA <= d); each eligible stock's 60 daily returns are regressed on [1, F] -> residuals eps (sum to zero).
   Trading time (A&L sec. 6, applied at residual level): eps_t *= clip(avgVol_10d / Vol_t, 0.25, 4), then re-centred,
   so idiosyncratic moves on abnormally high volume (news) count less than moves on thin volume (liquidity).
3. X_k = cumsum(eps), k = 1..60; AR(1) X_{k+1} = a + b X_k + zeta -> kappa = -ln(b)*252, m = a/(1-b),
   sigma_eq = sqrt(var(zeta)/(1-b^2)). Centred s-score (A&L 2010): s = (X_60 - (m - <m>)) / sigma_eq, with
   <m> the cross-sectional mean of m over eligible stocks with a valid OU fit (0 < b < 1). X_60 = 0 by construction.
4. Hysteresis per stock, evaluated on the first trading day of each week: open long s < -1.25, open short
   s > +1.25 (A&L entry band) - only if kappa > 252/30 (mean-reversion time < 30 trading days, half-life < ~21 days);
   close at equilibrium: long when s > 0, short when s < 0. (A&L close earlier, at -0.50 / +0.75, and decide daily;
   here that turned over ~130x/yr and lost 6.5%/yr to 5 bps/side costs, while the entry signal's edge keeps accruing
   for weeks - see results/runs/s03_pca_statarb/v1/notes.md.) A position is also closed when the stock leaves the
   universe or its OU fit breaks down (b >= 1 or b <= 0).
5. Book: equal dollars per open position, long and short legs each scaled to gross 1 -> dollar neutral, constant
   gross 2. A target row is emitted only on decision days when the set of open positions changes (else drift).

Hedge choice: dollar neutral, NO SPY hedge (option hedge="spy" kept). Positions are picked on factor residuals and both
legs are diversified S&P 500 baskets (~74 names each), so the book's market beta is small: 2000-2020 mean +0.004,
mean |beta| 0.06, sd 0.08 (rolling 252d betas). Because the 2x stock leverage cap binds on GROSS, a SPY leg makes the
engine rescale the whole stock book whenever |hedge| moves: ~0.5%/yr of extra costs with daily rows (run 1 design),
~0.15%/yr with weekly rows, to remove ~1.5% of vol in quadrature from a ~5.4% vol book - roughly a wash at best.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .base import Strategy
from .utils import rolling_beta


# ------------------------------------------------------------------ private helpers
def _first_of_period(index: pd.DatetimeIndex, freq: str = "W") -> np.ndarray:
    """True on the first trading day of each period. Known on that day from past dates only (unlike period ENDS,
    which would need tomorrow's calendar and make the schedule depend on where the data stop)."""
    key = index.to_period(freq)
    m = np.ones(len(index), dtype=bool)
    if len(index) > 1:
        m[1:] = np.asarray(key[1:] != key[:-1])
    return m


def _rolling_count(valid: np.ndarray, w: int) -> np.ndarray:
    """trailing w-row count of True per column (rows < w use what exists)"""
    c = np.cumsum(valid.astype(np.int64), axis=0)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def _ar1(X: np.ndarray):
    """column-wise OLS X[k+1] = a + b X[k] + zeta  ->  a, b, var(zeta) (dof-corrected)"""
    xl, xd = X[:-1], X[1:]
    n = xl.shape[0]
    ml, md = xl.mean(0), xd.mean(0)
    dl, dd = xl - ml, xd - md
    vl = (dl * dl).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (dl * dd).sum(0) / vl
        a = md - b * ml
        z = dd - b * dl
        vz = (z * z).sum(0) / (n - 2)
    return a, b, vz


# ------------------------------------------------------------------ strategy
class PCAStatArb(Strategy):
    name = "s03_pca_statarb"
    version = "v2"
    universe = "stocks"
    thesis = ("Idiosyncratic (factor-residual) price moves of large US stocks partly reflect temporary liquidity "
              "demand and overreaction, so they mean-revert over days to weeks (Lehmann 1990; Lo & MacKinlay 1990; "
              "Avellaneda & Lee 2010; Blitz et al. 2013). Stripping the common component with 15 PCA eigenportfolios "
              "isolates the residual, an Ornstein-Uhlenbeck fit turns it into an s-score, and hysteresis bands harvest "
              "the reversion across the whole S&P 500 cross-section (breadth) instead of a few hand-picked pairs.")
    inspired_by = ("RL Pairs Trading deck (relative value / mean reversion) - redone with breadth: PCA-residual OU "
                   "s-scores on all eligible S&P 500 stocks instead of 4 hand-picked pairs + an RL agent")

    def __init__(self, **params):
        super().__init__(**params)
        self._cache = {}
        self.diag = {}

    def default_params(self):
        return {"pca_window": 252,      # days in the correlation matrix (A&L: 1 year)
                "n_factors": 15,        # eigenportfolios kept (A&L: 15)
                "pca_freq": "W",        # re-estimate eigenportfolios on the first trading day of each week
                "reg_window": 60,       # residual regression / OU estimation window (A&L: 60 days)
                "max_missing": 3,       # max missing returns in the 60d window (filled with 0)
                "tau_max": 30,          # trade only if mean-reversion time 1/kappa < 30 days (kappa > 252/30)
                "s_open": 2.0,          # v2: stronger entry band (v1 1.25: edge per round trip <= cost after 2008)
                "s_close_long": 0.0,    # exit at equilibrium (A&L: close long at s > -0.50, short at s < +0.75)
                "s_close_short": 0.0,
                "trade_freq": "W",      # decide on the first trading day of each week ("D" = daily, as A&L)
                "center": True,         # centred s-score (subtract cross-sectional mean of m)
                "hedge": "none",        # "none" = dollar neutral (see module doc); "spy" = hedge book beta with SPY
                "beta_window": 252,
                "min_names": 100,       # min eligible stocks to run the PCA
                "volume_time": True,    # A&L sec. 6 trading-time residuals (eps * avgVol10/Vol, clipped, re-centred)
                "vol_avg": 10}

    # -------------------------------------------------------------- signals
    def _signals(self, ds):
        p = self.params
        tick = ds.tickers("stocks")
        U = ds.stock_universe().reindex(columns=tick).fillna(False).astype(bool)
        R = ds.ret(tick).reindex(U.index)
        cal = R.index
        Rv = R.to_numpy(dtype=float)
        nanR = np.isnan(Rv)
        R0 = np.where(nanR, 0.0, Rv)
        Uv = U.to_numpy()
        T, N = Rv.shape
        L, W, k = int(p["pca_window"]), int(p["reg_window"]), int(p["n_factors"])
        cntW = _rolling_count(~nanR, W)
        cntL = _rolling_count(~nanR, L)

        ratio = None
        if p["volume_time"]:
            V = ds.panel("volume", tick).reindex(cal)
            avg = V.rolling(int(p["vol_avg"]), min_periods=max(2, int(p["vol_avg"]) // 2)).mean()
            ratio = (avg / V.where(V > 0)).clip(0.25, 4.0).fillna(1.0).to_numpy()

        nU = Uv.sum(1)
        live = (np.arange(T) >= max(L, W)) & (nU >= p["min_names"])
        pca_days = np.flatnonzero(_first_of_period(cal, p["pca_freq"]) & live)
        # decision days: every day ("D") or the first trading day of each week ("W"); s-scores only needed there
        sched = np.ones(T, bool) if p["trade_freq"] == "D" else _first_of_period(cal, p["trade_freq"])

        S = np.full((T, N), np.nan)
        Bm = np.full((T, N), np.nan)
        n_pca = np.full(T, np.nan)
        var_k = np.full(T, np.nan)
        for j, ip in enumerate(pca_days):
            nxt = pca_days[j + 1] if j + 1 < len(pca_days) else T
            el = np.flatnonzero(Uv[ip] & (cntL[ip] >= int(0.95 * L)))
            if el.size < max(int(p["min_names"]), 3 * k):
                continue
            Rw = Rv[ip - L + 1: ip + 1][:, el]
            mu = np.nanmean(Rw, 0)
            sd = np.nanstd(Rw, 0, ddof=1)
            keep = np.isfinite(sd) & (sd > 0)
            el, Rw, mu, sd = el[keep], Rw[:, keep], mu[keep], sd[keep]
            Y = np.nan_to_num((Rw - mu) / sd)
            # top-k eigenvectors of the correlation matrix Y'Y/(L-1) via the small L x L Gram matrix YY'
            # (same subspace as an SVD of Y, ~3x faster and independent of the number of stocks)
            G = Y @ Y.T
            lam, Ug = np.linalg.eigh(G)                        # ascending
            lam_k, U_k = lam[-k:][::-1], Ug[:, -k:][:, ::-1]
            V = (Y.T @ U_k) / np.sqrt(lam_k)                   # unit-norm eigenvectors (Np x k)
            Q = V / sd[:, None]                                # eigenportfolio weights Q_ij = v_ij / sigma_i
            f0 = ip - W + 1
            F = R0[f0:nxt][:, el] @ Q                          # eigenportfolio returns, rows f0..nxt-1
            n_pca[ip:nxt] = el.size
            var_k[ip:nxt] = float(lam_k.sum() / np.trace(G))
            Z = np.empty((W, k + 1))
            Z[:, 0] = 1.0
            for t in range(ip, nxt):
                if not sched[t]:
                    continue
                cols = np.flatnonzero(Uv[t] & (cntW[t] >= W - int(p["max_missing"])))
                if cols.size < 2 * k:
                    continue
                Z[:, 1:] = F[t - W + 1 - f0: t + 1 - f0]
                Qz, _ = np.linalg.qr(Z)
                Rt = R0[t - W + 1: t + 1][:, cols]
                E = Rt - Qz @ (Qz.T @ Rt)                      # OLS residuals on [1, F]
                if ratio is not None:                          # trading-time residuals, re-centred
                    E = E * ratio[t - W + 1: t + 1][:, cols]
                    E = E - E.mean(0)
                X = np.cumsum(E, 0)
                a, b, vz = _ar1(X)
                fit = np.isfinite(b) & (b > 0) & (b < 1) & np.isfinite(vz) & (vz > 0)
                if fit.sum() < 2 * k:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    m = a / (1.0 - b)
                    seq = np.sqrt(vz / (1.0 - b * b))
                if p["center"]:
                    m = m - m[fit].mean()
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = (X[-1] - m) / seq
                s[~fit] = np.nan
                S[t, cols] = s
                Bm[t, cols] = b
        self.diag.update({"n_pca": pd.Series(n_pca, index=cal), "var_k": pd.Series(var_k, index=cal)})
        return cal, tick, Uv, S, Bm

    # -------------------------------------------------------------- book
    def _states(self, S, Bm, Uv):
        """hysteresis state machine -> {t: state vector} only on days when the set of positions changes"""
        p = self.params
        T, N = S.shape
        bmax = np.exp(-1.0 / float(p["tau_max"]))
        so, scl, scs = float(p["s_open"]), float(p["s_close_long"]), float(p["s_close_short"])
        state = np.zeros(N, dtype=np.int8)
        prev = state.copy()
        rows = {}
        started = False
        n_valid = np.full(T, np.nan)
        n_pass = np.full(T, np.nan)
        with np.errstate(invalid="ignore"):
            for t in range(T):
                s = S[t]
                fin = np.isfinite(s)
                if not fin.any():
                    continue                                   # no signal at all today (warm-up): hold
                valid = fin & Uv[t]
                pas = valid & (Bm[t] < bmax)
                n_valid[t], n_pass[t] = valid.sum(), pas.sum()
                lo, sh = state == 1, state == -1
                close = (lo & (~valid | (s > -scl))) | (sh & (~valid | (s < scs)))
                state[close] = 0
                flat = state == 0
                state[flat & pas & (s < -so)] = 1
                state[flat & pas & (s > so)] = -1
                if not started or (state != prev).any():
                    rows[t] = state.copy()
                    prev = state.copy()
                    started = True
        self.diag.update({"n_valid": n_valid, "n_pass": n_pass})
        return rows

    def target_weights(self, ds) -> pd.DataFrame:
        key = (str(ds.dir), str(ds.end), json.dumps(self.params, sort_keys=True, default=str))
        if key in self._cache:
            return self._cache[key].copy()
        p = self.params
        cal, tick, Uv, S, Bm = self._signals(ds)
        rows = self._states(S, Bm, Uv)
        T, N = S.shape
        Wt = np.full((T, N), np.nan)
        for t, st in rows.items():
            w = np.zeros(N)
            lo, sh = st == 1, st == -1
            nl, ns = int(lo.sum()), int(sh.sum())
            if nl:
                w[lo] = 1.0 / nl
            if ns:
                w[sh] = -1.0 / ns
            Wt[t] = w
        out = pd.DataFrame(Wt, index=cal, columns=tick)
        if p["hedge"] == "spy":
            beta = rolling_beta(ds.xret(tick).reindex(cal), ds.xret(["SPY"])["SPY"].reindex(cal),
                                window=int(p["beta_window"]), min_periods=int(p["beta_window"]) // 2)
            beta = beta.fillna(1.0)
            book_beta = (out.fillna(0.0) * beta).sum(axis=1)
            out["SPY"] = (-book_beta).where(out[tick].notna().any(axis=1))
            self.diag["book_beta"] = book_beta.where(out[tick].notna().any(axis=1))
        self.diag["S"] = S
        self.diag["B"] = Bm
        self._cache = {key: out}
        return out.copy()


STRATEGY = PCAStatArb
