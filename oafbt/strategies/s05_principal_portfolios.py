"""s05 - Principal portfolios (Kelly, Malamud & Pedersen 2023, Journal of Finance) on Ken French portfolios.

Inspired by the Macro Momentum deck TYPE (latent factors via PCA + ridge-style regularisation, walk-forward), but the
object is new: instead of compressing predictors, we decompose the PREDICTION MATRIX of a cross-section.

Setup (monthly). N assets with monthly returns R_{t+1} and signals S_t (cross-sectional ranks, unit gross). A linear
strategy holds w_t = L S_t and earns S_t' L' R_{t+1}; its expected return is <L, Pi>_F with the prediction matrix
Pi = E[R_{t+1} S_t'] (Pi_ij = how asset j's signal predicts asset i's return: own AND cross predictability).
  * symmetric part Pi_s = (Pi + Pi')/2 = sum_k lam_k v_k v_k'  ->  PEP_k: L = sign(lam_k) v_k v_k'      (E = |lam_k|)
    = timing of the eigen-portfolio v_k by its own signal (carries factor exposure: "principal EXPOSURE portfolios")
  * antisymmetric part Pi_a = (Pi - Pi')/2 = sum_k sig_k (y_k x_k' - x_k y_k')
        ->  PAP_k: L = y_k x_k' - x_k y_k'   (E = 2 sig_k): hold y_k sized by x_k's signal, short x_k sized by y_k's
    signal. Positions are orthogonal to the signal itself (w'S = S'L'S = 0), i.e. no bet in the signal direction:
    pure lead-lag "principal ALPHA portfolios" (beta-neutral when betas line up with the signal, KMP Prop. 6).
  The rank-K spectral-norm-optimal strategies keep the K largest |lam| (PEP) and the K largest sig (PAP).

Implementation (all walk-forward; a row dated T uses months <= T and days <= T only)
  * assets: KF value-weighted portfolios in BLOCKS (v1: 25 size x B/M | 12 industries), each block with its own
    prediction matrix; daily excess returns via utils.kf_portfolio_returns; monthly returns compounded from daily KF
    returns; only complete calendar months are used.
  * signal S_t: cross-sectional rank (utils.xs_rank, [-0.5, 0.5]) of the trailing 1-month return ("mom1", KMP
    baseline) or of the 12-1 month return ("mom12_1"), divided by its gross so sum|S_t| = 1.
  * Pi_T = mean over pairs (S_t, R~_{t+1}) with t >= est_start and t+1 <= T  (EXPANDING window, >= min_months pairs).
    R~ = cross-sectionally demeaned returns (Pi 1 = 0 = 1'Pi -> every principal portfolio is dollar-neutral, no market
    timing), divided by an EX-ANTE dispersion d_t (RMS of demeaned returns over the 12 months <= t) so that
    1999-2002 / 2008 do not dominate the estimate (WLS-style, predictability measured in units of cross-sect. risk).
  * shrinkage (Ledoit-Wolf style, intensity estimated from the data each month, no tuning): Pi <- (1-delta) Pi +
    delta * c * J, J = I - 11'/N, c = tr(Pi)/(N-1) (target: equal own-signal predictability = the 'simple factor'
    strategy; zero cross-predictability). delta = sum of sampling variances of Pi's entries / ||Pi - target||_F^2.
  * book at month end T: w_PEP = L_PEP S_T, w_PAP = L_PAP S_T (top-3 PEPs by |lam|, top-3 PAP pairs), each scaled
    to unit ex-ante vol (trailing 252d daily covariance, info <= T) and summed (equal risk); each block's book is
    scaled to unit vol and the blocks are summed (equal risk). Several signals -> same, one book per signal.
  * rows only on the last trading day of each month (utils.period_ends); the runner lags execution by one day,
    vol-targets to 10% and charges KF-portfolio costs (5 bps one-way, 30 bps/yr borrow, 10 bps/yr drag).

Development trials (all logged, dev data only; SR = net Sharpe 1975-2020 / 2004-2020):
  run1  KMP baseline: FF25, 1-month signal                      0.81 / -0.38  (gross SR ~2 pre-2005, ~0 after)
  run2  FF25, 12-1 momentum signal                              0.64 /  0.36
  run3  run2 + Pi projected on top return-PCs (Kaiser)          0.47 /  0.23  (rejected)
  run4  run2 + separate 12-industry block, equal risk  = v1     0.63 /  0.56
v1 = run4 (default_params). Notes: results/runs/s05_principal_portfolios/v1/notes.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .utils import kf_portfolio_returns, kf_rf, period_ends, xs_rank

TD = 252


# ------------------------------------------------------------------ private helpers
def _monthly_returns(P: pd.DataFrame) -> pd.DataFrame:
    """compound daily TOTAL returns into calendar-month returns; index = last trading day of the month.
    Only months whose last day is a true month end (utils.period_ends, published NYSE calendar) are kept."""
    per = P.index.to_period("M")
    m = np.expm1(np.log1p(P).groupby(per).sum(min_count=1))
    last = pd.Series(P.index, index=P.index).groupby(per).max()
    m.index = pd.DatetimeIndex(last.reindex(m.index).values)
    complete = period_ends(P.index, "M")
    return m.loc[m.index.isin(complete)]


def _signal(M: pd.DataFrame, kind: str) -> pd.DataFrame:
    """cross-sectional rank signal, normalised to unit gross; row t uses months <= t only"""
    if kind == "mom1":
        raw = M
    elif kind == "mom12_1":
        lm = np.log1p(M)
        raw = np.expm1(lm.shift(1).rolling(11, min_periods=11).sum())   # months t-11 .. t-1 (skip month t)
    else:
        raise ValueError(kind)
    S = xs_rank(raw)
    return S.div(S.abs().sum(axis=1).replace(0, np.nan), axis=0)


def _pep_pap(Pi: np.ndarray, k_pep: int, k_pap: int):
    """rank-K spectral-norm-optimal symmetric (PEP) and antisymmetric (PAP) position maps of a prediction matrix"""
    N = Pi.shape[0]
    Ps = 0.5 * (Pi + Pi.T)
    Pa = 0.5 * (Pi - Pi.T)
    lam, V = np.linalg.eigh(Ps)
    one = np.ones(N) / np.sqrt(N)
    tiny = 1e-10 * max(np.abs(lam).max(), 1e-300)
    ok = (np.abs(V.T @ one) < 0.5) & (np.abs(lam) > tiny)   # never select the null / equal-weight direction
    order = [i for i in np.argsort(-np.abs(lam)) if ok[i]][:k_pep]
    L_pep = np.zeros((N, N))
    for i in order:
        L_pep += np.sign(lam[i]) * np.outer(V[:, i], V[:, i])
    # antisymmetric: H = i*Pa is Hermitian; eigenvalue +sig with eigenvector z = (x + i y)/sqrt(2) gives
    # Pa x = sig y, Pa y = -sig x  =>  Pa = sum sig (y x' - x y')  and  y x' - x y' = 2 Im(z z^H)
    sig, Z = np.linalg.eigh(1j * Pa)
    top = np.argsort(-sig)[:k_pap]
    top = [i for i in top if sig[i] > 1e-10 * max(sig.max(), 1e-300)]
    Zk = Z[:, top]
    L_pap = 2.0 * np.imag(Zk @ Zk.conj().T)
    return L_pep, L_pap, lam[order], sig[top]


def _unit_vol(w: np.ndarray, C: np.ndarray) -> np.ndarray:
    v = float(np.sqrt(max(w @ C @ w, 0.0) * TD))
    return w / v if v > 1e-12 else np.zeros_like(w)


class PrincipalPortfolios(Strategy):
    name = "s05_principal_portfolios"
    version = "v1"
    universe = "kf"
    thesis = ("Style (size x B/M) and industry portfolios are predictable over 1-12 months by their own AND each "
              "other's past returns: style/industry momentum (Barberis-Shleifer 2003, Moskowitz-Grinblatt 1999) and "
              "slow cross-portfolio information diffusion (Lewellen 2002 cross-serial correlation, Menzly-Ozbas 2010). "
              "The eigen-structure of the prediction matrix E[R_{t+1} S_t'] gives the optimal linear strategies "
              "(Kelly-Malamud-Pedersen 2023): principal exposure portfolios (symmetric part, own/factor momentum) and "
              "principal alpha portfolios (antisymmetric part, lead-lag), estimated walk-forward with shrinkage.")
    inspired_by = "Macro Momentum deck (latent factors via PCA + ridge regularisation, walk-forward) -> new object: " \
                  "the prediction matrix of a cross-section (principal portfolios)"

    def default_params(self):
        # v1 = run 4: two blocks (FF25 size x B/M and 12 industries, one prediction matrix each), 12-1 momentum ranks
        return {"tables": [["25_Portfolios_5x5_Daily|VW"], ["12_Industry_Portfolios_daily|VW"]],
                "signals": ["mom12_1"], "est_start": "1963-07", "min_months": 120, "k_pep": 3, "k_pap": 3,
                "sleeves": ["pep", "pap"], "shrink": True, "pca": "none", "demean": True, "disp_norm": True,
                "disp_months": 12, "cov_days": 252}

    # ---------------------------------------------------------------- data
    def _blocks(self) -> list:
        """'tables' = list of table names (ONE joint universe / prediction matrix) or a list of lists (separate
        BLOCKS: one prediction matrix per block, the blocks' books combined at equal ex-ante risk)"""
        t = self.params["tables"]
        return [list(b) for b in t] if all(isinstance(b, (list, tuple)) for b in t) else [list(t)]

    def _rets(self, ds, tables=None) -> pd.DataFrame:
        """daily EXCESS returns of the KF portfolios (KF calendar)"""
        tables = tables or list(dict.fromkeys(t for b in self._blocks() for t in b))
        parts = [kf_portfolio_returns(ds, t) for t in tables]
        return pd.concat(parts, axis=1).sort_index()

    def asset_returns(self, ds, columns):
        return self._rets(ds).reindex(columns=columns)

    # ---------------------------------------------------------------- core
    def _block_books(self, ds, tables, details: bool = False):
        """one prediction matrix per signal for the assets of `tables`; returns {signal: unit-vol rows}, info"""
        p = self.params
        X = self._rets(ds, tables)
        rf = kf_rf(ds).reindex(X.index).fillna(0.0)
        P = X.add(rf, axis=0)                                       # daily total returns
        M = _monthly_returns(P)                                     # complete months only
        cols = list(M.columns)
        N = len(cols)
        Rd = M.sub(M.mean(axis=1), axis=0) if p["demean"] else M
        if p["disp_norm"]:
            disp = np.sqrt((Rd ** 2).mean(axis=1).rolling(p["disp_months"], min_periods=p["disp_months"] // 2).mean())
            U = Rd.div(disp.shift(1), axis=0)                       # R~_{t+1} / d_t  (d_t ex-ante)
        else:
            U = Rd
        start = pd.Period(p["est_start"], "M")
        mper = M.index.to_period("M")
        # daily covariance inputs
        Xa = X.reindex(columns=cols).to_numpy()
        dpos = {d: i for i, d in enumerate(X.index)}
        books, info = {}, []
        for kind in p["signals"]:
            S = _signal(M, kind)
            Sp = S.shift(1)                                          # S_t aligned with the return month t+1
            pair_ok = (U.notna().all(axis=1) & Sp.notna().all(axis=1) & (mper - 1 >= start)).to_numpy()
            Ua, Spa, Sa = U.to_numpy(), Sp.to_numpy(), S.to_numpy()
            A = np.zeros((N, N))
            Su, Suu = np.zeros(N), np.zeros((N, N))
            Q = 0.0
            n = 0
            rows = {}
            for j, T in enumerate(M.index):
                if pair_ok[j]:                                       # pair (S_{T-1}, R_T): complete at T's close
                    u, s = Ua[j], Spa[j]
                    A += np.outer(u, s)
                    Su += u
                    Suu += np.outer(u, u)
                    Q += float(u @ u) * float(s @ s)
                    n += 1
                if n < p["min_months"] or np.isnan(Sa[j]).any():
                    continue
                Pi = A / n
                delta, c = 0.0, np.trace(Pi) / max(N - 1, 1)
                if p["shrink"]:
                    J = np.eye(N) - np.ones((N, N)) / N if p["demean"] else np.eye(N)
                    tgt = c * J
                    noise = max(Q / n - float(np.sum(Pi * Pi)), 0.0) / n   # sum_ij Var(Pi_ij)
                    dist = float(np.sum((Pi - tgt) ** 2))
                    delta = float(np.clip(noise / dist, 0.0, 1.0)) if dist > 0 else 1.0
                    Pi = (1 - delta) * Pi + delta * tgt
                m_used = N
                if p["pca"] not in (None, "none"):
                    # keep only high-variance directions of the (same-sample) return covariance: Pi <- P Pi P
                    Cu = (Suu - np.outer(Su, Su) / n) / (n - 1)
                    ev, E = np.linalg.eigh(Cu)
                    if p["pca"] == "kaiser":                         # eigenvalue above the average one
                        keep = ev > np.sum(ev) / (N - 1 if p["demean"] else N)
                    else:
                        keep = np.zeros(N, bool)
                        keep[np.argsort(-ev)[:int(p["pca"])]] = True
                    Pm = E[:, keep] @ E[:, keep].T
                    Pi = Pm @ Pi @ Pm
                    m_used = int(keep.sum())
                L_pep, L_pap, lam, sig = _pep_pap(Pi, p["k_pep"], p["k_pap"])
                i = dpos[T]
                C = np.cov(np.nan_to_num(Xa[max(0, i - p["cov_days"] + 1):i + 1]), rowvar=False)
                s = Sa[j]
                sleeves = {"pep": _unit_vol(L_pep @ s, C), "pap": _unit_vol(L_pap @ s, C)}
                w = sum(sleeves[k] for k in p["sleeves"])
                rows[T] = _unit_vol(w, C)
                if details:
                    info.append({"date": T, "block": "+".join(tables), "signal": kind, "n_months": n, "delta": delta,
                                 "c_own": c, "m_pca": m_used, "lam": np.round(lam, 6).tolist(),
                                 "sig": np.round(sig, 6).tolist(), "ip_pep": float(np.sum(L_pep * Pi)),
                                 "ip_pap": float(np.sum(L_pap * Pi)), "cols": cols,
                                 "w_pep": sleeves["pep"], "w_pap": sleeves["pap"]})
            books[kind] = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
        return books, info

    def _build(self, ds, details: bool = False):
        books, info = [], []
        for tables in self._blocks():
            b, inf = self._block_books(ds, tables, details)
            books += list(b.values())
            info += inf
        cols = list(dict.fromkeys(c for b in books for c in b.columns))
        idx = sorted(set().union(*[b.index for b in books]))
        # unit-vol books summed (equal ex-ante risk); a row exists only where every block x signal has a book
        W = sum(b.reindex(columns=cols, fill_value=0.0).reindex(index=idx) for b in books)
        W = W.dropna(how="all")
        W.index = pd.DatetimeIndex(W.index)
        return (W, info) if details else W

    def target_weights(self, ds) -> pd.DataFrame:
        return self._build(ds)

    def details(self, ds):
        """diagnostics: (weights, list of per-decision dicts with delta, eigenvalues, sleeve weights)"""
        return self._build(ds, details=True)


STRATEGY = PrincipalPortfolios
