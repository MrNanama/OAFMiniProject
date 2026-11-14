import numpy as np
import pandas as pd
import pytest

from oafbt.engine.backtest import backtest, exante_vol
from oafbt.engine import metrics as M


def _cal(n=600):
    return pd.bdate_range("2010-01-01", periods=n)


def _rets(n=600, k=3, seed=0, vol=0.01):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0003, vol, (n, k)), index=_cal(n), columns=[f"A{i}" for i in range(k)])


def test_constant_weight_lag0_equals_asset():
    X = _rets(k=1)
    W = pd.DataFrame(1.0, index=X.index, columns=X.columns)
    r = backtest(W, X, lag=0, vol_target=None, drift=True)
    # holdings set at close of day 0 earn returns from day 1
    assert r.net.iloc[0] == 0.0
    np.testing.assert_allclose(r.net.iloc[1:].values, X["A0"].iloc[1:].values, atol=1e-12)


def test_lag_shifts_exposure():
    X = _rets(k=1)
    W = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    W.iloc[10] = 1.0
    r0 = backtest(W, X, lag=0, vol_target=None)
    r1 = backtest(W, X, lag=1, vol_target=None)
    assert (r0.net.iloc[:11] == 0).all() and r0.net.iloc[11] != 0
    assert (r1.net.iloc[:12] == 0).all() and r1.net.iloc[12] != 0


def test_costs_charged_on_trades():
    X = _rets(k=1) * 0.0                   # zero returns -> no drift, pnl = -costs
    W = pd.DataFrame(np.tile([[1.0], [0.0]], (300, 1)), index=X.index, columns=X.columns)
    r = backtest(W, X, cost_bps=10.0, lag=0, vol_target=None)
    assert np.isclose(r.turnover.sum(), 600.0)
    assert np.isclose(r.net.sum(), -600 * 10 / 1e4)


def test_buy_and_hold_drift():
    X = _rets(k=2, seed=3)
    W = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    W.iloc[0] = [0.5, 0.5]
    r = backtest(W, X, lag=0, vol_target=None)
    wealth = (1 + r.net).prod()
    expect = 0.5 * (1 + X["A0"].iloc[1:]).prod() + 0.5 * (1 + X["A1"].iloc[1:]).prod()
    assert np.isclose(wealth, expect, rtol=1e-10)
    assert np.isclose(r.turnover.sum(), 1.0)   # only the initial trade


def test_engine_no_lookahead_with_vol_target():
    X = _rets(k=3, seed=5)
    sig = np.sign(X.rolling(20).sum())
    W = sig.copy()
    W.loc[~(sig.index.day_of_week == 4)] = np.nan    # weekly rebalance, info <= t
    k = 400
    X2 = X.copy()
    X2.iloc[k + 1:] = np.random.default_rng(9).normal(0, 0.05, X2.iloc[k + 1:].shape)
    a = backtest(W, X, lag=1, vol_target=0.1, cost_bps=5)
    b = backtest(W, X2, lag=1, vol_target=0.1, cost_bps=5)
    np.testing.assert_allclose(a.net.iloc[:k + 1].values, b.net.iloc[:k + 1].values, atol=1e-14)
    np.testing.assert_allclose(a.scale.iloc[:k + 1].fillna(-1).values, b.scale.iloc[:k + 1].fillna(-1).values)


def test_vol_target_hits_target():
    X = _rets(n=3000, k=4, seed=11, vol=0.012)
    W = pd.DataFrame(0.25, index=X.index, columns=X.columns)
    r = backtest(W, X, lag=1, vol_target=0.10, lev_cap=10)
    realised = r.net.iloc[300:].std() * np.sqrt(252)
    assert abs(realised - 0.10) < 0.015


def test_leverage_cap():
    X = _rets(n=800, k=2, seed=2, vol=0.0005)   # tiny vol -> vol target would want huge leverage
    W = pd.DataFrame(1.0, index=X.index, columns=X.columns)
    r = backtest(W, X, lag=1, vol_target=0.10, lev_cap=3.0)
    assert r.gross_lev.max() <= 3.0 + 1e-6 * 3 + 0.05   # drift between rebalances allowed slightly


def test_delisting_liquidates():
    X = _rets(k=2, seed=4)
    X.iloc[300:, 1] = np.nan                   # asset 1 stops trading
    W = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    W.iloc[0] = [0.5, 0.5]
    r = backtest(W, X, lag=0, vol_target=None, keep_held=True)
    assert r.held.iloc[305, 1] == 0.0
    assert r.held.iloc[299, 1] != 0.0


def test_borrow_fee():
    X = _rets(k=1) * 0.0
    W = pd.DataFrame(-1.0, index=X.index, columns=X.columns)
    r = backtest(W, X, lag=0, vol_target=None, borrow_bps_yr=100.0)
    assert np.isclose(r.fees.iloc[5], 0.01 / 252)


def test_untradable_asset_not_entered():
    X = _rets(k=2, seed=7)
    X.iloc[:100, 1] = np.nan                   # asset 1 lists on day 100
    W = pd.DataFrame(0.5, index=X.index, columns=X.columns)
    r = backtest(W, X, lag=0, vol_target=None, keep_held=True)
    assert (r.held.iloc[:100, 1] == 0).all()
    assert r.held.iloc[100, 1] == 0.5


def test_metrics_basic():
    x = pd.Series(np.random.default_rng(0).normal(0.0004, 0.01, 2520), index=pd.bdate_range("2000-01-01", periods=2520))
    s = M.perf_stats(x)
    assert abs(s["sharpe"] - M.sharpe(x)) < 1e-12
    assert s["max_dd"] < 0
    lo, hi = M.boot_sharpe_ci(x)
    assert lo < s["sharpe"] < hi
    assert 0 <= s["psr0"] <= 1 and 0 <= s["dsr"] <= 1
    assert M.dsr(x, 100) < M.psr(x, 0)
