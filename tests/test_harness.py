"""The look-ahead harness must catch the leaky strategies the adversarial review built against the old harness."""
import numpy as np
import pandas as pd
import pytest

from oafbt.data.store import DataStore
from oafbt.engine.checks import lookahead_check
from oafbt.strategies.base import Strategy
from oafbt.strategies.utils import period_ends, keep_rows


class LeakyMonthStart(Strategy):
    name = "leak_month_start"
    universe = "multiasset"

    def target_weights(self, ds):
        X = ds.xret(["SPY", "TLT", "GLD", "EFA"]).loc["2005":]
        fut = X.rolling(10).sum().shift(-11)               # peeks at the next ~2 weeks
        W = np.sign(fut)
        first = X.index.to_series().groupby(X.index.to_period("M")).transform("min") == X.index.to_series()
        return W.where(first, np.nan).dropna(how="all")


class GoFlatBeforeBadDays(Strategy):
    name = "leak_flat"
    universe = "multiasset"

    def target_weights(self, ds):
        x = ds.xret(["SPY"]).loc["2005":]
        bad = x["SPY"].shift(-2) < -0.015
        W = pd.DataFrame(np.nan, index=x.index, columns=["SPY"])
        W.iloc[0] = 1.0
        W.loc[bad, "SPY"] = 0.0
        after = bad.shift(1, fill_value=False)
        W.loc[after & ~bad, "SPY"] = 1.0
        return W


class Honest(Strategy):
    name = "honest"
    universe = "multiasset"

    def target_weights(self, ds):
        X = ds.tri(["SPY", "TLT", "GLD", "EFA"]).loc["2004":]
        sig = np.sign(X / X.shift(126) - 1)
        return keep_rows(sig, period_ends(sig.index, "M"))


@pytest.mark.parametrize("cls", [LeakyMonthStart, GoFlatBeforeBadDays])
def test_harness_catches_leaks(cls):
    assert not lookahead_check(cls())["passed"]


def test_harness_passes_honest():
    assert lookahead_check(Honest())["passed"]
