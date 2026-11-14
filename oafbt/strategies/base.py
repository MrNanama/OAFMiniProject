"""Strategy interface.

A strategy is a class with
  name        unique id, e.g. "s03_pca_statarb" (also the module name)
  version     bumped every time the logic or parameters change ("v1", "v2", ...)
  universe    "multiasset" | "stocks" | "kf" | "crypto"  (selects eval window + leverage cap)
  params      dict of parameters (logged to the trials ledger)
  thesis      one paragraph: the economic reason this should work
  inspired_by which pitch-deck idea TYPE inspired it (the idea itself must be new)

and a method
  target_weights(ds) -> DataFrame (dates x assets) of target weights decided at each date's close, using
                        ONLY information available at that close (rows all-NaN = hold / let positions drift).

Returns for the assets are taken from `asset_returns(ds)` (default: ds.xret for the columns of the weights);
override it for synthetic assets (e.g. Ken French portfolios). The runner does sizing (vol target), costs,
execution lag and metrics, so strategies should output *relative* weights (any scale).
"""
from __future__ import annotations

import pandas as pd


class Strategy:
    name: str = "base"
    version: str = "v0"
    universe: str = "multiasset"
    thesis: str = ""
    inspired_by: str = ""
    lag: int = 1
    vol_target: float | None = 0.10
    lev_cap: float | None = None          # None -> config.LEV_CAP[universe]
    vol_window: int = 126
    vol_halflife: float = 42

    def __init__(self, **params):
        self.params = {**self.default_params(), **params}

    def default_params(self) -> dict:
        return {}

    def target_weights(self, ds) -> pd.DataFrame:
        raise NotImplementedError

    def asset_returns(self, ds, columns) -> pd.DataFrame:
        """daily EXCESS returns for the traded assets, on the strategy's calendar"""
        return ds.xret(columns)

    def asset_meta(self, ds, columns) -> pd.DataFrame:
        """cost_bps, borrow_bps, funded per asset"""
        return ds.meta(columns)
