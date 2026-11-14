"""s12 - Crypto time-series momentum (candidate replacement; momentum TYPE on a new asset class).

Thesis: crypto assets have strong return autocorrelation at 1-12 month horizons (Liu & Tsyvinski 2021 'Risks and
Returns of Cryptocurrency'; Liu, Tsyvinski & Wu 2022): investor attention and slow adoption make trends persist.
v1: BTC and ETH (Yahoo; daily close at 00:00 UTC, delayed one session in the data build so it is never ahead of
the NYSE close). Long-only (shorting crypto was costly/unavailable for most of the sample): weight = average over
lookbacks {21, 63, 126, 252} days of 1[trailing return > 0]; each asset sized inverse to its vol; weekly rebalance.
"""
import numpy as np
import pandas as pd

from .base import Strategy
from .utils import period_ends, keep_rows, ewm_vol


class CryptoTrend(Strategy):
    name = "s12_crypto_trend"
    version = "v1"
    universe = "crypto"
    thesis = "Crypto returns trend at 1-12 month horizons (attention, slow adoption)."
    inspired_by = "momentum type (BQ/CORPS decks), new asset class"

    def default_params(self):
        return {"assets": ["BTC-USD", "ETH-USD"], "lookbacks": [21, 63, 126, 252], "min_hist": 252}

    def target_weights(self, ds):
        p = self.params
        tri = ds.tri(p["assets"])
        X = ds.xret(p["assets"])
        vol = ewm_vol(X, halflife=42)
        sig = sum((tri / tri.shift(L) - 1 > 0).astype(float).where(tri.shift(L).notna()) for L in p["lookbacks"])
        sig = sig / len(p["lookbacks"])
        hist = X.notna().cumsum()
        W = (sig / vol).where(hist >= p["min_hist"])
        W = W.dropna(how="all").fillna(0.0)
        return keep_rows(W, period_ends(W.index, "W"))


STRATEGY = CryptoTrend
