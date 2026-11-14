"""s11 - Cross-asset carry, measured from market prices only (replaces s06; macro/relative-value TYPE, NEW premium).

Thesis (Koijen, Moskowitz, Pedersen & Vrugt 2018 'Carry'): an asset's carry - the return it earns if prices do not
move - predicts its return in every asset class (compensation for crash/liquidity risk, slow-moving capital).
None of the decks used carry. We infer carry WITHOUT macro databases, from the price data alone:
  * commodities: realised roll yield = trailing mean(ETF excess return - Yahoo continuous front-month return).
    (Yahoo's continuous futures omit roll yield - a data defect elsewhere - so the gap IS the roll yield.)
  * currencies: foreign deposit rate = trailing mean(currency-trust ETF return - spot FX return) + fees; carry =
    foreign rate - USD rate.
  * equities (country ETFs, unhedged): trailing 12m dividend yield from jumps in Yahoo's adjclose/close ratio,
    minus the USD bill rate.
v1: monthly; within each class weights = demeaned cross-sectional carry rank / asset vol (dollar-neutral per class);
classes combined with equal risk; engine scales the book to 10% ex-ante vol. Bonds excluded in v1 (an ETF curve
book is friction-bound, see retired s06).
"""
import numpy as np
import pandas as pd

from .base import Strategy
from .utils import period_ends, keep_rows, ewm_vol

COMMOD = {"USO": "CL=F", "UNG": "NG=F", "GLD": "GC=F", "SLV": "SI=F", "CPER": "HG=F", "UGA": "RB=F",
          "PPLT": "PL=F", "PALL": "PA=F"}
FXETF = {"FXE": "EURUSD=X", "FXY": "JPY=X", "FXB": "GBPUSD=X", "FXA": "AUDUSD=X", "FXC": "CAD=X", "FXF": "CHF=X"}
EQ = ["SPY", "EWJ", "EWG", "EWU", "EWC", "EWA", "EWH", "EWS", "EWT", "EWY", "EWZ", "EWW", "EWQ", "EWI", "EWP", "EWL",
      "EWD", "EWN", "FXI", "EZA", "EEM", "EWK", "EWO"]


class CrossAssetCarry(Strategy):
    name = "s11_cross_asset_carry"
    version = "v3"
    universe = "multiasset"
    thesis = "Carry predicts returns across asset classes; carry is inferred from ETF vs futures/spot/dividend data."
    inspired_by = "Macro Momentum deck (cross-asset macro type) - new premium: carry"

    def default_params(self):
        # v2: dev diagnostics (all logged): commodity roll-yield carry SR 0.36, G10 FX carry -0.44 (post-2008 carry
        # failure), equity dividend-yield carry -0.12. v2 keeps commodities only (an in-sample class selection -
        # counted in the ledger) and uses a 1-year carry window to reduce noise in the realised roll-yield estimate.
        return {"window": 252, "classes": ["commod"], "min_assets": 3, "vol_halflife": 63}

    def carry(self, ds):
        p = self.params
        w = p["window"]
        out = {}
        if "commod" in p["classes"]:
            etf = ds.xret(list(COMMOD))
            fut = ds.ret(list(COMMOD.values()))
            fut.columns = list(COMMOD)
            gap = (etf - fut)
            out["commod"] = gap.rolling(w, min_periods=int(0.8 * w)).mean() * 252
        if "fx" in p["classes"]:
            etf = ds.ret(list(FXETF))
            spot = ds.ret(list(FXETF.values()))
            spot.columns = list(FXETF)
            # cumulative gap over the window is robust to the 1-day staleness of Yahoo FX
            gap = (np.log1p(etf) - np.log1p(spot)).rolling(w, min_periods=int(0.8 * w)).sum() / w * 252
            out["fx"] = gap.sub(ds.rf * 252, axis=0)
        if "equity" in p["classes"]:
            adj, close = ds.px(EQ), ds.panel("close", EQ)
            ratio = adj / close
            j = (ratio / ratio.shift(1) - 1).clip(lower=0)      # dividend jumps at ex-dates
            j = j.where(j > 1e-4, 0.0)
            dy = j.rolling(252, min_periods=200).sum()
            out["equity"] = dy.sub(ds.rf * 252, axis=0)
        return out

    def target_weights(self, ds):
        p = self.params
        C = self.carry(ds)
        cols = sum([list(c.columns) for c in C.values()], [])
        X = ds.xret(cols)
        vol = ewm_vol(X, halflife=p["vol_halflife"])
        W = []
        for k, c in C.items():
            c = c.where(X[c.columns].notna() & vol[c.columns].notna())
            n = c.notna().sum(axis=1)
            r = c.rank(axis=1)
            r = r.sub(r.mean(axis=1), axis=0)
            w = (r / vol[c.columns])
            # v3 (final-review fix): make each class book DOLLAR-NEUTRAL after vol scaling (v1/v2 demeaned ranks
            # before dividing by vol, leaving a +17% average net-long tilt towards low-vol, high-carry gold/copper)
            lg, sh = w.clip(lower=0), w.clip(upper=0)
            lg = lg.div(lg.sum(axis=1).replace(0, np.nan), axis=0)
            sh = sh.div(-sh.sum(axis=1).replace(0, np.nan), axis=0)
            w = (lg.fillna(0.0) + sh.fillna(0.0)) / 2.0                         # unit gross, net zero
            # equal RISK per class: divide by the class book's own trailing vol
            book = (w.shift(1) * X[c.columns]).sum(axis=1)
            bvol = book.rolling(252, min_periods=63).std() * np.sqrt(252)
            w = w.div(bvol.replace(0, np.nan), axis=0).where(n >= p["min_assets"])
            W.append(w)
        W = pd.concat(W, axis=1).fillna(0.0)
        W = W[(W != 0).any(axis=1)]
        return keep_rows(W, period_ends(W.index, "M"))


STRATEGY = CrossAssetCarry
