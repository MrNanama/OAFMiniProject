"""s16 - Treasury auction cycle (supply-driven price pressure; flows/calendar TYPE, NEW signal).

Thesis (Lou, Yan & Zhang 2013, RFS 'Anticipated and Repeated Shocks in Liquid Markets'): ahead of large, scheduled
Treasury auctions, risk-averse dealers and investors make room for the new supply, so prices of existing long
bonds decline into the auction and recover after it (the supply shock is anticipated but liquidity is limited).
Dev event study (logged, 2003-2020, TLT excess returns): 10/20/30-year auctions: days -6..0 about -34 bps cumulative,
day +1 +4.3 bps, day +2 +13.3 bps (t=2.7). 5/7-year auctions are excluded: they sit in month-end week and would
double-count s10's month-end index-extension effect.

v1: for every 10-, 20- and 30-year auction (new issues and reopenings), point-in-time: an auction is used only once
it has been ANNOUNCED (ds.treasury_auctions() filters on announcement date; the decision for any day's exposure is
taken >= 1 session after the announcement). Short TLT over the pre-auction days (at most days -5..0), long TLT over
days +1..+2; overlapping events are netted. Inverse-vol sized; flat otherwise.
"""
import numpy as np
import pandas as pd

from .base import Strategy
from .utils import ewm_vol, trading_calendar


class TreasuryAuctionCycle(Strategy):
    name = "s16_treasury_auctions"
    version = "v1"
    universe = "multiasset"
    calendar_timed = True     # trades scheduled from published announcements, executed at the close
    thesis = "Anticipated auction supply depresses long Treasuries into the auction; prices recover after it."
    inspired_by = "flows/supply type (calendar), Macro deck's rates framing"

    def default_params(self):
        return {"asset": "TLT", "terms": ["10-Year", "20-Year", "30-Year"], "pre": 5, "post": [1, 2]}

    def target_weights(self, ds):
        p = self.params
        a = p["asset"]
        X = ds.xret([a])
        cal = ds.calendar
        full = trading_calendar(ds)
        au = ds.treasury_auctions()
        au = au[au.orig_term.isin(p["terms"])]
        expo = pd.Series(0.0, index=full)                         # exposure wanted over each day's return
        ann_pos = full.searchsorted(au.announce.values)            # first session on/after the announcement
        auc_pos = full.searchsorted(au.auction.values)
        for ia, iq in zip(ann_pos, auc_pos):
            if iq >= len(full) or iq <= 0:
                continue
            # decision for exposure on day t is made at close t-2 (lag 1) and must be >= the announcement session
            first_pre = max(iq - p["pre"], ia + 2)
            for t in range(first_pre, iq + 1):
                if 0 <= t < len(full):
                    expo.iloc[t] -= 1.0
            for k in p["post"]:
                if iq + k < len(full):
                    expo.iloc[iq + k] += 1.0
        expo = np.sign(expo)
        dec = expo.shift(-2, fill_value=0.0).reindex(cal).fillna(0.0)    # decide two sessions ahead
        vol = ewm_vol(X, halflife=63)[a]
        W = pd.DataFrame({a: dec / vol}, index=cal).fillna(0.0)
        W = W.loc[X[a].first_valid_index():]
        chg = W[a].diff().fillna(1.0) != 0
        return W.where(chg, np.nan)


STRATEGY = TreasuryAuctionCycle
