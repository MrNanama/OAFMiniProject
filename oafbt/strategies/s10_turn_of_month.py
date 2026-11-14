"""s10 - Month-turn flow effects in bonds and equities (wildcard: flows/calendar).

v1: long SPY over trading days -1..+3 around month end (classic turn-of-month). Dev SR 0.23.
v3: + equity exposure over scheduled FOMC announcement days (pre-announcement drift, Lucca & Moench 2015; dev
    event study: SPY +21 bps t=2.1 in 1994-2007 and +35 bps t=2.8 in 2008-2020; schedule published a year ahead).
v2: two flow-driven sleeves that rotate at the month turn:
  * Treasuries, days -2 and -1: bond indices extend duration at month end, so index-tracking money must buy
    longer bonds on the last days of the month ("index extension" demand). Dev event study 2004-2020 (raw price
    AND adjusted returns, so not a dividend artefact): last-day mean TLT +22 bps (t=3.9), IEF +11 bps (t=4.3),
    AGG t=6.1, SHY t=6.1; day -2 also positive.
  * Equities, days +1..+3: month-start inflows (payroll, pension contributions, rebalancing) - the turn-of-month
    effect; in 2004-2020 the equity gain sits on day +1 (+16 bps), not day -1 (-4 bps).
  Each sleeve is inverse-vol weighted; the engine scales the active book to 10% ex-ante vol.
The trading calendar (holidays) is published in advance, so locating day -k / +k is not look-ahead.
Research trials: the event study that picked the windows is counted in the ledger notes (3 exploratory looks).
"""
import numpy as np
import pandas as pd

from .base import Strategy
from .utils import ewm_vol, trading_calendar


def month_positions(cal: pd.DatetimeIndex):
    s = pd.Series(np.arange(len(cal)), index=cal)
    per = cal.to_period("M")
    from_end = s.groupby(per).transform(lambda x: x - x.max() - 1)      # -1 = last trading day of month
    from_start = s.groupby(per).transform(lambda x: x - x.min() + 1)    # +1 = first trading day of month
    return from_end, from_start


class MonthTurnFlows(Strategy):
    name = "s10_turn_of_month"
    version = "v3"
    universe = "multiasset"
    calendar_timed = True     # trades scheduled in advance (MOC orders): lag+1 test not meaningful
    thesis = "Month-end bond index extension and month-start equity inflows create predictable calendar returns."
    inspired_by = "wildcard (flows)"

    def default_params(self):
        return {"bond_assets": ["TLT", "IEF"], "bond_days_from_end": [-2, -1],
                "equity_assets": ["SPY"], "equity_days_from_start": [1, 2, 3], "vol_halflife": 63,
                "fomc_day": True}

    def target_weights(self, ds):
        p = self.params
        assets = p["bond_assets"] + p["equity_assets"]
        X = ds.xret(assets)
        cal = ds.calendar
        full = trading_calendar(ds)                      # published NYSE sessions incl. the next few weeks
        fe, fs = month_positions(full)
        bond_on = fe.isin(p["bond_days_from_end"])       # want exposure to THESE days' returns
        eq_on = fs.isin(p["equity_days_from_start"])
        if p.get("fomc_day"):                             # scheduled FOMC announcement day (Lucca & Moench 2015)
            eq_on = eq_on | pd.Series(full.isin(ds.fomc_dates()), index=full)
        # exposure over day t's return is held from the close of t-1, i.e. executed at close t-1, i.e. DECIDED
        # at close t-2 (engine lag = 1). Shift the flags back by 2 sessions on the published calendar.
        bond_dec = bond_on.shift(-2, fill_value=False).reindex(cal)
        eq_dec = eq_on.shift(-2, fill_value=False).reindex(cal)
        vol = ewm_vol(X, halflife=p["vol_halflife"])     # known at the decision date
        nb, ne = len(p["bond_assets"]), len(p["equity_assets"])
        W = pd.DataFrame(0.0, index=cal, columns=assets)
        for a in p["bond_assets"]:
            W[a] = np.where(bond_dec, (1.0 / nb) / vol[a], 0.0)
        for a in p["equity_assets"]:
            W[a] = np.where(eq_dec, (1.0 / ne) / vol[a], 0.0)
        W = W.where(vol.notna(), 0.0)
        first = X.apply(lambda s: s.first_valid_index()).max()
        W = W.loc[W.index >= first]
        chg = W.ne(0).astype(int).diff().abs().sum(axis=1).fillna(1.0) > 0
        return W.where(chg, np.nan)                      # emit rows only when the set of positions changes


STRATEGY = MonthTurnFlows
