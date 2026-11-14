"""s10 - Turn-of-the-month flow effect (wildcard: flows/calendar).

Thesis: month-end/month-start cash flows (payroll and pension contributions, fund rebalancing, window dressing)
concentrate equity buying in the last trading day and first few trading days of each month (Ariel 1987,
Lakonishok & Smidt 1988, Etula et al. 2020 "Patient capital outperformance... month-end liquidity").
Be long SPY only from the close of day -2 to the close of day +3 (earning days -1, +1, +2, +3).
The trading calendar (holidays) is published in advance, so knowing which day is "-2" is not look-ahead.
"""
import numpy as np
import pandas as pd

from .base import Strategy


class TurnOfMonth(Strategy):
    name = "s10_turn_of_month"
    version = "v1"
    universe = "multiasset"
    thesis = "Month-turn cash flows push equity prices up on trading days -1..+3 around month end."
    inspired_by = "wildcard (flows)"

    def default_params(self):
        return {"asset": "SPY", "enter_day": -2, "exit_day": 3}

    def target_weights(self, ds):
        p = self.params
        cal = ds.calendar
        cal = cal[cal >= ds.xret([p["asset"]])[p["asset"]].first_valid_index()]
        s = pd.Series(np.arange(len(cal)), index=cal)
        per = cal.to_period("M")
        # trading-day position relative to month end: last day of month = -1, first day of next month = +1
        pos_from_end = s.groupby(per).transform(lambda x: x - x.max() - 1)      # -1 = last day
        pos_from_start = s.groupby(per).transform(lambda x: x - x.min() + 1)    # +1 = first day
        hold = (pos_from_end >= p["enter_day"] + 1) | (pos_from_start <= p["exit_day"])
        # holdings must be in place at the close BEFORE the earning day. We return weights by DECISION date;
        # the runner executes with lag=1, so the decision for "hold over day t" is made at close t-2.
        held_over = hold.astype(float)                      # 1 if we want exposure to day t's return
        decide = held_over.shift(-2)                       # decided 2 closes before (calendar is known ahead)
        W = pd.DataFrame({p["asset"]: decide})
        chg = W[p["asset"]].diff().fillna(1.0) != 0
        W.loc[~chg] = np.nan                               # only emit rows when the target changes
        return W.iloc[:-2] if len(W) > 2 else W            # last two rows would need the future calendar


STRATEGY = TurnOfMonth
