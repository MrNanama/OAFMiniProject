"""Parameter-plateau check: perturb key parameters of each strategy around the chosen values and report the Sharpe.
Runs are logged with tag 'sens' (reported, NOT used to pick parameters).  python scripts/sensitivity.py"""
import json
import sys

import pandas as pd

from oafbt import config as C
from oafbt.data.store import DataStore
from oafbt.engine import runner
from oafbt.engine.metrics import perf_stats
from oafbt.strategies.registry import load

GRID = {
    "s12_crypto_trend": [{"lookbacks": [63, 126, 252]}, {"lookbacks": [21, 63]}, {"lookbacks": [126, 252]},
                         {"lookbacks": [42, 84, 168, 336]}],
    "s10_turn_of_month": [{"equity_days_from_start": [1, 2]}, {"equity_days_from_start": [1, 2, 3, 4]},
                          {"bond_days_from_end": [-1]}, {"bond_days_from_end": [-3, -2, -1]}, {"fomc_day": False}],
    "s09_vrp_timing": [{"ratio_max": 0.90}, {"ratio_max": 1.00}, {"rv_window": 10}, {"rv_window": 42}],
    "s02_factor_momentum": [{"lookbacks_m": [6, 12]}, {"lookbacks_m": [1, 3, 6, 12]}, {"lookbacks_m": [12]},
                            {"vol_halflife": 126}],
    "s05_principal_portfolios": [{"k_pep": 2, "k_pap": 2}, {"k_pep": 4, "k_pap": 4}, {"sleeves": ["pap"]},
                                 {"sleeves": ["pep"]}],
    "s16_treasury_auctions": [{"pre": 3}, {"pre": 7}, {"post": [1, 2, 3]}, {"post": [2]}, {"pre": 0}],
    "s15_rebalancing_flows": [{"z_min": 0.0}, {"z_min": 1.0}, {"late": [-3, -2, -1]}, {"early": [1, 2, 3]}],
    "s11_cross_asset_carry": [{"window": 126}, {"window": 504}],
    "s01_gp_trend": [{"freq": "W", "adjust": 0.5}, {"lengthscales": [126, 252]}, {"z_scale": 1.0}],
}


def main(names):
    ds = DataStore()
    rows = []
    for n in names:
        base = load(n)
        for ov in [{}] + GRID.get(n, []):
            st = load(n, **ov)
            out = runner.run(st, ds=ds, save=False, log=bool(ov), tag="sens", note=f"sensitivity {json.dumps(ov)}")
            r = out["res"]
            s0420 = perf_stats(r.net, start="2004-01-01", end=C.DEV_END, first_trade=r.info.get("first_trade"))
            rows.append({"strategy": n, "override": json.dumps(ov) if ov else "(chosen)",
                         "SR_dev": round(out["stats"]["sharpe"], 2), "SR_04_20": round(s0420["sharpe"], 2),
                         "maxDD": round(out["stats"]["max_dd"], 3), "turnover": round(out["stats"].get("turnover_yr", 0), 1)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "sensitivity.csv", index=False)
    (C.RESULTS / "sensitivity.md").write_text("# Parameter sensitivity (dev, net)\n\n" + df.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or list(GRID))
