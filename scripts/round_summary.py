"""Leaderboard for the active strategies (latest version of each), correlations and an equal-risk combination.

   python scripts/round_summary.py <round_number>     -> results/rounds/round_<n>.md (+ csv)
"""
import json
import re
import sys

import numpy as np
import pandas as pd

from oafbt import config as C
from oafbt.engine.metrics import perf_stats
from oafbt.engine.report import combined, corr_matrix
from oafbt.strategies.registry import ACTIVE

RUNS = C.RESULTS / "runs"


def latest_version(name):
    d = RUNS / name
    if not d.exists():
        return None
    vs = [p.name for p in d.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name) and (p / "summary.json").exists()]
    return max(vs, key=lambda v: int(v[1:])) if vs else None


def main(rnd: int):
    rows, rets = [], {}
    for n in ACTIVE:
        v = latest_version(n)
        if v is None:
            rows.append({"strategy": n, "version": "-"})
            continue
        s = json.loads((RUNS / n / v / "summary.json").read_text())
        daily = pd.read_parquet(RUNS / n / v / "daily.parquet")
        rets[n] = daily["net"]
        rows.append({"strategy": n, "version": v, "SR_dev": s.get("sharpe"), "SR_04_20": s.get("sharpe_2004_2020"),
                     "SR_2xcost": s.get("sharpe_cost2x"), "SR_lag+1": s.get("sharpe_lag+1"),
                     "ann_ret": s.get("ann_ret"), "vol": s.get("ann_vol"), "maxDD": s.get("max_dd"),
                     "calmar": s.get("calmar"), "turnover": s.get("turnover_yr"), "cost": s.get("cost_drag_yr"),
                     "beta": s.get("beta_spy"), "sub": s.get("sr_subperiods"), "dsr": s.get("dsr"),
                     "trials": s.get("n_trials"), "lookahead": s.get("lookahead_passed"),
                     "start": s.get("start")})
    df = pd.DataFrame(rows).set_index("strategy")
    from oafbt.strategies.registry import load
    cal = {n: getattr(load(n), "calendar_timed", False) for n in df.index if df.loc[n, "version"] != "-"}
    lagcol = df["SR_lag+1"].where(~df.index.map(lambda n: cal.get(n, False)), np.nan)
    df["score"] = pd.concat([df["SR_04_20"], df["SR_2xcost"], lagcol], axis=1).min(axis=1) * 0.5 + df["SR_04_20"] * 0.5
    out = C.RESULTS / "rounds"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"round_{rnd}.csv")
    lines = [f"# Round {rnd} leaderboard (dev data only, net of costs)\n",
             "score = 0.5*SR(2004-2020) + 0.5*min(SR 2004-2020, SR 2x cost, SR lag+1); lag+1 skipped for calendar-timed strategies\n"]
    cols = ["version", "score", "SR_04_20", "SR_dev", "SR_2xcost", "SR_lag+1", "ann_ret", "vol", "maxDD", "calmar",
            "turnover", "cost", "beta", "sub", "trials", "lookahead", "start"]
    lines.append("| strategy | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for n, r in df.sort_values("score", ascending=False).iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float) and not np.isnan(v):
                cells.append(f"{v:.1%}" if c in ("ann_ret", "vol", "maxDD", "cost") else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    if len(rets) >= 2:
        cm = corr_matrix(rets, start="2004-01-01", end=C.DEV_END)
        lines.append("\n## Weekly return correlations 2004-2020\n")
        lines.append(cm.round(2).to_markdown() if hasattr(cm, "to_markdown") else cm.round(2).to_string())
        comb = combined(rets, start="2004-01-01", end=C.DEV_END)
        st = perf_stats(comb, start="2005-01-01")
        lines.append(f"\n## Equal-risk combination of all active strategies (2005-2020, weights from trailing data)\n"
                     f"Sharpe {st['sharpe']:.2f}, ann.ret {st['ann_ret']:.1%}, vol {st['ann_vol']:.1%}, maxDD {st['max_dd']:.1%}, "
                     f"calmar {st['calmar']:.2f}, subperiods {st['sr_subperiods']}, positive years {st['pos_years']}")
        comb.to_frame("combined").to_parquet(out / f"round_{rnd}_combined.parquet")
    (out / f"round_{rnd}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
