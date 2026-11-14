"""ONE-SHOT holdout evaluation (2021-01-01 -> end of data). Requires the FROZEN file (created at freeze time with the
git commit hash). Runs every ACTIVE strategy exactly as frozen on the full data; metrics are computed on the holdout
window only; dev metrics of the same versions are reported side by side.  python scripts/run_holdout.py"""
import json

import numpy as np
import pandas as pd

from oafbt import config as C
from oafbt.data.store import DataStore
from oafbt.engine import runner
from oafbt.engine.metrics import perf_stats
from oafbt.engine.report import combined, corr_matrix
from oafbt.strategies.registry import ACTIVE, load

OUT = C.RESULTS / "holdout"


def main():
    if not C.FROZEN_FILE.exists():
        raise SystemExit("not frozen")
    frozen = json.loads(C.FROZEN_FILE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    ds_h = DataStore(holdout=True)
    rows, rets_h, rets_d = [], {}, {}
    for n in ACTIVE:
        st = load(n)
        assert st.version == frozen["versions"][n], f"{n}: version changed after freeze"
        out = runner.run(st, ds=ds_h, holdout=True, save=True, log=True, tag="holdout")
        r = out["res"]
        h = out["stats"]
        bench = runner.benchmark(ds_h, st.universe)
        d = perf_stats(r.net, bench=bench, start=C.EVAL_START[st.universe], end=C.DEV_END,
                       first_trade=r.info.get("first_trade"))
        rets_h[n] = r.net[r.net.index >= pd.Timestamp(C.HOLDOUT_START)]
        rets_d[n] = r.net[r.net.index <= pd.Timestamp(C.DEV_END)]
        rows.append({"strategy": n, "version": st.version, "SR_dev": d.get("sharpe"), "SR_holdout": h.get("sharpe"),
                     "ret_holdout": h.get("ann_ret"), "vol_holdout": h.get("ann_vol"), "maxDD_holdout": h.get("max_dd"),
                     "t_holdout": h.get("t_hac"), "beta_holdout": h.get("beta_spy"), "years_holdout": h.get("n_years"),
                     "holdout_end": h.get("end"), "pos_years_holdout": h.get("pos_years")})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows).set_index("strategy")
    df.to_csv(OUT / "holdout_metrics.csv")
    full = {n: pd.concat([rets_d[n], rets_h[n]]) for n in ACTIVE}
    comb = combined(full)
    cd = perf_stats(comb, start="2005-01-01", end=C.DEV_END)
    ch = perf_stats(comb, start=C.HOLDOUT_START)
    comb.to_frame("combined").to_parquet(OUT / "combined_daily.parquet")
    spy = ds_h.xret(["SPY"])["SPY"]
    sh = perf_stats(spy, start=C.HOLDOUT_START)
    lines = [f"# One-shot holdout ({C.HOLDOUT_START} -> {ch.get('end')}), frozen at {frozen['commit'][:10]} ({frozen['time']})\n",
             "| strategy | version | SR dev | SR holdout | ann.ret holdout | vol | maxDD | t (HAC) | beta | positive years |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in df.iterrows():
        lines.append(f"| {n} | {r.version} | {r.SR_dev:.2f} | {r.SR_holdout:.2f} | {r.ret_holdout:.1%} | {r.vol_holdout:.1%} | "
                     f"{r.maxDD_holdout:.1%} | {r.t_holdout:.2f} | {r.beta_holdout:.2f} | {r.pos_years_holdout} |")
    lines.append(f"\n**Equal-risk combination**: dev 2005-2020 Sharpe {cd['sharpe']:.2f} (ann {cd['ann_ret']:.1%}, maxDD "
                 f"{cd['max_dd']:.1%}); **holdout Sharpe {ch['sharpe']:.2f}** (ann {ch['ann_ret']:.1%}, vol {ch['ann_vol']:.1%}, "
                 f"maxDD {ch['max_dd']:.1%}, t {ch['t_hac']:.2f}, positive years {ch['pos_years']}).")
    lines.append(f"SPY excess over the same holdout: Sharpe {sh['sharpe']:.2f}, ann {sh['ann_ret']:.1%}, maxDD {sh['max_dd']:.1%}.")
    cm = corr_matrix(rets_h)
    lines.append("\n## Holdout weekly correlations\n" + cm.round(2).to_markdown())
    (OUT / "holdout.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
