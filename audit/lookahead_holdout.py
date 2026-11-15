"""Re-run the author's look-ahead check, but with cut dates INSIDE the 2021-2026 holdout window (the author's own
check used dev-period cuts only)."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/claude-0/audit")
from oafbt.data import store
from oafbt.engine import checks
from oafbt.strategies.registry import load
checks.DataStore = lambda end=None: store.DataStore(holdout=True, end=end)
from oafbt.strategies.utils import nyse_sessions
ses = nyse_sessions(); ses = ses[(ses >= "2021-03-01") & (ses <= "2026-09-30")]
rng = np.random.default_rng(11)
cuts = sorted(pd.DatetimeIndex(rng.choice(ses, size=6, replace=False)).tolist() + [pd.Timestamp(x) for x in ["2022-06-30","2024-08-05","2026-10-30"]])
for n in sys.argv[1:]:
    kw = {"fomc_day": False} if n == "s10_turn_of_month" else {}
    st = load(n, **kw)
    r = checks.lookahead_check(st, cuts=cuts)
    bad = {k: v for k, v in r.items() if k != "passed" and (v["hold_mismatch"] or v["value_mismatch"] or v["returns_mismatch"] or v["rows_after_cut"])}
    print(f"{n:28s} cuts={len(cuts)}  PASSED={r['passed']}  failing cuts: {bad if bad else 'none'}", flush=True)
