"""Train/test guard. Set TRAIN_END ONCE per project BEFORE looking at data (default ~70/30 time split + 1 month embargo).
Use train(x) for ALL exploration/tuning. Call unlock_test(reason) ONCE to access the held-out period; accesses are logged."""
import os, datetime as dt, pandas as pd
TRAIN_END = "2023-06-30"        # <- EDIT per dataset before exploring; first ~70% of history
TEST_START = (pd.Timestamp(TRAIN_END) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")   # 1-month embargo
LOG = "results/test_access_log.txt"

def train(x):
    """Return only the TRAIN window (index must be a DatetimeIndex)."""
    return x.loc[:TRAIN_END]

def unlock_test(reason):
    """Log and return a function that slices the TEST window. Call once, after the strategy is frozen and committed."""
    assert reason and len(reason) > 10, "state why/what is frozen"
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f: f.write(f"{dt.datetime.utcnow().isoformat()}Z TEST UNLOCKED: {reason}\n")
    if os.path.getsize(LOG) > 0 and sum(1 for _ in open(LOG)) > 1: print("WARNING: TEST accessed more than once; it is now contaminated. Report this.")
    return lambda x: x.loc[TEST_START:]
