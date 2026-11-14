"""Freeze the research: write ./FROZEN (commit hash, time, strategy versions, data fingerprint) - this unlocks
DataStore(holdout=True). Refuses to run with uncommitted changes.  python scripts/freeze.py"""
import datetime as dt
import hashlib
import json
import subprocess

from oafbt import config as C
from oafbt.strategies.registry import ACTIVE, load


def main():
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=C.ROOT).stdout.strip()
    if dirty:
        raise SystemExit(f"uncommitted changes:\n{dirty}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.ROOT).stdout.strip()
    fp = {}
    for f in sorted((C.DATA / "vault").glob("*")):
        fp[f.name] = hashlib.md5(f.read_bytes()).hexdigest()[:12]
    rec = {"commit": commit, "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "versions": {n: load(n).version for n in ACTIVE}, "vault_fingerprint": fp,
           "note": "One-shot holdout: strategies frozen at this commit; holdout = 2021-01-01 .. end of data."}
    C.FROZEN_FILE.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "vault_fingerprint"}, indent=1))


if __name__ == "__main__":
    main()
