#!/usr/bin/env python
"""
scan_injuries.py — twice-daily NFL injury scan (launchd: com.andrew.nfl-injury-scan).

Pulls Sleeper, diffs against the last snapshot, writes data/injuries_latest.json (which
the app reads to auto-fill the News slider), and iMessages Andrew the starter-level
changes. Deterministic, no LLM. Safe to run by hand:
    .venv/bin/python scan_injuries.py            # scan + text
    .venv/bin/python scan_injuries.py --dry-run  # scan, print, no text
"""
import datetime as dt
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import injuries as I  # noqa: E402
from notify import imessage  # noqa: E402

DATA = os.path.join(HERE, "data")
LATEST = os.path.join(DATA, "injuries_latest.json")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%F %T",
                    handlers=[logging.FileHandler("/tmp/nfl-injury-scan.log"), logging.StreamHandler(sys.stdout)])
log = logging.info


def main(dry=False):
    log("scan start")
    try:
        sl = I.fetch_sleeper()
    except Exception as ex:
        log(f"sleeper fetch failed: {ex}")
        if not dry:
            imessage(f"🏈 Injury scan failed: {type(ex).__name__}")
        return 1
    prev_ts, prev = I.load_snapshot(LATEST)
    cur = I.snapshot(sl)
    d = I.diff_snapshots(prev, cur)
    body = I.format_diff(d)
    when = "AM" if dt.datetime.now().hour < 12 else "PM"
    header = f"🏈 Injury scan {when} · {len(cur)} listed · since {prev_ts or 'first run'}"
    msg = f"{header}\n{body}"
    I.save_snapshot(cur, LATEST)
    log(f"new={len(d['new'])} changed={len(d['changed'])} cleared={len(d['cleared'])}")
    if dry:
        print(msg)
    else:
        imessage(msg)
        log("iMessage sent")
    return 0


if __name__ == "__main__":
    sys.exit(main(dry="--dry-run" in sys.argv))
