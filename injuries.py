"""
injuries.py — NFL injury feeds and a team-level impact score.

Sources
  * Sleeper (free, no key): every player with injury_status / body part / notes and
    depth-chart slot. Updated many times a day. Best for "what changed since this morning".
  * nflverse official injury reports (weekly, Wed–Fri practice + game status).

Impact score = win-probability hit (negative number) for a team, driven by which
depth-chart starters are Out / Doubtful / IR / PUP, with the QB weighted heaviest.
Pure functions except the two fetchers; see tests/test_injuries.py.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

SLEEPER_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_TO_NFLVERSE = {"LAR": "LA", "JAC": "JAX"}          # everything else matches
OUT_STATUSES = {"Out", "IR", "PUP", "Doubtful", "Sus", "NA", "DNR", "COV"}
QUESTIONABLE = {"Questionable"}

# Win-prob hit when the depth-chart #1 at this position is out. QB dwarfs everything;
# the rest are rough, and capped in total below. Questionable counts half.
POS_WEIGHT = {"QB": 0.060, "OT": 0.010, "OL": 0.008, "G": 0.006, "C": 0.006,
              "WR": 0.010, "TE": 0.006, "RB": 0.006,
              "DE": 0.008, "DL": 0.006, "DT": 0.006, "LB": 0.006, "CB": 0.008, "DB": 0.006, "S": 0.006,
              "K": 0.004, "P": 0.001}
BACKUP_QB_WEIGHT = 0.010     # QB2 out only matters a little (if QB1 also dinged it compounds)
MAX_NON_QB_HIT = 0.05


def fetch_sleeper(timeout: int = 30) -> pd.DataFrame:
    r = requests.get(SLEEPER_URL, timeout=timeout)
    r.raise_for_status()
    df = pd.DataFrame.from_dict(r.json(), orient="index")
    keep = ["full_name", "team", "position", "injury_status", "injury_body_part", "injury_notes",
            "depth_chart_order", "depth_chart_position", "status", "news_updated"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df[df["team"].notna()]
    df["team"] = df["team"].replace(SLEEPER_TO_NFLVERSE)
    df["source"] = "sleeper"
    return df.reset_index(drop=True)



def injured_players(sleeper: pd.DataFrame, team: str) -> pd.DataFrame:
    t = sleeper[(sleeper["team"] == team) & sleeper["injury_status"].notna()].copy()
    t["is_starter"] = t["depth_chart_order"].fillna(99).astype(float) <= 1
    sev = {**{s: 2 for s in OUT_STATUSES}, **{s: 1 for s in QUESTIONABLE}}
    t["severity"] = t["injury_status"].map(sev).fillna(0)
    return t.sort_values(["severity", "is_starter"], ascending=False)


def team_injury_impact(sleeper: pd.DataFrame, team: str, starting_qb: Optional[str] = None) -> Tuple[float, List[str]]:
    """
    Returns (win_prob_hit <= 0, notes). If `starting_qb` (from the nflverse schedule) is
    given and is NOT the injured QB, the QB hit is skipped: the schedule already knows
    who starts, and the market has priced that name.
    """
    t = injured_players(sleeper, team)
    hit, notes, non_qb = 0.0, [], 0.0
    for _, r in t.iterrows():
        st = r["injury_status"]
        mult = 1.0 if st in OUT_STATUSES else 0.5 if st in QUESTIONABLE else 0.0
        if mult == 0:
            continue
        pos, order = r["position"], float(r["depth_chart_order"]) if pd.notna(r["depth_chart_order"]) else 99
        if pos == "QB":
            if order == 1 and (starting_qb is None or _same_name(starting_qb, r["full_name"])):
                hit -= POS_WEIGHT["QB"] * mult
                notes.append(f"QB1 {r['full_name']} {st} ({r.get('injury_body_part') or '?'})")
            elif order == 2 and mult == 1.0:
                hit -= BACKUP_QB_WEIGHT
                notes.append(f"QB2 {r['full_name']} {st}")
            continue
        if order == 1:
            w = POS_WEIGHT.get(pos, 0.004) * mult
            non_qb += w
            notes.append(f"{pos}1 {r['full_name']} {st}")
    hit -= min(non_qb, MAX_NON_QB_HIT)
    return round(hit, 4), notes


def _same_name(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    return a == b or (a.split()[-1] == b.split()[-1] and a[0] == b[0])


def matchup_news_ticks(sleeper: pd.DataFrame, home: str, away: str,
                       home_qb: Optional[str] = None, away_qb: Optional[str] = None,
                       tick: float = 0.005) -> Tuple[int, List[str], List[str]]:
    """Default value for the card's News slider (+ favors home). One tick = 0.5% win prob."""
    h_hit, h_notes = team_injury_impact(sleeper, home, home_qb)
    a_hit, a_notes = team_injury_impact(sleeper, away, away_qb)
    ticks = int(round((h_hit - a_hit) / tick))
    return max(-10, min(10, ticks)), h_notes, a_notes


# ---------------------------------------------------------------- snapshots / diff

def snapshot(sleeper: pd.DataFrame) -> Dict[str, dict]:
    """Compact dict keyed by 'TEAM|Name' for players with any injury status."""
    t = sleeper[sleeper["injury_status"].notna()]
    return {f"{r['team']}|{r['full_name']}": {"team": r["team"], "name": r["full_name"], "pos": r["position"],
                                              "status": r["injury_status"], "part": r.get("injury_body_part"),
                                              "notes": r.get("injury_notes"),
                                              "starter": bool(pd.notna(r["depth_chart_order"]) and float(r["depth_chart_order"]) <= 1)}
            for _, r in t.iterrows()}


def diff_snapshots(prev: Dict[str, dict], cur: Dict[str, dict]) -> Dict[str, List[dict]]:
    new = [v for k, v in cur.items() if k not in prev]
    cleared = [v for k, v in prev.items() if k not in cur]
    changed = [{**cur[k], "was": prev[k]["status"]} for k in cur if k in prev and prev[k]["status"] != cur[k]["status"]]
    return {"new": new, "changed": changed, "cleared": cleared}


def format_diff(d: Dict[str, List[dict]], starters_only: bool = True, limit: int = 12) -> str:
    def pick(rows):
        rows = [r for r in rows if r.get("starter") or r.get("pos") == "QB"] if starters_only else rows
        return rows[:limit]
    lines = []
    for r in pick(d["new"]):
        lines.append(f"🆕 {r['team']} {r['pos']} {r['name']}: {r['status']}" + (f" ({r['part']})" if r.get("part") else ""))
    for r in pick(d["changed"]):
        lines.append(f"🔁 {r['team']} {r['pos']} {r['name']}: {r['was']} → {r['status']}")
    for r in pick(d["cleared"]):
        lines.append(f"✅ {r['team']} {r['pos']} {r['name']}: cleared")
    return "\n".join(lines) if lines else "No starter injury changes."


def save_snapshot(snap: Dict[str, dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"ts": dt.datetime.now().isoformat(timespec="minutes"), "players": snap}, f)


def load_snapshot(path: str) -> Tuple[Optional[str], Dict[str, dict]]:
    if not os.path.exists(path):
        return None, {}
    with open(path) as f:
        d = json.load(f)
    return d.get("ts"), d.get("players", {})
