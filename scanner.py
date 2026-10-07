"""
scanner.py — live pricing-error scanner across sportsbooks.

The sharpest free estimate of a game's true price is the CONSENSUS of many books after
removing each book's vig. A single book sitting off that consensus is a pricing error
you can bet into (or a stale line). This module is pure: feed it The Odds API payload.

Outputs, per game:
  * h2h / spreads / totals: each book's price vs consensus fair price → edge = p_fair·dec − 1
  * arbitrage: best-price synthetic hold < 0 across books
  * cross-book middles on spreads/totals (home −2.5 at A, away +3.5 at B)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from engine import decimal_to_american, no_vig_two_way, synthetic_hold
from hedge_engine import middle_breakdown, middle_window

EDGE_FLAG = 0.015  # 1.5% EV vs consensus
# The Odds API uses full team names; nflverse uses abbreviations. One map, imported everywhere.
TEAM_MAP = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
# The Odds API "us" region mixes regulated books with offshore ones. Offshore quotes are useful
# for consensus (more books = better fair price) but not placeable legally from Oregon.
OFFSHORE = {"betus", "bovada", "betonlineag", "lowvig", "mybookieag", "betanysports", "everygame", "unibet_us"}


def _pair_key(m: dict, home: str, away: str):
    """Map a market's outcomes into (side, point, dec) tuples with a canonical pair id."""
    out = []
    for o in m.get("outcomes", []):
        name, price, point = o.get("name"), o.get("price"), o.get("point")
        if m["key"] == "h2h":
            out.append(("home" if name == home else "away", None, price))
        elif m["key"] == "spreads":
            out.append(("home" if name == home else "away", point, price))
        elif m["key"] == "totals":
            out.append((name.lower(), point, price))
    return out


def parse_payload(games: List[dict], team_map: Dict[str, str]) -> pd.DataFrame:
    """Long table: game, market, book, side, point, dec, plus the opposing quote at the same book."""
    rows = []
    for g in games:
        h, a = team_map.get(g.get("home_team")), team_map.get(g.get("away_team"))
        if not (h and a):
            continue
        gid = f"{a}@{h}"
        for bk in g.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m["key"] not in ("h2h", "spreads", "totals"):
                    continue
                quotes = _pair_key(m, g["home_team"], g["away_team"])
                for side, point, dec in quotes:
                    # opposing quote: same book, same market, other side; for spreads the mirrored point
                    opp = None
                    for s2, p2, d2 in quotes:
                        if s2 != side and (m["key"] == "h2h" or (m["key"] == "totals" and p2 == point) or (m["key"] == "spreads" and p2 is not None and point is not None and abs(p2 + point) < 1e-9)):
                            opp = d2
                    rows.append({"game": gid, "home": h, "away": a, "commence": g.get("commence_time"), "market": m["key"],
                                 "book": bk["key"], "side": side, "point": point, "dec": float(dec), "opp_dec": opp})
    return pd.DataFrame(rows)


def consensus_edges(q: pd.DataFrame, min_books: int = 3) -> pd.DataFrame:
    """For every quote, fair prob = mean of no-vig probs across books quoting the same (market, side, point)."""
    if q.empty:
        return q
    q = q[q.opp_dec.notna()].copy()
    q["p_novig"] = [no_vig_two_way(d, o)[0] for d, o in zip(q.dec, q.opp_dec)]
    key = ["game", "market", "side", "point"]
    grp = q.groupby(key, dropna=False)["p_novig"]
    q["p_fair"] = grp.transform("mean")
    q["n_books"] = grp.transform("count")
    q = q[q.n_books >= min_books].copy()
    q["edge"] = q.p_fair * q.dec - 1
    q["fair_am"] = [decimal_to_american(1 / p) for p in q.p_fair]
    q["am"] = [decimal_to_american(d) for d in q.dec]
    return q.sort_values("edge", ascending=False).reset_index(drop=True)


def arbs(q: pd.DataFrame) -> pd.DataFrame:
    """Best price per side across books; hold < 0 is a risk-free arb (before limits/latency)."""
    if q.empty:
        return q
    out = []
    for (game, market, point_abs), g in q.assign(point_abs=q.point.abs()).groupby(["game", "market", "point_abs"], dropna=False):
        sides = g.side.unique()
        if len(sides) != 2:
            continue
        b = g.loc[g.groupby("side")["dec"].idxmax()]
        d1, d2 = b.dec.tolist()
        hold = synthetic_hold(d1, d2)
        out.append({"game": game, "market": market, "point": point_abs if pd.notna(point_abs) else None, "hold": hold,
                    "leg1": f"{b.iloc[0].side} {b.iloc[0].point if pd.notna(b.iloc[0].point) else ''} {decimal_to_american(d1):+d} @{b.iloc[0].book}",
                    "leg2": f"{b.iloc[1].side} {b.iloc[1].point if pd.notna(b.iloc[1].point) else ''} {decimal_to_american(d2):+d} @{b.iloc[1].book}"})
    df = pd.DataFrame(out)
    return df.sort_values("hold").reset_index(drop=True) if len(df) else df


def middles(q: pd.DataFrame, sd: float = 13.5, min_prob: float = 0.05) -> pd.DataFrame:
    """Cross-book spread middles: home at −x with one book, away at +y (y > x) with another."""
    if q.empty:
        return q
    out = []
    for game, g in q[q.market == "spreads"].groupby("game"):
        homes, aways = g[g.side == "home"], g[g.side == "away"]
        if homes.empty or aways.empty:
            continue
        cur = float(homes.point.median())          # consensus home spread, negative = home favoured
        for hrow in homes.itertuples():
            for arow in aways.itertuples():
                win = middle_window(hrow.point, arow.point)
                if not win:
                    continue
                pb, px, py = middle_breakdown(win, cur, sd)
                if pb < min_prob:
                    continue
                out.append({"game": game, "home_leg": f"{hrow.home if hasattr(hrow,'home') else 'home'} {hrow.point:+g} {decimal_to_american(hrow.dec):+d} @{hrow.book}",
                            "away_leg": f"away {arow.point:+g} {decimal_to_american(arow.dec):+d} @{arow.book}",
                            "window": f"{win[0]:g}–{win[1]:g}", "p_middle": round(pb, 3), "p_lose_both": 0.0,
                            "ev_per_unit_each": round(pb * ((hrow.dec - 1) + (arow.dec - 1)) + px * ((hrow.dec - 1) - 1) + py * ((arow.dec - 1) - 1), 3)})
    df = pd.DataFrame(out)
    return df.sort_values("ev_per_unit_each", ascending=False).reset_index(drop=True) if len(df) else df
