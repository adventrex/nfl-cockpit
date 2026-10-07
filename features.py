"""
features.py — leakage-free, pre-game feature construction for the win-prob model.

Everything a game's feature row uses is known BEFORE kickoff:
  * team EPA (pass/rush, off/def) = exponentially-weighted mean of the team's PRIOR
    games, carried across seasons with regression toward the league mean
  * QB EPA/dropback = same EWM but keyed by the PLAYER, so it follows a traded QB
  * backup-QB flag = scheduled starter differs from the team's last-game starter
  * context = rest-day diff, divisional game, outdoor roof, (wind: schedule actual —
    a forecast stand-in; small look-ahead, flagged in the report)
  * injuries = position-weighted count of Out/Doubtful on the official report
The market prior (no-vig closing moneyline logit) is always the first feature.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from engine import american_to_decimal, logit, no_vig_two_way

TEAM_STATS = ["epa_pass_off", "epa_rush_off", "epa_pass_def", "epa_rush_def"]
HALFLIFE_GAMES = 6          # EWM half-life in games
OFFSEASON_CARRY = 0.6       # 60% of last season's rating survives the offseason
QB_HALFLIFE = 8
QB_MIN_DROPBACKS = 10       # a game with fewer dropbacks doesn't update the QB rating
INJ_POS_WEIGHT = {"QB": 3.0, "OT": 1.0, "T": 1.0, "G": 0.6, "C": 0.6, "OL": 0.8, "WR": 1.0, "TE": 0.6, "RB": 0.6,
                  "DE": 0.8, "DT": 0.6, "DL": 0.6, "LB": 0.6, "CB": 0.8, "S": 0.6, "DB": 0.6, "K": 0.4}

FEATURE_SETS: Dict[str, List[str]] = {
    "market": ["logit_mkt"],
    "team": ["diff_pass", "diff_rush"],
    "qb": ["diff_qb", "diff_qb_change"],
    "context": ["rest_diff", "div_game", "outdoor", "wind"],
    "injuries": ["diff_inj"],
}


def _alpha(halflife: float) -> float:
    return 1 - 0.5 ** (1.0 / halflife)


# ---------------------------------------------------------------- team weekly -> rolling

def weekly_team_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team, game_id) with that game's EPA splits."""
    p = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    p["defteam"] = p["defteam"].fillna(p["posteam"])
    rows = []
    for side, col in (("off", "posteam"), ("def", "defteam")):
        g = p.groupby(["season", "week", "game_id", col, "play_type"])["epa"].mean().unstack("play_type")
        g = g.rename(columns={"pass": f"epa_pass_{side}", "run": f"epa_rush_{side}"}).reset_index().rename(columns={col: "team"})
        rows.append(g)
    out = rows[0].merge(rows[1], on=["season", "week", "game_id", "team"], how="outer")
    return out.sort_values(["team", "season", "week"]).reset_index(drop=True)


def rolling_team_ratings(weekly: pd.DataFrame, halflife: float = HALFLIFE_GAMES, carry: float = OFFSEASON_CARRY) -> pd.DataFrame:
    """Pre-game EWM rating for every (season, week, team) row in `weekly`.
    The rating for a row uses only games BEFORE it."""
    a = _alpha(halflife)
    league = weekly[TEAM_STATS].mean()
    out = []
    for team, g in weekly.groupby("team", sort=False):
        state = league.copy()
        last_season = None
        for _, r in g.sort_values(["season", "week"]).iterrows():
            if last_season is not None and r["season"] != last_season:
                state = carry * state + (1 - carry) * league
            out.append({"season": r["season"], "week": r["week"], "team": team, **{f"r_{k}": state[k] for k in TEAM_STATS}})
            obs = r[TEAM_STATS]
            state = np.where(obs.notna(), a * obs.fillna(0) + (1 - a) * state, state)
            state = pd.Series(state, index=TEAM_STATS)
            last_season = r["season"]
    return pd.DataFrame(out)


def latest_team_rating(ratings: pd.DataFrame, weekly: pd.DataFrame, team: str) -> Optional[pd.Series]:
    """Rating a team would carry into its NEXT game (post-update of its last game)."""
    g = weekly[weekly["team"] == team].sort_values(["season", "week"])
    if g.empty:
        return None
    a = _alpha(HALFLIFE_GAMES)
    last = g.iloc[-1]
    pre = ratings[(ratings["team"] == team) & (ratings["season"] == last["season"]) & (ratings["week"] == last["week"])]
    if pre.empty:
        return None
    state = pd.Series({k: pre.iloc[0][f"r_{k}"] for k in TEAM_STATS})
    obs = last[TEAM_STATS]
    return pd.Series(np.where(obs.notna(), a * obs.fillna(0) + (1 - a) * state, state), index=TEAM_STATS)


# ---------------------------------------------------------------- QB (player-keyed)

def weekly_qb_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[(pbp["play_type"] == "pass") & pbp["passer_player_id"].notna()]
    g = (p.groupby(["season", "week", "game_id", "posteam", "passer_player_id"])
         .agg(qb_epa=("epa", "mean"), dropbacks=("play_id", "count")).reset_index())
    return g[g["dropbacks"] >= QB_MIN_DROPBACKS].sort_values(["passer_player_id", "season", "week"]).reset_index(drop=True)


def rolling_qb_ratings(qb_weekly: pd.DataFrame, halflife: float = QB_HALFLIFE) -> Dict[str, List[tuple]]:
    """{qb_id: [(season, week, pre_game_rating), ...]} plus a final post-update rating under key (season=9999)."""
    a = _alpha(halflife)
    league = float(qb_weekly["qb_epa"].mean()) if len(qb_weekly) else 0.0
    out: Dict[str, List[tuple]] = {}
    for qb, g in qb_weekly.groupby("passer_player_id", sort=False):
        state, hist = league, []
        for _, r in g.iterrows():
            hist.append((r["season"], r["week"], state))
            state = a * r["qb_epa"] + (1 - a) * state
        hist.append((9999, 0, state))
        out[qb] = hist
    out["__league__"] = [(9999, 0, league)]
    return out


def qb_rating_before(qb_ratings: Dict[str, List[tuple]], qb_id, season: int, week: int) -> float:
    """Rating a QB carries into (season, week): the pre-game state of that row if he
    played it, else the pre-game state of his next row (= post-update of his last one),
    else his final post-update state. Unknown QB -> league mean."""
    league = qb_ratings["__league__"][0][2]
    if qb_id is None or qb_id not in qb_ratings:
        return league
    for s, w, r in qb_ratings[qb_id]:
        if s == 9999 or (s, w) >= (season, week):
            return r
    return league


# ---------------------------------------------------------------- injuries (official reports)

def injury_load(inj: pd.DataFrame) -> pd.DataFrame:
    """Position-weighted count of Out/Doubtful per (season, week, team)."""
    if inj is None or inj.empty:
        return pd.DataFrame(columns=["season", "week", "team", "inj_load"])
    x = inj[inj["report_status"].isin(["Out", "Doubtful"])].copy()
    x["w"] = x["position"].map(INJ_POS_WEIGHT).fillna(0.4) * np.where(x["report_status"] == "Out", 1.0, 0.7)
    return x.groupby(["season", "week", "team"])["w"].sum().reset_index().rename(columns={"w": "inj_load"})


# ---------------------------------------------------------------- game rows

def _prev_qb_map(sched: pd.DataFrame) -> Dict[tuple, Optional[str]]:
    """(season, week, team) -> qb_id the team started in its previous game (any season)."""
    rows = []
    for side in ("home", "away"):
        rows.append(sched[["season", "week", f"{side}_team", f"{side}_qb_id"]].rename(columns={f"{side}_team": "team", f"{side}_qb_id": "qb_id"}))
    t = pd.concat(rows).sort_values(["team", "season", "week"])
    t["prev_qb"] = t.groupby("team")["qb_id"].shift(1)
    return {(r.season, r.week, r.team): r.prev_qb for r in t.itertuples()}


def build_game_features(sched: pd.DataFrame, pbp: pd.DataFrame, injuries: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per REG-season game with a moneyline. Includes unplayed games (home_win NaN)."""
    s = sched[(sched["game_type"] == "REG") & sched["home_moneyline"].notna() & sched["away_moneyline"].notna()].copy()
    s = s.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    weekly = weekly_team_stats(pbp)
    ratings = rolling_team_ratings(weekly)
    qbw = weekly_qb_stats(pbp)
    qbr = rolling_qb_ratings(qbw)
    inj = injury_load(injuries)
    prev_qb = _prev_qb_map(s)

    # Team ratings: rows exist only for played games; for unplayed games use the latest post-update rating
    latest = {t: latest_team_rating(ratings, weekly, t) for t in weekly["team"].unique()}
    rkey = ratings.set_index(["season", "week", "team"])

    def team_rating(season, week, team):
        try:
            r = rkey.loc[(season, week, team)]
            return pd.Series({k: r[f"r_{k}"] for k in TEAM_STATS})
        except KeyError:
            lr = latest.get(team)
            return lr if lr is not None else pd.Series({k: 0.0 for k in TEAM_STATS})

    feats = []
    for r in s.itertuples():
        h, a = team_rating(r.season, r.week, r.home_team), team_rating(r.season, r.week, r.away_team)
        h_net_pass = h["epa_pass_off"] - a["epa_pass_def"]; a_net_pass = a["epa_pass_off"] - h["epa_pass_def"]
        h_net_rush = h["epa_rush_off"] - a["epa_rush_def"]; a_net_rush = a["epa_rush_off"] - h["epa_rush_def"]
        hq = qb_rating_before(qbr, r.home_qb_id, r.season, r.week)
        aq = qb_rating_before(qbr, r.away_qb_id, r.season, r.week)
        h_prev, a_prev = prev_qb.get((r.season, r.week, r.home_team)), prev_qb.get((r.season, r.week, r.away_team))
        h_chg = int(pd.notna(r.home_qb_id) and pd.notna(h_prev) and r.home_qb_id != h_prev)
        a_chg = int(pd.notna(r.away_qb_id) and pd.notna(a_prev) and r.away_qb_id != a_prev)
        hi = inj[(inj.season == r.season) & (inj.week == r.week) & (inj.team == r.home_team)]["inj_load"].sum() if len(inj) else 0.0
        ai = inj[(inj.season == r.season) & (inj.week == r.week) & (inj.team == r.away_team)]["inj_load"].sum() if len(inj) else 0.0
        pm = no_vig_two_way(american_to_decimal(r.home_moneyline), american_to_decimal(r.away_moneyline))[0]
        feats.append({
            "game_id": r.game_id, "season": r.season, "week": r.week, "gameday": r.gameday,
            "home_team": r.home_team, "away_team": r.away_team,
            "home_win": (1 if r.home_score > r.away_score else 0) if pd.notna(r.home_score) else np.nan,
            "p_mkt": pm, "logit_mkt": logit(pm),
            "home_dec": american_to_decimal(r.home_moneyline), "away_dec": american_to_decimal(r.away_moneyline),
            "diff_pass": h_net_pass - a_net_pass, "diff_rush": h_net_rush - a_net_rush,
            "diff_qb": hq - aq, "diff_qb_change": h_chg - a_chg,
            "rest_diff": (r.home_rest if pd.notna(r.home_rest) else 7) - (r.away_rest if pd.notna(r.away_rest) else 7),
            "div_game": int(r.div_game) if pd.notna(r.div_game) else 0,
            "outdoor": int(r.roof == "outdoors") if pd.notna(r.roof) else 0,
            "wind": float(r.wind) if pd.notna(r.wind) else 0.0,
            "diff_inj": hi - ai,
        })
    return pd.DataFrame(feats)


def feature_list(sets: List[str]) -> List[str]:
    cols: List[str] = []
    for s in sets:
        cols += FEATURE_SETS[s]
    return cols
