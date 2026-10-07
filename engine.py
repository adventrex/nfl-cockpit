"""
engine.py — pure math and decision logic for NFL Edge Cockpit.

No Streamlit imports here on purpose: everything in this file is unit-testable
with plain pytest (see tests/). app.py is the thin UI layer on top.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Odds math
# ---------------------------------------------------------------------------

def odds_api_key(path: str = None) -> Optional[str]:
    """The Odds API key from .streamlit/secrets.toml (ODDS_API_KEY), else the env var. One reader for app, picks and picker."""
    import os, tomllib
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml")
    try:
        with open(path, "rb") as f:
            return tomllib.load(f).get("ODDS_API_KEY") or os.environ.get("ODDS_API_KEY")
    except FileNotFoundError:
        return os.environ.get("ODDS_API_KEY")


def american_to_decimal(odds) -> float:
    """-110 -> 1.909, +150 -> 2.5. Unparseable input returns 2.0 (even money)."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return 2.0
    if odds == 0:
        return 2.0
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def decimal_to_american(dec: float) -> int:
    dec = float(dec)
    if dec <= 1.0:
        return -100000
    return int(round((dec - 1) * 100)) if dec >= 2.0 else int(round(-100 / (dec - 1)))


def no_vig_two_way(d1: float, d2: float) -> Tuple[float, float]:
    """Strip the hold from a two-way market. Returns (p1, p2) summing to 1."""
    p1, p2 = 1 / d1, 1 / d2
    s = p1 + p2
    return p1 / s, p2 / s


def synthetic_hold(d_best_1: float, d_best_2: float) -> float:
    """Hold you actually face if you take the best price on each side across books.
    Negative = arbitrage. Miller/Davidow rule of thumb: skip two-way markets > 2.5%."""
    return (1 / d_best_1 + 1 / d_best_2) - 1.0



def ev_per_dollar(p: float, d: float) -> float:
    return p * d - 1.0


def kelly_fraction(p: float, d: float) -> float:
    """Full Kelly. 0 if no edge."""
    b = d - 1.0
    if b <= 0:
        return 0.0
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return max(0.0, (b * p - (1 - p)) / b)


def half_kelly_fraction(p: float, d: float, cap: float = 0.05) -> float:
    return min(0.5 * kelly_fraction(p, d), cap)


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def inv_logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def z_score(p: float, d: float, se: float) -> float:
    """How many standard errors the model edge is above break-even (spec §4.2: want >= 1.28)."""
    if se <= 0:
        return float("inf") if p > 1 / d else float("-inf")
    return (p - 1 / d) / se


# ---------------------------------------------------------------------------
# Home-field advantage
# ---------------------------------------------------------------------------

def _logit_np(p):
    return np.log(p / (1.0 - p))


def compute_team_home_advantage(games: pd.DataFrame, min_games: int = 10, smooth: float = 0.5) -> Dict[str, float]:
    """
    Team-specific HFA *in excess of the league average*, as a win-probability bump.

    Why excess: the market line already prices the league-wide home edge. Adding a
    team's full HFA on top of the market double-counts it and biases every pick
    toward the home side (v3.2 bug). Only the part that is unusual for this team
    (e.g. DEN altitude, SEA crowd) is information the market might under-weight.
    """
    df = games.dropna(subset=["home_score", "away_score"]).copy()
    if df.empty:
        return {}
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # League baseline: pooled home win rate -> log-odds delta vs a neutral 50/50
    p_league = (df["home_win"].sum() + smooth) / (len(df) + 2 * smooth)
    league_delta = _logit_np(p_league) - _logit_np(1 - p_league)

    hfa_map: Dict[str, float] = {}
    for team in pd.unique(pd.concat([df["home_team"], df["away_team"]])):
        g_home = df[df["home_team"] == team]
        g_away = df[df["away_team"] == team]
        if len(g_home) < min_games or len(g_away) < min_games:
            hfa_map[team] = 0.0
            continue
        p_home = (g_home["home_win"].sum() + smooth) / (len(g_home) + 2 * smooth)
        p_away = ((1 - g_away["home_win"]).sum() + smooth) / (len(g_away) + 2 * smooth)
        delta = _logit_np(p_home) - _logit_np(p_away)
        excess = delta - league_delta
        # Shrink hard (÷4, not ÷2): home/away splits over 3 seasons are noisy and the
        # market already prices most of what is real. This is a nudge, not a thesis.
        hfa_map[team] = float(1.0 / (1.0 + np.exp(-excess / 4.0)) - 0.5)
    return hfa_map


# ---------------------------------------------------------------------------
# Sliders <-> EPA (one constant, both directions, so defaults round-trip)
# ---------------------------------------------------------------------------

EPA_PER_TICK = 0.03      # one slider tick = 0.03 EPA/play
HFA_PER_TICK = 0.015     # one HFA tick = 1.5% win prob
HFA_DEFAULT_MAX_TICKS = 2  # data-driven default never moves more than ±2 ticks (±3%) off league average
NEWS_MAX_SWING = 0.05    # news slider at +/-10 = +/-5% win prob toward home/away
WEATHER_MAX_SHRINK = 0.15  # weather slider at 10 shrinks the edge over 50% by 15%
REST_PER_DAY = 0.005


def epa_to_slider(epa: float, reverse: bool = False) -> int:
    val = 5 - epa / EPA_PER_TICK if reverse else 5 + epa / EPA_PER_TICK
    return int(round(min(max(val, 0), 10)))


def slider_delta_epa(slider_val: float, default_val: float) -> float:
    """EPA adjustment implied by moving a slider away from its data-driven default.
    At the default this is exactly 0, so the data isn't counted twice (v3.2 bug)."""
    return (slider_val - default_val) * EPA_PER_TICK




# ---------------------------------------------------------------------------
# Win probability
# ---------------------------------------------------------------------------

def calc_win_prob(
    market_prob: float,
    feature_row: Dict[str, float],
    sliders: Dict[str, float],
    defaults: Dict[str, float],
    clf=None,
    feature_names: Optional[list] = None,
    market_weight: float = 0.7,
) -> Tuple[float, Dict[str, float]]:
    """
    Blend the no-vig market probability (prior) with the logistic model and the
    user's manual adjustments. Returns (home_win_prob, breakdown).

    feature_row : pre-game features for this game (features.build_game_features row);
                  only diff_pass / diff_rush are touched by the sliders
    sliders : ph_qb ph_pwr ph_def pa_qb pa_pwr pa_def (0-10),
              news (-10..+10, + favors home), ww (0-10), hfa (0-10), wr (0-10), rh, ra
    defaults: h_qb h_pwr h_def a_qb a_pwr a_def hfa  (the data-driven slider defaults)
    """
    h_pass = slider_delta_epa(sliders["ph_qb"], defaults["h_qb"])
    h_rush = slider_delta_epa(sliders["ph_pwr"], defaults["h_pwr"])
    h_def = slider_delta_epa(sliders["ph_def"], defaults["h_def"])
    a_pass = slider_delta_epa(sliders["pa_qb"], defaults["a_qb"])
    a_rush = slider_delta_epa(sliders["pa_pwr"], defaults["a_pwr"])
    a_def = slider_delta_epa(sliders["pa_def"], defaults["a_def"])
    adj_net_pass = (h_pass - a_pass) + (h_def - a_def)
    adj_net_rush = (h_rush - a_rush) + (h_def - a_def)

    model_p = market_prob
    if clf is not None:
        names = feature_names or ["logit_mkt", "diff_pass", "diff_rush"]
        row = dict(feature_row)
        row["logit_mkt"] = logit(market_prob)
        row["diff_pass"] = row.get("diff_pass", 0.0) + adj_net_pass
        row["diff_rush"] = row.get("diff_rush", 0.0) + adj_net_rush
        x = pd.DataFrame([[row.get(n, 0.0) for n in names]], columns=names)
        model_raw = float(clf.predict_proba(x)[0, 1])
        model_p = market_weight * market_prob + (1 - market_weight) * model_raw

    # Manual / contextual nudges are applied in LOG-ODDS space, scaled so each equals its
    # stated size at a 50% game (d logit = 4 x d prob there). Adding raw probability points
    # manufactured huge fake EV on longshots: +3 pts on a 12% dog is a 24% relative bump.
    news_p = (sliders.get("news", 0) / 10.0) * NEWS_MAX_SWING
    rest_p = (sliders.get("rh", 7) - sliders.get("ra", 7)) * REST_PER_DAY * (sliders.get("wr", 0) / 2.0)
    hfa_p = (sliders.get("hfa", 5) - 5) * HFA_PER_TICK

    l0 = logit(model_p)
    p_news = inv_logit(l0 + 4 * news_p)
    p_rest = inv_logit(l0 + 4 * (news_p + rest_p))
    p_hfa = inv_logit(l0 + 4 * (news_p + rest_p + hfa_p))
    ww = sliders.get("ww", 0)
    l_final = (l0 + 4 * (news_p + rest_p + hfa_p)) * (1.0 - WEATHER_MAX_SHRINK * ww / 10.0)
    final = min(max(inv_logit(l_final), 0.01), 0.99)

    return final, {
        "AI Model (Stats)": model_p,
        "News / Injury": p_news - model_p,
        "Rest Advantage": p_rest - p_news,
        "Home Field (excess)": p_hfa - p_rest,
        "Weather (randomness)": final - p_hfa,
    }


# ---------------------------------------------------------------------------
# Decision + sizing
# ---------------------------------------------------------------------------

def decide(final_p: float, d_home: float, d_away: float, min_ev: float = 0.03) -> Optional[dict]:
    """Pick the +EV side (if any) above a minimum EV buffer. Returns None for no bet."""
    ev_h = ev_per_dollar(final_p, d_home)
    ev_a = ev_per_dollar(1 - final_p, d_away)
    best = max((ev_h, "home", final_p, d_home), (ev_a, "away", 1 - final_p, d_away))
    ev, side, p, d = best
    if ev < min_ev:
        return None
    return {"side": side, "ev": ev, "p": p, "d": d}


def stake_for(p: float, d: float, bankroll: float, risk_mult: float = 0.5, cap: float = 0.05) -> float:
    """Half-Kelly x risk multiplier, capped at `cap` of bankroll (after the multiplier)."""
    frac = half_kelly_fraction(p, d, cap=cap) * risk_mult
    return min(frac, cap * risk_mult) * bankroll


# ---------------------------------------------------------------------------
# Player props
# ---------------------------------------------------------------------------

def suggest_prop_line(recent_yards) -> float:
    """Median of recent games, rounded to the nearest 5 then +0.5 (book style)."""
    vals = [float(v) for v in recent_yards if pd.notnull(v)]
    if not vals:
        return 0.5
    med = float(np.median(vals))
    return round(med / 5) * 5 + 0.5


def prop_probability(recent_yards, line: float, prior_n: float = 4.0) -> float:
    """P(over) = hit rate in recent games, shrunk toward 50% with `prior_n` pseudo-games.
    Honest by construction: a line at the median gives ~50%, not a hard-coded 57%."""
    vals = [float(v) for v in recent_yards if pd.notnull(v)]
    hits = sum(1 for v in vals if v > line)
    return (hits + 0.5 * prior_n) / (len(vals) + prior_n)


# ---------------------------------------------------------------------------
# Bet log grading (result, P&L, closing-line value)
# ---------------------------------------------------------------------------

LOG_COLUMNS = [
    "timestamp", "game_id", "season", "away", "home", "market", "side", "book",
    "odds_am", "dec", "p_mkt", "p_model", "ev", "stake", "result", "pnl", "close_am", "clv_pct",
]


def grade_log(log: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Fill result / pnl / close_am / clv_pct for logged ML bets whose game has a final score.
    CLV = no-vig closing probability of your side minus the implied probability you paid.
    Positive CLV means you beat the close, the best leading indicator of real edge."""
    if log.empty:
        return log
    out = log.copy()
    s = sched.set_index("game_id") if "game_id" in sched.columns else pd.DataFrame()
    for i, r in out.iterrows():
        if r.get("market") != "ML" or r["game_id"] not in s.index:
            continue
        g = s.loc[r["game_id"]]
        if pd.isnull(g.get("home_score")) or pd.isnull(g.get("away_score")):
            continue
        side_is_home = r["side"] == r["home"]
        hs, as_ = float(g["home_score"]), float(g["away_score"])
        if hs == as_:
            result, pnl = "P", 0.0
        else:
            won = (hs > as_) == side_is_home
            result = "W" if won else "L"
            pnl = r["stake"] * (r["dec"] - 1) if won else -r["stake"]
        out.at[i, "result"], out.at[i, "pnl"] = result, round(pnl, 2)
        hm, am = g.get("home_moneyline"), g.get("away_moneyline")
        if pd.notnull(hm) and pd.notnull(am):
            ph, pa = no_vig_two_way(american_to_decimal(hm), american_to_decimal(am))
            p_close = ph if side_is_home else pa
            out.at[i, "close_am"] = hm if side_is_home else am
            out.at[i, "clv_pct"] = round(p_close - 1 / r["dec"], 4)
    return out


def log_summary(log: pd.DataFrame) -> dict:
    graded = log[log["result"].isin(["W", "L", "P"])] if "result" in log.columns else log.iloc[0:0]
    staked = float(graded["stake"].sum()) if len(graded) else 0.0
    pnl = float(graded["pnl"].sum()) if len(graded) else 0.0
    clv = graded["clv_pct"].dropna() if "clv_pct" in graded.columns else pd.Series(dtype=float)
    return {
        "bets": int(len(log)),
        "graded": int(len(graded)),
        "wins": int((graded["result"] == "W").sum()) if len(graded) else 0,
        "losses": int((graded["result"] == "L").sum()) if len(graded) else 0,
        "pnl": round(pnl, 2),
        "roi": round(pnl / staked, 4) if staked else 0.0,
        "avg_clv": round(float(clv.mean()), 4) if len(clv) else None,
        "beat_close_share": round(float((clv > 0).mean()), 3) if len(clv) else None,
    }
