"""
hedge_engine.py — hedging, locking, and middling math for open bets.

All functions are pure. "orig" is the bet you already hold, "hedge" is the
opposite side you are considering now. Odds are DECIMAL throughout.

Profit conventions (relative to doing nothing further):
  if the original bet wins : stake*(d_orig-1) - h
  if the hedge bet wins    : h*(d_hedge-1) - stake
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from engine import norm_cdf

NFL_MARGIN_SD = 13.5  # historical std-dev of NFL final margin vs the spread


@dataclass
class HedgeOption:
    name: str
    hedge_stake: float
    profit_if_orig_wins: float
    profit_if_hedge_wins: float
    ev: float            # expected profit given p_orig (your model)
    worst_case: float
    note: str = ""


def hedge_outcomes(stake: float, d_orig: float, h: float, d_hedge: float) -> Tuple[float, float]:
    return stake * (d_orig - 1) - h, h * (d_hedge - 1) - stake


def hedge_ev(p_orig: float, stake: float, d_orig: float, h: float, d_hedge: float) -> float:
    a, b = hedge_outcomes(stake, d_orig, h, d_hedge)
    return p_orig * a + (1 - p_orig) * b


def hedge_equal_profit(stake: float, d_orig: float, d_hedge: float) -> float:
    """Hedge stake that makes profit identical whichever side wins (a full 'lock')."""
    return stake * d_orig / d_hedge


def locked_profit(stake: float, d_orig: float, d_hedge: float) -> float:
    h = hedge_equal_profit(stake, d_orig, d_hedge)
    return stake * (d_orig - 1) - h


def kelly_optimal_hedge(stake: float, d_orig: float, p_orig: float, d_hedge: float, bankroll: float) -> float:
    """
    Hedge stake h that maximises E[log(wealth)] given YOUR probability p_orig that
    the original bet wins. Closed form from d/dh [p ln W_A + (1-p) ln W_B] = 0:

        h* = [ (1-p)(d_h-1)(B + s(d_o-1)) - p(B - s) ] / (d_h - 1)

    Clamped to [0, full lock]. h* = 0 means "your model says don't pay for insurance".
    """
    if d_hedge <= 1.0:
        return 0.0
    bh = d_hedge - 1.0
    h = ((1 - p_orig) * bh * (bankroll + stake * (d_orig - 1)) - p_orig * (bankroll - stake)) / bh
    return float(min(max(h, 0.0), hedge_equal_profit(stake, d_orig, d_hedge)))


def hedge_options(stake: float, d_orig: float, p_orig: float, d_hedge: float, bankroll: float) -> List[HedgeOption]:
    """The menu shown in the Hedge Desk: no hedge, Kelly-optimal, half lock, full lock."""
    h_lock = hedge_equal_profit(stake, d_orig, d_hedge)
    h_kelly = kelly_optimal_hedge(stake, d_orig, p_orig, d_hedge, bankroll)
    menu = [
        ("Let it ride", 0.0, "Max EV if your model prob is right; max variance."),
        ("Kelly-optimal", h_kelly, "Best long-run growth given your model prob and bankroll."),
        ("Half lock", 0.5 * h_lock, "Keeps half the upside, halves the downside."),
        ("Full lock", h_lock, "Same profit either way. Pays the book's vig for certainty."),
    ]
    out = []
    for name, h, note in menu:
        a, b = hedge_outcomes(stake, d_orig, h, d_hedge)
        out.append(HedgeOption(name, round(h, 2), round(a, 2), round(b, 2),
                               round(p_orig * a + (1 - p_orig) * b, 2), round(min(a, b), 2), note))
    return out


def hedge_cost_of_certainty(p_orig: float, stake: float, d_orig: float, d_hedge: float) -> float:
    """EV given up by fully locking vs letting it ride (positive = the lock costs you EV)."""
    return hedge_ev(p_orig, stake, d_orig, 0.0, d_hedge) - hedge_ev(
        p_orig, stake, d_orig, hedge_equal_profit(stake, d_orig, d_hedge), d_hedge)


# ---------------------------------------------------------------------------
# Spreads: middles
# ---------------------------------------------------------------------------

def middle_window(spread_on_x: float, spread_on_y: float) -> Optional[Tuple[float, float]]:
    """
    Original ticket: team X at `spread_on_x` (e.g. -3). Hedge ticket: opponent Y at
    `spread_on_y` (e.g. +6). Returns the (lo, hi) range of X's winning margin where
    BOTH tickets cash, or None if no middle exists.
    X covers  iff margin > -spread_on_x ;  Y covers iff margin < spread_on_y.
    """
    lo, hi = -spread_on_x, spread_on_y
    return (lo, hi) if hi > lo else None


def middle_probability(window: Tuple[float, float], current_spread_on_x: float, sd: float = NFL_MARGIN_SD) -> float:
    """P(margin lands inside the middle) with margin ~ N(-current_spread_on_x, sd),
    continuity-corrected for whole-number lines (ties push, they don't middle)."""
    lo, hi = window
    lo_eff = lo + 0.5 if float(lo).is_integer() else lo
    hi_eff = hi - 0.5 if float(hi).is_integer() else hi
    if hi_eff <= lo_eff:
        return 0.0
    mu = -current_spread_on_x
    return norm_cdf((hi_eff - mu) / sd) - norm_cdf((lo_eff - mu) / sd)


def middle_ev(stake_x: float, d_x: float, stake_y: float, d_y: float, p_middle: float, p_x_only: float, p_y_only: float) -> float:
    """Expected profit of holding both spread tickets. Remaining prob mass = pushes (0)."""
    both = stake_x * (d_x - 1) + stake_y * (d_y - 1)
    x_only = stake_x * (d_x - 1) - stake_y
    y_only = stake_y * (d_y - 1) - stake_x
    return p_middle * both + p_x_only * x_only + p_y_only * y_only


# ---------------------------------------------------------------------------
# Parlays: last-leg hedge is just a lock against the parlay's payout
# ---------------------------------------------------------------------------

def parlay_last_leg_options(parlay_stake: float, parlay_decimal: float, p_last_leg: float,
                            d_hedge: float, bankroll: float) -> List[HedgeOption]:
    """With every leg but one already won, the parlay behaves like a single bet
    at `parlay_decimal`. Hedge the opposite side of the final leg."""
    return hedge_options(parlay_stake, parlay_decimal, p_last_leg, d_hedge, bankroll)


def middle_breakdown(window: Tuple[float, float], current_spread_on_x: float, sd: float = NFL_MARGIN_SD) -> Tuple[float, float, float]:
    """(p_both_cash, p_only_X_cashes, p_only_Y_cashes) for a middle; leftover mass is pushes."""
    lo, hi = window
    lo_eff = lo + 0.5 if float(lo).is_integer() else lo
    hi_eff = hi - 0.5 if float(hi).is_integer() else hi
    mu = -current_spread_on_x
    p_both = max(0.0, norm_cdf((hi_eff - mu) / sd) - norm_cdf((lo_eff - mu) / sd))
    p_x_only = 1.0 - norm_cdf((hi_eff + (1.0 if float(hi).is_integer() else 0.0) - mu) / sd)
    p_y_only = norm_cdf((lo_eff - (1.0 if float(lo).is_integer() else 0.0) - mu) / sd)
    return p_both, p_x_only, p_y_only
