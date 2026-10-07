"""
parlay_engine.py — same-game-parlay correlation and pricing.

v4 changes (see tests/test_parlay.py):
  * Joint probability uses the legs' MODEL probabilities and a bounded pairwise
    correlation formula, P(A∩B) = pA·pB + ρ·sqrt(pA(1-pA)pB(1-pB)), applied
    sequentially. It can never exceed the smallest leg probability (the v3
    "boost" multiplier could).
  * EV is computed against the BOOK's parlay price. The old code priced the
    parlay from its own probability and then applied 15% vig, so every ticket
    showed EV = -15% by construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class PropLeg:
    leg_id: str
    description: str
    decimal_odds: float   # book price for this leg on its own
    p_model: float        # our probability the leg hits
    category: str         # Team Win | Passing | Rushing | Receiving
    team: str
    recent_stat: str


@dataclass
class ParlayResult:
    legs: List[PropLeg]
    final_odds: float     # the book's decimal payout
    win_prob: float       # our joint probability
    ev: float
    kelly_stake: float


class ParlayMath:
    # (same-team corr, opposing-team corr)
    CORR = {
        frozenset(["Team Win", "Passing"]): (0.50, -0.20),
        frozenset(["Team Win", "Rushing"]): (0.22, -0.22),   # measured 2023-25: lead-RB TD vs team win, 1,863 games, corr 0.224
        frozenset(["Team Win", "Receiving"]): (0.45, -0.20),
        frozenset(["Passing", "Receiving"]): (0.65, 0.10),
        frozenset(["Passing", "Rushing"]): (0.05, 0.00),
        frozenset(["Rushing", "Rushing"]): (-0.15, 0.00),
        frozenset(["Team Win"]): (1.0, -1.0),  # both MLs of the same game are mutually exclusive
    }

    @staticmethod
    def get_correlation(leg1: PropLeg, leg2: PropLeg) -> float:
        if leg1.leg_id == leg2.leg_id:
            return 1.0
        vals = ParlayMath.CORR.get(frozenset([leg1.category, leg2.category]), (0.05, 0.05))
        return vals[0] if leg1.team == leg2.team else vals[1]

    @staticmethod
    def joint_probability(legs: List[PropLeg]) -> float:
        """Sequential pairwise-correlation build-up, bounded by Fréchet limits."""
        if not legs:
            return 0.0
        joint = legs[0].p_model
        for i in range(1, len(legs)):
            p = legs[i].p_model
            rho = sum(ParlayMath.get_correlation(legs[j], legs[i]) for j in range(i)) / i
            raw = joint * p + rho * math.sqrt(joint * (1 - joint) * p * (1 - p))
            joint = min(max(raw, 0.0), min(joint, p))
        return joint

    @staticmethod
    def naive_odds(legs: List[PropLeg]) -> float:
        prod = 1.0
        for leg in legs:
            prod *= leg.decimal_odds
        return prod

    @staticmethod
    def find_best_additions(current_legs: List[PropLeg], candidates: List[PropLeg], top_n: int = 3) -> List[PropLeg]:
        scored = []
        for cand in candidates:
            if any(l.leg_id == cand.leg_id for l in current_legs):
                continue
            if cand.category == "Team Win" and any(l.category == "Team Win" for l in current_legs):
                continue  # never offer the other moneyline
            avg_corr = sum(ParlayMath.get_correlation(l, cand) for l in current_legs) / len(current_legs)
            ev = cand.p_model * cand.decimal_odds - 1
            scored.append((avg_corr * 2.0 + ev, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_n]]

    @staticmethod
    def calculate_ticket(legs: List[PropLeg], bankroll: float, book_decimal: float,
                         kelly_mult: float = 0.2, cap: float = 0.02) -> ParlayResult:
        """EV against the BOOK's quoted parlay price. There is no estimate path: pricing a parlay from our own
        probability is how v3 manufactured fake EV, and every real ticket has a quote."""
        if book_decimal is None or book_decimal <= 1.0:
            raise ValueError("book_decimal is required: read the payout off the bet slip")
        joint = ParlayMath.joint_probability(legs)
        odds = float(book_decimal)
        ev = joint * odds - 1
        b = odds - 1
        f = max(0.0, (b * joint - (1 - joint)) / b) if b > 0 else 0.0
        stake = min(f * kelly_mult, cap) * bankroll
        return ParlayResult(legs, odds, joint, ev, stake)
