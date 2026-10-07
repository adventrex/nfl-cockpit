"""
promo_engine.py — turn sportsbook promotions into locked profit ("matched betting").

Every plan is two bets: the PROMO bet at the promo book (DraftKings) and a HEDGE on the
opposite side at another book, sized so the profit is the same whichever side wins.
Profit is then locked at placement — the only risks are operational (line moves between
the two clicks, a void, limits, or the book restricting the account).

Decimal odds throughout. `d_bet` = promo-book price, `d_hedge` = other book's price on the
opposite side of the SAME line (mirrored spread / same total / the other moneyline).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd

from engine import decimal_to_american

PROMO_TYPES = {
    "bonus_bet": "Bonus bet (stake not returned)",
    "no_sweat": "No-sweat / risk-free bet (refund as bonus bet)",
    "profit_boost": "Profit boost (+X% on winnings)",
    "odds_boost": "Odds boost (book gives boosted price)",
    "qualifying": "Qualifying bet (bet $X to unlock $Y bonus)",
}
DEFAULT_REFUND_CONVERSION = 0.70   # a bonus-bet refund is worth ~70% of face when hedged well


@dataclass
class HedgePlan:
    promo_type: str
    game: str
    market: str
    bet_side: str
    bet_point: Optional[float]
    bet_book: str
    bet_stake: float
    bet_dec: float
    hedge_side: str
    hedge_book: str
    hedge_stake: float
    hedge_dec: float
    profit_if_bet_wins: float
    profit_if_hedge_wins: float
    locked_profit: float
    conversion: float          # locked profit / promo face value
    note: str = ""

    def ticket(self) -> str:
        pt = lambda p: f" {p:+g}" if p is not None and self.market == "spreads" else (f" {p:g}" if p is not None else "")
        return (f"[{self.game} {self.market}] 1) {self.bet_book}: ${self.bet_stake:.2f} on {self.bet_side}{pt(self.bet_point)} @ {decimal_to_american(self.bet_dec):+d}  "
                f"2) {self.hedge_book}: ${self.hedge_stake:.2f} on {self.hedge_side}{pt(-self.bet_point if self.bet_point is not None and self.market == 'spreads' else self.bet_point)} @ {decimal_to_american(self.hedge_dec):+d}  "
                f"→ ${self.profit_if_bet_wins:.2f} / ${self.profit_if_hedge_wins:.2f}")


# ---------------------------------------------------------------- core formulas (equal-profit hedges)

def bonus_bet(bonus: float, d_bet: float, d_hedge: float):
    """Free bet, stake not returned. Best at LONG odds (conversion rises with d_bet)."""
    h = bonus * (d_bet - 1) / d_hedge
    return h, bonus * (d_bet - 1) - h, h * (d_hedge - 1)


def no_sweat(stake: float, d_bet: float, d_hedge: float, refund_conv: float = DEFAULT_REFUND_CONVERSION):
    """Cash stake; if it loses you get it back as a bonus bet (valued at refund_conv × stake)."""
    h = stake * (d_bet - refund_conv) / d_hedge
    win = stake * (d_bet - 1) - h
    lose = -stake + h * (d_hedge - 1) + refund_conv * stake
    return h, win, lose


def profit_boost(stake: float, d_bet: float, boost_pct: float, d_hedge: float, max_win: Optional[float] = None):
    """Winnings multiplied by (1 + boost). Best on SHORT-ish odds where the cap doesn't bind."""
    w = stake * (d_bet - 1) * (1 + boost_pct)
    if max_win is not None:
        w = min(w, max_win)
    h = (stake + w) / d_hedge
    return h, w - h, h * (d_hedge - 1) - stake


def odds_boost(stake: float, d_boosted: float, d_hedge: float):
    w = stake * (d_boosted - 1)
    h = (stake + w) / d_hedge
    return h, w - h, h * (d_hedge - 1) - stake


def qualifying(stake: float, d_bet: float, d_hedge: float, bonus_value: float):
    """Bet $stake (cash) to unlock a bonus worth `bonus_value` (already discounted by conversion)."""
    h = stake * d_bet / d_hedge
    return h, stake * (d_bet - 1) - h + bonus_value, h * (d_hedge - 1) - stake + bonus_value


def plan_for(promo: dict, d_bet: float, d_hedge: float):
    """Dispatch. promo = {type, amount, boost_pct?, max_win?, refund_conv?, bonus_value?}. Returns (h, pA, pB, note)."""
    t, amt = promo["type"], float(promo["amount"])
    if t == "bonus_bet":
        h, a, b = bonus_bet(amt, d_bet, d_hedge); note = "stake not returned"
    elif t == "no_sweat":
        h, a, b = no_sweat(amt, d_bet, d_hedge, promo.get("refund_conv", DEFAULT_REFUND_CONVERSION)); note = f"lose-side includes refund valued at {promo.get('refund_conv', DEFAULT_REFUND_CONVERSION):.0%}"
    elif t == "profit_boost":
        h, a, b = profit_boost(amt, d_bet, promo.get("boost_pct", 0.0), d_hedge, promo.get("max_win")); note = f"{promo.get('boost_pct', 0):.0%} boost"
    elif t == "odds_boost":
        h, a, b = odds_boost(amt, promo.get("boosted_dec", d_bet), d_hedge); note = "boosted price"
    elif t == "qualifying":
        h, a, b = qualifying(amt, d_bet, d_hedge, promo.get("bonus_value", 0.0)); note = f"unlocks bonus worth ${promo.get('bonus_value', 0):.0f}"
    else:
        raise ValueError(t)
    return h, a, b, note


# ---------------------------------------------------------------- search live quotes

def best_plans(promo: dict, quotes: pd.DataFrame, promo_book: str = "draftkings", min_dec: float = 1.0,
               max_dec: float = 100.0, top_n: int = 8, hedge_books: Optional[set] = None) -> List[HedgePlan]:
    """
    quotes: scanner.parse_payload output (game, market, book, side, point, dec).
    For every promo-book quote, find the best price on the opposite side of the same line at
    an allowed hedge venue (default: any other book; pass hedge_books to restrict, and include
    promo_book itself to allow hedging at the same book — needed where it's the only legal app).
    """
    if quotes is None or quotes.empty:
        return []
    q = quotes.copy()
    q["opp_side"] = q.apply(lambda r: {"home": "away", "away": "home", "over": "under", "under": "over"}.get(r["side"]), axis=1)
    q["opp_point"] = q.apply(lambda r: (-r["point"] if r["market"] == "spreads" else r["point"]) if pd.notna(r["point"]) else None, axis=1)
    mine = q[(q.book == promo_book) & (q.dec >= min_dec) & (q.dec <= max_dec)]
    others = q[q.book.isin(hedge_books)] if hedge_books else q[q.book != promo_book]
    plans = []
    for r in mine.itertuples():
        cand = others[(others.game == r.game) & (others.market == r.market) & (others.side == r.opp_side)]
        if r.market != "h2h":
            cand = cand[cand.point.notna() & (cand.point.astype(float) == float(r.opp_point))]
        if cand.empty:
            continue
        best = cand.loc[cand.dec.idxmax()]
        d_bet = promo.get("boosted_dec", r.dec) if promo["type"] == "odds_boost" else r.dec
        h, a, b, note = plan_for(promo, d_bet, best.dec)
        locked = min(a, b)
        plans.append(HedgePlan(promo["type"], r.game, r.market, r.side, r.point if pd.notna(r.point) else None, promo_book,
                               round(float(promo["amount"]), 2), round(d_bet, 3), best.side, best.book, round(h, 2), round(best.dec, 3),
                               round(a, 2), round(b, 2), round(locked, 2), round(locked / float(promo["amount"]), 3), note))
    plans.sort(key=lambda p: p.locked_profit, reverse=True)
    return plans[:top_n]


def plans_table(plans: List[HedgePlan]) -> pd.DataFrame:
    return pd.DataFrame([asdict(p) | {"ticket": p.ticket()} for p in plans])
