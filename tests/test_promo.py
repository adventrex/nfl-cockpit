import pandas as pd
import pytest
import promo_engine as P
from engine import american_to_decimal as A


def test_bonus_bet_lock_is_equal_and_conversion_rises_with_odds():
    h, a, b = P.bonus_bet(50, A(+300), A(-350))
    assert a == pytest.approx(b, abs=1e-9) and a > 0
    _, short, _ = P.bonus_bet(50, A(-110), A(-110))
    _, long_, _ = P.bonus_bet(50, A(+400), A(-500))
    assert long_ > short                       # long shots convert bonus bets better
    assert 0.5 < long_ / 50 < 0.9              # typical 60-80% conversion

def test_no_sweat_equal_profit_with_refund_valued():
    h, win, lose = P.no_sweat(100, A(+350), A(-450), refund_conv=0.7)
    assert win == pytest.approx(lose, abs=1e-9)
    assert win > 0                              # long-shot no-sweat bets are +EV when hedged

def test_profit_boost_needs_boost_to_beat_vig():
    h, a, b = P.profit_boost(50, A(-110), 0.0, A(-110))
    assert a == pytest.approx(b) and a < 0      # no boost: you just pay the hold twice
    h, a, b = P.profit_boost(50, A(-110), 0.50, A(-110))
    assert a == pytest.approx(b) and a > 0      # 50% boost beats it
    _, capped, _ = P.profit_boost(500, A(+200), 1.0, A(-250), max_win=100)
    assert capped < 100                         # the cap binds, hedge still equalises

def test_qualifying_bet_cost_is_roughly_the_hold():
    h, a, b = P.qualifying(5, A(-110), A(-110), bonus_value=150 * 0.7)
    assert a == pytest.approx(b) and 100 < a < 110   # ~$105 locked on a "bet $5 get $150"

def test_best_plans_hedges_at_a_different_book_on_the_same_line():
    q = pd.DataFrame([
        {"game": "DET@BUF", "market": "h2h", "book": "draftkings", "side": "away", "point": None, "dec": 2.9},
        {"game": "DET@BUF", "market": "h2h", "book": "fanduel", "side": "home", "point": None, "dec": 1.45},
        {"game": "DET@BUF", "market": "h2h", "book": "betmgm", "side": "home", "point": None, "dec": 1.42},
        {"game": "DET@BUF", "market": "h2h", "book": "draftkings", "side": "home", "point": None, "dec": 1.42},
        {"game": "DET@BUF", "market": "spreads", "book": "draftkings", "side": "away", "point": 5.5, "dec": 1.91},
        {"game": "DET@BUF", "market": "spreads", "book": "fanduel", "side": "home", "point": -5.5, "dec": 1.95},
        {"game": "DET@BUF", "market": "spreads", "book": "betmgm", "side": "home", "point": -3.5, "dec": 1.91},   # different line: must NOT be used
    ])
    plans = P.best_plans({"type": "bonus_bet", "amount": 50}, q)
    assert plans and plans[0].market == "h2h" and plans[0].bet_side == "away" and plans[0].hedge_book == "fanduel"
    assert plans[0].profit_if_bet_wins == pytest.approx(plans[0].profit_if_hedge_wins, abs=0.01)
    sp = [p for p in plans if p.market == "spreads"][0]
    assert sp.hedge_book == "fanduel" and sp.hedge_dec == 1.95
    assert "draftkings" in plans[0].ticket() and "fanduel" in plans[0].ticket()
    assert P.best_plans({"type": "bonus_bet", "amount": 50}, pd.DataFrame()) == []


def test_hedge_venue_allowlist_and_same_book():
    q = pd.DataFrame([
        {"game": "DET@BUF", "market": "h2h", "book": "draftkings", "side": "away", "point": None, "dec": 2.9},
        {"game": "DET@BUF", "market": "h2h", "book": "draftkings", "side": "home", "point": None, "dec": 1.40},
        {"game": "DET@BUF", "market": "h2h", "book": "betus", "side": "home", "point": None, "dec": 1.50},
    ])
    off = P.best_plans({"type": "bonus_bet", "amount": 50}, q)
    assert off[0].hedge_book == "betus"                                   # unrestricted: best price wins
    same = P.best_plans({"type": "bonus_bet", "amount": 50}, q, hedge_books={"draftkings"})
    assert same[0].hedge_book == "draftkings" and same[0].locked_profit > 0   # Oregon mode still locks a profit
    assert P.best_plans({"type": "bonus_bet", "amount": 50}, q, hedge_books={"fanduel"}) == []
    assert same[0].ticket().startswith("[DET@BUF h2h]")
