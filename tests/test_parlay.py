import pytest
from parlay_engine import PropLeg, ParlayMath

def leg(i, cat, team, p=0.55, d=1.91):
    return PropLeg(f"{i}", f"{i}", d, p, cat, team, "")

def test_joint_prob_bounded_by_smallest_leg():
    legs = [leg(1, "Passing", "KC", 0.6), leg(2, "Receiving", "KC", 0.55)]
    j = ParlayMath.joint_probability(legs)
    assert 0.6 * 0.55 < j <= 0.55           # correlation lifts it above independence but never above a leg

def test_negative_correlation_lowers_joint():
    same = ParlayMath.joint_probability([leg(1, "Team Win", "KC", 0.6), leg(2, "Rushing", "KC", 0.5)])
    opp = ParlayMath.joint_probability([leg(1, "Team Win", "KC", 0.6), leg(2, "Rushing", "LAC", 0.5)])
    assert opp < 0.6 * 0.5 < same

def test_both_moneylines_are_mutually_exclusive():
    j = ParlayMath.joint_probability([leg(1, "Team Win", "KC", 0.6), leg(2, "Team Win", "LAC", 0.4)])
    assert j == 0.0
    adds = ParlayMath.find_best_additions([leg(1, "Team Win", "KC", 0.6)], [leg(2, "Team Win", "LAC", 0.4), leg(3, "Passing", "KC")])
    assert all(a.category != "Team Win" for a in adds)

def test_ev_is_against_book_price_not_tautological():
    legs = [leg(1, "Passing", "KC", 0.6), leg(2, "Receiving", "KC", 0.6)]
    with pytest.raises(ValueError):
        ParlayMath.calculate_ticket(legs, 1000, book_decimal=None)   # no estimate path: a book quote is required
    good = ParlayMath.calculate_ticket(legs, 1000, book_decimal=4.0)
    bad = ParlayMath.calculate_ticket(legs, 1000, book_decimal=1.5)
    assert good.ev > 0 > bad.ev and good.kelly_stake > 0 == bad.kelly_stake
    assert good.kelly_stake <= 0.02 * 1000

def test_single_leg_estimate_is_its_own_price():
    r = ParlayMath.calculate_ticket([leg(1, "Team Win", "KC", 0.6, 1.5)], 1000, book_decimal=1.5)
    assert r.final_odds == pytest.approx(1.5) and r.ev == pytest.approx(0.6 * 1.5 - 1)
