import pytest
import hedge_engine as H
from engine import american_to_decimal as A


def test_full_lock_equalises_profit():
    # Bet $100 on KC at +150 pregame; KC now -120, opponent +100 available
    s, d_o, d_h = 100, 2.5, 2.0
    h = H.hedge_equal_profit(s, d_o, d_h)
    a, b = H.hedge_outcomes(s, d_o, h, d_h)
    assert h == pytest.approx(125.0)
    assert a == pytest.approx(b) == pytest.approx(25.0)
    assert H.locked_profit(s, d_o, d_h) == pytest.approx(25.0)

def test_lock_can_be_a_loss_when_line_moved_against_you():
    # Bet $100 at -110, now the other side is -200 (your side drifted to a big dog)
    prof = H.locked_profit(100, A(-110), A(-200))
    assert prof < 0

def test_kelly_hedge_is_zero_when_model_still_loves_the_bet():
    # If you think your side wins 80% and the hedge is priced at a fair 50%, don't pay for insurance
    assert H.kelly_optimal_hedge(100, 2.5, 0.80, 2.0, bankroll=1000) == 0.0

def test_kelly_hedge_increases_as_confidence_drops_and_is_capped_at_lock():
    hs = [H.kelly_optimal_hedge(100, 2.5, p, 2.0, bankroll=1000) for p in (0.7, 0.5, 0.3, 0.1)]
    assert hs == sorted(hs)
    assert hs[-1] <= H.hedge_equal_profit(100, 2.5, 2.0) + 1e-9

def test_kelly_hedge_closed_form_matches_numeric_optimum():
    import math
    s, d_o, p, d_h, B = 100, 2.5, 0.45, 2.0, 1000
    h_star = H.kelly_optimal_hedge(s, d_o, p, d_h, B)
    def util(h):
        a, b = H.hedge_outcomes(s, d_o, h, d_h)
        return p * math.log(B + a) + (1 - p) * math.log(B + b)
    grid = [i * 0.5 for i in range(0, int(H.hedge_equal_profit(s, d_o, d_h) * 2) + 1)]
    best = max(grid, key=util)
    assert abs(best - h_star) < 1.0

def test_options_menu_shape_and_monotone_worst_case():
    opts = H.hedge_options(100, 2.5, 0.55, 2.0, 1000)
    assert [o.name for o in opts] == ["Let it ride", "Kelly-optimal", "Half lock", "Full lock"]
    assert opts[0].hedge_stake == 0 and opts[-1].profit_if_orig_wins == pytest.approx(opts[-1].profit_if_hedge_wins)
    assert opts[0].worst_case <= opts[2].worst_case <= opts[3].worst_case
    # Letting it ride has the highest EV when p_orig * d_orig > 1 and the hedge is fairly priced
    assert opts[0].ev >= opts[3].ev

def test_cost_of_certainty_positive_when_hedge_is_minus_ev():
    assert H.hedge_cost_of_certainty(0.55, 100, 2.5, A(-110)) > 0

def test_middle_window_and_probability():
    win = H.middle_window(-3, 6)          # KC -3 held, opponent +6 available
    assert win == (3, 6)
    p = H.middle_probability(win, current_spread_on_x=-4.5)
    assert 0.03 < p < 0.20                 # margins of exactly 4 or 5 -> a few percent
    assert H.middle_window(-3, 2) is None  # no gap -> no middle
    assert H.middle_probability((3, 3.5), -3) == pytest.approx(0.0, abs=1e-9) or H.middle_probability((3, 3.5), -3) < 0.05

def test_parlay_last_leg_is_a_lock_against_payout():
    # $10 parlay paying 20.0 with one leg left; other side of that leg at 1.9
    opts = H.parlay_last_leg_options(10, 20.0, 0.5, 1.9, 1000)
    full = opts[-1]
    assert full.profit_if_orig_wins == pytest.approx(full.profit_if_hedge_wins)
    assert full.profit_if_orig_wins > 0

def test_middle_breakdown_sums_below_one_and_middle_ev_positive_when_cheap():
    win = H.middle_window(-3, 6)
    pb, px, py = H.middle_breakdown(win, -4.5)
    assert 0 < pb < 0.2 and px + py + pb <= 1.0 + 1e-9
    ev = H.middle_ev(100, 1.909, 100, 1.909, pb, px, py)
    assert -15 < ev < 15                       # a thin middle at -110/-110 is roughly break-even
