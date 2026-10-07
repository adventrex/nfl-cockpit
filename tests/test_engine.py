import math
import pandas as pd
import pytest
import engine as E


# --- odds math -------------------------------------------------------------
def test_american_decimal_roundtrip():
    assert E.american_to_decimal(-110) == pytest.approx(1.9091, abs=1e-4)
    assert E.american_to_decimal(150) == 2.5
    assert E.decimal_to_american(2.5) == 150
    assert E.decimal_to_american(1.9091) == -110
    assert E.american_to_decimal("garbage") == 2.0

def test_no_vig_sums_to_one_and_strips_hold():
    p1, p2 = E.no_vig_two_way(1.9091, 1.9091)
    assert p1 == pytest.approx(0.5) and p1 + p2 == pytest.approx(1.0)
    ph, pa = E.no_vig_two_way(E.american_to_decimal(-238), E.american_to_decimal(195))  # DET@BUF tonight
    assert 0.66 < ph < 0.69

def test_synthetic_hold():
    assert E.synthetic_hold(1.9091, 1.9091) == pytest.approx(0.0476, abs=1e-3)   # -110/-110 = 4.76%
    assert E.synthetic_hold(2.05, 2.05) < 0                                       # arb

def test_kelly_worked_example_from_spec_section_13():
    # Spec §13 claims full Kelly ~10% here. Correct arithmetic: (0.9091*0.55-0.45)/0.9091 = 5.5%.
    d = E.american_to_decimal(-110)
    assert E.ev_per_dollar(0.55, d) == pytest.approx(0.05, abs=1e-3)
    assert E.kelly_fraction(0.55, d) == pytest.approx(0.055, abs=1e-3)
    assert E.half_kelly_fraction(0.55, d, cap=0.05) == pytest.approx(0.0275, abs=1e-3)
    assert E.half_kelly_fraction(0.75, d, cap=0.05) == 0.05     # cap engages on a huge edge
    assert E.kelly_fraction(0.40, d) == 0.0

def test_z_score():
    assert E.z_score(0.55, E.american_to_decimal(-110), 0.02) == pytest.approx(1.31, abs=0.01)


# --- HFA: excess over league, not raw --------------------------------------
def _fake_games(n_per=12, team_a_home_wr=0.9, league_home_wr=0.55):
    rows = []
    teams = ["A", "B", "C", "D"]
    for t in teams:
        for opp in teams:
            if t == opp: continue
            for k in range(n_per // 3 + 1):
                wr = team_a_home_wr if t == "A" else league_home_wr
                home_win = 1 if (k / (n_per // 3 + 1)) < wr else 0
                rows.append({"home_team": t, "away_team": opp, "home_score": 20 + home_win, "away_score": 20})
    return pd.DataFrame(rows)

def test_hfa_is_excess_over_league_average():
    hfa = E.compute_team_home_advantage(_fake_games())
    assert hfa["A"] > 0.02                       # unusual home edge shows up (shrunk)
    assert abs(hfa["B"]) < 0.06                  # average teams sit near zero, not near the league HFA
    assert E.compute_team_home_advantage(pd.DataFrame(columns=["home_score", "away_score", "home_team", "away_team"])) == {}


# --- sliders don't double count --------------------------------------------
DEFAULTS = {"h_qb": 7, "h_pwr": 4, "h_def": 6, "a_qb": 5, "a_pwr": 5, "a_def": 5, "hfa": 5}

def _sliders(**over):
    s = {"ph_qb": 7, "ph_pwr": 4, "ph_def": 6, "pa_qb": 5, "pa_pwr": 5, "pa_def": 5,
         "news": 0, "ww": 0, "hfa": 5, "wr": 0, "rh": 7, "ra": 7}
    s.update(over); return s

def test_untouched_sliders_return_market_prob_when_no_model():
    p, br = E.calc_win_prob(0.62, {"diff_pass": 0.10, "diff_rush": 0.05}, _sliders(), DEFAULTS, clf=None)
    assert p == pytest.approx(0.62)
    assert all(abs(v) < 1e-9 for k, v in br.items() if k != "AI Model (Stats)")

def test_slider_roundtrip_is_zero_delta():
    assert E.slider_delta_epa(E.epa_to_slider(0.06), E.epa_to_slider(0.06)) == 0.0
    assert E.epa_to_slider(0.06) == 7 and E.epa_to_slider(0.06, reverse=True) == 3

def test_news_is_signed_and_deterministic():
    p_plus, _ = E.calc_win_prob(0.5, {}, _sliders(news=10), DEFAULTS)
    p_minus, _ = E.calc_win_prob(0.5, {}, _sliders(news=-10), DEFAULTS)
    assert p_plus == pytest.approx(0.55, abs=0.001) and p_minus == pytest.approx(0.45, abs=0.001)
    assert E.calc_win_prob(0.5, {}, _sliders(), DEFAULTS)[0] == E.calc_win_prob(0.5, {}, _sliders(), DEFAULTS)[0]

def test_weather_shrinks_toward_coinflip_never_flips_favorite():
    p, br = E.calc_win_prob(0.70, {}, _sliders(ww=10), DEFAULTS)
    assert 0.5 < p < 0.70 and br["Weather (randomness)"] < 0
    p_dog, _ = E.calc_win_prob(0.30, {}, _sliders(ww=10), DEFAULTS)
    assert 0.30 < p_dog < 0.5                    # symmetric: the dog gains what the fav loses

def test_hfa_slider_at_default_adds_nothing():
    p, br = E.calc_win_prob(0.55, {}, _sliders(hfa=5), DEFAULTS)
    assert br["Home Field (excess)"] == 0

def test_model_blend_uses_market_weight():
    class FakeClf:
        def predict_proba(self, x):
            import numpy as np; return np.array([[0.1, 0.9]])
    p, br = E.calc_win_prob(0.5, {}, _sliders(), DEFAULTS, clf=FakeClf(), market_weight=0.7)
    assert br["AI Model (Stats)"] == pytest.approx(0.7 * 0.5 + 0.3 * 0.9)


# --- decision --------------------------------------------------------------
def test_decide_picks_plus_ev_side_and_respects_buffer():
    dh, da = E.american_to_decimal(-110), E.american_to_decimal(-110)
    assert E.decide(0.5, dh, da) is None                          # coin flip at -110 is -EV both ways
    d = E.decide(0.58, dh, da, min_ev=0.03)
    assert d["side"] == "home" and d["ev"] > 0.03
    d = E.decide(0.40, dh, da, min_ev=0.03)
    assert d["side"] == "away"
    assert E.decide(0.53, dh, da, min_ev=0.03) is None            # +1.2% EV is inside the buffer

def test_stake_caps():
    assert E.stake_for(0.9, 2.0, 1000, risk_mult=0.5) == pytest.approx(25.0)   # 5% cap x 0.5
    assert E.stake_for(0.9, 2.0, 1000, risk_mult=1.0) == pytest.approx(50.0)
    assert E.stake_for(0.4, 2.0, 1000) == 0.0


# --- props -----------------------------------------------------------------
def test_prop_line_and_probability_are_honest():
    yds = [80, 95, 60, 110, 70, 100, 90, 85]
    line = E.suggest_prop_line(yds)
    assert line == 85.5 or line == 90.5
    assert 0.35 < E.prop_probability(yds, line) < 0.65       # a median line is ~a coin flip
    assert E.prop_probability(yds, 10.5) > 0.8                # obviously-low line, shrunk not 100%
    assert E.prop_probability([], 50.5) == 0.5


# --- bet log grading -------------------------------------------------------
def test_grade_log_results_pnl_and_clv():
    sched = pd.DataFrame([{"game_id": "g1", "home_score": 27, "away_score": 20, "home_moneyline": -200, "away_moneyline": 170},
                          {"game_id": "g2", "home_score": None, "away_score": None, "home_moneyline": -120, "away_moneyline": 100}])
    log = pd.DataFrame([
        {"game_id": "g1", "home": "BUF", "away": "DET", "market": "ML", "side": "BUF", "stake": 10.0, "dec": 1.5, "result": None, "pnl": None, "close_am": None, "clv_pct": None},
        {"game_id": "g1", "home": "BUF", "away": "DET", "market": "ML", "side": "DET", "stake": 10.0, "dec": 2.8, "result": None, "pnl": None, "close_am": None, "clv_pct": None},
        {"game_id": "g2", "home": "KC", "away": "LAC", "market": "ML", "side": "KC", "stake": 10.0, "dec": 1.8, "result": None, "pnl": None, "close_am": None, "clv_pct": None},
    ])
    g = E.grade_log(log, sched)
    assert list(g["result"]) == ["W", "L", None]
    assert g.loc[0, "pnl"] == 5.0 and g.loc[1, "pnl"] == -10.0
    # took BUF at 1.5 (66.7% implied); closed -200/+170 -> ~64.4% no-vig -> negative CLV
    assert g.loc[0, "clv_pct"] < 0
    # took DET at 2.8 (35.7%); close no-vig ~35.6% -> roughly flat
    assert abs(g.loc[1, "clv_pct"]) < 0.02
    s = E.log_summary(g)
    assert s["graded"] == 2 and s["wins"] == 1 and s["pnl"] == -5.0 and s["roi"] == pytest.approx(-0.25)


def test_slider_deltas_flow_into_feature_row():
    captured = {}
    class Spy:
        def predict_proba(self, x):
            captured.update(x.iloc[0].to_dict()); import numpy as np; return np.array([[0.5, 0.5]])
    E.calc_win_prob(0.5, {"diff_pass": 0.1, "diff_rush": 0.0, "diff_qb": 0.2}, _sliders(ph_qb=9), DEFAULTS, clf=Spy(),
                    feature_names=["logit_mkt", "diff_pass", "diff_rush", "diff_qb"])
    assert captured["diff_pass"] == pytest.approx(0.1 + 2 * E.EPA_PER_TICK) and captured["diff_qb"] == 0.2 and captured["logit_mkt"] == 0.0


def test_nudges_are_small_for_longshots():
    """v4.2 bug: +3 HFA pts and a weather shrink turned a 12.5% dog into 17.6% (+27% fake EV at +625)."""
    away_dog_home_p = 0.875
    p, _ = E.calc_win_prob(away_dog_home_p, {}, _sliders(hfa=3, ww=6), DEFAULTS)   # nudges favour the dog
    dog = 1 - p
    assert 0.125 < dog < 0.16                       # moves, but nothing like +5 points
    p50, _ = E.calc_win_prob(0.5, {}, _sliders(hfa=7), DEFAULTS)
    assert p50 == pytest.approx(0.53, abs=0.001)    # same slider is still worth its stated 3% at a coin flip
