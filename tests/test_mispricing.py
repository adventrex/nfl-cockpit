import numpy as np
import pandas as pd
import pytest
import mispricing as M

def sched(n=400, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        season = 2005 + i % 20
        spread = float(rng.choice([-7, -3.5, -1, 0, 2.5, 4, 7, 10]))
        p_home = 0.5 + spread * 0.025
        p_home = min(max(p_home, 0.1), 0.9)
        hm = -int(100 * p_home / (1 - p_home)) if p_home >= 0.5 else int(100 * (1 - p_home) / p_home)
        am = -int(100 * (1 - p_home) / p_home) if p_home < 0.5 else int(100 * p_home / (1 - p_home))
        margin = rng.normal(spread, 13)
        total = 45.0
        pts = rng.normal(total, 10)
        rows.append({"game_id": f"g{i}", "season": season, "week": 1 + i % 18, "game_type": "REG", "home_team": "H", "away_team": "A",
                     "spread_line": spread, "total_line": total, "home_moneyline": hm, "away_moneyline": am,
                     "home_rest": 7, "away_rest": 7 if i % 5 else 14, "div_game": i % 2, "roof": "outdoors", "temp": 30 if i % 3 == 0 else 60,
                     "wind": 5, "weekday": "Sunday", "gametime": "13:00",
                     "home_score": round((pts + margin) / 2), "away_score": round((pts - margin) / 2)})
    return pd.DataFrame(rows)

def test_normalise_and_scoring_shapes():
    g = M.normalise(sched())
    assert {"p_home", "dog", "fav", "primetime", "divisional"} <= set(g.columns)
    r = M.score(g[g.dog == "home"], "ML", g[g.dog == "home"].dog)
    assert r["n"] > 50 and 0 < r["win_rate"] < 1 and "p" in r
    r2 = M.score(g, "ATS", pd.Series("home", index=g.index))
    assert abs(r2["expected_rate"] - 0.5) < 1e-9

def test_bh_fdr():
    assert M.bh_fdr([0.001, 0.02, 0.5, 0.9], q=0.10) == [True, True, False, False]
    assert M.bh_fdr([0.2, 0.3], q=0.10) == [False, False]

def test_no_false_positives_on_efficient_fake_market():
    """Outcomes are drawn FROM the spread, so an honest scan must validate nothing."""
    df = M.run_hypotheses(M.normalise(sched(n=1500)))
    assert df.validated.sum() == 0
    assert set(df.columns) >= {"hypothesis", "market", "rule", "disc_p", "val_roi", "fdr_pass", "validated"}

def test_calibration_buckets_run():
    cal = M.calibration_buckets(M.normalise(sched()))
    assert len(cal) >= 3 and (cal.n > 0).all()
