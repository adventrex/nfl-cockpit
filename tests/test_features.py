import numpy as np
import pandas as pd
import features as F


def fake_pbp():
    """Team A great passing offense for 3 weeks in 2024, then bad in 2025 wk1. Team B average. QB X moves A->B in 2025."""
    rows = []
    def game(season, week, gid, off, deff, epa, qb, n=12):
        for i in range(n):
            rows.append({"season": season, "week": week, "game_id": gid, "posteam": off, "defteam": deff, "play_type": "pass",
                         "epa": epa, "passer_player_id": qb, "play_id": i})
            rows.append({"season": season, "week": week, "game_id": gid, "posteam": off, "defteam": deff, "play_type": "run",
                         "epa": 0.0, "passer_player_id": None, "play_id": 100 + i})
    for w in (1, 2, 3):
        game(2024, w, f"2024_{w}_B_A", "A", "B", 0.5, "X")
        game(2024, w, f"2024_{w}_B_A", "B", "A", 0.0, "Y")
    game(2025, 1, "2025_1_A_B", "A", "B", -0.5, "Z")
    game(2025, 1, "2025_1_A_B", "B", "A", 0.5, "X")     # X now throws for B
    return pd.DataFrame(rows)


def test_rolling_rating_is_pre_game_and_carries_across_seasons():
    w = F.weekly_team_stats(fake_pbp())
    r = F.rolling_team_ratings(w)
    a = r[r.team == "A"].sort_values(["season", "week"])
    league = w["epa_pass_off"].mean()
    assert a.iloc[0]["r_epa_pass_off"] == league                # week 1 knows nothing yet
    assert a.iloc[1]["r_epa_pass_off"] > a.iloc[0]["r_epa_pass_off"]  # learned from week 1 only
    pre_2025 = a[a.season == 2025].iloc[0]["r_epa_pass_off"]
    post_2024 = F.latest_team_rating(r[r.season == 2024], w[w.season == 2024], "A")["epa_pass_off"]
    assert abs(pre_2025 - (F.OFFSEASON_CARRY * post_2024 + (1 - F.OFFSEASON_CARRY) * league)) < 1e-9
    # 2025 wk1 rating for A does NOT include A's terrible 2025 wk1 game
    assert pre_2025 > 0


def test_qb_rating_follows_player_across_teams():
    q = F.weekly_qb_stats(fake_pbp())
    qr = F.rolling_qb_ratings(q)
    x_into_2025 = F.qb_rating_before(qr, "X", 2025, 1)
    z_into_2025 = F.qb_rating_before(qr, "Z", 2025, 1)
    assert x_into_2025 > z_into_2025                             # X's history came with him to team B
    assert F.qb_rating_before(qr, "NOBODY", 2025, 1) == qr["__league__"][0][2]
    assert F.qb_rating_before(qr, "X", 2024, 1) == qr["__league__"][0][2]  # nothing known before his first game


def test_build_game_features_flags_qb_change_and_uses_prior_ratings():
    sched = pd.DataFrame([
        {"game_id": "2024_1_B_A", "season": 2024, "week": 1, "gameday": "2024-09-08", "game_type": "REG", "home_team": "A", "away_team": "B",
         "home_score": 30, "away_score": 10, "home_moneyline": -150, "away_moneyline": 130, "home_qb_id": "X", "away_qb_id": "Y",
         "home_rest": 7, "away_rest": 7, "div_game": 1, "roof": "outdoors", "wind": 12},
        {"game_id": "2024_2_B_A", "season": 2024, "week": 2, "gameday": "2024-09-15", "game_type": "REG", "home_team": "A", "away_team": "B",
         "home_score": 20, "away_score": 21, "home_moneyline": -200, "away_moneyline": 170, "home_qb_id": "X", "away_qb_id": "Y",
         "home_rest": 7, "away_rest": 10, "div_game": 1, "roof": "dome", "wind": None},
        {"game_id": "2025_1_A_B", "season": 2025, "week": 1, "gameday": "2025-09-07", "game_type": "REG", "home_team": "B", "away_team": "A",
         "home_score": None, "away_score": None, "home_moneyline": -110, "away_moneyline": -110, "home_qb_id": "X", "away_qb_id": "Z",
         "home_rest": 7, "away_rest": 7, "div_game": 0, "roof": "outdoors", "wind": None},
    ])
    inj = pd.DataFrame([{"season": 2024, "week": 2, "team": "A", "position": "QB", "report_status": "Out"},
                        {"season": 2024, "week": 2, "team": "B", "position": "WR", "report_status": "Questionable"}])
    f = F.build_game_features(sched, fake_pbp(), inj)
    assert list(f.game_id) == ["2024_1_B_A", "2024_2_B_A", "2025_1_A_B"]
    assert f.iloc[0]["diff_pass"] == 0.0                         # week 1: nobody knows anything
    assert f.iloc[1]["diff_pass"] == 0.0                         # two-team world: A's offence == B's bad defence, cancels exactly
    assert f.iloc[1]["diff_qb"] > 0                              # but X has a track record and Y doesn't
    assert f.iloc[1]["diff_inj"] == 3.0 and f.iloc[0]["diff_inj"] == 0
    assert f.iloc[1]["rest_diff"] == -3 and f.iloc[1]["outdoor"] == 0 and f.iloc[0]["wind"] == 12
    last = f.iloc[2]
    assert np.isnan(last["home_win"]) and last["diff_qb_change"] == 1 - 1   # both teams changed QB
    assert last["diff_qb"] > 0                                   # B now has X, the proven QB
    assert set(F.feature_list(["market", "qb"])) == {"logit_mkt", "diff_qb", "diff_qb_change"}
