#!/usr/bin/env python
"""
backtest.py — walk-forward test of the win-prob model against real outcomes.

For every (season, week) in the target seasons, train on every game strictly before
that week (features are pre-game by construction, see features.py), predict the week,
and score. Five nested feature sets show what each ingredient adds. Betting P&L is
simulated at the CLOSING moneyline with the app's own rule (min EV, ¼-Kelly capped),
which is the hardest possible test: beating the close consistently is real edge.

    .venv/bin/python backtest.py                 # 2024 + 2025 (+ 2026 to date)
    .venv/bin/python backtest.py --seasons 2025
Writes BACKTEST.md and data/backtest.json.
"""
import argparse
import datetime as dt
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402
from engine import decide, half_kelly_fraction  # noqa: E402

SETS = [("market", ["market"]), ("+team", ["market", "team"]), ("+qb", ["market", "team", "qb"]),
        ("+context", ["market", "team", "qb", "context"]), ("+injuries", ["market", "team", "qb", "context", "injuries"])]
MIN_TRAIN = 200
MIN_EV = 0.03
RISK_MULT = 0.5
APP_BLEND = 0.3      # app: p = 0.7 * market + 0.3 * model


def load(seasons_needed):
    import nfl_data_py as nfl
    sched = nfl.import_schedules(seasons_needed)
    pbp = pd.concat([nfl.import_pbp_data([y], cache=False) for y in seasons_needed])
    try:
        inj = nfl.import_injuries(seasons_needed)
    except Exception:
        inj = None
    return sched, pbp, inj


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))


def simulate_bets(df, p_col):
    """Flat 1-unit and ¼-Kelly (100u bankroll) P&L using the app's decision rule at closing odds."""
    flat_pnl, n, wins, bank = 0.0, 0, 0, 100.0
    for p, dh, da, hw in zip(df[p_col], df["home_dec"], df["away_dec"], df["home_win"]):
        d = decide(p, dh, da, min_ev=MIN_EV)
        if not d:
            continue
        n += 1
        won = (d["side"] == "home") == (hw == 1)
        wins += int(won)
        flat_pnl += (d["d"] - 1) if won else -1
        stake = min(half_kelly_fraction(d["p"], d["d"], cap=0.05) * RISK_MULT, 0.025) * bank
        bank += stake * (d["d"] - 1) if won else -stake
    return {"bets": n, "hit": round(wins / n, 3) if n else None, "flat_roi": round(flat_pnl / n, 4) if n else None,
            "kelly_end_bank": round(bank, 1)}


def run(target_seasons):
    seasons_needed = sorted(set([min(target_seasons) - 2, min(target_seasons) - 1] + list(target_seasons)))
    print("loading", seasons_needed, flush=True)
    sched, pbp, inj = load(seasons_needed)
    feats = F.build_game_features(sched, pbp, inj)
    played = feats.dropna(subset=["home_win"]).copy()
    played["home_win"] = played["home_win"].astype(int)
    played = played.sort_values(["season", "week"]).reset_index(drop=True)
    print(f"{len(played)} played games with lines; injuries={'yes' if inj is not None else 'no'}", flush=True)

    preds = {name: pd.Series(index=played.index, dtype=float) for name, _ in SETS}
    for season in target_seasons:
        for week in sorted(played[played.season == season].week.unique()):
            test_idx = played[(played.season == season) & (played.week == week)].index
            train = played[(played.season < season) | ((played.season == season) & (played.week < week))]
            if len(train) < MIN_TRAIN:
                continue
            for name, sets in SETS:
                cols = F.feature_list(sets)
                m = model().fit(train[cols], train["home_win"])
                preds[name].loc[test_idx] = m.predict_proba(played.loc[test_idx, cols])[:, 1]

    scored = played[preds["market"].notna()].copy()
    for name in preds:
        scored[f"p_{name}"] = preds[name]
        scored[f"pb_{name}"] = (1 - APP_BLEND) * scored["p_mkt"] + APP_BLEND * scored[f"p_{name}"]
    y = scored["home_win"]

    report = {"generated": dt.datetime.now().isoformat(timespec="minutes"), "target_seasons": target_seasons,
              "n_games": int(len(scored)), "baseline": {}, "sets": {}, "by_season": {}, "calibration": []}
    report["baseline"] = {"logloss": round(log_loss(y, scored["p_mkt"]), 4), "brier": round(brier_score_loss(y, scored["p_mkt"]), 4),
                          "home_win_rate": round(float(y.mean()), 3)}
    for name, sets in SETS:
        p = scored[f"p_{name}"]
        report["sets"][name] = {"features": F.feature_list(sets),
                                "logloss": round(log_loss(y, p), 4), "brier": round(brier_score_loss(y, p), 4),
                                "acc": round(float(((p > 0.5) == y).mean()), 3),
                                "bets_raw": simulate_bets(scored, f"p_{name}"),
                                "bets_app_blend": simulate_bets(scored, f"pb_{name}")}
    best = "+injuries"
    for season, g in scored.groupby("season"):
        report["by_season"][int(season)] = {"n": int(len(g)), "market_logloss": round(log_loss(g.home_win, g.p_mkt), 4),
                                            "model_logloss": round(log_loss(g.home_win, g[f"p_{best}"]), 4),
                                            "bets_app_blend": simulate_bets(g, f"pb_{best}")}
    scored["bucket"] = pd.cut(scored[f"p_{best}"], [0, .3, .4, .5, .6, .7, .8, 1.0])
    for b, g in scored.groupby("bucket", observed=True):
        report["calibration"].append({"bucket": str(b), "n": int(len(g)), "pred": round(float(g[f"p_{best}"].mean()), 3),
                                      "actual": round(float(g.home_win.mean()), 3)})
    # feature weights of the full model on everything (for the README)
    full = model().fit(scored[F.feature_list(SETS[-1][1])], y)
    coefs = dict(zip(F.feature_list(SETS[-1][1]), np.round(full[-1].coef_[0], 3)))
    report["full_model_std_coefs"] = {k: float(v) for k, v in coefs.items()}

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(report, open(os.path.join(HERE, "data", "backtest.json"), "w"), indent=2)
    write_markdown(report)
    return report


def write_markdown(r):
    L = [f"# Backtest — walk-forward, seasons {r['target_seasons']}", "",
         f"Generated {r['generated']} · {r['n_games']} games · trained only on games before each week · scored at the **closing** moneyline.", "",
         "## Does anything beat the market?", "",
         "| feature set | log-loss | Brier | acc | bets (raw) | hit | flat ROI | bets (app blend) | hit | flat ROI | ¼-Kelly bank (100 start) |",
         "|---|---|---|---|---|---|---|---|---|---|---|",
         f"| **market alone (no model)** | {r['baseline']['logloss']} | {r['baseline']['brier']} | — | — | — | — | — | — | — | — |"]
    for name, s in r["sets"].items():
        b, a = s["bets_raw"], s["bets_app_blend"]
        f = lambda v, pct=False: ("—" if v is None else (f"{v:+.1%}" if pct else v))
        L.append(f"| {name} | {s['logloss']} | {s['brier']} | {s['acc']} | {b['bets']} | {f(b['hit'])} | {f(b['flat_roi'], True)} | {a['bets']} | {f(a['hit'])} | {f(a['flat_roi'], True)} | {a['kelly_end_bank']} |")
    L += ["", "Lower log-loss / Brier = better probabilities. A feature set earns its place only if it lowers log-loss **below the market row**. Flat ROI is per 1-unit bet at the closing price with the app's 3% min-EV rule.", "",
          "## By season (full model, app blend)", "", "| season | games | market log-loss | model log-loss | bets | hit | flat ROI | ¼-Kelly bank |", "|---|---|---|---|---|---|---|---|"]
    for s, v in r["by_season"].items():
        b = v["bets_app_blend"]
        L.append(f"| {s} | {v['n']} | {v['market_logloss']} | {v['model_logloss']} | {b['bets']} | {b['hit'] if b['hit'] is not None else '—'} | {('%+.1f%%' % (b['flat_roi']*100)) if b['flat_roi'] is not None else '—'} | {b['kelly_end_bank']} |")
    L += ["", "## Calibration (full model)", "", "| predicted home-win bucket | n | mean predicted | actual |", "|---|---|---|---|"]
    for c in r["calibration"]:
        L.append(f"| {c['bucket']} | {c['n']} | {c['pred']} | {c['actual']} |")
    L += ["", "## Standardised coefficients, full model", "", "| feature | coef |", "|---|---|"]
    for k, v in r["full_model_std_coefs"].items():
        L.append(f"| {k} | {v:+.3f} |")
    L += ["", "Caveats: `wind` is the schedule's recorded game-day wind (a stand-in for the forecast the app uses live). Injury loads use the official Wed–Fri reports, which the closing line has already seen. Team ratings are EWM (half-life 6 games, 60% offseason carry); QB ratings follow the player (half-life 8 games)."]
    open(os.path.join(HERE, "BACKTEST.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[2024, 2025, 2026])
    a = ap.parse_args()
    rep = run(a.seasons)
    print(open(os.path.join(HERE, "BACKTEST.md")).read())
