"""
mispricing.py — hunt for statistically significant market pricing errors in past games.

Method (pre-registered, so we don't fool ourselves):
  1. A FIXED list of hypotheses (classic "the market is wrong about X" claims). Each names
     a market (ML / ATS / TOT), a filter on pre-game facts, and which side to bet.
  2. DISCOVERY window (seasons <= 2017): score each hypothesis at the closing price.
     ML: expected wins under an efficient market = Σ no-vig p; z = (W − Σp)/sqrt(Σp(1−p)).
     ATS/TOT: win rate vs 50% excluding pushes; ROI at −110.
     Control the false-discovery rate with Benjamini–Hochberg at q = 0.10 across ALL tests.
  3. VALIDATION window (2018+): a survivor must ALSO show positive ROI and p < 0.10
     out of sample. Anything that only "works" in discovery is a data-mining artefact.
  4. Market calibration by implied-probability bucket (favourite–longshot bias check).

Filters use only pre-game columns so the app can apply validated rules to upcoming games.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from engine import american_to_decimal, no_vig_two_way, norm_cdf

DISCOVERY_MAX_SEASON = 2017
FDR_Q = 0.10
VALIDATION_P = 0.10
MIN_N = 60
ATS_DEC = 1.9091  # -110


# ---------------------------------------------------------------- normalisation

def normalise(sched: pd.DataFrame) -> pd.DataFrame:
    """Pre-game facts + (if played) outcomes, one row per REG game."""
    s = sched[sched["game_type"] == "REG"].copy()
    g = pd.DataFrame({
        "game_id": s["game_id"], "season": s["season"], "week": s["week"],
        "home": s["home_team"], "away": s["away_team"],
        "spread": s["spread_line"],                  # + = home favoured by that many
        "total": s["total_line"],
        "home_ml": s["home_moneyline"], "away_ml": s["away_moneyline"],
        "home_rest": s["home_rest"].fillna(7), "away_rest": s["away_rest"].fillna(7),
        "divisional": s["div_game"].fillna(0).astype(int),
        "outdoor": (s["roof"] == "outdoors").astype(int),
        "temp": s["temp"], "wind": s["wind"],
        "weekday": s["weekday"], "gametime": s["gametime"].fillna("13:00"),
        "result": s["home_score"] - s["away_score"],
        "points": s["home_score"] + s["away_score"],
    })
    g["primetime"] = (g["weekday"].isin(["Monday", "Thursday"]) | (g["gametime"] >= "20:00")).astype(int)
    g["home_fav"] = (g["spread"] > 0).astype(int)
    g["pick_em"] = (g["spread"] == 0).astype(int)
    g["dog"] = np.where(g["spread"] > 0, "away", np.where(g["spread"] < 0, "home", None))
    g["fav"] = np.where(g["spread"] > 0, "home", np.where(g["spread"] < 0, "away", None))
    g["dog_rest"] = np.where(g["dog"] == "home", g["home_rest"], g["away_rest"])
    g["fav_rest"] = np.where(g["dog"] == "home", g["away_rest"], g["home_rest"])
    has_ml = g["home_ml"].notna() & g["away_ml"].notna()
    ph = [no_vig_two_way(american_to_decimal(h), american_to_decimal(a))[0] if ok else np.nan
          for h, a, ok in zip(g["home_ml"], g["away_ml"], has_ml)]
    g["p_home"] = ph
    g["p_away"] = 1 - g["p_home"]
    g["p_dog"] = np.where(g["dog"] == "home", g["p_home"], g["p_away"])
    return g.reset_index(drop=True)


# ---------------------------------------------------------------- hypotheses

# name, market, filter(g)->mask, side(g)->Series of 'home'/'away' or 'over'/'under', plain-English rule
HYPOTHESES: List[Tuple[str, str, Callable, Callable, str]] = [
    ("Home dogs ML", "ML", lambda g: g.dog == "home", lambda g: g.dog, "bet the home underdog on the moneyline"),
    ("Road dogs ML", "ML", lambda g: g.dog == "away", lambda g: g.dog, "bet the road underdog on the moneyline"),
    ("Dogs +100 to +150 ML", "ML", lambda g: g.p_dog.between(0.40, 0.50), lambda g: g.dog, "bet underdogs priced 40–50%"),
    ("Dogs +150 to +250 ML", "ML", lambda g: g.p_dog.between(0.286, 0.40), lambda g: g.dog, "bet underdogs priced 29–40%"),
    ("Dogs +250 to +400 ML", "ML", lambda g: g.p_dog.between(0.20, 0.286), lambda g: g.dog, "bet underdogs priced 20–29%"),
    ("Big dogs > +400 ML", "ML", lambda g: g.p_dog < 0.20, lambda g: g.dog, "bet underdogs priced under 20%"),
    ("Heavy favourites ML", "ML", lambda g: (1 - g.p_dog) > 0.80, lambda g: g.fav, "bet favourites priced over 80%"),
    ("Divisional dogs ML", "ML", lambda g: (g.divisional == 1) & g.dog.notna(), lambda g: g.dog, "bet the underdog in division games"),
    ("Dogs off a bye ML", "ML", lambda g: (g.dog_rest >= 13) & g.dog.notna(), lambda g: g.dog, "bet underdogs coming off a bye"),
    ("Dogs with 3+ days more rest ML", "ML", lambda g: ((g.dog_rest - g.fav_rest) >= 3) & g.dog.notna(), lambda g: g.dog, "bet underdogs with a 3+ day rest edge"),
    ("Primetime dogs ML", "ML", lambda g: (g.primetime == 1) & g.dog.notna(), lambda g: g.dog, "bet underdogs in primetime"),
    ("Late-season dogs ML (wk 15+)", "ML", lambda g: (g.week >= 15) & g.dog.notna(), lambda g: g.dog, "bet underdogs from week 15 on"),
    ("Cold-weather home dogs ML", "ML", lambda g: (g.dog == "home") & (g.outdoor == 1) & (g.temp < 35), lambda g: g.dog, "bet home dogs outdoors under 35°F"),
    ("Home dogs ATS", "ATS", lambda g: g.dog == "home", lambda g: g.dog, "take the points with home underdogs"),
    ("Road favourites ATS", "ATS", lambda g: g.fav == "away", lambda g: g.fav, "lay the points with road favourites"),
    ("Big favourites (7+) ATS: take dog", "ATS", lambda g: g.spread.abs() >= 7, lambda g: g.dog, "take the points against 7+ point favourites"),
    ("Big favourites (7+) ATS: lay it", "ATS", lambda g: g.spread.abs() >= 7, lambda g: g.fav, "lay 7+ points with big favourites"),
    ("Divisional dogs ATS", "ATS", lambda g: (g.divisional == 1) & g.dog.notna(), lambda g: g.dog, "take the points in division games"),
    ("Team off a bye ATS", "ATS", lambda g: (g.home_rest >= 13) ^ (g.away_rest >= 13), lambda g: np.where(g.home_rest >= 13, "home", "away"), "bet the team coming off a bye ATS"),
    ("Rest edge 3+ days ATS", "ATS", lambda g: (g.home_rest - g.away_rest).abs() >= 3, lambda g: np.where(g.home_rest > g.away_rest, "home", "away"), "bet the better-rested team ATS"),
    ("Thursday home team ATS", "ATS", lambda g: g.weekday == "Thursday", lambda g: pd.Series("home", index=g.index), "bet the home team on Thursday night"),
    ("Primetime dogs ATS", "ATS", lambda g: (g.primetime == 1) & g.dog.notna(), lambda g: g.dog, "take the points in primetime"),
    ("Week 1 dogs ATS", "ATS", lambda g: (g.week == 1) & g.dog.notna(), lambda g: g.dog, "take the points in week 1"),
    ("Late-season dogs ATS (wk 15+)", "ATS", lambda g: (g.week >= 15) & g.dog.notna(), lambda g: g.dog, "take the points from week 15 on"),
    ("Cold-weather dogs ATS", "ATS", lambda g: (g.outdoor == 1) & (g.temp < 35) & g.dog.notna(), lambda g: g.dog, "take the points outdoors under 35°F"),
    ("Windy dogs ATS (15+ mph)", "ATS", lambda g: (g.outdoor == 1) & (g.wind >= 15) & g.dog.notna(), lambda g: g.dog, "take the points in 15+ mph wind"),
    ("Under in wind 15+", "TOT", lambda g: (g.outdoor == 1) & (g.wind >= 15), lambda g: pd.Series("under", index=g.index), "bet the under in 15+ mph wind"),
    ("Under in cold (<35°F)", "TOT", lambda g: (g.outdoor == 1) & (g.temp < 35), lambda g: pd.Series("under", index=g.index), "bet the under outdoors under 35°F"),
    ("Under in division games", "TOT", lambda g: g.divisional == 1, lambda g: pd.Series("under", index=g.index), "bet the under in division games"),
    ("Under when total 50+", "TOT", lambda g: g.total >= 50, lambda g: pd.Series("under", index=g.index), "bet the under on totals of 50 or more"),
    ("Over when total <= 40", "TOT", lambda g: g.total <= 40, lambda g: pd.Series("over", index=g.index), "bet the over on totals of 40 or less"),
    ("Over in domes", "TOT", lambda g: g.outdoor == 0, lambda g: pd.Series("over", index=g.index), "bet the over indoors"),
    ("Under in primetime", "TOT", lambda g: g.primetime == 1, lambda g: pd.Series("under", index=g.index), "bet the under in primetime"),
    ("Over in week 1", "TOT", lambda g: g.week == 1, lambda g: pd.Series("over", index=g.index), "bet the over in week 1"),
]


# ---------------------------------------------------------------- scoring

def _score_ml(g: pd.DataFrame, side: pd.Series) -> dict:
    g = g[g.p_home.notna() & g.result.notna()]
    side = pd.Series(side, index=g.index) if not isinstance(side, pd.Series) else side.loc[g.index]
    if len(g) == 0:
        return {"n": 0}
    won = np.where(side == "home", g.result > 0, g.result < 0)
    push = g.result == 0
    p = np.where(side == "home", g.p_home, g.p_away)
    dec = np.array([american_to_decimal(h if s == "home" else a) for h, a, s in zip(g.home_ml, g.away_ml, side)])
    keep = ~push
    W, exp, var = won[keep].sum(), p[keep].sum(), (p[keep] * (1 - p[keep])).sum()
    z = (W - exp) / math.sqrt(var) if var > 0 else 0.0
    profit = np.where(won[keep], dec[keep] - 1, -1.0).sum()
    n = int(keep.sum())
    return {"n": n, "wins": int(W), "win_rate": round(W / n, 4), "expected_rate": round(exp / n, 4),
            "roi": round(profit / n, 4), "z": round(z, 2), "p": round(1 - norm_cdf(z), 4)}


def _score_binary(g: pd.DataFrame, side: pd.Series, market: str) -> dict:
    g = g[g.result.notna() & g.points.notna()]
    side = pd.Series(side, index=g.index) if not isinstance(side, pd.Series) else side.loc[g.index]
    if market == "ATS":
        g = g[g.spread.notna()]; side = side.loc[g.index]
        margin = g.result - g.spread                     # >0 home covers
        won = np.where(side == "home", margin > 0, margin < 0)
        push = margin == 0
    else:
        g = g[g.total.notna()]; side = side.loc[g.index]
        diff = g.points - g.total
        won = np.where(side == "over", diff > 0, diff < 0)
        push = diff == 0
    keep = ~push
    n = int(keep.sum())
    if n == 0:
        return {"n": 0}
    W = int(won[keep].sum())
    z = (W - 0.5 * n) / math.sqrt(0.25 * n)
    profit = W * (ATS_DEC - 1) - (n - W)
    return {"n": n, "wins": W, "win_rate": round(W / n, 4), "expected_rate": 0.5,
            "roi": round(profit / n, 4), "z": round(z, 2), "p": round(1 - norm_cdf(z), 4)}


def score(g: pd.DataFrame, market: str, side) -> dict:
    return _score_ml(g, side) if market == "ML" else _score_binary(g, side, market)


def bh_fdr(pvals: List[float], q: float = FDR_Q) -> List[bool]:
    """Benjamini–Hochberg: which tests survive at false-discovery rate q."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    passed = [False] * m
    k_max = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= q * rank / m:
            k_max = rank
    for rank, i in enumerate(order, start=1):
        if rank <= k_max:
            passed[i] = True
    return passed


def run_hypotheses(g: pd.DataFrame, discovery_max: int = DISCOVERY_MAX_SEASON) -> pd.DataFrame:
    rows = []
    for name, market, filt, side_fn, rule in HYPOTHESES:
        mask = filt(g).fillna(False).astype(bool)
        sub = g[mask]
        side = side_fn(sub)
        disc, val = sub[sub.season <= discovery_max], sub[sub.season > discovery_max]
        sd = score(disc, market, pd.Series(side, index=sub.index).loc[disc.index] if len(disc) else pd.Series(dtype=object))
        sv = score(val, market, pd.Series(side, index=sub.index).loc[val.index] if len(val) else pd.Series(dtype=object))
        sa = score(sub, market, pd.Series(side, index=sub.index))
        rows.append({"hypothesis": name, "market": market, "rule": rule,
                     **{f"disc_{k}": v for k, v in sd.items()}, **{f"val_{k}": v for k, v in sv.items()}, **{f"all_{k}": v for k, v in sa.items()}})
    df = pd.DataFrame(rows)
    df["disc_p"] = df["disc_p"].fillna(1.0)
    ok_n = df["disc_n"].fillna(0) >= MIN_N
    df["fdr_pass"] = False
    df.loc[ok_n, "fdr_pass"] = bh_fdr(df.loc[ok_n, "disc_p"].tolist())
    df["validated"] = df["fdr_pass"] & (df["val_n"].fillna(0) >= MIN_N // 2) & (df["val_roi"].fillna(-1) > 0) & (df["val_p"].fillna(1) < VALIDATION_P)
    # Watch list: suggestive in BOTH windows but not FDR-proof. Shown in the app with a caveat, never auto-bet.
    df["watch"] = ~df["validated"] & (df["disc_p"] < 0.05) & (df["val_roi"].fillna(-1) > 0) & (df["val_p"].fillna(1) < VALIDATION_P)
    return df.sort_values("disc_p").reset_index(drop=True)


def calibration_buckets(g: pd.DataFrame) -> pd.DataFrame:
    """Market implied prob (no-vig, home side) vs actual home win rate + z."""
    x = g[g.p_home.notna() & g.result.notna() & (g.result != 0)].copy()
    x["bucket"] = pd.cut(x.p_home, [0, .2, .3, .4, .5, .6, .7, .8, 1.0])
    out = []
    for b, s in x.groupby("bucket", observed=True):
        n, exp, var = len(s), s.p_home.sum(), (s.p_home * (1 - s.p_home)).sum()
        W = (s.result > 0).sum()
        z = (W - exp) / math.sqrt(var) if var else 0
        out.append({"bucket": str(b), "n": n, "implied": round(exp / n, 3), "actual": round(W / n, 3), "z": round(z, 2)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- report

def write_report(df: pd.DataFrame, cal: pd.DataFrame, path: str, seasons: Tuple[int, int]) -> None:
    surv = df[df.validated]
    L = [f"# Market mispricing scan — NFL {seasons[0]}–{seasons[1]}", "",
         f"{len(HYPOTHESES)} pre-registered hypotheses · discovery ≤ {DISCOVERY_MAX_SEASON}, validation {DISCOVERY_MAX_SEASON+1}+ · "
         f"BH false-discovery rate q={FDR_Q} · a rule is **validated** only if it passes FDR in discovery AND shows ROI > 0 with p < {VALIDATION_P} out of sample.", "",
         "## Validated pricing errors", ""]
    if surv.empty:
        L.append("**None.** Every classic bias either never existed at the closing line or has been arbitraged away in the validation window. See the table: several look great in discovery and die out of sample — that is what data mining looks like.")
    else:
        L += ["| hypothesis | market | discovery n / ROI / p | validation n / ROI / p |", "|---|---|---|---|"]
        for r in surv.itertuples():
            L.append(f"| **{r.hypothesis}** | {r.market} | {r.disc_n} / {r.disc_roi:+.1%} / {r.disc_p} | {r.val_n} / {r.val_roi:+.1%} / {r.val_p} |")
    watch = df[df.watch]
    L += ["", "## Watch list (suggestive in both windows, NOT proven)", ""]
    if watch.empty:
        L.append("Nothing.")
    else:
        L += ["| hypothesis | market | discovery n / ROI / p | validation n / ROI / p |", "|---|---|---|---|"]
        for r in watch.itertuples():
            L.append(f"| {r.hypothesis} | {r.market} | {r.disc_n} / {r.disc_roi:+.1%} / {r.disc_p} | {r.val_n} / {r.val_roi:+.1%} / {r.val_p} |")
        L.append("")
        L.append("Treat these as 'don't bet against them', not as a system. Combined they survive neither the FDR bar nor a 50-bet live test yet.")
    L += ["", "## All hypotheses", "", "| hypothesis | mkt | disc n | disc win% | exp% | disc ROI | disc p | FDR | val n | val win% | val ROI | val p | validated |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        f = lambda v, pct=False: "—" if pd.isna(v) else (f"{v:+.1%}" if pct else v)
        L.append(f"| {r.hypothesis} | {r.market} | {f(r.disc_n)} | {f(r.disc_win_rate)} | {f(r.disc_expected_rate)} | {f(r.disc_roi, True)} | {f(r.disc_p)} | {'✅' if r.fdr_pass else ''} | {f(r.val_n)} | {f(r.val_win_rate)} | {f(r.val_roi, True)} | {f(r.val_p)} | {'✅' if r.validated else ''} |")
    L += ["", "ML rows: exp% = what an efficient market predicts (mean no-vig probability of the bet side); ROI at the actual closing price. ATS/TOT rows: ROI at −110 (breakeven 52.4%).", "",
          "## Is the moneyline calibrated? (favourite–longshot check)", "", "| home implied prob | n | implied | actual | z |", "|---|---|---|---|---|"]
    for r in cal.itertuples():
        L.append(f"| {r.bucket} | {r.n} | {r.implied} | {r.actual} | {r.z:+.2f} |")
    L += ["", "|z| above ~2 in a bucket means the market systematically misprices that range."]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    import json, os, sys, warnings
    warnings.filterwarnings("ignore")
    import nfl_data_py as nfl
    here = os.path.dirname(os.path.abspath(__file__))
    seasons = list(range(1999, 2027))
    g = normalise(nfl.import_schedules(seasons))
    df = run_hypotheses(g)
    cal = calibration_buckets(g)
    write_report(df, cal, os.path.join(here, "MISPRICING.md"), (1999, int(g[g.result.notna()].season.max())))
    os.makedirs(os.path.join(here, "data"), exist_ok=True)
    df.to_json(os.path.join(here, "data", "mispricing.json"), orient="records", indent=1)
    print(open(os.path.join(here, "MISPRICING.md")).read())


def match_rules(g_upcoming: pd.DataFrame, results: pd.DataFrame, include_watch: bool = True) -> List[dict]:
    """For each upcoming game (normalised row), list validated / watch-list rules it triggers and the side."""
    names = set(results[results.validated].hypothesis) | (set(results[results.watch].hypothesis) if include_watch else set())
    out = []
    for name, market, filt, side_fn, rule in HYPOTHESES:
        if name not in names:
            continue
        mask = filt(g_upcoming)
        mask = mask.fillna(False).astype(bool) if hasattr(mask, "fillna") else pd.Series(bool(mask), index=g_upcoming.index)
        if not mask.any():
            continue
        sub = g_upcoming[mask]
        sides = pd.Series(side_fn(sub), index=sub.index)
        r = results[results.hypothesis == name].iloc[0]
        for gid, side in zip(sub.game_id, sides):
            out.append({"game_id": gid, "hypothesis": name, "market": market, "side": side, "rule": rule,
                        "tier": "validated" if r.validated else "watch", "val_roi": r.val_roi, "val_n": r.val_n})
    return out
