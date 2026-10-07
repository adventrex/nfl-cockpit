#!/usr/bin/env python
"""
picks.py — headless run of the cockpit for one date: ranks every moneyline side by EV.

    .venv/bin/python picks.py                 # today
    .venv/bin/python picks.py 2026-09-20 --book draftkings --top 10

Uses the app's own defaults: market prior, 10% model weight, excess-HFA nudge (capped),
injury-driven News, weather shrink. Prices: the chosen book (Oregon = draftkings) AND the
cross-book consensus, so a DK price that is off the market shows up as a second signal.
"""
import argparse, datetime as dt, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pandas as pd, requests
import engine as E, injuries as I, scanner as SC, mispricing as MP, stadiums as ST



def main():
    ap = argparse.ArgumentParser(); ap.add_argument("date", nargs="?", default=str(dt.date.today()))
    ap.add_argument("--book", default="draftkings"); ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(); day = dt.date.fromisoformat(a.date)
    import nfl_data_py as nfl
    from autopicker import cockpit_ml_probs, MODEL_W
    season = day.year if day.month >= 3 else day.year - 1
    sched = nfl.import_schedules([season]); sched["gameday"] = sched["gameday"].astype(str)
    games = sched[sched.gameday == str(day)]
    if games.empty:
        print("No games on", day); return
    prob_fn = cockpit_ml_probs(season)
    sl = I.fetch_sleeper()
    r = requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/", params={"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "decimal", "apiKey": E.odds_api_key()}, timeout=15)
    q = SC.parse_payload(r.json(), SC.TEAM_MAP); cons = SC.consensus_edges(q, min_books=3)
    print(f"# Picks for {day} · {len(games)} games · book={a.book} · model weight {MODEL_W:.0%} · API credits left {r.headers.get('x-requests-remaining')}\n")

    rows = []
    for _, g in games.iterrows():
        h, aw = g.home_team, g.away_team; gid = f"{aw}@{h}"
        bk = q[(q.game == gid) & (q.market == "h2h") & (q.book == a.book)]
        if len(bk) < 2:
            continue
        dh, da = float(bk[bk.side == "home"].dec.iloc[0]), float(bk[bk.side == "away"].dec.iloc[0])
        c = cons[(cons.game == gid) & (cons.market == "h2h")]
        p_cons = float(c[c.side == "home"].p_fair.iloc[0]) if len(c[c.side == "home"]) else E.no_vig_two_way(dh, da)[0]
        w = ST.get_forecast(h, day)
        news, hn, an = I.matchup_news_ticks(sl, h, aw, g.get("home_qb_name"), g.get("away_qb_name"))
        p, _, br = prob_fn(g, p_cons)
        for side, team, pp, dec in (("home", h, p, dh), ("away", aw, 1 - p, da)):
            best = q[(q.game == gid) & (q.market == "h2h") & (q.side == side) & ~q.book.isin(SC.OFFSHORE)].sort_values("dec").iloc[-1]
            rows.append({"game": gid, "pick": team, "odds": E.decimal_to_american(dec), "p_cockpit": pp, "p_consensus": p_cons if side == "home" else 1 - p_cons,
                         "ev": E.ev_per_dollar(pp, dec), "ev_vs_consensus_only": E.ev_per_dollar(p_cons if side == "home" else 1 - p_cons, dec),
                         "model": br["AI Model (Stats)"] - p_cons if side == "home" else p_cons - br["AI Model (Stats)"],
                         "news": br["News / Injury"] * (1 if side == "home" else -1), "hfa": br["Home Field (excess)"] * (1 if side == "home" else -1),
                         "wx": w["desc"] if w else "", "inj": "; ".join((an if side == "home" else hn)[:3]),
                         "best_other": f"{E.decimal_to_american(best.dec):+d} {best.book}", "kickoff": g.gametime})
    if not rows:
        print("No", a.book, "moneylines quoted for these games (already kicked off, or the book is dark)."); return
    df = pd.DataFrame(rows).sort_values("ev", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    top = df.head(a.top).copy()
    for c in ("p_cockpit", "p_consensus"): top[c] = top[c].map("{:.1%}".format)
    for c in ("ev", "ev_vs_consensus_only", "model", "news", "hfa"): top[c] = top[c].map("{:+.1%}".format)
    print(top[["game", "kickoff", "pick", "odds", "p_consensus", "p_cockpit", "ev", "ev_vs_consensus_only", "model", "news", "hfa", "best_other"]].to_string(index=False))
    print("\nOpponent starters out/questionable for each pick:")
    for r_ in df.head(a.top).itertuples(): print(f"  {r_.pick:>3} ({r_.game}): {r_.inj or '—'} | wx {r_.wx}")
    n_bar = int((df.ev >= 0.03).sum()); print(f"\n{n_bar} sides clear the app's 3% min-EV bar at {a.book}.")

    # spreads / totals: DK prices off consensus
    oc = cons[(cons.book == a.book) & cons.game.isin([f'{g.away_team}@{g.home_team}' for g in games.itertuples()]) & (cons.edge > 0)].head(8)
    print(f"\n## {a.book} prices above cross-book consensus today (any market)")
    print(oc[["game", "market", "side", "point", "am", "fair_am", "n_books", "edge"]].assign(edge=oc.edge.map("{:+.1%}".format)).to_string(index=False) if len(oc) else "none")

    # wind-under watch rule
    up = games.copy()
    for i, g in up.iterrows():
        w = ST.get_forecast(g.home_team, day)
        if w and not w["is_closed"]: up.at[i, "temp"], up.at[i, "wind"] = w["temp"], w["wind"]
    res = pd.read_json(os.path.join(HERE, "data", "mispricing.json"))
    m = MP.match_rules(MP.normalise(up), res)
    print("\n## Watch-list rule matches (under in 15+ mph wind)")
    for x in m:
        aw_, h_ = x["game_id"].split("_")[2], x["game_id"].split("_")[3]; gid = f"{aw_}@{h_}"
        dk = q[(q.game == gid) & (q.market == "totals") & (q.book == a.book) & (q.side == "under")]
        ce = cons[(cons.game == gid) & (cons.market == "totals") & (cons.book == a.book) & (cons.side == "under")]
        w = ST.get_forecast(h_, day)
        if len(dk):
            print(f"  {gid}: UNDER {dk.point.iloc[0]:g} @ {E.decimal_to_american(dk.dec.iloc[0]):+d} ({a.book}) · wind {w['wind']} mph, rain {w['rain']}% · vs consensus {ce.edge.iloc[0]:+.1%}" if len(ce) else f"  {gid}: UNDER {dk.point.iloc[0]:g} @ {E.decimal_to_american(dk.dec.iloc[0]):+d} · wind {w['wind']} mph")
    if not m: print("none")


if __name__ == "__main__":
    main()
