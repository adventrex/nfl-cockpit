#!/usr/bin/env python
"""
autopicker.py — self-improving weekly NFL card built on the cockpit's own engines.

    .venv/bin/python autopicker.py run        # grade -> learn -> snapshot closers -> new card (what launchd calls)
    .venv/bin/python autopicker.py pick       # just build a card for games kicking off in the next 40h
    .venv/bin/python autopicker.py snapshot   # record closing prices for open picks (run near kickoff)
    .venv/bin/python autopicker.py grade      # grade finished picks + parlays
    .venv/bin/python autopicker.py learn      # update tag biases from graded results, write CHANGELOG
    .venv/bin/python autopicker.py report     # scoreboard: what is working / what is not

How a pick is made
  p_fair  = cross-book no-vig consensus (scanner.py) — the sharpest free estimate of truth
  p_final = p_fair nudged in log-odds by (a) the cockpit model + injury news (moneylines only),
            (b) the wind-under watch rule, (c) LEARNED per-tag biases from this picker's own record
  EV      = p_final x DraftKings price - 1   (DK = the only legal Oregon book)
Every card has >= 10 singles and 5 parlays so the learner gets data, but each pick carries a TIER:
  A / B = real-money stake suggested;  PAPER = tracked at 1 unit, stake $0.
Not losing money on picks that are negative-EV is the first and cheapest hedge.

Everything lives in picker_data/: ledger.csv, parlays.csv, weights.json, CHANGELOG.md, cards/.
"""
import argparse, datetime as dt, itertools, json, math, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd, requests
import engine as E, scanner as SC, hedge_engine as H, injuries as I, stadiums as ST
from notify import imessage
MODEL_W = 0.10   # cockpit model weight; backtest found no feature set beats the close, so small on purpose

DATA = os.path.join(HERE, "picker_data")
CARDS = os.path.join(DATA, "cards")
LEDGER, PARLAYS = os.path.join(DATA, "ledger.csv"), os.path.join(DATA, "parlays.csv")
WEIGHTS, CHANGELOG, CONFIG = os.path.join(DATA, "weights.json"), os.path.join(DATA, "CHANGELOG.md"), os.path.join(DATA, "config.json")

DEFAULT_CONFIG = {
    "bankroll": 500.0,            # dollars the stakes are sized against — edit to your real number
    "mode": "paper",              # "paper" = all stakes $0 until the gate below passes AND you flip this to "real"
    "book": "draftkings",
    "min_picks": 10, "n_parlays": 5,
    "tier_a_ev": 0.015, "tier_b_ev": 0.0,      # A: EV >= +1.5% ; B: EV >= 0 ; else PAPER
    "kelly_mult": 0.25, "max_stake_pct": 0.02, "tier_b_stake_pct": 0.005,
    "weekly_risk_pct": 0.06,      # total real stakes on one card never exceed this share of bankroll
    "parlay_share": 0.20,         # parlays get at most this share of the weekly risk budget
    "drawdown_brake_pct": 0.10,   # season real P&L below -10% of bankroll -> stakes halved
    "drawdown_stop_pct": 0.20,    # below -20% -> forced back to paper
    "gate_min_graded": 50,        # pre-registered: no real money until 50 graded singles ...
    "gate_min_clv": 0.0,          # ... show average CLV above this
    "horizon_hours": 40,
}
LEDGER_COLS = ["pick_id", "run_ts", "season", "week", "game_id", "game", "kickoff_utc", "market", "side", "label", "point", "dec", "odds_am",
               "opp_dec", "p_cons", "p_final", "ev", "tier", "stake", "tags", "src", "close_dec", "close_point", "p_close", "clv",
               "result", "pnl", "pnl_unit", "graded_ts"]
PARLAY_COLS = ["parlay_id", "run_ts", "season", "week", "legs", "desc", "dec", "p", "ev", "tier", "stake", "style", "hedge_note", "result", "pnl", "pnl_unit", "graded_ts"]
PT_VALUE = {"SPREAD": 0.030, "TOTAL": 0.025}   # approx win-prob value of one point, used only for CLV when the line moved
WIND_UNDER_NUDGE = 0.06                        # logit nudge for the one watch-list rule (under, 15+ mph wind)


# ---------------------------------------------------------------- storage
def _load(path, cols):
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"pick_id": str, "parlay_id": str, "legs": str, "tags": str, "result": str})
        for c in cols:
            if c not in df.columns: df[c] = np.nan
        return df[cols]
    return pd.DataFrame(columns=cols)


def _save(df, path): df.to_csv(path, index=False)


def cfg():
    os.makedirs(CARDS, exist_ok=True)
    c = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG): c.update(json.load(open(CONFIG)))
    else: json.dump(c, open(CONFIG, "w"), indent=2)
    return c


def weights():
    if os.path.exists(WEIGHTS): return json.load(open(WEIGHTS))
    return {"version": 1, "tag_bias": {}, "benched": [], "signal_w": {"model": 1.0, "wind_under": 1.0}}


def changelog(lines):
    new = not os.path.exists(CHANGELOG)
    with open(CHANGELOG, "a") as f:
        if new: f.write("# Autopicker changelog — every change the learner makes to itself\n")
        f.write(f"\n## {dt.datetime.now():%Y-%m-%d %H:%M}\n" + "\n".join(f"- {x}" for x in lines) + "\n")



# ---------------------------------------------------------------- data
def fetch_odds():
    r = requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/", timeout=20,
                     params={"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "decimal", "apiKey": E.odds_api_key()})
    r.raise_for_status()
    q = SC.parse_payload(r.json(), SC.TEAM_MAP)
    q["commence"] = pd.to_datetime(q["commence"], utc=True)
    return q, SC.consensus_edges(q, min_books=3), r.headers.get("x-requests-remaining")


def season_of(day): return day.year if day.month >= 3 else day.year - 1


def schedule(season):
    import nfl_data_py as nfl
    s = nfl.import_schedules([season])
    s["gameday"] = s["gameday"].astype(str)
    return s


def cockpit_ml_probs(season):
    """The cockpit's win-probability recipe (10% model + injury news + rest), shared with picks.py.
    Returns prob(game_row, p_home_consensus) -> (p_home, news_tick, breakdown). Heavy (downloads pbp)."""
    import nfl_data_py as nfl, features as F
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    yrs = [season - 2, season - 1, season]
    hist = nfl.import_schedules(yrs)
    pbp = pd.concat([nfl.import_pbp_data([y], cache=False) for y in yrs])
    try: inj_hist = nfl.import_injuries(yrs)
    except Exception: inj_hist = None
    feats = F.build_game_features(hist, pbp, inj_hist)
    names = F.feature_list(["market", "team", "qb", "context", "injuries"])
    tr = feats.dropna(subset=["home_win"])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)).fit(tr[names], tr["home_win"].astype(int))
    fx = feats.set_index("game_id")
    sl = I.fetch_sleeper()
    d = {"h_qb": 5, "h_pwr": 5, "h_def": 5, "a_qb": 5, "a_pwr": 5, "a_def": 5}

    def prob(g, p_cons):
        news, hn, an = I.matchup_news_ticks(sl, g.home_team, g.away_team, g.get("home_qb_name"), g.get("away_qb_name"))
        rest_w = 4 if abs(g.home_rest - g.away_rest) > 3 else 0
        sliders = {"ph_qb": 5, "ph_pwr": 5, "ph_def": 5, "pa_qb": 5, "pa_pwr": 5, "pa_def": 5, "news": news, "ww": 0, "hfa": 5, "wr": rest_w, "rh": g.home_rest, "ra": g.away_rest}
        frow = fx.loc[g.game_id].to_dict() if g.game_id in fx.index else {}
        p, br = E.calc_win_prob(p_cons, frow, sliders, {**d, "hfa": 5}, clf=clf, feature_names=names, market_weight=1 - MODEL_W)
        return p, news, br
    return prob


# ---------------------------------------------------------------- candidates
def build_candidates(q, cons, sched, c, W, use_model=True):
    now = pd.Timestamp.now(tz="UTC")
    book = c["book"]
    live = q[(q.commence > now) & (q.commence <= now + pd.Timedelta(hours=c["horizon_hours"]))]
    gids = list(live.game.unique())
    if not gids: return pd.DataFrame(), "no games in horizon"
    note = ""
    prob_fn = None
    if use_model:
        try: prob_fn = cockpit_ml_probs(int(sched.season.iloc[0]))
        except Exception as ex: note = f"⚠️ COCKPIT MODEL UNAVAILABLE ({type(ex).__name__}: {ex}) — moneylines priced on consensus only"
    rows = []
    for gid in gids:
        aw, h = gid.split("@")
        kick = live[live.game == gid].commence.iloc[0]
        local_day = kick.tz_convert("America/New_York").date()
        sg = sched[(sched.home_team == h) & (sched.away_team == aw) & (sched.gameday.isin([str(local_day), str(local_day - dt.timedelta(days=1))]))]
        if sg.empty: continue
        g = sg.iloc[0]
        wx = ST.get_forecast(h, local_day) or {}
        windy = (not wx.get("is_closed", True)) and float(wx.get("wind", 0) or 0) >= 15
        bq = q[(q.game == gid) & (q.book == book) & q.opp_dec.notna()]
        p_home_cons = None
        for r in bq.itertuples():
            mkt = {"h2h": "ML", "spreads": "SPREAD", "totals": "TOTAL"}[r.market]
            ce = cons[(cons.game == gid) & (cons.market == r.market) & (cons.side == r.side) & (cons.book == book) & ((cons.point == r.point) | (cons.point.isna() & pd.isna(r.point)))]
            if len(ce): p_cons, src = float(ce.p_fair.iloc[0]), f"consensus({int(ce.n_books.iloc[0])})"
            else: p_cons, src = E.no_vig_two_way(r.dec, r.opp_dec)[0], "dk-novig"
            tags, nudge = [mkt], 0.0
            if mkt == "ML":
                tags += ["home" if r.side == "home" else "away", "fav" if p_cons >= 0.5 else "dog"]
                if prob_fn is not None:
                    p_home_c = p_cons if r.side == "home" else 1 - p_cons
                    p_home_m, news, _ = prob_fn(g, p_home_c)
                    shift = E.logit(p_home_m) - E.logit(p_home_c)
                    shift = shift if r.side == "home" else -shift
                    nudge += W["signal_w"].get("model", 1.0) * shift
                    if shift > 0.01: tags.append("model_agree")
                    if (news > 0) == (r.side == "home") and news != 0: tags.append("inj_edge")
                label = f"{h if r.side == 'home' else aw} ML"
            elif mkt == "SPREAD":
                tags += ["home" if r.side == "home" else "away", "fav" if r.point < 0 else "dog"]
                if abs(r.point) in (3, 7): tags.append("key_number")
                label = f"{h if r.side == 'home' else aw} {r.point:+g}"
            else:
                tags.append(r.side)
                if windy and r.side == "under":
                    tags.append("wind_under")
                    nudge += W["signal_w"].get("wind_under", 1.0) * WIND_UNDER_NUDGE
                label = f"{r.side.title()} {r.point:g} ({gid})"
            if p_cons * r.dec - 1 >= 0.01: tags.append("dk_off_market")
            biases = [W["tag_bias"].get(t, 0.0) for t in tags]
            nudge += float(np.clip(np.mean(biases), -0.15, 0.15)) if biases else 0.0
            p_final = float(np.clip(E.inv_logit(E.logit(p_cons) + nudge), 0.02, 0.98))
            rows.append({"season": int(g.season), "week": int(g.week), "game_id": g.game_id, "game": gid, "kickoff_utc": kick.isoformat(), "market": mkt, "side": r.side,
                         "label": label, "point": r.point, "dec": r.dec, "odds_am": E.decimal_to_american(r.dec), "opp_dec": r.opp_dec, "p_cons": p_cons,
                         "p_final": p_final, "ev": p_final * r.dec - 1, "tags": "|".join(tags), "src": src, "wx": wx.get("desc", "")})
    return pd.DataFrame(rows), note


def select(cand, c, W, already):
    """Best EV first; per game at most ONE side (ML or spread) and ONE total — stacking correlated bets on a game is un-hedging."""
    cand = cand[~cand.apply(lambda r: (r.game_id, r.market, r.side) in already, axis=1)].copy()
    cand["benched"] = cand.tags.map(lambda t: any(x in W["benched"] for x in t.split("|")))
    cand = cand[(cand.dec >= 1.25) & (cand.dec <= 4.5)]       # no -400 juice, no lottery dogs: both are where vig hides
    cand = cand.sort_values(["benched", "ev"], ascending=[True, False])
    picks, used = [], set()
    n_games = cand.game_id.nunique()
    target = min(c["min_picks"], n_games * 2)
    for r in cand.itertuples():
        slot = (r.game_id, "total" if r.market == "TOTAL" else "side")
        if slot in used: continue
        if len(picks) >= target and r.ev < c["tier_b_ev"]: break
        picks.append(r.Index)
        used.add(slot)
    return cand.loc[picks].copy()


def size(picks, c, real_pnl):
    B = c["bankroll"]
    mult = 1.0
    why = ""
    if c["mode"] != "real": mult, why = 0.0, "paper mode"
    elif real_pnl <= -c["drawdown_stop_pct"] * B: mult, why = 0.0, "drawdown STOP hit — paper until reviewed"
    elif real_pnl <= -c["drawdown_brake_pct"] * B: mult, why = 0.5, "drawdown brake — stakes halved"
    tiers, stakes = [], []
    for r in picks.itertuples():
        if r.ev >= c["tier_a_ev"]: t, s = "A", min(E.kelly_fraction(r.p_final, r.dec) * c["kelly_mult"], c["max_stake_pct"]) * B
        elif r.ev >= c["tier_b_ev"]: t, s = "B", c["tier_b_stake_pct"] * B
        else: t, s = "PAPER", 0.0
        tiers.append(t)
        stakes.append(s * mult)
    picks["tier"], picks["stake"] = tiers, stakes
    budget = c["weekly_risk_pct"] * B * (1 - c["parlay_share"])
    if picks.stake.sum() > budget > 0: picks["stake"] *= budget / picks.stake.sum()
    picks["stake"] = picks.stake.round(2)
    return picks, why


# ---------------------------------------------------------------- parlays
def build_parlays(picks, c, mult_zero):
    """Cross-game parlays only (DK pays the true product; same-game parlays carry ~15% hidden vig).
    Legs ordered by kickoff so the LAST leg is hedgeable once the rest have won."""
    pool = picks.sort_values("ev", ascending=False).head(10)
    combos = []
    for k in (2, 3, 4):
        for idx in itertools.combinations(pool.index, k):
            legs = pool.loc[list(idx)]
            if legs.game_id.nunique() < k: continue
            legs = legs.sort_values("kickoff_utc")
            dec, p = float(legs.dec.prod()), float(legs.p_final.prod())
            last = legs.iloc[-1]
            gap = (pd.Timestamp(last.kickoff_utc) - pd.Timestamp(legs.iloc[-2].kickoff_utc)).total_seconds() / 3600
            combos.append({"idx": list(legs.index), "k": k, "dec": dec, "p": p, "ev": p * dec - 1, "hedgeable": gap >= 3, "last": last})
    if not combos: return []
    out, uses = [], {}

    def take(filt, style):
        for cb in sorted([x for x in combos if filt(x)], key=lambda x: -x["ev"]):
            if any(cb["idx"] == o["idx"] for o in out) or any(uses.get(i, 0) >= 3 for i in cb["idx"]): continue
            for i in cb["idx"]: uses[i] = uses.get(i, 0) + 1
            cb["style"] = style
            out.append(cb)
            return
    take(lambda x: x["k"] == 2, "safest 2-leg")
    take(lambda x: x["k"] == 3 and x["hedgeable"], "3-leg, hedgeable last leg")
    take(lambda x: x["k"] == 2 and x["hedgeable"], "2-leg, hedgeable last leg")
    take(lambda x: x["k"] == 3, "3-leg")
    take(lambda x: x["k"] == 4, "4-leg long shot")
    while len(out) < c["n_parlays"] and len(out) < len(combos):
        n = len(out)
        take(lambda x: True, "best remaining")
        if len(out) == n: uses = {}     # relax reuse cap rather than come up short
    B = c["bankroll"]
    budget = c["weekly_risk_pct"] * B * c["parlay_share"]
    for cb in out:
        cb["tier"] = "B" if cb["ev"] >= 0 else "PAPER"
        cb["stake"] = 0.0 if (mult_zero or cb["tier"] == "PAPER") else round(budget / c["n_parlays"], 2)
        ref = cb["stake"] or 10.0
        last = cb["last"]
        lock_h = H.hedge_equal_profit(ref, cb["dec"], last.opp_dec)
        lock = H.locked_profit(ref, cb["dec"], last.opp_dec)
        kel = H.kelly_optimal_hedge(ref, cb["dec"], last.p_final, last.opp_dec, B)
        cb["hedge_note"] = (f"per ${ref:.0f} staked: if all legs before '{last.label}' win, betting ${lock_h:.2f} on the other side at {E.decimal_to_american(last.opp_dec):+d} "
                            f"locks ${lock:.2f} either way; growth-optimal hedge ${kel:.2f}" + ("" if cb["hedgeable"] else " (legs overlap in time — hedge window is live-betting only)"))
    return out[:c["n_parlays"]]


# ---------------------------------------------------------------- commands
def cmd_pick(a):
    c, W = cfg(), weights()
    led, par = _load(LEDGER, LEDGER_COLS), _load(PARLAYS, PARLAY_COLS)
    q, cons, credits = fetch_odds()
    sched = schedule(season_of(dt.date.today()))
    cand, note = build_candidates(q, cons, sched, c, W, use_model=not a.no_model)
    if cand.empty:
        print("No games in the next", c["horizon_hours"], "hours.", note)
        return None
    open_ = led[led.result.isna()]
    already = set(zip(open_.game_id, open_.market, open_.side)) | {(g, m, {"home": "away", "away": "home", "over": "under", "under": "over"}[s]) for g, m, s in zip(open_.game_id, open_.market, open_.side)}
    carded_games = set(open_.game_id)
    cand = cand[~cand.game_id.isin(carded_games)] if not a.force else cand
    if cand.empty:
        print("Every game in the horizon is already carded (use --force to add more).")
        return None
    picks = select(cand, c, W, already)
    real_pnl = float(led.pnl.fillna(0).sum() + par.pnl.fillna(0).sum())
    gate = gate_status(led, c)
    if c["mode"] == "real" and not gate["passed"]:
        c = {**c, "mode": "paper"}
        note += f"\n⚠️ mode=real ignored: gate not passed ({gate['text']})"
    picks, why = size(picks, c, real_pnl)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M")
    picks["run_ts"] = ts
    picks["pick_id"] = [f"{ts}-{i+1:02d}" for i in range(len(picks))]
    parlays = build_parlays(picks, c, mult_zero=(c["mode"] != "real" or bool(why and "STOP" in why)))
    prow = [{"parlay_id": f"{ts}-P{i+1}", "run_ts": ts, "season": int(picks.season.iloc[0]), "week": int(picks.week.iloc[0]), "legs": ";".join(picks.loc[cb["idx"]].pick_id),
             "desc": " + ".join(picks.loc[cb["idx"]].label), "dec": round(cb["dec"], 3), "p": round(cb["p"], 4), "ev": round(cb["ev"], 4), "tier": cb["tier"],
             "stake": cb["stake"], "style": cb["style"], "hedge_note": cb["hedge_note"]} for i, cb in enumerate(parlays)]
    card = render_card(picks, prow, c, W, gate, note, why, credits)
    if a.dry:
        print(card)
        return card
    _save(pd.concat([led, picks.reindex(columns=LEDGER_COLS)], ignore_index=True), LEDGER)
    _save(pd.concat([par, pd.DataFrame(prow).reindex(columns=PARLAY_COLS)], ignore_index=True), PARLAYS)
    path = os.path.join(CARDS, f"{ts}-week{int(picks.week.iloc[0])}.md")
    open(path, "w").write(card)
    print(card)
    print(f"\nsaved → {path}")
    if a.imessage:
        top = "\n".join(f"{r.tier:>5} {r.label} {r.odds_am:+d} (EV {r.ev:+.1%})" for r in picks.head(10).itertuples())
        imessage(f"🏈 Autopicker wk{int(picks.week.iloc[0])}: {len(picks)} picks, {len(prow)} parlays · mode {c['mode']}\n{top}\nFull card: {path}")
    return card


def render_card(picks, prow, c, W, gate, note, why, credits):
    L = [f"# Autopicker card · week {int(picks.week.iloc[0])} · {dt.datetime.now():%a %b %d %H:%M} · {c['book']} · mode **{c['mode']}** · bankroll ${c['bankroll']:.0f}",
         f"_Gate for real money: {gate['text']}. Learner v{W['version']} · benched tags: {', '.join(W['benched']) or 'none'} · Odds API credits left: {credits}_"]
    if note: L.append(note.strip())
    if why: L.append(f"**Sizing override:** {why}")
    L += ["", "## Singles (ranked by EV at the DraftKings price)", "| # | Tier | Pick | Odds | Fair % | Our % | EV | Stake | Kick (PT) | Why |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(picks.itertuples(), 1):
        k = pd.Timestamp(r.kickoff_utc).tz_convert("America/Los_Angeles").strftime("%a %H:%M")
        L.append(f"| {i} | {r.tier} | {r.label} | {r.odds_am:+d} | {r.p_cons:.1%} | {r.p_final:.1%} | {r.ev:+.1%} | ${r.stake:.2f} | {k} | {r.tags.replace('|', ', ')} |")
    n_real = int((picks.tier != "PAPER").sum())
    L += ["", f"**{n_real} of {len(picks)} singles clear EV ≥ 0.** PAPER picks are tracked at 1 unit for the learner; betting them is paying ~{-picks[picks.tier == 'PAPER'].ev.mean():.1%} vig per dollar for entertainment." if n_real < len(picks) else ""]
    L += ["", "## Parlays (cross-game only; legs in kickoff order)", "| # | Tier | Legs | Pays | Hit % | EV | Stake | Style |", "|---|---|---|---|---|---|---|---|"]
    for i, p in enumerate(prow, 1):
        L.append(f"| P{i} | {p['tier']} | {p['desc']} | {E.decimal_to_american(p['dec']):+d} | {p['p']:.1%} | {p['ev']:+.1%} | ${p['stake']:.2f} | {p['style']} |")
    L += ["", "### Hedge plans"] + [f"- **P{i}** {p['hedge_note']}" for i, p in enumerate(prow, 1)]
    exp = picks.groupby("game").stake.sum()
    tot = picks.stake.sum() + sum(p["stake"] for p in prow)
    L += ["", "## Risk", f"- Total real money at risk this card: **${tot:.2f}** (cap ${c['weekly_risk_pct'] * c['bankroll']:.2f} = {c['weekly_risk_pct']:.0%} of bankroll). Worst case = lose exactly that.",
          f"- Most exposure on one game: ${exp.max():.2f}. One side + one total per game max, so no single upset sinks the card.",
          "- Hedging rule of thumb: a hedge at one book always costs the vig. Use it to lock a parlay's last leg, not on singles."]
    return "\n".join(x for x in L if x is not None)


def cmd_snapshot(a):
    c = cfg()
    led = _load(LEDGER, LEDGER_COLS)
    now = pd.Timestamp.now(tz="UTC")
    k = pd.to_datetime(led.kickoff_utc, utc=True)
    need = led[led.result.isna() & (k > now) & (k <= now + pd.Timedelta(hours=a.within))]
    if need.empty:
        print("snapshot: nothing kicking off within", a.within, "h")
        return
    q, cons, credits = fetch_odds()
    n = 0
    mk = {"ML": "h2h", "SPREAD": "spreads", "TOTAL": "totals"}
    for i, r in need.iterrows():
        bq = q[(q.game == r.game) & (q.book == c["book"]) & (q.market == mk[r.market]) & (q.side == r.side) & q.opp_dec.notna()]
        if bq.empty: continue
        b = bq.iloc[0]
        ce = cons[(cons.game == r.game) & (cons.market == mk[r.market]) & (cons.side == r.side) & (cons.book == c["book"])]
        p_close = float(ce.p_fair.iloc[0]) if len(ce) else E.no_vig_two_way(b.dec, b.opp_dec)[0]
        if r.market != "ML" and pd.notna(b.point) and b.point != r.point:
            better = (r.point - b.point) if r.market == "SPREAD" or r.side == "under" else (b.point - r.point)   # points in my favour vs the close
            p_close = float(np.clip(p_close + better * PT_VALUE[r.market], 0.02, 0.98))
        led.loc[i, ["close_dec", "close_point", "p_close", "clv"]] = [b.dec, b.point, round(p_close, 4), round(p_close * r.dec - 1, 4)]
        n += 1
    _save(led, LEDGER)
    print(f"snapshot: closing prices recorded for {n} picks · credits left {credits}")


def _grade_row(r, g):
    hs, as_ = float(g.home_score), float(g.away_score)
    m = hs - as_
    if r.market == "ML": x = m if r.side == "home" else -m
    elif r.market == "SPREAD": x = (m if r.side == "home" else -m) + float(r.point)
    else: x = (hs + as_ - float(r.point)) * (1 if r.side == "over" else -1)
    return "W" if x > 0 else "L" if x < 0 else "P"


def cmd_grade(a):
    led, par = _load(LEDGER, LEDGER_COLS), _load(PARLAYS, PARLAY_COLS)
    if led.result.notna().all():
        print("grade: nothing open")
        return
    s = pd.concat([schedule(int(y)) for y in led[led.result.isna()].season.unique()]).set_index("game_id")
    n = 0
    for i, r in led[led.result.isna()].iterrows():
        if r.game_id not in s.index or pd.isnull(s.loc[r.game_id].home_score): continue
        g = s.loc[r.game_id]
        res = _grade_row(r, g)
        stake = float(r.stake or 0)
        if pd.isna(r.clv) and r.market == "ML" and pd.notnull(g.get("home_moneyline")):      # fallback close from nflverse
            ph, pa = E.no_vig_two_way(E.american_to_decimal(g.home_moneyline), E.american_to_decimal(g.away_moneyline))
            pc = ph if r.side == "home" else pa
            led.loc[i, ["p_close", "clv"]] = [round(pc, 4), round(pc * r.dec - 1, 4)]
        led.loc[i, ["result", "pnl", "pnl_unit", "graded_ts"]] = [res, round({"W": stake * (r.dec - 1), "L": -stake, "P": 0}[res], 2), round({"W": r.dec - 1, "L": -1, "P": 0}[res], 3), dt.datetime.now().isoformat(timespec="minutes")]
        n += 1
    res_by_id, dec_by_id = dict(zip(led.pick_id, led.result)), dict(zip(led.pick_id, led.dec))
    np_ = 0
    for i, p in par[par.result.isna()].iterrows():
        rs = [res_by_id.get(x) for x in p.legs.split(";")]
        if "L" in rs: res, dec = "L", p.dec
        elif any(pd.isna(x) or x is None for x in rs): continue
        else:
            dec = float(np.prod([dec_by_id[x] for x, y in zip(p.legs.split(";"), rs) if y == "W"]))
            res = "W" if "W" in rs else "P"
        stake = float(p.stake or 0)
        par.loc[i, ["result", "pnl", "pnl_unit", "graded_ts"]] = [res, round({"W": stake * (dec - 1), "L": -stake, "P": 0}[res], 2), round({"W": dec - 1, "L": -1, "P": 0}[res], 3), dt.datetime.now().isoformat(timespec="minutes")]
        np_ += 1
    _save(led, LEDGER)
    _save(par, PARLAYS)
    print(f"grade: {n} singles, {np_} parlays graded")


def tag_table(led):
    g = led[led.result.isin(["W", "L", "P"])]
    rows = []
    for t in sorted({x for ts in g.tags.dropna() for x in ts.split("|")}) + ["ALL"]:
        d = g if t == "ALL" else g[g.tags.str.split("|").map(lambda xs: t in xs)]
        dd = d[d.result != "P"]
        clv = d.clv.dropna()
        rows.append({"tag": t, "n": len(d), "W": int((d.result == "W").sum()), "L": int((d.result == "L").sum()), "exp_W": round(float(dd.p_final.sum()), 1),
                     "roi_unit": round(float(d.pnl_unit.sum() / len(d)), 3) if len(d) else 0.0, "n_clv": len(clv), "clv": round(float(clv.mean()), 4) if len(clv) else np.nan})
    return pd.DataFrame(rows)


def gate_status(led, c):
    g = led[led.result.isin(["W", "L", "P"])]
    clv = g.clv.dropna()
    n, m = len(g), float(clv.mean()) if len(clv) else float("nan")
    ok = n >= c["gate_min_graded"] and len(clv) >= c["gate_min_graded"] // 2 and m > c["gate_min_clv"]
    return {"passed": bool(ok), "text": f"{n}/{c['gate_min_graded']} graded, avg CLV {m:+.2%}" + (" — PASSED" if ok else " — not passed, paper only") if n else f"0/{c['gate_min_graded']} graded — paper only"}


def cmd_learn(a):
    """Conservative by design: NFL results are ~all noise at small n, so CLV (did the market move toward us?) is the main teacher.
    A tag's bias moves only with >= 20 CLV samples (or >= 40 results), is shrunk hard, smoothed 50/50 with its old value, and capped."""
    led, W = _load(LEDGER, LEDGER_COLS), weights()
    t = tag_table(led)
    log = []
    if t.empty or t[t.tag == "ALL"].n.iloc[0] == 0:
        print("learn: no graded picks yet")
        return
    for r in t[t.tag != "ALL"].itertuples():
        old = W["tag_bias"].get(r.tag, 0.0)
        target = None
        if r.n_clv >= 20:
            target = 4 * float(np.clip(r.clv * r.n_clv / (r.n_clv + 30), -0.03, 0.03))
            basis = f"CLV {r.clv:+.2%} over {r.n_clv}"
        elif r.W + r.L >= 40:
            target = 4 * float(np.clip((r.W - r.exp_W) / (r.W + r.L) * (r.W + r.L) / (r.W + r.L + 100), -0.03, 0.03))
            basis = f"{r.W}W vs {r.exp_W} expected"
        if target is not None:
            new = round(0.5 * old + 0.5 * target, 4)
            if abs(new - old) >= 0.002:
                W["tag_bias"][r.tag] = new
                log.append(f"bias[{r.tag}] {old:+.3f} → {new:+.3f} logit ({basis})")
        bench = r.n_clv >= 30 and r.clv < -0.02 and r.tag not in ("ML", "SPREAD", "TOTAL")
        if bench and r.tag not in W["benched"]:
            W["benched"].append(r.tag)
            log.append(f"BENCHED tag '{r.tag}' (CLV {r.clv:+.2%} over {r.n_clv}) — picks with it go to the back of the line")
        if not bench and r.tag in W["benched"] and r.clv > -0.01:
            W["benched"].remove(r.tag)
            log.append(f"un-benched '{r.tag}' (CLV recovered to {r.clv:+.2%})")
    allr = t[t.tag == "ALL"].iloc[0]
    summary = f"record {allr.W}-{allr.L}, flat-unit ROI {allr.roi_unit:+.1%}, avg CLV {allr.clv:+.2%} on {allr.n_clv}" if allr.n_clv else f"record {allr.W}-{allr.L}, flat-unit ROI {allr.roi_unit:+.1%}, no CLV yet"
    if log:
        W["version"] += 1
        json.dump(W, open(WEIGHTS, "w"), indent=2)
    changelog([f"learn pass · {summary}"] + (log or ["no changes — sample too small to justify any (need 20 CLV samples or 40 results per tag)"]))
    print("learn:", summary)
    [print("  ", x) for x in log]


def cmd_report(a):
    c = cfg()
    led, par = _load(LEDGER, LEDGER_COLS), _load(PARLAYS, PARLAY_COLS)
    print(f"# Autopicker scoreboard · {dt.date.today()}\nGate: {gate_status(led, c)['text']}\nReal P&L: singles ${led.pnl.fillna(0).sum():.2f} · parlays ${par.pnl.fillna(0).sum():.2f}\n")
    t = tag_table(led)
    print(t.sort_values("n", ascending=False).to_string(index=False) if len(t) and t.n.sum() else "no graded singles yet")
    pg = par[par.result.isin(["W", "L", "P"])]
    if len(pg): print(f"\nParlays: {int((pg.result == 'W').sum())}-{int((pg.result == 'L').sum())}, flat-unit ROI {pg.pnl_unit.sum() / len(pg):+.1%} (expected {pg.ev.mean():+.1%})")
    print(f"\nOpen: {int(led.result.isna().sum())} singles, {int(par.result.isna().sum())} parlays")


def cmd_run(a):
    for f in (cmd_grade, cmd_learn):
        try: f(a)
        except Exception as ex: print(f"{f.__name__} failed: {type(ex).__name__}: {ex}")
    try: cmd_snapshot(a)
    except Exception as ex: print(f"snapshot failed: {type(ex).__name__}: {ex}")
    try: cmd_pick(a)
    except Exception as ex:
        print(f"pick failed: {type(ex).__name__}: {ex}")
        if a.imessage: imessage(f"🏈 Autopicker pick FAILED: {type(ex).__name__}: {ex}")
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "pick", "snapshot", "grade", "learn", "report"])
    ap.add_argument("--dry", action="store_true", help="pick: print the card, write nothing")
    ap.add_argument("--force", action="store_true", help="pick: allow games that already have open picks")
    ap.add_argument("--no-model", action="store_true", help="pick: skip the heavy cockpit model, consensus only")
    ap.add_argument("--imessage", action="store_true")
    ap.add_argument("--within", type=float, default=3.0, help="snapshot: hours to kickoff")
    a = ap.parse_args()
    {"run": cmd_run, "pick": cmd_pick, "snapshot": cmd_snapshot, "grade": cmd_grade, "learn": cmd_learn, "report": cmd_report}[a.cmd](a)
