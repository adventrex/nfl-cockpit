"""
NFL Edge Cockpit — v4.0.0
Thin Streamlit UI over engine.py (win-prob + sizing), hedge_engine.py (hedging),
and parlay_engine.py (same-game parlays). All math lives in those modules and is
covered by tests/. Run:  streamlit run app.py
"""
import datetime
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

import engine as E
import hedge_engine as H
import injuries as I
import features as F
import scanner as SC
import mispricing as MP
import promo_engine as PE
import json
import parlay_engine as parlay
import stadiums as ST

APP_VERSION = "4.2.1"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bet_log.csv")
MODES = ["🏈 Cockpit", "🔎 Scanner", "🎁 Promo Hedger", "🛡️ Hedge Desk", "📒 Bet Log"]

st.set_page_config(page_title="NFL Edge Cockpit Pro", page_icon="🏈", layout="wide")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🏈 NFL Cockpit")
    st.caption(f"v{APP_VERSION}")
    if "_goto_mode" in st.session_state:
        st.session_state["mode"] = st.session_state.pop("_goto_mode")
    mode = st.radio("Mode", MODES, key="mode")
    st.divider()
    bankroll = st.number_input("Bankroll ($)", value=100, step=10, min_value=1)
    risk_mult = st.selectbox("Risk Profile", [0.5, 1.0], index=0,
                             format_func=lambda x: "Conservative (¼ Kelly)" if x == 0.5 else "Aggressive (½ Kelly)")
    min_ev = st.slider("Min EV to bet (%)", 1, 10, 3, help="Buffer above break-even before a bet is suggested. 3% ≈ the juice on a -110 line.") / 100
    model_w = st.slider("Model weight (%)", 0, 50, 10, help="How much the model moves you off the market price. Walk-forward backtest (BACKTEST.md) found no feature set beats the closing line, so the default is deliberately small.") / 100
    max_wager = bankroll * risk_mult * 0.05
    st.metric("Max wager (5% cap × risk)", f"${max_wager:.2f}")
    st.divider()
    if st.button("🔄 Clear Cache & Reload"):
        st.cache_resource.clear(); st.cache_data.clear(); st.rerun()

# ---------------------------------------------------------------- secrets
ODDS_API_KEY = E.odds_api_key()   # .streamlit/secrets.toml, else env

try:
    import nfl_data_py as nfl
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
except ImportError as e:
    st.error(f"Missing libraries: {e}. Run: pip install -r requirements.txt"); st.stop()

# ---------------------------------------------------------------- data services
LIVE_BOOKS = ("draftkings", "fanduel", "betmgm")   # the card shows best price among these; the Scanner tab uses all books


def fetch_live_odds(api_key):
    """Best moneyline per side among LIVE_BOOKS + synthetic hold + DK home spread, keyed (home, away).
    Derived from the same cached all-markets payload the Scanner uses, so the app makes ONE Odds API call per refresh."""
    payload, msg = fetch_all_markets(api_key)
    if not payload:
        return {}, msg + " — using schedule lines."
    q = SC.parse_payload(payload, SC.TEAM_MAP)
    out = {}
    for gid, g in q[q.book.isin(LIVE_BOOKS)].groupby("game"):
        ml = g[g.market == "h2h"]
        if ml.empty:
            continue
        bh = ml[ml.side == "home"].sort_values("dec").iloc[-1]
        ba = ml[ml.side == "away"].sort_values("dec").iloc[-1]
        sp = g[(g.market == "spreads") & (g.side == "home")]
        out[(bh.home, bh.away)] = {
            "books": {b: {"home_dec": x[x.side == "home"].dec.iloc[0], "away_dec": x[x.side == "away"].dec.iloc[0]} for b, x in ml.groupby("book") if len(x) == 2},
            "home_dec": bh.dec, "home_book": bh.book, "away_dec": ba.dec, "away_book": ba.book,
            "hold": E.synthetic_hold(bh.dec, ba.dec),
            "spread_home": float(sp[sp.book == "draftkings"].point.iloc[0]) if (sp.book == "draftkings").any() else (float(sp.point.iloc[0]) if len(sp) else None),
            "ts": datetime.datetime.now().strftime("%H:%M"),
        }
    return out, f"Live: {len(out)} games · {msg.split(' · ', 1)[-1]}"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_markets(api_key):
    """Every US book, h2h + spreads + totals. Costs 3 API credits; cached 30 minutes."""
    if not api_key:
        return [], "No ODDS_API_KEY"
    try:
        r = requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/",
                         params={"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "decimal", "apiKey": api_key}, timeout=15)
        r.raise_for_status()
        return r.json(), f"{len(r.json())} games · {r.headers.get('x-requests-remaining', '?')} credits left · {datetime.datetime.now():%H:%M}"
    except Exception as ex:
        return [], f"Odds API failed ({type(ex).__name__})"


@st.cache_resource(ttl=86400, show_spinner=False)
def load_mispricing():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mispricing.json")
    if os.path.exists(p):
        return pd.read_json(p)
    g = MP.normalise(nfl.import_schedules(list(range(1999, datetime.date.today().year + 1))))
    return MP.run_hypotheses(g)


get_forecast = st.cache_data(ttl=3600, show_spinner=False)(ST.get_forecast)   # stadiums.py holds the table + Open-Meteo call


def _read_nflverse(kind, year):
    url = f"https://github.com/nflverse/nflverse-data/releases/download/{kind}_{year}.csv.gz"
    return pd.read_csv(url, compression="gzip", low_memory=False)


@st.cache_resource(ttl=3600, show_spinner=False)
def load_nfl_data():
    today = datetime.date.today()
    season = today.year if today.month >= 3 else today.year - 1
    status = {}

    try:
        sched = nfl.import_schedules([season])
    except Exception as ex:
        sched = pd.DataFrame(); status["schedule"] = f"❌ {type(ex).__name__}"
    if not sched.empty:
        status["schedule"] = f"✅ {season}: {len(sched)} games, {int(sched['home_score'].notna().sum())} played"

    hfa = {}
    try:
        hfa = E.compute_team_home_advantage(nfl.import_schedules([season - 3, season - 2, season - 1]))
        status["hfa"] = f"✅ excess HFA from {season-3}–{season-1}"
    except Exception as ex:
        status["hfa"] = f"⚠️ {type(ex).__name__}"

    pbp_all, weekly_all = [], []
    for yr in (season - 2, season - 1, season):
        try:
            p = nfl.import_pbp_data([yr], cache=False)
            if p.empty:
                raise ValueError("empty")
        except Exception:
            try:
                p = _read_nflverse("pbp/play_by_play", yr)
            except Exception:
                p = pd.DataFrame()
        if not p.empty:
            pbp_all.append(p)
        try:  # nflverse moved weekly player stats in 2025; nfl_data_py's old URL 404s
            w = _read_nflverse("stats_player/stats_player_week", yr)
            if "recent_team" not in w.columns and "team" in w.columns:
                w = w.rename(columns={"team": "recent_team"})
            weekly_all.append(w[w["season_type"] == "REG"] if "season_type" in w.columns else w)
        except Exception:
            pass
    pbp = pd.concat(pbp_all) if pbp_all else pd.DataFrame()
    weekly = pd.concat(weekly_all) if weekly_all else pd.DataFrame()
    status["pbp"] = f"✅ seasons {sorted(pbp['season'].unique().tolist())}" if not pbp.empty else "❌ no play-by-play"
    status["weekly"] = f"✅ seasons {sorted(weekly['season'].unique().tolist())}, thru wk {int(weekly[weekly['season']==weekly['season'].max()]['week'].max())}" if not weekly.empty else "❌ no weekly stats"

    clf, team_stats, qb_stats, val = None, pd.DataFrame(), pd.DataFrame(), {}
    feats, feat_names, rating_now = pd.DataFrame(), [], {}
    if not pbp.empty:
        pbp["defteam"] = pbp["defteam"].fillna(pbp["posteam"])
        pas, run = pbp[pbp["play_type"] == "pass"], pbp[pbp["play_type"] == "run"]
        def agg(df, col, **kw):
            return df.groupby(["season", col], dropna=True).agg(**kw).reset_index().rename(columns={col: "team"})
        team_stats = agg(pas, "posteam", epa_pass_off=("epa", "mean"), pass_yds_off=("yards_gained", "sum"))
        for part in [agg(run, "posteam", epa_rush_off=("epa", "mean"), rush_yds_off=("yards_gained", "sum")),
                     pbp.groupby(["season", "posteam"])["game_id"].nunique().reset_index().rename(columns={"game_id": "games", "posteam": "team"}),
                     agg(pbp, "posteam", interceptions=("interception", "sum"), fumbles_lost=("fumble_lost", "sum")),
                     agg(pas, "defteam", epa_pass_def=("epa", "mean"), pass_yds_def=("yards_gained", "sum")),
                     agg(run, "defteam", epa_rush_def=("epa", "mean"), rush_yds_def=("yards_gained", "sum")),
                     agg(pbp, "defteam", def_int=("interception", "sum"), def_fumbles=("fumble_lost", "sum"))]:
            team_stats = team_stats.merge(part, on=["season", "team"], how="outer")
        for k in ("pass_off", "rush_off", "pass_def", "rush_def"):
            team_stats[f"avg_{k}"] = team_stats[f"{k.split('_')[0]}_yds_{k.split('_')[1]}"] / team_stats["games"]
        team_stats["avg_tos_off"] = (team_stats["interceptions"] + team_stats["fumbles_lost"]) / team_stats["games"]
        team_stats["avg_tos_def"] = (team_stats["def_int"] + team_stats["def_fumbles"]) / team_stats["games"]
        team_stats["epa_net_pass"] = team_stats["epa_pass_off"] - team_stats["epa_pass_def"]
        team_stats["epa_net_rush"] = team_stats["epa_rush_off"] - team_stats["epa_rush_def"]

        if "cpoe" not in pbp.columns:
            pbp["cpoe"] = 0.0
        qb_stats = (pas.groupby(["season", "posteam", "passer_player_name"])
                    .agg(qb_epa=("epa", "mean"), qb_cpoe=("cpoe", "mean"), dropbacks=("play_id", "count"))
                    .reset_index().sort_values("dropbacks", ascending=False).drop_duplicates(["season", "posteam"]))

        # Train on leakage-free pre-game features (features.py); walk-forward validated in backtest.py
        try:
            hist = nfl.import_schedules([season - 2, season - 1, season])
            try:
                inj_hist = nfl.import_injuries([season - 2, season - 1, season])
            except Exception:
                inj_hist = None
            feats = F.build_game_features(hist, pbp, inj_hist)
            feat_names = F.feature_list(["market", "team", "qb", "context", "injuries"])
            tr = feats.dropna(subset=["home_win"]).sort_values(["season", "week"])
            if len(tr) > 60:
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))
                cut = int(len(tr) * 0.8)
                m = mk().fit(tr[feat_names].iloc[:cut], tr["home_win"].iloc[:cut])
                pv = m.predict_proba(tr[feat_names].iloc[cut:])[:, 1]
                yv = tr["home_win"].iloc[cut:]
                val = {"n_val": int(len(yv)), "acc": float(((pv > 0.5) == yv).mean()),
                       "logloss_model": float(log_loss(yv, pv)), "logloss_market": float(log_loss(yv, tr["p_mkt"].iloc[cut:]))}
                clf = mk().fit(tr[feat_names], tr["home_win"])
                status["model"] = f"✅ trained on {len(tr)} games, {len(feat_names)} pre-game features"
            weekly_ts = F.weekly_team_stats(pbp)
            ratings = F.rolling_team_ratings(weekly_ts)
            rating_now = {t: F.latest_team_rating(ratings, weekly_ts, t) for t in weekly_ts["team"].unique()}
        except Exception as ex:
            status["model"] = f"⚠️ {type(ex).__name__}: {str(ex)[:80]}"

    return {"clf": clf, "team_stats": team_stats, "weekly": weekly, "sched": sched, "qb_stats": qb_stats,
            "hfa": hfa, "status": status, "val": val, "season": season,
            "feats": feats.set_index("game_id") if not feats.empty else feats, "feat_names": feat_names, "rating_now": rating_now}


@st.cache_data(ttl=1800, show_spinner=False)
def load_injuries():
    try:
        return I.fetch_sleeper(), "✅ Sleeper live"
    except Exception as ex:
        return pd.DataFrame(columns=["full_name", "team", "position", "injury_status", "injury_body_part", "injury_notes", "depth_chart_order"]), f"⚠️ {type(ex).__name__}"


with st.spinner("Loading schedules, play-by-play, injuries and odds…"):
    D = load_nfl_data()
    INJ, inj_msg = load_injuries()
    live_odds, odds_msg = fetch_live_odds(ODDS_API_KEY)

with st.sidebar:
    st.divider(); st.markdown("### 💾 Data Health")
    for k, v in D["status"].items():
        st.caption(f"**{k}:** {v}")
    st.caption(f"**odds:** {odds_msg}")
    st.caption(f"**injuries:** {inj_msg}")
    if D["val"]:
        v = D["val"]
        beats = v["logloss_model"] < v["logloss_market"]
        st.markdown("### 🤖 Model Vitality")
        st.caption(f"Holdout n={v['n_val']} · acc {v['acc']:.1%} · log-loss model {v['logloss_model']:.3f} vs market {v['logloss_market']:.3f}")
        st.caption(("✅ beats market on holdout" if beats else "⚠️ market alone is sharper — keep model weight low") + " · pre-game features only; see BACKTEST.md for the walk-forward test.")

if D["sched"].empty:
    st.error("Unable to load the NFL schedule. Check your connection and try again."); st.stop()

# ---------------------------------------------------------------- cockpit helpers
TS, WK, QB, HFA = D["team_stats"], D["weekly"], D["qb_stats"], D["hfa"]


def latest_team_row(team):
    if TS.empty:
        return None
    r = TS[TS["team"] == team].sort_values("season", ascending=False).head(1)
    return None if r.empty else r.iloc[0]


def team_leaders(team, n_games=8):
    """Most recent starter QB (by attempts last game) + RB/WR leaders, with recent game logs for props."""
    if WK.empty:
        return {}
    r = WK[WK["recent_team"] == team].sort_values(["season", "week"], ascending=False)
    if r.empty:
        return {}
    out = {}
    last = r[(r["season"] == r["season"].max()) & (r["week"] == r["week"].max())]
    for pos, col, filt in (("QB", "passing_yards", last.sort_values("attempts", ascending=False)),
                           ("RB", "rushing_yards", r[r["position"] == "RB"].groupby("player_display_name")[["rushing_yards"]].sum().sort_values("rushing_yards", ascending=False).reset_index() if "position" in r.columns else r),
                           ("WR", "receiving_yards", r[r["position"].isin(["WR", "TE"])].groupby("player_display_name")[["receiving_yards"]].sum().sort_values("receiving_yards", ascending=False).reset_index() if "position" in r.columns else r)):
        if filt.empty:
            continue
        name = filt.iloc[0]["player_display_name"]
        log = r[r["player_display_name"] == name].head(n_games)[col].tolist()
        out[pos] = {"name": name, "recent": log, "stat": f"Last {len(log)}: " + ", ".join(str(int(x)) for x in log[:5])}
    return out


def qb_metrics(team):
    if QB.empty:
        return None
    r = QB[QB["posteam"] == team].sort_values("season", ascending=False)
    return None if r.empty else r.iloc[0]


def default_sliders(row, weather):
    s = {"h_qb": 5, "h_pwr": 5, "h_def": 5, "a_qb": 5, "a_pwr": 5, "a_def": 5, "rest": 0, "news": 0, "weath": 0, "hfa": 5, "h_inj": [], "a_inj": []}
    for pre, team in (("h", row["home_team"]), ("a", row["away_team"])):
        t = D["rating_now"].get(team)
        if t is None:
            t = latest_team_row(team)
        if t is not None:
            s[f"{pre}_qb"] = E.epa_to_slider(t["epa_pass_off"])
            s[f"{pre}_pwr"] = E.epa_to_slider(t["epa_rush_off"])
            s[f"{pre}_def"] = E.epa_to_slider((t["epa_pass_def"] + t["epa_rush_def"]) / 2, reverse=True)
    s["hfa"] = 5                                   # opt-in: excess HFA never survived the backtest, so the default is neutral
    s["hfa_hint"] = HFA.get(row["home_team"], 0.0)
    if abs(row.get("home_rest", 7) - row.get("away_rest", 7)) > 3:
        s["rest"] = 4
    if not INJ.empty:
        s["news"], s["h_inj"], s["a_inj"] = I.matchup_news_ticks(INJ, row["home_team"], row["away_team"],
                                                                  row.get("home_qb_name"), row.get("away_qb_name"))
    # weather no longer moves SIDES by default (windy dogs failed validation); wind feeds the totals watch rule instead
    return s


def game_features(row):
    """Pre-game feature row for this game (from features.py); falls back to ratings-only."""
    gid = row.get("game_id")
    if not D["feats"].empty and gid in D["feats"].index:
        return D["feats"].loc[gid].to_dict()
    h, a = D["rating_now"].get(row["home_team"]), D["rating_now"].get(row["away_team"])
    if h is None or a is None:
        return {}
    return {"diff_pass": float((h["epa_pass_off"] - a["epa_pass_def"]) - (a["epa_pass_off"] - h["epa_pass_def"])),
            "diff_rush": float((h["epa_rush_off"] - a["epa_rush_def"]) - (a["epa_rush_off"] - h["epa_rush_def"]))}


def gold_projection(home, away):
    h, a = latest_team_row(home), latest_team_row(away)
    if h is None or a is None:
        return None
    ex = lambda o, d: (o + d) / 2
    g = {"h_pass": ex(h["avg_pass_off"], a["avg_pass_def"]), "h_rush": ex(h["avg_rush_off"], a["avg_rush_def"]), "h_to": ex(h["avg_tos_off"], a["avg_tos_def"]),
         "a_pass": ex(a["avg_pass_off"], h["avg_pass_def"]), "a_rush": ex(a["avg_rush_off"], h["avg_rush_def"]), "a_to": ex(a["avg_tos_off"], h["avg_tos_def"])}
    g["h_score"] = 2.5 + g["h_pass"] * 0.045 + g["h_rush"] * 0.06 - g["h_to"] * 4.0
    g["a_score"] = 2.5 + g["a_pass"] * 0.045 + g["a_rush"] * 0.06 - g["a_to"] * 4.0
    return g


def generate_props(home, away, h_p, dh, da):
    props = [parlay.PropLeg(f"{home}_ML", f"{home} to win", dh, h_p, "Team Win", home, f"Model {h_p:.0%}"),
             parlay.PropLeg(f"{away}_ML", f"{away} to win", da, 1 - h_p, "Team Win", away, f"Model {1-h_p:.0%}")]
    for team in (home, away):
        for pos, cat in (("QB", "Passing"), ("RB", "Rushing"), ("WR", "Receiving")):
            p = team_leaders(team).get(pos)
            if not p or not p["recent"]:
                continue
            line = E.suggest_prop_line(p["recent"])
            prob = E.prop_probability(p["recent"], line)
            desc = f"{p['name']} over {line} {cat.lower()} yds"
            props.append(parlay.PropLeg(desc, desc, 1.91, prob, cat, team, p["stat"]))
    return props


def append_log(rec):
    df = pd.DataFrame([rec], columns=E.LOG_COLUMNS)
    df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)


def load_log():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=E.LOG_COLUMNS)
    return pd.read_csv(LOG_PATH)


# ---------------------------------------------------------------- game card
def render_game_card(i, row):
    home, away = row["home_team"], row["away_team"]
    lo = live_odds.get((home, away))
    if lo:
        dh0, da0, src = lo["home_dec"], lo["away_dec"], f"Best of DK/FD/MGM · {lo['ts']}"
    else:
        dh0 = E.american_to_decimal(row["home_moneyline"] if pd.notnull(row.get("home_moneyline")) else -110)
        da0 = E.american_to_decimal(row["away_moneyline"] if pd.notnull(row.get("away_moneyline")) else -110)
        src = "Schedule line (cached)"
    weather = get_forecast(home, sel_date)
    defs = default_sliders(row, weather)
    hq, aq = qb_metrics(home), qb_metrics(away)

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        c1.subheader(f"{away} @ {home}")
        c1.caption(f"{row.get('gametime', '')} ET · {weather['desc'] if weather else ''}" + (f" · Spread {home} {lo['spread_home']:+}" if lo and lo.get("spread_home") is not None else ""))
        ctx_bits = [f"Rest {away} {int(row.get('away_rest', 7))}d / {home} {int(row.get('home_rest', 7))}d"]
        if row.get("div_game") == 1: ctx_bits.append("🏟️ Division game")
        if pd.notna(row.get("home_qb_name")) and pd.notna(row.get("away_qb_name")): ctx_bits.append(f"QBs {row['away_qb_name']} vs {row['home_qb_name']}")
        c1.caption(" · ".join(ctx_bits))
        with c2:
            st.markdown("**QB Matchup**")
            if hq is not None and aq is not None:
                edge = "Even"
                if hq["qb_epa"] > aq["qb_epa"] + 0.05: edge = f"✅ {hq['passer_player_name']}"
                elif aq["qb_epa"] > hq["qb_epa"] + 0.05: edge = f"✅ {aq['passer_player_name']}"
                st.caption(f"Edge: {edge}")
        with st.expander(f"🏥 Injuries (auto-sets News to {defs['news']:+d})"):
            ci_a, ci_h = st.columns(2)
            for col, t, key in ((ci_a, away, "a_inj"), (ci_h, home, "h_inj")):
                col.markdown(f"**{t}**")
                notes = defs.get(key) or []
                col.write("\n".join(f"- {n}" for n in notes) if notes else "- no starters listed")
                if not INJ.empty:
                    others = I.injured_players(INJ, t)
                    others = others[~others["is_starter"]].head(6)
                    if len(others): col.caption("Depth: " + ", ".join(f"{r.full_name} ({r.injury_status})" for r in others.itertuples()))
        with st.expander("QB efficiency (EPA/play, CPOE)"):
            ca, ch = st.columns(2)
            for col, q, t in ((ca, aq, away), (ch, hq, home)):
                if q is not None:
                    col.markdown(f"**{q['passer_player_name']}** ({t})")
                    col.metric("EPA/play", f"{q['qb_epa']:.2f}"); col.metric("CPOE", f"{q['qb_cpoe']:.1f}%")
        st.divider()

        c_odds, c_away, c_home, c_res = st.columns([1, 1.5, 1.5, 1.3])
        with c_odds:
            st.markdown(f"##### 🏦 {src}")
            oa = st.number_input(away, value=E.decimal_to_american(da0), step=5, key=f"oa_{i}")
            oh = st.number_input(home, value=E.decimal_to_american(dh0), step=5, key=f"oh_{i}")
            da, dh = E.american_to_decimal(oa), E.american_to_decimal(oh)
            pmkt = E.no_vig_two_way(dh, da)[0]
            st.progress(pmkt, f"No-vig {home}: {pmkt:.1%}")
            if lo:
                hold = lo["hold"]
                st.caption(f"Best {home}: {lo['home_book']} · Best {away}: {lo['away_book']}")
                st.caption(("✅" if hold <= 0.025 else "⚠️") + f" Synthetic hold {hold:.1%}")
        with c_away:
            st.markdown(f"##### {away} adjust")
            pa_qb = st.slider("QB", 0, 10, defs["a_qb"], key=f"pa_qb_{i}")
            pa_pwr = st.slider("Run / Power", 0, 10, defs["a_pwr"], key=f"pa_pwr_{i}")
            pa_def = st.slider("Defense", 0, 10, defs["a_def"], key=f"pa_def_{i}")
        with c_home:
            st.markdown(f"##### {home} adjust")
            ph_qb = st.slider("QB", 0, 10, defs["h_qb"], key=f"ph_qb_{i}")
            ph_pwr = st.slider("Run / Power", 0, 10, defs["h_pwr"], key=f"ph_pwr_{i}")
            ph_def = st.slider("Defense", 0, 10, defs["h_def"], key=f"ph_def_{i}")

        st.markdown("##### Context")
        cc1, cc2, cc3, cc4 = st.columns(4)
        wr = cc1.slider("Rest weight", 0, 10, defs["rest"], key=f"wr_{i}", help="Market already prices rest; leave at 0 unless a short week is being ignored.")
        news = cc2.slider(f"News (− {away} / + {home})", -10, 10, defs["news"], key=f"wn_{i}", help="Injury / QB news you believe the line hasn't absorbed. ±10 = ±5% win prob.")
        hfa = cc3.slider("Home field (5 = league avg)", 0, 10, defs["hfa"], key=f"hfa_{i}", help="Only this team's EXCESS home edge vs league average is added; the market already has the average.")
        closed = bool(weather and weather["is_closed"])
        ww = cc4.slider("Weather randomness", 0, 10, defs["weath"], disabled=closed, key=f"ww_{i}", help="Wind/rain shrink the favorite's edge toward 50/50.")
        if closed: cc4.caption("🔒 Indoor stadium")

        with c_res:
            sliders = {"pa_qb": pa_qb, "pa_pwr": pa_pwr, "pa_def": pa_def, "ph_qb": ph_qb, "ph_pwr": ph_pwr, "ph_def": ph_def,
                       "wr": wr, "news": news, "ww": ww, "hfa": hfa, "rh": row.get("home_rest", 7), "ra": row.get("away_rest", 7)}
            frow = game_features(row)
            final_p, breakdown = E.calc_win_prob(pmkt, frow, sliders, defs, clf=D["clf"], feature_names=D["feat_names"], market_weight=1 - model_w)
            st.markdown("##### 🚀 Verdict")
            st.metric(f"{home} win prob", f"{final_p:.1%}", delta=f"{final_p - pmkt:+.1%} vs market")
            with st.expander("🧮 Math"):
                for k, v in breakdown.items():
                    st.write(f"- {k}: {v:.1%}" if k.startswith("AI") else f"- {k}: {v:+.1%}")
                st.write(f"- EV {home}: {E.ev_per_dollar(final_p, dh):+.1%} · EV {away}: {E.ev_per_dollar(1-final_p, da):+.1%}")
                if frow:
                    st.caption("Model inputs: " + " · ".join(f"{k} {frow.get(k, 0):+.2f}" for k in ("diff_pass", "diff_rush", "diff_qb", "diff_qb_change", "rest_diff", "diff_inj") if k in frow))
            dec = E.decide(final_p, dh, da, min_ev=min_ev)
            if dec:
                side = home if dec["side"] == "home" else away
                book = (lo["home_book"] if dec["side"] == "home" else lo["away_book"]) if lo else "schedule"
                stake = min(E.stake_for(dec["p"], dec["d"], bankroll, risk_mult), max_wager)
                st.success(f"BET **{side}** \\${stake:.0f} @ {E.decimal_to_american(dec['d']):+d} ({book}) · EV {dec['ev']:+.1%}")
                if st.button("📝 Log this bet", key=f"log_{i}"):
                    append_log({"timestamp": datetime.datetime.now().isoformat(timespec="minutes"), "game_id": row.get("game_id"),
                                "season": row.get("season"), "away": away, "home": home, "market": "ML", "side": side, "book": book,
                                "odds_am": E.decimal_to_american(dec["d"]), "dec": round(dec["d"], 4), "p_mkt": round(pmkt if dec["side"] == "home" else 1 - pmkt, 4),
                                "p_model": round(dec["p"], 4), "ev": round(dec["ev"], 4), "stake": round(stake, 2),
                                "result": None, "pnl": None, "close_am": None, "clv_pct": None})
                    st.toast(f"Logged {side} \\${stake:.0f}")
                if st.button("🛡️ Plan a hedge", key=f"hedge_{i}"):
                    st.session_state["hedge_prefill"] = {"desc": f"{side} ML ({away} @ {home})", "odds": E.decimal_to_american(dec["d"]),
                                                         "stake": round(stake, 2), "p": round(dec["p"], 3),
                                                         "hedge_odds": E.decimal_to_american(da if dec["side"] == "home" else dh)}
                    st.session_state["_goto_mode"] = MODES[3]; st.rerun()
            else:
                best = max(E.ev_per_dollar(final_p, dh), E.ev_per_dollar(1 - final_p, da))
                st.info(f"No edge (best EV {best:+.1%}, need {min_ev:+.0%})")

        with st.expander("🏆 Simple projection (yards / turnovers / score)"):
            g = gold_projection(home, away)
            if g:
                ca, ch = st.columns(2)
                for col, t, k in ((ca, away, "a"), (ch, home, "h")):
                    col.markdown(f"**{t}**"); col.write(f"Pass {g[k+'_pass']:.0f} · Rush {g[k+'_rush']:.0f} · TO {g[k+'_to']:.1f}"); col.metric("Score", f"{g[k+'_score']:.1f}")
                diff = g["h_score"] - g["a_score"]
                st.caption(f"Projection: {home if diff > 0 else away} by {abs(diff):.1f}")
            else:
                st.caption("Insufficient data")

        if st.button(f"🧠 Strategy Lab: {away} @ {home}", key=f"sl_{i}"):
            st.session_state.update({"sl_active": True, "sl_home": home, "sl_away": away, "sl_h_prob": final_p,
                                     "sl_dh": dh, "sl_da": da, "sl_legs": [], "sl_pool": None}); st.rerun()


# ---------------------------------------------------------------- strategy lab
def render_strategy_lab():
    home, away, h_p = st.session_state["sl_home"], st.session_state["sl_away"], st.session_state["sl_h_prob"]
    st.markdown(f"## 🧠 Strategy Lab: {away} @ {home}")
    if st.button("← Back to schedule"):
        st.session_state["sl_active"] = False; st.rerun()
    if not st.session_state.get("sl_pool"):
        st.session_state["sl_pool"] = generate_props(home, away, h_p, st.session_state["sl_dh"], st.session_state["sl_da"])
    pool, legs = st.session_state["sl_pool"], st.session_state["sl_legs"]
    st.caption("Prop lines default to the player's recent median (a coin flip by construction) at -110. Edit a leg's line/odds to match your book before trusting the EV.")
    st.divider()
    cb, ct = st.columns([2, 1])
    with cb:
        if not legs:
            st.subheader("Step 1: pick the winner")
            c1, c2 = st.columns(2)
            for col, t in ((c1, home), (c2, away)):
                ml = next(p for p in pool if p.leg_id == f"{t}_ML")
                if col.button(f"🏆 {t} ({ml.p_model:.0%})"):
                    legs.append(ml); st.rerun()
        elif len(legs) < 5:
            st.subheader(f"Correlated with: {legs[-1].description}")
            fits = parlay.ParlayMath.find_best_additions(legs, pool, top_n=3)
            if fits:
                for col, leg in zip(st.columns(3), fits):
                    with col, st.container(border=True):
                        st.markdown(f"**{leg.description}**"); st.caption(f"{leg.category} · P {leg.p_model:.0%} · {leg.recent_stat}")
                        if st.button("➕ Add", key=f"add_{leg.leg_id}_{len(legs)}"):
                            legs.append(leg); st.rerun()
            else:
                st.info("No more correlated props.")
        else:
            st.success("Ticket full (5 legs).")
        if legs and st.button("🔄 Reset ticket"):
            st.session_state["sl_legs"] = []; st.rerun()
    with ct:
        st.markdown("### 🎫 Ticket")
        if not legs:
            st.info("Empty"); return
        for n, leg in enumerate(legs, 1):
            st.write(f"{n}. {leg.description}")
        book_am = st.number_input("Book parlay odds (American, 0 = estimate)", value=0, step=10)
        res = parlay.ParlayMath.calculate_ticket(legs, bankroll, book_decimal=E.american_to_decimal(book_am) if book_am else None)
        st.metric("Payout odds", f"{E.decimal_to_american(res.final_odds):+d}" + (" (est.)" if res.odds_is_estimate else ""))
        st.metric("Joint win prob", f"{res.win_prob:.1%}")
        (st.success if res.ev > 0 else st.warning)(f"EV {res.ev:+.1%}")
        if res.ev > 0:
            st.markdown(f"### Bet \\${res.kelly_stake:.0f}")
            st.caption("⅕ Kelly, capped at 2% of bankroll — parlays are where sizing mistakes compound.")


# ---------------------------------------------------------------- hedge desk
def render_hedge_desk():
    st.markdown("## 🛡️ Hedge Desk")
    st.caption("Hedging buys certainty with EV. Every option below shows what you keep if either side wins, so the trade-off is explicit.")
    pre = st.session_state.get("hedge_prefill", {})
    tab1, tab2, tab3 = st.tabs(["Moneyline / any single bet", "Spread middle finder", "Parlay last leg"])

    with tab1:
        c1, c2 = st.columns(2)
        desc = c1.text_input("Open bet", value=pre.get("desc", "e.g. BUF ML"))
        odds = c1.number_input("Odds you took (American)", value=int(pre.get("odds", -110)), step=5)
        stake = c1.number_input("Stake ($)", value=float(pre.get("stake", 25.0)), step=5.0, min_value=0.0)
        p = c2.slider("Your win prob for that bet now (%)", 1, 99, int(round(pre.get("p", 0.5) * 100))) / 100
        hedge_odds = c2.number_input("Other side, best current odds (American)", value=int(pre.get("hedge_odds", -110)), step=5)
        d_o, d_h = E.american_to_decimal(odds), E.american_to_decimal(hedge_odds)
        opts = H.hedge_options(stake, d_o, p, d_h, bankroll)
        df = pd.DataFrame([{"Option": o.name, "Hedge stake": o.hedge_stake, f"If {desc} wins": o.profit_if_orig_wins,
                            "If hedge wins": o.profit_if_hedge_wins, "EV (your prob)": o.ev, "Worst case": o.worst_case, "Note": o.note} for o in opts])
        st.dataframe(df, hide_index=True, use_container_width=True)
        cost = H.hedge_cost_of_certainty(p, stake, d_o, d_h)
        lp = H.locked_profit(stake, d_o, d_h)
        st.info(f"Full lock guarantees **\\${lp:+.2f}**. It costs **\\${cost:.2f}** of EV versus letting it ride, if your {p:.0%} is right. "
                + ("The Kelly line says don't hedge at all." if opts[1].hedge_stake == 0 else f"Kelly-optimal hedge: **\\${opts[1].hedge_stake:.2f}**."))

    with tab2:
        st.caption("You hold team X at a spread; the line moved and you can now take the opponent Y at a better number. If both cash, that's a middle.")
        c1, c2, c3 = st.columns(3)
        sx = c1.number_input("Your ticket: X spread", value=-3.0, step=0.5)
        ox = c1.number_input("Your odds (American)", value=-110, step=5)
        stx = c1.number_input("Your stake on X ($)", value=25.0, step=5.0)
        sy = c2.number_input("Available now: Y spread", value=6.0, step=0.5)
        oy = c2.number_input("Y odds (American)", value=-110, step=5)
        sty = c2.number_input("Stake on Y ($)", value=25.0, step=5.0)
        cur = c3.number_input("Current market spread on X", value=-4.5, step=0.5, help="Used as the centre of the margin distribution (σ = 13.5).")
        win = H.middle_window(sx, sy)
        if not win:
            st.warning("No middle: the two numbers overlap. Taking Y here is a plain hedge, use the first tab.")
        else:
            pb, px, py = H.middle_breakdown(win, cur)
            ev = H.middle_ev(stx, E.american_to_decimal(ox), sty, E.american_to_decimal(oy), pb, px, py)
            st.metric("Middle window (X wins by)", f"{win[0]:g} < margin < {win[1]:g}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P(both cash)", f"{pb:.1%}"); m2.metric("P(only X)", f"{px:.1%}"); m3.metric("P(only Y)", f"{py:.1%}"); m4.metric("EV of adding Y", f"${ev:+.2f}")
            (st.success if ev > 0 else st.warning)("Positive-EV middle." if ev > 0 else "This middle doesn't pay for its juice; skip unless you want the variance cut.")

    with tab3:
        c1, c2 = st.columns(2)
        pst = c1.number_input("Parlay stake ($)", value=10.0, step=1.0)
        pod = c1.number_input("Parlay payout odds (American)", value=1200, step=50)
        pl = c2.slider("Your prob the last leg hits (%)", 1, 99, 50) / 100
        ho = c2.number_input("Opposite side of the last leg (American)", value=-110, step=5)
        opts = H.parlay_last_leg_options(pst, E.american_to_decimal(pod), pl, E.american_to_decimal(ho), bankroll)
        st.dataframe(pd.DataFrame([{"Option": o.name, "Hedge stake": o.hedge_stake, "If parlay hits": o.profit_if_orig_wins,
                                    "If hedge hits": o.profit_if_hedge_wins, "EV": o.ev, "Worst case": o.worst_case} for o in opts]),
                     hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- scanner
def _quotes():
    payload, msg = fetch_all_markets(ODDS_API_KEY)
    return SC.parse_payload(payload, SC.TEAM_MAP), msg


def render_scanner():
    st.markdown("## 🔎 Pricing-error scanner")
    q, msg = _quotes()
    show_offshore = st.checkbox("Include offshore books in the off-consensus list", value=False, help="Offshore quotes always feed the consensus; this only controls whether they are listed as bettable.")
    st.caption(f"Live books: {msg}. Consensus = mean no-vig probability across every book quoting the same line; a book off consensus is a pricing error or a stale line. Move fast: these close in minutes.")
    if q.empty:
        st.warning("No live quotes (no API key or the API failed)."); 
    else:
        e = SC.consensus_edges(q, min_books=3)
        hot = e[(e.edge >= SC.EDGE_FLAG) & (show_offshore | ~e.book.isin(SC.OFFSHORE))]
        st.markdown(f"### 💰 Off-consensus prices (edge ≥ {SC.EDGE_FLAG:.1%}) — {len(hot)} found across {q.book.nunique()} books")
        if len(hot):
            show = hot[["game", "market", "side", "point", "book", "am", "fair_am", "n_books", "edge"]].copy()
            show["edge"] = show["edge"].map(lambda v: f"{v:+.1%}")
            st.dataframe(show.head(25), hide_index=True, use_container_width=True)
        else:
            st.info("Nothing off consensus right now. That is the normal state of an efficient market — check again close to kickoff and after injury news.")
        ar = SC.arbs(q)
        ar = ar[ar.hold < 0.005] if len(ar) else ar
        st.markdown(f"### 🔁 Arbitrage / near-zero hold — {len(ar)}")
        if len(ar):
            ar["hold"] = ar["hold"].map(lambda v: f"{v:+.2%}")
            st.dataframe(ar.head(15), hide_index=True, use_container_width=True)
        else:
            st.caption("No cross-book arbs.")
        md = SC.middles(q)
        st.markdown(f"### ↔️ Cross-book spread middles — {len(md)}")
        if len(md):
            st.dataframe(md.head(15), hide_index=True, use_container_width=True)
        else:
            st.caption("No middles worth 5%+.")

    st.divider()
    st.markdown("### 📜 Historical-bias matches this week")
    res = load_mispricing()
    nv, nw = int(res.validated.sum()), int(res.watch.sum())
    st.caption(f"{len(res)} pre-registered hypotheses tested 1999–{D['season']} at the closing line with false-discovery control: **{nv} validated**, {nw} on the watch list. Full table in MISPRICING.md.")
    today = datetime.date.today()
    up = D["sched"][(pd.to_datetime(D["sched"]["gameday"]).dt.date >= today) & (pd.to_datetime(D["sched"]["gameday"]).dt.date <= today + datetime.timedelta(days=7))].copy()
    if up.empty:
        st.info("No games in the next 7 days."); return
    for c in ("temp", "wind"):
        if c not in up.columns:
            up[c] = np.nan
    for i, r in up.iterrows():                      # forecast stands in for game-day weather
        w = get_forecast(r["home_team"], pd.to_datetime(r["gameday"]).date())
        if w and not w["is_closed"]:
            up.at[i, "temp"], up.at[i, "wind"] = w["temp"], w["wind"]
    g = MP.normalise(up)
    matches = MP.match_rules(g, res)
    if not matches:
        st.info("No upcoming game triggers a validated or watch-list rule."); return
    m = pd.DataFrame(matches)
    m["game"] = m.game_id.map(lambda gid: gid.split("_", 2)[-1].replace("_", " @ "))
    m["val_roi"] = m["val_roi"].map(lambda v: f"{v:+.1%}")
    st.dataframe(m[["game", "tier", "hypothesis", "market", "side", "rule", "val_roi", "val_n"]], hide_index=True, use_container_width=True)
    st.caption("Watch-list rules are suggestive, not proven. Log any bet you make on one so the Bet Log can grade the rule live.")


# ---------------------------------------------------------------- promo hedger
def render_promo():
    st.markdown("## 🎁 Promo Hedger")
    st.caption("Turn a DraftKings promo into locked profit: the promo bet at DK, the opposite side of the SAME line at another book, sized so you win the same amount either way. Profit is fixed at placement; the risks are operational (line moves between clicks, voids, limits, account restrictions).")
    c1, c2, c3 = st.columns(3)
    ptype = c1.selectbox("Promo type", list(PE.PROMO_TYPES), format_func=lambda k: PE.PROMO_TYPES[k])
    amount = c1.number_input("Promo amount ($)", value=50.0, step=5.0, min_value=1.0)
    promo = {"type": ptype, "amount": amount}
    if ptype == "profit_boost":
        promo["boost_pct"] = c2.number_input("Boost (%)", value=50, step=5, min_value=1) / 100
        mw = c2.number_input("Max extra winnings ($, 0 = none)", value=0.0, step=5.0)
        promo["max_win"] = None if mw <= 0 else mw
    elif ptype == "odds_boost":
        promo["boosted_dec"] = E.american_to_decimal(c2.number_input("Boosted odds (American)", value=200, step=5))
    elif ptype == "no_sweat":
        promo["refund_conv"] = c2.slider("Value of the refund bonus bet (%)", 50, 80, 70) / 100
    elif ptype == "qualifying":
        face = c2.number_input("Bonus unlocked ($ face)", value=150.0, step=10.0)
        promo["bonus_value"] = face * c2.slider("…worth (% of face when hedged)", 50, 80, 70) / 100
    min_am = c3.number_input("Min odds allowed by promo (American, 0 = any)", value=0, step=5)
    crowns = c3.number_input("DK Crowns value per $ wagered (0 = ignore)", value=0.0, step=0.001, format="%.3f", help="Dynasty rewards: ~1 Crown per $ on straight bets; value them yourself (often well under 1¢).")
    min_dec = E.american_to_decimal(min_am) if min_am else 1.0

    q, msg = _quotes()
    st.caption(f"Live books: {msg}")
    books = sorted(b for b in q.book.unique() if b not in SC.OFFSHORE) if not q.empty else []
    venue_mode = st.radio("Hedge venue", ["DraftKings only (Oregon: the one legal app)", "Any regulated book (best price)"], horizontal=True,
                          help="Oregon has one legal sportsbook app, so the hedge goes on DK's own opposite side (you pay the hold twice, the promo still converts). Use the manual calculator with a Kalshi price for a cheaper hedge.")
    allowed = {"draftkings"} if venue_mode.startswith("DraftKings") else set(books) - {"draftkings"}
    plans = PE.best_plans(promo, q, promo_book="draftkings", min_dec=min_dec, top_n=8, hedge_books=allowed) if allowed else []
    if not plans:
        st.warning("No hedgeable DraftKings line at the venues you allowed (or no API key). In Oregon tick 'Allow hedging at DraftKings itself' or use the manual calculator with a Kalshi price."); 
    else:
        t = PE.plans_table(plans)
        best = plans[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Best locked profit", f"${best.locked_profit:+.2f}")
        m2.metric("Conversion of promo", f"{best.conversion:.0%}")
        m3.metric("Crowns bonus (est.)", f"${best.bet_stake * crowns:.2f}")
        st.markdown("#### Ranked plans")
        show = t[["game", "market", "bet_side", "bet_point", "bet_dec", "hedge_book", "hedge_side", "hedge_dec", "bet_stake", "hedge_stake", "profit_if_bet_wins", "profit_if_hedge_wins", "locked_profit", "conversion"]].copy()
        for c in ("bet_dec", "hedge_dec"):
            show[c] = show[c].map(lambda d: f"{E.decimal_to_american(d):+d}")
        show["conversion"] = show["conversion"].map(lambda v: f"{v:.0%}")
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.markdown("#### Tickets (copy these exactly)")
        for p in plans[:3]:
            st.code(p.ticket())
        st.caption("Place the DK leg first, then re-check the hedge price before placing it. If the hedge line moved, re-run this page: never place the hedge at a worse number than shown.")

    st.divider()
    st.markdown("#### Manual calculator (no API needed)")
    c1, c2, c3 = st.columns(3)
    d_bet = E.american_to_decimal(c1.number_input("DK odds on your promo bet (American)", value=250, step=5))
    d_h = E.american_to_decimal(c2.number_input("Other book, opposite side (American)", value=-300, step=5))
    h, pa, pb, note = PE.plan_for(promo, d_bet, d_h)
    c3.metric("Hedge stake", f"${h:.2f}")
    c3.metric("Profit either way", f"${min(pa, pb):+.2f}", help=f"if promo bet wins ${pa:+.2f} · if hedge wins ${pb:+.2f} · {note}")


# ---------------------------------------------------------------- bet log
def render_bet_log():
    st.markdown("## 📒 Bet Log")
    log = E.grade_log(load_log(), D["sched"])
    if log.empty:
        st.info("No bets logged yet. Use **📝 Log this bet** on a game card."); return
    log.to_csv(LOG_PATH, index=False)
    s = E.log_summary(log)
    m = st.columns(5)
    m[0].metric("Bets", f"{s['bets']} ({s['graded']} graded)"); m[1].metric("Record", f"{s['wins']}-{s['losses']}")
    m[2].metric("P&L", f"${s['pnl']:+.2f}"); m[3].metric("ROI", f"{s['roi']:+.1%}")
    m[4].metric("Avg CLV", f"{s['avg_clv']:+.1%}" if s["avg_clv"] is not None else "—",
                help="Closing-line value: your no-vig edge vs the closing line. Positive over 50+ bets = real edge.")
    st.caption("PROGRAM.md stopping rule: evaluate after 50 graded bets. Results auto-grade from nflverse final scores; CLV uses the schedule's closing moneyline.")
    edited = st.data_editor(log, hide_index=True, use_container_width=True, num_rows="dynamic")
    if st.button("💾 Save edits"):
        edited.to_csv(LOG_PATH, index=False); st.toast("Saved")
    st.download_button("⬇️ Download CSV", log.to_csv(index=False), "bet_log.csv", "text/csv")


# ---------------------------------------------------------------- router
if mode == MODES[1]:
    render_scanner()
elif mode == MODES[2]:
    render_promo()
elif mode == MODES[3]:
    render_hedge_desk()
elif mode == MODES[4]:
    render_bet_log()
elif st.session_state.get("sl_active"):
    render_strategy_lab()
else:
    c_date, c_msg = st.columns([1, 3])
    sel_date = c_date.date_input("📅 Game date", datetime.date.today())
    c_msg.success(f"Season {D['season']} · {odds_msg}")
    sched = D["sched"].copy()
    sched["gameday"] = sched["gameday"].astype(str)
    games = sched[sched["gameday"] == str(sel_date)]
    if games.empty:
        nxt = sched[(sched["gameday"] > str(sel_date)) & sched["home_score"].isna()]["gameday"].min()
        st.warning(f"No games on {sel_date}." + (f" Next slate: {nxt}" if isinstance(nxt, str) else ""))
    else:
        st.markdown(f"### 🔥 {len(games)} game{'s' if len(games) > 1 else ''}")
        for i, row in games.iterrows():
            render_game_card(i, row)
