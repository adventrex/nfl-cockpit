import streamlit as st
import pandas as pd
import numpy as np
import datetime
import math
import random
import requests
import streamlit.components.v1 as components
from math import erf, sqrt

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="NFL Edge Cockpit Pro", page_icon="🏈", layout="wide")

# --- HACKER LOADING SCREEN HTML ---
LOADING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    :root { --bg-color: #0e1117; --primary-neon: #20f7ff; --accent-neon: #ff4dd2; --code-font: monospace; }
    body { background-color: var(--bg-color); color: #cdd4ff; font-family: var(--code-font); overflow: hidden; margin: 0; padding: 20px; }
    .loading-shell { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 400px; gap: 20px; text-align: center; }
    .title-block h1 { color: var(--primary-neon); text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 10px; text-shadow: 0 0 10px rgba(32, 247, 255, 0.5); }
    .progress-bar { width: 300px; height: 4px; background: #333; border-radius: 2px; overflow: hidden; position: relative; margin: 0 auto; }
    .progress-inner { position: absolute; top: 0; left: 0; height: 100%; width: 100%; background: var(--primary-neon); animation: progress 2s infinite ease-in-out; transform-origin: left; }
    @keyframes progress { 0% { transform: scaleX(0); } 50% { transform: scaleX(0.7); } 100% { transform: scaleX(0); transform-origin: right; } }
    .code-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; width: 100%; max-width: 800px; font-size: 10px; opacity: 0.7; text-align: left; }
    .code-col { border: 1px solid #333; padding: 10px; height: 150px; overflow: hidden; position: relative; background: #000; }
    .code-line { color: var(--accent-neon); white-space: nowrap; animation: scroll 3s infinite linear; }
    @keyframes scroll { 0% { transform: translateY(100%); opacity: 0; } 20% { opacity: 1; } 100% { transform: translateY(-150%); opacity: 0; } }
  </style>
</head>
<body>
  <div class="loading-shell">
    <div class="title-block">
      <h1>NFL Edge Engine v2.5</h1>
      <div style="font-size: 12px; color: #52ff6b;">CALIBRATING LOGISTIC REGRESSION MODEL...</div>
      <div class="progress-bar"><div class="progress-inner"></div></div>
    </div>
    <div class="code-grid">
      <div class="code-col">
        <div class="code-line">IMPORTING nfl_data_py...</div>
        <div class="code-line" style="animation-delay: 0.5s">FETCHING 2024-2025 PBP...</div>
        <div class="code-line" style="animation-delay: 1.0s">CALCULATING EPA/PLAY...</div>
        <div class="code-line" style="animation-delay: 1.5s">MERGING SCHEDULE DATA...</div>
      </div>
      <div class="code-col">
        <div class="code-line">OPTIMIZING KELLY CRITERION...</div>
        <div class="code-line" style="animation-delay: 0.7s">SYNCING DRAFTKINGS API...</div>
        <div class="code-line" style="animation-delay: 1.2s">REMOVING VIG...</div>
        <div class="code-line" style="animation-delay: 1.8s">CHECKING LINE MOVEMENT...</div>
      </div>
      <div class="code-col">
        <div class="code-line">BUILDING CORRELATION MATRIX...</div>
        <div class="code-line" style="animation-delay: 0.3s">SCANNING INJURY REPORTS...</div>
        <div class="code-line" style="animation-delay: 0.9s">DETECTING MARKET INEFFICIENCIES...</div>
        <div class="code-line" style="animation-delay: 1.4s">RUNNING MONTE CARLO SIMS...</div>
      </div>
    </div>
  </div>
</body>
</html>
"""

# --- INITIALIZE LOADING PLACEHOLDER ---
loading_placeholder = st.empty()
with loading_placeholder:
    components.html(LOADING_HTML, height=450)

with st.sidebar:
    st.title("🏈 NFL Cockpit")
    bankroll = st.number_input("Bankroll ($)", value=100, step=10)
    kelly = st.selectbox("Risk Profile", [0.5, 1.0], index=0, format_func=lambda x: "Conservative (0.5x)" if x==0.5 else "Aggressive (1.0x)")
    
    max_wager = bankroll * kelly * 0.05
    st.metric("Max Wager Limit (5% Cap)", f"${max_wager:.2f}", help="The absolute maximum bet size allowed per game to prevent ruin.")
    
    st.divider()
    status_slot = st.empty()
    
    # Placeholder for Years Loaded
    years_slot = st.empty()

# Date Picker
c_date, c_status = st.columns([1, 3])
sel_date = c_date.date_input("📅 Select Game Date", datetime.date.today())

# ==========================================
# 2. IMPORTS & API KEYS
# ==========================================
ODDS_API_KEY = "bb2b1af235a1f0273f9b085b82d6be81"

try:
    import nfl_data_py as nfl
    from sklearn.linear_model import LogisticRegression
    import parlay_engine as parlay
except ImportError as e:
    st.error(f"Missing Libraries: {e}. Please run: pip install nfl_data_py scikit-learn pandas numpy requests")
    st.stop()

# ==========================================
# 3. SECONDARY DATA SOURCES
# ==========================================
class OddsService:
    # Map API Team Names to nflverse Abbreviations
    TEAM_MAP = {
        "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
    }

    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_live_odds():
        try:
            url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?regions=us&markets=h2h&bookmakers=draftkings&apiKey={ODDS_API_KEY}"
            response = requests.get(url, timeout=5)
            if not response.ok: return {}
            data = response.json()
            
            live_data = {}
            
            for game in data:
                h_name = game.get('home_team')
                a_name = game.get('away_team')
                h_abbr = OddsService.TEAM_MAP.get(h_name)
                a_abbr = OddsService.TEAM_MAP.get(a_name)
                
                if h_abbr and a_abbr and game['bookmakers']:
                    dk = game['bookmakers'][0]
                    outcomes = dk['markets'][0]['outcomes']
                    h_price, a_price = None, None
                    
                    for outcome in outcomes:
                        if outcome['name'] == h_name: h_price = outcome['price']
                        elif outcome['name'] == a_name: a_price = outcome['price']
                    
                    def dec_to_amer(dec):
                        if dec >= 2.0: return int((dec - 1) * 100)
                        else: return int(-100 / (dec - 1))
                        
                    if h_price and a_price:
                        live_data[(h_abbr, a_abbr)] = {'home_am': dec_to_amer(h_price), 'away_am': dec_to_amer(a_price)}
            return live_data
        except: return {}

class StadiumService:
    # Map Team to (Lat, Lon, Type). Type: 'open', 'dome', 'retractable'
    STADIUMS = {
        "ARI": {"lat": 33.5276, "lon": -112.2626, "type": "retractable"},
        "ATL": {"lat": 33.7554, "lon": -84.4010, "type": "retractable"},
        "BAL": {"lat": 39.2780, "lon": -76.6227, "type": "open"},
        "BUF": {"lat": 42.7738, "lon": -78.7870, "type": "open"},
        "CAR": {"lat": 35.2258, "lon": -80.8528, "type": "open"},
        "CHI": {"lat": 41.8623, "lon": -87.6167, "type": "open"},
        "CIN": {"lat": 39.0955, "lon": -84.5161, "type": "open"},
        "CLE": {"lat": 41.5061, "lon": -81.6995, "type": "open"},
        "DAL": {"lat": 32.7473, "lon": -97.0945, "type": "retractable"},
        "DEN": {"lat": 39.7439, "lon": -105.0201, "type": "open"},
        "DET": {"lat": 42.3400, "lon": -83.0456, "type": "dome"},
        "GB":  {"lat": 44.5013, "lon": -88.0622, "type": "open"},
        "HOU": {"lat": 29.6847, "lon": -95.4107, "type": "retractable"},
        "IND": {"lat": 39.7601, "lon": -86.1639, "type": "retractable"},
        "JAX": {"lat": 30.3240, "lon": -81.6373, "type": "open"},
        "KC":  {"lat": 39.0489, "lon": -94.4839, "type": "open"},
        "LV":  {"lat": 36.0909, "lon": -115.1833, "type": "dome"},
        "LAC": {"lat": 33.9535, "lon": -118.3390, "type": "dome"},
        "LA":  {"lat": 33.9535, "lon": -118.3390, "type": "dome"},
        "MIA": {"lat": 25.9580, "lon": -80.2389, "type": "open"},
        "MIN": {"lat": 44.9735, "lon": -93.2575, "type": "dome"},
        "NE":  {"lat": 42.0909, "lon": -71.2643, "type": "open"},
        "NO":  {"lat": 29.9511, "lon": -90.0812, "type": "dome"},
        "NYG": {"lat": 40.8135, "lon": -74.0745, "type": "open"},
        "NYJ": {"lat": 40.8135, "lon": -74.0745, "type": "open"},
        "PHI": {"lat": 39.9008, "lon": -75.1675, "type": "open"},
        "PIT": {"lat": 40.4468, "lon": -80.0158, "type": "open"},
        "SF":  {"lat": 37.4032, "lon": -121.9698, "type": "open"},
        "SEA": {"lat": 47.5952, "lon": -122.3316, "type": "open"},
        "TB":  {"lat": 27.9759, "lon": -82.5033, "type": "open"},
        "TEN": {"lat": 36.1665, "lon": -86.7713, "type": "open"},
        "WAS": {"lat": 38.9077, "lon": -76.8645, "type": "open"}
    }

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_forecast(team_abbr, date_obj):
        """Fetches weather from Open-Meteo API"""
        stadium = StadiumService.STADIUMS.get(team_abbr)
        if not stadium: return None

        # If Dome/Retractable, assume closed/good conditions
        if stadium['type'] in ['dome', 'retractable']:
            return {"desc": "🏟️ Stadium Closed/Dome", "is_closed": True, "temp": 72, "wind": 0, "rain": 0}

        # If Open, fetch API
        try:
            date_str = date_obj.strftime("%Y-%m-%d")
            url = f"https://api.open-meteo.com/v1/forecast?latitude={stadium['lat']}&longitude={stadium['lon']}&daily=temperature_2m_max,precipitation_probability_max,windspeed_10m_max&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=auto&start_date={date_str}&end_date={date_str}"
            
            res = requests.get(url, timeout=5)
            if res.ok:
                data = res.json()
                if 'daily' in data:
                    temp = data['daily']['temperature_2m_max'][0]
                    rain = data['daily']['precipitation_probability_max'][0]
                    wind = data['daily']['windspeed_10m_max'][0]
                    
                    desc = f"🌡️ {temp}°F  💨 {wind} mph  💧 {rain}% Rain"
                    return {"desc": desc, "is_closed": False, "temp": temp, "wind": wind, "rain": rain}
        except:
            pass
            
        return {"desc": "Weather Unavailable", "is_closed": False, "temp": 70, "wind": 5, "rain": 0}

# ==========================================
# 4. MATH HELPERS
# ==========================================
def american_to_decimal(odds):
    try: odds = float(odds)
    except: return 2.0
    if odds > 0: return 1 + (odds / 100) 
    else: return 1 + (100 / abs(odds))

def no_vig_two_way(d1, d2):
    p1, p2 = 1/d1, 1/d2
    return p1/(p1+p2), p2/(p1+p2)

def synthetic_hold(d1, d2):
    return (1/d1 + 1/d2) - 1

def half_kelly(p, d, cap=0.05):
    # Correct logic: Calculate FULL, halve it, then clip to cap
    b = d - 1
    q = 1 - p
    if b <= 0: return 0
    f_full = (b * p - q) / b
    f_half = f_full * 0.5
    return max(0, min(f_half, cap))

def logit(p):
    return math.log(max(p, 1e-6) / (1 - max(p, 1e-6)))

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def get_last_nfl_date_from_sched(sched_df):
    """Return last gameday with a completed game, fallback to today."""
    if sched_df is None or sched_df.empty or 'gameday' not in sched_df.columns:
        return datetime.date.today()
    if 'home_score' in sched_df.columns:
        completed = sched_df.dropna(subset=['home_score'])
        if not completed.empty:
            return pd.to_datetime(completed['gameday']).max().date()
    return datetime.date.today()

def compute_team_home_advantage(games, min_games=10, alpha=0.05, smooth=0.5):
    """
    Calculates specific Home Field Advantage per team using Log-Odds ratio.
    Returns a dictionary mapping Team -> HFA Probability Bonus (e.g. 0.03)
    """
    df = games.copy()
    # Ensure numeric
    df = df.dropna(subset=['home_score', 'away_score'])
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]]))
    hfa_map = {}

    for team in teams:
        g_home = df[df["home_team"] == team]
        g_away = df[df["away_team"] == team]

        n_home = len(g_home)
        n_away = len(g_away)

        if n_home < min_games or n_away < min_games:
            hfa_map[team] = 0.0
            continue

        w_home = g_home["home_win"].sum()
        w_away = (1 - g_away["home_win"]).sum()

        p_home = (w_home + smooth) / (n_home + 2 * smooth)
        p_away = (w_away + smooth) / (n_away + 2 * smooth)

        l_home = np.log(p_home / (1.0 - p_home))
        l_away = np.log(p_away / (1.0 - p_away))
        delta = l_home - l_away

        # Relaxed significance: use raw delta if sample is decent
        home_prob_equal = 1.0 / (1.0 + np.exp(-delta / 2.0))
        raw_hfa = home_prob_equal - 0.5
        # Shrinkage factor: don't trust small samples fully
        n_total = n_home + n_away
        shrink = min(1.0, n_total / (2.0 * min_games))
        hfa_map[team] = raw_hfa * shrink
            
    return hfa_map

# ==========================================
# 5. DATA LOADER (ROBUST MIX)
# ==========================================
def try_load_nflverse_csv(url, label, status_report, yr):
    try:
        df = pd.read_csv(url, compression='gzip', low_memory=False)
        if df.empty:
            status_report[yr] = f"❌ {label}: Empty"
            return pd.DataFrame()
        status_report[yr] = f"✅ {label}: Loaded (Direct CSV)"
        return df
    except Exception as e:
        # status_report[yr] = f"❌ {label}: Unavailable"
        return pd.DataFrame()

@st.cache_resource(ttl=3600)
def load_nfl_data():
    current_year = datetime.date.today().year
    if datetime.date.today().month < 3: current_year -= 1
    
    target_season = current_year

    # 1. SCHEDULE (Required)
    sched = pd.DataFrame()
    try:
        sched = nfl.import_schedules([target_season])
        if 'gameday' in sched.columns:
             sched['gameday'] = pd.to_datetime(sched['gameday']).dt.date
    except Exception as e:
        st.error(f"Unable to load schedule for {target_season}: {e}")
        st.stop()

    if sched is None or sched.empty:
        # Last ditch: fallback to URL
        try:
            url = f"https://raw.githubusercontent.com/nflverse/nflverse-data/master/data/schedules/schedule_{target_season}.csv"
            sched = pd.read_csv(url)
            sched['gameday'] = pd.to_datetime(sched['gameday']).dt.date
        except:
            st.error(f"No schedule found for {target_season}.")
            st.stop()
        
    last_played_date = get_last_nfl_date_from_sched(sched)

    # 2. HFA DATA
    hfa_dict = {}
    try:
        hfa_years = [target_season - 3, target_season - 2, target_season - 1]
        sched_hfa = nfl.import_schedules(hfa_years)
        hfa_dict = compute_team_home_advantage(sched_hfa)
    except: pass

    # 3. STATS - ROBUST LOOP
    pbp_all = []
    weekly_all = []
    loaded_years = []
    status_report = {}
    
    for yr in [target_season - 1, target_season]:
        success = False
        
        # A. Library
        try:
            p = nfl.import_pbp_data([yr], cache=False)
            w = nfl.import_weekly_data([yr])
            if not p.empty:
                pbp_all.append(p)
                weekly_all.append(w)
                loaded_years.append(yr)
                status_report[yr] = "✅ Loaded (Library)"
                success = True
        except: pass
        
        # B. Releases URL (Direct)
        if not success:
            try:
                url_p = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{yr}.csv.gz"
                url_w = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player_week/stats_player_week_{yr}.csv.gz"
                
                p_direct = try_load_nflverse_csv(url_p, "PBP", status_report, yr)
                w_direct = try_load_nflverse_csv(url_w, "Weekly", status_report, yr)
                
                if not p_direct.empty:
                    pbp_all.append(p_direct)
                    weekly_all.append(w_direct)
                    loaded_years.append(yr)
                    success = True
            except: pass
        
        if not success:
            status_report[yr] = "❌ Unavailable"

    pbp = pd.concat(pbp_all, ignore_index=True) if pbp_all else pd.DataFrame()
    weekly = pd.concat(weekly_all, ignore_index=True) if weekly_all else pd.DataFrame()

    clf = None
    team_stats = pd.DataFrame()
    qb_stats = pd.DataFrame()
    loaded_year = target_season
    
    if not pbp.empty:
        # Feature Engineering
        pass_plays = pbp[pbp["play_type"] == "pass"]
        run_plays = pbp[pbp["play_type"] == "run"]
        
        off_pass = pass_plays.groupby(["season", "posteam"], dropna=True).agg(epa_pass_off=("epa", "mean"), pass_yds_off=("yards_gained", "sum")).reset_index().rename(columns={"posteam": "team"})
        off_run = run_plays.groupby(["season", "posteam"], dropna=True).agg(epa_rush_off=("epa", "mean"), rush_yds_off=("yards_gained", "sum")).reset_index().rename(columns={"posteam": "team"})
        game_counts = pbp.groupby(['season', 'posteam'])['game_id'].nunique().reset_index().rename(columns={'game_id': 'games', 'posteam': 'team'})
        tos = pbp.groupby(["season", "posteam"], dropna=True).agg(interceptions=("interception", "sum"), fumbles_lost=("fumble_lost", "sum")).reset_index().rename(columns={"posteam": "team"})

        pbp["defteam"] = pbp["defteam"].fillna(pbp["posteam"])
        def_pass = pass_plays.groupby(["season", "defteam"], dropna=True).agg(epa_pass_def=("epa", "mean"), pass_yds_def=("yards_gained", "sum")).reset_index().rename(columns={"defteam": "team"})
        def_run = run_plays.groupby(["season", "defteam"], dropna=True).agg(epa_rush_def=("epa", "mean"), rush_yds_def=("yards_gained", "sum")).reset_index().rename(columns={"defteam": "team"})
        takeaways = pbp.groupby(["season", "defteam"], dropna=True).agg(def_int=("interception", "sum"), def_fumbles=("fumble_lost", "sum")).reset_index().rename(columns={"defteam": "team"})

        team_stats = pd.merge(off_pass, off_run, on=["season", "team"], how="outer")
        team_stats = pd.merge(team_stats, game_counts, on=["season", "team"], how="outer")
        team_stats = pd.merge(team_stats, tos, on=["season", "team"], how="outer")
        team_stats = pd.merge(team_stats, def_pass, on=["season", "team"], how="outer")
        team_stats = pd.merge(team_stats, def_run, on=["season", "team"], how="outer")
        team_stats = pd.merge(team_stats, takeaways, on=["season", "team"], how="outer")
        
        team_stats['avg_pass_off'] = team_stats['pass_yds_off'] / team_stats['games']
        team_stats['avg_rush_off'] = team_stats['rush_yds_off'] / team_stats['games']
        team_stats['avg_tos_off'] = (team_stats['interceptions'] + team_stats['fumbles_lost']) / team_stats['games']
        team_stats['avg_pass_def'] = team_stats['pass_yds_def'] / team_stats['games']
        team_stats['avg_rush_def'] = team_stats['rush_yds_def'] / team_stats['games']
        team_stats['avg_tos_def'] = (team_stats['def_int'] + team_stats['def_fumbles']) / team_stats['games']
        
        team_stats['epa_net_pass'] = team_stats['epa_pass_off'] - team_stats['epa_pass_def']
        team_stats['epa_net_rush'] = team_stats['epa_rush_off'] - team_stats['epa_rush_def']
        
        if 'cpoe' not in pbp.columns: pbp['cpoe'] = 0.0
        qb_stats = pbp[pbp['play_type']=='pass'].groupby(['season', 'posteam', 'passer_player_name']).agg(
            qb_epa=('epa', 'mean'),
            qb_cpoe=('cpoe', 'mean'),
            dropbacks=('play_id', 'count')
        ).reset_index().sort_values('dropbacks', ascending=False).drop_duplicates(['season', 'posteam'])

        # Training
        games_train = sched.dropna(subset=['home_score', 'home_moneyline']).copy()
        if not games_train.empty:
            games_train['gameday_dt'] = pd.to_datetime(games_train['gameday']).dt.date
            games_train = games_train[games_train['gameday_dt'] <= last_played_date]

        if not games_train.empty:
            games_train['home_win'] = (games_train['home_score'] > games_train['away_score']).astype(int)
            games_train = games_train.merge(team_stats.add_prefix("home_"), left_on=["season", "home_team"], right_on=["home_season", "home_team"], how="left")
            games_train = games_train.merge(team_stats.add_prefix("away_"), left_on=["season", "away_team"], right_on=["away_season", "away_team"], how="left")
            
            games_train["diff_net_pass"] = games_train["home_epa_net_pass"] - games_train["away_epa_net_pass"]
            games_train["diff_net_rush"] = games_train["home_epa_net_rush"] - games_train["away_epa_net_rush"]
            
            def get_logit(r):
                try: return logit(no_vig_two_way(american_to_decimal(r['home_moneyline']), american_to_decimal(r['away_moneyline']))[0])
                except: return np.nan
            games_train['logit_mkt'] = games_train.apply(get_logit, axis=1)
            
            features = ['logit_mkt', 'diff_net_pass', 'diff_net_rush']
            train_clean = games_train.dropna(subset=['home_win'] + features)
            if not train_clean.empty:
                clf = LogisticRegression()
                # Weighted Training: 3x weight for current season
                train_clean['weight'] = train_clean['season'].map(lambda x: 3.0 if x == current_year else 1.0)
                clf.fit(train_clean[features], train_clean['home_win'], sample_weight=train_clean['weight'])
        
        loaded_year = pbp['season'].max()

    return clf, team_stats, weekly, sched, qb_stats, loaded_year, hfa_dict, status_report, loaded_years

# LOAD SEQUENCE
loading_placeholder.empty()
with loading_placeholder:
    components.html(LOADING_HTML, height=450)

model_clf, team_stats_db, weekly_stats_db, sched_db, qb_stats_db, sched_source, hfa_db, status_report, loaded_years_list = load_nfl_data()
live_odds_map = OddsService.fetch_live_odds()

loading_placeholder.empty()

st.sidebar.markdown("### 💾 Data Diagnostics")
for item, status in status_report.items():
    st.sidebar.caption(f"**{item}:** {status}")

if loaded_years_list:
    years_str = ", ".join(str(y) for y in sorted(list(set(loaded_years_list))))
    years_slot.caption(f"Stats Loaded: {years_str}")

if hfa_db: status_slot.caption("HFA Data: 3 Years Loaded")

if sched_db is None or sched_db.empty:
    st.error("Unable to load NFL Schedule. Check connection.")
    st.stop()

# ==========================================
# 6. LOGIC ENGINE
# ==========================================
class CockpitEngine:
    @staticmethod
    def get_team_leaders(team_abbr):
        if weekly_stats_db.empty: return {}
        
        # Safety: Ensure required columns exist in weekly stats
        req_cols = {'recent_team', 'passing_yards', 'rushing_yards', 'receiving_yards', 'player_display_name', 'week'}
        if not req_cols.issubset(set(weekly_stats_db.columns)):
             return {}

        # 1. Filter for this team
        team_rows = weekly_stats_db[weekly_stats_db['recent_team'] == team_abbr]
        if team_rows.empty: return {}

        # 2. Find the MAX season and MAX week (The "Now")
        max_season = team_rows['season'].max()
        current_season_rows = team_rows[team_rows['season'] == max_season]
        max_week = current_season_rows['week'].max()
        
        # 3. Get only players from that specific last game
        last_game_stats = current_season_rows[current_season_rows['week'] == max_week]
        
        if last_game_stats.empty: return {}

        leaders = {}
        
        # QB: Sort by passing yards
        qb = last_game_stats.sort_values('passing_yards', ascending=False).head(1)
        if not qb.empty:
            leaders['QB'] = {'name': qb.iloc[0]['player_display_name'], 'raw_yds': qb.iloc[0]['passing_yards'], 'stat': f"Last: {qb.iloc[0]['passing_yards']} yds"}
            
        # RB: Sort by rushing yards
        rb = last_game_stats.sort_values('rushing_yards', ascending=False).head(1)
        if not rb.empty:
            leaders['RB'] = {'name': rb.iloc[0]['player_display_name'], 'raw_yds': rb.iloc[0]['rushing_yards'], 'stat': f"Last: {rb.iloc[0]['rushing_yards']} yds"}
            
        # WR: Sort by receiving yards
        wr = last_game_stats.sort_values('receiving_yards', ascending=False).head(1)
        if not wr.empty:
            leaders['WR'] = {'name': wr.iloc[0]['player_display_name'], 'raw_yds': wr.iloc[0]['receiving_yards'], 'stat': f"Last: {wr.iloc[0]['receiving_yards']} yds"}
            
        return leaders

    @staticmethod
    def get_qb_metrics(team_abbr, season):
        if qb_stats_db.empty: return None
        starter = qb_stats_db[(qb_stats_db['posteam'] == team_abbr)].sort_values('season', ascending=False)
        if starter.empty: return None
        return starter.iloc[0]

    @staticmethod
    def generate_props(home, away, h_win_prob, a_win_prob):
        h_lead = CockpitEngine.get_team_leaders(home)
        a_lead = CockpitEngine.get_team_leaders(away)
        props = []
        
        props.append(parlay.PropLeg(f"{home}_ML", f"{home} To Win", 1.0 / max(h_win_prob, 0.01), h_win_prob, "Team Win", home, "Moneyline"))
        props.append(parlay.PropLeg(f"{away}_ML", f"{away} To Win", 1.0 / max(a_win_prob, 0.01), a_win_prob, "Team Win", away, "Moneyline"))

        def add(p_data, cat, team, m=1.0):
            if not p_data: return
            line = round(p_data['raw_yds'] * m / 5) * 5 + 0.5
            desc = f"{p_data['name']} Over {line} {cat} Yds"
            props.append(parlay.PropLeg(desc, desc, 1.91, 0.50, cat, team, p_data['stat']))

        if 'QB' in h_lead: add(h_lead['QB'], "Passing", home, 0.95)
        if 'RB' in h_lead: add(h_lead['RB'], "Rushing", home, 0.85)
        if 'WR' in h_lead: add(h_lead['WR'], "Receiving", home, 0.90)
        if 'QB' in a_lead: add(a_lead['QB'], "Passing", away, 0.95)
        if 'RB' in a_lead: add(a_lead['RB'], "Rushing", away, 0.85)
        if 'WR' in a_lead: add(a_lead['WR'], "Receiving", away, 0.90)
        return props

    @staticmethod
    def get_default_sliders(row, weather_data):
        s_def = {'h_qb': 5, 'h_pwr': 5, 'h_def': 5, 'a_qb': 5, 'a_pwr': 5, 'a_def': 5, 'rest': 5, 'news': 5, 'weath': 0, 'hfa': 5}
        if not team_stats_db.empty:
            h_team, a_team = row['home_team'], row['away_team']
            h = team_stats_db[(team_stats_db['team']==h_team)].sort_values('season', ascending=False).head(1)
            a = team_stats_db[(team_stats_db['team']==a_team)].sort_values('season', ascending=False).head(1)
            def epa_to_10(epa, reverse=False):
                val = 5 + (epa / 0.04)
                if reverse: val = 5 - (epa / 0.04) 
                return int(min(max(val, 0), 10))
            if not h.empty:
                s_def['h_qb'] = epa_to_10(h['epa_pass_off'].values[0])
                s_def['h_pwr'] = epa_to_10(h['epa_rush_off'].values[0])
                avg_def_epa = (h['epa_pass_def'].values[0] + h['epa_rush_def'].values[0]) / 2
                s_def['h_def'] = epa_to_10(avg_def_epa, reverse=True)
            if not a.empty:
                s_def['a_qb'] = epa_to_10(a['epa_pass_off'].values[0])
                s_def['a_pwr'] = epa_to_10(a['epa_rush_off'].values[0])
                avg_def_epa = (a['epa_pass_def'].values[0] + a['epa_rush_def'].values[0]) / 2
                s_def['a_def'] = epa_to_10(avg_def_epa, reverse=True)
        
        hfa_val = hfa_db.get(row['home_team'], 0.0)
        hfa_ticks = hfa_val / 0.015
        s_def['hfa'] = int(min(max(5 + hfa_ticks, 0), 10))

        rest_h = row.get('home_rest', 7)
        rest_a = row.get('away_rest', 7)
        if abs(rest_h - rest_a) > 3: 
             s_def['rest'] = 8 if rest_h > rest_a else 2
        
        if weather_data and not weather_data['is_closed']:
            if weather_data['wind'] > 15 or weather_data['rain'] > 50: s_def['weath'] = 6
        return s_def
        
    @staticmethod
    def get_simple_gold_prediction(home_team, away_team, season):
        if team_stats_db.empty: return None
        h = team_stats_db[(team_stats_db['team'] == home_team)].sort_values('season', ascending=False).head(1)
        a = team_stats_db[(team_stats_db['team'] == away_team)].sort_values('season', ascending=False).head(1)
        if h.empty or a.empty: return None
        def expected(val_off, val_def): return (val_off + val_def) / 2
        h_pass = expected(h['avg_pass_off'].values[0], a['avg_pass_def'].values[0])
        h_rush = expected(h['avg_rush_off'].values[0], a['avg_rush_def'].values[0])
        h_to   = expected(h['avg_tos_off'].values[0], a['avg_tos_def'].values[0])
        a_pass = expected(a['avg_pass_off'].values[0], h['avg_pass_def'].values[0])
        a_rush = expected(a['avg_rush_off'].values[0], h['avg_rush_def'].values[0])
        a_to   = expected(a['avg_tos_off'].values[0], h['avg_tos_def'].values[0])
        h_score = 2.0 + (h_pass * 0.04) + (h_rush * 0.04) - (h_to * 4.0) + 2.0
        a_score = 2.0 + (a_pass * 0.04) + (a_rush * 0.04) - (a_to * 4.0)
        return {'h_score': h_score, 'a_score': a_score, 'h_pass': h_pass, 'h_rush': h_rush, 'h_to': h_to, 'a_pass': a_pass, 'a_rush': a_rush, 'a_to': a_to}

    @staticmethod
    def calc_win_prob(market_prob, row, sliders):
        logit_mkt = logit(market_prob)
        def s_to_epa(val): return (val - 5) * 0.03 
        h_pass = s_to_epa(sliders['ph_qb'])
        h_rush = s_to_epa(sliders['ph_pwr'])
        h_def_val = s_to_epa(sliders['ph_def'])
        a_pass = s_to_epa(sliders['pa_qb'])
        a_rush = s_to_epa(sliders['pa_pwr'])
        a_def_val = s_to_epa(sliders['pa_def'])

        adj_net_pass = (h_pass - a_pass) + (h_def_val - a_def_val)
        adj_net_rush = (h_rush - a_rush) + (h_def_val - a_def_val)
        
        base_diff_pass, base_diff_rush = 0.0, 0.0
        if not team_stats_db.empty:
            h = team_stats_db[(team_stats_db['team']==row['home_team'])].sort_values('season', ascending=False).head(1)
            a = team_stats_db[(team_stats_db['team']==row['away_team'])].sort_values('season', ascending=False).head(1)
            if not h.empty and not a.empty:
                base_diff_pass = h['epa_net_pass'].values[0] - a['epa_net_pass'].values[0]
                base_diff_rush = h['epa_net_rush'].values[0] - a['epa_net_rush'].values[0]

        model_p = market_prob
        if model_clf:
            x = pd.DataFrame([[logit_mkt, base_diff_pass + adj_net_pass, base_diff_rush + adj_net_rush]], 
                           columns=['logit_mkt', 'diff_net_pass', 'diff_net_rush'])
            model_raw = model_clf.predict_proba(x)[0, 1]
            model_p = (0.7 * market_prob) + (0.3 * model_raw)

        news = (5 - sliders['wn']) * 0.015
        weather = -0.02 * (sliders['ww'] / 5) if sliders['ww'] > 0 else 0
        rest = ((sliders.get('rh',7)-sliders.get('ra',7))*0.005) * (sliders['wr']/2.0)
        hfa_boost = (sliders['hfa'] - 5) * 0.015

        final = min(max(model_p + news + weather + rest + hfa_boost, 0.01), 0.99)
        return final, {"AI Model (Stats)": model_p, "News Variance": news, "Weather Penalty": weather, "Rest Advantage": rest, "Home Field": hfa_boost}

# ==========================================
# 8. UI RENDERERS
# ==========================================
def render_game_card(i, row, bankroll, kelly):
    home, away = row['home_team'], row['away_team']
    season = row['season']
    
    def safe_int(val):
        try: return int(val)
        except: return -110
    
    def_oa = safe_int(row.get('away_moneyline'))
    def_oh = safe_int(row.get('home_moneyline'))
    src = "Schedule (Cached)"

    dh_default, da_default = american_to_decimal(def_oh), american_to_decimal(def_oa)
    hold_default = synthetic_hold(dh_default, da_default)
    
    live_odds = live_odds_map.get((home, away))
    if not live_odds: live_odds = live_odds_map.get((away, home))
    if live_odds:
        def_oh, def_oa = live_odds['home_am'], live_odds['away_am']
        src = "DraftKings (Live)"
    
    h_qb = CockpitEngine.get_qb_metrics(home, season)
    a_qb = CockpitEngine.get_qb_metrics(away, season)
    weather_info = StadiumService.get_forecast(home, sel_date)
    defs = CockpitEngine.get_default_sliders(row, weather_info)

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        c1.subheader(f"{away} @ {home}")
        c1.caption(f"{row['gametime']} ET")
        if weather_info: c1.caption(f"{weather_info['desc']}")
        
        if hold_default > 0.05:
            c1.caption(f"⚠️ Rich market (feed hold {hold_default:.1%})")

        with c2:
            st.markdown("**QB Matchup**")
            if h_qb is not None and a_qb is not None:
                edge_txt = "Even"
                if h_qb['qb_epa'] > a_qb['qb_epa'] + 0.05: edge_txt = f"✅ {h_qb['passer_player_name']}"
                elif a_qb['qb_epa'] > h_qb['qb_epa'] + 0.05: edge_txt = f"✅ {a_qb['passer_player_name']}"
                st.caption(f"Edge: {edge_txt}")
            else: st.caption("Data Unavailable")

        with st.expander("See QB Efficiency Stats (EPA & CPOE)"):
            c_qa, c_qh = st.columns(2)
            if a_qb is not None:
                c_qa.markdown(f"**{a_qb['passer_player_name']}** ({away})")
                c_qa.metric("EPA/Play", f"{a_qb['qb_epa']:.2f}")
                c_qa.metric("CPOE", f"{a_qb['qb_cpoe']:.1f}%")
            if h_qb is not None:
                c_qh.markdown(f"**{h_qb['passer_player_name']}** ({home})")
                c_qh.metric("EPA/Play", f"{h_qb['qb_epa']:.2f}")
                c_qh.metric("CPOE", f"{h_qb['qb_cpoe']:.1f}%")

        st.divider()
        c_odds, c_away, c_home, c_res = st.columns([1, 1.5, 1.5, 1.2])
        with c_odds:
            st.markdown(f"##### 🏦 {src}")
            oa = st.number_input(f"{away}", value=def_oa, step=5, key=f"oa_{i}")
            oh = st.number_input(f"{home}", value=def_oh, step=5, key=f"oh_{i}")
            da_live, dh_live = american_to_decimal(oa), american_to_decimal(oh)
            pmkt = no_vig_two_way(dh_live, da_live)[0]
            
            hold_live = synthetic_hold(dh_live, da_live)
            
            # Dynamic Market Prob Display
            # pmkt[0] is Home. pmkt[1] is Away.
            # If Home >= 50%, show Home. Else show Away.
            if pmkt >= 0.5:
                disp_mkt_prob = pmkt
                disp_mkt_team = home
            else:
                disp_mkt_prob = 1.0 - pmkt
                disp_mkt_team = away
                
            st.progress(disp_mkt_prob, f"Mkt: {disp_mkt_team} {disp_mkt_prob:.1%}")
            
            if hold_live > 0.07:
                st.warning(f"Hold {hold_live:.1%} > 7%. Skipping.")
                return
            elif hold_live > 0.05:
                st.caption(f"Hold: {hold_live:.1%} (High)")

        with c_away:
            st.markdown(f"##### {away} Adjust")
            pa_qb = st.slider(f"QB Rating", 0, 10, defs['a_qb'], key=f"pa_qb_{i}", help="0=Bench, 5=Avg, 10=MVP")
            pa_pwr = st.slider(f"Run/Power", 0, 10, defs['a_pwr'], key=f"pa_pwr_{i}", help="Run Game/O-Line Strength")
            pa_def = st.slider(f"Defense", 0, 10, defs['a_def'], key=f"pa_def_{i}", help="Ability to stop scores")

        with c_home:
            st.markdown(f"##### {home} Adjust")
            ph_qb = st.slider(f"QB Rating", 0, 10, defs['h_qb'], key=f"ph_qb_{i}", help="0=Bench, 5=Avg, 10=MVP")
            ph_pwr = st.slider(f"Run/Power", 0, 10, defs['h_pwr'], key=f"ph_pwr_{i}", help="Run Game/O-Line Strength")
            ph_def = st.slider(f"Defense", 0, 10, defs['h_def'], key=f"ph_def_{i}", help="Ability to stop scores")
            
        st.markdown("##### Context")
        cc1, cc2, cc3, cc4 = st.columns(4)
        wr = st.slider("Rest", 0, 10, defs['rest'], key=f"wr_{i}", help="5=Big Advantage (Bye vs Short Week)")
        wn = st.slider("News (Home vs Away)", 0, 10, defs['news'], key=f"wn_{i}", help="0=Good for Home, 5=Neutral, 10=Good for Away")
        hfa = st.slider("Home Field", 0, 10, defs['hfa'], help="Auto-calculated advantage from history", key=f"hfa_{i}")
        weather_disable = weather_info and weather_info['is_closed']
        ww = st.slider("Weather", 0, 10, defs['weath'], disabled=weather_disable, key=f"ww_{i}")
        if weather_disable: cc4.caption("🔒 Indoor Stadium")

        with c_res:
            sliders = {'pa_qb': pa_qb, 'pa_pwr': pa_pwr, 'pa_def': pa_def, 'ph_qb': ph_qb, 'ph_pwr': ph_pwr, 'ph_def': ph_def, 'wr': wr, 'wn': wn, 'ww': ww, 'hfa': hfa, 'rh': row.get('home_rest',7), 'ra': row.get('away_rest',7)}
            final_p_home, breakdown = CockpitEngine.calc_win_prob(pmkt, row, sliders)
            
            # Determine Verdict Side
            if final_p_home >= 0.5:
                pick_team = home
                pick_prob = final_p_home
                pick_odds = dh_live
                mkt_ref = pmkt # Home market prob
            else:
                pick_team = away
                pick_prob = 1.0 - final_p_home
                pick_odds = da_live
                mkt_ref = 1.0 - pmkt # Away market prob
            
            p_be = 1 / pick_odds
            edge = pick_prob - p_be
            se_p = 0.04 
            z_score = edge / se_p
            ev = pick_prob * pick_odds - 1
            
            st.markdown("##### 🚀 Verdict")
            st.markdown(f"**Model Pick:** {pick_team}")
            st.metric("Win Prob", f"{pick_prob:.1%}", delta=f"{pick_prob-mkt_ref:.1%}")
            
            with st.expander("🧮 View Math"):
                st.markdown("**Component Contribution:**")
                for k, v in breakdown.items():
                    val_fmt = f"{v:.1%}" if k == "AI Model (Stats)" else f"{v:+.1%}"
                    st.write(f"- {k}: {val_fmt}")
            
            # Bet Logic with CI Filter
            max_wager_val = bankroll * 0.05
            
            if z_score >= 1.28 and ev > 0:
                stake = half_kelly(pick_prob, pick_odds) * bankroll * kelly
                stake = min(stake, max_wager_val)
                st.success(f"BET {pick_team} ${stake:.0f}")
            else:
                if ev > 0: st.warning("Edge too small (CI filter)")
                else: st.info("No Edge")

        with st.expander("🏆 Simple Gold Standard (Outcome Checker)"):
            gold = CockpitEngine.get_simple_gold_prediction(home, away, season)
            if gold:
                c_g1, c_g2 = st.columns(2)
                with c_g1:
                    st.markdown(f"**{away} Projected**")
                    st.write(f"Pass: {gold['a_pass']:.0f} yds")
                    st.write(f"Rush: {gold['a_rush']:.0f} yds")
                    st.write(f"Turnovers: {gold['a_to']:.1f}")
                    st.metric("Score", f"{gold['a_score']:.1f}")
                with c_g2:
                    st.markdown(f"**{home} Projected**")
                    st.write(f"Pass: {gold['h_pass']:.0f} yds")
                    st.write(f"Rush: {gold['h_rush']:.0f} yds")
                    st.write(f"Turnovers: {gold['h_to']:.1f}")
                    st.metric("Score", f"{gold['h_score']:.1f}")
                diff = gold['h_score'] - gold['a_score']
                winner = home if diff > 0 else away
                st.success(f"**Prediction: {winner} wins by {abs(diff):.1f} points**")
            else: st.caption("Insufficient Data for Gold Standard Model")

        if st.button(f"🧠 Open Strategy Lab: {away} vs {home}", key=f"sl_{i}"):
            st.session_state['sl_active'] = True
            st.session_state['sl_home'] = home
            st.session_state['sl_away'] = away
            st.session_state['sl_h_prob'] = final_p_home
            st.session_state['sl_legs'] = []
            st.rerun()

# ==========================================
# 9. STRATEGY LAB UI
# ==========================================
def render_strategy_lab(bankroll):
    st.markdown(f"## 🧠 Strategy Lab: {st.session_state['sl_away']} @ {st.session_state['sl_home']}")
    if st.button("← Back to Schedule"):
        st.session_state['sl_active'] = False
        st.rerun()
    home = st.session_state['sl_home']
    away = st.session_state['sl_away']
    h_prob = st.session_state['sl_h_prob']
    
    a_prob = 1.0 - h_prob
    
    if 'sl_pool' not in st.session_state or not st.session_state['sl_legs']:
        st.session_state['sl_pool'] = CockpitEngine.generate_props(home, away, h_prob, a_prob)
    all_props = st.session_state['sl_pool']
    current_legs = st.session_state['sl_legs']
    MAX_LEGS = 4
    st.divider()
    c_build, c_ticket = st.columns([2, 1])
    with c_build:
        if len(current_legs) == 0:
            st.subheader("Step 1: Pick the Winner")
            c1, c2 = st.columns(2)
            h_ml = next((p for p in all_props if p.description == f"{home} To Win"), None)
            a_ml = next((p for p in all_props if p.description == f"{away} To Win"), None)
            if h_ml and c1.button(f"🏆 {home}"):
                st.session_state['sl_legs'].append(h_ml)
                st.rerun()
            if a_ml and c2.button(f"🏆 {away}"):
                st.session_state['sl_legs'].append(a_ml)
                st.rerun()
        elif len(current_legs) < MAX_LEGS:
            last_leg = current_legs[-1]
            st.subheader(f"Next Step: What correlates with {last_leg.description}?")
            fits = parlay.ParlayMath.find_best_additions(current_legs, all_props, top_n=3)
            if fits:
                cols = st.columns(3)
                for idx, leg in enumerate(fits):
                    with cols[idx]:
                        with st.container(border=True):
                            st.markdown(f"**{leg.description}**")
                            st.caption(f"Type: {leg.category}")
                            st.markdown(f"*:gray[{leg.recent_stat}]*") 
                            if st.button("➕ Add", key=f"add_{leg.leg_id}_{len(current_legs)}"):
                                st.session_state['sl_legs'].append(leg)
                                st.rerun()
            else: st.info("No more high-correlation props found.")
        else: st.success(f"Ticket Full ({MAX_LEGS} Legs).")
            
        if len(current_legs) > 0:
            if st.button("🔄 Reset Ticket"):
                st.session_state['sl_legs'] = []
                st.rerun()
    with c_ticket:
        st.markdown("### 🎫 Ticket")
        if current_legs:
            res = parlay.ParlayMath.calculate_ticket(current_legs, bankroll)
            for i, leg in enumerate(current_legs): st.write(f"{i+1}. {leg.description}")
            st.divider()
            st.metric("Odds", f"{res.final_odds:.2f}")
            st.metric("Win Prob", f"{res.win_prob:.1%}")
            if res.ev > 0:
                st.success(f"**EV: +{res.ev:.1%}**")
                st.markdown(f"### Bet: ${res.kelly_stake:.0f}")
            else: st.warning(f"EV: {res.ev:.1%}")
            st.caption(f"Targeting {MAX_LEGS}-Leg SGP")
        else: st.info("Empty")

if st.session_state.get('sl_active', False):
    render_strategy_lab(bankroll)
else:
    if not sched_db.empty:
        sched_db['gameday'] = sched_db['gameday'].astype(str)
        day_games = sched_db[sched_db['gameday'] == str(sel_date)]
        if day_games.empty:
            st.warning(f"No games found for {sel_date}.")
        else:
            st.markdown(f"### 🔥 {len(day_games)} Games Found")
            for i, row in day_games.iterrows(): render_game_card(i, row, bankroll, kelly)
