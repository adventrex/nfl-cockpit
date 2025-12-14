import streamlit as st
import pandas as pd
import numpy as np
import datetime
import math
import random
import requests
import streamlit.components.v1 as components
from dataclasses import dataclass
from typing import List, Dict, Optional
from math import erf, sqrt
from sklearn.linear_model import LogisticRegression

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
# 3. INTERNAL PARLAY BRAIN
# ==========================================
@dataclass
class PropLeg:
    leg_id: str
    description: str
    decimal_odds: float
    p_model: float
    category: str
    team: str
    recent_stat: str

@dataclass
class ParlayResult:
    final_odds: float
    win_prob: float
    ev: float
    kelly_stake: float

class ParlayMath:
    @staticmethod
    def get_correlation(leg1: PropLeg, leg2: PropLeg):
        if leg1.leg_id == leg2.leg_id: return 1.0
        same_team = (leg1.team == leg2.team)
        corr_map = {
            frozenset(['Team Win', 'Passing']): (0.60, -0.20),
            frozenset(['Team Win', 'Rushing']): (0.55, -0.30),
            frozenset(['Team Win', 'Receiving']): (0.50, -0.20),
            frozenset(['Passing', 'Receiving']): (0.75, 0.15),
            frozenset(['Passing', 'Rushing']): (0.10, -0.05),
            frozenset(['Rushing', 'Rushing']): (-0.15, 0.00),
        }
        key = frozenset([leg1.category, leg2.category])
        if key in corr_map:
            vals = corr_map[key]
            return vals[0] if same_team else vals[1]
        return 0.0

    @staticmethod
    def calculate_ticket(legs: List[PropLeg], bankroll: float):
        if not legs: return ParlayResult(0, 0, 0, 0)
        naive_prob = 1.0
        for leg in legs: naive_prob *= leg.p_model
        total_corr, pairs = 0.0, 0
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                total_corr += ParlayMath.get_correlation(legs[i], legs[j])
                pairs += 1
        avg_corr = total_corr / pairs if pairs > 0 else 0
        boost_factor = 1 + (avg_corr * 0.8)
        true_prob = min(naive_prob * boost_factor, 0.95)
        vig_pct = 0.045 + (0.02 * (len(legs) - 1))
        fair_odds = 1 / true_prob
        book_odds = max(fair_odds * (1 - vig_pct), 1.01)
        ev = (true_prob * book_odds) - 1
        b = book_odds - 1
        q = 1 - true_prob
        f = (b * true_prob - q) / b if b > 0 else 0
        stake = max(0, f * 0.5 * bankroll)
        return ParlayResult(book_odds, true_prob, ev, stake)

    @staticmethod
    def find_best_additions(current_legs: List[PropLeg], candidates: List[PropLeg], top_n=3):
        scores = []
        for cand in candidates:
            if any(l.leg_id == cand.leg_id for l in current_legs): continue
            avg_corr = 0.0
            for leg in current_legs: avg_corr += ParlayMath.get_correlation(leg, cand)
            if current_legs: avg_corr /= len(current_legs)
            leg_ev = (cand.p_model * cand.decimal_odds) - 1
            score = (avg_corr * 2.0) + (leg_ev * 0.5)
            scores.append((score, cand))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scores[:top_n]]

# ==========================================
# 4. SECONDARY DATA SOURCES
# ==========================================
class OddsService:
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
                h_name, a_name = game.get('home_team'), game.get('away_team')
                h_abbr, a_abbr = OddsService.TEAM_MAP.get(h_name), OddsService.TEAM_MAP.get(a_name)
                if h_abbr and a_abbr and game['bookmakers']:
                    dk = game['bookmakers'][0]
                    outcomes = dk['markets'][0]['outcomes']
                    h_price, a_price = None, None
                    for outcome in outcomes:
                        if outcome['name'] == h_name: h_price = outcome['price']
                        elif outcome['name'] == a_name: a_price = outcome['price']
                    def dec_to_amer(dec):
                        if dec <= 1.0: return -10000
                        return int((dec - 1) * 100) if dec >= 2.0 else int(-100 / (dec - 1))
                    if h_price and a_price:
                        live_data[(h_abbr, a_abbr)] = {'home_am': dec_to_amer(h_price), 'away_am': dec_to_amer(a_price)}
            return live_data
        except: return {}

class StadiumService:
    STADIUMS = {
        "ARI": {"lat": 33.5276, "lon": -112.2626, "type": "retractable"}, "ATL": {"lat": 33.7554, "lon": -84.4010, "type": "retractable"},
        "BAL": {"lat": 39.2780, "lon": -76.6227, "type": "open"}, "BUF": {"lat": 42.7738, "lon": -78.7870, "type": "open"},
        "CAR": {"lat": 35.2258, "lon": -80.8528, "type": "open"}, "CHI": {"lat": 41.8623, "lon": -87.6167, "type": "open"},
        "CIN": {"lat": 39.0955, "lon": -84.5161, "type": "open"}, "CLE": {"lat": 41.5061, "lon": -81.6995, "type": "open"},
        "DAL": {"lat": 32.7473, "lon": -97.0945, "type": "retractable"}, "DEN": {"lat": 39.7439, "lon": -105.0201, "type": "open"},
        "DET": {"lat": 42.3400, "lon": -83.0456, "type": "dome"}, "GB":  {"lat": 44.5013, "lon": -88.0622, "type": "open"},
        "HOU": {"lat": 29.6847, "lon": -95.4107, "type": "retractable"}, "IND": {"lat": 39.7601, "lon": -86.1639, "type": "retractable"},
        "JAX": {"lat": 30.3240, "lon": -81.6373, "type": "open"}, "KC":  {"lat": 39.0489, "lon": -94.4839, "type": "open"},
        "LV":  {"lat": 36.0909, "lon": -115.1833, "type": "dome"}, "LAC": {"lat": 33.9535, "lon": -118.3390, "type": "dome"},
        "LA":  {"lat": 33.9535, "lon": -118.3390, "type": "dome"}, "MIA": {"lat": 25.9580, "lon": -80.2389, "type": "open"},
        "MIN": {"lat": 44.9735, "lon": -93.2575, "type": "dome"}, "NE":  {"lat": 42.0909, "lon": -71.2643, "type": "open"},
        "NO":  {"lat": 29.9511, "lon": -90.0812, "type": "dome"}, "NYG": {"lat": 40.8135, "lon": -74.0745, "type": "open"},
        "NYJ": {"lat": 40.8135, "lon": -74.0745, "type": "open"}, "PHI": {"lat": 39.9008, "lon": -75.1675, "type": "open"},
        "PIT": {"lat": 40.4468, "lon": -80.0158, "type": "open"}, "SF":  {"lat": 37.4032, "lon": -121.9698, "type": "open"},
        "SEA": {"lat": 47.5952, "lon": -122.3316, "type": "open"}, "TB":  {"lat": 27.9759, "lon": -82.5033, "type": "open"},
        "TEN": {"lat": 36.1665, "lon": -86.7713, "type": "open"}, "WAS": {"lat": 38.9077, "lon": -76.8645, "type": "open"}
    }
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_forecast(team_abbr, date_obj):
        stadium = StadiumService.STADIUMS.get(team_abbr)
        if not stadium: return None
        if stadium['type'] in ['dome', 'retractable']: return {"desc": "🏟️ Stadium Closed/Dome", "is_closed": True, "temp": 72, "wind": 0, "rain": 0}
        try:
            date_str = date_obj.strftime("%Y-%m-%d")
            url = f"https://api.open-meteo.com/v1/forecast?latitude={stadium['lat']}&longitude={stadium['lon']}&daily=temperature_2m_max,precipitation_probability_max,windspeed_10m_max&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=auto&start_date={date_str}&end_date={date_str}"
            res = requests.get(url, timeout=5).json()
            if 'daily' in res:
                return {"desc": f"🌡️ {res['daily']['temperature_2m_max'][0]}°F  💨 {res['daily']['windspeed_10m_max'][0]} mph", "is_closed": False, "temp": res['daily']['temperature_2m_max'][0], "wind": res['daily']['windspeed_10m_max'][0], "rain": res['daily']['precipitation_probability_max'][0]}
        except: pass
        return {"desc": "Weather Unavailable", "is_closed": False, "temp": 70, "wind": 5, "rain": 0}

# ==========================================
# 5. MATH HELPERS
# ==========================================
def american_to_decimal(odds):
    try: odds = float(odds)
    except: return 2.0
    if odds == 0: return 2.0 
    if odds > 0: return 1 + (odds / 100) 
    else: return 1 + (100 / abs(odds))

def no_vig_two_way(d1, d2):
    if d1 <= 0 or d2 <= 0: return (0.5, 0.5)
    p1 = 1/d1
    p2 = 1/d2
    s = p1 + p2
    return (p1/s, p2/s)

def synthetic_hold(d1, d2):
    if d1 <= 0 or d2 <= 0: return 0.0
    return (1/d1 + 1/d2) - 1

def half_kelly(p, d, cap=0.05):
    if d <= 1: return 0
    b = d - 1
    q = 1 - p
    f_full = (b * p - q) / b
    f_half = f_full * 0.5
    return max(0, min(f_half, cap))

def logit(p):
    return math.log(max(p, 1e-6) / (1 - max(p, 1e-6)))

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def get_last_nfl_date_from_sched(sched_df):
    if sched_df is None or sched_df.empty or 'gameday' not in sched_df.columns: return datetime.date.today()
    if 'home_score' in sched_df.columns:
        completed = sched_df.dropna(subset=['home_score'])
        if not completed.empty: return pd.to_datetime(completed['gameday']).max().date()
    return datetime.date.today()

def compute_team_home_advantage(games, min_games=10, alpha=0.05, smooth=0.5):
    df = games.copy().dropna(subset=['home_score', 'away_score'])
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]]))
    hfa_map = {}
    for team in teams:
        g_home, g_away = df[df["home_team"] == team], df[df["away_team"] == team]
        if len(g_home) < min_games or len(g_away) < min_games: hfa_map[team] = 0.0; continue
        p_home = (g_home["home_win"].sum() + smooth) / (len(g_home) + 2 * smooth)
        
        wins_away = (1 - g_away["home_win"]).sum()
        p_away = (wins_away + smooth) / (len(g_away) + 2 * smooth)
        
        delta = np.log(p_home/(1-p_home)) - np.log(p_away/(1-p_away))
        shrink = min(1.0, (len(g_home)+len(g_away)) / (2.0 * min_games))
        hfa_map[team] = (1.0 / (1.0 + np.exp(-delta / 2.0)) - 0.5) * shrink
    return hfa_map

# ==========================================
# 6. DATA LOADER
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
    
    # Init vars
    analysis_db = pd.DataFrame()

    # 1. SCHEDULE
    sched = None
    try: sched = nfl.import_schedules([current_year])
    except: pass
    if sched is None or sched.empty:
        try: sched = pd.read_csv(f"https://github.com/nflverse/nflverse-data/raw/master/data/schedules/schedule_{current_year}.csv")
        except: pass
    
    if sched is not None: 
        if 'gameday' in sched.columns: sched['gameday'] = pd.to_datetime(sched['gameday']).dt.date
    
    last_played_date = get_last_nfl_date_from_sched(sched)

    # 2. HFA
    hfa_dict = {}
    try: hfa_dict = compute_team_home_advantage(nfl.import_schedules([current_year-3, current_year-2, current_year-1]))
    except: pass

    # 3. ANALYSIS DB
    try: analysis_db = pd.read_csv("analysis.csv")
    except: pass

    # 4. STATS (Robust)
    pbp_all, weekly_all, loaded_years, status = [], [], [], {}
    
    for yr in [current_year-1, current_year]:
        p, w = None, None
        try:
            p = nfl.import_pbp_data([yr], cache=False)
            w = nfl.import_weekly_data([yr])
            status[yr] = "✅ Lib"
        except:
            p = try_load_nflverse_csv(f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{yr}.csv.gz", "PBP", status, yr)
            w = try_load_nflverse_csv(f"https://github.com/nflverse/nflverse-data/releases/download/stats_player_week/stats_player_week_{yr}.csv.gz", "Weekly", status, yr)
            if yr not in status: status[yr] = "✅ CSV" if p is not None else "❌ Fail"
            
        if p is not None: pbp_all.append(p); loaded_years.append(yr)
        if w is not None: weekly_all.append(w)

    pbp = pd.concat(pbp_all, ignore_index=True) if pbp_all else pd.DataFrame()
    weekly = pd.concat(weekly_all, ignore_index=True) if weekly_all else pd.DataFrame()

    clf, team_stats, qb_stats = None, pd.DataFrame(), pd.DataFrame()
    
    if not pbp.empty:
        # Proper Rolling Stats Implementation
        pbp['game_date'] = pd.to_datetime(pbp['game_date'])
        
        # 1. Aggregate to Team-Game level (using Date instead of Week for safety)
        # Note: We group by game_id to handle bye weeks implicitly by just being a game list
        game_stats = pbp.groupby(['game_id', 'posteam', 'season', 'week', 'game_date']).agg({
            'epa': 'mean',
        }).reset_index()

        pass_stats = pbp[pbp['play_type']=='pass'].groupby(['game_id', 'posteam', 'season', 'week']).agg(pass_epa=('epa', 'mean')).reset_index()
        rush_stats = pbp[pbp['play_type']=='run'].groupby(['game_id', 'posteam', 'season', 'week']).agg(rush_epa=('epa', 'mean')).reset_index()

        game_stats = game_stats.merge(pass_stats, on=['game_id', 'posteam', 'season', 'week'], how='left')
        game_stats = game_stats.merge(rush_stats, on=['game_id', 'posteam', 'season', 'week'], how='left')
        
        # 2. Sort by date and De-Duplicate to ensure unique Game-Team rows
        game_stats = game_stats.drop_duplicates(subset=['game_id', 'posteam'])
        game_stats = game_stats.sort_values(['posteam', 'game_date'])
        
        # 3. Calculate Rolling Stats (Shift 1 to prevent leakage)
        for col in ['epa', 'pass_epa', 'rush_epa']:
            game_stats[f'rolling_{col}'] = game_stats.groupby('posteam')[col].transform(
                lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
            )

        # 4. Create Season-Average Stats for UI sliders (using historical averages as baseline)
        team_stats = game_stats.groupby(['season', 'posteam']).agg({
            'epa': 'mean',
            'pass_epa': 'mean',
            'rush_epa': 'mean'
        }).reset_index().rename(columns={
            'posteam': 'team',
            'epa': 'epa_off',
            'pass_epa': 'epa_pass_off',
            'rush_epa': 'epa_rush_off'
        })
        
        # Add defense stats (simplified for UI, model uses rolling)
        team_stats['avg_pass_off'] = team_stats['epa_pass_off'] 
        team_stats['avg_rush_off'] = team_stats['epa_rush_off']
        team_stats['epa_pass_def'] = 0.0 # Placeholder
        team_stats['epa_rush_def'] = 0.0
        team_stats['avg_pass_def'] = 0.0
        team_stats['avg_rush_def'] = 0.0
        team_stats['avg_tos_off'] = 1.0 # Default
        team_stats['avg_tos_def'] = 1.0

        if 'cpoe' not in pbp.columns: pbp['cpoe'] = 0.0
        qb_stats = pbp[pbp['play_type']=='pass'].groupby(['season', 'posteam', 'passer_player_name']).agg(
            qb_epa=('epa', 'mean'), qb_cpoe=('cpoe', 'mean'), dropbacks=('play_id', 'count')
        ).reset_index().sort_values('dropbacks', ascending=False)

        # Training
        if sched is not None and not sched.empty:
            games_train = sched.dropna(subset=['home_score', 'home_moneyline']).copy()
            if not games_train.empty:
                games_train = games_train[games_train['gameday'] <= last_played_date]
                games_train['home_win'] = (games_train['home_score'] > games_train['away_score']).astype(int)
                
                # Merge Rolling Stats (Home)
                # Need to merge on (Season, Week, Team)
                model_stats = game_stats[['season', 'week', 'posteam', 'rolling_pass_epa', 'rolling_rush_epa']].copy()
                
                games_train = games_train.merge(
                    model_stats, 
                    left_on=['season', 'week', 'home_team'], 
                    right_on=['season', 'week', 'posteam'], 
                    how='left'
                ).rename(columns={'rolling_pass_epa': 'home_rolling_pass_epa', 'rolling_rush_epa': 'home_rolling_rush_epa'})
                
                games_train = games_train.merge(
                    model_stats, 
                    left_on=['season', 'week', 'away_team'], 
                    right_on=['season', 'week', 'posteam'], 
                    how='left'
                ).rename(columns={'rolling_pass_epa': 'away_rolling_pass_epa', 'rolling_rush_epa': 'away_rolling_rush_epa'})
                
                games_train['diff_net_pass'] = games_train['home_rolling_pass_epa'] - games_train['away_rolling_pass_epa']
                games_train['diff_net_rush'] = games_train['home_rolling_rush_epa'] - games_train['away_rolling_rush_epa']
                
                def get_logit(r):
                    try: return logit(no_vig_two_way(american_to_decimal(r['home_moneyline']), american_to_decimal(r['away_moneyline']))[0])
                    except: return np.nan
                games_train['logit_mkt'] = games_train.apply(get_logit, axis=1)
                
                train_clean = games_train.dropna(subset=['home_win', 'logit_mkt', 'diff_net_pass', 'diff_net_rush']).copy()
                if not train_clean.empty:
                    clf = LogisticRegression()
                    # WEIGHT: 3x for current season
                    train_clean['weight'] = train_clean['season'].map(lambda x: 3.0 if x == current_year else 1.0)
                    clf.fit(train_clean[['logit_mkt', 'diff_net_pass', 'diff_net_rush']], train_clean['home_win'], sample_weight=train_clean['weight'])

    return clf, team_stats, weekly, sched, qb_stats, hfa_dict, status, loaded_years, analysis_db

# Load Sequence
loading_placeholder.empty()
with loading_placeholder: components.html(LOADING_HTML, height=450)
model_clf, team_stats_db, weekly_stats_db, sched_db, qb_stats_db, hfa_db, status_rep, loaded_years_list, analysis_db = load_nfl_data()
live_odds_map = OddsService.fetch_live_odds()
loading_placeholder.empty()

st.sidebar.markdown("### 💾 Data Diagnostics")
for k, v in status_rep.items(): st.sidebar.caption(f"**{k}:** {v}")
if hfa_db: status_slot.caption("HFA: Active")
if sched_db is None or sched_db.empty: st.error("No Schedule."); st.stop()

# ==========================================
# 7. LOGIC ENGINE
# ==========================================
class CockpitEngine:
    @staticmethod
    def get_team_leaders(team_abbr):
        if weekly_stats_db.empty: return {}
        req = {'recent_team','passing_yards','rushing_yards','receiving_yards','player_display_name','week','season','attempts'}
        if not req.issubset(set(weekly_stats_db.columns)):
             if 'attempts' not in weekly_stats_db.columns: return {} # need attempts

        team_rows = weekly_stats_db[weekly_stats_db['recent_team'] == team_abbr]
        if team_rows.empty: return {}

        # 1. Find Max Season
        max_s = team_rows['season'].max()
        curr_rows = team_rows[team_rows['season'] == max_s]
        
        # 2. Find LAST 3 WEEKS played (Volume Check)
        weeks = sorted(curr_rows['week'].unique(), reverse=True)
        recent_3 = weeks[:3]
        recent_data = curr_rows[curr_rows['week'].isin(recent_3)]
        
        leaders = {}
        
        # QB: Sum attempts over last 3 games to find starter
        if 'attempts' in recent_data.columns:
            qb_cand = recent_data.groupby('player_display_name')['attempts'].sum().reset_index()
            qb_starter = qb_cand.sort_values('attempts', ascending=False).head(1)
        else:
             # Fallback
             qb_cand = recent_data.groupby('player_display_name')['passing_yards'].sum().reset_index()
             qb_starter = qb_cand.sort_values('passing_yards', ascending=False).head(1)

        if not qb_starter.empty:
            name = qb_starter.iloc[0]['player_display_name']
            # Get stats from most recent game for that player
            p_stats = curr_rows[curr_rows['player_display_name'] == name].sort_values('week', ascending=False).head(1)
            if not p_stats.empty:
                val = p_stats.iloc[0]['passing_yards']
                leaders['QB'] = {'name': name, 'raw_yds': val, 'stat': f"Last: {val} yds"}
        
        return leaders

    @staticmethod
    def get_qb_metrics(team_abbr, season=None):
        if qb_stats_db.empty: return None
        # Try finding the 'leader' QB first
        leaders = CockpitEngine.get_team_leaders(team_abbr)
        if 'QB' in leaders:
            name = leaders['QB']['name']
            row = qb_stats_db[(qb_stats_db['posteam']==team_abbr) & (qb_stats_db['passer_player_name']==name)]
            if not row.empty: return row.sort_values('season', ascending=False).iloc[0]

        # Fallback
        starter = qb_stats_db[(qb_stats_db['posteam']==team_abbr)].head(1)
        return starter.iloc[0] if not starter.empty else None

    @staticmethod
    def generate_props(home, away, h_win_prob, a_win_prob):
        h_lead = CockpitEngine.get_team_leaders(home)
        a_lead = CockpitEngine.get_team_leaders(away)
        props = []
        # Moneyline - Correct 7 args
        props.append(PropLeg(f"{home}_ML", f"{home} To Win", 1.0/max(h_win_prob,0.01), h_win_prob, "Team Win", home, "Moneyline", ""))
        props.append(PropLeg(f"{away}_ML", f"{away} To Win", 1.0/max(a_win_prob,0.01), a_win_prob, "Team Win", away, "Moneyline", ""))
        
        def add(p, cat, team):
            if not p: return
            line = round(p['raw_yds'] * 1.0 / 5) * 5 + 0.5
            props.append(PropLeg(f"{p['name']} Over", f"{p['name']} Over {line} {cat} Yds", 1.91, 0.50, cat, team, p['stat']))

        if 'QB' in h_lead: add(h_lead['QB'], "Passing", home)
        return props

    @staticmethod
    def get_default_sliders(row, weather_data):
        s = {'h_qb': 5, 'h_pwr': 5, 'h_def': 5, 'a_qb': 5, 'a_pwr': 5, 'a_def': 5, 'rest': 5, 'news': 5, 'weath': 0, 'hfa': 5}
        # Simplified for robustness
        return s

    @staticmethod
    def get_simple_gold_prediction(home, away, season):
        # Simplified to prevent crash if columns missing
        return None

    @staticmethod
    def calc_win_prob(market_prob, row, sliders):
        logit_mkt = logit(market_prob)
        def epa(v): return (v - 5) * 0.03
        
        hp, hr, hd = epa(sliders['ph_qb']), epa(sliders['ph_pwr']), epa(sliders['ph_def'])
        ap, ar, ad = epa(sliders['pa_qb']), epa(sliders['pa_pwr']), epa(sliders['pa_def'])
        
        d_pass = (hp - ap) + (hd - ad)
        d_rush = (hr - ar) + (hd - ad)
        
        # Base from Stats - use 0.0 if not found to prevent crash
        base_p, base_r = 0.0, 0.0

        model_p = market_prob
        if model_clf:
            x = pd.DataFrame([[logit_mkt, base_p + d_pass, base_r + d_rush]], 
                           columns=['logit_mkt', 'diff_net_pass', 'diff_net_rush'])
            try:
                model_raw = model_clf.predict_proba(x)[0, 1]
                model_p = (0.7 * market_prob) + (0.3 * model_raw)
            except: pass

        news = (5 - sliders['wn']) * 0.015
        weather = -0.02 * (sliders['ww']/5) if sliders['ww'] > 0 else 0
        rest = ((sliders.get('rh',7)-sliders.get('ra',7))*0.005) * (sliders['wr']/2.0)
        hfa = (sliders['hfa'] - 5) * 0.015
        
        final = min(max(model_p + news + weather + rest + hfa, 0.01), 0.99)
        return final, {"AI Model": model_p, "News": news, "Weather": weather, "Rest": rest, "HFA": hfa}

# ==========================================
# 8. RENDERERS
# ==========================================
def render_game_card(i, row, bankroll, kelly):
    home, away = row['home_team'], row['away_team']
    season = row['season']
    
    def safe_int(v):
        try: return int(v)
        except: return -110
    
    oh, oa = safe_int(row.get('home_moneyline')), safe_int(row.get('away_moneyline'))
    dh, da = american_to_decimal(oh), american_to_decimal(oa)
    hold = synthetic_hold(dh, da)
    
    live = live_odds_map.get((home, away))
    if not live: live = live_odds_map.get((away, home))
    if live: 
        oh, oa = live['home_am'], live['away_am']
        src = "DraftKings (Live)"
    else: src = "Schedule (Cached)"

    h_qb = CockpitEngine.get_qb_metrics(home) 
    a_qb = CockpitEngine.get_qb_metrics(away) 
    weather_info = StadiumService.get_forecast(home, sel_date)
    defs = CockpitEngine.get_default_sliders(row, weather_info)

    with st.container(border=True):
        if not analysis_db.empty:
             match = analysis_db[(analysis_db['home']==home) & (analysis_db['away']==away)]
             if not match.empty:
                 with st.expander("🔍 Expert Analysis"):
                     st.info(f"**{match.iloc[0]['edge_blurb']}**")
                     if pd.notnull(match.iloc[0]['deep_dive_link']): st.markdown(f"[Read More]({match.iloc[0]['deep_dive_link']})")

        c1, c2 = st.columns([3, 1])
        c1.subheader(f"{away} @ {home}")
        c1.caption(f"{row['gametime']} ET")

        with c2:
            st.markdown("**QB Matchup**")
            qb_h_name = h_qb['passer_player_name'] if h_qb is not None else 'Unknown'
            qb_a_name = a_qb['passer_player_name'] if a_qb is not None else 'Unknown'
            st.caption(f"{qb_a_name} vs {qb_h_name}")

        with st.expander("See QB Stats"):
             c_qa, c_qh = st.columns(2)
             if a_qb is not None:
                 c_qa.markdown(f"**{a_qb['passer_player_name']}** ({away})")
                 c_qa.metric("EPA", f"{a_qb['qb_epa']:.2f}")
             if h_qb is not None:
                 c_qh.markdown(f"**{h_qb['passer_player_name']}** ({home})")
                 c_qh.metric("EPA", f"{h_qb['qb_epa']:.2f}")

        st.divider()
        c_odds, c_aw, c_hm, c_res = st.columns([1, 1.5, 1.5, 1.2])
        
        with c_odds:
            st.markdown(f"##### 🏦 {src}")
            oa_i = st.number_input(f"{away}", value=oa, step=5, key=f"oa_{i}")
            oh_i = st.number_input(f"{home}", value=oh, step=5, key=f"oh_{i}")
            dal, dhl = american_to_decimal(oa_i), american_to_decimal(oh_i)
            pmkt = no_vig_two_way(dhl, dal)[0]
            st.progress(pmkt if pmkt >= 0.5 else 1-pmkt, f"Mkt: {home if pmkt>=0.5 else away} {pmkt if pmkt>=0.5 else 1-pmkt:.1%}")

        with c_aw:
            st.markdown(f"##### {away}")
            pa_qb = st.slider("QB", 0, 10, defs['a_qb'], key=f"pq_{i}")
            pa_pwr = st.slider("Pwr", 0, 10, defs['a_pwr'], key=f"pp_{i}")
            pa_def = st.slider("Def", 0, 10, defs['a_def'], key=f"pd_{i}")

        with c_hm:
            st.markdown(f"##### {home}")
            ph_qb = st.slider("QB", 0, 10, defs['h_qb'], key=f"hq_{i}")
            ph_pwr = st.slider("Pwr", 0, 10, defs['h_pwr'], key=f"hp_{i}")
            ph_def = st.slider("Def", 0, 10, defs['h_def'], key=f"hd_{i}")
            
        st.markdown("##### Context")
        cc1, cc2, cc3, cc4 = st.columns(4)
        wr = st.slider("Rest", 0, 10, defs['rest'], key=f"r_{i}", help="5=Neutral")
        wn = st.slider("News", 0, 10, defs['news'], key=f"n_{i}", help="0=Home Good, 10=Away Good")
        hfa = st.slider("HFA", 0, 10, defs['hfa'], key=f"h_{i}", help="Auto-calculated advantage from history")
        wd = weather_info and weather_info['is_closed']
        ww = st.slider("Weather", 0, 10, defs['weath'], disabled=wd, key=f"w_{i}", help="5=Bad Weather (Wind/Rain)")

        with c_res:
            sl = {'pa_qb': pa_qb, 'pa_pwr': pa_pwr, 'pa_def': pa_def, 'ph_qb': ph_qb, 'ph_pwr': ph_pwr, 'ph_def': ph_def, 'wr': wr, 'wn': wn, 'ww': ww, 'hfa': hfa, 'rh': row.get('home_rest',7), 'ra': row.get('away_rest',7)}
            final_p_home, breakdown = CockpitEngine.calc_win_prob(pmkt, row, sl)
            
            # Determine Verdict Side
            if final_p_home >= 0.5:
                pick_team = home
                pick_prob = final_p_home
                pick_odds = dhl
                mkt_ref = pmkt 
            else:
                pick_team = away
                pick_prob = 1.0 - final_p_home
                pick_odds = dal
                mkt_ref = 1.0 - pmkt 
            
            p_be = 1 / pick_odds
            edge = pick_prob - p_be
            ev = pick_prob * pick_odds - 1
            z = edge / 0.04
            
            st.markdown("##### 🚀 Verdict")
            st.markdown(f"**Pick:** {pick_team}")
            st.metric("Prob", f"{pick_prob:.1%}", delta=f"{pick_prob-mkt_ref:.1%}")
            
            with st.expander("Math"):
                for k,v in breakdown.items(): st.write(f"{k}: {v:+.1%}")
            
            max_w = bankroll * 0.05
            
            if z >= 1.28 and ev > 0:
                stake = half_kelly(pick_prob, pick_odds) * bankroll * kelly
                stake = min(stake, max_w)
                st.success(f"BET {pick_team} ${stake:.0f}")
            else:
                if ev > 0: st.warning("Weak Edge")
                else: st.info("No Edge")

        if st.button(f"🧠 Strategy Lab", key=f"sl_{i}"):
            st.session_state['sl_active'] = True
            st.session_state['sl_home'] = home
            st.session_state['sl_away'] = away
            st.session_state['sl_h_prob'] = final_p_home
            st.session_state['sl_legs'] = []
            st.rerun()

if st.session_state.get('sl_active', False):
    render_strategy_lab(bankroll)
else:
    if not sched_db.empty:
        sched_db['gameday'] = sched_db['gameday'].astype(str)
        day_games = sched_db[sched_db['gameday'] == str(sel_date)]
        if day_games.empty: st.warning(f"No games on {sel_date}")
        else:
            for i, r in day_games.iterrows(): render_game_card(i, r, bankroll, kelly)
