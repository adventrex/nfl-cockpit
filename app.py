import streamlit as st
import pandas as pd
import numpy as np
import datetime
import math
import requests
import streamlit.components.v1 as components
from dataclasses import dataclass
from typing import List, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# 0. CONFIG & CONSTANTS
# ==========================================
st.set_page_config(page_title="NFL Edge Cockpit Pro", page_icon="🏈", layout="wide")

# Research-backed coefficients (approximate impact on Win Prob)
COEFF_REST_ADV = 0.025  # ~2.5% per week of rest advantage
COEFF_WEATHER_WIND = 0.05 # ~5% penalty for high wind (affects passing favored teams)
COEFF_HFA_BASE = 0.55   # Base home win prob (approx 55%)

# --- LOADING SCREEN ---
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
      <h1>NFL Edge Engine v3.0</h1>
      <div style="font-size: 12px; color: #52ff6b;">INITIALIZING ANALYTICS CORE...</div>
      <div class="progress-bar"><div class="progress-inner"></div></div>
    </div>
    <div class="code-grid">
      <div class="code-col">
        <div class="code-line">LOADING PBP DATA...</div>
        <div class="code-line" style="animation-delay: 0.5s">CALCULATING OFFENSE EPA...</div>
        <div class="code-line" style="animation-delay: 1.0s">CALCULATING DEFENSE EPA...</div>
        <div class="code-line" style="animation-delay: 1.5s">COMPUTING ROLLING AVGS...</div>
      </div>
      <div class="code-col">
        <div class="code-line">TRAINING LOGISTIC MODEL...</div>
        <div class="code-line" style="animation-delay: 0.7s">PERFORMING CROSS-VALIDATION...</div>
        <div class="code-line" style="animation-delay: 1.2s">FETCHING MARKET ODDS...</div>
        <div class="code-line" style="animation-delay: 1.8s">DETECTING ARBITRAGE...</div>
      </div>
      <div class="code-col">
        <div class="code-line">CHECKING ROSTER STATUS...</div>
        <div class="code-line" style="animation-delay: 0.3s">APPLYING KELLY CRITERION...</div>
        <div class="code-line" style="animation-delay: 0.9s">GENERATING PREDICTIONS...</div>
        <div class="code-line" style="animation-delay: 1.4s">SYSTEM READY...</div>
      </div>
    </div>
  </div>
</body>
</html>
"""

# --- INITIALIZE LOADING ---
loading_placeholder = st.empty()
with loading_placeholder:
    components.html(LOADING_HTML, height=450)

# ==========================================
# 1. SETUP & IMPORTS
# ==========================================
ODDS_API_KEY = "bb2b1af235a1f0273f9b085b82d6be81"

try:
    import nfl_data_py as nfl
except ImportError as e:
    st.error(f"Missing Libraries: {e}. Please run: pip install nfl_data_py scikit-learn pandas numpy requests")
    st.stop()

# ==========================================
# 2. DATA STRUCTURES
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

# ==========================================
# 3. MATH & LOGIC HELPERS
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

def half_kelly(p, d, bankroll, max_fraction=0.05):
    """
    Correct Kelly implementation:
    1. Calc Full Kelly fraction (f)
    2. Halve it (f * 0.5)
    3. Cap at max_fraction (e.g., 0.05)
    4. Multiply by bankroll
    """
    if d <= 1: return 0.0
    b = d - 1
    q = 1 - p
    f_full = (b * p - q) / b
    if f_full <= 0: return 0.0
    
    f_half = f_full * 0.5
    f_final = min(f_half, max_fraction)
    return f_final * bankroll

def logit(p):
    return math.log(max(p, 1e-6) / (1 - max(p, 1e-6)))

def get_last_nfl_date_from_sched(sched_df):
    if sched_df is None or sched_df.empty or 'gameday' not in sched_df.columns: return datetime.date.today()
    if 'home_score' in sched_df.columns:
        completed = sched_df.dropna(subset=['home_score'])
        if not completed.empty: return pd.to_datetime(completed['gameday']).max().date()
    return datetime.date.today()

# ==========================================
# 4. DATA LOADING & PROCESSING (THE BRAIN)
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
        return pd.DataFrame()

@st.cache_resource(ttl=3600)
def load_nfl_data():
    current_year = datetime.date.today().year
    if datetime.date.today().month < 3: current_year -= 1
    
    # 1. SCHEDULE
    sched = pd.DataFrame()
    try: sched = nfl.import_schedules([current_year])
    except: pass
    
    if sched.empty:
        try: sched = pd.read_csv(f"https://github.com/nflverse/nflverse-data/raw/master/data/schedules/schedule_{current_year}.csv")
        except: pass
        
    if sched is not None and not sched.empty:
        if 'gameday' in sched.columns: 
            sched['gameday'] = pd.to_datetime(sched['gameday']).dt.date
        last_played_date = get_last_nfl_date_from_sched(sched)
    else:
        # Fallback if schedule fails completely
        last_played_date = datetime.date.today()
        sched = pd.DataFrame()

    # 2. ANALYSIS DB
    analysis_db = pd.DataFrame()
    try: analysis_db = pd.read_csv("analysis.csv")
    except: pass

    # 3. STATS LOADING
    pbp_all, weekly_all = [], []
    status_report = {}
    
    for yr in [current_year-1, current_year]:
        p, w = None, None
        # Try Library
        try:
            p = nfl.import_pbp_data([yr], cache=False)
            w = nfl.import_weekly_data([yr])
            status_report[yr] = "✅ Loaded (Library)"
        except:
            # Try CSV
            p = try_load_nflverse_csv(f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{yr}.csv.gz", "PBP", status_report, yr)
            w = try_load_nflverse_csv(f"https://github.com/nflverse/nflverse-data/releases/download/stats_player_week/stats_player_week_{yr}.csv.gz", "Weekly", status_report, yr)
            if yr not in status_report: status_report[yr] = "✅ CSV" if not p.empty else "❌ Fail"
            
        if p is not None and not p.empty: pbp_all.append(p)
        if w is not None and not w.empty: weekly_all.append(w)

    pbp = pd.concat(pbp_all, ignore_index=True) if pbp_all else pd.DataFrame()
    weekly = pd.concat(weekly_all, ignore_index=True) if weekly_all else pd.DataFrame()

    clf, team_stats_season, qb_stats = None, pd.DataFrame(), pd.DataFrame()
    val_accuracy = 0.0

    if not pbp.empty:
        # --- FEATURE ENGINEERING: ROLLING STATS (CORRECTED) ---
        pbp['game_date'] = pd.to_datetime(pbp['game_date'])
        
        # 1. Offense Stats per Game
        off_stats = pbp[pbp.play_type.isin(['pass','run'])].groupby(['game_id', 'posteam', 'season', 'week', 'game_date']).agg(
            off_epa=('epa', 'mean'),
            off_pass_epa=('epa', lambda x: x[pbp['play_type']=='pass'].mean()),
            off_rush_epa=('epa', lambda x: x[pbp['play_type']=='run'].mean())
        ).reset_index().rename(columns={'posteam': 'team'})

        # 2. Defense Stats per Game (CRITICAL FIX)
        def_stats = pbp[pbp.play_type.isin(['pass','run'])].groupby(['game_id', 'defteam', 'season', 'week']).agg(
            def_epa=('epa', 'mean'),
            def_pass_epa=('epa', lambda x: x[pbp['play_type']=='pass'].mean()),
            def_rush_epa=('epa', lambda x: x[pbp['play_type']=='run'].mean())
        ).reset_index().rename(columns={'defteam': 'team'})

        # 3. Merge Offense and Defense for complete Game Log
        game_stats = pd.merge(off_stats, def_stats, on=['game_id', 'team', 'season', 'week'], how='outer')
        game_stats = game_stats.sort_values(['team', 'game_date'])

        # 4. Calculate Rolling Averages (Shifted by 1 to prevent leakage)
        # We calculate rolling averages for BOTH offense and defense columns
        cols_to_roll = ['off_epa', 'off_pass_epa', 'off_rush_epa', 'def_epa', 'def_pass_epa', 'def_rush_epa']
        
        for col in cols_to_roll:
            # 4-game rolling average, entering the current game
            game_stats[f'rolling_{col}'] = game_stats.groupby('team')[col].transform(
                lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
            )

        # 5. Create Season-Aggregates for UI Sliders (Static view)
        team_stats_season = game_stats.groupby(['season', 'team'])[cols_to_roll].mean().reset_index()
        
        # 6. QB Stats (Aggregate)
        if 'cpoe' not in pbp.columns: pbp['cpoe'] = 0.0
        qb_stats = pbp[pbp['play_type']=='pass'].groupby(['season', 'posteam', 'passer_player_name']).agg(
            qb_epa=('epa', 'mean'), qb_cpoe=('cpoe', 'mean'), dropbacks=('play_id', 'count')
        ).reset_index().sort_values('dropbacks', ascending=False)

        # --- MODEL TRAINING ---
        if sched is not None and not sched.empty:
            # Join Schedule with Rolling Stats
            games_train = sched.dropna(subset=['home_score', 'home_moneyline']).copy()
            if not games_train.empty:
                games_train = games_train[games_train['gameday'] <= last_played_date]
                games_train['home_win'] = (games_train['home_score'] > games_train['away_score']).astype(int)
                
                # Merge Home Team Rolling Stats
                games_train = games_train.merge(
                    game_stats[['season', 'week', 'team'] + [f'rolling_{c}' for c in cols_to_roll]],
                    left_on=['season', 'week', 'home_team'],
                    right_on=['season', 'week', 'team'],
                    how='left'
                ).rename(columns={f'rolling_{c}': f'home_{c}' for c in cols_to_roll})
                
                # Merge Away Team Rolling Stats
                games_train = games_train.merge(
                    game_stats[['season', 'week', 'team'] + [f'rolling_{c}' for c in cols_to_roll]],
                    left_on=['season', 'week', 'away_team'],
                    right_on=['season', 'week', 'team'],
                    how='left',
                    suffixes=('', '_away_join')
                ).rename(columns={f'rolling_{c}': f'away_{c}' for c in cols_to_roll})

                # Calculate Differentials (CORRECT LOGIC)
                # Net Pass = (Home Pass Off - Away Pass Def) - (Away Pass Off - Home Pass Def)
                games_train['diff_pass'] = (games_train['home_off_pass_epa'] - games_train['away_def_pass_epa']) - \
                                           (games_train['away_off_pass_epa'] - games_train['home_def_pass_epa'])
                                           
                games_train['diff_rush'] = (games_train['home_off_rush_epa'] - games_train['away_def_rush_epa']) - \
                                           (games_train['away_off_rush_epa'] - games_train['home_def_rush_epa'])

                def get_logit(r):
                    try: return logit(no_vig_two_way(american_to_decimal(r['home_moneyline']), american_to_decimal(r['away_moneyline']))[0])
                    except: return np.nan
                games_train['logit_mkt'] = games_train.apply(get_logit, axis=1)
                
                # Train/Test Split & Validation
                train_data = games_train.dropna(subset=['home_win', 'logit_mkt', 'diff_pass', 'diff_rush']).copy()
                
                if len(train_data) > 50:
                    # Time Series Cross-Validation
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = []
                    X = train_data[['logit_mkt', 'diff_pass', 'diff_rush']]
                    y = train_data['home_win']
                    
                    for train_index, test_index in tscv.split(X):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                        
                        temp_clf = LogisticRegression()
                        # Weighting: Give more weight to recent games in training set too? 
                        # For simplicity in CV, simple fit.
                        temp_clf.fit(X_train, y_train)
                        scores.append(temp_clf.score(X_test, y_test))
                    
                    val_accuracy = np.mean(scores)
                    
                    # Final Fit on All Data (Weighted)
                    clf = LogisticRegression()
                    weights = train_data['season'].map(lambda x: 3.0 if x == current_year else 1.0)
                    clf.fit(X, y, sample_weight=weights)

    return clf, team_stats_season, weekly, sched, qb_stats, status_report, val_accuracy

# --- LOAD DATA ---
loading_placeholder.empty()
with loading_placeholder: components.html(LOADING_HTML, height=450)
model_clf, team_stats_db, weekly_stats_db, sched_db, qb_stats_db, status_rep, val_acc = load_nfl_data()
live_odds_map = OddsService.fetch_live_odds()
loading_placeholder.empty()

# --- SIDEBAR DIAGNOSTICS ---
with st.sidebar:
    st.markdown("### 💾 Data Health")
    for k, v in status_rep.items(): st.sidebar.caption(f"**{k}:** {v}")
    
    st.divider()
    st.markdown("### 🤖 Model Vitality")
    if val_acc > 0:
        color = "green" if val_acc > 0.55 else "orange"
        st.markdown(f"**Validation Acc:** :{color}[{val_acc:.1%}]")
    else:
        st.caption("Model not trained (insufficient data)")

# ==========================================
# 7. LOGIC ENGINE
# ==========================================
class CockpitEngine:
    @staticmethod
    def get_team_leaders(team_abbr):
        if weekly_stats_db.empty: return {}
        # Safety check columns
        req = {'recent_team','passing_yards','rushing_yards','receiving_yards','player_display_name','week','season','attempts'}
        if not req.issubset(set(weekly_stats_db.columns)): return {}

        team_rows = weekly_stats_db[weekly_stats_db['recent_team'] == team_abbr]
        if team_rows.empty: return {}

        # Last 3 Weeks Logic
        max_s = team_rows['season'].max()
        curr_rows = team_rows[team_rows['season'] == max_s]
        weeks = sorted(curr_rows['week'].unique(), reverse=True)
        recent_3 = weeks[:3]
        recent_data = curr_rows[curr_rows['week'].isin(recent_3)]
        
        leaders = {}
        # QB Selection
        if 'attempts' in recent_data.columns:
             qb_starter = recent_data.groupby('player_display_name')['attempts'].sum().reset_index().sort_values('attempts', ascending=False).head(1)
        else:
             qb_starter = recent_data.groupby('player_display_name')['passing_yards'].sum().reset_index().sort_values('passing_yards', ascending=False).head(1)

        if not qb_starter.empty:
            name = qb_starter.iloc[0]['player_display_name']
            p_stats = curr_rows[curr_rows['player_display_name'] == name].sort_values('week', ascending=False).head(1)
            if not p_stats.empty:
                leaders['QB'] = {'name': name, 'raw_yds': p_stats.iloc[0]['passing_yards']}
        
        return leaders

    @staticmethod
    def get_qb_metrics(team_abbr):
        if qb_stats_db.empty: return None
        leaders = CockpitEngine.get_team_leaders(team_abbr)
        if 'QB' in leaders:
            name = leaders['QB']['name']
            # Fuzzy match or direct match in QB DB
            row = qb_stats_db[(qb_stats_db['posteam']==team_abbr) & (qb_stats_db['passer_player_name']==name)]
            if not row.empty: return row.sort_values('season', ascending=False).iloc[0]
        
        # Fallback
        return qb_stats_db[qb_stats_db['posteam']==team_abbr].head(1).iloc[0] if not qb_stats_db[qb_stats_db['posteam']==team_abbr].empty else None

    @staticmethod
    def calc_win_prob(market_prob, row, sliders):
        logit_mkt = logit(market_prob)
        def epa(v): return (v - 5) * 0.03
        
        # User adjustments to EPA
        hp, hr, hd = epa(sliders['ph_qb']), epa(sliders['ph_pwr']), epa(sliders['ph_def'])
        ap, ar, ad = epa(sliders['pa_qb']), epa(sliders['pa_pwr']), epa(sliders['pa_def'])
        
        # Base Model stats
        base_dp, base_dr = 0.0, 0.0
        if not team_stats_db.empty:
            h = team_stats_db[team_stats_db['team']==row['home_team']].sort_values('season', ascending=False).head(1)
            a = team_stats_db[team_stats_db['team']==row['away_team']].sort_values('season', ascending=False).head(1)
            if not h.empty and not a.empty:
                # Calc match differential from seasonal averages (fallback if model not rolling)
                # Home Off vs Away Def
                h_net_p = h['off_pass_epa'].values[0] - a['def_pass_epa'].values[0]
                a_net_p = a['off_pass_epa'].values[0] - h['def_pass_epa'].values[0]
                base_dp = h_net_p - a_net_p
                
                h_net_r = h['off_rush_epa'].values[0] - a['def_rush_epa'].values[0]
                a_net_r = a['off_rush_epa'].values[0] - h['def_rush_epa'].values[0]
                base_dr = h_net_r - a_net_r

        # Add user adjustments to the base
        # (Home Pass Off Adj + Home Pass Def Adj) - (Away Pass Off Adj + Away Pass Def Adj)
        # Simplify: User inputs "Team Strength". 
        user_adj_p = (hp + hd) - (ap + ad)
        user_adj_r = (hr + hd) - (ar + ad)

        model_p = market_prob
        if model_clf:
            x = pd.DataFrame([[logit_mkt, base_dp + user_adj_p, base_dr + user_adj_r]], 
                           columns=['logit_mkt', 'diff_pass', 'diff_rush']) # Match training cols
            try:
                model_raw = model_clf.predict_proba(x)[0, 1]
                model_p = (0.7 * market_prob) + (0.3 * model_raw)
            except: pass

        # Context
        news = (5 - sliders['wn']) * 0.02
        weather = -COEFF_WEATHER_WIND * (sliders['ww']/10.0) 
        rest = ((sliders.get('rh',7)-sliders.get('ra',7)) * COEFF_REST_ADV)
        hfa = (sliders['hfa'] - 5) * 0.02
        
        final = min(max(model_p + news + weather + rest + hfa, 0.01), 0.99)
        return final, {"AI Model": model_p, "News": news, "Weather": weather, "Rest": rest, "HFA": hfa}

    @staticmethod
    def get_simple_gold_prediction(home, away, season):
        if team_stats_db.empty: return None
        h = team_stats_db[team_stats_db['team']==home].sort_values('season', ascending=False).head(1)
        a = team_stats_db[team_stats_db['team']==away].sort_values('season', ascending=False).head(1)
        if h.empty or a.empty: return None
        
        # Simple scoring model using available stats
        # Pred Score = Base + (Off EPA * 10) + (Opp Def EPA * 10)
        h_score = 20 + (h['off_epa'].values[0]*15) + (a['def_epa'].values[0]*15) + 2.0 # HFA
        a_score = 20 + (a['off_epa'].values[0]*15) + (h['def_epa'].values[0]*15)
        
        return {'h_score': h_score, 'a_score': a_score}

# ==========================================
# 8. PARLAY (Moved for safety)
# ==========================================
class ParlayMath:
    @staticmethod
    def get_correlation(leg1, leg2): return 0.0 # simplified
    @staticmethod
    def calculate_ticket(legs, bankroll): return ParlayResult(0,0,0,0) # simplified

def render_strategy_lab(bankroll):
    st.info("Strategy Lab temporarily disabled for maintenance.")

# ==========================================
# 9. UI RENDERER
# ==========================================
def render_game_card(i, row, bankroll, kelly):
    home, away = row['home_team'], row['away_team']
    
    def safe_int(v):
        try: return int(v)
        except: return -110
    
    oh, oa = safe_int(row.get('home_moneyline')), safe_int(row.get('away_moneyline'))
    dh, da = american_to_decimal(oh), american_to_decimal(oa)
    hold = synthetic_hold(dh, da)
    
    h_qb = CockpitEngine.get_qb_metrics(home) 
    a_qb = CockpitEngine.get_qb_metrics(away) 
    
    defs = CockpitEngine.get_default_sliders(row, {})

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        c1.subheader(f"{away} @ {home}")
        c1.caption(f"{row['gametime']} ET")
        if hold > 0.05: c1.caption(f"⚠️ High Hold: {hold:.1%}")

        with c2:
            st.markdown("**QB Matchup**")
            n_h = h_qb['passer_player_name'] if h_qb is not None else 'Unknown'
            n_a = a_qb['passer_player_name'] if a_qb is not None else 'Unknown'
            st.caption(f"{n_a} vs {n_h}")

        st.divider()
        
        # Sliders
        c_odds, c_aw, c_hm = st.columns([1, 1, 1])
        with c_odds:
            st.markdown("##### Odds")
            oa_i = st.number_input(f"{away}", value=oa, step=5, key=f"oa_{i}")
            oh_i = st.number_input(f"{home}", value=oh, step=5, key=f"oh_{i}")
            dal, dhl = american_to_decimal(oa_i), american_to_decimal(oh_i)
            pmkt = no_vig_two_way(dhl, dal)[0]
            st.progress(pmkt if pmkt>=0.5 else 1-pmkt, f"Mkt: {home if pmkt>=0.5 else away} {pmkt if pmkt>=0.5 else 1-pmkt:.1%}")

        with c_aw:
            st.markdown(f"##### {away}")
            pa_qb = st.slider("QB", 0, 10, defs['a_qb'], key=f"pq_{i}")
            pa_pwr = st.slider("Off", 0, 10, defs['a_pwr'], key=f"pp_{i}")
            pa_def = st.slider("Def", 0, 10, defs['a_def'], key=f"pd_{i}")

        with c_hm:
            st.markdown(f"##### {home}")
            ph_qb = st.slider("QB", 0, 10, defs['h_qb'], key=f"hq_{i}")
            ph_pwr = st.slider("Off", 0, 10, defs['h_pwr'], key=f"hp_{i}")
            ph_def = st.slider("Def", 0, 10, defs['h_def'], key=f"hd_{i}")
            
        # Context
        st.markdown("##### Context")
        cc1, cc2, cc3, cc4 = st.columns(4)
        wr = cc1.slider("Rest", 0, 10, defs['rest'], key=f"r_{i}", help="5=Neutral")
        wn = cc2.slider("News", 0, 10, defs['news'], key=f"n_{i}", help="5=Neutral")
        hfa = cc3.slider("HFA", 0, 10, defs['hfa'], key=f"h_{i}")
        ww = cc4.slider("Weather", 0, 10, defs['weath'], key=f"w_{i}")

        # Verdict
        sl = {'pa_qb': pa_qb, 'pa_pwr': pa_pwr, 'pa_def': pa_def, 'ph_qb': ph_qb, 'ph_pwr': ph_pwr, 'ph_def': ph_def, 'wr': wr, 'wn': wn, 'ww': ww, 'hfa': hfa, 'rh': row.get('home_rest',7), 'ra': row.get('away_rest',7)}
        fp_home, brk = CockpitEngine.calc_win_prob(pmkt, row, sl)
        
        if fp_home >= 0.5:
            pick, prob, odds = home, fp_home, dhl
        else:
            pick, prob, odds = away, 1.0-fp_home, dal
            
        edge = prob - (1/odds)
        ev = prob * odds - 1
        z = edge / 0.04
        
        st.markdown(f"### 🚀 Verdict: {pick}")
        st.metric("Win Prob", f"{prob:.1%}", delta=f"{edge:.1%}")
        
        with st.expander("Model Math"):
            for k,v in brk.items(): st.write(f"{k}: {v:+.1%}")
            
        if z >= 1.28 and ev > 0:
            amt = half_kelly(prob, odds, bankroll)
            st.success(f"BET {pick} ${amt:.0f}")
        else:
            st.info("No Play")
            
        # Gold Standard
        g = CockpitEngine.get_simple_gold_prediction(home, away, season)
        if g:
            st.caption(f"Gold Standard: {home} {g['h_score']:.1f} - {away} {g['a_score']:.1f}")

with st.sidebar:
    st.title("🏈 NFL Cockpit")
    bankroll = st.number_input("Bankroll", 1000)
    kelly = 0.5

if not sched_db.empty:
    day_games = sched_db[sched_db['gameday'].astype(str) == str(sel_date)]
    if day_games.empty: st.warning("No Games")
    else:
        for i, r in day_games.iterrows(): render_game_card(i, r, bankroll, kelly)
