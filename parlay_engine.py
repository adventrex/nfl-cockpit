import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class PropLeg:
    leg_id: str
    description: str
    decimal_odds: float
    p_model: float
    category: str
    team: str          # New: Helps us correlate teammates
    recent_stat: str   # New: Context for the user (e.g. "Last Wk: 250 yds")

@dataclass
class ParlayResult:
    legs: List[PropLeg]
    final_odds: float
    win_prob: float
    ev: float
    kelly_stake: float

class ParlayMath:
    @staticmethod
    def get_correlation(leg1: PropLeg, leg2: PropLeg):
        """
        Calculates correlation between two legs based on category AND team relationship.
        """
        # 1. Same Leg Check
        if leg1.leg_id == leg2.leg_id: return 1.0
        
        # 2. Team Relationship
        same_team = (leg1.team == leg2.team)
        
        # 3. Base Matrix (Simplified Logic)
        # Keys: (Category1, Category2)
        # Values: (Same Team Corr, Opponent Corr)
        corr_map = {
            frozenset(['Team Win', 'Passing']): (0.50, -0.20),   # Win correlates with QB stats
            frozenset(['Team Win', 'Rushing']): (0.40, -0.30),   # Win correlates with RB stats (clock kill)
            frozenset(['Team Win', 'Receiving']): (0.45, -0.20),
            frozenset(['Passing', 'Receiving']): (0.65, 0.10),   # QB/WR highly correlated
            frozenset(['Passing', 'Rushing']): (0.05, 0.00),     # Mildly uncorrelated
            frozenset(['Rushing', 'Rushing']): (-0.15, 0.00),    # RBs steal from each other
        }
        
        # Default if category not found
        base_corr = 0.05
        
        key = frozenset([leg1.category, leg2.category])
        if key in corr_map:
            vals = corr_map[key]
            base_corr = vals[0] if same_team else vals[1]
            
        return base_corr

    @staticmethod
    def price_parlay(legs: List[PropLeg]):
        """
        Prices a multi-leg parlay with correlation adjustments.
        """
        if not legs: return 1.0, 0.0
        
        # 1. Naive Probability (Product of all legs)
        naive_prob = 1.0
        for leg in legs:
            naive_prob *= (1 / leg.decimal_odds) # Implied prob from odds
            
        # 2. Correlation Boost
        # We calculate the average pairwise correlation of the entire set
        total_corr = 0.0
        pairs = 0
        
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                c = ParlayMath.get_correlation(legs[i], legs[j])
                total_corr += c
                pairs += 1
                
        avg_corr = total_corr / pairs if pairs > 0 else 0
        
        # Boost factor: If highly correlated, outcome is MORE likely than naive product
        # Heuristic: prob_boost = (1 + avg_corr * decay)
        boost = 1 + (avg_corr * 0.6) 
        
        final_prob = min(naive_prob * boost, 0.95)
        
        # 3. Convert to Odds (add SGP vig)
        fair_odds = 1 / final_prob
        sgp_odds = fair_odds * 0.85 # 15% vig
        
        return max(sgp_odds, 1.01), final_prob

    @staticmethod
    def find_best_additions(current_legs: List[PropLeg], candidates: List[PropLeg], top_n=3):
        """
        Finds the best next leg to add to the current ticket.
        """
        scored_candidates = []
        
        for cand in candidates:
            # Skip if already in ticket
            if any(l.leg_id == cand.leg_id for l in current_legs): continue
            
            # Skip if conflicts (e.g. Same category for same player? simplified here)
            
            # Score based on Average Correlation with CURRENT ticket
            avg_corr_with_ticket = 0.0
            for existing in current_legs:
                avg_corr_with_ticket += ParlayMath.get_correlation(existing, cand)
            
            avg_corr_with_ticket /= len(current_legs)
            
            # EV Score
            ev_score = (cand.p_model * cand.decimal_odds) - 1
            
            # Combined Score: We want High Correlation + High EV
            final_score = (avg_corr_with_ticket * 2.0) + (ev_score * 1.0)
            
            scored_candidates.append((final_score, cand))
            
        # Sort descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [x[1] for x in scored_candidates[:top_n]]

    @staticmethod
    def calculate_ticket(legs: List[PropLeg], bankroll: float):
        final_odds, joint_prob = ParlayMath.price_parlay(legs)
        
        # EV
        ev = (joint_prob * final_odds) - 1
        
        # Kelly
        b = final_odds - 1
        q = 1 - joint_prob
        f = (b * joint_prob - q) / b if b > 0 else 0
        
        # SGP sizing is conservative (divide by leg count to reduce risk?)
        stake = max(0, f * 0.20 * bankroll) 
        
        return ParlayResult(
            legs=legs,
            final_odds=final_odds,
            win_prob=joint_prob,
            ev=ev,
            kelly_stake=stake
        )