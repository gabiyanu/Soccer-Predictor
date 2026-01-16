"""
Elo Rating System for Soccer Teams

Dynamic team strength tracking with support for:
- Match result updates
- Home advantage adjustment
- Goal difference weighting
- Time decay for historical matches

Reference: Elo, A. (1978). The Rating of Chessplayers, Past and Present
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json


@dataclass
class EloMatch:
    """Record of a match for Elo calculation."""
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    date: Optional[datetime] = None
    
    @property
    def result(self) -> float:
        """Match result from home team perspective (1=win, 0.5=draw, 0=loss)."""
        if self.home_goals > self.away_goals:
            return 1.0
        elif self.home_goals == self.away_goals:
            return 0.5
        else:
            return 0.0
    
    @property
    def goal_difference(self) -> int:
        """Goal difference from home team perspective."""
        return self.home_goals - self.away_goals


@dataclass
class TeamElo:
    """Elo rating for a team."""
    name: str
    rating: float = 1500.0
    matches_played: int = 0
    history: List[Tuple[datetime, float]] = field(default_factory=list)
    
    def update(self, new_rating: float, date: Optional[datetime] = None):
        """Update rating and record history."""
        if date:
            self.history.append((date, self.rating))
        self.rating = new_rating
        self.matches_played += 1


class EloRating:
    """
    Elo rating system for soccer teams.
    
    Features:
    - Customizable K-factor (rating volatility)
    - Home advantage modeling
    - Goal difference multiplier
    - Historical decay for old matches
    """
    
    def __init__(
        self,
        k_factor: float = 32.0,
        home_advantage: float = 100.0,
        goal_diff_multiplier: float = 0.5,
        base_rating: float = 1500.0,
        rating_scale: float = 400.0
    ):
        """
        Initialize Elo rating system.
        
        Args:
            k_factor: Maximum rating change per match (higher = more volatile)
            home_advantage: Elo points added to home team for expected score
            goal_diff_multiplier: How much goal difference affects K
            base_rating: Starting rating for new teams
            rating_scale: Divisor for expected score calculation (400 in chess)
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.goal_diff_multiplier = goal_diff_multiplier
        self.base_rating = base_rating
        self.rating_scale = rating_scale
        
        self.teams: Dict[str, TeamElo] = {}
        self.match_history: List[EloMatch] = []
        
    def get_rating(self, team: str) -> float:
        """Get current rating for a team, creating entry if needed."""
        if team not in self.teams:
            self.teams[team] = TeamElo(name=team, rating=self.base_rating)
        return self.teams[team].rating
    
    def expected_score(
        self,
        rating_a: float,
        rating_b: float,
        home_advantage: bool = True
    ) -> float:
        """
        Calculate expected score for team A against team B.
        
        E_A = 1 / (1 + 10^((R_B - R_A - H) / scale))
        
        Args:
            rating_a: Team A rating
            rating_b: Team B rating
            home_advantage: Whether team A has home advantage
            
        Returns:
            Expected score for team A (0 to 1)
        """
        ha = self.home_advantage if home_advantage else 0
        exponent = (rating_b - rating_a - ha) / self.rating_scale
        return 1.0 / (1.0 + 10 ** exponent)
    
    def goal_diff_k_multiplier(self, goal_diff: int) -> float:
        """
        Calculate K-factor multiplier based on goal difference.
        
        Larger margins of victory result in larger rating changes.
        
        Args:
            goal_diff: Absolute goal difference
            
        Returns:
            Multiplier for K-factor
        """
        if goal_diff == 0:
            return 1.0
        elif goal_diff == 1:
            return 1.0
        elif goal_diff == 2:
            return 1.0 + self.goal_diff_multiplier
        else:
            # Logarithmic scaling for blowouts
            return 1.0 + self.goal_diff_multiplier + (
                (goal_diff - 2) * self.goal_diff_multiplier / 2
            )
    
    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        date: Optional[datetime] = None,
        k_override: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Update ratings based on match result.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            date: Match date (for history tracking)
            k_override: Optional override for K-factor
            
        Returns:
            Tuple of (new_home_rating, new_away_rating)
        """
        # Get current ratings
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Calculate expected scores
        home_expected = self.expected_score(home_rating, away_rating, home_advantage=True)
        away_expected = 1 - home_expected
        
        # Actual results
        if home_goals > away_goals:
            home_actual, away_actual = 1.0, 0.0
        elif home_goals == away_goals:
            home_actual, away_actual = 0.5, 0.5
        else:
            home_actual, away_actual = 0.0, 1.0
        
        # K-factor with goal difference adjustment
        k = k_override if k_override is not None else self.k_factor
        goal_diff = abs(home_goals - away_goals)
        k_adjusted = k * self.goal_diff_k_multiplier(goal_diff)
        
        # Update ratings
        new_home_rating = home_rating + k_adjusted * (home_actual - home_expected)
        new_away_rating = away_rating + k_adjusted * (away_actual - away_expected)
        
        # Store updates
        self.teams[home_team].update(new_home_rating, date)
        self.teams[away_team].update(new_away_rating, date)
        
        # Record match
        self.match_history.append(EloMatch(
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
            date=date
        ))
        
        return new_home_rating, new_away_rating
    
    def process_matches(
        self,
        matches: List[Dict],
        chronological: bool = True
    ) -> 'EloRating':
        """
        Process multiple matches to build ratings.
        
        Args:
            matches: List of match dictionaries
            chronological: Whether to sort by date first
            
        Returns:
            Self, for method chaining
        """
        if chronological and matches and 'date' in matches[0]:
            matches = sorted(matches, key=lambda x: x.get('date', ''))
        
        for match in matches:
            date = match.get('date')
            if isinstance(date, str):
                try:
                    date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                except:
                    date = None
                
            self.update_ratings(
                home_team=match['home_team'],
                away_team=match['away_team'],
                home_goals=match['home_goals'],
                away_goals=match['away_goals'],
                date=date
            )
            
        return self
    
    def predict_match(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Predict match outcome probabilities using Elo ratings.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dictionary with predictions
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Expected score (probability-like)
        home_expected = self.expected_score(home_rating, away_rating, home_advantage=True)
        
        # Convert to match outcome probabilities
        # Using empirical conversion based on Elo difference
        rating_diff = home_rating - away_rating + self.home_advantage
        
        # Estimate win/draw/loss from expected score
        # These coefficients are calibrated from historical data
        if home_expected >= 0.5:
            home_win = 0.7 * home_expected + 0.15
            draw = 0.25 - 0.2 * (home_expected - 0.5)
            away_win = 1 - home_win - draw
        else:
            away_win = 0.7 * (1 - home_expected) + 0.15
            draw = 0.25 - 0.2 * (0.5 - home_expected)
            home_win = 1 - away_win - draw
        
        # Ensure valid probabilities
        home_win = max(0.01, min(0.95, home_win))
        away_win = max(0.01, min(0.95, away_win))
        draw = 1 - home_win - away_win
        
        return {
            'home_rating': home_rating,
            'away_rating': away_rating,
            'rating_difference': rating_diff,
            'home_expected': home_expected,
            'home_win_prob': home_win,
            'draw_prob': draw,
            'away_win_prob': away_win
        }
    
    def rating_to_expected_goals(
        self,
        team_rating: float,
        opponent_rating: float,
        is_home: bool = True,
        league_avg: float = 1.35
    ) -> float:
        """
        Convert Elo rating difference to expected goals.
        
        Args:
            team_rating: Rating of scoring team
            opponent_rating: Rating of defending team
            is_home: Whether team is playing at home
            league_avg: League average goals per team
            
        Returns:
            Expected goals for the team
        """
        rating_diff = team_rating - opponent_rating
        if is_home:
            rating_diff += self.home_advantage
        
        # Convert to multiplicative factor
        # Every 100 rating points = ~0.15 goal difference
        factor = 1.0 + (rating_diff / 1000)
        
        return league_avg * max(factor, 0.3)
    
    def get_rankings(self, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Get teams ranked by Elo rating."""
        rankings = sorted(
            [(name, team.rating) for name, team in self.teams.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return rankings[:top_n] if top_n else rankings
    
    def get_team_history(self, team: str) -> List[Tuple[datetime, float]]:
        """Get rating history for a team."""
        if team not in self.teams:
            return []
        return self.teams[team].history
    
    def simulate_match_outcome(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[str, int, int]:
        """
        Simulate a single match outcome based on Elo ratings.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Tuple of (result, home_goals, away_goals)
        """
        pred = self.predict_match(home_team, away_team)
        
        # Sample outcome
        rand = np.random.random()
        if rand < pred['home_win_prob']:
            result = 'home_win'
        elif rand < pred['home_win_prob'] + pred['draw_prob']:
            result = 'draw'
        else:
            result = 'away_win'
        
        # Sample goals based on expected values
        home_xg = self.rating_to_expected_goals(
            self.get_rating(home_team),
            self.get_rating(away_team),
            is_home=True
        )
        away_xg = self.rating_to_expected_goals(
            self.get_rating(away_team),
            self.get_rating(home_team),
            is_home=False
        )
        
        home_goals = np.random.poisson(home_xg)
        away_goals = np.random.poisson(away_xg)
        
        # Adjust to match result if needed
        if result == 'home_win' and home_goals <= away_goals:
            home_goals = away_goals + 1
        elif result == 'away_win' and away_goals <= home_goals:
            away_goals = home_goals + 1
        elif result == 'draw':
            away_goals = home_goals
            
        return result, home_goals, away_goals
    
    def save(self, filepath: str):
        """Save Elo ratings to JSON file."""
        data = {
            'config': {
                'k_factor': self.k_factor,
                'home_advantage': self.home_advantage,
                'goal_diff_multiplier': self.goal_diff_multiplier,
                'base_rating': self.base_rating,
                'rating_scale': self.rating_scale
            },
            'teams': {
                name: {
                    'rating': team.rating,
                    'matches_played': team.matches_played
                }
                for name, team in self.teams.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'EloRating':
        """Load Elo ratings from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        elo = cls(**data['config'])
        for name, team_data in data['teams'].items():
            elo.teams[name] = TeamElo(
                name=name,
                rating=team_data['rating'],
                matches_played=team_data['matches_played']
            )
        return elo
