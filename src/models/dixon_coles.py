"""
Dixon-Coles Model for Soccer Match Prediction

This model improves upon basic Poisson by introducing a correlation parameter (rho)
that adjusts probabilities for low-scoring outcomes (0-0, 1-0, 0-1, 1-1).

Reference: Dixon, M. J., & Coles, S. G. (1997)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings


@dataclass
class DixonColesParams:
    """Parameters for the Dixon-Coles model."""
    attack: Dict[str, float]  # Team attack strengths
    defense: Dict[str, float]  # Team defense strengths
    home_advantage: float  # Home field advantage
    rho: float  # Low-score correlation parameter


class DixonColes:
    """
    Dixon-Coles model for predicting soccer match outcomes.
    
    The model uses team-specific attack and defense parameters along with
    a correlation factor (rho) that adjusts for the observed dependency
    in low-scoring games.
    """
    
    def __init__(
        self,
        rho: float = -0.13,
        home_advantage: float = 0.25,
        time_decay: float = 0.005
    ):
        """
        Initialize the Dixon-Coles model.
        
        Args:
            rho: Correlation parameter for low-scoring outcomes (typically negative)
            home_advantage: Log-scale home advantage factor
            time_decay: Exponential decay rate for historical matches
        """
        self.rho = rho
        self.home_advantage = home_advantage
        self.time_decay = time_decay
        self.params: Optional[DixonColesParams] = None
        self._teams: List[str] = []
        
    def tau(
        self,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        lambda_away: float,
        rho: float
    ) -> float:
        """
        Calculate the tau correction factor for low-scoring outcomes.
        
        Args:
            home_goals: Number of home team goals
            away_goals: Number of away team goals
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team
            rho: Correlation parameter
            
        Returns:
            Tau correction factor
        """
        if home_goals == 0 and away_goals == 0:
            return 1.0 - lambda_home * lambda_away * rho
        elif home_goals == 0 and away_goals == 1:
            return 1.0 + lambda_home * rho
        elif home_goals == 1 and away_goals == 0:
            return 1.0 + lambda_away * rho
        elif home_goals == 1 and away_goals == 1:
            return 1.0 - rho
        else:
            return 1.0
    
    def match_probability(
        self,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        lambda_away: float,
        rho: Optional[float] = None
    ) -> float:
        """
        Calculate probability of a specific scoreline.
        
        Args:
            home_goals: Number of home team goals
            away_goals: Number of away team goals
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team
            rho: Correlation parameter (uses instance default if None)
            
        Returns:
            Probability of the given scoreline
        """
        if rho is None:
            rho = self.rho
            
        tau_factor = self.tau(home_goals, away_goals, lambda_home, lambda_away, rho)
        
        home_prob = poisson.pmf(home_goals, lambda_home)
        away_prob = poisson.pmf(away_goals, lambda_away)
        
        return tau_factor * home_prob * away_prob
    
    def calculate_expected_goals(
        self,
        home_attack: float,
        home_defense: float,
        away_attack: float,
        away_defense: float,
        home_advantage: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate expected goals for both teams.
        
        Args:
            home_attack: Home team attack strength
            home_defense: Home team defense strength
            away_attack: Away team attack strength
            away_defense: Away team defense strength
            home_advantage: Home advantage factor (uses instance default if None)
            
        Returns:
            Tuple of (expected_home_goals, expected_away_goals)
        """
        if home_advantage is None:
            home_advantage = self.home_advantage
            
        lambda_home = np.exp(home_attack + away_defense + home_advantage)
        lambda_away = np.exp(away_attack + home_defense)
        
        return lambda_home, lambda_away
    
    def score_matrix(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = 10,
        rho: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate probability matrix for all scorelines up to max_goals.
        
        Args:
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team
            max_goals: Maximum number of goals to consider
            rho: Correlation parameter
            
        Returns:
            2D numpy array where [i,j] = P(home=i, away=j)
        """
        if rho is None:
            rho = self.rho
            
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                matrix[home_goals, away_goals] = self.match_probability(
                    home_goals, away_goals, lambda_home, lambda_away, rho
                )
                
        # Normalize to ensure probabilities sum to 1
        matrix /= matrix.sum()
        
        return matrix
    
    def outcome_probabilities(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = 10,
        rho: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate match outcome probabilities (home win, draw, away win).
        
        Args:
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team
            max_goals: Maximum goals to consider
            rho: Correlation parameter
            
        Returns:
            Dictionary with 'home_win', 'draw', 'away_win' probabilities
        """
        matrix = self.score_matrix(lambda_home, lambda_away, max_goals, rho)
        
        home_win = np.sum(np.tril(matrix, k=-1))  # Below diagonal
        draw = np.sum(np.diag(matrix))  # Diagonal
        away_win = np.sum(np.triu(matrix, k=1))  # Above diagonal
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win
        }
    
    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        matches: List[Dict],
        teams: List[str]
    ) -> float:
        """
        Calculate negative log-likelihood for parameter optimization.
        
        Args:
            params: Flattened parameter array
            matches: List of match dictionaries
            teams: List of team names
            
        Returns:
            Negative log-likelihood value
        """
        n_teams = len(teams)
        
        # Unpack parameters
        attack = dict(zip(teams, params[:n_teams]))
        defense = dict(zip(teams, params[n_teams:2*n_teams]))
        home_adv = params[2*n_teams]
        rho = params[2*n_teams + 1]
        
        # Constraint: sum of attack/defense = 0 for identifiability
        attack_sum = sum(attack.values())
        defense_sum = sum(defense.values())
        
        log_likelihood = 0.0
        
        for match in matches:
            home_team = match['home_team']
            away_team = match['away_team']
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            weight = match.get('weight', 1.0)
            
            if home_team not in teams or away_team not in teams:
                continue
                
            lambda_home, lambda_away = self.calculate_expected_goals(
                attack[home_team],
                defense[home_team],
                attack[away_team],
                defense[away_team],
                home_adv
            )
            
            prob = self.match_probability(
                home_goals, away_goals, lambda_home, lambda_away, rho
            )
            
            if prob > 0:
                log_likelihood += weight * np.log(prob)
        
        # Add regularization for constraint
        penalty = 100 * (attack_sum ** 2 + defense_sum ** 2)
        
        return -log_likelihood + penalty
    
    def fit(
        self,
        matches: List[Dict],
        teams: Optional[List[str]] = None,
        apply_time_decay: bool = True
    ) -> 'DixonColes':
        """
        Fit the model to historical match data.
        
        Args:
            matches: List of match dictionaries with keys:
                     'home_team', 'away_team', 'home_goals', 'away_goals',
                     'date' (optional, for time decay)
            teams: List of teams to include (auto-detected if None)
            apply_time_decay: Whether to apply time decay weights
            
        Returns:
            Self, for method chaining
        """
        if teams is None:
            teams = list(set(
                [m['home_team'] for m in matches] + 
                [m['away_team'] for m in matches]
            ))
        
        self._teams = teams
        n_teams = len(teams)
        
        # Apply time decay weights
        if apply_time_decay and matches and 'date' in matches[0]:
            matches = self._apply_time_weights(matches)
        else:
            for m in matches:
                m['weight'] = 1.0
        
        # Initial parameters
        initial_attack = np.zeros(n_teams)
        initial_defense = np.zeros(n_teams)
        initial_home_adv = 0.25
        initial_rho = -0.1
        
        x0 = np.concatenate([
            initial_attack,
            initial_defense,
            [initial_home_adv, initial_rho]
        ])
        
        # Bounds
        bounds = (
            [(-2, 2)] * n_teams +  # Attack
            [(-2, 2)] * n_teams +  # Defense
            [(0, 1)] +              # Home advantage
            [(-0.5, 0.1)]          # Rho
        )
        
        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._negative_log_likelihood,
                x0,
                args=(matches, teams),
                method='L-BFGS-B',
                bounds=bounds
            )
        
        # Store parameters
        self.params = DixonColesParams(
            attack=dict(zip(teams, result.x[:n_teams])),
            defense=dict(zip(teams, result.x[n_teams:2*n_teams])),
            home_advantage=result.x[2*n_teams],
            rho=result.x[2*n_teams + 1]
        )
        
        self.home_advantage = self.params.home_advantage
        self.rho = self.params.rho
        
        return self
    
    def _apply_time_weights(self, matches: List[Dict]) -> List[Dict]:
        """Apply exponential time decay weights to matches."""
        import pandas as pd
        
        dates = pd.to_datetime([m['date'] for m in matches])
        max_date = dates.max()
        days_ago = (max_date - dates).days
        
        for i, m in enumerate(matches):
            m['weight'] = np.exp(-self.time_decay * days_ago[i])
            
        return matches
    
    def predict(
        self,
        home_team: str,
        away_team: str,
        max_goals: int = 10
    ) -> Dict:
        """
        Predict match outcome probabilities.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            max_goals: Maximum goals to consider
            
        Returns:
            Dictionary with prediction results
        """
        if self.params is None:
            raise ValueError("Model must be fit before making predictions")
            
        lambda_home, lambda_away = self.calculate_expected_goals(
            self.params.attack.get(home_team, 0.0),
            self.params.defense.get(home_team, 0.0),
            self.params.attack.get(away_team, 0.0),
            self.params.defense.get(away_team, 0.0)
        )
        
        outcomes = self.outcome_probabilities(lambda_home, lambda_away, max_goals)
        score_matrix = self.score_matrix(lambda_home, lambda_away, max_goals)
        
        return {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'home_win_prob': outcomes['home_win'],
            'draw_prob': outcomes['draw'],
            'away_win_prob': outcomes['away_win'],
            'score_matrix': score_matrix,
            'most_likely_score': np.unravel_index(
                np.argmax(score_matrix), score_matrix.shape
            )
        }
    
    def get_team_ratings(self) -> Dict[str, Dict[str, float]]:
        """Get attack and defense ratings for all teams."""
        if self.params is None:
            raise ValueError("Model must be fit first")
            
        return {
            team: {
                'attack': self.params.attack[team],
                'defense': self.params.defense[team],
                'overall': self.params.attack[team] - self.params.defense[team]
            }
            for team in self._teams
        }
