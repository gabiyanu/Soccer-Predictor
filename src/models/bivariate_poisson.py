"""
Bivariate Poisson Model for Soccer Match Prediction

This model accounts for the correlation between team scores using a
covariance term (lambda_3) in the joint distribution.

Reference: Karlis, D., & Ntzoufras, I. (2003)
"""

import numpy as np
from scipy.special import factorial
from scipy.stats import poisson
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings


@dataclass
class BivariateParams:
    """Parameters for the Bivariate Poisson model."""
    lambda_home: Dict[str, float]  # Team home scoring rates
    lambda_away: Dict[str, float]  # Team away scoring rates
    lambda_covariance: float  # Covariance term (lambda_3)
    home_boost: float  # Home advantage multiplier


class BivariatePoisson:
    """
    Bivariate Poisson distribution model for soccer match prediction.
    
    Models the joint distribution of (X, Y) where X = home goals, Y = away goals
    with correlation through a shared Poisson component:
    
    X = X₁ + X₃
    Y = X₂ + X₃
    
    Where X₁ ~ Poisson(λ₁), X₂ ~ Poisson(λ₂), X₃ ~ Poisson(λ₃)
    """
    
    def __init__(
        self,
        lambda_covariance: float = 0.1,
        home_boost: float = 1.25
    ):
        """
        Initialize the Bivariate Poisson model.
        
        Args:
            lambda_covariance: Covariance parameter (λ₃) - typically small positive
            home_boost: Multiplicative home advantage factor
        """
        self.lambda_covariance = lambda_covariance
        self.home_boost = home_boost
        self.params: Optional[BivariateParams] = None
        
    def bivariate_poisson_pmf(
        self,
        x: int,
        y: int,
        lambda_1: float,
        lambda_2: float,
        lambda_3: float
    ) -> float:
        """
        Calculate the bivariate Poisson probability mass function.
        
        P(X=x, Y=y) = exp(-(λ₁+λ₂+λ₃)) × Σₖ (λ₁^(x-k) × λ₂^(y-k) × λ₃^k) / ((x-k)!(y-k)!k!)
        
        Args:
            x: Home team goals
            y: Away team goals
            lambda_1: Home team independent scoring rate (λ₁)
            lambda_2: Away team independent scoring rate (λ₂)
            lambda_3: Covariance term (λ₃)
            
        Returns:
            Probability of (x, y) scoreline
        """
        if x < 0 or y < 0:
            return 0.0
            
        # Ensure positive lambdas
        lambda_1 = max(lambda_1, 1e-10)
        lambda_2 = max(lambda_2, 1e-10)
        lambda_3 = max(lambda_3, 1e-10)
        
        k_max = min(x, y)
        
        # Calculate using log-space for numerical stability
        log_exp_term = -(lambda_1 + lambda_2 + lambda_3)
        
        total = 0.0
        for k in range(k_max + 1):
            log_term = (
                (x - k) * np.log(lambda_1) +
                (y - k) * np.log(lambda_2) +
                k * np.log(lambda_3) -
                np.log(factorial(x - k, exact=False)) -
                np.log(factorial(y - k, exact=False)) -
                np.log(factorial(k, exact=False))
            )
            total += np.exp(log_term)
            
        return np.exp(log_exp_term) * total
    
    def score_matrix(
        self,
        lambda_1: float,
        lambda_2: float,
        lambda_3: Optional[float] = None,
        max_goals: int = 10
    ) -> np.ndarray:
        """
        Generate probability matrix for all scorelines.
        
        Args:
            lambda_1: Home team independent rate
            lambda_2: Away team independent rate
            lambda_3: Covariance term (uses instance default if None)
            max_goals: Maximum goals to consider
            
        Returns:
            2D array where [i,j] = P(home=i, away=j)
        """
        if lambda_3 is None:
            lambda_3 = self.lambda_covariance
            
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                matrix[x, y] = self.bivariate_poisson_pmf(
                    x, y, lambda_1, lambda_2, lambda_3
                )
                
        # Normalize
        matrix_sum = matrix.sum()
        if matrix_sum > 0:
            matrix /= matrix_sum
            
        return matrix
    
    def correlation_coefficient(
        self,
        lambda_1: float,
        lambda_2: float,
        lambda_3: float
    ) -> float:
        """
        Calculate the correlation coefficient between X and Y.
        
        For bivariate Poisson:
        Corr(X, Y) = λ₃ / sqrt((λ₁ + λ₃)(λ₂ + λ₃))
        
        Args:
            lambda_1, lambda_2, lambda_3: Model parameters
            
        Returns:
            Correlation coefficient
        """
        numerator = lambda_3
        denominator = np.sqrt((lambda_1 + lambda_3) * (lambda_2 + lambda_3))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
    
    def expected_goals(
        self,
        lambda_1: float,
        lambda_2: float,
        lambda_3: float
    ) -> Tuple[float, float]:
        """
        Calculate expected goals for each team.
        
        E[X] = λ₁ + λ₃
        E[Y] = λ₂ + λ₃
        
        Args:
            lambda_1, lambda_2, lambda_3: Model parameters
            
        Returns:
            Tuple of (expected_home, expected_away)
        """
        return lambda_1 + lambda_3, lambda_2 + lambda_3
    
    def outcome_probabilities(
        self,
        lambda_1: float,
        lambda_2: float,
        lambda_3: Optional[float] = None,
        max_goals: int = 10
    ) -> Dict[str, float]:
        """
        Calculate match outcome probabilities.
        
        Args:
            lambda_1: Home team independent rate
            lambda_2: Away team independent rate
            lambda_3: Covariance term
            max_goals: Maximum goals to consider
            
        Returns:
            Dictionary with outcome probabilities
        """
        matrix = self.score_matrix(lambda_1, lambda_2, lambda_3, max_goals)
        
        home_win = np.sum(np.tril(matrix, k=-1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, k=1))
        
        # Both teams to score
        btts = matrix.sum() - matrix[0, :].sum() - matrix[:, 0].sum() + matrix[0, 0]
        
        # Over/under goals
        total_goals_probs = {}
        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
            over = sum(
                matrix[i, j]
                for i in range(max_goals + 1)
                for j in range(max_goals + 1)
                if i + j > threshold
            )
            total_goals_probs[f'over_{threshold}'] = over
            total_goals_probs[f'under_{threshold}'] = 1 - over
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            'btts': btts,
            **total_goals_probs
        }
    
    def estimate_lambdas(
        self,
        home_attack: float,
        home_defense: float,
        away_attack: float,
        away_defense: float,
        league_avg: float = 1.35
    ) -> Tuple[float, float, float]:
        """
        Estimate lambda parameters from team strengths.
        
        Args:
            home_attack: Home team attack strength (multiplicative)
            home_defense: Home team defense strength (multiplicative)
            away_attack: Away team attack strength (multiplicative)
            away_defense: Away team defense strength (multiplicative)
            league_avg: League average goals per team per match
            
        Returns:
            Tuple of (lambda_1, lambda_2, lambda_3)
        """
        # Home team rate = attack × opponent defense weakness × home boost
        lambda_total_home = league_avg * home_attack * (1 / away_defense) * self.home_boost
        
        # Away team rate = attack × opponent defense weakness
        lambda_total_away = league_avg * away_attack * (1 / home_defense)
        
        # Split into independent and shared components
        # The covariance term represents shared factors (weather, referee, match tempo)
        lambda_3 = self.lambda_covariance
        
        lambda_1 = max(lambda_total_home - lambda_3, 0.1)
        lambda_2 = max(lambda_total_away - lambda_3, 0.1)
        
        return lambda_1, lambda_2, lambda_3
    
    def fit(
        self,
        matches: List[Dict],
        method: str = 'mle'
    ) -> 'BivariatePoisson':
        """
        Fit the model to historical match data.
        
        Args:
            matches: List of match dictionaries with:
                     'home_goals', 'away_goals', 'home_team', 'away_team'
            method: Fitting method ('mle' for maximum likelihood)
            
        Returns:
            Self, for method chaining
        """
        teams = list(set(
            [m['home_team'] for m in matches] +
            [m['away_team'] for m in matches]
        ))
        n_teams = len(teams)
        
        def negative_log_likelihood(params):
            """Objective function for MLE."""
            lambda_home = dict(zip(teams, params[:n_teams]))
            lambda_away = dict(zip(teams, params[n_teams:2*n_teams]))
            lambda_3 = params[2*n_teams]
            home_boost = params[2*n_teams + 1]
            
            log_lik = 0.0
            for match in matches:
                ht, at = match['home_team'], match['away_team']
                hg, ag = match['home_goals'], match['away_goals']
                
                l1 = lambda_home.get(ht, 1.0) * home_boost / lambda_away.get(at, 1.0)
                l2 = lambda_home.get(at, 1.0) / lambda_away.get(ht, 1.0)
                
                prob = self.bivariate_poisson_pmf(hg, ag, l1, l2, lambda_3)
                if prob > 1e-10:
                    log_lik += np.log(prob)
                    
            return -log_lik
        
        # Initial parameters
        x0 = np.concatenate([
            np.ones(n_teams) * 1.3,  # Home lambdas
            np.ones(n_teams) * 1.0,  # Away lambdas
            [0.1, 1.2]  # lambda_3, home_boost
        ])
        
        bounds = (
            [(0.3, 3.0)] * n_teams +  # Home
            [(0.3, 3.0)] * n_teams +  # Away
            [(0.01, 0.5), (1.0, 1.5)]  # Covariance, home boost
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                negative_log_likelihood,
                x0,
                method='L-BFGS-B',
                bounds=bounds
            )
        
        self.params = BivariateParams(
            lambda_home=dict(zip(teams, result.x[:n_teams])),
            lambda_away=dict(zip(teams, result.x[n_teams:2*n_teams])),
            lambda_covariance=result.x[2*n_teams],
            home_boost=result.x[2*n_teams + 1]
        )
        
        self.lambda_covariance = self.params.lambda_covariance
        self.home_boost = self.params.home_boost
        
        return self
    
    def predict(
        self,
        home_team: str,
        away_team: str,
        home_attack: Optional[float] = None,
        away_attack: Optional[float] = None,
        home_defense: Optional[float] = None,
        away_defense: Optional[float] = None,
        max_goals: int = 10
    ) -> Dict:
        """
        Predict match outcome.
        
        Can use either fitted parameters or provided team strengths.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_attack, away_attack: Attack strengths (optional)
            home_defense, away_defense: Defense strengths (optional)
            max_goals: Maximum goals to consider
            
        Returns:
            Dictionary with predictions
        """
        if self.params is not None:
            # Use fitted parameters
            lambda_1 = (
                self.params.lambda_home.get(home_team, 1.3) *
                self.params.home_boost /
                self.params.lambda_away.get(away_team, 1.0)
            )
            lambda_2 = (
                self.params.lambda_home.get(away_team, 1.3) /
                self.params.lambda_away.get(home_team, 1.0)
            )
            lambda_3 = self.params.lambda_covariance
        elif all(v is not None for v in [home_attack, away_attack, home_defense, away_defense]):
            # Use provided strengths
            lambda_1, lambda_2, lambda_3 = self.estimate_lambdas(
                home_attack, home_defense,
                away_attack, away_defense
            )
        else:
            raise ValueError("Either fit model or provide team strengths")
        
        outcomes = self.outcome_probabilities(lambda_1, lambda_2, lambda_3, max_goals)
        matrix = self.score_matrix(lambda_1, lambda_2, lambda_3, max_goals)
        exp_home, exp_away = self.expected_goals(lambda_1, lambda_2, lambda_3)
        
        return {
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'expected_home_goals': exp_home,
            'expected_away_goals': exp_away,
            'correlation': self.correlation_coefficient(lambda_1, lambda_2, lambda_3),
            'home_win_prob': outcomes['home_win'],
            'draw_prob': outcomes['draw'],
            'away_win_prob': outcomes['away_win'],
            'btts_prob': outcomes['btts'],
            'over_2_5_prob': outcomes.get('over_2.5', outcomes.get('over_2_5', 0)),
            'score_matrix': matrix,
            'most_likely_score': np.unravel_index(np.argmax(matrix), matrix.shape)
        }
    
    def simulate_match(
        self,
        lambda_1: float,
        lambda_2: float,
        lambda_3: float
    ) -> Tuple[int, int]:
        """
        Simulate a single match outcome.
        
        Args:
            lambda_1, lambda_2, lambda_3: Model parameters
            
        Returns:
            Tuple of (home_goals, away_goals)
        """
        # Generate independent Poisson components
        x1 = np.random.poisson(lambda_1)
        x2 = np.random.poisson(lambda_2)
        x3 = np.random.poisson(lambda_3)
        
        # Combine for final scores
        home_goals = x1 + x3
        away_goals = x2 + x3
        
        return home_goals, away_goals
