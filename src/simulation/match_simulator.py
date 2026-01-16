"""
Match Simulator

Orchestrates all prediction models to produce comprehensive match predictions.
Combines Dixon-Coles, Bivariate Poisson, Elo ratings, and player-level factors.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from ..models import (
    DixonColes, 
    BivariatePoisson, 
    EloRating,
    PlayerModel,
    Squad,
    Player,
    Position
)
from .monte_carlo import MonteCarloEngine, SimulationResult


@dataclass
class SimulationConfig:
    """Configuration for match simulation."""
    n_simulations: int = 10000
    
    # Home advantage
    home_advantage: float = 0.25  # Log-scale for Dixon-Coles
    home_advantage_elo: float = 100  # Elo points
    home_boost_xg: float = 1.25  # Multiplicative for xG
    
    # Dixon-Coles parameters
    dixon_coles_rho: float = -0.13  # Low-score correlation
    
    # Bivariate Poisson
    bivariate_lambda3: float = 0.1  # Covariance term
    
    # Elo settings
    elo_k_factor: float = 32.0
    time_decay_factor: float = 0.005
    
    # Player model
    player_impact_weight: float = 0.3  # 0-1, how much player factors affect
    key_player_impact: float = 0.15
    
    # Model weighting for ensemble
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        'dixon_coles': 0.35,
        'bivariate_poisson': 0.35,
        'elo': 0.20,
        'player_model': 0.10
    })
    
    # League average for xG baseline
    league_avg_goals: float = 1.35


@dataclass
class TeamData:
    """Team information for match simulation."""
    name: str
    elo: float = 1500.0
    attack_strength: float = 1.0  # Multiplicative, 1.0 = average
    defense_strength: float = 1.0
    squad: Optional[Squad] = None
    recent_form: Optional[List[str]] = None  # List of 'W', 'D', 'L'
    
    @classmethod
    def from_dict(cls, data: Dict, player_model: Optional[PlayerModel] = None) -> 'TeamData':
        """Create TeamData from dictionary."""
        squad = None
        if 'players' in data and player_model:
            squad = player_model.create_squad_from_dict(data)
            
        return cls(
            name=data.get('name', 'Unknown'),
            elo=data.get('elo', 1500.0),
            attack_strength=data.get('attack_strength', 1.0),
            defense_strength=data.get('defense_strength', 1.0),
            squad=squad,
            recent_form=data.get('recent_form')
        )


class MatchSimulator:
    """
    Main match simulation class combining all models.
    
    Uses an ensemble approach to combine predictions from:
    - Dixon-Coles (Poisson with low-score adjustment)
    - Bivariate Poisson (correlated scores)
    - Elo ratings (team strength)
    - Player-level factors (injuries, lineups)
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        dixon_coles: Optional[DixonColes] = None,
        bivariate_poisson: Optional[BivariatePoisson] = None,
        elo_system: Optional[EloRating] = None,
        player_model: Optional[PlayerModel] = None
    ):
        """
        Initialize match simulator.
        
        Args:
            config: Simulation configuration
            dixon_coles: Pre-fitted Dixon-Coles model
            bivariate_poisson: Pre-fitted Bivariate Poisson model
            elo_system: Pre-fitted Elo rating system
            player_model: Player model for lineup analysis
        """
        self.config = config or SimulationConfig()
        
        # Initialize models
        self.dixon_coles = dixon_coles or DixonColes(
            rho=self.config.dixon_coles_rho,
            home_advantage=self.config.home_advantage,
            time_decay=self.config.time_decay_factor
        )
        
        self.bivariate_poisson = bivariate_poisson or BivariatePoisson(
            lambda_covariance=self.config.bivariate_lambda3,
            home_boost=self.config.home_boost_xg
        )
        
        self.elo_system = elo_system or EloRating(
            k_factor=self.config.elo_k_factor,
            home_advantage=self.config.home_advantage_elo
        )
        
        self.player_model = player_model or PlayerModel(
            key_player_impact=self.config.key_player_impact
        )
        
        # Monte Carlo engine
        self.mc_engine = MonteCarloEngine(
            n_simulations=self.config.n_simulations
        )
    
    def _calculate_base_xg(
        self,
        home_team: TeamData,
        away_team: TeamData
    ) -> Tuple[float, float]:
        """
        Calculate base expected goals from team strengths.
        
        Args:
            home_team: Home team data
            away_team: Away team data
            
        Returns:
            Tuple of (home_xg, away_xg)
        """
        league_avg = self.config.league_avg_goals
        
        # Home team xG
        home_xg = (
            league_avg *
            home_team.attack_strength *
            (1 / away_team.defense_strength) *
            self.config.home_boost_xg
        )
        
        # Away team xG
        away_xg = (
            league_avg *
            away_team.attack_strength *
            (1 / home_team.defense_strength)
        )
        
        return home_xg, away_xg
    
    def _apply_elo_adjustment(
        self,
        base_home_xg: float,
        base_away_xg: float,
        home_team: TeamData,
        away_team: TeamData
    ) -> Tuple[float, float]:
        """
        Adjust xG based on Elo rating difference.
        """
        elo_diff = home_team.elo - away_team.elo + self.config.home_advantage_elo
        
        # Convert Elo diff to xG multiplier
        # ~100 Elo points = ~0.1 xG difference
        adjustment_factor = 1.0 + (elo_diff / 1000)
        
        home_xg = base_home_xg * max(0.5, adjustment_factor)
        away_xg = base_away_xg * max(0.5, 2 - adjustment_factor)
        
        return home_xg, away_xg
    
    def _apply_player_adjustment(
        self,
        home_xg: float,
        away_xg: float,
        home_team: TeamData,
        away_team: TeamData
    ) -> Tuple[float, float]:
        """
        Adjust xG based on player-level factors.
        """
        weight = self.config.player_impact_weight
        
        if home_team.squad:
            home_impact = self.player_model.injury_impact(home_team.squad)
            home_xg = home_xg * (
                1 - weight + weight * home_impact['attack_factor']
            )
            away_xg = away_xg * (
                1 - weight + weight * (1 / home_impact['defense_factor'])
            )
        
        if away_team.squad:
            away_impact = self.player_model.injury_impact(away_team.squad)
            away_xg = away_xg * (
                1 - weight + weight * away_impact['attack_factor']
            )
            home_xg = home_xg * (
                1 - weight + weight * (1 / away_impact['defense_factor'])
            )
        
        return home_xg, away_xg
    
    def _apply_form_adjustment(
        self,
        home_xg: float,
        away_xg: float,
        home_team: TeamData,
        away_team: TeamData
    ) -> Tuple[float, float]:
        """
        Adjust xG based on recent form.
        """
        def form_factor(form: Optional[List[str]]) -> float:
            if not form:
                return 1.0
            
            points = {'W': 3, 'D': 1, 'L': 0}
            total = sum(points.get(r, 1) for r in form[-5:])
            expected = 5 * 1.5  # 1.5 points per game average
            
            return 0.9 + 0.2 * (total / expected)
        
        home_factor = form_factor(home_team.recent_form)
        away_factor = form_factor(away_team.recent_form)
        
        return home_xg * home_factor, away_xg * away_factor
    
    def calculate_match_xg(
        self,
        home_team: TeamData,
        away_team: TeamData
    ) -> Tuple[float, float]:
        """
        Calculate final expected goals combining all factors.
        
        Args:
            home_team: Home team data
            away_team: Away team data
            
        Returns:
            Tuple of (home_xg, away_xg)
        """
        # Base calculation
        home_xg, away_xg = self._calculate_base_xg(home_team, away_team)
        
        # Apply adjustments
        home_xg, away_xg = self._apply_elo_adjustment(
            home_xg, away_xg, home_team, away_team
        )
        home_xg, away_xg = self._apply_player_adjustment(
            home_xg, away_xg, home_team, away_team
        )
        home_xg, away_xg = self._apply_form_adjustment(
            home_xg, away_xg, home_team, away_team
        )
        
        # Ensure reasonable bounds
        home_xg = np.clip(home_xg, 0.3, 4.0)
        away_xg = np.clip(away_xg, 0.2, 3.5)
        
        return home_xg, away_xg
    
    def _simulate_single_match_dixon_coles(
        self,
        home_xg: float,
        away_xg: float
    ) -> Dict[str, int]:
        """Simulate match using Dixon-Coles adjusted Poisson."""
        # Sample from Dixon-Coles adjusted distribution
        max_goals = 10
        matrix = self.dixon_coles.score_matrix(home_xg, away_xg, max_goals)
        
        # Flatten and sample
        flat_probs = matrix.flatten()
        idx = np.random.choice(len(flat_probs), p=flat_probs)
        home_goals, away_goals = np.unravel_index(idx, matrix.shape)
        
        return {'home_goals': int(home_goals), 'away_goals': int(away_goals)}
    
    def _simulate_single_match_bivariate(
        self,
        home_xg: float,
        away_xg: float
    ) -> Dict[str, int]:
        """Simulate match using Bivariate Poisson."""
        lambda_3 = self.config.bivariate_lambda3
        lambda_1 = max(0.1, home_xg - lambda_3)
        lambda_2 = max(0.1, away_xg - lambda_3)
        
        home_goals, away_goals = self.bivariate_poisson.simulate_match(
            lambda_1, lambda_2, lambda_3
        )
        
        return {'home_goals': int(home_goals), 'away_goals': int(away_goals)}
    
    def _simulate_single_match(
        self,
        home_xg: float,
        away_xg: float,
        method: str = 'ensemble'
    ) -> Dict[str, int]:
        """
        Simulate a single match outcome.
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            method: 'dixon_coles', 'bivariate', or 'ensemble'
            
        Returns:
            Dictionary with home_goals and away_goals
        """
        if method == 'dixon_coles':
            return self._simulate_single_match_dixon_coles(home_xg, away_xg)
        elif method == 'bivariate':
            return self._simulate_single_match_bivariate(home_xg, away_xg)
        else:
            # Ensemble: randomly choose method based on weights
            weights = self.config.model_weights
            dc_weight = weights.get('dixon_coles', 0.5)
            
            if np.random.random() < dc_weight:
                return self._simulate_single_match_dixon_coles(home_xg, away_xg)
            else:
                return self._simulate_single_match_bivariate(home_xg, away_xg)
    
    def simulate_match(
        self,
        home_team: TeamData,
        away_team: TeamData,
        n_simulations: Optional[int] = None,
        method: str = 'ensemble'
    ) -> SimulationResult:
        """
        Run full Monte Carlo simulation for a match.
        
        Args:
            home_team: Home team data
            away_team: Away team data
            n_simulations: Override for number of simulations
            method: Simulation method
            
        Returns:
            SimulationResult with full prediction analysis
        """
        n = n_simulations or self.config.n_simulations
        
        # Calculate expected goals
        home_xg, away_xg = self.calculate_match_xg(home_team, away_team)
        
        # Create simulation function
        def simulate():
            return self._simulate_single_match(home_xg, away_xg, method)
        
        # Run Monte Carlo simulation
        self.mc_engine.n_simulations = n
        result = self.mc_engine.run_simulation(simulate)
        
        # Add metadata
        result.expected_home_goals = home_xg
        result.expected_away_goals = away_xg
        
        return result
    
    def quick_predict(
        self,
        home_team: TeamData,
        away_team: TeamData
    ) -> Dict[str, Any]:
        """
        Quick prediction without full Monte Carlo (analytical).
        
        Uses Dixon-Coles directly for faster results.
        
        Args:
            home_team: Home team data
            away_team: Away team data
            
        Returns:
            Dictionary with prediction results
        """
        home_xg, away_xg = self.calculate_match_xg(home_team, away_team)
        
        outcomes = self.dixon_coles.outcome_probabilities(home_xg, away_xg)
        score_matrix = self.dixon_coles.score_matrix(home_xg, away_xg)
        most_likely = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        
        # Market probabilities
        btts = 1 - score_matrix[0, :].sum() - score_matrix[:, 0].sum() + score_matrix[0, 0]
        over_2_5 = sum(
            score_matrix[i, j]
            for i in range(11)
            for j in range(11)
            if i + j > 2.5
        )
        
        return {
            'home_team': home_team.name,
            'away_team': away_team.name,
            'expected_home_goals': home_xg,
            'expected_away_goals': away_xg,
            'home_win_prob': outcomes['home_win'],
            'draw_prob': outcomes['draw'],
            'away_win_prob': outcomes['away_win'],
            'most_likely_score': f"{most_likely[0]}-{most_likely[1]}",
            'btts_prob': btts,
            'over_2_5_prob': over_2_5
        }
    
    def simulate_multiple_matches(
        self,
        matches: List[Tuple[TeamData, TeamData]],
        n_simulations_per_match: int = 5000
    ) -> List[SimulationResult]:
        """
        Simulate multiple matches.
        
        Args:
            matches: List of (home_team, away_team) tuples
            n_simulations_per_match: Simulations per match
            
        Returns:
            List of SimulationResult objects
        """
        results = []
        for home_team, away_team in matches:
            result = self.simulate_match(
                home_team, away_team,
                n_simulations=n_simulations_per_match
            )
            results.append(result)
        return results
    
    def fit_from_historical(
        self,
        matches: List[Dict]
    ) -> 'MatchSimulator':
        """
        Fit all models from historical match data.
        
        Args:
            matches: List of historical match dictionaries
            
        Returns:
            Self, for method chaining
        """
        # Fit Dixon-Coles
        self.dixon_coles.fit(matches)
        
        # Fit Bivariate Poisson
        self.bivariate_poisson.fit(matches)
        
        # Build Elo ratings
        self.elo_system.process_matches(matches)
        
        return self
