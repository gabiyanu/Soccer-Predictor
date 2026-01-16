"""
Monte Carlo Match Predictor

Combines Dixon-Coles model with StatsBomb data for accurate match predictions.
"""

import numpy as np
from scipy.stats import poisson
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from .statsbomb_loader import TeamStats


@dataclass
class PredictionResult:
    """Container for match prediction results."""
    home_team: str
    away_team: str
    
    # Expected goals
    home_xg: float
    away_xg: float
    
    # Outcome probabilities
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    
    # Confidence intervals (95%)
    home_win_ci: Tuple[float, float] = (0.0, 0.0)
    draw_ci: Tuple[float, float] = (0.0, 0.0)
    away_win_ci: Tuple[float, float] = (0.0, 0.0)
    
    # Score distribution
    score_distribution: Dict[str, float] = field(default_factory=dict)
    most_likely_score: str = "0-0"
    
    # Market probabilities
    btts_prob: float = 0.0
    over_1_5_prob: float = 0.0
    over_2_5_prob: float = 0.0
    over_3_5_prob: float = 0.0
    clean_sheet_home: float = 0.0
    clean_sheet_away: float = 0.0
    
    # Model info
    n_simulations: int = 10000
    model_weights: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'match': f"{self.home_team} vs {self.away_team}",
            'expected_goals': {
                'home': round(self.home_xg, 2),
                'away': round(self.away_xg, 2)
            },
            'outcome_probabilities': {
                'home_win': round(self.home_win_prob * 100, 1),
                'draw': round(self.draw_prob * 100, 1),
                'away_win': round(self.away_win_prob * 100, 1)
            },
            'predicted_winner': self._get_winner(),
            'most_likely_score': self.most_likely_score,
            'markets': {
                'btts': round(self.btts_prob * 100, 1),
                'over_1_5': round(self.over_1_5_prob * 100, 1),
                'over_2_5': round(self.over_2_5_prob * 100, 1),
                'over_3_5': round(self.over_3_5_prob * 100, 1),
            },
            'top_scores': dict(list(self.score_distribution.items())[:5])
        }
    
    def _get_winner(self) -> str:
        """Get predicted winner."""
        if self.home_win_prob > self.draw_prob and self.home_win_prob > self.away_win_prob:
            return self.home_team
        elif self.away_win_prob > self.draw_prob:
            return self.away_team
        return "Draw"
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"\n{'='*60}",
            f"  {self.home_team} vs {self.away_team}",
            f"{'='*60}",
            f"\n  Expected Goals: {self.home_xg:.2f} - {self.away_xg:.2f}",
            f"\n  Outcome Probabilities:",
            f"    Home Win:  {self.home_win_prob*100:5.1f}%  {'█' * int(self.home_win_prob*20)}",
            f"    Draw:      {self.draw_prob*100:5.1f}%  {'█' * int(self.draw_prob*20)}",
            f"    Away Win:  {self.away_win_prob*100:5.1f}%  {'█' * int(self.away_win_prob*20)}",
            f"\n  Prediction: {self._get_winner()} ({max(self.home_win_prob, self.draw_prob, self.away_win_prob)*100:.1f}% confidence)",
            f"  Most Likely Score: {self.most_likely_score}",
            f"\n  Markets:",
            f"    BTTS:      {self.btts_prob*100:5.1f}%",
            f"    Over 1.5:  {self.over_1_5_prob*100:5.1f}%",
            f"    Over 2.5:  {self.over_2_5_prob*100:5.1f}%",
            f"    Over 3.5:  {self.over_3_5_prob*100:5.1f}%",
            f"\n{'='*60}\n"
        ]
        return '\n'.join(lines)


class DixonColesModel:
    """
    Dixon-Coles model for soccer match prediction.
    
    Adjusts standard Poisson for the observed dependency in low-scoring games
    using a tau correction factor.
    """
    
    def __init__(self, rho: float = -0.13):
        """
        Initialize Dixon-Coles model.
        
        Args:
            rho: Correlation parameter (typically negative, around -0.1 to -0.2)
        """
        self.rho = rho
    
    def tau(
        self,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        lambda_away: float
    ) -> float:
        """
        Calculate tau correction factor for low-scoring outcomes.
        
        The tau function adjusts probabilities for:
        - 0-0 games (often more common than Poisson predicts)
        - 1-0, 0-1 games
        - 1-1 games
        """
        if home_goals == 0 and away_goals == 0:
            return 1.0 - lambda_home * lambda_away * self.rho
        elif home_goals == 0 and away_goals == 1:
            return 1.0 + lambda_home * self.rho
        elif home_goals == 1 and away_goals == 0:
            return 1.0 + lambda_away * self.rho
        elif home_goals == 1 and away_goals == 1:
            return 1.0 - self.rho
        else:
            return 1.0
    
    def score_probability(
        self,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        lambda_away: float
    ) -> float:
        """Calculate probability of a specific scoreline."""
        tau_factor = self.tau(home_goals, away_goals, lambda_home, lambda_away)
        home_prob = poisson.pmf(home_goals, lambda_home)
        away_prob = poisson.pmf(away_goals, lambda_away)
        return tau_factor * home_prob * away_prob
    
    def score_matrix(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = 10
    ) -> np.ndarray:
        """Generate probability matrix for all scorelines."""
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                matrix[h, a] = self.score_probability(h, a, lambda_home, lambda_away)
        
        # Normalize
        matrix /= matrix.sum()
        return matrix


class MonteCarloPredictor:
    """
    Monte Carlo simulation engine for match prediction.
    
    Combines multiple factors:
    - Base team strengths (attack/defense)
    - xG-based strengths
    - Elo ratings
    - Recent form
    - Home advantage
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        home_advantage: float = 0.25,
        rho: float = -0.13,
        use_xg: bool = True,
        xg_weight: float = 0.6,
        random_seed: Optional[int] = None
    ):
        """
        Initialize predictor.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            home_advantage: Home advantage factor (log-scale)
            rho: Dixon-Coles correlation parameter
            use_xg: Whether to use xG-based strengths
            xg_weight: Weight for xG vs goals-based strength (0-1)
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.home_advantage = home_advantage
        self.use_xg = use_xg
        self.xg_weight = xg_weight
        
        self.dixon_coles = DixonColesModel(rho=rho)
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def calculate_expected_goals(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        league_avg: float = 1.35
    ) -> Tuple[float, float]:
        """
        Calculate expected goals for both teams.
        
        Combines multiple factors:
        - Team attack/defense strengths
        - xG-based strengths (if available)
        - Elo rating difference
        - Recent form
        - Home advantage
        """
        # Get attack/defense strengths
        if self.use_xg and home_stats.xg_for > 0:
            home_attack = (
                self.xg_weight * home_stats.xg_attack_strength +
                (1 - self.xg_weight) * home_stats.attack_strength
            )
            home_defense = (
                self.xg_weight * home_stats.xg_defense_strength +
                (1 - self.xg_weight) * home_stats.defense_strength
            )
            away_attack = (
                self.xg_weight * away_stats.xg_attack_strength +
                (1 - self.xg_weight) * away_stats.attack_strength
            )
            away_defense = (
                self.xg_weight * away_stats.xg_defense_strength +
                (1 - self.xg_weight) * away_stats.defense_strength
            )
        else:
            home_attack = home_stats.attack_strength
            home_defense = home_stats.defense_strength
            away_attack = away_stats.attack_strength
            away_defense = away_stats.defense_strength
        
        # Base xG calculation
        home_xg = league_avg * home_attack * (1 / away_defense)
        away_xg = league_avg * away_attack * (1 / home_defense)
        
        # Elo adjustment
        elo_diff = home_stats.elo_rating - away_stats.elo_rating
        elo_factor = 1 + (elo_diff / 800)  # ~100 Elo = ~12.5% adjustment
        home_xg *= max(0.7, min(1.5, elo_factor))
        away_xg *= max(0.7, min(1.5, 2 - elo_factor))
        
        # Form adjustment
        home_form_factor = self._calculate_form_factor(home_stats.recent_form)
        away_form_factor = self._calculate_form_factor(away_stats.recent_form)
        home_xg *= home_form_factor
        away_xg *= away_form_factor
        
        # Home advantage
        home_xg *= np.exp(self.home_advantage)
        
        # Bounds
        home_xg = np.clip(home_xg, 0.4, 4.5)
        away_xg = np.clip(away_xg, 0.3, 3.5)
        
        return home_xg, away_xg
    
    def _calculate_form_factor(self, form: List[str], n_matches: int = 5) -> float:
        """
        Calculate form factor from recent results.
        
        Uses exponentially weighted average where recent matches matter more.
        """
        if not form:
            return 1.0
        
        recent = form[-n_matches:]
        weights = [0.3, 0.25, 0.2, 0.15, 0.1][:len(recent)]
        weights = weights[::-1]  # Most recent = highest weight
        
        points_map = {'W': 3, 'D': 1, 'L': 0}
        total_points = sum(
            points_map.get(r, 1) * w 
            for r, w in zip(recent, weights)
        )
        max_points = sum(3 * w for w in weights)
        
        # Scale to 0.85 - 1.15 range
        return 0.85 + 0.30 * (total_points / max_points)
    
    def simulate_match(
        self,
        home_xg: float,
        away_xg: float
    ) -> Tuple[int, int]:
        """Simulate a single match using Dixon-Coles."""
        matrix = self.dixon_coles.score_matrix(home_xg, away_xg)
        
        # Flatten and sample
        flat_probs = matrix.flatten()
        idx = np.random.choice(len(flat_probs), p=flat_probs)
        home_goals, away_goals = np.unravel_index(idx, matrix.shape)
        
        return int(home_goals), int(away_goals)
    
    def predict(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        n_simulations: Optional[int] = None
    ) -> PredictionResult:
        """
        Run Monte Carlo simulation and return prediction.
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            n_simulations: Override for number of simulations
            
        Returns:
            PredictionResult with comprehensive prediction data
        """
        n = n_simulations or self.n_simulations
        
        # Calculate expected goals
        home_xg, away_xg = self.calculate_expected_goals(home_stats, away_stats)
        
        # Run simulations
        results = {
            'home_wins': 0,
            'draws': 0,
            'away_wins': 0,
            'scores': {},
            'total_goals': [],
            'home_clean_sheets': 0,
            'away_clean_sheets': 0,
            'btts': 0,
        }
        
        for _ in range(n):
            home_goals, away_goals = self.simulate_match(home_xg, away_xg)
            
            # Outcome
            if home_goals > away_goals:
                results['home_wins'] += 1
            elif home_goals < away_goals:
                results['away_wins'] += 1
            else:
                results['draws'] += 1
            
            # Score tracking
            score_key = f"{home_goals}-{away_goals}"
            results['scores'][score_key] = results['scores'].get(score_key, 0) + 1
            
            # Markets
            results['total_goals'].append(home_goals + away_goals)
            if away_goals == 0:
                results['home_clean_sheets'] += 1
            if home_goals == 0:
                results['away_clean_sheets'] += 1
            if home_goals > 0 and away_goals > 0:
                results['btts'] += 1
        
        # Calculate probabilities
        home_win_prob = results['home_wins'] / n
        draw_prob = results['draws'] / n
        away_win_prob = results['away_wins'] / n
        
        # Score distribution
        score_dist = {
            k: v / n 
            for k, v in sorted(
                results['scores'].items(), 
                key=lambda x: -x[1]
            )[:15]
        }
        
        most_likely = max(results['scores'].items(), key=lambda x: x[1])[0]
        
        # Market probabilities
        total_goals = np.array(results['total_goals'])
        
        # Confidence intervals (Wilson score)
        home_win_ci = self._wilson_ci(results['home_wins'], n)
        draw_ci = self._wilson_ci(results['draws'], n)
        away_win_ci = self._wilson_ci(results['away_wins'], n)
        
        return PredictionResult(
            home_team=home_stats.name,
            away_team=away_stats.name,
            home_xg=home_xg,
            away_xg=away_xg,
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            home_win_ci=home_win_ci,
            draw_ci=draw_ci,
            away_win_ci=away_win_ci,
            score_distribution=score_dist,
            most_likely_score=most_likely,
            btts_prob=results['btts'] / n,
            over_1_5_prob=np.sum(total_goals > 1.5) / n,
            over_2_5_prob=np.sum(total_goals > 2.5) / n,
            over_3_5_prob=np.sum(total_goals > 3.5) / n,
            clean_sheet_home=results['home_clean_sheets'] / n,
            clean_sheet_away=results['away_clean_sheets'] / n,
            n_simulations=n,
        )
    
    def _wilson_ci(self, successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval."""
        if n == 0:
            return (0.0, 0.0)
        
        from scipy import stats
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        p = successes / n
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    def quick_predict(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats
    ) -> Dict[str, Any]:
        """
        Quick analytical prediction without Monte Carlo.
        
        Faster but less accurate than full simulation.
        """
        home_xg, away_xg = self.calculate_expected_goals(home_stats, away_stats)
        
        matrix = self.dixon_coles.score_matrix(home_xg, away_xg)
        
        # Outcome probabilities
        home_win = np.sum(np.tril(matrix, k=-1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, k=1))
        
        # Most likely score
        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        
        return {
            'home_team': home_stats.name,
            'away_team': away_stats.name,
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2),
            'home_win_prob': round(home_win * 100, 1),
            'draw_prob': round(draw * 100, 1),
            'away_win_prob': round(away_win * 100, 1),
            'most_likely_score': f"{max_idx[0]}-{max_idx[1]}"
        }
