"""
Monte Carlo Simulation Engine

Core engine for running stochastic simulations with:
- Parallelizable simulation runs
- Statistical aggregation
- Confidence interval calculation
- Variance tracking
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import warnings


@dataclass
class SimulationResult:
    """Container for simulation results with statistical measures."""
    n_simulations: int
    raw_results: List[Dict] = field(default_factory=list, repr=False)
    
    # Outcome probabilities
    home_win_prob: float = 0.0
    draw_prob: float = 0.0
    away_win_prob: float = 0.0
    
    # Expected values
    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0
    
    # Score distribution
    score_distribution: Dict[Tuple[int, int], float] = field(default_factory=dict)
    most_likely_score: Tuple[int, int] = (0, 0)
    
    # Confidence intervals
    home_win_ci: Tuple[float, float] = (0.0, 0.0)
    draw_ci: Tuple[float, float] = (0.0, 0.0)
    away_win_ci: Tuple[float, float] = (0.0, 0.0)
    
    # Market probabilities
    btts_prob: float = 0.0  # Both teams to score
    over_1_5_prob: float = 0.0
    over_2_5_prob: float = 0.0
    over_3_5_prob: float = 0.0
    clean_sheet_home: float = 0.0
    clean_sheet_away: float = 0.0
    
    # Additional stats
    std_home_goals: float = 0.0
    std_away_goals: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'n_simulations': self.n_simulations,
            'outcomes': {
                'home_win': self.home_win_prob,
                'draw': self.draw_prob,
                'away_win': self.away_win_prob
            },
            'expected_goals': {
                'home': self.expected_home_goals,
                'away': self.expected_away_goals
            },
            'confidence_intervals': {
                'home_win': self.home_win_ci,
                'draw': self.draw_ci,
                'away_win': self.away_win_ci
            },
            'markets': {
                'btts': self.btts_prob,
                'over_1_5': self.over_1_5_prob,
                'over_2_5': self.over_2_5_prob,
                'over_3_5': self.over_3_5_prob,
                'clean_sheet_home': self.clean_sheet_home,
                'clean_sheet_away': self.clean_sheet_away
            },
            'most_likely_score': self.most_likely_score,
            'score_distribution': {
                f"{k[0]}-{k[1]}": v 
                for k, v in sorted(self.score_distribution.items(), key=lambda x: -x[1])[:10]
            }
        }


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for stochastic match prediction.
    
    Features:
    - Configurable number of simulations
    - Bootstrap confidence intervals
    - Score distribution tracking
    - Parallel execution support
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None,
        parallel: bool = False,
        n_workers: int = 4
    ):
        """
        Initialize Monte Carlo engine.
        
        Args:
            n_simulations: Number of simulations to run
            confidence_level: Confidence level for intervals (0.95 = 95%)
            random_seed: Seed for reproducibility
            parallel: Whether to use parallel execution
            n_workers: Number of parallel workers
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.parallel = parallel
        self.n_workers = n_workers
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def run_simulation(
        self,
        simulation_fn: Callable[[], Dict],
        n_simulations: Optional[int] = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            simulation_fn: Function that returns a single simulation result
                          Expected to return dict with 'home_goals', 'away_goals'
            n_simulations: Override for number of simulations
            
        Returns:
            SimulationResult with aggregated statistics
        """
        n = n_simulations or self.n_simulations
        
        if self.parallel:
            results = self._run_parallel(simulation_fn, n)
        else:
            results = [simulation_fn() for _ in range(n)]
        
        return self._aggregate_results(results, n)
    
    def _run_parallel(
        self,
        simulation_fn: Callable[[], Dict],
        n: int
    ) -> List[Dict]:
        """Run simulations in parallel."""
        chunk_size = n // self.n_workers
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(lambda: [simulation_fn() for _ in range(chunk_size)])
                for _ in range(self.n_workers)
            ]
            results = []
            for future in futures:
                results.extend(future.result())
                
        # Handle remainder
        remainder = n - len(results)
        results.extend([simulation_fn() for _ in range(remainder)])
        
        return results
    
    def _aggregate_results(
        self,
        results: List[Dict],
        n: int
    ) -> SimulationResult:
        """Aggregate individual simulation results into statistics."""
        if not results:
            return SimulationResult(n_simulations=0)
        
        home_goals = np.array([r['home_goals'] for r in results])
        away_goals = np.array([r['away_goals'] for r in results])
        
        # Outcome counts
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        # Probabilities
        home_win_prob = home_wins / n
        draw_prob = draws / n
        away_win_prob = away_wins / n
        
        # Score distribution
        score_counts: Dict[Tuple[int, int], int] = {}
        for hg, ag in zip(home_goals, away_goals):
            key = (int(hg), int(ag))
            score_counts[key] = score_counts.get(key, 0) + 1
        
        score_distribution = {k: v / n for k, v in score_counts.items()}
        most_likely = max(score_counts.items(), key=lambda x: x[1])[0]
        
        # Confidence intervals using Wilson score
        home_win_ci = self._wilson_ci(home_wins, n)
        draw_ci = self._wilson_ci(draws, n)
        away_win_ci = self._wilson_ci(away_wins, n)
        
        # Market probabilities
        btts = np.sum((home_goals > 0) & (away_goals > 0)) / n
        total_goals = home_goals + away_goals
        over_1_5 = np.sum(total_goals > 1.5) / n
        over_2_5 = np.sum(total_goals > 2.5) / n
        over_3_5 = np.sum(total_goals > 3.5) / n
        clean_sheet_home = np.sum(away_goals == 0) / n
        clean_sheet_away = np.sum(home_goals == 0) / n
        
        return SimulationResult(
            n_simulations=n,
            raw_results=results,
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            expected_home_goals=float(np.mean(home_goals)),
            expected_away_goals=float(np.mean(away_goals)),
            score_distribution=score_distribution,
            most_likely_score=most_likely,
            home_win_ci=home_win_ci,
            draw_ci=draw_ci,
            away_win_ci=away_win_ci,
            btts_prob=btts,
            over_1_5_prob=over_1_5,
            over_2_5_prob=over_2_5,
            over_3_5_prob=over_3_5,
            clean_sheet_home=clean_sheet_home,
            clean_sheet_away=clean_sheet_away,
            std_home_goals=float(np.std(home_goals)),
            std_away_goals=float(np.std(away_goals))
        )
    
    def _wilson_ci(
        self,
        successes: int,
        n: int
    ) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval.
        
        More accurate than normal approximation for proportions,
        especially near 0 or 1.
        """
        if n == 0:
            return (0.0, 0.0)
            
        from scipy import stats
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        
        p = successes / n
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
        
        lower = max(0, center - margin)
        upper = min(1, center + margin)
        
        return (lower, upper)
    
    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: Original data array
            statistic: Function to calculate statistic
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(sample))
        
        alpha = (1 - self.confidence_level) / 2
        lower = np.percentile(bootstrap_stats, alpha * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)
        
        return (lower, upper)
    
    def sensitivity_analysis(
        self,
        base_params: Dict,
        param_variations: Dict[str, List[float]],
        simulation_fn: Callable[[Dict], Dict]
    ) -> Dict[str, List[SimulationResult]]:
        """
        Run sensitivity analysis varying one parameter at a time.
        
        Args:
            base_params: Base parameter dictionary
            param_variations: Dict mapping param name to list of values to test
            simulation_fn: Function taking params and returning simulation result
            
        Returns:
            Dictionary mapping param names to lists of results
        """
        results = {}
        
        for param_name, values in param_variations.items():
            results[param_name] = []
            
            for value in values:
                params = base_params.copy()
                params[param_name] = value
                
                result = self.run_simulation(
                    lambda p=params: simulation_fn(p),
                    n_simulations=self.n_simulations // len(values)
                )
                results[param_name].append(result)
        
        return results
    
    def convergence_check(
        self,
        simulation_fn: Callable[[], Dict],
        checkpoints: List[int] = None,
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Check simulation convergence by running at different n values.
        
        Args:
            simulation_fn: Simulation function
            checkpoints: List of n values to test
            tolerance: Convergence tolerance for probability estimates
            
        Returns:
            Dictionary with convergence analysis
        """
        if checkpoints is None:
            checkpoints = [100, 500, 1000, 2500, 5000, 10000]
        
        checkpoint_results = []
        
        for n in checkpoints:
            result = self.run_simulation(simulation_fn, n_simulations=n)
            checkpoint_results.append({
                'n': n,
                'home_win': result.home_win_prob,
                'draw': result.draw_prob,
                'away_win': result.away_win_prob,
                'home_xg': result.expected_home_goals,
                'away_xg': result.expected_away_goals
            })
        
        # Check if converged
        if len(checkpoint_results) >= 2:
            last = checkpoint_results[-1]
            prev = checkpoint_results[-2]
            
            converged = all([
                abs(last['home_win'] - prev['home_win']) < tolerance,
                abs(last['draw'] - prev['draw']) < tolerance,
                abs(last['away_win'] - prev['away_win']) < tolerance
            ])
        else:
            converged = False
        
        return {
            'checkpoints': checkpoint_results,
            'converged': converged,
            'recommended_n': checkpoints[-1] if converged else checkpoints[-1] * 2
        }
