"""
Statistical Utilities

Helper functions for probability calibration, scoring rules, and statistical analysis.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Results from probability calibration analysis."""
    bins: np.ndarray
    observed_frequencies: np.ndarray
    expected_frequencies: np.ndarray
    sample_sizes: np.ndarray
    reliability_score: float
    resolution_score: float
    brier_score: float


def brier_score(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray
) -> float:
    """
    Calculate Brier score for probability predictions.
    
    Brier = (1/n) * Σ(p_i - o_i)²
    
    Lower is better. Range: 0 (perfect) to 1 (worst).
    
    Args:
        predicted_probs: Predicted probabilities (0-1)
        outcomes: Binary outcomes (0 or 1)
        
    Returns:
        Brier score
    """
    return np.mean((predicted_probs - outcomes) ** 2)


def ranked_probability_score(
    predicted_probs: np.ndarray,
    actual_outcome: int
) -> float:
    """
    Calculate Ranked Probability Score for ordered categorical predictions.
    
    Used for home win/draw/away win predictions.
    
    Args:
        predicted_probs: Array of [p_home_win, p_draw, p_away_win]
        actual_outcome: 0=home win, 1=draw, 2=away win
        
    Returns:
        RPS (0-1, lower is better)
    """
    n_outcomes = len(predicted_probs)
    actual = np.zeros(n_outcomes)
    actual[actual_outcome] = 1.0
    
    # Cumulative distributions
    cum_pred = np.cumsum(predicted_probs)
    cum_actual = np.cumsum(actual)
    
    # RPS = (1/(n-1)) * Σ(cum_pred_i - cum_actual_i)²
    rps = np.sum((cum_pred - cum_actual) ** 2) / (n_outcomes - 1)
    
    return rps


def log_loss(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray,
    eps: float = 1e-15
) -> float:
    """
    Calculate logarithmic loss (cross-entropy).
    
    Args:
        predicted_probs: Predicted probabilities
        outcomes: Binary outcomes
        eps: Small value to prevent log(0)
        
    Returns:
        Log loss (lower is better)
    """
    # Clip probabilities to avoid log(0)
    clipped = np.clip(predicted_probs, eps, 1 - eps)
    
    return -np.mean(
        outcomes * np.log(clipped) +
        (1 - outcomes) * np.log(1 - clipped)
    )


def calibration_curve(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> CalibrationResult:
    """
    Calculate calibration curve for probability predictions.
    
    Args:
        predicted_probs: Predicted probabilities
        outcomes: Binary outcomes
        n_bins: Number of bins for calibration
        strategy: 'uniform' or 'quantile' binning
        
    Returns:
        CalibrationResult with bin-level metrics
    """
    if strategy == 'uniform':
        bins = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(predicted_probs, percentiles)
        bins[0] = 0
        bins[-1] = 1
    
    bin_indices = np.digitize(predicted_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    observed = np.zeros(n_bins)
    expected = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            observed[i] = np.mean(outcomes[mask])
            expected[i] = np.mean(predicted_probs[mask])
            counts[i] = np.sum(mask)
    
    # Reliability (calibration error)
    n = len(outcomes)
    reliability = np.sum(counts * (observed - expected) ** 2) / n
    
    # Resolution
    overall_freq = np.mean(outcomes)
    resolution = np.sum(counts * (observed - overall_freq) ** 2) / n
    
    # Brier
    brier = brier_score(predicted_probs, outcomes)
    
    return CalibrationResult(
        bins=bins,
        observed_frequencies=observed,
        expected_frequencies=expected,
        sample_sizes=counts,
        reliability_score=reliability,
        resolution_score=resolution,
        brier_score=brier
    )


def wilson_confidence_interval(
    successes: int,
    n: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.
    
    More accurate than normal approximation for small samples
    and proportions near 0 or 1.
    
    Args:
        successes: Number of successes
        n: Total trials
        confidence: Confidence level (0.95 = 95%)
        
    Returns:
        Tuple of (lower, upper) bounds
    """
    if n == 0:
        return (0.0, 1.0)
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
    
    return (max(0, center - margin), min(1, center + margin))


def jeffreys_interval(
    successes: int,
    n: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Jeffreys prior confidence interval.
    
    Bayesian interval using Beta(0.5, 0.5) prior.
    
    Args:
        successes: Number of successes
        n: Total trials
        confidence: Confidence level
        
    Returns:
        Tuple of (lower, upper) bounds
    """
    alpha = (1 - confidence) / 2
    
    # Beta distribution parameters
    a = successes + 0.5
    b = n - successes + 0.5
    
    lower = stats.beta.ppf(alpha, a, b)
    upper = stats.beta.ppf(1 - alpha, a, b)
    
    return (lower, upper)


def poisson_confidence_interval(
    count: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for Poisson rate.
    
    Args:
        count: Observed count
        confidence: Confidence level
        
    Returns:
        Tuple of (lower, upper) rate bounds
    """
    alpha = (1 - confidence) / 2
    
    if count == 0:
        lower = 0
    else:
        lower = stats.chi2.ppf(alpha, 2 * count) / 2
    
    upper = stats.chi2.ppf(1 - alpha, 2 * (count + 1)) / 2
    
    return (lower, upper)


def expected_value(
    outcomes: Dict[any, float],
    values: Optional[Dict[any, float]] = None
) -> float:
    """
    Calculate expected value from probability distribution.
    
    Args:
        outcomes: Dict mapping outcomes to probabilities
        values: Dict mapping outcomes to values (uses keys as values if None)
        
    Returns:
        Expected value
    """
    if values is None:
        # Assume outcomes keys are numeric
        return sum(k * p for k, p in outcomes.items())
    
    return sum(values.get(k, 0) * p for k, p in outcomes.items())


def variance(
    outcomes: Dict[any, float],
    values: Optional[Dict[any, float]] = None
) -> float:
    """
    Calculate variance of probability distribution.
    
    Args:
        outcomes: Dict mapping outcomes to probabilities
        values: Dict mapping outcomes to values
        
    Returns:
        Variance
    """
    if values is None:
        vals = {k: k for k in outcomes.keys()}
    else:
        vals = values
    
    mean = expected_value(outcomes, vals)
    
    return sum(
        ((vals.get(k, 0) - mean) ** 2) * p 
        for k, p in outcomes.items()
    )


def kelly_criterion(
    win_prob: float,
    odds: float,
    bankroll_fraction: float = 1.0
) -> float:
    """
    Calculate Kelly criterion optimal bet size.
    
    f* = (bp - q) / b
    
    where:
    - b = decimal odds - 1 (profit multiple)
    - p = probability of winning
    - q = probability of losing (1 - p)
    
    Args:
        win_prob: Estimated probability of winning
        odds: Decimal odds (e.g., 2.5 for +150)
        bankroll_fraction: Maximum fraction of bankroll to risk
        
    Returns:
        Optimal bet fraction (0 = don't bet)
    """
    b = odds - 1
    p = win_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Cap at bankroll fraction and ensure non-negative
    return max(0, min(kelly, bankroll_fraction))


def value_bet_detection(
    estimated_prob: float,
    market_odds: float,
    min_edge: float = 0.05
) -> Dict[str, any]:
    """
    Detect potential value bets.
    
    Args:
        estimated_prob: Your estimated probability
        market_odds: Market decimal odds
        min_edge: Minimum edge required
        
    Returns:
        Dict with value analysis
    """
    implied_prob = 1 / market_odds
    edge = estimated_prob - implied_prob
    
    return {
        'estimated_prob': estimated_prob,
        'implied_prob': implied_prob,
        'edge': edge,
        'is_value': edge > min_edge,
        'kelly_fraction': kelly_criterion(estimated_prob, market_odds),
        'expected_return': estimated_prob * (market_odds - 1) - (1 - estimated_prob)
    }


def bootstrap_mean_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for mean.
    
    Args:
        data: Data array
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Tuple of (lower, upper) bounds
    """
    n = len(data)
    means = np.array([
        np.mean(np.random.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    
    alpha = (1 - confidence) / 2
    return (
        np.percentile(means, alpha * 100),
        np.percentile(means, (1 - alpha) * 100)
    )


def hypothesis_test_proportion(
    successes: int,
    n: int,
    null_prob: float = 0.5,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Perform hypothesis test for proportion.
    
    Args:
        successes: Number of successes
        n: Total trials
        null_prob: Null hypothesis probability
        alternative: 'two-sided', 'greater', or 'less'
        
    Returns:
        Dict with test results
    """
    observed_prop = successes / n
    
    # Standard error under null
    se = np.sqrt(null_prob * (1 - null_prob) / n)
    
    # Z-statistic
    z = (observed_prop - null_prob) / se
    
    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z)
    else:  # less
        p_value = stats.norm.cdf(z)
    
    return {
        'observed_proportion': observed_prop,
        'z_statistic': z,
        'p_value': p_value,
        'significant_at_05': p_value < 0.05,
        'significant_at_01': p_value < 0.01
    }


def chi_square_test(
    observed: np.ndarray,
    expected: np.ndarray
) -> Dict[str, float]:
    """
    Perform chi-square goodness of fit test.
    
    Args:
        observed: Observed frequencies
        expected: Expected frequencies
        
    Returns:
        Dict with test results
    """
    chi2, p_value = stats.chisquare(observed, expected)
    
    return {
        'chi_square': chi2,
        'p_value': p_value,
        'degrees_of_freedom': len(observed) - 1,
        'significant': p_value < 0.05
    }
