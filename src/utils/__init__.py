"""
Utility Functions

This package contains helper utilities for statistical analysis and visualization:

- stats: Statistical scoring rules, calibration, confidence intervals
- visualization: Plotting functions for predictions and analysis
"""

from .stats import (
    brier_score,
    ranked_probability_score,
    log_loss,
    calibration_curve,
    wilson_confidence_interval,
    jeffreys_interval,
    poisson_confidence_interval,
    expected_value,
    variance,
    kelly_criterion,
    value_bet_detection,
    bootstrap_mean_ci,
    hypothesis_test_proportion,
    chi_square_test,
    CalibrationResult
)

from .visualization import (
    plot_score_matrix,
    plot_outcome_probabilities,
    plot_elo_history,
    plot_simulation_distribution,
    plot_calibration_curve,
    plot_team_comparison,
    plot_soccer_pitch,
    create_prediction_report
)

__all__ = [
    # Stats
    'brier_score',
    'ranked_probability_score',
    'log_loss',
    'calibration_curve',
    'wilson_confidence_interval',
    'jeffreys_interval',
    'poisson_confidence_interval',
    'expected_value',
    'variance',
    'kelly_criterion',
    'value_bet_detection',
    'bootstrap_mean_ci',
    'hypothesis_test_proportion',
    'chi_square_test',
    'CalibrationResult',
    
    # Visualization
    'plot_score_matrix',
    'plot_outcome_probabilities',
    'plot_elo_history',
    'plot_simulation_distribution',
    'plot_calibration_curve',
    'plot_team_comparison',
    'plot_soccer_pitch',
    'create_prediction_report'
]
