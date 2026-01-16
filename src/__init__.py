"""
Soccer Match Predictor with StatsBomb Open Data

A comprehensive soccer match prediction system combining multiple statistical models
with StatsBomb's free open data.

NO CREDENTIALS OR AUTHENTICATION REQUIRED.

Modules:
- models: Dixon-Coles, Bivariate Poisson, Elo Rating, Player Model
- simulation: Monte Carlo engine and Match Simulator
- data: StatsBomb data loader and predictor
- utils: Statistical utilities and visualization

Usage:
    from src import StatsBombLoader, MonteCarloPredictor
    from src.models import DixonColes, EloRating
    from src.simulation import MatchSimulator
"""

# Data loading
from .data import (
    StatsBombLoader,
    TeamStats,
    MonteCarloPredictor,
    PredictionResult,
    DixonColesModel,
    get_world_cup_stats,
    get_euro_stats,
    get_copa_america_stats,
    get_afcon_stats,
    list_available_data
)

# Models
from .models import (
    DixonColes,
    DixonColesParams,
    BivariatePoisson,
    BivariateParams,
    EloRating,
    EloMatch,
    TeamElo,
    PlayerModel,
    Player,
    Squad,
    Formation,
    Position,
    FORMATIONS
)

# Simulation
from .simulation import (
    MonteCarloEngine,
    SimulationResult,
    MatchSimulator,
    SimulationConfig,
    TeamData
)

# Utilities
from .utils import (
    brier_score,
    ranked_probability_score,
    log_loss,
    calibration_curve,
    wilson_confidence_interval,
    kelly_criterion,
    value_bet_detection,
    plot_score_matrix,
    plot_outcome_probabilities,
    create_prediction_report
)

__version__ = '2.0.0'
__author__ = 'Soccer Predictor Team'

__all__ = [
    # Data
    'StatsBombLoader',
    'TeamStats',
    'MonteCarloPredictor',
    'PredictionResult',
    'DixonColesModel',
    'get_world_cup_stats',
    'get_euro_stats',
    'get_copa_america_stats',
    'get_afcon_stats',
    'list_available_data',
    
    # Models
    'DixonColes',
    'DixonColesParams',
    'BivariatePoisson',
    'BivariateParams',
    'EloRating',
    'EloMatch',
    'TeamElo',
    'PlayerModel',
    'Player',
    'Squad',
    'Formation',
    'Position',
    'FORMATIONS',
    
    # Simulation
    'MonteCarloEngine',
    'SimulationResult',
    'MatchSimulator',
    'SimulationConfig',
    'TeamData',
    
    # Utils
    'brier_score',
    'ranked_probability_score',
    'log_loss',
    'calibration_curve',
    'wilson_confidence_interval',
    'kelly_criterion',
    'value_bet_detection',
    'plot_score_matrix',
    'plot_outcome_probabilities',
    'create_prediction_report'
]
