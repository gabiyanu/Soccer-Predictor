"""
Data module for StatsBomb soccer match predictor.

Uses ONLY FREE OPEN DATA - No credentials required!
Data Source: https://github.com/statsbomb/open-data

Provides:
- StatsBombLoader: Load and process StatsBomb open data
- TeamStats: Computed team statistics
- MonteCarloPredictor: Prediction engine
- PredictionResult: Prediction results container
- Convenience functions for popular tournaments
"""

from .statsbomb_loader import (
    StatsBombLoader,
    TeamStats,
    get_world_cup_stats,
    get_euro_stats,
    get_copa_america_stats,
    get_afcon_stats,
    list_available_data,
    STATSBOMB_AVAILABLE
)

from .predictor import (
    MonteCarloPredictor,
    PredictionResult,
    DixonColesModel
)

__all__ = [
    # Main classes
    'StatsBombLoader',
    'TeamStats',
    'MonteCarloPredictor',
    'PredictionResult',
    'DixonColesModel',
    
    # Convenience functions (all use FREE data)
    'get_world_cup_stats',
    'get_euro_stats',
    'get_copa_america_stats',
    'get_afcon_stats',
    'list_available_data',
    
    # Status
    'STATSBOMB_AVAILABLE'
]
