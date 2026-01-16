"""
Soccer Match Prediction Models

This package contains various statistical models for predicting soccer match outcomes:

- DixonColes: Adjusted Poisson model for low-scoring games
- BivariatePoisson: Joint distribution accounting for score correlation
- EloRating: Dynamic team strength tracking system
- PlayerModel: Player-level impact modeling
"""

from .dixon_coles import DixonColes, DixonColesParams
from .bivariate_poisson import BivariatePoisson, BivariateParams
from .elo_rating import EloRating, EloMatch, TeamElo
from .player_model import (
    PlayerModel, 
    Player, 
    Squad, 
    Formation,
    Position,
    FORMATIONS
)

__all__ = [
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
    'FORMATIONS'
]
