"""
Soccer Match Simulation

This package contains the Monte Carlo simulation engine and match simulator:

- MonteCarloEngine: Core stochastic simulation with confidence intervals
- MatchSimulator: Combines all models for comprehensive predictions
- SimulationResult: Container for simulation outputs
- SimulationConfig: Configuration settings for simulation
"""

from .monte_carlo import MonteCarloEngine, SimulationResult
from .match_simulator import MatchSimulator, SimulationConfig, TeamData

__all__ = [
    'MonteCarloEngine',
    'SimulationResult',
    'MatchSimulator',
    'SimulationConfig',
    'TeamData'
]
