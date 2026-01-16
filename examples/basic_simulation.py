"""
Basic Simulation Example

Demonstrates how to use the soccer predictor with StatsBomb open data.

NO CREDENTIALS OR AUTHENTICATION REQUIRED - uses free open data.
"""

import sys
sys.path.insert(0, '..')

from src.data import StatsBombLoader, MonteCarloPredictor, get_world_cup_stats
from src.models import DixonColes, EloRating, BivariatePoisson
from src.simulation import MatchSimulator, SimulationConfig, TeamData


def basic_prediction_example():
    """
    Basic example: Load World Cup data and predict a match.
    """
    print("=" * 60)
    print("  Basic Match Prediction Example")
    print("=" * 60)
    
    # Load World Cup 2022 team statistics
    print("\n1. Loading World Cup 2022 data from StatsBomb...")
    team_stats = get_world_cup_stats(year=2022, calculate_xg=False)
    
    print(f"   Loaded stats for {len(team_stats)} teams")
    
    # Get two teams
    home_team = team_stats.get('Argentina')
    away_team = team_stats.get('France')
    
    if not home_team or not away_team:
        print("   Teams not found, using available teams...")
        teams = list(team_stats.keys())
        home_team = team_stats[teams[0]]
        away_team = team_stats[teams[1]]
    
    print(f"\n2. Teams selected:")
    print(f"   Home: {home_team.name} (Elo: {home_team.elo_rating:.0f})")
    print(f"   Away: {away_team.name} (Elo: {away_team.elo_rating:.0f})")
    
    # Create predictor and run simulation
    print("\n3. Running Monte Carlo simulation (10,000 iterations)...")
    predictor = MonteCarloPredictor(n_simulations=10000)
    result = predictor.predict(home_team, away_team)
    
    # Display results
    print(result)
    
    return result


def advanced_simulation_example():
    """
    Advanced example using the full MatchSimulator with all models.
    """
    print("\n" + "=" * 60)
    print("  Advanced Simulation Example")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading Euro 2024 data...")
    loader = StatsBombLoader()
    matches = loader.load_matches_by_key('euro_2024')
    
    if not matches:
        print("   Could not load Euro 2024 data")
        return
    
    print(f"   Loaded {len(matches)} matches")
    
    # Convert to format needed for model fitting
    match_dicts = []
    for _, match in matches.iterrows():
        match_dicts.append({
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'home_goals': match['home_score'],
            'away_goals': match['away_score'],
            'date': str(match.get('match_date', ''))
        })
    
    # Initialize simulator with custom config
    print("\n2. Initializing MatchSimulator with ensemble models...")
    config = SimulationConfig(
        n_simulations=5000,
        home_advantage=0.25,
        dixon_coles_rho=-0.13,
        model_weights={
            'dixon_coles': 0.4,
            'bivariate_poisson': 0.3,
            'elo': 0.2,
            'player_model': 0.1
        }
    )
    
    simulator = MatchSimulator(config=config)
    
    # Fit models on historical data
    print("   Fitting models on historical match data...")
    simulator.fit_from_historical(match_dicts)
    
    # Get Elo rankings
    print("\n3. Elo Rankings after fitting:")
    rankings = simulator.elo_system.get_rankings(top_n=10)
    for i, (team, elo) in enumerate(rankings, 1):
        print(f"   {i:2d}. {team}: {elo:.0f}")
    
    # Predict a match
    print("\n4. Predicting match...")
    
    # Create team data from Elo system
    top_teams = [team for team, _ in rankings[:2]]
    
    home_data = TeamData(
        name=top_teams[0],
        elo=simulator.elo_system.get_rating(top_teams[0]),
        attack_strength=1.1,
        defense_strength=0.9
    )
    
    away_data = TeamData(
        name=top_teams[1],
        elo=simulator.elo_system.get_rating(top_teams[1]),
        attack_strength=1.05,
        defense_strength=0.95
    )
    
    # Run simulation
    result = simulator.simulate_match(home_data, away_data)
    
    print(f"\n   {home_data.name} vs {away_data.name}")
    print(f"   Expected Goals: {result.expected_home_goals:.2f} - {result.expected_away_goals:.2f}")
    print(f"   Home Win: {result.home_win_prob:.1%}")
    print(f"   Draw: {result.draw_prob:.1%}")
    print(f"   Away Win: {result.away_win_prob:.1%}")
    print(f"   Most Likely Score: {result.most_likely_score[0]}-{result.most_likely_score[1]}")
    print(f"   BTTS: {result.btts_prob:.1%}")
    print(f"   Over 2.5: {result.over_2_5_prob:.1%}")
    
    return result


def model_comparison_example():
    """
    Compare different prediction models.
    """
    print("\n" + "=" * 60)
    print("  Model Comparison Example")
    print("=" * 60)
    
    # Set up a match
    home_xg = 1.8
    away_xg = 1.2
    
    print(f"\n  Match setup: Home xG = {home_xg}, Away xG = {away_xg}")
    
    # Dixon-Coles
    print("\n1. Dixon-Coles Model:")
    dc = DixonColes(rho=-0.13)
    dc_outcomes = dc.outcome_probabilities(home_xg, away_xg)
    print(f"   Home Win: {dc_outcomes['home_win']:.1%}")
    print(f"   Draw: {dc_outcomes['draw']:.1%}")
    print(f"   Away Win: {dc_outcomes['away_win']:.1%}")
    
    # Bivariate Poisson
    print("\n2. Bivariate Poisson Model:")
    bp = BivariatePoisson(lambda_covariance=0.1)
    bp_outcomes = bp.outcome_probabilities(home_xg - 0.1, away_xg - 0.1, lambda_3=0.1)
    print(f"   Home Win: {bp_outcomes['home_win']:.1%}")
    print(f"   Draw: {bp_outcomes['draw']:.1%}")
    print(f"   Away Win: {bp_outcomes['away_win']:.1%}")
    print(f"   BTTS: {bp_outcomes['btts']:.1%}")
    
    # Standard Poisson (using Dixon-Coles with rho=0)
    print("\n3. Standard Poisson (no low-score adjustment):")
    poisson = DixonColes(rho=0)
    poisson_outcomes = poisson.outcome_probabilities(home_xg, away_xg)
    print(f"   Home Win: {poisson_outcomes['home_win']:.1%}")
    print(f"   Draw: {poisson_outcomes['draw']:.1%}")
    print(f"   Away Win: {poisson_outcomes['away_win']:.1%}")


def list_available_competitions():
    """
    List all available free competitions.
    """
    print("\n" + "=" * 60)
    print("  Available Free Competitions")
    print("=" * 60)
    
    from src.data import list_available_data
    list_available_data()


if __name__ == "__main__":
    # List available data
    list_available_competitions()
    
    # Run examples
    try:
        basic_prediction_example()
    except Exception as e:
        print(f"Basic example failed: {e}")
    
    try:
        model_comparison_example()
    except Exception as e:
        print(f"Model comparison failed: {e}")
    
    try:
        advanced_simulation_example()
    except Exception as e:
        print(f"Advanced example failed: {e}")
    
    print("\n" + "=" * 60)
    print("  Examples completed!")
    print("=" * 60)
