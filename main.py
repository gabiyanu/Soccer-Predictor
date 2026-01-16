#!/usr/bin/env python3
"""
Soccer Match Predictor - Interactive CLI

Predicts soccer match outcomes using StatsBomb FREE OPEN DATA.
NO CREDENTIALS OR AUTHENTICATION REQUIRED.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import StatsBombLoader, MonteCarloPredictor, list_available_data


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print application header."""
    print("\n" + "=" * 60)
    print("  ⚽ SOCCER MATCH PREDICTOR ⚽")
    print("  Using StatsBomb Free Open Data")
    print("=" * 60)


def print_menu():
    """Print main menu options."""
    print("\n  What would you like to do?\n")
    print("  1. Predict a match")
    print("  2. View team rankings")
    print("  3. View team statistics")
    print("  4. List available competitions")
    print("  5. Exit")
    print()


def get_competition_choice(loader):
    """Let user select a competition."""
    competitions = loader.FREE_COMPETITIONS
    
    print("\n  Available Competitions:\n")
    comp_list = list(competitions.items())
    
    for i, (key, info) in enumerate(comp_list, 1):
        print(f"  {i:2d}. {info['name']}")
    
    print()
    while True:
        try:
            choice = input("  Enter competition number (or 'b' to go back): ").strip()
            if choice.lower() == 'b':
                return None, None
            
            idx = int(choice) - 1
            if 0 <= idx < len(comp_list):
                key, info = comp_list[idx]
                return key, info
            else:
                print("  ❌ Invalid choice. Please try again.")
        except ValueError:
            print("  ❌ Please enter a number.")


def get_team_choice(teams, prompt="Select team"):
    """Let user select a team from list."""
    print(f"\n  {prompt}:\n")
    
    # Sort teams alphabetically
    sorted_teams = sorted(teams)
    
    # Display in columns
    col_width = 25
    cols = 3
    
    for i in range(0, len(sorted_teams), cols):
        row = ""
        for j in range(cols):
            if i + j < len(sorted_teams):
                team_str = f"{i+j+1:2d}. {sorted_teams[i+j]}"
                row += team_str.ljust(col_width)
        print(f"  {row}")
    
    print()
    while True:
        try:
            choice = input(f"  Enter team number (or 'b' to go back): ").strip()
            if choice.lower() == 'b':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_teams):
                return sorted_teams[idx]
            else:
                print("  ❌ Invalid choice. Please try again.")
        except ValueError:
            print("  ❌ Please enter a number.")


def get_simulations():
    """Get number of simulations from user."""
    print("\n  How many simulations to run?")
    print("  (More = more accurate but slower)")
    print("  Recommended: 10000\n")
    
    while True:
        try:
            choice = input("  Enter number (or press Enter for 10000): ").strip()
            if choice == "":
                return 10000
            
            n = int(choice)
            if 100 <= n <= 100000:
                return n
            else:
                print("  ❌ Please enter a number between 100 and 100000.")
        except ValueError:
            print("  ❌ Please enter a valid number.")


def predict_match():
    """Main prediction flow."""
    loader = StatsBombLoader()
    
    # Step 1: Choose competition
    print("\n" + "-" * 60)
    print("  STEP 1: Choose Competition")
    print("-" * 60)
    
    comp_key, comp_info = get_competition_choice(loader)
    if comp_key is None:
        return
    
    print(f"\n  ✓ Selected: {comp_info['name']}")
    
    # Step 2: Load data
    print("\n" + "-" * 60)
    print("  STEP 2: Loading Data...")
    print("-" * 60)
    
    print(f"\n  Loading {comp_info['name']} data from StatsBomb...")
    
    try:
        team_stats = loader.build_team_stats(
            competition_id=comp_info['competition_id'],
            season_id=comp_info['season_id'],
            calculate_xg=False,
            verbose=False
        )
    except Exception as e:
        print(f"\n  ❌ Error loading data: {e}")
        print("  Please check your internet connection and try again.")
        input("\n  Press Enter to continue...")
        return
    
    if not team_stats:
        print("\n  ❌ No team data found for this competition.")
        input("\n  Press Enter to continue...")
        return
    
    teams = list(team_stats.keys())
    print(f"  ✓ Loaded {len(teams)} teams")
    
    # Step 3: Select home team
    print("\n" + "-" * 60)
    print("  STEP 3: Select Home Team")
    print("-" * 60)
    
    home_team = get_team_choice(teams, "Select HOME team")
    if home_team is None:
        return
    
    print(f"\n  ✓ Home team: {home_team}")
    
    # Step 4: Select away team
    print("\n" + "-" * 60)
    print("  STEP 4: Select Away Team")
    print("-" * 60)
    
    # Remove home team from options
    away_teams = [t for t in teams if t != home_team]
    
    away_team = get_team_choice(away_teams, "Select AWAY team")
    if away_team is None:
        return
    
    print(f"\n  ✓ Away team: {away_team}")
    
    # Step 5: Get simulation count
    print("\n" + "-" * 60)
    print("  STEP 5: Simulation Settings")
    print("-" * 60)
    
    n_simulations = get_simulations()
    print(f"\n  ✓ Running {n_simulations:,} simulations")
    
    # Step 6: Run prediction
    print("\n" + "-" * 60)
    print("  STEP 6: Running Prediction...")
    print("-" * 60)
    
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    print(f"\n  {home_team} (Elo: {home_stats.elo_rating:.0f}) vs {away_team} (Elo: {away_stats.elo_rating:.0f})")
    print("\n  Simulating match outcomes...")
    
    predictor = MonteCarloPredictor(n_simulations=n_simulations)
    result = predictor.predict(home_stats, away_stats)
    
    # Display results
    print(result)
    
    # Ask to predict another
    print("\n" + "-" * 60)
    choice = input("  Predict another match? (y/n): ").strip().lower()
    if choice == 'y':
        predict_match()


def view_rankings():
    """View team rankings for a competition."""
    loader = StatsBombLoader()
    
    print("\n" + "-" * 60)
    print("  TEAM RANKINGS")
    print("-" * 60)
    
    comp_key, comp_info = get_competition_choice(loader)
    if comp_key is None:
        return
    
    print(f"\n  Loading {comp_info['name']} data...")
    
    try:
        team_stats = loader.build_team_stats(
            competition_id=comp_info['competition_id'],
            season_id=comp_info['season_id'],
            calculate_xg=False,
            verbose=False
        )
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        input("\n  Press Enter to continue...")
        return
    
    if not team_stats:
        print("\n  ❌ No data found.")
        input("\n  Press Enter to continue...")
        return
    
    # Sort by Elo rating
    rankings = sorted(team_stats.items(), key=lambda x: x[1].elo_rating, reverse=True)
    
    print(f"\n  {comp_info['name']} - Team Rankings by Elo\n")
    print("  " + "-" * 50)
    print(f"  {'Rank':<6}{'Team':<25}{'Elo':<10}{'W-D-L'}")
    print("  " + "-" * 50)
    
    for i, (name, stats) in enumerate(rankings, 1):
        record = f"{stats.wins}-{stats.draws}-{stats.losses}"
        print(f"  {i:<6}{name:<25}{stats.elo_rating:<10.0f}{record}")
    
    print("  " + "-" * 50)
    
    input("\n  Press Enter to continue...")


def view_team_stats():
    """View detailed stats for a specific team."""
    loader = StatsBombLoader()
    
    print("\n" + "-" * 60)
    print("  TEAM STATISTICS")
    print("-" * 60)
    
    comp_key, comp_info = get_competition_choice(loader)
    if comp_key is None:
        return
    
    print(f"\n  Loading {comp_info['name']} data...")
    
    try:
        team_stats = loader.build_team_stats(
            competition_id=comp_info['competition_id'],
            season_id=comp_info['season_id'],
            calculate_xg=False,
            verbose=False
        )
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        input("\n  Press Enter to continue...")
        return
    
    if not team_stats:
        print("\n  ❌ No data found.")
        input("\n  Press Enter to continue...")
        return
    
    teams = list(team_stats.keys())
    team = get_team_choice(teams, "Select team to view")
    if team is None:
        return
    
    stats = team_stats[team]
    
    print(f"\n  {'-' * 40}")
    print(f"  {team} - Detailed Statistics")
    print(f"  {'-' * 40}")
    print(f"\n  Matches Played: {stats.matches_played}")
    print(f"  Record: {stats.wins}W - {stats.draws}D - {stats.losses}L")
    print(f"  Points: {stats.wins * 3 + stats.draws}")
    print(f"\n  Goals Scored: {stats.goals_scored}")
    print(f"  Goals Conceded: {stats.goals_conceded}")
    print(f"  Goal Difference: {stats.goals_scored - stats.goals_conceded:+d}")
    print(f"\n  Elo Rating: {stats.elo_rating:.0f}")
    print(f"  Attack Strength: {stats.attack_strength:.2f}")
    print(f"  Defense Strength: {stats.defense_strength:.2f}")
    
    if stats.recent_form:
        form_str = "-".join(stats.recent_form[-5:])
        print(f"\n  Recent Form: {form_str}")
    
    print(f"  {'-' * 40}")
    
    input("\n  Press Enter to continue...")


def list_competitions():
    """List all available competitions."""
    print("\n" + "-" * 60)
    print("  AVAILABLE COMPETITIONS (FREE DATA)")
    print("-" * 60)
    
    list_available_data()
    
    input("\n  Press Enter to continue...")


def main():
    """Main application loop."""
    print_header()
    
    while True:
        print_menu()
        
        choice = input("  Enter your choice (1-5): ").strip()
        
        if choice == '1':
            predict_match()
        elif choice == '2':
            view_rankings()
        elif choice == '3':
            view_team_stats()
        elif choice == '4':
            list_competitions()
        elif choice == '5':
            print("\n  Thanks for using Soccer Match Predictor!")
            print("  Goodbye! ⚽\n")
            break
        else:
            print("\n  ❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Goodbye! ⚽\n")
        sys.exit(0)
