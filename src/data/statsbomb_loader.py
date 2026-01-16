"""
StatsBomb Open Data Loader

Fetches and processes match data from StatsBomb's FREE OPEN DATA.
NO CREDENTIALS OR AUTHENTICATION REQUIRED.

Data Source: https://github.com/statsbomb/open-data

Available Free Competitions (as of 2024):
- FIFA World Cup: 2018, 2022
- UEFA Euro: 2020, 2024
- Copa America: 2024
- Africa Cup of Nations: 2023
- FA Women's Super League: Multiple seasons
- NWSL: Multiple seasons  
- FIFA Women's World Cup: 2019, 2023
- La Liga: Select matches (Messi data 2004-2021)
- Premier League: 2003/2004 (Arsenal Invincibles)
- Champions League: Select finals/matches
- Bundesliga: Select seasons

Note: statsbombpy automatically uses open data when no credentials provided.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import warnings
import urllib.request

# Try statsbombpy first, fall back to direct GitHub access
try:
    from statsbombpy import sb
    STATSBOMB_AVAILABLE = True
except ImportError:
    STATSBOMB_AVAILABLE = False
    warnings.warn(
        "statsbombpy not installed. Install with: pip install statsbombpy\n"
        "Alternatively, data can be loaded directly from GitHub."
    )

# GitHub raw URLs for direct access (no library needed)
GITHUB_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


@dataclass
class TeamStats:
    """Computed statistics for a team."""
    name: str
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0
    shots_for: int = 0
    shots_against: int = 0
    shots_on_target_for: int = 0
    shots_on_target_against: int = 0
    possession_avg: float = 50.0
    passes_completed: int = 0
    passes_attempted: int = 0
    recent_form: List[str] = field(default_factory=list)
    elo_rating: float = 1500.0
    
    @property
    def attack_strength(self) -> float:
        """Calculate attack strength relative to league average."""
        if self.matches_played == 0:
            return 1.0
        goals_per_game = self.goals_scored / self.matches_played
        # Assume league average is ~1.35 goals per game
        return max(0.5, min(2.0, goals_per_game / 1.35))
    
    @property
    def defense_strength(self) -> float:
        """Calculate defense strength (lower is better, inverted for model)."""
        if self.matches_played == 0:
            return 1.0
        goals_against_per_game = self.goals_conceded / self.matches_played
        # Inverted: fewer goals = lower value = better defense
        return max(0.5, min(2.0, goals_against_per_game / 1.35))
    
    @property
    def xg_attack_strength(self) -> float:
        """Attack strength based on xG."""
        if self.matches_played == 0:
            return 1.0
        xg_per_game = self.xg_for / self.matches_played
        return max(0.5, min(2.0, xg_per_game / 1.35))
    
    @property
    def xg_defense_strength(self) -> float:
        """Defense strength based on xG against."""
        if self.matches_played == 0:
            return 1.0
        xga_per_game = self.xg_against / self.matches_played
        return max(0.5, min(2.0, xga_per_game / 1.35))
    
    @property
    def shot_conversion_rate(self) -> float:
        """Goals per shot."""
        if self.shots_for == 0:
            return 0.1
        return self.goals_scored / self.shots_for
    
    @property
    def pass_accuracy(self) -> float:
        """Pass completion percentage."""
        if self.passes_attempted == 0:
            return 0.75
        return self.passes_completed / self.passes_attempted
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'matches_played': self.matches_played,
            'wins': self.wins,
            'draws': self.draws,
            'losses': self.losses,
            'goals_scored': self.goals_scored,
            'goals_conceded': self.goals_conceded,
            'goal_difference': self.goals_scored - self.goals_conceded,
            'points': self.wins * 3 + self.draws,
            'xg_for': round(self.xg_for, 2),
            'xg_against': round(self.xg_against, 2),
            'attack_strength': round(self.attack_strength, 3),
            'defense_strength': round(self.defense_strength, 3),
            'xg_attack_strength': round(self.xg_attack_strength, 3),
            'xg_defense_strength': round(self.xg_defense_strength, 3),
            'elo_rating': round(self.elo_rating, 1),
            'recent_form': self.recent_form[-5:],
            'possession_avg': round(self.possession_avg, 1),
        }


class StatsBombLoader:
    """
    Load and process data from StatsBomb's FREE OPEN DATA.
    
    NO AUTHENTICATION OR CREDENTIALS REQUIRED.
    
    The statsbombpy library automatically accesses open data when
    no credentials are provided. Alternatively, data can be loaded
    directly from GitHub JSON files.
    
    Verified Free Competitions (competition_id: season_id):
    --------------------------------------------------------
    - FIFA World Cup 2022: (43, 106)
    - FIFA World Cup 2018: (43, 3)
    - UEFA Euro 2024: (55, 282)
    - UEFA Euro 2020: (55, 43)
    - Copa America 2024: (223, 282)
    - Africa Cup of Nations 2023: (1267, 107)
    - FIFA Women's World Cup 2023: (72, 107)
    - FIFA Women's World Cup 2019: (72, 30)
    - FA Women's Super League 2023/24: (37, 108)
    - NWSL 2018: (49, 3)
    - La Liga 2020/21 (partial): (11, 90)
    - La Liga 2019/20 (partial): (11, 42)
    - Premier League 2003/04: (2, 44)
    - Champions League 2018/19 (select): (16, 4)
    """
    
    # VERIFIED FREE DATA - Competition IDs and their available seasons
    # Only includes competitions confirmed in open-data repository
    FREE_COMPETITIONS = {
        # International Tournaments (Full Coverage)
        'world_cup_2022': {'competition_id': 43, 'season_id': 106, 'name': 'FIFA World Cup 2022'},
        'world_cup_2018': {'competition_id': 43, 'season_id': 3, 'name': 'FIFA World Cup 2018'},
        'euro_2024': {'competition_id': 55, 'season_id': 282, 'name': 'UEFA Euro 2024'},
        'euro_2020': {'competition_id': 55, 'season_id': 43, 'name': 'UEFA Euro 2020'},
        'copa_america_2024': {'competition_id': 223, 'season_id': 282, 'name': 'Copa America 2024'},
        'afcon_2023': {'competition_id': 1267, 'season_id': 107, 'name': 'Africa Cup of Nations 2023'},
        
        # Women's Competitions
        'womens_world_cup_2023': {'competition_id': 72, 'season_id': 107, 'name': "FIFA Women's World Cup 2023"},
        'womens_world_cup_2019': {'competition_id': 72, 'season_id': 30, 'name': "FIFA Women's World Cup 2019"},
        'fa_wsl_2023_24': {'competition_id': 37, 'season_id': 108, 'name': 'FA WSL 2023/24'},
        'nwsl_2018': {'competition_id': 49, 'season_id': 3, 'name': 'NWSL 2018'},
        
        # Club Competitions (Partial/Select Matches)
        'la_liga_2020_21': {'competition_id': 11, 'season_id': 90, 'name': 'La Liga 2020/21 (Partial)'},
        'la_liga_2019_20': {'competition_id': 11, 'season_id': 42, 'name': 'La Liga 2019/20 (Partial)'},
        'premier_league_2003_04': {'competition_id': 2, 'season_id': 44, 'name': 'Premier League 2003/04'},
        'champions_league_2018_19': {'competition_id': 16, 'season_id': 4, 'name': 'Champions League 2018/19 (Select)'},
    }
    
    def __init__(self, use_statsbombpy: bool = True):
        """
        Initialize the StatsBomb Open Data loader.
        
        Args:
            use_statsbombpy: If True, use statsbombpy library (recommended).
                           If False, load directly from GitHub JSON files.
        
        NO CREDENTIALS OR AUTHENTICATION REQUIRED for open data.
        """
        self.use_statsbombpy = use_statsbombpy and STATSBOMB_AVAILABLE
        
        if not self.use_statsbombpy:
            print("Using direct GitHub access for StatsBomb open data.")
        
        self.competitions_df = None
        self.matches_df = None
        self.events_cache = {}
        self.team_stats: Dict[str, TeamStats] = {}
    
    @classmethod
    def get_free_competitions(cls) -> Dict[str, Dict]:
        """
        Get dictionary of verified free/open competitions.
        
        Returns:
            Dictionary with competition keys and their IDs/names
        """
        return cls.FREE_COMPETITIONS.copy()
    
    @classmethod
    def list_free_competitions(cls) -> None:
        """Print available free competitions."""
        print("\n" + "="*60)
        print("  STATSBOMB FREE OPEN DATA - Available Competitions")
        print("  No credentials or authentication required!")
        print("="*60)
        
        categories = {
            'International Tournaments': ['world_cup', 'euro', 'copa_america', 'afcon'],
            "Women's Competitions": ['womens_world_cup', 'fa_wsl', 'nwsl'],
            'Club Competitions (Partial)': ['la_liga', 'premier_league', 'champions_league'],
        }
        
        for category, prefixes in categories.items():
            print(f"\n  {category}:")
            print("  " + "-"*50)
            for key, info in cls.FREE_COMPETITIONS.items():
                if any(key.startswith(p) for p in prefixes):
                    print(f"    • {info['name']}")
                    print(f"      Key: '{key}' | ID: ({info['competition_id']}, {info['season_id']})")
        print()
    
    def _load_from_github(self, endpoint: str) -> Any:
        """
        Load JSON data directly from StatsBomb GitHub repository.
        
        Args:
            endpoint: Path after base URL (e.g., 'competitions.json')
            
        Returns:
            Parsed JSON data
        """
        url = f"{GITHUB_BASE_URL}/{endpoint}"
        try:
            with urllib.request.urlopen(url) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            raise ConnectionError(f"Failed to load from GitHub: {url}\nError: {e}")
        
    def get_available_competitions(self) -> pd.DataFrame:
        """
        Get list of available competitions from StatsBomb open data.
        
        Note: Only returns competitions available in free/open data.
        """
        if self.competitions_df is None:
            if self.use_statsbombpy:
                # statsbombpy automatically uses open data without credentials
                self.competitions_df = sb.competitions()
            else:
                # Direct GitHub access
                data = self._load_from_github("competitions.json")
                self.competitions_df = pd.DataFrame(data)
        return self.competitions_df
    
    def get_seasons_for_competition(self, competition_id: int) -> List[Dict]:
        """Get available seasons for a competition."""
        comps = self.get_available_competitions()
        seasons = comps[comps['competition_id'] == competition_id]
        return seasons[['season_id', 'season_name']].to_dict('records')
    
    def load_matches(
        self, 
        competition_id: int, 
        season_id: int
    ) -> pd.DataFrame:
        """
        Load all matches for a competition and season from OPEN DATA.
        
        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID
            
        Returns:
            DataFrame with match information
            
        Note: Only works with free/open data competitions.
        """
        if self.use_statsbombpy:
            self.matches_df = sb.matches(
                competition_id=competition_id, 
                season_id=season_id
            )
        else:
            # Direct GitHub access
            endpoint = f"matches/{competition_id}/{season_id}.json"
            data = self._load_from_github(endpoint)
            self.matches_df = pd.DataFrame(data)
            
            # Normalize team names from nested structure
            if 'home_team' in self.matches_df.columns:
                if isinstance(self.matches_df['home_team'].iloc[0], dict):
                    self.matches_df['home_team'] = self.matches_df['home_team'].apply(
                        lambda x: x.get('home_team_name', x) if isinstance(x, dict) else x
                    )
                    self.matches_df['away_team'] = self.matches_df['away_team'].apply(
                        lambda x: x.get('away_team_name', x) if isinstance(x, dict) else x
                    )
        
        return self.matches_df
    
    def load_matches_by_key(self, competition_key: str) -> pd.DataFrame:
        """
        Load matches using a friendly competition key.
        
        Args:
            competition_key: Key from FREE_COMPETITIONS (e.g., 'world_cup_2022')
            
        Returns:
            DataFrame with match information
        """
        if competition_key not in self.FREE_COMPETITIONS:
            available = list(self.FREE_COMPETITIONS.keys())
            raise ValueError(
                f"Unknown competition key: '{competition_key}'\n"
                f"Available keys: {available}"
            )
        
        info = self.FREE_COMPETITIONS[competition_key]
        print(f"Loading: {info['name']}")
        return self.load_matches(info['competition_id'], info['season_id'])
    
    def load_match_events(self, match_id: int) -> pd.DataFrame:
        """
        Load all events for a specific match from OPEN DATA.
        
        Args:
            match_id: StatsBomb match ID
            
        Returns:
            DataFrame with all match events
        """
        if match_id not in self.events_cache:
            if self.use_statsbombpy:
                self.events_cache[match_id] = sb.events(match_id=match_id)
            else:
                # Direct GitHub access
                endpoint = f"events/{match_id}.json"
                data = self._load_from_github(endpoint)
                self.events_cache[match_id] = pd.DataFrame(data)
        return self.events_cache[match_id]
    
    def calculate_match_xg(self, match_id: int) -> Dict[str, float]:
        """
        Calculate expected goals for each team from match events.
        
        Args:
            match_id: StatsBomb match ID
            
        Returns:
            Dictionary with home_xg and away_xg
        """
        events = self.load_match_events(match_id)
        
        # Filter for shots
        shots = events[events['type'] == 'Shot'].copy()
        
        if shots.empty or 'shot_statsbomb_xg' not in shots.columns:
            return {'home_xg': 0.0, 'away_xg': 0.0}
        
        # Get match info
        match = self.matches_df[self.matches_df['match_id'] == match_id].iloc[0]
        home_team = match['home_team']
        away_team = match['away_team']
        
        # Sum xG by team
        home_shots = shots[shots['team'] == home_team]
        away_shots = shots[shots['team'] == away_team]
        
        home_xg = home_shots['shot_statsbomb_xg'].sum() if not home_shots.empty else 0.0
        away_xg = away_shots['shot_statsbomb_xg'].sum() if not away_shots.empty else 0.0
        
        return {
            'home_xg': home_xg,
            'away_xg': away_xg,
            'home_shots': len(home_shots),
            'away_shots': len(away_shots),
        }
    
    def calculate_match_stats(self, match_id: int) -> Dict:
        """
        Calculate comprehensive statistics for a match.
        
        Args:
            match_id: StatsBomb match ID
            
        Returns:
            Dictionary with detailed match statistics
        """
        events = self.load_match_events(match_id)
        match = self.matches_df[self.matches_df['match_id'] == match_id].iloc[0]
        
        home_team = match['home_team']
        away_team = match['away_team']
        
        stats = {
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': match['home_score'],
            'away_score': match['away_score'],
            'match_date': match['match_date'],
        }
        
        # Shots
        shots = events[events['type'] == 'Shot']
        if not shots.empty:
            home_shots = shots[shots['team'] == home_team]
            away_shots = shots[shots['team'] == away_team]
            
            stats['home_shots'] = len(home_shots)
            stats['away_shots'] = len(away_shots)
            
            if 'shot_statsbomb_xg' in shots.columns:
                stats['home_xg'] = home_shots['shot_statsbomb_xg'].sum()
                stats['away_xg'] = away_shots['shot_statsbomb_xg'].sum()
            
            # Shots on target
            if 'shot_outcome' in shots.columns:
                stats['home_shots_on_target'] = len(
                    home_shots[home_shots['shot_outcome'].isin(['Goal', 'Saved'])]
                )
                stats['away_shots_on_target'] = len(
                    away_shots[away_shots['shot_outcome'].isin(['Goal', 'Saved'])]
                )
        
        # Passes
        passes = events[events['type'] == 'Pass']
        if not passes.empty:
            home_passes = passes[passes['team'] == home_team]
            away_passes = passes[passes['team'] == away_team]
            
            stats['home_passes'] = len(home_passes)
            stats['away_passes'] = len(away_passes)
            
            # Completed passes (no 'pass_outcome' means successful)
            if 'pass_outcome' in passes.columns:
                stats['home_passes_completed'] = len(
                    home_passes[home_passes['pass_outcome'].isna()]
                )
                stats['away_passes_completed'] = len(
                    away_passes[away_passes['pass_outcome'].isna()]
                )
        
        # Possession (estimated from events)
        total_events = len(events[events['team'].isin([home_team, away_team])])
        if total_events > 0:
            home_events = len(events[events['team'] == home_team])
            stats['home_possession'] = (home_events / total_events) * 100
            stats['away_possession'] = 100 - stats['home_possession']
        
        return stats
    
    def build_team_stats(
        self, 
        competition_id: int, 
        season_id: int,
        calculate_xg: bool = True,
        verbose: bool = True
    ) -> Dict[str, TeamStats]:
        """
        Build comprehensive team statistics from a full season.
        
        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID
            calculate_xg: Whether to load events and calculate xG (slower)
            verbose: Print progress
            
        Returns:
            Dictionary mapping team names to TeamStats objects
        """
        # Load matches
        matches = self.load_matches(competition_id, season_id)
        
        if verbose:
            print(f"Processing {len(matches)} matches...")
        
        # Initialize team stats
        self.team_stats = {}
        
        # Initialize Elo ratings
        elo_ratings = {}
        K_FACTOR = 32
        HOME_ADVANTAGE = 100
        
        # Sort matches by date
        matches = matches.sort_values('match_date')
        
        for idx, match in matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_score = match['home_score']
            away_score = match['away_score']
            
            # Initialize teams if needed
            for team in [home_team, away_team]:
                if team not in self.team_stats:
                    self.team_stats[team] = TeamStats(name=team)
                    elo_ratings[team] = 1500.0
            
            home_stats = self.team_stats[home_team]
            away_stats = self.team_stats[away_team]
            
            # Update basic stats
            home_stats.matches_played += 1
            away_stats.matches_played += 1
            home_stats.goals_scored += home_score
            away_stats.goals_scored += away_score
            home_stats.goals_conceded += away_score
            away_stats.goals_conceded += home_score
            
            # Update wins/draws/losses and form
            if home_score > away_score:
                home_stats.wins += 1
                away_stats.losses += 1
                home_stats.recent_form.append('W')
                away_stats.recent_form.append('L')
                actual_home = 1.0
            elif home_score < away_score:
                home_stats.losses += 1
                away_stats.wins += 1
                home_stats.recent_form.append('L')
                away_stats.recent_form.append('W')
                actual_home = 0.0
            else:
                home_stats.draws += 1
                away_stats.draws += 1
                home_stats.recent_form.append('D')
                away_stats.recent_form.append('D')
                actual_home = 0.5
            
            # Update Elo ratings
            home_elo = elo_ratings[home_team]
            away_elo = elo_ratings[away_team]
            
            expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - HOME_ADVANTAGE) / 400))
            
            # Goal difference multiplier
            goal_diff = abs(home_score - away_score)
            multiplier = np.log(goal_diff + 1) + 1 if goal_diff > 0 else 1
            
            elo_change = K_FACTOR * multiplier * (actual_home - expected_home)
            elo_ratings[home_team] += elo_change
            elo_ratings[away_team] -= elo_change
            
            # Calculate xG if requested
            if calculate_xg:
                try:
                    match_stats = self.calculate_match_stats(match['match_id'])
                    
                    if 'home_xg' in match_stats:
                        home_stats.xg_for += match_stats['home_xg']
                        away_stats.xg_against += match_stats['home_xg']
                        away_stats.xg_for += match_stats['away_xg']
                        home_stats.xg_against += match_stats['away_xg']
                    
                    if 'home_shots' in match_stats:
                        home_stats.shots_for += match_stats.get('home_shots', 0)
                        away_stats.shots_for += match_stats.get('away_shots', 0)
                        home_stats.shots_against += match_stats.get('away_shots', 0)
                        away_stats.shots_against += match_stats.get('home_shots', 0)
                    
                    if 'home_passes_completed' in match_stats:
                        home_stats.passes_completed += match_stats.get('home_passes_completed', 0)
                        away_stats.passes_completed += match_stats.get('away_passes_completed', 0)
                        home_stats.passes_attempted += match_stats.get('home_passes', 0)
                        away_stats.passes_attempted += match_stats.get('away_passes', 0)
                        
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not process match {match['match_id']}: {e}")
            
            if verbose and (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(matches)} matches...")
        
        # Store final Elo ratings
        for team, elo in elo_ratings.items():
            self.team_stats[team].elo_rating = elo
        
        if verbose:
            print(f"Completed! Built stats for {len(self.team_stats)} teams.")
        
        return self.team_stats
    
    def get_team_stats(self, team_name: str) -> Optional[TeamStats]:
        """Get statistics for a specific team."""
        return self.team_stats.get(team_name)
    
    def get_all_teams(self) -> List[str]:
        """Get list of all teams with statistics."""
        return sorted(self.team_stats.keys())
    
    def get_standings(self) -> pd.DataFrame:
        """Get league standings sorted by points."""
        data = [stats.to_dict() for stats in self.team_stats.values()]
        df = pd.DataFrame(data)
        return df.sort_values('points', ascending=False).reset_index(drop=True)
    
    def export_team_stats(self, filepath: str):
        """Export team statistics to JSON."""
        data = {name: stats.to_dict() for name, stats in self.team_stats.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_team_stats(self, filepath: str):
        """Load team statistics from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.team_stats = {}
        for name, stats_dict in data.items():
            stats = TeamStats(name=name)
            stats.matches_played = stats_dict.get('matches_played', 0)
            stats.wins = stats_dict.get('wins', 0)
            stats.draws = stats_dict.get('draws', 0)
            stats.losses = stats_dict.get('losses', 0)
            stats.goals_scored = stats_dict.get('goals_scored', 0)
            stats.goals_conceded = stats_dict.get('goals_conceded', 0)
            stats.xg_for = stats_dict.get('xg_for', 0.0)
            stats.xg_against = stats_dict.get('xg_against', 0.0)
            stats.elo_rating = stats_dict.get('elo_rating', 1500.0)
            stats.recent_form = stats_dict.get('recent_form', [])
            self.team_stats[name] = stats


def get_world_cup_stats(year: int = 2022, calculate_xg: bool = True) -> Dict[str, TeamStats]:
    """
    Load FIFA World Cup team statistics from FREE OPEN DATA.
    
    Args:
        year: World Cup year (2018 or 2022 available)
        calculate_xg: Whether to calculate xG from events
        
    Returns:
        Dictionary of team statistics
    """
    loader = StatsBombLoader()
    
    key_map = {2022: 'world_cup_2022', 2018: 'world_cup_2018'}
    
    if year not in key_map:
        raise ValueError(f"World Cup {year} not in free data. Available: {list(key_map.keys())}")
    
    info = loader.FREE_COMPETITIONS[key_map[year]]
    
    return loader.build_team_stats(
        competition_id=info['competition_id'],
        season_id=info['season_id'],
        calculate_xg=calculate_xg
    )


def get_euro_stats(year: int = 2024, calculate_xg: bool = True) -> Dict[str, TeamStats]:
    """
    Load UEFA Euro team statistics from FREE OPEN DATA.
    
    Args:
        year: Euro year (2020 or 2024 available)
        calculate_xg: Whether to calculate xG from events
        
    Returns:
        Dictionary of team statistics
    """
    loader = StatsBombLoader()
    
    key_map = {2024: 'euro_2024', 2020: 'euro_2020'}
    
    if year not in key_map:
        raise ValueError(f"Euro {year} not in free data. Available: {list(key_map.keys())}")
    
    info = loader.FREE_COMPETITIONS[key_map[year]]
    
    return loader.build_team_stats(
        competition_id=info['competition_id'],
        season_id=info['season_id'],
        calculate_xg=calculate_xg
    )


def get_copa_america_stats(calculate_xg: bool = True) -> Dict[str, TeamStats]:
    """
    Load Copa America 2024 team statistics from FREE OPEN DATA.
    
    Returns:
        Dictionary of team statistics
    """
    loader = StatsBombLoader()
    info = loader.FREE_COMPETITIONS['copa_america_2024']
    
    return loader.build_team_stats(
        competition_id=info['competition_id'],
        season_id=info['season_id'],
        calculate_xg=calculate_xg
    )


def get_afcon_stats(calculate_xg: bool = True) -> Dict[str, TeamStats]:
    """
    Load Africa Cup of Nations 2023 team statistics from FREE OPEN DATA.
    
    Returns:
        Dictionary of team statistics
    """
    loader = StatsBombLoader()
    info = loader.FREE_COMPETITIONS['afcon_2023']
    
    return loader.build_team_stats(
        competition_id=info['competition_id'],
        season_id=info['season_id'],
        calculate_xg=calculate_xg
    )


def list_available_data():
    """Print all available free competitions."""
    StatsBombLoader.list_free_competitions()
