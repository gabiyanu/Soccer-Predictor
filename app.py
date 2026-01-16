"""
Soccer Predictor Web API

Flask backend serving predictions via REST API.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import StatsBombLoader, MonteCarloPredictor

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

# Cache for loaded data
data_cache = {}


def get_loader():
    """Get or create StatsBomb loader."""
    if 'loader' not in data_cache:
        data_cache['loader'] = StatsBombLoader()
    return data_cache['loader']


@app.route('/')
def serve_index():
    """Serve the main web page."""
    return send_from_directory('web', 'index.html')


@app.route('/api/competitions', methods=['GET'])
def get_competitions():
    """Get list of available competitions."""
    loader = get_loader()
    competitions = []
    
    for key, info in loader.FREE_COMPETITIONS.items():
        competitions.append({
            'key': key,
            'name': info['name'],
            'competition_id': info['competition_id'],
            'season_id': info['season_id']
        })
    
    return jsonify({'competitions': competitions})


@app.route('/api/teams/<competition_key>', methods=['GET'])
def get_teams(competition_key):
    """Get teams for a competition."""
    loader = get_loader()
    
    if competition_key not in loader.FREE_COMPETITIONS:
        return jsonify({'error': 'Competition not found'}), 404
    
    # Check cache
    cache_key = f'teams_{competition_key}'
    if cache_key in data_cache:
        team_stats = data_cache[cache_key]
    else:
        info = loader.FREE_COMPETITIONS[competition_key]
        try:
            team_stats = loader.build_team_stats(
                competition_id=info['competition_id'],
                season_id=info['season_id'],
                calculate_xg=False,
                verbose=False
            )
            data_cache[cache_key] = team_stats
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    teams = []
    for name, stats in sorted(team_stats.items()):
        teams.append({
            'name': name,
            'elo': round(stats.elo_rating),
            'matches': stats.matches_played,
            'wins': stats.wins,
            'draws': stats.draws,
            'losses': stats.losses,
            'goals_scored': stats.goals_scored,
            'goals_conceded': stats.goals_conceded,
            'attack_strength': round(stats.attack_strength, 2),
            'defense_strength': round(stats.defense_strength, 2)
        })
    
    return jsonify({'teams': teams})


@app.route('/api/predict', methods=['POST'])
def predict_match():
    """Run match prediction."""
    data = request.json
    
    competition_key = data.get('competition')
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    n_simulations = min(data.get('simulations', 2500), 10000)  # Cap at 10k for memory
    
    if not all([competition_key, home_team, away_team]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    loader = get_loader()
    
    # Get team stats from cache
    cache_key = f'teams_{competition_key}'
    if cache_key not in data_cache:
        return jsonify({'error': 'Please load teams first'}), 400
    
    team_stats = data_cache[cache_key]
    
    if home_team not in team_stats:
        return jsonify({'error': f'Home team "{home_team}" not found'}), 404
    if away_team not in team_stats:
        return jsonify({'error': f'Away team "{away_team}" not found'}), 404
    
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    # Run prediction with memory-efficient settings
    predictor = MonteCarloPredictor(n_simulations=n_simulations)
    result = predictor.predict(home_stats, away_stats)
    
    # Format response (don't store raw results to save memory)
    prediction = {
        'home_team': home_team,
        'away_team': away_team,
        'home_elo': round(home_stats.elo_rating),
        'away_elo': round(away_stats.elo_rating),
        'expected_goals': {
            'home': round(result.home_xg, 2),
            'away': round(result.away_xg, 2)
        },
        'probabilities': {
            'home_win': round(result.home_win_prob * 100, 1),
            'draw': round(result.draw_prob * 100, 1),
            'away_win': round(result.away_win_prob * 100, 1)
        },
        'most_likely_score': result.most_likely_score,
        'markets': {
            'btts': round(result.btts_prob * 100, 1),
            'over_1_5': round(result.over_1_5_prob * 100, 1),
            'over_2_5': round(result.over_2_5_prob * 100, 1),
            'over_3_5': round(result.over_3_5_prob * 100, 1),
            'clean_sheet_home': round(result.clean_sheet_home * 100, 1),
            'clean_sheet_away': round(result.clean_sheet_away * 100, 1)
        },
        'top_scores': dict(list(result.score_distribution.items())[:8]),
        'simulations': n_simulations
    }
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    
    return jsonify(prediction)


@app.route('/api/rankings/<competition_key>', methods=['GET'])
def get_rankings(competition_key):
    """Get team rankings for a competition."""
    loader = get_loader()
    
    cache_key = f'teams_{competition_key}'
    if cache_key not in data_cache:
        # Load if not cached
        if competition_key not in loader.FREE_COMPETITIONS:
            return jsonify({'error': 'Competition not found'}), 404
        
        info = loader.FREE_COMPETITIONS[competition_key]
        try:
            team_stats = loader.build_team_stats(
                competition_id=info['competition_id'],
                season_id=info['season_id'],
                calculate_xg=False,
                verbose=False
            )
            data_cache[cache_key] = team_stats
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        team_stats = data_cache[cache_key]
    
    # Sort by Elo
    rankings = sorted(
        team_stats.items(),
        key=lambda x: x[1].elo_rating,
        reverse=True
    )
    
    result = []
    for rank, (name, stats) in enumerate(rankings, 1):
        result.append({
            'rank': rank,
            'name': name,
            'elo': round(stats.elo_rating),
            'record': f"{stats.wins}-{stats.draws}-{stats.losses}",
            'gd': stats.goals_scored - stats.goals_conceded,
            'points': stats.wins * 3 + stats.draws
        })
    
    return jsonify({'rankings': result})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 50)
    print("  ⚽ Soccer Predictor Web Server")
    print("=" * 50)
    print(f"\n  Open http://localhost:{port} in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=port)
