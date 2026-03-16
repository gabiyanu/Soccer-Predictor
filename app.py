"""
Soccer Predictor Web API - Google Cloud Edition

Lightweight Flask backend using analytical predictions (no Monte Carlo).
Memory-efficient for free tier hosting.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
from scipy.stats import poisson
import os

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

# ============================================================
# LIGHTWEIGHT PREDICTION ENGINE (No Monte Carlo - saves memory)
# ============================================================

class DixonColesPredictor:
    """Analytical Dixon-Coles predictor - fast and memory efficient."""
    
    def __init__(self, rho=-0.13, home_advantage=0.25):
        self.rho = rho
        self.home_advantage = home_advantage
    
    def tau(self, home_goals, away_goals, lambda_home, lambda_away):
        """Tau correction for low-scoring games."""
        if home_goals == 0 and away_goals == 0:
            return 1.0 - lambda_home * lambda_away * self.rho
        elif home_goals == 0 and away_goals == 1:
            return 1.0 + lambda_home * self.rho
        elif home_goals == 1 and away_goals == 0:
            return 1.0 + lambda_away * self.rho
        elif home_goals == 1 and away_goals == 1:
            return 1.0 - self.rho
        return 1.0
    
    def score_probability(self, home_goals, away_goals, lambda_home, lambda_away):
        """Calculate probability of a specific scoreline."""
        tau = self.tau(home_goals, away_goals, lambda_home, lambda_away)
        return tau * poisson.pmf(home_goals, lambda_home) * poisson.pmf(away_goals, lambda_away)
    
    def predict(self, home_xg, away_xg, max_goals=8):
        """Generate full prediction analytically."""
        # Build score matrix
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                matrix[i, j] = self.score_probability(i, j, home_xg, away_xg)
        
        # Normalize
        matrix /= matrix.sum()
        
        # Outcomes
        home_win = np.sum(np.tril(matrix, k=-1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, k=1))
        
        # Most likely score
        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        
        # Markets
        btts = 1 - matrix[0, :].sum() - matrix[:, 0].sum() + matrix[0, 0]
        over_1_5 = sum(matrix[i, j] for i in range(max_goals+1) for j in range(max_goals+1) if i+j > 1.5)
        over_2_5 = sum(matrix[i, j] for i in range(max_goals+1) for j in range(max_goals+1) if i+j > 2.5)
        over_3_5 = sum(matrix[i, j] for i in range(max_goals+1) for j in range(max_goals+1) if i+j > 3.5)
        clean_home = matrix[:, 0].sum()
        clean_away = matrix[0, :].sum()
        
        # Top scores
        flat_idx = np.argsort(matrix.flatten())[::-1][:8]
        top_scores = {}
        for idx in flat_idx:
            i, j = np.unravel_index(idx, matrix.shape)
            top_scores[f"{i}-{j}"] = float(matrix[i, j])
        
        return {
            'home_win': float(home_win),
            'draw': float(draw),
            'away_win': float(away_win),
            'most_likely_score': f"{max_idx[0]}-{max_idx[1]}",
            'btts': float(btts),
            'over_1_5': float(over_1_5),
            'over_2_5': float(over_2_5),
            'over_3_5': float(over_3_5),
            'clean_sheet_home': float(clean_home),
            'clean_sheet_away': float(clean_away),
            'top_scores': top_scores
        }


# ============================================================
# TEAM DATA (Pre-computed to avoid loading StatsBomb)
# ============================================================

# Pre-computed team stats for major competitions
COMPETITIONS = {
    'world_cup_2022': {
        'name': 'FIFA World Cup 2022',
        'teams': {
            'Argentina': {'elo': 1770, 'attack': 1.35, 'defense': 0.85},
            'France': {'elo': 1755, 'attack': 1.40, 'defense': 0.90},
            'Croatia': {'elo': 1710, 'attack': 1.10, 'defense': 0.80},
            'Morocco': {'elo': 1680, 'attack': 0.95, 'defense': 0.70},
            'Brazil': {'elo': 1750, 'attack': 1.45, 'defense': 0.88},
            'Netherlands': {'elo': 1695, 'attack': 1.20, 'defense': 0.85},
            'England': {'elo': 1720, 'attack': 1.30, 'defense': 0.82},
            'Portugal': {'elo': 1705, 'attack': 1.25, 'defense': 0.88},
            'Spain': {'elo': 1715, 'attack': 1.28, 'defense': 0.85},
            'Germany': {'elo': 1680, 'attack': 1.22, 'defense': 0.95},
            'Japan': {'elo': 1620, 'attack': 1.05, 'defense': 0.90},
            'South Korea': {'elo': 1605, 'attack': 1.00, 'defense': 0.92},
            'Australia': {'elo': 1560, 'attack': 0.95, 'defense': 1.05},
            'USA': {'elo': 1595, 'attack': 1.02, 'defense': 0.95},
            'Senegal': {'elo': 1615, 'attack': 1.05, 'defense': 0.88},
            'Switzerland': {'elo': 1640, 'attack': 1.08, 'defense': 0.90},
            'Poland': {'elo': 1610, 'attack': 1.10, 'defense': 0.98},
            'Belgium': {'elo': 1680, 'attack': 1.15, 'defense': 0.95},
            'Mexico': {'elo': 1590, 'attack': 1.00, 'defense': 1.00},
            'Uruguay': {'elo': 1645, 'attack': 1.12, 'defense': 0.92},
            'Denmark': {'elo': 1650, 'attack': 1.08, 'defense': 0.88},
            'Tunisia': {'elo': 1520, 'attack': 0.85, 'defense': 1.00},
            'Saudi Arabia': {'elo': 1480, 'attack': 0.88, 'defense': 1.08},
            'Ecuador': {'elo': 1560, 'attack': 0.98, 'defense': 0.98},
            'Iran': {'elo': 1535, 'attack': 0.90, 'defense': 0.95},
            'Wales': {'elo': 1545, 'attack': 0.92, 'defense': 1.02},
            'Ghana': {'elo': 1505, 'attack': 0.95, 'defense': 1.08},
            'Cameroon': {'elo': 1520, 'attack': 1.00, 'defense': 1.10},
            'Serbia': {'elo': 1575, 'attack': 1.05, 'defense': 1.02},
            'Canada': {'elo': 1500, 'attack': 0.92, 'defense': 1.15},
            'Costa Rica': {'elo': 1465, 'attack': 0.80, 'defense': 1.12},
            'Qatar': {'elo': 1440, 'attack': 0.75, 'defense': 1.20},
        }
    },
    'euro_2024': {
        'name': 'UEFA Euro 2024',
        'teams': {
            'Spain': {'elo': 1760, 'attack': 1.38, 'defense': 0.78},
            'England': {'elo': 1745, 'attack': 1.32, 'defense': 0.82},
            'France': {'elo': 1755, 'attack': 1.35, 'defense': 0.80},
            'Netherlands': {'elo': 1710, 'attack': 1.25, 'defense': 0.85},
            'Germany': {'elo': 1730, 'attack': 1.30, 'defense': 0.88},
            'Portugal': {'elo': 1720, 'attack': 1.28, 'defense': 0.85},
            'Switzerland': {'elo': 1665, 'attack': 1.12, 'defense': 0.88},
            'Austria': {'elo': 1640, 'attack': 1.15, 'defense': 0.92},
            'Turkey': {'elo': 1620, 'attack': 1.10, 'defense': 0.95},
            'Belgium': {'elo': 1680, 'attack': 1.18, 'defense': 0.90},
            'Italy': {'elo': 1700, 'attack': 1.20, 'defense': 0.85},
            'Denmark': {'elo': 1660, 'attack': 1.10, 'defense': 0.88},
            'Slovenia': {'elo': 1560, 'attack': 0.95, 'defense': 0.98},
            'Romania': {'elo': 1545, 'attack': 0.98, 'defense': 1.00},
            'Slovakia': {'elo': 1530, 'attack': 0.92, 'defense': 1.02},
            'Georgia': {'elo': 1510, 'attack': 0.90, 'defense': 1.05},
            'Ukraine': {'elo': 1590, 'attack': 1.02, 'defense': 0.95},
            'Poland': {'elo': 1610, 'attack': 1.08, 'defense': 0.98},
            'Czech Republic': {'elo': 1580, 'attack': 1.00, 'defense': 0.95},
            'Hungary': {'elo': 1555, 'attack': 0.95, 'defense': 0.98},
            'Scotland': {'elo': 1540, 'attack': 0.92, 'defense': 1.02},
            'Croatia': {'elo': 1695, 'attack': 1.15, 'defense': 0.88},
            'Albania': {'elo': 1485, 'attack': 0.85, 'defense': 1.08},
            'Serbia': {'elo': 1590, 'attack': 1.05, 'defense': 1.00},
        }
    },
    'copa_america_2024': {
        'name': 'Copa America 2024',
        'teams': {
            'Argentina': {'elo': 1780, 'attack': 1.42, 'defense': 0.78},
            'Colombia': {'elo': 1710, 'attack': 1.25, 'defense': 0.85},
            'Uruguay': {'elo': 1720, 'attack': 1.28, 'defense': 0.82},
            'Brazil': {'elo': 1740, 'attack': 1.35, 'defense': 0.88},
            'Venezuela': {'elo': 1580, 'attack': 1.00, 'defense': 0.98},
            'Ecuador': {'elo': 1605, 'attack': 1.05, 'defense': 0.95},
            'Mexico': {'elo': 1620, 'attack': 1.08, 'defense': 0.95},
            'Panama': {'elo': 1520, 'attack': 0.88, 'defense': 1.05},
            'USA': {'elo': 1625, 'attack': 1.10, 'defense': 0.92},
            'Canada': {'elo': 1560, 'attack': 0.95, 'defense': 1.00},
            'Chile': {'elo': 1595, 'attack': 1.02, 'defense': 0.98},
            'Peru': {'elo': 1565, 'attack': 0.95, 'defense': 1.00},
            'Paraguay': {'elo': 1545, 'attack': 0.92, 'defense': 1.02},
            'Bolivia': {'elo': 1420, 'attack': 0.75, 'defense': 1.18},
            'Costa Rica': {'elo': 1495, 'attack': 0.85, 'defense': 1.08},
            'Jamaica': {'elo': 1455, 'attack': 0.80, 'defense': 1.12},
        }
    }
}

# Global predictor
predictor = DixonColesPredictor()


# ============================================================
# API ROUTES
# ============================================================

@app.route('/')
def serve_index():
    """Serve the main web page."""
    return send_from_directory('web', 'index.html')


@app.route('/api/competitions', methods=['GET'])
def get_competitions():
    """Get list of available competitions."""
    comps = [
        {'key': key, 'name': info['name']}
        for key, info in COMPETITIONS.items()
    ]
    return jsonify({'competitions': comps})


@app.route('/api/teams/<competition_key>', methods=['GET'])
def get_teams(competition_key):
    """Get teams for a competition."""
    if competition_key not in COMPETITIONS:
        return jsonify({'error': 'Competition not found'}), 404
    
    teams = []
    for name, stats in COMPETITIONS[competition_key]['teams'].items():
        teams.append({
            'name': name,
            'elo': stats['elo'],
            'attack_strength': stats['attack'],
            'defense_strength': stats['defense']
        })
    
    # Sort by Elo
    teams.sort(key=lambda x: x['elo'], reverse=True)
    return jsonify({'teams': teams})


@app.route('/api/predict', methods=['POST'])
def predict_match():
    """Run match prediction."""
    data = request.json
    
    competition_key = data.get('competition')
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    
    if not all([competition_key, home_team, away_team]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    if competition_key not in COMPETITIONS:
        return jsonify({'error': 'Competition not found'}), 404
    
    teams = COMPETITIONS[competition_key]['teams']
    
    if home_team not in teams:
        return jsonify({'error': f'Team "{home_team}" not found'}), 404
    if away_team not in teams:
        return jsonify({'error': f'Team "{away_team}" not found'}), 404
    
    home = teams[home_team]
    away = teams[away_team]
    
    # Calculate expected goals
    league_avg = 1.35
    home_advantage = 1.25
    
    # Elo adjustment
    elo_diff = home['elo'] - away['elo']
    elo_factor = 1 + (elo_diff / 800)
    
    home_xg = league_avg * home['attack'] * (1 / away['defense']) * home_advantage
    home_xg *= max(0.7, min(1.4, elo_factor))
    
    away_xg = league_avg * away['attack'] * (1 / home['defense'])
    away_xg *= max(0.7, min(1.4, 2 - elo_factor))
    
    # Bounds
    home_xg = max(0.5, min(3.5, home_xg))
    away_xg = max(0.3, min(3.0, away_xg))
    
    # Get prediction
    result = predictor.predict(home_xg, away_xg)
    
    return jsonify({
        'home_team': home_team,
        'away_team': away_team,
        'home_elo': home['elo'],
        'away_elo': away['elo'],
        'expected_goals': {
            'home': round(home_xg, 2),
            'away': round(away_xg, 2)
        },
        'probabilities': {
            'home_win': round(result['home_win'] * 100, 1),
            'draw': round(result['draw'] * 100, 1),
            'away_win': round(result['away_win'] * 100, 1)
        },
        'most_likely_score': result['most_likely_score'],
        'markets': {
            'btts': round(result['btts'] * 100, 1),
            'over_1_5': round(result['over_1_5'] * 100, 1),
            'over_2_5': round(result['over_2_5'] * 100, 1),
            'over_3_5': round(result['over_3_5'] * 100, 1),
            'clean_sheet_home': round(result['clean_sheet_home'] * 100, 1),
            'clean_sheet_away': round(result['clean_sheet_away'] * 100, 1)
        },
        'top_scores': result['top_scores']
    })


@app.route('/api/rankings/<competition_key>', methods=['GET'])
def get_rankings(competition_key):
    """Get team rankings for a competition."""
    if competition_key not in COMPETITIONS:
        return jsonify({'error': 'Competition not found'}), 404
    
    teams = COMPETITIONS[competition_key]['teams']
    
    rankings = sorted(
        [{'name': name, 'elo': stats['elo'], 'attack': stats['attack'], 'defense': stats['defense']} 
         for name, stats in teams.items()],
        key=lambda x: x['elo'],
        reverse=True
    )
    
    for i, team in enumerate(rankings, 1):
        team['rank'] = i
    
    return jsonify({'rankings': rankings})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
