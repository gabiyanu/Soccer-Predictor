# Soccer Match Predictor with StatsBomb Open Data

A comprehensive soccer match prediction system combining multiple statistical models with StatsBomb's free open data.

**NO CREDENTIALS OR AUTHENTICATION REQUIRED** - Uses 100% free open data.

## Features

### Statistical Models
- **Dixon-Coles**: Adjusted Poisson model with tau correction for low-scoring games
- **Bivariate Poisson**: Joint distribution accounting for score correlation
- **Elo Rating System**: Dynamic team strength tracking with history
- **Player Model**: Squad-level analysis with injuries, formations, key players

### Simulation Engine
- **Monte Carlo Simulation**: 10,000+ iteration stochastic simulations
- **Confidence Intervals**: Wilson score intervals for probability estimates
- **Score Distribution**: Full matrix of scoreline probabilities
- **Market Probabilities**: BTTS, Over/Under, Clean Sheets

### Statistical Utilities
- **Calibration Analysis**: Brier score, RPS, log loss
- **Betting Analytics**: Kelly criterion, value bet detection
- **Visualization**: Score matrices, Elo history, prediction reports

## Available Free Data (StatsBomb Open Data)

### International Tournaments (Full Match Coverage)
| Competition | Seasons | Competition ID |
|------------|---------|----------------|
| FIFA World Cup | 2018, 2022 | 43 |
| UEFA Euro | 2020, 2024 | 55 |
| Copa America | 2024 | 223 |
| Africa Cup of Nations | 2023 | 1267 |

### Women's Competitions
| Competition | Seasons | Competition ID |
|------------|---------|----------------|
| FIFA Women's World Cup | 2019, 2023 | 72 |
| FA Women's Super League | 2018-2024 | 37 |
| NWSL | 2018 | 49 |

### Club Competitions (Select Matches)
| Competition | Seasons | Notes |
|------------|---------|-------|
| La Liga | 2015-2021 | Messi matches |
| Premier League | 2003/04 | Arsenal Invincibles |
| Champions League | 2018/19 | Select matches |

## Installation

```bash
# Clone or download the project
cd soccer-predictor-statsbomb

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 🌐 Web Interface (Recommended)

The easiest way to use the predictor:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web server
python app.py
```

Then open **http://localhost:5000** in your browser!

![Web Interface](https://via.placeholder.com/800x400?text=Beautiful+Web+Interface)

Features:
- Select any competition from dropdown
- Choose home and away teams
- Adjust simulation count
- See beautiful visualizations of predictions

### 💻 Command Line Interface

```bash
# List available competitions
python main.py --list

# Predict a World Cup match
python main.py --competition world_cup_2022 --home "Argentina" --away "France"

# Predict Euro match with more simulations
python main.py --competition euro_2024 --home "Spain" --away "England" --simulations 50000
```

### Python API

```python
from src import StatsBombLoader, MonteCarloPredictor, get_world_cup_stats

# Load World Cup 2022 data
team_stats = get_world_cup_stats(year=2022)

# Get team statistics
argentina = team_stats['Argentina']
france = team_stats['France']

print(f"Argentina Elo: {argentina.elo_rating:.0f}")
print(f"France Elo: {france.elo_rating:.0f}")

# Make prediction
predictor = MonteCarloPredictor(n_simulations=10000)
result = predictor.predict(argentina, france)

print(result)  # Formatted prediction output
```

### Advanced: Using All Models

```python
from src.models import DixonColes, BivariatePoisson, EloRating, PlayerModel
from src.simulation import MatchSimulator, SimulationConfig, TeamData

# Configure simulation
config = SimulationConfig(
    n_simulations=20000,
    home_advantage=0.25,
    dixon_coles_rho=-0.13,
    model_weights={
        'dixon_coles': 0.35,
        'bivariate_poisson': 0.35,
        'elo': 0.20,
        'player_model': 0.10
    }
)

# Initialize simulator
simulator = MatchSimulator(config=config)

# Fit on historical data
simulator.fit_from_historical(match_data)

# Create team data
home = TeamData(name="Spain", elo=1850, attack_strength=1.15, defense_strength=0.85)
away = TeamData(name="England", elo=1820, attack_strength=1.10, defense_strength=0.90)

# Run simulation
result = simulator.simulate_match(home, away)

print(f"Home Win: {result.home_win_prob:.1%}")
print(f"Draw: {result.draw_prob:.1%}")
print(f"Away Win: {result.away_win_prob:.1%}")
print(f"Expected Goals: {result.expected_home_goals:.2f} - {result.expected_away_goals:.2f}")
```

## Project Structure

```
soccer-predictor-statsbomb/
├── app.py                     # Flask web server
├── main.py                    # Interactive CLI
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── web/
│   └── index.html            # Web interface
│
├── src/
│   ├── __init__.py           # Main exports
│   │
│   ├── data/
│   │   ├── statsbomb_loader.py  # StatsBomb data access
│   │   └── predictor.py         # Integrated predictor
│   │
│   ├── models/
│   │   ├── dixon_coles.py       # Dixon-Coles model
│   │   ├── bivariate_poisson.py # Bivariate Poisson model
│   │   ├── elo_rating.py        # Elo rating system
│   │   └── player_model.py      # Player/squad modeling
│   │
│   ├── simulation/
│   │   ├── monte_carlo.py       # MC simulation engine
│   │   └── match_simulator.py   # Ensemble simulator
│   │
│   └── utils/
│       ├── stats.py             # Statistical utilities
│       └── visualization.py     # Plotting functions
│
├── webapp/
│   └── StatsBombPredictor.jsx   # React UI component (alternative)
│
└── examples/
    └── basic_simulation.py      # Usage examples
```

## Model Details

### Dixon-Coles Model
Adjusts standard Poisson for observed dependency in low-scoring games using tau correction:
- τ(0,0) = 1 - λ₁λ₂ρ
- τ(0,1) = 1 + λ₁ρ
- τ(1,0) = 1 + λ₂ρ
- τ(1,1) = 1 - ρ

### Bivariate Poisson
Models correlation through shared component:
- X = X₁ + X₃
- Y = X₂ + X₃
Where X₃ ~ Poisson(λ₃) captures correlation

### Elo System
Dynamic ratings with:
- K-factor: 32 (configurable)
- Home advantage: 100 Elo points
- Goal difference multiplier
- Historical tracking

### Player Model
Squad-level factors:
- Position-weighted squad strength
- Injury/suspension impact
- Formation matchups
- Key player dependency

## Output Example

```
============================================================
  Argentina vs France
============================================================

  Expected Goals: 1.72 - 1.45

  Outcome Probabilities:
    Home Win:  42.3%  ████████
    Draw:      25.8%  █████
    Away Win:  31.9%  ██████

  Prediction: Argentina (42.3% confidence)
  Most Likely Score: 1-1

  Markets:
    BTTS:      58.2%
    Over 1.5:  72.4%
    Over 2.5:  48.1%
    Over 3.5:  25.3%

============================================================
```

## License

This project uses StatsBomb Open Data, available under CC BY-NC-SA 4.0 license.
See: https://github.com/statsbomb/open-data

## Data Source

All match data provided by [StatsBomb](https://statsbomb.com/).
Data accessed through the `statsbombpy` library or direct GitHub API.

## References

- Dixon, M. J., & Coles, S. G. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market.
- Karlis, D., & Ntzoufras, I. (2003). Analysis of sports data by using bivariate Poisson models.
- Elo, A. (1978). The Rating of Chessplayers, Past and Present.
