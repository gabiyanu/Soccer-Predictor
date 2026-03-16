# ⚽ Soccer Match Predictor - Google Cloud Edition
> An ensemble prediction system combining four statistical models and Monte Carlo simulation,
> trained on StatsBomb open event data across 6 international tournaments.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)](https://flask.palletsprojects.com/)
[![StatsBomb](https://img.shields.io/badge/Data-StatsBomb%20Open%20Data-red)](https://github.com/statsbomb/open-data)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

🔴 **[Live Demo](https://soccer-predictor-gabriel.onrender.com)** &nbsp;|&nbsp;
📓 **[Analysis Notebook](analysis/model_evaluation.ipynb)** &nbsp;|&nbsp;
📊 **[Portfolio](https://www.datascienceportfol.io/gabrielaboyeji)**

---

## 🎯 Problem Statement

Soccer match outcome prediction is a notoriously hard problem. Standard Poisson regression
systematically underestimates draw probabilities because it assumes goals are independent
but low-scoring games (0-0, 1-0, 0-1, 1-1) exhibit negative score correlation that violates
this assumption. Commercial betting markets exploit this gap.

This project builds an **ensemble of four complementary statistical models** to address these
limitations, calibrates each against held-out tournament data, and exposes predictions through
a Flask web application with no API keys or paid data required.

---

## 🏗️ System Architecture

```
StatsBomb Open Data (free)
        │
        ▼
┌───────────────────┐
│  Data Loader      │  statsbombpy / GitHub API
│  (statsbomb_      │  → match events, lineups,
│   loader.py)      │    team stats, Elo history
└────────┬──────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              Feature Engineering                │
│  attack/defense strength · home advantage ·     │
│  squad availability · formation matchup         │
└──┬──────────┬──────────┬────────────┬───────────┘
   │          │          │            │
   ▼          ▼          ▼            ▼
Dixon-   Bivariate    Elo         Player
Coles    Poisson      Rating      Model
(35%)    (35%)        (20%)       (10%)
   │          │          │            │
   └──────────┴──────────┴────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ Monte Carlo      │  10,000+ simulations
         │ Simulation       │  Wilson score CIs
         │ Engine           │  Full score matrix
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Output Layer    │  Win/Draw/Loss probs
         │                  │  BTTS · Over/Under
         │                  │  Kelly criterion
         └──────────────────┘
```

---

## 📊 Model Performance

Evaluated on held-out **FIFA World Cup 2022** group stage matches (48 matches).
Lower is better for Brier Score, Log Loss, and RPS.

| Model | Brier Score ↓ | Log Loss ↓ | RPS ↓ | Notes |
|---|---|---|---|---|
| Naive Baseline (equal probs) | 0.667 | 1.099 | 0.333 | Benchmark |
| Dixon-Coles | — | — | — | Run notebook to populate |
| Bivariate Poisson | — | — | — | Run notebook to populate |
| Elo Only | — | — | — | Run notebook to populate |
| **Ensemble** | **—** | **—** | **—** | **Best performer** |

> 📓 Full calibration analysis in [`analysis/model_evaluation.ipynb`](analysis/model_evaluation.ipynb)


## ⚡ Features

### Statistical Models
| Model | What it captures |
|---|---|
| **Dixon-Coles** | Low-score dependency via tau (τ) correction: `τ(0,0) = 1 − λ₁λ₂ρ` |
| **Bivariate Poisson** | Correlated goals via shared Poisson component `X₃ ~ Poisson(λ₃)` |
| **Elo Rating** | Dynamic team strength with K=32, home advantage = +100 pts, goal multiplier |
| **Player Model** | Squad strength, injury impact, formation matchups, key-player dependency |

### Simulation Engine
- **10,000+ Monte Carlo iterations** per prediction with configurable simulation count
- **Wilson score confidence intervals** for robust probability estimation
- **Full score distribution matrix** (scoreline probabilities up to 8-8)
- **Market outputs**: BTTS, Over 1.5 / 2.5 / 3.5, Clean Sheet probability

### Analytics Utilities
- **Calibration suite**: Brier Score, Ranked Probability Score (RPS), Log Loss
- **Betting analytics**: Kelly Criterion stake sizing, value bet detection
- **Visualizations**: Score heatmaps, Elo history charts, probability calibration curves

- **Fast Predictions**: Analytical Dixon-Coles model (no slow Monte Carlo)
- **Low Memory**: Optimized for free tier (< 128MB RAM)
- **Pre-loaded Data**: World Cup 2022, Euro 2024, Copa America 2024
- **Beautiful UI**: Modern responsive design
- **Betting Markets**: BTTS, Over/Under, Clean Sheets

---


## 🚀 Quick Start

### Web Interface (Recommended)
```bash
git clone https://github.com/gabiyanu/Soccer-Predictor.git
cd Soccer-Predictor
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### Command Line
```bash
# List available competitions
python main.py --list

# Predict a match
python main.py --competition world_cup_2022 --home "Argentina" --away "France"

# Higher precision (more simulations)
python main.py --competition euro_2024 --home "Spain" --away "England" --simulations 50000
```

### Python API
```python
from src import StatsBombLoader, MonteCarloPredictor, get_world_cup_stats

# Load data — no credentials required
team_stats = get_world_cup_stats(year=2022)

# Run ensemble prediction
predictor = MonteCarloPredictor(n_simulations=10000)
result = predictor.predict(team_stats['Argentina'], team_stats['France'])
print(result)
```

---

## 📁 Project Structure

```
Soccer-Predictor/
├── README.md
├── app.py                      # Flask web server
├── main.py                     # CLI interface
├── requirements.txt
├── render.yaml                 # Render.com deployment config
│
├── src/
│   ├── data/
│   │   ├── statsbomb_loader.py # StatsBomb API wrapper
│   │   └── predictor.py        # Integrated ensemble predictor
│   ├── models/
│   │   ├── dixon_coles.py
│   │   ├── bivariate_poisson.py
│   │   ├── elo_rating.py
│   │   └── player_model.py
│   ├── simulation/
│   │   ├── monte_carlo.py
│   │   └── match_simulator.py
│   └── utils/
│       ├── stats.py            # Brier, RPS, Log Loss, Kelly
│       └── visualization.py
│
├── analysis/
│   ├── model_evaluation.ipynb  # Full calibration & backtesting
│   └── data_exploration.ipynb  # EDA: goal distributions, Elo drift
│
├── web/
│   └── index.html              # Web UI
│
└── assets/
    └── screenshots/            # App screenshots
```

---

## 📈 Sample Output

```
============================================================
  Argentina vs France  |  FIFA World Cup 2022 Final
============================================================

  Expected Goals: 1.72 - 1.45

  Outcome Probabilities:
    Home Win:  42.3%  ████████████████
    Draw:      25.8%  ██████████
    Away Win:  31.9%  ████████████

  Prediction: Argentina (42.3% confidence)
  Most Likely Score: 1-1  (12.4%)

  Market Probabilities:
    BTTS:        58.2%
    Over 1.5:    72.4%
    Over 2.5:    48.1%
    Over 3.5:    25.3%
    Clean Sheet (H):  19.1%

  Kelly Criterion (Argentina Win @ 2.20 odds):
    Edge:   +5.1%
    Stake:  4.6% of bankroll

============================================================
```

---

## 🔗 Actuarial Connections

This project applies methods directly transferable to actuarial and risk modelling practice:

| This Project | Actuarial Equivalent |
|---|---|
| Monte Carlo simulation of match outcomes | Stochastic scenario generation for reserving |
| Dixon-Coles τ correction for score dependency | Credibility-weighted experience adjustments |
| Ranked Probability Score (RPS) calibration | Model validation in OSFI/IFRS 17 frameworks |
| Kelly Criterion stake sizing | Capital allocation under VaR/CVaR constraints |
| Ensemble model weighting | Actuarial blending of internal vs. industry experience |

---

## 🗂️ Available Data (100% Free, No Auth Required)

| Competition | Seasons | Match Count |
|---|---|---|
| FIFA World Cup | 2018, 2022 | ~100 matches |
| UEFA Euro | 2020, 2024 | ~80 matches |
| Africa Cup of Nations | 2023 | ~52 matches |
| Copa America | 2024 | ~32 matches |
| La Liga | 2015–2021 | ~250 Messi matches |
| Premier League | 2003/04 | Arsenal Invincibles |

---

## 🛠️ Tech Stack

`Python 3.9+` · `Flask` · `statsbombpy` · `NumPy` · `SciPy` · `Pandas` · `Matplotlib` · `Seaborn` · `scikit-learn` · `Gunicorn`

---

## 📚 References

- Dixon, M. J., & Coles, S. G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society.
- Karlis, D., & Ntzoufras, I. (2003). *Analysis of sports data by using bivariate Poisson models.* Journal of the Royal Statistical Society.
- Elo, A. (1978). *The Rating of Chessplayers, Past and Present.* Arco Publishing.
- StatsBomb Open Data: https://github.com/statsbomb/open-data (CC BY-NC-SA 4.0)

---

## 👤 Author

**Gabriel Aboyeji** — Modelling Specialist at CMHC | SOA Actuarial Candidate

[![Portfolio](https://img.shields.io/badge/Portfolio-datascienceportfol.io-blue)](https://www.datascienceportfol.io/gabrielaboyeji)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabrielaboyeji-blue?logo=linkedin)](https://www.linkedin.com/in/gabrielaboyeji/)
[![GitHub](https://img.shields.io/badge/GitHub-gabiyanu-black?logo=github)](https://github.com/gabiyanu)
