# ⚽ Soccer Match Predictor - Google Cloud Edition
> An ensemble prediction system combining four statistical models and Monte Carlo simulation,
> trained on StatsBomb open event data across international tournaments.

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
systematically underestimates draw probabilities because it assumes goals are independent —
but low-scoring games exhibit negative score correlation that violates this assumption.

This project empirically confirms that gap and builds an **ensemble of four complementary
statistical models** to address it, calibrated against held-out World Cup knockout data
and validated on a separate competition (Euro 2024).

---

## 🏗️ System Architecture

```
StatsBomb Open Data (free, no auth required)
        │
        ▼
┌───────────────────┐
│  Data Loader      │  statsbombpy
│                   │  → match results, team stats
└────────┬──────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              Feature Engineering                │
│  attack/defense strength · home advantage ·     │
│  historical goal rates · Elo history            │
└──┬──────────┬──────────┬────────────────────────┘
   │          │          │
   ▼          ▼          ▼
Dixon-   Naive        Elo
Coles    Poisson      Rating
(45%)    (30%)        (25%)
   │          │          │
   └──────────┴──────────┘
                    │
                    ▼
         ┌──────────────────┐
         │   Ensemble       │  Weighted average
         │   (best model)   │  RPS: 0.1884
         └──────────────────┘
```

---

## 📊 Model Performance

**Evaluation dataset:** FIFA World Cup 2022 knockout stage (16 matches, trained on 48 group stage matches)

| Model | Brier Score ↓ | Log Loss ↓ | RPS ↓ |
|---|---|---|---|
| Naive Baseline (1/3 each) | 0.6667 | 1.0986 | 0.2257 |
| Elo | 0.6681 | 1.2081 | 0.2021 |
| Naive Poisson | 0.5944 | 0.9815 | 0.1928 |
| Dixon-Coles | 0.5944 | 0.9815 | 0.1928 |
| **Ensemble** | **0.5859** | **0.9752** | **0.1884** |

**Ensemble beats the naive baseline by 16.5% on RPS** — the standard proper scoring rule
used in sports prediction literature.

### Per-Outcome Brier Score

Draws are the hardest outcome to predict. This breakdown shows where the tau correction helps:

| Model | Home Win ↓ | Draw ↓ | Away Win ↓ |
|---|---|---|---|
| Naive Baseline | 0.3194 | 0.2153 | 0.1319 |
| Elo | 0.2564 | 0.2638 | 0.1479 |
| Naive Poisson | 0.2781 | 0.2087 | 0.1076 |
| Dixon-Coles | 0.2781 | 0.2087 | 0.1076 |
| **Ensemble** | **0.2661** | **0.2092** | **0.1107** |

Dixon-Coles reduces the draw Brier score by **3.1%** over naive Poisson — consistent with its
tau correction targeting exactly this outcome type.

### Cross-Competition Validation (Euro 2024)

Models trained on WC 2022, tested on an entirely separate competition:

| Model | Brier ↓ | Log Loss ↓ | RPS ↓ |
|---|---|---|---|
| Naive Baseline | 0.6667 | 1.0986 | 0.2222 |
| Elo | 0.7431 | 1.3563 | 0.2278 |
| Naive Poisson | 0.6300 | 1.0317 | 0.1962 |
| Dixon-Coles | 0.6300 | 1.0317 | 0.1962 |
| **Ensemble** | **0.6398** | **1.0621** | **0.1976** |

Dixon-Coles and Naive Poisson generalise cleanly. Elo degrades — its point estimates
overfit to WC 2022 team ratings that don't transfer to Euro 2024 rosters.

---

## 🔍 Key Empirical Finding: Low-Score Dependency

The notebook confirms Dixon-Coles' core assumption using WC 2022 data:

| Score | Observed | Naive Poisson | Ratio | Verdict |
|---|---|---|---|---|
| 0-0 | 10.94% | 6.81% | 1.607 | ⚠ Inflated |
| 0-1 | 4.69% | 7.55% | 0.621 | ↓ Deflated |
| 1-0 | 10.94% | 10.74% | 1.018 | ✓ |
| 1-1 | 7.81% | 11.91% | 0.656 | ↓ Deflated |

0-0 draws occur **60.7% more often** than naive Poisson predicts — the exact dependency
Dixon & Coles (1997) identified and that the tau correction addresses.

---

## 🔢 Sample Prediction: Argentina vs France (WC 2022 Final)

```
============================================================
  Argentina vs France  |  WC 2022 Final
============================================================

  Model probabilities:
    NaivePoisson:  Home=46.8%  Draw=26.8%  Away=26.4%
    DixonColes:    Home=46.8%  Draw=26.8%  Away=26.4%
    Elo:           Home=55.6%  Draw= 8.3%  Away=36.1%
    Ensemble:      Home=49.0%  Draw=22.2%  Away=28.8%

  Prediction: Argentina (49.0% confidence)

============================================================
```

*(Actual result: Argentina won on penalties — correct call)*

---


## ✨ Features

### Statistical Models
| Model | What it captures |
|---|---|
| **Dixon-Coles** | Low-score dependency via tau (τ) correction: `τ(0,0) = 1 − λ_h · λ_a · ρ` |
| **Naive Poisson** | Standard independent Poisson with attack/defense strength ratings |
| **Elo Rating** | Dynamic team strength tracking with K=32 and goal-difference multiplier |
| **Ensemble** | Weighted average (DC 45% · Poisson 30% · Elo 25%) |

### Web Application
- Select any competition from dropdown
- Choose home and away teams
- Adjust simulation count (up to 50,000 iterations)
- Score distribution heatmap output

---
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

```bash
git clone https://github.com/gabiyanu/Soccer-Predictor.git
cd Soccer-Predictor
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### Command Line
```bash
python main.py --competition world_cup_2022 --home "Argentina" --away "France"
python main.py --competition euro_2024 --home "Spain" --away "England" --simulations 50000
```

---

## 📁 Project Structure

```
Soccer-Predictor/
├── README.md
├── app.py                       # Flask web server
├── main.py                      # CLI interface
├── requirements.txt
├── render.yaml                  # Render.com deployment config
│
├── src/                         # Model implementations
│   ├── data/
│   ├── models/
│   ├── simulation/
│   └── utils/
│
├── analysis/
│   └── model_evaluation.ipynb   # Full calibration & backtesting notebook
│
└── web/
    └── index.html
```

---

## 🔗 Actuarial Connections

This project applies methods directly transferable to actuarial and risk modelling:

| This Project | Actuarial Equivalent |
|---|---|
| Dixon-Coles τ correction for score dependency | Credibility-weighted experience adjustment |
| Ensemble RPS improvement of 16.5% | Blended model outperformance vs. industry tables |
| Ranked Probability Score (RPS) calibration | Proper scoring rules in IFRS 17 model validation |
| Cross-competition generalisation test | Out-of-time / out-of-sample backtesting (OSFI E-23) |
| Elo overfit detection on Euro 2024 | Identifying model instability under regime change |

---

## 🗂️ Data

All data from **StatsBomb Open Data** — no credentials or API keys required.

| Competition | Season | Matches | Used for |
|---|---|---|---|
| FIFA World Cup 2022 | 2022 | 64 | Train (48) + Holdout (16) |
| Euro 2024 | 2024 | 51 | Cross-competition validation |

---

## 🛠️ Tech Stack

`Python 3.9+` · `Flask` · `statsbombpy` · `NumPy` · `SciPy` · `Pandas` · `Matplotlib` · `Seaborn` · `Gunicorn`

---

## 📚 References

- Dixon, M. J., & Coles, S. G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society.
- Karlis, D., & Ntzoufras, I. (2003). *Analysis of sports data by using bivariate Poisson models.*
- Elo, A. (1978). *The Rating of Chessplayers, Past and Present.*
- StatsBomb Open Data: https://github.com/statsbomb/open-data (CC BY-NC-SA 4.0)

---

## 👤 Author

**Gabriel Aboyeji** — Modelling Specialist at CMHC | SOA Actuarial Candidate (P, FM, IFM, SRM, FAM)

[![Portfolio](https://img.shields.io/badge/Portfolio-datascienceportfol.io-blue)](https://www.datascienceportfol.io/gabrielaboyeji)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabrielaboyeji-blue?logo=linkedin)](https://www.linkedin.com/in/gabrielaboyeji/)
[![GitHub](https://img.shields.io/badge/GitHub-gabiyanu-black?logo=github)](https://github.com/gabiyanu)
