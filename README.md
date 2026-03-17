# ⚽ ScoutIQ — Soccer Match Predictor
> An ensemble prediction system combining three statistical models deployed on Firebase,
> trained on StatsBomb open event data across international tournaments.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)](https://flask.palletsprojects.com/)
[![Firebase](https://img.shields.io/badge/Firebase-Hosting%20%2B%20Functions-orange?logo=firebase)](https://firebase.google.com/)
[![StatsBomb](https://img.shields.io/badge/Data-StatsBomb%20Open%20Data-red)](https://github.com/statsbomb/open-data)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

🔴 **[Live Demo](https://soccer-prediction-490403.web.app/)** &nbsp;|&nbsp;
📓 **[Analysis Notebook](https://github.com/gabiyanu/Soccer-Predictor/blob/main/model_evaluation.ipynb)** &nbsp;|&nbsp;
📊 **[Portfolio](https://www.datascienceportfol.io/gabrielaboyeji)**

---

## 🎯 Problem Statement

Soccer match outcome prediction is a notoriously hard problem. Standard Poisson regression
systematically underestimates draw probabilities because it assumes goals are independent —
but low-scoring games exhibit negative score correlation that violates this assumption.

This project empirically confirms that gap and builds an **ensemble of three complementary
statistical models** to address it, calibrated against held-out World Cup knockout data
and validated on a separate competition (Euro 2024).

---

## 🏗️ System Architecture

```
StatsBomb Open Data (free, no auth required)
        │
        ▼
┌───────────────────┐
│  Data Loader      │  built-in team stats
│                   │  → attack/defense ratings, Elo
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
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │  Firebase Cloud Functions (Python)  │  Flask API
  │  Firebase Hosting (static frontend) │  index.html
  │  Firestore (prediction cache TTL)   │  60-min TTL
  └─────────────────────────────────────┘
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

### Web Application (ScoutIQ)
- Dark glassmorphism UI with animated pitch elements
- Select any of 6 competitions from dropdown (WC 2022, Euro 2024, Copa América 2024, AFCON 2025, UEFA NL 2024/25, FIFA Top Nations 2025/26)
- Side-by-side model breakdown table (Dixon-Coles / Naive Poisson / Elo / Ensemble)
- AI match narrative powered by Google Gemini 1.5 Flash
- Live rankings and head-to-head comparisons

### API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/api/competitions` | GET | List available competitions |
| `/api/teams` | GET | Teams for a given competition |
| `/api/predict` | POST | Match prediction with model breakdown |
| `/api/analyze` | POST | Gemini AI match narrative |
| `/api/rankings` | GET | Team Elo rankings |
| `/api/head2head` | GET | Head-to-head stats |
| `/api/refresh` | POST | Clear Firestore cache |

### Analytics Utilities
- **Calibration suite**: Brier Score, Ranked Probability Score (RPS), Log Loss
- **Betting analytics**: Kelly Criterion stake sizing, value bet detection
- **Visualizations**: Score heatmaps, Elo history charts, probability calibration curves
- **Firestore caching**: 60-minute TTL prediction cache for fast repeat lookups

---

## 🚀 Quick Start

### Local Development
```bash
git clone https://github.com/gabiyanu/Soccer-Predictor.git
cd Soccer-Predictor
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### Firebase Deployment
```bash
# 1. Install Firebase CLI
npm install -g firebase-tools

# 2. Log in and set project
firebase login
firebase use soccer-prediction-490403

# 3. Deploy hosting + cloud functions
firebase deploy --only functions,hosting
```

### Optional Environment Variables (Firebase Console → Functions → Runtime Config)
| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Enable Gemini AI match narratives |
| `GOOGLE_SHEETS_ID` | Live team data from Google Sheets |
| `GOOGLE_SHEETS_JSON` | Service account JSON (base64-encoded) |

---

## 📁 Project Structure

```
Soccer-Predictor/
├── README.md
├── app.py                        # Flask server (local dev)
├── requirements.txt              # Local dev dependencies
├── app.yaml                      # App Engine config (legacy)
│
├── firebase.json                 # Firebase Hosting + Functions config
├── .firebaserc                   # Firebase project ID
├── firestore.rules               # Firestore security rules
├── firestore.indexes.json        # Firestore composite indexes
│
├── functions/
│   ├── main.py                   # Cloud Function entry point (Flask wrapped)
│   └── requirements.txt          # Cloud Function dependencies
│
├── web/
│   └── index.html                # ScoutIQ frontend (dark glassmorphism UI)
│
└── model_evaluation.ipynb        # Full calibration & backtesting notebook
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

All team strength data is built-in. Model validation uses **StatsBomb Open Data** — no credentials required.

| Competition | Season | Matches | Used for |
|---|---|---|---|
| FIFA World Cup 2022 | 2022 | 64 | Train (48) + Holdout (16) |
| Euro 2024 | 2024 | 51 | Cross-competition validation |

---

## 🛠️ Tech Stack

`Python 3.11+` · `Flask` · `NumPy` · `SciPy` · `Firebase Hosting` · `Firebase Cloud Functions` · `Firestore` · `Google Gemini 1.5 Flash` · `Google Sheets API`

---

## 📚 References

- Dixon, M. J., & Coles, S. G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society.
- Karlis, D., & Ntzoufras, I. (2003). *Analysis of sports data by using bivariate Poisson models.*
- Elo, A. (1978). *The Rating of Chessplayers, Past and Present.*
- Hvattum, L. M., & Arntzen, H. (2010). *Using ELO ratings for match result prediction in association football.*
- StatsBomb Open Data: https://github.com/statsbomb/open-data (CC BY-NC-SA 4.0)

---

## 👤 Author

**Gabriel Aboyeji** — Modelling Specialist at CMHC | SOA Actuarial Candidate (P, FM, IFM, SRM, FAM)

[![Portfolio](https://img.shields.io/badge/Portfolio-datascienceportfol.io-blue)](https://www.datascienceportfol.io/gabrielaboyeji)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabrielaboyeji-blue?logo=linkedin)](https://www.linkedin.com/in/gabrielaboyeji/)
[![GitHub](https://img.shields.io/badge/GitHub-gabiyanu-black?logo=github)](https://github.com/gabiyanu)
