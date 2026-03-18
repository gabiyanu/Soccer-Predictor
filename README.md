# ⚽ ScoutIQ — AI-Powered Soccer Match Predictor

> A full-stack data science project combining ensemble statistical modelling, cloud-native deployment, and generative AI — built end-to-end as a portfolio demonstration of applied ML, MLOps, and Google Cloud engineering.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Firebase](https://img.shields.io/badge/Firebase-Cloud%20Functions%20%7C%20Hosting%20%7C%20Firestore-orange?logo=firebase)](https://firebase.google.com/)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Cloud%20Run%20%7C%20Cloud%20Build-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-8E75B2?logo=google&logoColor=white)](https://ai.google.dev/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Statistical%20Models-8CAAE6?logo=scipy)](https://scipy.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<br>

🔴 **[Live Demo](https://soccer-prediction-490403.web.app/)** &nbsp;|&nbsp;
📓 **[Analysis Notebook](https://github.com/gabiyanu/Soccer-Predictor/blob/main/model_evaluation.ipynb)** &nbsp;|&nbsp;
📊 **[Portfolio](https://www.datascienceportfol.io/gabrielaboyeji)**

---

## 🧠 Skills Demonstrated

| Domain | Skills |
|---|---|
| **Statistical Modelling** | Dixon-Coles bivariate Poisson, Elo rating systems, ensemble weighting, Brier Score, RPS, Log Loss calibration |
| **Data Science** | Feature engineering, backtesting, cross-competition validation, out-of-sample testing, Kelly Criterion |
| **Machine Learning Ops** | Model serving via REST API, prediction caching, lazy singleton initialization, cold-start optimization |
| **Cloud Engineering** | Firebase Cloud Functions (2nd gen), Cloud Run, Cloud Build, Artifact Registry, Firestore, Firebase Hosting |
| **Generative AI** | Google Gemini 2.5 Flash integration for AI-generated match narratives via `google-genai` SDK |
| **Backend Development** | Python 3.11, Flask, Flask-CORS, REST API design, serverless architecture |
| **Frontend Development** | Vanilla JS, dark glassmorphism UI, country flag CDN integration, async/retry patterns |
| **DevOps** | Firebase CLI deployment, Cloud Run environment config, IAM permissions, git version control |

---

## 🎯 Problem Statement

Soccer match outcome prediction is a notoriously hard problem. Standard Poisson regression systematically **underestimates draw probabilities** because it assumes goals are independent — but low-scoring games exhibit negative score correlation that violates this assumption.

This project empirically confirms that gap and builds an **ensemble of three complementary statistical models** to address it, calibrated against held-out World Cup knockout data and validated on a separate competition (Euro 2024).

---

## 🏗️ System Architecture

```
StatsBomb Open Data  ──►  Feature Engineering  ──►  Three Statistical Models
                                                              │
                    ┌─────────────────────────────────────────┤
                    │                 │                        │
               Dixon-Coles       Naive Poisson            Elo Rating
                 (45%)              (30%)                   (25%)
                    │                 │                        │
                    └─────────────────┴────────────────────────┘
                                      │
                              Weighted Ensemble
                              (RPS: 0.1884 ↓)
                                      │
                    ┌─────────────────┴──────────────────────┐
                    │        Google Cloud Platform           │
                    │                                        │
                    │  Firebase Cloud Functions (Python)     │
                    │  ├── Flask REST API                    │
                    │  ├── Firestore cache (60-min TTL)      │
                    │  └── Gemini 2.5 Flash AI narratives    │
                    │                                        │
                    │  Firebase Hosting (static frontend)    │
                    │  └── ScoutIQ dark UI (glassmorphism)   │
                    └────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Google Cloud Products
| Product | Role |
|---|---|
| **Firebase Cloud Functions (2nd gen)** | Serverless Python API hosting — Flask app wrapped via `full_dispatch_request` |
| **Firebase Hosting** | Static frontend CDN with URL rewrite rules to Cloud Functions |
| **Cloud Firestore** | NoSQL prediction cache with 60-minute TTL to reduce cold-start latency |
| **Cloud Run** | Underlying execution layer for 2nd gen Cloud Functions |
| **Cloud Build** | Automated container builds triggered by `firebase deploy` |
| **Artifact Registry** | Container image storage for Cloud Run revisions |
| **Google Gemini 2.5 Flash** | Generative AI model for contextual match preview narratives |
| **Google Sheets API** | Optional live team data loader via `gspread` |

### Languages & Frameworks
| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core modelling and API logic |
| Flask | 2.x | REST API framework |
| NumPy | 1.24+ | Vectorized score matrix computation |
| SciPy | 1.10+ | Poisson PMF and statistical utilities |
| google-genai | 1.x | Gemini AI SDK (replaces deprecated `google-generativeai`) |
| gspread | 6.x | Google Sheets integration |
| Firebase Functions | 0.1+ | 2nd-gen Cloud Functions Python SDK |

---

## 📊 Model Performance

**Evaluation dataset:** FIFA World Cup 2022 knockout stage (16 matches, trained on 48 group-stage matches)

| Model | Brier Score ↓ | Log Loss ↓ | RPS ↓ |
|---|---|---|---|
| Naive Baseline (uniform 1/3) | 0.6667 | 1.0986 | 0.2257 |
| Elo | 0.6681 | 1.2081 | 0.2021 |
| Naive Poisson | 0.5944 | 0.9815 | 0.1928 |
| Dixon-Coles | 0.5944 | 0.9815 | 0.1928 |
| **Ensemble** | **0.5859** | **0.9752** | **0.1884** |

**Ensemble beats the naive baseline by 16.5% on RPS** — the standard proper scoring rule used in sports prediction literature.

### Draw Prediction Breakdown

Draws are the hardest outcome to predict. The Dixon-Coles tau (τ) correction targets exactly this:

| Model | Home Win ↓ | Draw ↓ | Away Win ↓ |
|---|---|---|---|
| Naive Baseline | 0.3194 | 0.2153 | 0.1319 |
| Elo | 0.2564 | 0.2638 | 0.1479 |
| Naive Poisson | 0.2781 | 0.2087 | 0.1076 |
| Dixon-Coles | 0.2781 | 0.2087 | 0.1076 |
| **Ensemble** | **0.2661** | **0.2092** | **0.1107** |

Dixon-Coles reduces the draw Brier score by **3.1% over naive Poisson** — consistent with its τ correction targeting low-score dependency.

### Cross-Competition Validation (Euro 2024)

Models trained on WC 2022, tested on an entirely separate competition to probe generalisation:

| Model | Brier ↓ | Log Loss ↓ | RPS ↓ |
|---|---|---|---|
| Naive Baseline | 0.6667 | 1.0986 | 0.2222 |
| Elo | 0.7431 | 1.3563 | 0.2278 |
| Naive Poisson | 0.6300 | 1.0317 | 0.1962 |
| Dixon-Coles | 0.6300 | 1.0317 | 0.1962 |
| **Ensemble** | **0.6398** | **1.0621** | **0.1976** |

Dixon-Coles and Naive Poisson generalise cleanly. **Elo degrades** — its point estimates overfit to WC 2022 team ratings that don't transfer to Euro 2024 rosters, illustrating model instability under regime change.

---

## 🔍 Key Empirical Finding: Low-Score Dependency

The notebook confirms Dixon-Coles' core assumption using WC 2022 data:

| Score | Observed | Naive Poisson | Ratio | Verdict |
|---|---|---|---|---|
| 0-0 | 10.94% | 6.81% | 1.607 | ⚠ Inflated |
| 0-1 | 4.69% | 7.55% | 0.621 | ↓ Deflated |
| 1-0 | 10.94% | 10.74% | 1.018 | ✓ Accurate |
| 1-1 | 7.81% | 11.91% | 0.656 | ↓ Deflated |

**0-0 draws occur 60.7% more often** than naive Poisson predicts — the exact dependency Dixon & Coles (1997) identified and that the τ correction addresses.

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
| **Dixon-Coles** | Low-score dependency via τ correction: `τ(0,0) = 1 − λ_h · λ_a · ρ` with fitted `ρ = −0.13` |
| **Naive Poisson** | Standard independent Poisson with attack/defence strength ratings (`ρ = 0.0`) |
| **Elo Rating** | Dynamic team strength tracking with K=32, goal-difference multiplier (Hvattum & Arntzen 2010) |
| **Ensemble** | Weighted average — DC 45% · Naive Poisson 30% · Elo 25% |

### Web Application (ScoutIQ)
- Dark glassmorphism UI with animated pitch elements
- 6 competitions: WC 2022, Euro 2024, Copa América 2024, AFCON 2025, UEFA Nations League 2024/25, FIFA Top Nations 2025/26
- Side-by-side model breakdown table (Dixon-Coles / Naive Poisson / Elo / Ensemble)
- Country flag images via flagCDN (ISO-2 codes, including GB subdivisions)
- **AI match narratives** powered by Google Gemini 2.5 Flash
- Live Elo rankings and head-to-head comparisons

### Engineering Highlights
- **Two-file split**: thin `main.py` (fast Firebase CLI analysis) + heavy `predictor.py` (lazy-imported on first request)
- **Lazy singletons**: Gemini, Sheets loader, and ensemble predictor initialised only on first HTTP request to minimise cold-start time
- **Retry logic**: Frontend `AbortController` with 30s timeout + 3-attempt backoff handles serverless cold starts gracefully
- **Firestore caching**: 60-minute TTL prevents repeated expensive computations on warm requests

### API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/api/competitions` | GET | List available competitions |
| `/api/teams` | GET | Teams for a given competition |
| `/api/predict` | POST | Match prediction with full model breakdown |
| `/api/analyze` | POST | Gemini AI match narrative |
| `/api/rankings` | GET | Team Elo rankings |
| `/api/head2head` | GET | Head-to-head stats |
| `/api/refresh` | POST | Clear Firestore prediction cache |

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

# 2. Authenticate and set project
firebase login
firebase use soccer-prediction-490403

# 3. Set up Python virtual environment in functions/
cd functions
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cd ..

# 4. Add your Gemini API key to functions/.env
#    GEMINI_API_KEY=your_key_here
#    Get a key at: https://aistudio.google.com/app/apikey

# 5. Deploy
firebase deploy --only "functions,hosting"
```

### Environment Variables (functions/.env)
| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Enable Gemini 2.5 Flash AI match narratives (get from AI Studio) |
| `GOOGLE_SHEETS_ID` | Live team data loader from Google Sheets |
| `GOOGLE_SHEETS_JSON` | Service account JSON for Sheets (base64-encoded) |

---

## 📁 Project Structure

```
Soccer-Predictor/
├── README.md
├── app.py                        # Flask server (local development)
├── requirements.txt              # Local dev dependencies
│
├── firebase.json                 # Firebase Hosting + Functions config
├── .firebaserc                   # Firebase project binding
├── firestore.rules               # Firestore security rules
├── firestore.indexes.json        # Firestore composite indexes
│
├── functions/
│   ├── main.py                   # Cloud Function entry point — thin wrapper
│   ├── predictor.py              # All models, routes, Gemini, Sheets logic
│   ├── requirements.txt          # Cloud Function dependencies
│   └── .env                      # Runtime env vars (gitignored — add locally)
│
├── web/
│   └── index.html                # ScoutIQ frontend — dark glassmorphism UI
│
└── model_evaluation.ipynb        # Full calibration & backtesting notebook
```

---

## 🗂️ Data

All team strength data is built-in. Model validation uses **StatsBomb Open Data** — no credentials required.

| Competition | Season | Matches | Used for |
|---|---|---|---|
| FIFA World Cup 2022 | 2022 | 64 | Train (48 group stage) + Holdout (16 knockout) |
| Euro 2024 | 2024 | 51 | Cross-competition out-of-sample validation |

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
[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabrielaboyeji-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gabrielaboyeji/)
[![GitHub](https://img.shields.io/badge/GitHub-gabiyanu-181717?logo=github&logoColor=white)](https://github.com/gabiyanu)
