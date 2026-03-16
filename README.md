# ⚽ Soccer Match Predictor - Google Cloud Edition

A lightweight soccer match prediction web app using the Dixon-Coles statistical model.

**Optimized for free tier hosting** - No Monte Carlo simulations, uses fast analytical predictions.

## 🚀 Deploy to Google Cloud App Engine

### Prerequisites
1. Google Cloud account (free tier available)
2. Google Cloud SDK installed: https://cloud.google.com/sdk/docs/install

### Step-by-Step Deployment

#### Step 1: Create a Google Cloud Project
```bash
# Login to Google Cloud
gcloud auth login

# Create a new project (choose a unique ID)
gcloud projects create soccer-predictor-123 --name="Soccer Predictor"

# Set it as active project
gcloud config set project soccer-predictor-123

# Enable App Engine
gcloud app create --region=us-central
```

#### Step 2: Deploy the App
```bash
# Navigate to project folder
cd soccer-predictor-gcloud

# Deploy to App Engine
gcloud app deploy

# When prompted, type 'Y' to confirm
```

#### Step 3: Open Your App
```bash
# Open in browser
gcloud app browse
```

Your app will be live at: `https://soccer-predictor-123.uc.r.appspot.com`

---

## 🏃 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open http://localhost:8080
```

---

## 📁 Project Structure

```
soccer-predictor-gcloud/
├── app.py              # Flask application
├── app.yaml            # Google App Engine config
├── requirements.txt    # Python dependencies
└── web/
    └── index.html      # Web interface
```

---

## ⚡ Features

- **Fast Predictions**: Analytical Dixon-Coles model (no slow Monte Carlo)
- **Low Memory**: Optimized for free tier (< 128MB RAM)
- **Pre-loaded Data**: World Cup 2022, Euro 2024, Copa America 2024
- **Beautiful UI**: Modern responsive design
- **Betting Markets**: BTTS, Over/Under, Clean Sheets

---

## 💰 Google Cloud Free Tier

App Engine Free Tier includes:
- 28 instance hours per day
- 1 GB outbound data/day
- 5 GB Cloud Storage

This app easily fits within free limits for personal use!

---

## 🔧 Customization

### Add More Teams
Edit the `COMPETITIONS` dictionary in `app.py`:

```python
COMPETITIONS = {
    'your_competition': {
        'name': 'Your Competition Name',
        'teams': {
            'Team A': {'elo': 1600, 'attack': 1.1, 'defense': 0.9},
            'Team B': {'elo': 1550, 'attack': 1.0, 'defense': 1.0},
        }
    }
}
```

---

## 📊 How It Works

The Dixon-Coles model improves upon basic Poisson by:
1. Adjusting for low-scoring game correlation (0-0, 1-0, 0-1, 1-1)
2. Using Elo ratings to weight team strength
3. Applying home advantage factors

This gives more accurate predictions than simple Poisson models.
