"""
Soccer Predictor — Flask application logic (v2.0)
==================================================
All heavy imports (numpy, scipy, flask, firebase_admin, google-generativeai,
gspread) live here, NOT in main.py. This file is imported lazily — only when
the first real HTTP request arrives — so the Firebase CLI's 10-second source-
analysis window is never exhausted by slow imports.

Exported symbol:
    flask_app  — fully configured Flask application (all /api/* routes registered)
"""

import os
import json
import hashlib
import logging
import time

import numpy as np
from scipy.stats import poisson

from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Firebase Admin SDK (lazy client) ──────────────────────────────────────────
import firebase_admin
from firebase_admin import firestore as firebase_firestore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy Firestore client — initialised on first cache hit/miss, not at import time.
_db = None

def _get_db():
    global _db
    if _db is None:
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        _db = firebase_firestore.client()
    return _db


# ══════════════════════════════════════════════════════════════════════════════
# FIRESTORE CACHE
# ══════════════════════════════════════════════════════════════════════════════
_CACHE_COLL = "predictions"
_CACHE_TTL  = 3600  # seconds


def _cache_get(key: str):
    try:
        doc = _get_db().collection(_CACHE_COLL).document(key).get()
        if doc.exists:
            data = doc.to_dict()
            if data.get("expires_at", 0) > time.time():
                return data.get("payload")
    except Exception as exc:
        logger.warning("Firestore get error: %s", exc)
    return None


def _cache_set(key: str, payload: dict):
    try:
        _get_db().collection(_CACHE_COLL).document(key).set({
            "payload":    payload,
            "expires_at": time.time() + _CACHE_TTL,
            "created_at": time.time(),
        })
    except Exception as exc:
        logger.warning("Firestore set error: %s", exc)


def _cache_clear():
    try:
        docs = _get_db().collection(_CACHE_COLL).stream()
        for doc in docs:
            doc.reference.delete()
    except Exception as exc:
        logger.warning("Firestore clear error: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE INTEGRATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_AVAILABLE = False
_gemini_model    = None
_gemini_init_done = False

def _init_gemini():
    """Lazily initialise Gemini so the env-var is read at first request,
    not at module-import time (important for Cloud Functions cold starts).
    Uses google-genai SDK (1.x) which supports current Gemini models."""
    global GEMINI_AVAILABLE, _gemini_model, _gemini_init_done
    if _gemini_init_done:
        return
    _gemini_init_done = True
    try:
        from google import genai as _genai
        _key = os.environ.get("GEMINI_API_KEY", "")
        if _key:
            _gemini_model = _genai.Client(api_key=_key)
            GEMINI_AVAILABLE = True
            logger.info("Gemini: enabled (google-genai 1.x)")
        else:
            logger.info("Gemini: GEMINI_API_KEY not set — AI narratives disabled")
    except ImportError:
        logger.info("google-genai not installed")

# ── Google Sheets ─────────────────────────────────────────────────────────────
SHEETS_AVAILABLE = False
try:
    import gspread
    from google.oauth2.service_account import Credentials as SACredentials
    SHEETS_AVAILABLE = True
except ImportError:
    logger.info("gspread not installed")


class GeminiAnalyzer:
    def generate(self, home_team, away_team, home_elo, away_elo,
                 xg_home, xg_away, probs, competition_name):
        _init_gemini()          # ensure env-var is read before first use
        if not GEMINI_AVAILABLE or _gemini_model is None:
            return None
        try:
            prompt = (
                f"You are a professional soccer analyst. "
                f"Write a concise, insightful 3-sentence match preview for "
                f"{home_team} vs {away_team} ({competition_name}).\n\n"
                f"Model stats:\n"
                f"  {home_team} Elo {home_elo}  |  {away_team} Elo {away_elo}\n"
                f"  Expected goals: {home_team} {xg_home:.2f} — {away_team} {xg_away:.2f}\n"
                f"  Win probabilities: {home_team} {probs['home_win']:.0%}, "
                f"Draw {probs['draw']:.0%}, {away_team} {probs['away_win']:.0%}\n\n"
                f"Structure: (1) competitive context, "
                f"(2) xG/Elo expectation, (3) prediction.\n"
                f"Tone: punchy, analytical, no fluff. Plain text only."
            )
            resp = _gemini_model.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return resp.text.strip()
        except Exception as exc:
            logger.warning("Gemini error: %s", exc)
            return None


class GoogleSheetsLoader:
    _SCOPES = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    def __init__(self):
        self._client   = None
        self._sheet_id = os.environ.get("GOOGLE_SHEETS_ID", "")

    def _get_client(self):
        if not SHEETS_AVAILABLE or not self._sheet_id:
            return None
        if self._client is not None:
            return self._client
        try:
            creds_json = os.environ.get("GOOGLE_SHEETS_JSON", "")
            if creds_json:
                info  = json.loads(creds_json)
                creds = SACredentials.from_service_account_info(info, scopes=self._SCOPES)
            else:
                path  = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
                creds = SACredentials.from_service_account_file(path, scopes=self._SCOPES)
            self._client = gspread.authorize(creds)
            logger.info("Sheets: connected to %s", self._sheet_id)
        except Exception as exc:
            logger.warning("Sheets auth failed: %s", exc)
        return self._client

    def load(self):
        client = self._get_client()
        if not client:
            return None
        try:
            spreadsheet = client.open_by_key(self._sheet_id)
            result = {}
            for ws in spreadsheet.worksheets():
                key  = ws.title.lower().replace(" ", "_")
                rows = ws.get_all_values()
                if len(rows) < 3:
                    continue
                comp_name = rows[0][0] if rows[0] else key
                teams = {}
                for row in rows[2:]:
                    if len(row) < 4 or not row[0].strip():
                        continue
                    try:
                        teams[row[0].strip()] = {
                            "elo":     int(float(row[1])),
                            "attack":  float(row[2]),
                            "defense": float(row[3]),
                        }
                    except (ValueError, IndexError):
                        continue
                if teams:
                    result[key] = {"name": comp_name, "teams": teams}
            return result or None
        except Exception as exc:
            logger.warning("Sheets load failed: %s", exc)
            return None


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL MODELS
# ══════════════════════════════════════════════════════════════════════════════

class PoissonPredictor:
    """
    Vectorised Poisson-based predictor.
    rho=0.0   → Naive Independent Poisson
    rho=-0.13 → Dixon-Coles with tau correction
    """

    def __init__(self, rho=0.0, max_goals=8):
        self.rho = rho
        self.max_goals = max_goals
        g = np.arange(max_goals + 1)
        gh, ga = np.meshgrid(g, g, indexing="ij")
        self._goal_sum = gh + ga

    def _matrix(self, lam_h, lam_a):
        g = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
        if self.rho != 0.0:
            r = self.rho
            m[0, 0] *= max(0.0, 1.0 - lam_h * lam_a * r)
            m[0, 1] *= max(0.0, 1.0 + lam_h * r)
            m[1, 0] *= max(0.0, 1.0 + lam_a * r)
            m[1, 1] *= max(0.0, 1.0 - r)
        total = m.sum()
        return m / total if total > 0 else m

    def predict(self, lam_h, lam_a):
        m  = self._matrix(lam_h, lam_a)
        gs = self._goal_sum
        home_win = float(np.tril(m, k=-1).sum())
        draw     = float(np.trace(m))
        away_win = float(np.triu(m, k=1).sum())
        btts       = float(1 - m[0, :].sum() - m[:, 0].sum() + m[0, 0])
        over_1_5   = float(m[gs > 1].sum())
        over_2_5   = float(m[gs > 2].sum())
        over_3_5   = float(m[gs > 3].sum())
        clean_home = float(m[:, 0].sum())
        clean_away = float(m[0, :].sum())
        flat    = m.flatten()
        top_idx = np.argpartition(flat, -8)[-8:]
        top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
        top_scores = {
            f"{i}-{j}": float(m[i, j])
            for i, j in (np.unravel_index(k, m.shape) for k in top_idx)
        }
        return dict(
            home_win=home_win, draw=draw, away_win=away_win,
            btts=btts, over_1_5=over_1_5, over_2_5=over_2_5,
            over_3_5=over_3_5, clean_sheet_home=clean_home,
            clean_sheet_away=clean_away, top_scores=top_scores,
        )


class EloPredictor:
    """Logistic Elo-based outcome probabilities. Ref: Hvattum & Arntzen (2010)."""

    def __init__(self, home_advantage_elo=50, draw_base=0.28):
        self.ha        = home_advantage_elo
        self.draw_base = draw_base

    def predict(self, home_elo, away_elo):
        adj  = (home_elo - away_elo) + self.ha
        E    = 1.0 / (1.0 + 10.0 ** (-adj / 400.0))
        draw = max(0.10, self.draw_base - 0.18 * abs(2.0 * E - 1.0))
        hw   = max(0.04, E - 0.5 * draw)
        aw   = max(0.04, 1.0 - hw - draw)
        t    = hw + draw + aw
        return dict(home_win=hw/t, draw=draw/t, away_win=aw/t)


class EnsemblePredictor:
    """
    Weighted ensemble: Dixon-Coles (45%) + Naive Poisson (30%) + Elo (25%).
    Markets sourced from Dixon-Coles (most accurate for scoreline detail).
    """

    WEIGHTS = {"dixon_coles": 0.45, "naive_poisson": 0.30, "elo": 0.25}

    def __init__(self):
        self._dc  = PoissonPredictor(rho=-0.13)
        self._np  = PoissonPredictor(rho=0.0)
        self._elo = EloPredictor()

    def predict(self, lam_h, lam_a, home_elo, away_elo):
        dc  = self._dc.predict(lam_h, lam_a)
        np_ = self._np.predict(lam_h, lam_a)
        elo = self._elo.predict(home_elo, away_elo)
        w   = self.WEIGHTS
        ens = {
            "home_win": w["dixon_coles"]*dc["home_win"]  + w["naive_poisson"]*np_["home_win"] + w["elo"]*elo["home_win"],
            "draw":     w["dixon_coles"]*dc["draw"]      + w["naive_poisson"]*np_["draw"]     + w["elo"]*elo["draw"],
            "away_win": w["dixon_coles"]*dc["away_win"]  + w["naive_poisson"]*np_["away_win"] + w["elo"]*elo["away_win"],
        }
        for key in ("btts","over_1_5","over_2_5","over_3_5","clean_sheet_home","clean_sheet_away","top_scores"):
            ens[key] = dc[key]
        breakdown = {
            "dixon_coles":   {k: dc[k]  for k in ("home_win","draw","away_win")},
            "naive_poisson": {k: np_[k] for k in ("home_win","draw","away_win")},
            "elo":           {k: elo[k] for k in ("home_win","draw","away_win")},
            "ensemble":      {k: ens[k] for k in ("home_win","draw","away_win")},
            "weights": w,
        }
        return ens, breakdown


# ══════════════════════════════════════════════════════════════════════════════
# COMPETITION DATA  (static fallback; overridden by Google Sheets if set)
# ══════════════════════════════════════════════════════════════════════════════

_STATIC_COMPETITIONS = {
    "world_cup_2022": {
        "name": "FIFA World Cup 2022",
        "teams": {
            "Argentina":    {"elo": 1770, "attack": 1.35, "defense": 0.85},
            "France":       {"elo": 1755, "attack": 1.40, "defense": 0.90},
            "Croatia":      {"elo": 1710, "attack": 1.10, "defense": 0.80},
            "Morocco":      {"elo": 1680, "attack": 0.95, "defense": 0.70},
            "Brazil":       {"elo": 1750, "attack": 1.45, "defense": 0.88},
            "Netherlands":  {"elo": 1695, "attack": 1.20, "defense": 0.85},
            "England":      {"elo": 1720, "attack": 1.30, "defense": 0.82},
            "Portugal":     {"elo": 1705, "attack": 1.25, "defense": 0.88},
            "Spain":        {"elo": 1715, "attack": 1.28, "defense": 0.85},
            "Germany":      {"elo": 1680, "attack": 1.22, "defense": 0.95},
            "Japan":        {"elo": 1620, "attack": 1.05, "defense": 0.90},
            "South Korea":  {"elo": 1605, "attack": 1.00, "defense": 0.92},
            "Australia":    {"elo": 1560, "attack": 0.95, "defense": 1.05},
            "USA":          {"elo": 1595, "attack": 1.02, "defense": 0.95},
            "Senegal":      {"elo": 1615, "attack": 1.05, "defense": 0.88},
            "Switzerland":  {"elo": 1640, "attack": 1.08, "defense": 0.90},
            "Poland":       {"elo": 1610, "attack": 1.10, "defense": 0.98},
            "Belgium":      {"elo": 1680, "attack": 1.15, "defense": 0.95},
            "Mexico":       {"elo": 1590, "attack": 1.00, "defense": 1.00},
            "Uruguay":      {"elo": 1645, "attack": 1.12, "defense": 0.92},
            "Denmark":      {"elo": 1650, "attack": 1.08, "defense": 0.88},
            "Tunisia":      {"elo": 1520, "attack": 0.85, "defense": 1.00},
            "Saudi Arabia": {"elo": 1480, "attack": 0.88, "defense": 1.08},
            "Ecuador":      {"elo": 1560, "attack": 0.98, "defense": 0.98},
            "Iran":         {"elo": 1535, "attack": 0.90, "defense": 0.95},
            "Wales":        {"elo": 1545, "attack": 0.92, "defense": 1.02},
            "Ghana":        {"elo": 1505, "attack": 0.95, "defense": 1.08},
            "Cameroon":     {"elo": 1520, "attack": 1.00, "defense": 1.10},
            "Serbia":       {"elo": 1575, "attack": 1.05, "defense": 1.02},
            "Canada":       {"elo": 1500, "attack": 0.92, "defense": 1.15},
            "Costa Rica":   {"elo": 1465, "attack": 0.80, "defense": 1.12},
            "Qatar":        {"elo": 1440, "attack": 0.75, "defense": 1.20},
        },
    },
    "euro_2024": {
        "name": "UEFA Euro 2024",
        "teams": {
            "Spain":          {"elo": 1760, "attack": 1.38, "defense": 0.78},
            "England":        {"elo": 1745, "attack": 1.32, "defense": 0.82},
            "France":         {"elo": 1755, "attack": 1.35, "defense": 0.80},
            "Netherlands":    {"elo": 1710, "attack": 1.25, "defense": 0.85},
            "Germany":        {"elo": 1730, "attack": 1.30, "defense": 0.88},
            "Portugal":       {"elo": 1720, "attack": 1.28, "defense": 0.85},
            "Switzerland":    {"elo": 1665, "attack": 1.12, "defense": 0.88},
            "Austria":        {"elo": 1640, "attack": 1.15, "defense": 0.92},
            "Turkey":         {"elo": 1620, "attack": 1.10, "defense": 0.95},
            "Belgium":        {"elo": 1680, "attack": 1.18, "defense": 0.90},
            "Italy":          {"elo": 1700, "attack": 1.20, "defense": 0.85},
            "Denmark":        {"elo": 1660, "attack": 1.10, "defense": 0.88},
            "Croatia":        {"elo": 1695, "attack": 1.15, "defense": 0.88},
            "Ukraine":        {"elo": 1590, "attack": 1.02, "defense": 0.95},
            "Poland":         {"elo": 1610, "attack": 1.08, "defense": 0.98},
            "Czech Republic": {"elo": 1580, "attack": 1.00, "defense": 0.95},
            "Romania":        {"elo": 1545, "attack": 0.98, "defense": 1.00},
            "Slovakia":       {"elo": 1530, "attack": 0.92, "defense": 1.02},
            "Hungary":        {"elo": 1555, "attack": 0.95, "defense": 0.98},
            "Scotland":       {"elo": 1540, "attack": 0.92, "defense": 1.02},
            "Slovenia":       {"elo": 1560, "attack": 0.95, "defense": 0.98},
            "Georgia":        {"elo": 1510, "attack": 0.90, "defense": 1.05},
            "Serbia":         {"elo": 1590, "attack": 1.05, "defense": 1.00},
            "Albania":        {"elo": 1485, "attack": 0.85, "defense": 1.08},
        },
    },
    "copa_america_2024": {
        "name": "Copa America 2024",
        "teams": {
            "Argentina": {"elo": 1780, "attack": 1.42, "defense": 0.78},
            "Colombia":  {"elo": 1710, "attack": 1.25, "defense": 0.85},
            "Uruguay":   {"elo": 1720, "attack": 1.28, "defense": 0.82},
            "Brazil":    {"elo": 1740, "attack": 1.35, "defense": 0.88},
            "Venezuela": {"elo": 1580, "attack": 1.00, "defense": 0.98},
            "Ecuador":   {"elo": 1605, "attack": 1.05, "defense": 0.95},
            "Mexico":    {"elo": 1620, "attack": 1.08, "defense": 0.95},
            "Panama":    {"elo": 1520, "attack": 0.88, "defense": 1.05},
            "USA":       {"elo": 1625, "attack": 1.10, "defense": 0.92},
            "Canada":    {"elo": 1560, "attack": 0.95, "defense": 1.00},
            "Chile":     {"elo": 1595, "attack": 1.02, "defense": 0.98},
            "Peru":      {"elo": 1565, "attack": 0.95, "defense": 1.00},
            "Paraguay":  {"elo": 1545, "attack": 0.92, "defense": 1.02},
            "Bolivia":   {"elo": 1420, "attack": 0.75, "defense": 1.18},
            "Costa Rica":{"elo": 1495, "attack": 0.85, "defense": 1.08},
            "Jamaica":   {"elo": 1455, "attack": 0.80, "defense": 1.12},
        },
    },
    "afcon_2025": {
        "name": "AFCON 2025 — Morocco",
        "teams": {
            "Morocco":           {"elo": 1695, "attack": 1.20, "defense": 0.78},
            "Ivory Coast":       {"elo": 1665, "attack": 1.18, "defense": 0.88},
            "Nigeria":           {"elo": 1640, "attack": 1.15, "defense": 0.90},
            "Senegal":           {"elo": 1650, "attack": 1.15, "defense": 0.85},
            "Egypt":             {"elo": 1620, "attack": 1.10, "defense": 0.88},
            "Algeria":           {"elo": 1635, "attack": 1.12, "defense": 0.90},
            "Mali":              {"elo": 1610, "attack": 1.08, "defense": 0.92},
            "Cameroon":          {"elo": 1605, "attack": 1.08, "defense": 0.95},
            "South Africa":      {"elo": 1590, "attack": 1.05, "defense": 0.95},
            "Tunisia":           {"elo": 1595, "attack": 1.05, "defense": 0.92},
            "DR Congo":          {"elo": 1580, "attack": 1.02, "defense": 0.98},
            "Ghana":             {"elo": 1570, "attack": 1.00, "defense": 1.00},
            "Guinea":            {"elo": 1545, "attack": 0.95, "defense": 1.00},
            "Cape Verde":        {"elo": 1530, "attack": 0.90, "defense": 1.02},
            "Angola":            {"elo": 1515, "attack": 0.88, "defense": 1.05},
            "Zambia":            {"elo": 1490, "attack": 0.85, "defense": 1.08},
            "Benin":             {"elo": 1470, "attack": 0.82, "defense": 1.08},
            "Equatorial Guinea": {"elo": 1480, "attack": 0.82, "defense": 1.08},
            "Zimbabwe":          {"elo": 1460, "attack": 0.80, "defense": 1.10},
            "Comoros":           {"elo": 1445, "attack": 0.75, "defense": 1.15},
            "Tanzania":          {"elo": 1430, "attack": 0.75, "defense": 1.15},
            "Mozambique":        {"elo": 1420, "attack": 0.72, "defense": 1.18},
            "Sudan":             {"elo": 1410, "attack": 0.70, "defense": 1.20},
            "Botswana":          {"elo": 1415, "attack": 0.72, "defense": 1.18},
        },
    },
    "nations_league_2025": {
        "name": "UEFA Nations League 2024/25 — League A",
        "teams": {
            "Spain":       {"elo": 1835, "attack": 1.43, "defense": 0.70},
            "France":      {"elo": 1790, "attack": 1.38, "defense": 0.78},
            "Germany":     {"elo": 1785, "attack": 1.36, "defense": 0.80},
            "Portugal":    {"elo": 1760, "attack": 1.32, "defense": 0.82},
            "Netherlands": {"elo": 1730, "attack": 1.28, "defense": 0.85},
            "Italy":       {"elo": 1720, "attack": 1.22, "defense": 0.82},
            "Croatia":     {"elo": 1690, "attack": 1.15, "defense": 0.88},
            "Denmark":     {"elo": 1660, "attack": 1.12, "defense": 0.88},
            "Switzerland": {"elo": 1660, "attack": 1.10, "defense": 0.87},
            "Belgium":     {"elo": 1665, "attack": 1.12, "defense": 0.92},
            "Poland":      {"elo": 1615, "attack": 1.08, "defense": 0.97},
            "Serbia":      {"elo": 1600, "attack": 1.05, "defense": 0.98},
            "Hungary":     {"elo": 1580, "attack": 0.98, "defense": 0.97},
            "Scotland":    {"elo": 1555, "attack": 0.95, "defense": 1.00},
            "Bosnia":      {"elo": 1525, "attack": 0.90, "defense": 1.02},
            "Israel":      {"elo": 1510, "attack": 0.88, "defense": 1.04},
        },
    },
    "fifa_top_nations_2026": {
        "name": "FIFA Top Nations 2025/26",
        "teams": {
            "Spain":        {"elo": 1840, "attack": 1.43, "defense": 0.70},
            "Argentina":    {"elo": 1825, "attack": 1.42, "defense": 0.75},
            "France":       {"elo": 1795, "attack": 1.38, "defense": 0.78},
            "Germany":      {"elo": 1790, "attack": 1.36, "defense": 0.80},
            "England":      {"elo": 1780, "attack": 1.33, "defense": 0.80},
            "Brazil":       {"elo": 1768, "attack": 1.35, "defense": 0.82},
            "Portugal":     {"elo": 1762, "attack": 1.30, "defense": 0.83},
            "Netherlands":  {"elo": 1733, "attack": 1.28, "defense": 0.85},
            "Italy":        {"elo": 1722, "attack": 1.22, "defense": 0.82},
            "Colombia":     {"elo": 1718, "attack": 1.25, "defense": 0.88},
            "Morocco":      {"elo": 1700, "attack": 1.20, "defense": 0.80},
            "Uruguay":      {"elo": 1712, "attack": 1.20, "defense": 0.85},
            "Croatia":      {"elo": 1688, "attack": 1.15, "defense": 0.88},
            "Japan":        {"elo": 1678, "attack": 1.18, "defense": 0.88},
            "Ivory Coast":  {"elo": 1665, "attack": 1.15, "defense": 0.88},
            "Belgium":      {"elo": 1665, "attack": 1.12, "defense": 0.92},
            "Switzerland":  {"elo": 1662, "attack": 1.10, "defense": 0.88},
            "Senegal":      {"elo": 1652, "attack": 1.12, "defense": 0.88},
            "Denmark":      {"elo": 1652, "attack": 1.10, "defense": 0.88},
            "Turkey":       {"elo": 1645, "attack": 1.12, "defense": 0.90},
            "USA":          {"elo": 1648, "attack": 1.12, "defense": 0.92},
            "Nigeria":      {"elo": 1642, "attack": 1.12, "defense": 0.92},
            "South Korea":  {"elo": 1638, "attack": 1.08, "defense": 0.90},
            "Mexico":       {"elo": 1628, "attack": 1.08, "defense": 0.95},
            "Egypt":        {"elo": 1612, "attack": 1.08, "defense": 0.90},
            "Ecuador":      {"elo": 1615, "attack": 1.05, "defense": 0.95},
            "Poland":       {"elo": 1618, "attack": 1.08, "defense": 0.97},
            "Australia":    {"elo": 1592, "attack": 1.02, "defense": 0.95},
            "Canada":       {"elo": 1582, "attack": 1.00, "defense": 0.98},
            "Saudi Arabia": {"elo": 1545, "attack": 0.92, "defense": 1.02},
            "Iran":         {"elo": 1548, "attack": 0.90, "defense": 1.00},
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETONS + HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Lazy singletons — created on first request to minimise cold-start time.
# Importing predictor.py (numpy, scipy, etc.) already takes several seconds;
# building the numpy goal grids inside EnsemblePredictor adds more. Deferring
# until the first actual HTTP request keeps cold-start under ~5 s.
_sheets_loader = None
_gemini        = None
_ensemble      = None


def _get_singletons():
    global _sheets_loader, _gemini, _ensemble
    if _ensemble is None:
        _sheets_loader = GoogleSheetsLoader()
        _gemini        = GeminiAnalyzer()
        _ensemble      = EnsemblePredictor()


def _get_competitions():
    _get_singletons()
    live = _sheets_loader.load()
    return live if live else _STATIC_COMPETITIONS


def _compute_xg(home, away, league_avg=1.35, home_advantage=1.25):
    """
    Dixon-Coles xG: lambda_home = mu * alpha_home * beta_away * gamma
                    lambda_away = mu * alpha_away * beta_home
    Smooth tanh Elo correction bounded ±25%.
    """
    lam_h = league_avg * home["attack"] * away["defense"] * home_advantage
    lam_a = league_avg * away["attack"] * home["defense"]
    elo_mult = 1.0 + float(np.tanh((home["elo"] - away["elo"]) / 600.0)) * 0.25
    lam_h   *= elo_mult
    lam_a   *= (2.0 - elo_mult)
    return float(np.clip(lam_h, 0.30, 4.0)), float(np.clip(lam_a, 0.20, 3.5))


# ══════════════════════════════════════════════════════════════════════════════
# FLASK APP + ROUTES
# ══════════════════════════════════════════════════════════════════════════════

flask_app = Flask(__name__)
CORS(flask_app)


@flask_app.route("/api/competitions", methods=["GET"])
def get_competitions():
    comps = _get_competitions()
    return jsonify({
        "competitions": [
            {"key": k, "name": v["name"], "team_count": len(v["teams"])}
            for k, v in comps.items()
        ]
    })


@flask_app.route("/api/teams/<competition_key>", methods=["GET"])
def get_teams(competition_key):
    comps = _get_competitions()
    if competition_key not in comps:
        return jsonify({"error": "Competition not found"}), 404
    teams = sorted(
        [{"name": n, "elo": s["elo"],
          "attack_strength": s["attack"], "defense_strength": s["defense"]}
         for n, s in comps[competition_key]["teams"].items()],
        key=lambda x: x["elo"], reverse=True,
    )
    return jsonify({"teams": teams})


@flask_app.route("/api/predict", methods=["POST"])
def predict_match():
    _get_singletons()
    data = request.json or {}
    ckey = data.get("competition")
    home = data.get("home_team")
    away = data.get("away_team")
    if not all([ckey, home, away]):
        return jsonify({"error": "Missing required fields"}), 400
    comps = _get_competitions()
    if ckey not in comps:
        return jsonify({"error": "Competition not found"}), 404
    teams = comps[ckey]["teams"]
    if home not in teams:
        return jsonify({"error": f'Team "{home}" not found'}), 404
    if away not in teams:
        return jsonify({"error": f'Team "{away}" not found'}), 404

    cache_key = hashlib.md5(f"{ckey}|{home}|{away}".encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached)

    hs, as_ = teams[home], teams[away]
    lam_h, lam_a = _compute_xg(hs, as_)
    ens, breakdown = _ensemble.predict(lam_h, lam_a, hs["elo"], as_["elo"])

    def pct(v): return round(v * 100, 1)

    bd_pct = {
        m: ({o: pct(p) for o, p in probs.items()} if isinstance(probs, dict) else probs)
        for m, probs in breakdown.items()
    }
    response = {
        "home_team": home, "away_team": away,
        "home_elo": hs["elo"], "away_elo": as_["elo"],
        "expected_goals": {"home": round(lam_h, 2), "away": round(lam_a, 2)},
        "probabilities": {
            "home_win": pct(ens["home_win"]),
            "draw":     pct(ens["draw"]),
            "away_win": pct(ens["away_win"]),
        },
        "markets": {
            "btts":             pct(ens["btts"]),
            "over_1_5":         pct(ens["over_1_5"]),
            "over_2_5":         pct(ens["over_2_5"]),
            "over_3_5":         pct(ens["over_3_5"]),
            "clean_sheet_home": pct(ens["clean_sheet_home"]),
            "clean_sheet_away": pct(ens["clean_sheet_away"]),
        },
        "top_scores":      ens["top_scores"],
        "model_breakdown": bd_pct,
        "competition_name": comps[ckey]["name"],
    }
    _cache_set(cache_key, response)
    return jsonify(response)


@flask_app.route("/api/analyze", methods=["POST"])
def analyze_match():
    _get_singletons()
    data = request.json or {}
    if "probabilities" in data:
        pred = data
    else:
        ckey = data.get("competition")
        home = data.get("home_team")
        away = data.get("away_team")
        if not all([ckey, home, away]):
            return jsonify({"error": "Missing required fields"}), 400
        comps = _get_competitions()
        if ckey not in comps or home not in comps[ckey]["teams"] or away not in comps[ckey]["teams"]:
            return jsonify({"error": "Invalid competition or team"}), 404
        hs, as_ = comps[ckey]["teams"][home], comps[ckey]["teams"][away]
        lam_h, lam_a = _compute_xg(hs, as_)
        ens, _ = _ensemble.predict(lam_h, lam_a, hs["elo"], as_["elo"])
        def pct(v): return round(v * 100, 1)
        pred = {
            "home_team": home, "away_team": away,
            "home_elo": hs["elo"], "away_elo": as_["elo"],
            "expected_goals": {"home": round(lam_h, 2), "away": round(lam_a, 2)},
            "probabilities": {"home_win": pct(ens["home_win"]), "draw": pct(ens["draw"]), "away_win": pct(ens["away_win"])},
            "competition_name": comps[ckey]["name"],
        }

    narrative = _gemini.generate(
        home_team=pred["home_team"], away_team=pred["away_team"],
        home_elo=pred["home_elo"],  away_elo=pred["away_elo"],
        xg_home=pred["expected_goals"]["home"],
        xg_away=pred["expected_goals"]["away"],
        probs={k: v/100 for k, v in pred["probabilities"].items()},
        competition_name=pred.get("competition_name", ""),
    )
    if narrative is None:
        p   = pred["probabilities"]
        ldr = pred["home_team"] if p["home_win"] > p["away_win"] and p["home_win"] > p["draw"] \
              else (pred["away_team"] if p["away_win"] > p["home_win"] else "either side")
        narrative = (
            f"{pred['home_team']} (Elo {pred['home_elo']}) host "
            f"{pred['away_team']} (Elo {pred['away_elo']}) with expected goals "
            f"{pred['expected_goals']['home']:.2f} – {pred['expected_goals']['away']:.2f}. "
            f"The ensemble gives {pred['home_team']} a {p['home_win']}% win probability, "
            f"{p['draw']}% draw, {pred['away_team']} {p['away_win']}%. "
            f"Model marginally favours {ldr}. "
            f"(Set GEMINI_API_KEY to enable AI-generated narratives.)"
        )
    return jsonify({"narrative": narrative, "ai_powered": GEMINI_AVAILABLE})


@flask_app.route("/api/rankings/<competition_key>", methods=["GET"])
def get_rankings(competition_key):
    comps = _get_competitions()
    if competition_key not in comps:
        return jsonify({"error": "Competition not found"}), 404
    ranked = sorted(
        [{"name": n, "elo": s["elo"], "attack": s["attack"], "defense": s["defense"]}
         for n, s in comps[competition_key]["teams"].items()],
        key=lambda x: x["elo"], reverse=True,
    )
    for i, t in enumerate(ranked, 1):
        t["rank"] = i
    return jsonify({"rankings": ranked})


@flask_app.route("/api/head2head", methods=["GET"])
def head2head():
    ckey = request.args.get("competition")
    home = request.args.get("home")
    away = request.args.get("away")
    if not all([ckey, home, away]):
        return jsonify({"error": "Provide competition, home, away query params"}), 400
    comps = _get_competitions()
    if ckey not in comps:
        return jsonify({"error": "Competition not found"}), 404
    teams = comps[ckey]["teams"]
    if home not in teams or away not in teams:
        return jsonify({"error": "One or both teams not found"}), 404
    hs, as_ = teams[home], teams[away]
    lam_h, lam_a = _compute_xg(hs, as_)
    ens, breakdown = _ensemble.predict(lam_h, lam_a, hs["elo"], as_["elo"])
    def pct(v): return round(v * 100, 1)
    return jsonify({
        "home_team": home, "away_team": away,
        "elo_diff": hs["elo"] - as_["elo"],
        "expected_goals": {"home": round(lam_h, 2), "away": round(lam_a, 2)},
        "probabilities": {"home_win": pct(ens["home_win"]), "draw": pct(ens["draw"]), "away_win": pct(ens["away_win"])},
        "model_breakdown": {
            m: ({o: pct(p) for o, p in probs.items()} if isinstance(probs, dict) else probs)
            for m, probs in breakdown.items()
        },
    })


@flask_app.route("/api/refresh", methods=["POST"])
def refresh_data():
    _get_singletons()
    _sheets_loader._client = None
    _cache_clear()
    live = _sheets_loader.load()
    source = "google_sheets" if live else "static_fallback"
    return jsonify({
        "status": "refreshed",
        "source": source,
        "competitions_loaded": len(live or _STATIC_COMPETITIONS),
    })
