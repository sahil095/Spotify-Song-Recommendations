"""
Configuration file for the Music Recommendation System
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Data file
DATA_FILE = DATA_DIR / "df_clean.csv"

# Model files
MODEL_FILE = MODELS_DIR / "recommendation_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
MODEL_DF_FILE = MODELS_DIR / "model_df.pkl"

# Feature configuration
NUMERICAL_FEATURES = [
    'danceability', 'energy', 'loudness', 'explicit',
    'instrumentalness', 'tempo', 'popularity', 'valence',
    'speechiness', 'acousticness', 'liveness'
]

CATEGORICAL_FEATURES = ['mode']

ID_COLUMNS = ['track_name', 'artist_name(s)']

# Model parameters
N_NEIGHBORS = 10
METRIC = 'cosine'

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Music Recommendation API"
API_VERSION = "1.0.0"
