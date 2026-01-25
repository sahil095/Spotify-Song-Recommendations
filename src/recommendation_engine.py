"""
Recommendation Engine for Music Recommendation System
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import pickle
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ID_COLUMNS,
    MODEL_FILE, PREPROCESSOR_FILE, MODEL_DF_FILE, DATA_FILE
)


class RecommendationEngine:
    """Music recommendation engine using Nearest Neighbors"""
    
    def __init__(self):
        self.pipe = None
        self.model_df = None
        self.name_to_idx = None
        self.id_to_idx = None
        self.numerical_features = NUMERICAL_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.id_columns = ID_COLUMNS
        
    def load_data(self):
        """Load and prepare the dataset"""
        print(f"Loading data from {DATA_FILE}...")
        df_clean = pd.read_csv(DATA_FILE)
        
        # Filter available features
        available_numerical = [f for f in self.numerical_features if f in df_clean.columns]
        available_categorical = [f for f in self.categorical_features if f in df_clean.columns]
        available_id_cols = [c for c in self.id_columns if c in df_clean.columns]
        
        print(f"Available numerical features: {available_numerical}")
        print(f"Available categorical features: {available_categorical}")
        
        # Create model dataframe
        self.model_df = df_clean[available_id_cols + available_numerical + available_categorical].copy()
        
        # Fill missing values
        for c in available_numerical:
            self.model_df[c] = self.model_df[c].fillna(self.model_df[c].median())
        for c in available_categorical:
            self.model_df[c] = self.model_df[c].fillna('Unknown')
        
        # Update feature lists to only include available features
        self.numerical_features = available_numerical
        self.categorical_features = available_categorical
        
        print(f"Model dataframe shape: {self.model_df.shape}")
        return self.model_df
    
    def build_model(self):
        """Build and train the recommendation model"""
        if self.model_df is None:
            self.load_data()
        
        print("Building preprocessing pipeline...")
        # Preprocessing: scale numericals, one-hot encode categoricals
        preprocess = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features),
            ],
            remainder='drop'
        )
        
        # Nearest Neighbors model
        nn_model = NearestNeighbors(metric='cosine', algorithm='auto')
        
        # Pipeline
        self.pipe = Pipeline([
            ('preprocess', preprocess),
            ('nn', nn_model)
        ])
        
        # Prepare features
        X = self.model_df[self.numerical_features + self.categorical_features]
        
        print("Training model...")
        self.pipe.fit(X)
        print("Model training complete!")
        
        # Build lookup indices
        self._build_lookup_indices()
        
    def _build_lookup_indices(self):
        """Build lookup indices for quick song search"""
        self.name_to_idx = {}
        self.id_to_idx = {}
        
        if 'track_name' in self.model_df.columns:
            # Handle duplicates by keeping first occurrence
            for idx, name in self.model_df['track_name'].astype(str).items():
                if name not in self.name_to_idx:
                    self.name_to_idx[name] = idx
        
        if 'track_id' in self.model_df.columns:
            for idx, track_id in self.model_df['track_id'].astype(str).items():
                if track_id not in self.id_to_idx:
                    self.id_to_idx[track_id] = idx
    
    def save_model(self):
        """Save the trained model and data"""
        print(f"Saving model to {MODEL_FILE}...")
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(self.pipe, f)
        
        print(f"Saving model dataframe to {MODEL_DF_FILE}...")
        with open(MODEL_DF_FILE, 'wb') as f:
            pickle.dump(self.model_df, f)
        
        print("Model saved successfully!")
    
    def load_model(self):
        """Load a previously saved model"""
        if not MODEL_FILE.exists() or not MODEL_DF_FILE.exists():
            raise FileNotFoundError("Model files not found. Please train the model first.")
        
        print(f"Loading model from {MODEL_FILE}...")
        with open(MODEL_FILE, 'rb') as f:
            self.pipe = pickle.load(f)
        
        print(f"Loading model dataframe from {MODEL_DF_FILE}...")
        with open(MODEL_DF_FILE, 'rb') as f:
            self.model_df = pickle.load(f)
        
        # Rebuild lookup indices
        self._build_lookup_indices()
        
        # Extract feature names from the pipeline
        preprocessor = self.pipe.named_steps['preprocess']
        # Get numerical features
        num_transformer = preprocessor.transformers_[0]
        self.numerical_features = num_transformer[2] if len(num_transformer) > 2 else []
        
        # Get categorical features
        cat_transformer = preprocessor.transformers_[1]
        self.categorical_features = cat_transformer[2] if len(cat_transformer) > 2 else []
        
        print("Model loaded successfully!")
    
    def recommend_songs(self, song, k=10):
        """
        Return k closest songs for a given input song
        
        Parameters
        ----------
        song : str
            A track name (e.g., "Blinding Lights") OR a track_id
        k : int
            Number of recommendations to return
        
        Returns
        -------
        dict
            Dictionary containing input song info and recommendations
        """
        if self.pipe is None or self.model_df is None:
            raise ValueError("Model not loaded. Please load or train the model first.")
        
        if song is None or str(song).strip() == '':
            raise ValueError('Please provide a non-empty song name or track_id.')
        
        song = str(song).strip()
        
        # Resolve index
        idx = None
        matched_song = song
        
        if self.id_to_idx is not None and song in self.id_to_idx:
            idx = self.id_to_idx[song]
        elif self.name_to_idx is not None and song in self.name_to_idx:
            idx = self.name_to_idx[song]
            matched_song = self.model_df.loc[idx, 'track_name']
        else:
            # Fallback: case-insensitive contains match on track_name
            if 'track_name' in self.model_df.columns:
                mask = self.model_df['track_name'].astype(str).str.lower().str.contains(song.lower(), na=False)
                if mask.any():
                    idx = self.model_df.loc[mask].index[0]
                    matched_song = self.model_df.loc[idx, 'track_name']
        
        if idx is None:
            raise KeyError(f"Song '{song}' not found. Try an exact track_name or a valid track_id.")
        
        # Get input song info
        input_song_info = {}
        input_track_name = None
        input_artist_name = None
        
        for col in self.id_columns:
            if col in self.model_df.columns:
                value = str(self.model_df.loc[idx, col])
                input_song_info[col] = value
                if col == 'track_name':
                    input_track_name = value
                elif col == 'artist_name(s)':
                    input_artist_name = value
        
        # Query neighbors - request k+1 to account for the input song itself
        # The first result is always the query point itself (distance = 0)
        query_X = self.model_df.loc[[idx], self.numerical_features + self.categorical_features]
        distances, indices = self.pipe.named_steps['nn'].kneighbors(
            self.pipe.named_steps['preprocess'].transform(query_X),
            n_neighbors=min(k + 1, len(self.model_df))
        )
        
        distances = distances.ravel()
        indices = indices.ravel()
        
        # Convert indices back to original row indices
        neighbor_df_indices = self.model_df.iloc[indices].index.values
        
        
        # Build results - skip the first result (it's always the input song itself)
        results = []
        for i, neighbor_idx in enumerate(neighbor_df_indices):
            # Always skip the first result (index 0) as it's the query point itself
            # if i == 0:
            #     continue
            
            print(i, neighbor_idx)
            neighbor_track_name = str(self.model_df.loc[neighbor_idx, 'track_name']) if 'track_name' in self.model_df.columns else None
            neighbor_artist_name = str(self.model_df.loc[neighbor_idx, 'artist_name(s)']) if 'artist_name(s)' in self.model_df.columns else None
            print(neighbor_track_name, neighbor_artist_name)
            # Additional safety check: skip if it's the same song (by index or track_name)
            is_same_song = (
                neighbor_idx == idx or  # Same DataFrame index
                (input_track_name and neighbor_track_name and 
                 input_track_name.lower().strip() == neighbor_track_name.lower().strip())  # Same track name (case-insensitive)
            )
            
            if is_same_song:
                continue
            
            result = {
                'track_name': neighbor_track_name or 'Unknown',
                'artist_name': neighbor_artist_name or 'Unknown',
                'popularity': int(self.model_df.loc[neighbor_idx, 'popularity']) if 'popularity' in self.model_df.columns else 0,
                'similarity': float(1 - distances[i])
            }
            results.append(result)
            
            if len(results) >= k:
                break
        
        return {
            'input_song': input_song_info,
            'recommendations': results
        }
