"""
Script to train and save the recommendation model
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.recommendation_engine import RecommendationEngine
from config.config import MODEL_FILE, MODEL_DF_FILE

def main():
    """Train and save the recommendation model"""
    print("=" * 60)
    print("Music Recommendation Model Training")
    print("=" * 60)
    
    # Initialize engine
    engine = RecommendationEngine()
    
    # Load data
    engine.load_data()
    
    # Build and train model
    engine.build_model()
    
    # Save model
    engine.save_model()
    
    print("\n" + "=" * 60)
    print("Training complete! Model saved successfully.")
    print(f"Model file: {MODEL_FILE}")
    print(f"Data file: {MODEL_DF_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
