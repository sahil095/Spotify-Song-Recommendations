"""
FastAPI application for Music Recommendation System
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from pydantic import BaseModel
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.recommendation_engine import RecommendationEngine
from config.config import API_TITLE, API_VERSION, STATIC_DIR, TEMPLATES_DIR

# Initialize FastAPI app
app = FastAPI(title=API_TITLE, version=API_VERSION)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize recommendation engine
engine = RecommendationEngine()

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load the recommendation model on application startup"""
    try:
        engine.load_model()
        print("✓ Recommendation model loaded successfully!")
    except FileNotFoundError:
        print("⚠ Model not found. Please run train_model.py first.")
        print("⚠ The API will not work until the model is trained.")
    except Exception as e:
        print(f"✗ Error loading model: {e}")

# Request/Response models
class RecommendationRequest(BaseModel):
    song: str
    k: int = 10

class RecommendationResponse(BaseModel):
    input_song: dict
    recommendations: list

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get music recommendations for a given song
    
    Parameters:
    - song: Song name to get recommendations for
    - k: Number of recommendations (default: 10)
    
    Returns:
    - Dictionary with input song info and list of recommendations
    """
    try:
        if engine.pipe is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first by running train_model.py"
            )
        
        result = engine.recommend_songs(request.song, k=request.k)
        return result
    
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": engine.pipe is not None
    }

if __name__ == "__main__":
    import uvicorn
    from config.config import API_HOST, API_PORT
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)
