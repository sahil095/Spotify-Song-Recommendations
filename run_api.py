"""
Convenience script to run the FastAPI application
"""
import uvicorn
from config.config import API_HOST, API_PORT

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Music Recommendation API")
    print("=" * 60)
    print(f"Server will be available at: http://{API_HOST}:{API_PORT}")
    print(f"API Documentation: http://{API_HOST}:{API_PORT}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "src.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True  # Auto-reload on code changes
    )
