# Music Recommendation System - Web Application

A web-based music recommendation system that uses machine learning to suggest similar songs based on audio features. Built with FastAPI backend and a simple HTML/CSS frontend.

## Features

- 🎵 **Content-Based Recommendations**: Uses audio features (danceability, energy, tempo, etc.) to find similar songs
- 🔍 **Smart Song Search**: Supports exact song names or partial matches
- 📊 **Top-K Recommendations**: Get 1-20 recommendations per query
- 🚀 **Fast API**: Built with FastAPI for high performance
- 💻 **Simple UI**: Clean and intuitive web interface

## Project Structure

```
.
├── config/
│   └── config.py              # Configuration settings
├── src/
│   ├── app.py                 # FastAPI application
│   ├── recommendation_engine.py  # Core recommendation logic
│   └── train_model.py         # Model training script
├── static/
│   └── style.css              # CSS styles
├── templates/
│   └── index.html             # HTML template
├── models/                     # Saved models (created after training)
├── data/
│   └── df_clean.csv           # Dataset
├── requirements_api.txt        # Python dependencies
└── README_API.md              # This file
```

## Installation

1. **Install Dependencies**

   ```bash
   pip install -r requirements_api.txt
   ```

2. **Train the Model**

   Before running the web application, you need to train and save the recommendation model:

   ```bash
   python src/train_model.py
   ```

   This will:
   - Load the data from `data/df_clean.csv`
   - Train the Nearest Neighbors model
   - Save the model to `models/recommendation_model.pkl`
   - Save the processed data to `models/model_df.pkl`

## Running the Application

### Option 1: Using Python

```bash
python src/app.py
```

### Option 2: Using Uvicorn directly

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## Usage

### Web Interface

1. Open your browser and navigate to `http://localhost:8000`
2. Enter a song name in the search box
3. Select the number of recommendations (default: 10)
4. Click "Get Recommendations"
5. View the recommended songs with similarity scores

### API Endpoints

#### 1. Get Recommendations

**POST** `/api/recommend`

Request body:
```json
{
  "song": "I Know You Want Me (Calle Ocho)",
  "k": 10
}
```

Response:
```json
{
  "input_song": {
    "track_name": "I Know You Want Me (Calle Ocho)",
    "artist_name(s)": "Pitbull"
  },
  "recommendations": [
    {
      "track_name": "Super Freaky Girl",
      "artist_name": "Nicki Minaj",
      "popularity": 53,
      "similarity": 0.9256
    },
    ...
  ]
}
```

#### 2. Health Check

**GET** `/api/health`

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Configuration

Edit `config/config.py` to customize:

- **Features**: Add or remove numerical/categorical features
- **Model Parameters**: Change number of neighbors, metric, etc.
- **API Settings**: Modify host, port, title, etc.

## Troubleshooting

### Model Not Found Error

If you see "Model not loaded" error:
1. Make sure you've run `python src/train_model.py` first
2. Check that `models/recommendation_model.pkl` exists
3. Verify the data file path in `config/config.py`

### Song Not Found

- Try using the exact song name as it appears in the dataset
- The search is case-insensitive and supports partial matches
- Check the dataset to see available songs

### Port Already in Use

If port 8000 is already in use:
1. Change the port in `config/config.py`
2. Or specify a different port: `uvicorn src.app:app --port 8080`

## Development

### Adding New Features

1. Update `NUMERICAL_FEATURES` or `CATEGORICAL_FEATURES` in `config/config.py`
2. Retrain the model: `python src/train_model.py`
3. Restart the application

### Customizing the UI

- Edit `templates/index.html` for HTML structure
- Edit `static/style.css` for styling

## License

Same as the main project.

## Notes

- The model uses cosine similarity to find similar songs
- Recommendations are based on audio features, not user behavior
- The system requires the `df_clean.csv` file in the `data/` directory
- Model training may take a few minutes depending on dataset size
