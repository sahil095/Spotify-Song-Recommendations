# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements_api.txt
```

## Step 2: Train the Model

Before running the web application, train the recommendation model:

```bash
python src/train_model.py
```

This will:
- Load data from `data/df_clean.csv`
- Train the Nearest Neighbors model
- Save the model to `models/recommendation_model.pkl`

**Expected output:**
```
============================================================
Music Recommendation Model Training
============================================================
Loading data from data/df_clean.csv...
Available numerical features: [...]
Model dataframe shape: (8582, 14)
Building preprocessing pipeline...
Training model...
Model training complete!
Saving model to models/recommendation_model.pkl...
Model saved successfully!
```

## Step 3: Run the Web Application

### Option A: Using the convenience script
```bash
python run_api.py
```

### Option B: Using uvicorn directly
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### Option C: Using Python
```bash
python src/app.py
```

## Step 4: Access the Application

1. **Web Interface**: Open your browser and go to:
   ```
   http://localhost:8000
   ```

2. **API Documentation**: Interactive API docs at:
   ```
   http://localhost:8000/docs
   ```

3. **Health Check**: Verify the API is running:
   ```
   http://localhost:8000/api/health
   ```

## Testing the System

1. Open the web interface at `http://localhost:8000`
2. Enter a song name (e.g., "I Know You Want Me (Calle Ocho)")
3. Click "Get Recommendations"
4. View the top 10 similar songs!

## Troubleshooting

### Error: "Model not found"
- Make sure you ran `python src/train_model.py` first
- Check that `models/recommendation_model.pkl` exists

### Error: "Song not found"
- Try using the exact song name from the dataset
- The search supports partial matches (case-insensitive)

### Port 8000 already in use
- Change the port in `config/config.py` (API_PORT)
- Or use: `uvicorn src.app:app --port 8080`

## Next Steps

- Customize features in `config/config.py`
- Modify the UI in `templates/index.html` and `static/style.css`
- Add more endpoints in `src/app.py`
