from fastapi import FastAPI
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.service.popular_service import get_popular_movies
from src.service.content_based_service import get_content_recommendations
app = FastAPI(title="Movie Recommendation API")


@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/popular")
def popular_movies(top_n: int = 10):
    return get_popular_movies(top_n)

@app.get("/test-recommend")
def test_recommend(movie: str):
    return get_content_recommendations(movie)
    


