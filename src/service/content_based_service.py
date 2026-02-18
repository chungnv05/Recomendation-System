import pickle
import difflib
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from data.load_data.data_loader import data_load_from_db
from src.recommend_logic.content_based.data_process import preprocess_data


df = data_load_from_db()
df = preprocess_data(df)



with open('src/models/content-based/similarity_matrix.pkl', "rb") as f:
    similarity = pickle.load(f)


def get_content_recommendations(movie_name: str, top_n: int = 10):

    titles = df['title'].tolist()

    matches = difflib.get_close_matches(movie_name, titles, n=1)

    if not matches:
        return {"error": "Movie not found"}

    closest_match = matches[0]

    movie_index = df[df.title == closest_match]['index'].values[0]

    similarity_scores = list(enumerate(similarity[movie_index]))

    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for idx, score in sorted_scores[1:top_n+1]:
        recommendations.append({
            "title": df.iloc[idx]['title'],
            "score": round(score, 3)
        })

    return {
        "searched_movie": closest_match,
        "recommendations": recommendations
    }
