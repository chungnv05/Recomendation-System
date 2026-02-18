from pathlib import Path
import sys
import pickle
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

with open('src/models/popular/popular.pkl', 'rb') as f:
        popular_list =  pickle.load(f)

def get_popular_movies(top_n: int = 10):
    
    popular_list = popular_list.head(top_n)
    columms = ['title', 'WR_score']
    popular_list = popular_list[columms]
    return popular_list.to_dict(orient="records")

    
