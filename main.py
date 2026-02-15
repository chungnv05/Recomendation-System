import pandas as pd
import numpy as np

from src.data_loader import data_load
from src.models.popular import PopularRecommender
from src.models.content_based.data_process import preprocess_data
from src.models.content_based.TF_IDF import Tf_idfVectorizer


def main():
    df = data_load('data/raw/movies.csv')
    # POPULAR MODEL
    print("=== POPULAR MODEL ===")

    popular_model = PopularRecommender(df)
    popular_result = popular_model.top_popular(df)

    print(popular_result.head())
    # CONTENT BASED MODEL
    print("\n=== CONTENT BASED MODEL ===")
    df = preprocess_data(df)
    tfidf_vectors = Tf_idfVectorizer(df)
    print("Số phim:", len(tfidf_vectors))
    print("Số chiều vector:", len(tfidf_vectors[0]))
    print("\nVector phim đầu tiên (20 phần tử đầu):")
    print(tfidf_vectors[0][:20])
if __name__ == "__main__":
    main()