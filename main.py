import difflib
import pandas as pd
import numpy as np
import pickle
from data.load_data.data_loader import data_load_from_db
from src.recommend_logic.popular.popular import PopularRecommender
from src.recommend_logic.content_based.data_process import preprocess_data


def main():
    
    df = data_load_from_db()
    # test content-based model
    df = preprocess_data(df)
    with open("src/models/content-based/similarity_matrix.pkl", "rb") as f:
        similarity = pickle.load(f)

    while True:
        i = input("Nhấn 1 để tìm phim nhấn 2 để thoát: ")

        if (i == '1'):
            movie = input("Nhập phim: ")
            list_of_title = df['title'].tolist()

            find_close_match_movie = difflib.get_close_matches(movie, list_of_title)
            closest_match = find_close_match_movie[0]

            matches = df[df.title == closest_match]
            if not matches.empty:
                index_of_movie = matches['index'].values[0]
            else:
                print("Movie not found")

            similarity_score = list(enumerate(similarity[index_of_movie]))

            sorted_similarity = sorted(similarity_score, key= lambda x:x[1], reverse=True)

            print("Danh sách 5 phim gợi ý cho bạn: ")
            count = 0 
            for idx, score in sorted_similarity:

                if idx == index_of_movie:
                    continue

                title = df.iloc[idx]['title']
                print(f"{title}")

                count += 1
                if count == 30:
                    break
        else: 
            print("thank")
            break
    

    







    
if __name__ == "__main__":
    main()