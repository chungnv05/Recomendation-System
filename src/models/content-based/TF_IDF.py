import pandas as pd 
import numpy as np
from src.data_loader import data_load


def TfVectorizer(df: pd.DataFrame) -> list:
    word_set = set()
    raw_genres = df['genres'].to_list()
    for movie_genre in raw_genres:
        for genre in movie_genre:
            word_set.add(genre)
    
    word_set = sorted(list(word_set))


    tf_vector = []
    for movie_genre in raw_genres:
        vector = [0] * len(word_set)
        current_movie = movie_genre
        for genre in current_movie:
            if genre in word_set:
                index = word_set.index(genre)
                vector[index] = 1 
        tf_vector.append(vector)

    return tf_vector

    
        








