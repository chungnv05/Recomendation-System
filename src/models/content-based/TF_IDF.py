import pandas as pd 
import numpy as np
from src.data_loader import data_load
from collections import Counter
import math



def TfVectorizer(df: pd.DataFrame) -> list: # tính vector tf cho từng phim
    raw_genres = df['genres'].to_list()

    vocab = sorted(set(g.lower() for movie in raw_genres for g in movie))
    word_idx = {g:i for i,g in enumerate(vocab)}

    tf_vectors = []

    for movie in raw_genres:
        counter = Counter(movie)
        vector = [0]*len(vocab)
        for g, count in counter.items():
            vector[word_idx[g]] = count
        tf_vectors.append(vector)

    return tf_vectors

def IdfVectorizer(df: pd.DataFrame) -> list: # tính vector idf
    raw_genres = df['genres'].to_list()

    vocab = sorted(set(g.lower() for movie in raw_genres for g in movie))
    idf_vector = [0]*len(vocab)
    word_idx = {g:i for i,g in enumerate(vocab)}
    for word in vocab:
        freq = 0
        for movie in raw_genres:
            if word in set(movie):
                freq += 1
        idf_vector[word_idx[word]] = math.log((len(raw_genres) + 1)/(freq + 1)) + 1
    
    return idf_vector
        

def Tf_idfVectorizer(df: pd.DataFrame) -> list: # tính tf_idf vector cho tất cả phim
    tf_vectors = np.array(TfVectorizer(df))
    idf_vector = np.array(IdfVectorizer(df))

    return (tf_vectors * idf_vector).tolist()
    



    
    


    
        








