import pandas as pd 
import numpy as np
from src.data_loader import data_load
from collections import Counter
import math



def build_vocab(df: pd.DataFrame, col_name: str):
    raw_genres = df[col_name].to_list()

    docs = [[g.lower().strip() for g in movie] for movie in raw_genres]
    vocab = sorted(set(g for movie in docs for g in movie))
    word_idx = {g:i for i, g in enumerate(vocab)}

    return docs, vocab, word_idx

def TfVectorizer(df: pd.DataFrame) -> list: # tính vector tf cho từng phim
    
    docs,vocab, word_idx = build_vocab(df,col_name= 'soup')

    tf_vectors = []

    for movie in docs:
        counter = Counter(movie)
        vector = [0]*len(vocab)
        for g, count in counter.items():
            vector[word_idx[g]] = count
        tf_vectors.append(vector)

    return tf_vectors

def IdfVectorizer(df: pd.DataFrame) -> list: # tính vector idf
    docs, vocab, word_idx = build_vocab(df,col_name='soup')
    idf_vector = [0]*len(vocab)
    word_set = [set(movie) for movie in docs]
    for word in vocab:
        freq = 0
        for movie in word_set:
            if word in movie:
                freq += 1
        idf_vector[word_idx[word]] = math.log((len(docs) + 1)/(freq + 1)) + 1
    
    return idf_vector
        

def Tf_idfVectorizer(df: pd.DataFrame) -> list: # tính tf_idf vector cho tất cả phim
    tf_vectors = np.array(TfVectorizer(df))
    idf_vector = np.array(IdfVectorizer(df))

    return (tf_vectors * idf_vector).tolist()
    



    
    


    
        








