import pandas as pd 
import numpy as np
from collections import Counter
import math


def build_vocab(df: pd.DataFrame, col_name: str):
    raw_docs = df[col_name].to_list()
    docs = [movie.lower().split() for movie in raw_docs]
    vocab = sorted(set(g for movie in docs for g in movie))
    word_idx = {g: i for i, g in enumerate(vocab)}
    return docs, vocab, word_idx


def TfVectorizer(df: pd.DataFrame) -> list:
    docs, vocab, word_idx = build_vocab(df, col_name='soup')
    tf_vectors = []

    for movie in docs:
        counter = Counter(movie)
        vector = [0] * len(vocab)
        for g, count in counter.items():
            vector[word_idx[g]] = count 
        tf_vectors.append(vector)
    return tf_vectors


def IdfVectorizer(df: pd.DataFrame) -> list: 
    docs, vocab, word_idx = build_vocab(df, col_name='soup')
    idf_vector = [0] * len(vocab)
    word_set = [set(movie) for movie in docs]
    for word in vocab:
        freq = 0
        for movie in word_set:
            if word in movie:
                freq += 1

        idf_vector[word_idx[word]] = math.log((len(docs) + 1) / (freq + 1)) + 1
    return idf_vector
        

def Tf_idfVectorizer(df: pd.DataFrame) -> list:
    docs, vocab, word_idx = build_vocab(df, col_name='soup')
    
    # TF
    tf_vectors = []
    for movie in docs:
        counter = Counter(movie)
        vector = [0] * len(vocab)
        for g, count in counter.items():
            vector[word_idx[g]] = count 
        tf_vectors.append(vector)

    # IDF 
    idf_vector = [0] * len(vocab)
    word_set = [set(movie) for movie in docs]
    for word in vocab:
        freq = 0
        for movie in word_set:
            if word in movie:
                freq += 1

        idf_vector[word_idx[word]] = math.log((len(docs) + 1) / (freq + 1)) + 1
    
    tf_vectors = np.array(tf_vectors)
    idf_vector = np.array(idf_vector)
    
    tf_idf = tf_vectors * idf_vector
    norms = np.linalg.norm(tf_idf, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10 

    return tf_idf / norms