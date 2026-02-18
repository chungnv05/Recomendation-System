import pandas as pd 
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))

from TF_IDF import Tf_idfVectorizer
from src.data_loader import data_load
from data_process import preprocess_data
import pickle

def cosine_similarity(X, Y = None):

    if Y is None:
        Y = X
    

    similarity = np.dot(X,Y.T)

    return similarity


def train_model():
    df = data_load('data/raw/movies.csv')
    df = preprocess_data(df)
    
    tfidf = Tf_idfVectorizer(df)
    similarity_matrix = cosine_similarity(tfidf)
    
    with open("src/models/content-based/similarity_matrix.pkl", "wb") as f:
        pickle.dump(similarity_matrix, f)
        

    
    
    
    
    
    

