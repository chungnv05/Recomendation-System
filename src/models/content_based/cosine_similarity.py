import pandas as pd 
import numpy as np



def cosine_similarity(X, Y = None):

    if Y == None:
        Y = X
    

    similarity = np.dot(X,Y.T)

    return similarity


    

