import pandas as pd
import re

def preprocess_data(df: pd.DataFrame):
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        df[feature] = df[feature].fillna('')
    df['soup'] = (
        df['genres'] + ' ' +
        df['keywords'] + ' ' +
        df['tagline'] + ' ' +
        df['cast'] + ' ' +
        df['director']
    )
    df['soup'] = df['soup'].str.lower()
    df['soup'] = df['soup'].apply(
        lambda x: re.sub(r'[^a-z0-9\s]', ' ', x)
    )
    return df