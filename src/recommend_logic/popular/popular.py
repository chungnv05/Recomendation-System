import pandas as pd 
from pathlib import Path
import sys
import pickle

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))

from data.load_data.data_loader import data_load_from_db

# Sử dụng công thức IMDB Formula là công thức tính Weighted Rating dùng để tránh bias cho phim có ít lượt vote nhưng điểm cao bất thường.
# Công thức IMDB: 
class PopularRecommender:
    def __init__(self, df):
        self.C = df['vote_average'].mean() # Điểm trung bình toàn các phim
        self.m = df['vote_count'].quantile(0.8) # Số vote tối thiểu

    def vectorized_filter(self, df) -> pd.DataFrame:
        qualified_movies = df[df['vote_count'] >= self.m].copy() # Lọc bỏ các phim có vote_count < m sử dụng kĩ thuật vectorized_filter
        return qualified_movies
    
    def calculate_WR(self, movie): # tính WR cho 1 bộ phim cụ thể 
        v = movie['vote_count']
        R = movie['vote_average']
        return ((v/(v + self.m))*R) + ((self.m/(v + self.m))*self.C)
    
    def top_popular(self, df, top_n=10):
        # 1. Filter 
        qualified = self.vectorized_filter(df)

        # 2. Tính WR (vectorized)
        qualified.loc[:,'WR_score'] = (
            (qualified['vote_count']/(qualified['vote_count']+self.m))*qualified['vote_average']
            + (self.m/(qualified['vote_count']+self.m))*self.C
        )
        # Làm tròn
        qualified['WR_score'] = qualified['WR_score'].round(2)
        # 3. Sort
        qualified = qualified.sort_values(by='WR_score', ascending=False)
        # 4. Return top N
        return qualified
    
# Tạo model
def create_model():
    df = data_load_from_db() 
    popular = PopularRecommender(df)
    top_movie = popular.top_popular(df)
    
    with open('src/models/popular/popular.pkl', 'wb') as f:
        pickle.dump(top_movie, f)

# create_model()
    


    

    


    
        


