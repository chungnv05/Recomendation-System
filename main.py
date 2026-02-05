import pandas as pd 
from src.models.popular import PopularRecommender
from src.data_loader import data_load


# Test thử
def main(): 
    df = data_load('data/raw/movies.csv')
    
    popular_model = PopularRecommender(df)
    res = popular_model.top_popular(df)

    print(res)
    res.to_csv('data/raw/example.csv')
    



if __name__ == "__main__":
    main()

    