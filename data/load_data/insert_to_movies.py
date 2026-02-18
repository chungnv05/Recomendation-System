import pandas as pd
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.config.database_connect import get_engine

engine = get_engine()

df = pd.read_csv("data/raw/movies.csv")

columns = [
    "index",
    "title",
    "genres",
    "keywords",
    "tagline",
    "cast",
    "director",
    "overview",
    "vote_average",
    "vote_count"
]

df = df[columns]

df.to_sql(
    name="movies",
    con=engine,
    if_exists="append",
    index=False
)

