import pandas as pd
from pathlib import Path
import sys
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.config.database_connect import get_engine

engine = get_engine()

def data_load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def data_load_from_db():
    query = 'SELECT * FROM movies'
    df = pd.read_sql(query,engine)
    return df
    