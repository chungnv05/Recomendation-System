import pandas as pd

def data_load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)
