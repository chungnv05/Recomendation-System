import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")

def connection_string():
    driver = DB_DRIVER.replace(" ", "+")
    if not DB_USER:
        return f"mssql+pyodbc://@{DB_SERVER}/{DB_NAME}?driver={driver}&trusted_connection=yes"
    return f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}?driver={driver}"

def get_engine():
    return create_engine(connection_string(), fast_executemany=True, pool_pre_ping=True)