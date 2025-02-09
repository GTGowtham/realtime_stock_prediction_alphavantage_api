import sqlite3
import pandas as pd
import requests
import io
import time
import datetime

API_KEY = "PPTEH0JD06XO2PH1"
DB_NAME = "stock_data.db"

STOCK_SYMBOLS = {
    "apple": "AAPL",
    "nvidia": "NVDA",
    "google": "GOOGL",
    "microsoft": "MSFT",
    "amazon": "AMZN"
}

def fetch_latest_stock_data(symbol, api_key):
    """Fetch only the last few days of stock data."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&datatype=csv"
    response = requests.get(url)

    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text))
        
        # Debug: Print the first few rows of data
        print(f"\n✅ Data received for {symbol}:")
        print(df.head())

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        print(f"\n⚠️ Failed to fetch data for {symbol}. HTTP Status: {response.status_code}")
        return None

def check_latest_date_in_db(table_name, db_name=DB_NAME):
    """Check the latest date available in the database for a given stock table."""
    conn = sqlite3.connect(db_name)

    query_check = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    table_exists = pd.read_sql(query_check, conn)

    if table_exists.empty:
        print(f"Table {table_name} does not exist. Creating a new one.")
        conn.close()
        return None

    query = f"SELECT MAX(timestamp) AS latest_date FROM {table_name}"
    result = pd.read_sql(query, conn)
    conn.close()

    if result.iloc[0, 0]:
        return pd.to_datetime(result.iloc[0, 0])
    
    return None

def store_data_in_db(df, table_name, db_name=DB_NAME):
    """Store new data in the database only if it is missing."""
    latest_db_date = check_latest_date_in_db(table_name)

    if latest_db_date:
        df = df[df['timestamp'] > latest_db_date]  # Insert only new data

    if not df.empty:
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists="append", index=False)
        conn.close()
        print(f"✅ New stock data for {table_name} stored successfully.")
    else:
        print(f"🔹 No new data for {table_name}. Data is already up-to-date.")

if __name__ == "__main__":
    while True:
        for table_name, symbol in STOCK_SYMBOLS.items():
            data = fetch_latest_stock_data(symbol, API_KEY)
            if data is not None:
                store_data_in_db(data, table_name)
        print("Stock data update completed.")
        print("Sleeping for 24 hours...")
        time.sleep(86400)  # Runs daily
