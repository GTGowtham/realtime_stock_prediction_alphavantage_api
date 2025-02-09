import numpy as np
import pandas as pd
import sqlite3
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Constants
DB_NAME = "stock_data.db"
MODEL_PATH = "model_av.h5"
LSTM_SEQUENCE_LENGTH = 50

# Load the trained model without compiling
model = load_model(MODEL_PATH, compile=False)
scaler = MinMaxScaler()

# Function to get the next trading day (skip weekends)
def get_next_trading_day(last_date):
    next_date = last_date + pd.Timedelta(days=1)
    if next_date.weekday() == 5:  # Saturday → Move to Monday
        next_date += pd.Timedelta(days=2)
    elif next_date.weekday() == 6:  # Sunday → Move to Monday
        next_date += pd.Timedelta(days=1)
    return next_date

# Function to load the latest stock data
def load_latest_data(db_name, table_name, days=None):
    conn = sqlite3.connect(db_name)
    query = f"SELECT * FROM {table_name} WHERE timestamp IS NOT NULL ORDER BY timestamp DESC"
    df = pd.read_sql(query, conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')  # Convert timestamp column to datetime
    df = df.dropna(subset=['timestamp'])  # Drop rows with invalid timestamps
    df = df.sort_values(by='timestamp')  # Ensure chronological order
    
    if days:
        df = df.tail(days)  # Get last 'days' worth of data
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# Function to make predictions
def predict_next_day(model, data):
    data_scaled = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume']])
    X_input = np.array([data_scaled])  # Shape: (1, 50, 5)
    prediction_scaled = model.predict(X_input)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0]

# Streamlit UI
st.title("Stock Price Prediction for Next Trading Day")

st.write("This app loads a trained LSTM model and predicts the next valid trading day's Open, High, Low, Close, and Volume based on the latest data from the database.")

# Dropdown to select stock table
stock_options = ["apple", "nvidia", "google", "microsoft", "amazon"]
selected_stock = st.selectbox("Select Stock:", stock_options)

db_table = selected_stock.lower()

if st.button("Predict Next Day Prices"):
    latest_data = load_latest_data(DB_NAME, db_table, LSTM_SEQUENCE_LENGTH)
    
    if len(latest_data) < LSTM_SEQUENCE_LENGTH:
        st.error("Not enough data to make a prediction. Please collect more stock data.")
    else:
        prediction = predict_next_day(model, latest_data)
        last_date = latest_data['timestamp'].max()
        next_trading_day = get_next_trading_day(last_date)  # Adjust for weekends
        
        st.write(f"### Predicted Prices for {next_trading_day.strftime('%Y-%m-%d')}")
        st.write(f"**Open:** {prediction[0]:.2f}")
        st.write(f"**High:** {prediction[1]:.2f}")
        st.write(f"**Low:** {prediction[2]:.2f}")
        st.write(f"**Close:** {prediction[3]:.2f}")
        st.write(f"**Volume:** {prediction[4]:,.0f}")

# Fetch last 30 days data
if st.button("Fetch Last 30 Days Data"):
    last_30_days_data = load_latest_data(DB_NAME, db_table, 30)
    
    if last_30_days_data.empty:
        st.error("No data available for the last 30 days.")
    else:
        st.write("### Last 30 Days Stock Data")
        st.dataframe(last_30_days_data)
