import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Constants
DB_NAME = "stock_data.db"
LSTM_SEQUENCE_LENGTH = 50
TRAIN_TEST_SPLIT = 0.8
MODEL_NAME = "model.h5"

def load_stock_data(db_name):
    """Retrieve stock data from the database."""
    conn = sqlite3.connect(db_name)
    query = "SELECT * FROM stocks ORDER BY timestamp ASC"
    df = pd.read_sql(query, conn)
    conn.close()
    
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def prepare_data(df, sequence_length):
    """Prepare data for training and testing."""
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(sequence_length):
    """Build LSTM model architecture."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 5)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(5)  # Predicting Open, High, Low, Close, Volume
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

if __name__ == "__main__":
    df = load_stock_data(DB_NAME)
    X, y, scaler = prepare_data(df, LSTM_SEQUENCE_LENGTH)
    
    split = int(TRAIN_TEST_SPLIT * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = build_lstm_model(LSTM_SEQUENCE_LENGTH)
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
    model.save(MODEL_NAME)
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Training Completed. MAE: {mae:.4f}, MSE: {mse:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:, 3], label='Actual Close Prices', color='blue')
    plt.plot(predictions[:, 3], label='Predicted Close Prices', color='red')
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.title("Actual vs Predicted Stock Prices")
    plt.show()
