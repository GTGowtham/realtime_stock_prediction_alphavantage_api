# Stock Price Prediction for Next Trading Day

This project predicts the next trading day's stock prices using an LSTM model trained on historical stock market data. It fetches data using the **Alpha Vantage API**, stores it in an SQLite database, and provides predictions through a **Streamlit UI**.

## Features
- Fetch real-time stock data using the **Alpha Vantage API**
- Store stock data in an SQLite database
- Train and use an **LSTM model** for price prediction
- Predict the **Open, High, Low, Close, and Volume** for the next trading day
- Skip weekends and predict for Monday if the next day is Saturday or Sunday
- Streamlit UI for visualization

---

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/GTGowtham/stock-prediction.git
   cd stock-prediction
   
2.Installing libraries required:
   ```bash
   pip install -r requirements.txt
  


