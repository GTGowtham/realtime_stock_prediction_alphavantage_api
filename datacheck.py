import requests
import json

API_KEY = "PPTEH0JD06XO2PH1"  # Replace with your API Key
SYMBOL = "AAPL"  # Change to any stock symbol

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&apikey={API_KEY}&datatype=json"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    time_series = data.get("Time Series (Daily)", {})
    
    if time_series:
        latest_date = max(time_series.keys())  # Get the most recent date
        latest_data = time_series[latest_date]
        print(f"✅ Latest available data for {SYMBOL}: {latest_date}")
        print(json.dumps(latest_data, indent=2))
    else:
        print("⚠️ No time series data available. Check API limits.")
else:
    print(f"⚠️ API request failed with status code {response.status_code}")
