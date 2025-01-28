import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'LUMN', 'INTC']

# Download full historical data for the tickers
data = yf.download(tickers, start="1900-01-01")  # Adjust start date if needed

# Display the first few rows of the data
print(data.head())