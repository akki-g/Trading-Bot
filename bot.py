import yfinance as yf
import pandas as pd
import numpy as np 
import time


def get_SMAs(data, slow, fast):
    data['SMA_Fast'] = data['Close'].rolling(window=fast).mean()
    data['SMA_Slow'] = data['Close'].rolling(window=slow).mean()
    return data


def get_ema(data, period):
    data['EMA'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

    
def get_macd(data, period_long, period_short, period_signal):
    shortEMA = data['Close'].ewm(span=period_short, adjust=False).mean()
    longEMA = data['Close'].ewm(span=period_long, adjust=False).mean()
    data['MACD'] = shortEMA - longEMA
    data['Signal_Line'] = data['MACD'].ewm(span=period_signal, adjust=False).mean()
    return data

    
def get_rsi(data, period):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))
    return data
 

def get_BollingerBands(data, period):
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['20dSTD'] = data['Close'].rolling(window=period).std()
    data['UpperBand'] = data['SMA'] + (data['20dSTD'] * 1)
    data['LowerBand'] = data['SMA'] - (data['20dSTD'] * 1)
    return data

def get_beta(data, window):
    data['Return'] = np.log(data['Close']).diff()
    benchmark = yf.download('^GSPC', start='1900-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    benchmark['BenchReturn'] = np.log(benchmark['Close']).diff()
    
    combined = pd.concat([data['Return'], benchmark['BenchReturn']], axis= 1).dropna()
    combined['Covariance'] = combined['Return'].rolling(window = window).cov(combined['BenchReturn'])
    combined['Variance'] = combined['BenchReturn'].rolling(window = window).var()

    combined['Beta'] = combined['Covariance'] / combined['Variance']
    data['Beta'] = combined['Beta']
    data['Beta'] = data['Beta'].fillna(method='ffill')
    return data


def set_signals(data):
    data = data.dropna() 
    data = data.reset_index(drop=True)
    print("DataFrame Info Before Signals:")
    print(data.info())
    print("First few rows:")
    print(data.head())

    # SMA signal
    data['SMAsignal'] = 0
    data.loc[data['SMA_Fast'] > data['SMA_Slow'], 'SMAsignal'] = 1
    data.loc[data['SMA_Fast'] <= data['SMA_Slow'], 'SMAsignal'] = 0

    # EMA signal
    data['EMAsignal'] = 0
    data.loc[data['Close'] > data['EMA'], 'EMAsignal'] = 1
    data.loc[data['Close'] <= data['EMA'], 'EMAsignal'] = 0

    # MACD signal
    data['MACDsignal'] = 0
    data.loc[data['MACD'] > data['Signal_Line'], 'MACDsignal'] = 1
    data.loc[data['MACD'] <= data['Signal_Line'], 'MACDsignal'] = 0

    # RSI signal
    data['RSIsignal'] = 0
    data.loc[data['RSI'] > 70, 'RSIsignal'] = 1
    data.loc[data['RSI'] < 30, 'RSIsignal'] = 0

    # Bollinger Bands signal
    data['BBsignal'] = 0
    data.loc[data['Close'] > data['UpperBand'], 'BBsignal'] = 1
    data.loc[data['Close'] < data['LowerBand'], 'BBsignal'] = 0
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
       
    return data


def get_data(symbol):
    date = time.strftime("%Y-%m-%d")
    data = yf.download(symbol, start="1900-01-01", end=date)
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten by selecting the first level
        data.columns = data.columns.get_level_values(0)
    data = get_SMAs(data, 100, 10)
    data = get_macd(data, 26, 12, 9)
    data = get_rsi(data, 14)
    data = get_BollingerBands(data, 20)
    data = get_ema(data, 9)
    data = get_beta(data, 100)
    data = set_signals(data)
    print(data)
    return data

    