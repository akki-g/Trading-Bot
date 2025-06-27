import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

def get_SMAs(data, slow, fast):
    data['SMA_Fast'] = data['Close'].rolling(window=fast).mean()
    data['SMA_Slow'] = data['Close'].rolling(window=slow).mean()
    return data

def get_ema(data, period):
    data[f'EMA{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

def get_macd(data, period_short, period_long, period_signal):
    """Updated MACD with strategy periods (8,21,5)"""
    shortEMA = data['Close'].ewm(span=period_short, adjust=False).mean()
    longEMA = data['Close'].ewm(span=period_long, adjust=False).mean()
    data['MACD'] = shortEMA - longEMA
    data['Signal_Line'] = data['MACD'].ewm(span=period_signal, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    return data

def get_rsi(data, periods):
    """Calculate RSI for multiple periods"""
    for period in periods:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        RS = gain / loss
        data[f'RSI{period}'] = 100 - (100 / (1 + RS))
    return data

def get_BollingerBands(data, period, num_std=2):
    data['BB_SMA'] = data['Close'].rolling(window=period).mean()
    data['BB_STD'] = data['Close'].rolling(window=period).std()
    data['UpperBand'] = data['BB_SMA'] + (data['BB_STD'] * num_std)
    data['LowerBand'] = data['BB_SMA'] - (data['BB_STD'] * num_std)
    data['BB_Width'] = data['UpperBand'] - data['LowerBand']
    return data

def get_stochastic(data, k_period=9, d_period=3, smooth_k=3):
    """Calculate Stochastic Oscillator (9,3,3)"""
    # Calculate %K
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    data['Stoch_K'] = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    
    # Smooth %K
    data['Stoch_K'] = data['Stoch_K'].rolling(window=smooth_k).mean()
    
    # Calculate %D (3-period SMA of %K)
    data['Stoch_D'] = data['Stoch_K'].rolling(window=d_period).mean()
    
    return data

def calculate_indicators(data):
    """Calculate all technical indicators"""
    # EMA200 for trend filter
    data = get_ema(data, 200)
    
    # RSI with multiple periods (6,14,24)
    data = get_rsi(data, [6, 14, 24])
    
    # MACD with strategy periods (8,21,5)
    data = get_macd(data, 8, 21, 5)
    
    # Bollinger Bands
    data = get_BollingerBands(data, 20)
    
    # Stochastic (9,3,3)
    data = get_stochastic(data, 9, 3, 3)
    
    return data

def generate_signals(data):
    """Generate buy/sell signals based on strategy rules"""
    data['Signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
    data['Entry'] = False
    data['Exit'] = False
    
    # Calculate helper columns
    # Check if price is above EMA200 (trend filter)
    data['Above_EMA200'] = data['Close'] > data['EMA200']
    
    # RSI conditions (using RSI14 as main)
    data['RSI_Bull'] = data['RSI14'] > 50
    data['RSI_Bear'] = data['RSI14'] < 50
    
    # Bollinger Bands conditions
    # Close to support = within 5% of lower band
    data['BB_NearSupport'] = (data['Close'] - data['LowerBand']) / data['LowerBand'] <= 0.05
    # Close to resistance = within 5% of upper band
    data['BB_NearResistance'] = (data['UpperBand'] - data['Close']) / data['UpperBand'] <= 0.05
    
    # MACD crossovers
    data['MACD_BullCross'] = (data['MACD'] > data['Signal_Line']) & (data['MACD'].shift(1) <= data['Signal_Line'].shift(1))
    data['MACD_BearCross'] = (data['MACD'] < data['Signal_Line']) & (data['MACD'].shift(1) >= data['Signal_Line'].shift(1))
    
    # Stochastic crossovers
    data['Stoch_BullCross'] = (data['Stoch_K'] > data['Stoch_D']) & (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1))
    data['Stoch_BearCross'] = (data['Stoch_K'] < data['Stoch_D']) & (data['Stoch_K'].shift(1) >= data['Stoch_D'].shift(1))
    
    # BUY CONDITIONS (all must be true)
    buy_conditions = (
        data['Above_EMA200'] &  # Trend filter
        data['RSI_Bull'] &  # RSI above 50
        data['BB_NearSupport'] &  # Price near lower band
        data['MACD_BullCross'] &  # MACD bullish crossover
        data['Stoch_BullCross']  # Stochastic bullish crossover
    )
    
    # SELL CONDITIONS (all must be true)
    sell_conditions = (
        data['Above_EMA200'] &  # Still need trend filter
        data['RSI_Bear'] &  # RSI below 50
        data['BB_NearResistance'] &  # Price near upper band
        data['MACD_BearCross'] &  # MACD bearish crossover
        data['Stoch_BearCross']  # Stochastic bearish crossover
    )
    
    data.loc[buy_conditions, 'Signal'] = 1
    data.loc[sell_conditions, 'Signal'] = -1
    
    return data

def apply_position_management(data, take_profit_pct=0.002, stop_loss_pct=0.001):
    """Apply take profit and stop loss logic"""
    data['Position'] = 0
    data['Entry_Price'] = np.nan
    data['Exit_Price'] = np.nan
    data['PnL'] = 0
    data['Exit_Reason'] = ''
    
    position = 0
    entry_price = 0
    
    for i in range(1, len(data)):
        # If no position
        if position == 0:
            if data['Signal'].iloc[i] == 1:  # Buy signal
                position = 1
                entry_price = data['Close'].iloc[i]
                data.loc[data.index[i], 'Position'] = 1
                data.loc[data.index[i], 'Entry_Price'] = entry_price
                data.loc[data.index[i], 'Entry'] = True
                
            elif data['Signal'].iloc[i] == -1:  # Sell signal (for short)
                position = -1
                entry_price = data['Close'].iloc[i]
                data.loc[data.index[i], 'Position'] = -1
                data.loc[data.index[i], 'Entry_Price'] = entry_price
                data.loc[data.index[i], 'Entry'] = True
        
        # If in long position
        elif position == 1:
            current_price = data['Close'].iloc[i]
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Check take profit
            if pnl_pct >= take_profit_pct:
                data.loc[data.index[i], 'Exit_Price'] = current_price
                data.loc[data.index[i], 'PnL'] = pnl_pct
                data.loc[data.index[i], 'Exit'] = True
                data.loc[data.index[i], 'Exit_Reason'] = 'Take Profit'
                position = 0
                
            # Check stop loss
            elif pnl_pct <= -stop_loss_pct:
                data.loc[data.index[i], 'Exit_Price'] = current_price
                data.loc[data.index[i], 'PnL'] = pnl_pct
                data.loc[data.index[i], 'Exit'] = True
                data.loc[data.index[i], 'Exit_Reason'] = 'Stop Loss'
                position = 0
                
            # Check for exit signal
            elif data['Signal'].iloc[i] == -1:
                data.loc[data.index[i], 'Exit_Price'] = current_price
                data.loc[data.index[i], 'PnL'] = pnl_pct
                data.loc[data.index[i], 'Exit'] = True
                data.loc[data.index[i], 'Exit_Reason'] = 'Signal'
                position = 0
            else:
                data.loc[data.index[i], 'Position'] = position
        
        # If in short position
        elif position == -1:
            current_price = data['Close'].iloc[i]
            pnl_pct = (entry_price - current_price) / entry_price
            
            # Check take profit
            if pnl_pct >= take_profit_pct:
                data.loc[data.index[i], 'Exit_Price'] = current_price
                data.loc[data.index[i], 'PnL'] = pnl_pct
                data.loc[data.index[i], 'Exit'] = True
                data.loc[data.index[i], 'Exit_Reason'] = 'Take Profit'
                position = 0
                
            # Check stop loss
            elif pnl_pct <= -stop_loss_pct:
                data.loc[data.index[i], 'Exit_Price'] = current_price
                data.loc[data.index[i], 'PnL'] = pnl_pct
                data.loc[data.index[i], 'Exit'] = True
                data.loc[data.index[i], 'Exit_Reason'] = 'Stop Loss'
                position = 0
                
            # Check for exit signal
            elif data['Signal'].iloc[i] == 1:
                data.loc[data.index[i], 'Exit_Price'] = current_price
                data.loc[data.index[i], 'PnL'] = pnl_pct
                data.loc[data.index[i], 'Exit'] = True
                data.loc[data.index[i], 'Exit_Reason'] = 'Signal'
                position = 0
            else:
                data.loc[data.index[i], 'Position'] = position
    
    return data

def calculate_performance_metrics(data):
    """Calculate trading performance metrics"""
    trades = data[data['Exit'] == True].copy()
    
    if len(trades) == 0:
        return {
            'Total Trades': 0,
            'Win Rate': 0,
            'Average PnL': 0,
            'Total PnL': 0,
            'Sharpe Ratio': 0
        }
    
    winning_trades = trades[trades['PnL'] > 0]
    losing_trades = trades[trades['PnL'] < 0]
    
    metrics = {
        'Total Trades': len(trades),
        'Winning Trades': len(winning_trades),
        'Losing Trades': len(losing_trades),
        'Win Rate': len(winning_trades) / len(trades) * 100,
        'Average PnL': trades['PnL'].mean() * 100,
        'Total PnL': trades['PnL'].sum() * 100,
        'Max Win': trades['PnL'].max() * 100,
        'Max Loss': trades['PnL'].min() * 100,
        'Take Profit Exits': len(trades[trades['Exit_Reason'] == 'Take Profit']),
        'Stop Loss Exits': len(trades[trades['Exit_Reason'] == 'Stop Loss']),
        'Signal Exits': len(trades[trades['Exit_Reason'] == 'Signal'])
    }
    
    # Calculate Sharpe Ratio (simplified)
    if trades['PnL'].std() > 0:
        metrics['Sharpe Ratio'] = (trades['PnL'].mean() / trades['PnL'].std()) * np.sqrt(252)
    else:
        metrics['Sharpe Ratio'] = 0
    
    return metrics

def run_trading_algorithm(symbol, start_date="2020-01-01", end_date=None):
    """Main function to run the complete trading algorithm"""
    print(f"Running trading algorithm for {symbol}")
    
    # Download data
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Handle multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Calculate all indicators
    data = calculate_indicators(data)
    
    # Generate trading signals
    data = generate_signals(data)
    
    # Apply position management (TP/SL)
    data = apply_position_management(data, take_profit_pct=0.002, stop_loss_pct=0.001)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(data)
    
    # Print results
    print("\n=== TRADING PERFORMANCE ===")
    for key, value in metrics.items():
        if 'PnL' in key or 'Win' in key or 'Loss' in key:
            print(f"{key}: {value:.4f}%")
        else:
            print(f"{key}: {value:.2f}")
    
    return data, metrics

# Example usage for backtesting
if __name__ == "__main__":
    # Test with NVDA
    symbol = 'NVDA'
    data, metrics = run_trading_algorithm(symbol, start_date="2022-01-01")
    
    # Save results to CSV
    data.to_csv(f'{symbol}_trading_results.csv')
    
    # Create a summary of trades
    trades_summary = data[data['Exit'] == True][['Entry_Price', 'Exit_Price', 'PnL', 'Exit_Reason']]
    print("\n=== RECENT TRADES ===")
    print(trades_summary.tail(10))
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Price and signals
    ax1.plot(data.index, data['Close'], label='Close Price', color='black', linewidth=1)
    ax1.plot(data.index, data['EMA200'], label='EMA200', color='blue', alpha=0.7)
    ax1.plot(data.index, data['UpperBand'], label='Upper BB', color='red', alpha=0.5)
    ax1.plot(data.index, data['LowerBand'], label='Lower BB', color='green', alpha=0.5)
    
    # Mark entries and exits
    entries = data[data['Entry'] == True]
    exits = data[data['Exit'] == True]
    ax1.scatter(entries.index, entries['Close'], color='green', marker='^', s=100, label='Entry')
    ax1.scatter(exits.index, exits['Close'], color='red', marker='v', s=100, label='Exit')
    
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2.plot(data.index, data['RSI14'], label='RSI(14)', color='purple')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Cumulative PnL
    cumulative_pnl = data['PnL'].cumsum() * 100
    ax3.plot(data.index, cumulative_pnl, label='Cumulative PnL (%)', color='blue')
    ax3.fill_between(data.index, 0, cumulative_pnl, alpha=0.3)
    ax3.set_ylabel('Cumulative PnL (%)')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{symbol}_trading_performance.png', dpi=150)
    plt.show()