import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
from bot import run_trading_algorithm, calculate_indicators, generate_signals, apply_position_management, calculate_performance_metrics

# Major currency pairs for forex trading
MAJOR_FOREX_PAIRS = [
    'EURUSD=X',  # Euro/US Dollar
    'USDJPY=X',  # US Dollar/Japanese Yen
    'GBPUSD=X',  # British Pound/US Dollar
    'USDCHF=X',  # US Dollar/Swiss Franc
    'AUDUSD=X',  # Australian Dollar/US Dollar
    'USDCAD=X',  # US Dollar/Canadian Dollar
    'NZDUSD=X',  # New Zealand Dollar/US Dollar
]

def bulk_download_forex_data(symbols, start_date, end_date, max_retries=3):
    """Download data for multiple symbols at once to avoid rate limiting"""
    for attempt in range(max_retries):
        try:
            print(f"Bulk downloading {len(symbols)} forex pairs (attempt {attempt + 1}/{max_retries})")
            # Download all symbols at once
            data = yf.download(symbols, start=start_date, end=end_date, progress=False, group_by='ticker')
            
            if data.empty:
                raise ValueError(f"No data returned for symbols: {symbols}")
            
            # Unpack the data for each symbol
            symbol_data = {}
            for symbol in symbols:
                if len(symbols) == 1:
                    # Single symbol case - no multi-index
                    symbol_data[symbol] = data
                else:
                    # Multiple symbols case - extract each symbol's data
                    try:
                        symbol_data[symbol] = data[symbol].dropna()
                        if symbol_data[symbol].empty:
                            print(f"Warning: No data for {symbol}")
                    except KeyError:
                        print(f"Warning: {symbol} not found in downloaded data")
                        symbol_data[symbol] = pd.DataFrame()
            
            return symbol_data
            
        except Exception as e:
            print(f"Bulk download attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(2, 5)  # Exponential backoff
                print(f"Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise e

def download_data_with_retry(symbol, start_date, end_date, max_retries=3):
    """Download data with retry logic - kept for backward compatibility"""
    symbol_data = bulk_download_forex_data([symbol], start_date, end_date, max_retries)
    return symbol_data[symbol]

def run_trading_algorithm_with_retry(symbol, start_date="2020-01-01", end_date=None):
    """Run trading algorithm with retry logic for data download"""
    print(f"Running trading algorithm for {symbol}")
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Download data with retry
    data = download_data_with_retry(symbol, start_date, end_date)
    
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

def backtest_forex_pairs(pairs=None, start_date="2023-01-01", end_date=None):
    """Backtest the trading strategy on multiple forex pairs using bulk download"""
    if pairs is None:
        pairs = MAJOR_FOREX_PAIRS
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Bulk download all forex data at once
    print(f"\n{'='*60}")
    print(f"BULK DOWNLOADING {len(pairs)} FOREX PAIRS")
    print(f"{'='*60}")
    
    try:
        all_data = bulk_download_forex_data(pairs, start_date, end_date)
    except Exception as e:
        print(f"Bulk download failed: {str(e)}")
        return {}, pd.DataFrame()
    
    results = {}
    summary_metrics = []
    
    # Process each pair's data
    for pair in pairs:
        print(f"\n{'='*50}")
        print(f"Processing {pair}")
        print(f"{'='*50}")
        
        try:
            if pair not in all_data or all_data[pair].empty:
                raise ValueError(f"No data available for {pair}")
            
            data = all_data[pair].copy()
            
            # Calculate all indicators
            data = calculate_indicators(data)
            
            # Generate trading signals
            data = generate_signals(data)
            
            # Apply position management (TP/SL)
            data = apply_position_management(data, take_profit_pct=0.002, stop_loss_pct=0.001)
            
            # Calculate performance metrics
            metrics = calculate_performance_metrics(data)
            
            results[pair] = {
                'data': data,
                'metrics': metrics
            }
            
            # Add pair name to metrics
            metrics['Pair'] = pair
            summary_metrics.append(metrics)
            
            # Print individual results
            print("\n=== TRADING PERFORMANCE ===")
            for key, value in metrics.items():
                if 'PnL' in key or 'Win' in key or 'Loss' in key:
                    print(f"{key}: {value:.4f}%")
                elif key != 'Pair':
                    print(f"{key}: {value:.2f}")
            
        except Exception as e:
            print(f"Error processing {pair}: {str(e)}")
            # Create empty metrics for failed pairs
            empty_metrics = {
                'Total Trades': 0, 'Win Rate': 0, 'Average PnL': 0, 
                'Total PnL': 0, 'Sharpe Ratio': 0, 'Pair': pair
            }
            summary_metrics.append(empty_metrics)
            continue
    
    # Create summary report
    summary_df = pd.DataFrame(summary_metrics)
    summary_df = summary_df.sort_values('Win Rate', ascending=False)
    
    print("\n" + "="*80)
    print("FOREX PAIRS PERFORMANCE SUMMARY")
    print("="*80)
    print(summary_df.to_string())
    
    # Save summary to CSV
    summary_df.to_csv('forex_backtest_summary.csv', index=False)
    
    return results, summary_df

def analyze_best_trading_times(data):
    """Analyze which hours/days are most profitable for trading"""
    # Add hour and day of week
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    
    # Group by hour and calculate average PnL
    hourly_pnl = data.groupby('Hour')['PnL'].agg(['mean', 'count', 'sum'])
    hourly_pnl['mean'] = hourly_pnl['mean'] * 100  # Convert to percentage
    
    # Group by day of week
    daily_pnl = data.groupby('DayOfWeek')['PnL'].agg(['mean', 'count', 'sum'])
    daily_pnl['mean'] = daily_pnl['mean'] * 100  # Convert to percentage
    
    return hourly_pnl, daily_pnl

def optimize_parameters(symbol, param_ranges):
    """Optimize strategy parameters for a specific symbol"""
    best_params = {}
    best_win_rate = 0
    
    results = []
    
    # Download data once using bulk download (single symbol)
    symbol_data = bulk_download_forex_data([symbol], "2022-01-01", datetime.now().strftime("%Y-%m-%d"))
    data = symbol_data[symbol]
    
    # Test different parameter combinations
    for tp in param_ranges['take_profit']:
        for sl in param_ranges['stop_loss']:
            for bb_threshold in param_ranges['bb_threshold']:
                # Make a copy of data for this test
                test_data = data.copy()
                
                # Calculate indicators
                test_data = calculate_indicators(test_data)
                
                # Modify BB threshold for signal generation
                test_data['BB_NearSupport'] = (test_data['Close'] - test_data['LowerBand']) / test_data['LowerBand'] <= bb_threshold/100
                test_data['BB_NearResistance'] = (test_data['UpperBand'] - test_data['Close']) / test_data['UpperBand'] <= bb_threshold/100
                
                # Generate signals with modified conditions
                test_data = generate_signals(test_data)
                
                # Apply position management
                test_data = apply_position_management(test_data, take_profit_pct=tp, stop_loss_pct=sl)
                
                # Calculate metrics
                metrics = calculate_performance_metrics(test_data)
                
                # Store results
                result = {
                    'Take_Profit': tp * 100,
                    'Stop_Loss': sl * 100,
                    'BB_Threshold': bb_threshold * 100,
                    'Win_Rate': metrics['Win Rate'],
                    'Total_PnL': metrics['Total PnL'],
                    'Total_Trades': metrics['Total Trades'],
                    'Sharpe_Ratio': metrics['Sharpe Ratio']
                }
                results.append(result)
                
                # Check if this is the best so far
                if metrics['Win Rate'] > best_win_rate and metrics['Total Trades'] > 10:
                    best_win_rate = metrics['Win Rate']
                    best_params = {
                        'take_profit': tp,
                        'stop_loss': sl,
                        'bb_threshold': bb_threshold,
                        'metrics': metrics
                    }
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Win_Rate', ascending=False)
    
    print(f"\nOptimization Results for {symbol}:")
    print(results_df.head(10))
    
    return best_params, results_df

# Multi-pair trading bot
class MultiPairForexBot:
    def __init__(self, pairs=None, initial_capital=10000, leverage=1, position_size_pct=0.1):
        self.pairs = pairs if pairs else MAJOR_FOREX_PAIRS
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.position_size_pct = position_size_pct
        self.current_positions = {}  # {pair: position_info}
        self.trade_history = []
        self.pair_data = {}  # Store data for each pair
        self.last_signals = {}  # Track last signals for each pair
        
    def load_data(self, start_date, end_date=None):
        """Load data for all trading pairs"""
        print(f"Loading data for {len(self.pairs)} forex pairs...")
        self.pair_data = bulk_download_forex_data(self.pairs, start_date, end_date)
        
        # Prepare data with indicators and signals for each pair
        for pair in self.pairs:
            if pair in self.pair_data and not self.pair_data[pair].empty:
                data = self.pair_data[pair].copy()
                data = calculate_indicators(data)
                data = generate_signals(data)
                self.pair_data[pair] = data
                self.last_signals[pair] = 0
                print(f"Prepared {len(data)} data points for {pair}")
            else:
                print(f"Warning: No data available for {pair}")
                
    def get_current_signals(self, timestamp):
        """Get current trading signals for all pairs at given timestamp"""
        signals = {}
        for pair in self.pairs:
            if pair in self.pair_data:
                data = self.pair_data[pair]
                # Find the closest timestamp
                try:
                    idx = data.index.get_indexer([timestamp], method='nearest')[0]
                    if idx >= 0 and idx < len(data):
                        signals[pair] = {
                            'signal': data['Signal'].iloc[idx],
                            'price': data['Close'].iloc[idx],
                            'timestamp': data.index[idx]
                        }
                except:
                    signals[pair] = {'signal': 0, 'price': 0, 'timestamp': timestamp}
        return signals
        
    def execute_trade(self, pair, signal, price, timestamp):
        """Execute a trade for a specific pair"""
        if signal == 0:  # No signal
            return
            
        available_capital = self.capital - sum([pos['position_value'] for pos in self.current_positions.values()])
        position_size = available_capital * self.position_size_pct * self.leverage
        
        if position_size <= 0:
            return  # No capital available
            
        if signal == 1:  # Buy signal
            if pair not in self.current_positions:
                units = position_size / price
                self.current_positions[pair] = {
                    'units': units,
                    'entry_price': price,
                    'entry_time': timestamp,
                    'type': 'long',
                    'position_value': position_size
                }
                print(f"BUY {pair}: {units:.2f} units at {price:.4f}")
                
        elif signal == -1:  # Sell signal
            if pair in self.current_positions and self.current_positions[pair]['type'] == 'long':
                # Close long position
                position = self.current_positions[pair]
                pnl = (price - position['entry_price']) * position['units']
                self.capital += pnl
                
                trade_record = {
                    'pair': pair,
                    'entry_price': position['entry_price'],
                    'exit_price': price,
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'units': position['units'],
                    'pnl': pnl,
                    'pnl_pct': (price - position['entry_price']) / position['entry_price'] * 100,
                    'duration': timestamp - position['entry_time']
                }
                
                self.trade_history.append(trade_record)
                del self.current_positions[pair]
                print(f"SELL {pair}: PnL ${pnl:.2f} ({trade_record['pnl_pct']:.2f}%)")
                
    def run_multi_pair_trading(self, start_date, end_date=None):
        """Run trading simulation across multiple pairs simultaneously"""
        # Load data for all pairs
        self.load_data(start_date, end_date)
        
        # Get all unique timestamps across all pairs
        all_timestamps = set()
        for pair, data in self.pair_data.items():
            if not data.empty:
                all_timestamps.update(data.index)
        
        all_timestamps = sorted(list(all_timestamps))
        print(f"Running multi-pair trading across {len(all_timestamps)} time periods...")
        
        # Process each timestamp
        for i, timestamp in enumerate(all_timestamps):
            if i % 1000 == 0:  # Progress update
                print(f"Processing {i}/{len(all_timestamps)} ({i/len(all_timestamps)*100:.1f}%)")
                
            # Get signals for all pairs at this timestamp
            current_signals = self.get_current_signals(timestamp)
            
            # Execute trades for each pair
            for pair, signal_info in current_signals.items():
                if signal_info['signal'] != self.last_signals.get(pair, 0):
                    # Signal changed, execute trade
                    self.execute_trade(pair, signal_info['signal'], 
                                     signal_info['price'], signal_info['timestamp'])
                    self.last_signals[pair] = signal_info['signal']
        
        # Close any remaining positions at the end
        if all_timestamps:
            final_timestamp = all_timestamps[-1]
            for pair in list(self.current_positions.keys()):
                final_signals = self.get_current_signals(final_timestamp)
                if pair in final_signals:
                    self.execute_trade(pair, -1, final_signals[pair]['price'], 
                                     final_signals[pair]['timestamp'])
                                     
        return self.get_performance_summary()
        
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        if not self.trade_history:
            return "No trades executed"
            
        trades_df = pd.DataFrame(self.trade_history)
        
        # Overall performance
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # Per-pair performance
        pair_performance = trades_df.groupby('pair').agg({
            'pnl': ['sum', 'mean', 'count'],
            'pnl_pct': 'mean'
        }).round(4)
        
        summary = {
            'Overall Performance': {
                'Initial Capital': self.initial_capital,
                'Final Capital': self.capital,
                'Total Return': total_return,
                'Total Trades': total_trades,
                'Winning Trades': winning_trades,
                'Win Rate': win_rate,
                'Average PnL per Trade': trades_df['pnl'].mean(),
                'Total PnL': trades_df['pnl'].sum()
            },
            'Per-Pair Performance': pair_performance,
            'Trade History': trades_df
        }
        
        return summary

# Paper trading simulation (kept for backward compatibility)
class PaperTrader:
    def __init__(self, initial_capital=10000, leverage=1):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.current_positions = {}
        
    def execute_trade(self, symbol, signal, price, timestamp, position_size_pct=0.1):
        """Execute a paper trade"""
        position_size = self.capital * position_size_pct * self.leverage
        
        if signal == 1:  # Buy
            if symbol not in self.current_positions:
                units = position_size / price
                self.current_positions[symbol] = {
                    'units': units,
                    'entry_price': price,
                    'entry_time': timestamp,
                    'type': 'long'
                }
                print(f"BUY {symbol}: {units:.2f} units at ${price:.4f}")
                
        elif signal == -1:  # Sell/Short
            if symbol in self.current_positions and self.current_positions[symbol]['type'] == 'long':
                # Close long position
                position = self.current_positions[symbol]
                pnl = (price - position['entry_price']) * position['units']
                self.capital += pnl
                
                self.trade_history.append({
                    'symbol': symbol,
                    'entry_price': position['entry_price'],
                    'exit_price': price,
                    'pnl': pnl,
                    'pnl_pct': (price - position['entry_price']) / position['entry_price'] * 100,
                    'duration': timestamp - position['entry_time']
                })
                
                del self.current_positions[symbol]
                print(f"SELL {symbol}: PnL ${pnl:.2f} ({pnl/position_size*100:.2f}%)")
                
    def get_performance_summary(self):
        """Get paper trading performance summary"""
        if not self.trade_history:
            return "No trades executed yet"
        
        trades_df = pd.DataFrame(self.trade_history)
        
        summary = {
            'Initial Capital': self.initial_capital,
            'Current Capital': self.capital,
            'Total Return': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'Total Trades': len(trades_df),
            'Winning Trades': len(trades_df[trades_df['pnl'] > 0]),
            'Win Rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100,
            'Average PnL': trades_df['pnl'].mean(),
            'Total PnL': trades_df['pnl'].sum()
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    print("Starting Multi-Pair Forex Trading System...")
    print("="*60)
    
    # Option 1: Traditional individual pair backtesting
    print("\n1. Individual Pair Backtesting (for comparison)")
    results, summary = backtest_forex_pairs(start_date="2023-01-01")
    
    # Option 2: Multi-pair simultaneous trading
    print("\n" + "="*60)
    print("2. Multi-Pair Simultaneous Trading")
    print("="*60)
    
    # Initialize multi-pair bot
    forex_bot = MultiPairForexBot(
        pairs=MAJOR_FOREX_PAIRS[:4],  # Start with top 4 pairs
        initial_capital=10000,
        leverage=1,
        position_size_pct=0.2  # 20% of available capital per position
    )
    
    try:
        # Run multi-pair trading simulation
        multi_pair_performance = forex_bot.run_multi_pair_trading(
            start_date="2023-01-01",
            end_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Display results
        print("\n" + "="*60)
        print("MULTI-PAIR TRADING RESULTS")
        print("="*60)
        
        if isinstance(multi_pair_performance, dict):
            # Overall performance
            overall = multi_pair_performance['Overall Performance']
            print("\nOverall Performance:")
            for key, value in overall.items():
                if isinstance(value, (int, float)):
                    if 'Rate' in key or 'Return' in key:
                        print(f"{key}: {value:.2f}%")
                    elif 'PnL' in key or 'Capital' in key:
                        print(f"{key}: ${value:.2f}")
                    else:
                        print(f"{key}: {value:.0f}")
            
            # Per-pair performance
            print("\nPer-Pair Performance:")
            pair_perf = multi_pair_performance['Per-Pair Performance']
            print(pair_perf.to_string())
            
            # Save detailed results
            trade_history = multi_pair_performance['Trade History']
            trade_history.to_csv('multi_pair_trading_results.csv', index=False)
            print(f"\nDetailed trade history saved to 'multi_pair_trading_results.csv'")
            
        else:
            print(multi_pair_performance)
            
    except Exception as e:
        print(f"Multi-pair trading failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Option 3: Parameter optimization for best performing pair
    if len(summary) > 0 and summary['Total Trades'].max() > 0:
        print("\n" + "="*60)
        print("3. Parameter Optimization")
        print("="*60)
        
        # Find pair with most trades (more reliable for optimization)
        best_pair = summary.loc[summary['Total Trades'].idxmax(), 'Pair']
        print(f"\nOptimizing parameters for: {best_pair}")
        
        param_ranges = {
            'take_profit': [0.001, 0.002, 0.003, 0.004],  # 0.1% to 0.4%
            'stop_loss': [0.0005, 0.001, 0.0015, 0.002],  # 0.05% to 0.2%
            'bb_threshold': [0.03, 0.05, 0.07, 0.10]  # 3% to 10%
        }
        
        try:
            best_params, optimization_results = optimize_parameters(best_pair, param_ranges)
            
            print(f"\nBest Parameters for {best_pair}:")
            if best_params and 'take_profit' in best_params:
                print(f"Take Profit: {best_params['take_profit']*100:.2f}%")
                print(f"Stop Loss: {best_params['stop_loss']*100:.2f}%")
                print(f"BB Threshold: {best_params['bb_threshold']*100:.2f}%")
            else:
                print("No optimal parameters found - insufficient profitable trades")
        except Exception as e:
            print(f"Parameter optimization failed: {str(e)}")