#!/usr/bin/env python3
"""
Incremental Data Updater for Forex Trading System
Fetches only new data since the last update in the database
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Tuple
import time
import schedule
from csv_data_manager import CSVDataManager
from database_manager import DatabaseManager
from config import ForexConfig

class IncrementalUpdater:
    """Handles incremental updates of forex data"""
    
    def __init__(self, csv_manager: Optional[CSVDataManager] = None):
        self.csv_manager = csv_manager or CSVDataManager()
        self.db_manager = self.csv_manager.db_manager
        self.logger = logging.getLogger(__name__)
        
        # Major forex pairs with yfinance symbols
        self.forex_pairs = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'NZDUSD': 'NZDUSD=X'
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_missing_data_ranges(self, pair: str) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """Get ranges of missing data for both daily and minute intervals"""
        missing_ranges = {'1d': [], '1m': []}
        
        for interval in ['1d', '1m']:
            last_timestamp = self.csv_manager.get_last_timestamp(pair, interval)
            
            if last_timestamp is None:
                # No data exists, need full range
                if interval == '1d':
                    start_date = datetime.now() - timedelta(days=30)  # Last 30 days
                else:  # 1m
                    start_date = datetime.now() - timedelta(days=7)   # Last 7 days
                missing_ranges[interval].append((start_date, datetime.now()))
            else:
                # Get data from last timestamp to now
                gap_start = last_timestamp + (timedelta(days=1) if interval == '1d' else timedelta(minutes=1))
                gap_end = datetime.now()
                
                if gap_end > gap_start:
                    missing_ranges[interval].append((gap_start, gap_end))
        
        return missing_ranges
    
    def fetch_yfinance_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                           interval: str = '1d') -> Optional[pd.DataFrame]:
        """Fetch data from yfinance with error handling"""
        try:
            # Map our intervals to yfinance intervals
            yf_interval_map = {
                '1d': '1d',
                '1m': '1m'
            }
            
            yf_interval = yf_interval_map.get(interval, '1d')
            
            self.logger.info(f"Fetching {symbol} data from {start_date} to {end_date} ({yf_interval})")
            
            # Add buffer for timezone issues
            start_buffer = start_date - timedelta(days=1)
            end_buffer = end_date + timedelta(days=1)
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_buffer.strftime('%Y-%m-%d'),
                end=end_buffer.strftime('%Y-%m-%d'),
                interval=yf_interval,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                self.logger.warning(f"No data returned for {symbol} ({yf_interval})")
                return None
            
            # Clean and format data
            data = data.reset_index()
            
            # Rename columns to match our schema
            column_mapping = {
                'Date': 'datetime',
                'Datetime': 'datetime',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            data = data.rename(columns=column_mapping)
            
            # Add required columns
            pair_name = symbol.replace('=X', '')  # Convert EURUSD=X to EURUSD
            data['symbol'] = pair_name
            data['interval'] = interval
            
            # Filter to exact date range
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]
            
            # Ensure we have the required columns
            required_cols = ['datetime', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume']
            data = data[required_cols]
            
            # Validate data
            data, errors = self.csv_manager.validate_csv_data(data)
            
            if errors:
                self.logger.warning(f"Data validation issues for {symbol}: {errors}")
            
            self.logger.info(f"Fetched {len(data)} records for {symbol} ({interval})")
            return data if not data.empty else None
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def update_pair_data(self, pair: str, max_retries: int = 3) -> bool:
        """Update data for a specific forex pair"""
        symbol = self.forex_pairs.get(pair)
        if not symbol:
            self.logger.error(f"Unknown forex pair: {pair}")
            return False
        
        success = True
        
        # Get missing data ranges
        missing_ranges = self.get_missing_data_ranges(pair)
        
        for interval, ranges in missing_ranges.items():
            if not ranges:
                self.logger.info(f"{pair} ({interval}): No missing data")
                continue
            
            for start_date, end_date in ranges:
                # Skip if the range is too small
                min_gap = timedelta(hours=1) if interval == '1m' else timedelta(days=1)
                if end_date - start_date < min_gap:
                    continue
                
                # Fetch data with retries
                data = None
                for attempt in range(max_retries):
                    try:
                        data = self.fetch_yfinance_data(symbol, start_date, end_date, interval)
                        if data is not None:
                            break
                    except Exception as e:
                        self.logger.warning(f"Attempt {attempt + 1} failed for {pair}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                
                if data is not None and not data.empty:
                    try:
                        # Insert the new data
                        self.csv_manager.bulk_insert_data(data, batch_size=1000)
                        self.logger.info(f"Updated {len(data)} records for {pair} ({interval})")
                    except Exception as e:
                        self.logger.error(f"Error inserting data for {pair}: {e}")
                        success = False
                else:
                    self.logger.warning(f"No data fetched for {pair} ({interval}) from {start_date} to {end_date}")
                    success = False
        
        return success
    
    def update_all_pairs(self) -> Dict[str, bool]:
        """Update data for all forex pairs"""
        self.logger.info("Starting incremental update for all forex pairs")
        
        results = {}
        for pair in self.forex_pairs.keys():
            try:
                results[pair] = self.update_pair_data(pair)
                # Small delay between pairs to be respectful to API
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error updating {pair}: {e}")
                results[pair] = False
        
        # Refresh materialized view after all updates
        try:
            self.csv_manager.refresh_latest_view()
        except Exception as e:
            self.logger.warning(f"Could not refresh materialized view: {e}")
        
        # Log summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        self.logger.info(f"Update completed: {successful}/{total} pairs updated successfully")
        
        return results
    
    def check_data_freshness(self) -> Dict[str, Dict[str, Optional[datetime]]]:
        """Check how fresh the data is for each pair and interval"""
        freshness = {}
        
        for pair in self.forex_pairs.keys():
            freshness[pair] = {}
            for interval in ['1d', '1m']:
                last_timestamp = self.csv_manager.get_last_timestamp(pair, interval)
                freshness[pair][interval] = last_timestamp
        
        return freshness
    
    def print_data_status(self):
        """Print current data status for all pairs"""
        freshness = self.check_data_freshness()
        now = datetime.now()
        
        self.logger.info("=== FOREX DATA STATUS ===")
        for pair, intervals in freshness.items():
            self.logger.info(f"\n{pair}:")
            for interval, last_time in intervals.items():
                if last_time:
                    age = now - last_time.replace(tzinfo=None)
                    self.logger.info(f"  {interval}: {last_time} (age: {age})")
                else:
                    self.logger.info(f"  {interval}: NO DATA")
    
    def schedule_updates(self):
        """Schedule automatic updates"""
        # Update 1-minute data every 5 minutes during market hours
        schedule.every(5).minutes.do(self.update_minute_data)
        
        # Update daily data once per day at 6 AM UTC (after market close)
        schedule.every().day.at("06:00").do(self.update_daily_data)
        
        # Status check every hour
        schedule.every().hour.do(self.print_data_status)
        
        self.logger.info("Scheduled automatic updates:")
        self.logger.info("- 1-minute data: every 5 minutes")
        self.logger.info("- Daily data: daily at 6:00 AM UTC")
        self.logger.info("- Status check: every hour")
    
    def update_minute_data(self):
        """Update only 1-minute data"""
        self.logger.info("Running scheduled 1-minute data update")
        
        for pair in self.forex_pairs.keys():
            try:
                missing_ranges = self.get_missing_data_ranges(pair)
                if missing_ranges['1m']:
                    self.update_pair_data(pair)
                time.sleep(0.5)  # Small delay between pairs
            except Exception as e:
                self.logger.error(f"Error in scheduled update for {pair}: {e}")
    
    def update_daily_data(self):
        """Update only daily data"""
        self.logger.info("Running scheduled daily data update")
        
        for pair in self.forex_pairs.keys():
            try:
                missing_ranges = self.get_missing_data_ranges(pair)
                if missing_ranges['1d']:
                    self.update_pair_data(pair)
                time.sleep(0.5)  # Small delay between pairs
            except Exception as e:
                self.logger.error(f"Error in daily update for {pair}: {e}")
    
    def run_scheduler(self):
        """Run the scheduled updater"""
        self.schedule_updates()
        
        self.logger.info("Starting scheduled forex data updater...")
        self.logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            self.logger.info("Scheduled updater stopped by user")


def main():
    """Main function for incremental updates"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Incremental forex data updater')
    parser.add_argument('--pair', help='Update specific pair only')
    parser.add_argument('--status', action='store_true', help='Show data status only')
    parser.add_argument('--schedule', action='store_true', help='Run scheduled updates')
    parser.add_argument('--update-all', action='store_true', help='Update all pairs once')
    
    args = parser.parse_args()
    
    try:
        updater = IncrementalUpdater()
        
        if args.status:
            updater.print_data_status()
        elif args.pair:
            success = updater.update_pair_data(args.pair.upper())
            if success:
                print(f"Successfully updated {args.pair}")
            else:
                print(f"Failed to update {args.pair}")
                return 1
        elif args.schedule:
            updater.run_scheduler()
        elif args.update_all:
            results = updater.update_all_pairs()
            failed = [pair for pair, success in results.items() if not success]
            if failed:
                print(f"Failed to update: {', '.join(failed)}")
                return 1
            else:
                print("All pairs updated successfully")
        else:
            print("Use --help to see available options")
            return 1
            
    except Exception as e:
        print(f"Update failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())