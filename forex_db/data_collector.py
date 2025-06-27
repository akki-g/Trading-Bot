import requests
import pandas as pd
import yfinance as yf
from alpha_vantage.foreignexchange import ForeignExchange
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp
from retry import retry
import json
from config import ForexConfig, API_ENDPOINTS

logger = logging.getLogger(__name__)

class ForexDataCollector:
    """Collects forex data from multiple sources with fallback mechanisms."""
    
    def __init__(self):
        self.config = ForexConfig()
        self.alpha_vantage = None
        self.session = None
        
        # Initialize Alpha Vantage if API key is available
        if self.config.ALPHA_VANTAGE_API_KEY:
            self.alpha_vantage = ForeignExchange(key=self.config.ALPHA_VANTAGE_API_KEY)
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @retry(tries=3, delay=1, backoff=2)
    def get_historical_data_alpha_vantage(self, pair: str, 
                                        interval: str = '1min') -> List[Dict]:
        """Get historical data from Alpha Vantage API."""
        if not self.alpha_vantage:
            raise ValueError("Alpha Vantage API key not configured")
        
        try:
            # Convert pair format (EURUSD -> EUR, USD)
            from_currency = pair[:3]
            to_currency = pair[3:]
            
            if interval == '1min':
                data, meta_data = self.alpha_vantage.get_currency_exchange_intraday(
                    from_symbol=from_currency,
                    to_symbol=to_currency,
                    interval='1min',
                    outputsize='full'
                )
            elif interval == 'daily':
                data, meta_data = self.alpha_vantage.get_currency_exchange_daily(
                    from_symbol=from_currency,
                    to_symbol=to_currency,
                    outputsize='full'
                )
            else:
                raise ValueError(f"Unsupported interval: {interval}")
            
            # Convert to our standard format
            formatted_data = []
            for timestamp, values in data.items():
                formatted_data.append({
                    'timestamp': datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
                    'pair': pair,
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': None,  # Alpha Vantage doesn't provide volume for forex
                    'spread': None,
                    'tick_volume': None,
                    'source': 'alpha_vantage'
                })
            
            logger.info(f"Retrieved {len(formatted_data)} records from Alpha Vantage for {pair}")
            return formatted_data
            
        except Exception as e:
            logger.error(f"Alpha Vantage API error for {pair}: {e}")
            raise
    
    def get_historical_data_yfinance(self, pair: str, 
                                   period: str = '1y',
                                   interval: str = '1m') -> List[Dict]:
        """Get historical data from Yahoo Finance (limited forex support)."""
        try:
            # Convert pair format for Yahoo Finance (EURUSD -> EURUSD=X)
            yahoo_symbol = f"{pair}=X"
            
            # Yahoo Finance has limitations on intraday data
            if interval == '1m':
                period = '7d'  # Yahoo only allows 7 days for 1-minute data
            
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data returned from Yahoo Finance for {pair}")
                return []
            
            # Convert to our standard format
            formatted_data = []
            for timestamp, row in data.iterrows():
                formatted_data.append({
                    'timestamp': timestamp.to_pydatetime(),
                    'pair': pair,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']) if pd.notna(row['Volume']) else None,
                    'spread': None,
                    'tick_volume': None,
                    'source': 'yahoo_finance'
                })
            
            logger.info(f"Retrieved {len(formatted_data)} records from Yahoo Finance for {pair}")
            return formatted_data
            
        except Exception as e:
            logger.error(f"Yahoo Finance API error for {pair}: {e}")
            return []
    
    async def get_historical_data_oanda(self, pair: str, 
                                      granularity: str = 'M1',
                                      count: int = 5000) -> List[Dict]:
        """Get historical data from OANDA API."""
        if not self.config.OANDA_API_KEY:
            raise ValueError("OANDA API key not configured")
        
        try:
            # Convert pair format (EURUSD -> EUR_USD)
            oanda_pair = f"{pair[:3]}_{pair[3:]}"
            
            headers = {
                'Authorization': f'Bearer {self.config.OANDA_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'granularity': granularity,
                'count': count
            }
            
            url = f"{API_ENDPOINTS['oanda']['base_url']}/instruments/{oanda_pair}/candles"
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    formatted_data = []
                    for candle in data.get('candles', []):
                        if candle.get('complete'):
                            mid = candle['mid']
                            formatted_data.append({
                                'timestamp': datetime.fromisoformat(candle['time'].replace('Z', '+00:00')),
                                'pair': pair,
                                'open': float(mid['o']),
                                'high': float(mid['h']),
                                'low': float(mid['l']),
                                'close': float(mid['c']),
                                'volume': int(candle.get('volume', 0)),
                                'spread': None,  # Could calculate from bid/ask if available
                                'tick_volume': int(candle.get('volume', 0)),
                                'source': 'oanda'
                            })
                    
                    logger.info(f"Retrieved {len(formatted_data)} records from OANDA for {pair}")
                    return formatted_data
                else:
                    logger.error(f"OANDA API error {response.status} for {pair}")
                    return []
                    
        except Exception as e:
            logger.error(f"OANDA API error for {pair}: {e}")
            return []
    
    def get_forex_data_with_fallback(self, pair: str, 
                                   data_type: str = 'historical',
                                   **kwargs) -> List[Dict]:
        """Get forex data with fallback to multiple sources."""
        data_sources = []
        
        # Try Alpha Vantage first (most reliable for historical data)
        if self.alpha_vantage and data_type == 'historical':
            try:
                data = self.get_historical_data_alpha_vantage(pair, **kwargs)
                if data:
                    data_sources.append(('alpha_vantage', data))
            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {pair}: {e}")
        
        # Try Yahoo Finance as fallback
        try:
            data = self.get_historical_data_yfinance(pair, **kwargs)
            if data:
                data_sources.append(('yahoo_finance', data))
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {pair}: {e}")
        
        # Return the best available data source
        if data_sources:
            # Prefer Alpha Vantage, then others
            for source_name, data in data_sources:
                if source_name == 'alpha_vantage':
                    return data
            
            # Return first available if Alpha Vantage not available
            return data_sources[0][1]
        
        logger.error(f"All data sources failed for {pair}")
        return []
    
    async def get_real_time_data(self, pairs: List[str]) -> List[Dict]:
        """Get real-time data for multiple pairs."""
        real_time_data = []
        
        # Use OANDA for real-time data if available
        if self.config.OANDA_API_KEY:
            for pair in pairs:
                try:
                    data = await self.get_historical_data_oanda(pair, 'M1', 1)
                    if data:
                        real_time_data.extend(data)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Real-time data error for {pair}: {e}")
        
        # Fallback to Yahoo Finance for real-time quotes
        if not real_time_data:
            for pair in pairs:
                try:
                    data = self.get_historical_data_yfinance(pair, '1d', '1m')
                    if data:
                        # Get only the latest data point
                        real_time_data.append(data[-1])
                except Exception as e:
                    logger.error(f"Yahoo Finance real-time error for {pair}: {e}")
        
        return real_time_data
    
    def get_historical_data_bulk(self, pairs: List[str], 
                               years: int = 10) -> Dict[str, List[Dict]]:
        """Get bulk historical data for multiple pairs."""
        bulk_data = {}
        
        for pair in pairs:
            logger.info(f"Fetching historical data for {pair}")
            
            try:
                # Try different intervals and combine
                data_1min = []
                data_daily = []
                
                # Get recent 1-minute data (limited by APIs)
                try:
                    data_1min = self.get_forex_data_with_fallback(
                        pair, 'historical', interval='1min'
                    )
                except Exception as e:
                    logger.warning(f"Failed to get 1-minute data for {pair}: {e}")
                
                # Get daily data for longer history
                try:
                    data_daily = self.get_forex_data_with_fallback(
                        pair, 'historical', interval='daily'
                    )
                except Exception as e:
                    logger.warning(f"Failed to get daily data for {pair}: {e}")
                
                # Combine data, prioritizing 1-minute data for recent periods
                combined_data = data_1min + data_daily
                
                # Remove duplicates and sort
                seen_timestamps = set()
                unique_data = []
                for record in sorted(combined_data, key=lambda x: x['timestamp']):
                    timestamp_key = (record['timestamp'], record['source'])
                    if timestamp_key not in seen_timestamps:
                        seen_timestamps.add(timestamp_key)
                        unique_data.append(record)
                
                bulk_data[pair] = unique_data
                logger.info(f"Collected {len(unique_data)} records for {pair}")
                
                # Rate limiting between pairs
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to collect data for {pair}: {e}")
                bulk_data[pair] = []
        
        return bulk_data
    
    def validate_data(self, data: List[Dict]) -> List[Dict]:
        """Validate and clean forex data."""
        validated_data = []
        
        for record in data:
            try:
                # Check required fields
                required_fields = ['timestamp', 'pair', 'open', 'high', 'low', 'close', 'source']
                if not all(field in record for field in required_fields):
                    logger.warning(f"Missing required fields in record: {record}")
                    continue
                
                # Validate OHLC data
                ohlc = [record['open'], record['high'], record['low'], record['close']]
                if any(price <= 0 for price in ohlc):
                    logger.warning(f"Invalid OHLC prices in record: {record}")
                    continue
                
                # Validate high/low logic
                if record['high'] < max(record['open'], record['close']) or \
                   record['low'] > min(record['open'], record['close']):
                    logger.warning(f"Invalid high/low logic in record: {record}")
                    continue
                
                # Validate timestamp
                if not isinstance(record['timestamp'], datetime):
                    logger.warning(f"Invalid timestamp in record: {record}")
                    continue
                
                validated_data.append(record)
                
            except Exception as e:
                logger.warning(f"Error validating record {record}: {e}")
                continue
        
        logger.info(f"Validated {len(validated_data)} out of {len(data)} records")
        return validated_data

class DataEnricher:
    """Enriches forex data with additional calculated fields."""
    
    @staticmethod
    def calculate_technical_indicators(data: List[Dict]) -> List[Dict]:
        """Calculate basic technical indicators."""
        if not data:
            return data
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp')
        
        # Calculate simple moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Calculate price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        
        # Calculate volatility (rolling standard deviation)
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Convert back to list of dictionaries
        enriched_data = df.to_dict('records')
        
        # Clean NaN values
        for record in enriched_data:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
        
        return enriched_data
    
    @staticmethod
    def calculate_spreads(data: List[Dict], typical_spread: float = 0.0001) -> List[Dict]:
        """Calculate estimated spreads for data that doesn't include them."""
        for record in data:
            if record.get('spread') is None:
                # Estimate spread based on volatility or use typical spread
                if 'volatility' in record and record['volatility']:
                    estimated_spread = record['volatility'] * 0.1  # Simple estimation
                else:
                    estimated_spread = typical_spread
                
                record['spread'] = estimated_spread
        
        return data