#!/usr/bin/env python3
from data_collector import ForexDataCollector
from database_manager import DatabaseManager
import asyncio
import logging

# Setup logging to see more details
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_data_collection():
    """Test data collection from APIs."""
    print("Testing data collection...")
    
    collector = ForexDataCollector()
    db_manager = DatabaseManager()
    
    # Test Alpha Vantage connection with daily data (more reliable format)
    try:
        print("Testing Alpha Vantage API with daily data...")
        data = collector.get_historical_data_alpha_vantage('EURUSD', 'daily')
        if data:
            print(f"✅ Alpha Vantage: Retrieved {len(data)} data points")
            
            # Test data insertion
            validated_data = collector.validate_data(data[:5])  # Test with just 5 records
            if validated_data:
                success = db_manager.insert_forex_data(validated_data)
                if success:
                    print("✅ Alpha Vantage data successfully inserted into database!")
                else:
                    print("❌ Failed to insert Alpha Vantage data into database")
        else:
            print("⚠️ Alpha Vantage: No data retrieved")
    except Exception as e:
        print(f"❌ Alpha Vantage error: {e}")
    
    # Test Yahoo Finance connection
    try:
        print("Testing Yahoo Finance API...")
        data = collector.get_historical_data_yfinance('GBPUSD', '5d', '1h')
        if data:
            print(f"✅ Yahoo Finance: Retrieved {len(data)} data points")
            
            # Test data insertion
            validated_data = collector.validate_data(data[:5])  # Test with just 5 records
            if validated_data:
                success = db_manager.insert_forex_data(validated_data)
                if success:
                    print("✅ Yahoo Finance data successfully inserted into database!")
                else:
                    print("❌ Failed to insert Yahoo Finance data into database")
        else:
            print("⚠️ Yahoo Finance: No data retrieved")
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
    
    # Check what's in the database
    try:
        print("\n📊 Checking database contents...")
        stats = db_manager.get_data_statistics()
        if stats:
            print(f"Total records in database: {stats.get('total_records', 0)}")
            for pair_stat in stats.get('pair_statistics', []):
                print(f"  {pair_stat['pair']} ({pair_stat['source']}): {pair_stat['record_count']} records")
        else:
            print("No data found in database")
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    asyncio.run(test_data_collection())#!/usr/bin/env python3
from data_collector import ForexDataCollector
from database_manager import DatabaseManager
import asyncio
import logging

# Setup logging to see more details
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_data_collection():
    """Test data collection from APIs."""
    print("Testing data collection...")
    
    collector = ForexDataCollector()
    db_manager = DatabaseManager()
    
    # Test Alpha Vantage connection with daily data (more reliable format)
    try:
        print("Testing Alpha Vantage API with daily data...")
        data = collector.get_historical_data_alpha_vantage('EURUSD', 'daily')
        if data:
            print(f"✅ Alpha Vantage: Retrieved {len(data)} data points")
            
            # Test data insertion
            validated_data = collector.validate_data(data[:5])  # Test with just 5 records
            if validated_data:
                success = db_manager.insert_forex_data(validated_data)
                if success:
                    print("✅ Alpha Vantage data successfully inserted into database!")
                else:
                    print("❌ Failed to insert Alpha Vantage data into database")
        else:
            print("⚠️ Alpha Vantage: No data retrieved")
    except Exception as e:
        print(f"❌ Alpha Vantage error: {e}")
    
    # Test Yahoo Finance connection
    try:
        print("Testing Yahoo Finance API...")
        data = collector.get_historical_data_yfinance('GBPUSD', '5d', '1h')
        if data:
            print(f"✅ Yahoo Finance: Retrieved {len(data)} data points")
            
            # Test data insertion
            validated_data = collector.validate_data(data[:5])  # Test with just 5 records
            if validated_data:
                success = db_manager.insert_forex_data(validated_data)
                if success:
                    print("✅ Yahoo Finance data successfully inserted into database!")
                else:
                    print("❌ Failed to insert Yahoo Finance data into database")
        else:
            print("⚠️ Yahoo Finance: No data retrieved")
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
    
    # Check what's in the database
    try:
        print("\n📊 Checking database contents...")
        stats = db_manager.get_data_statistics()
        if stats:
            print(f"Total records in database: {stats.get('total_records', 0)}")
            for pair_stat in stats.get('pair_statistics', []):
                print(f"  {pair_stat['pair']} ({pair_stat['source']}): {pair_stat['record_count']} records")
        else:
            print("No data found in database")
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    asyncio.run(test_data_collection())