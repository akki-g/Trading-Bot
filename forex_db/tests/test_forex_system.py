#!/usr/bin/env python3
"""
Comprehensive test suite for the Forex Data Collection System.

Tests cover:
- Database operations
- Data collection
- System configuration
- Data validation
- Performance testing
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Import system components
from config import ForexConfig
from database_manager import DatabaseManager
from data_collector import ForexDataCollector, DataEnricher
from forex_system import ForexDataSystem

class TestConfig:
    """Test configuration class."""
    
    @pytest.fixture(scope="session")
    def test_config(self):
        """Create test configuration."""
        config = ForexConfig()
        # Override with test values
        config.DB_NAME = 'forex_test_db'
        config.DB_USER = 'test_user'
        config.DB_PASSWORD = 'test_password'
        config.MAJOR_PAIRS = ['EURUSD', 'GBPUSD']  # Limit for testing
        config.BATCH_SIZE = 100
        return config
    
    @pytest.fixture(scope="session")
    def test_database(self, test_config):
        """Create test database."""
        # Create test database
        try:
            conn = psycopg2.connect(
                host=test_config.DB_HOST,
                port=test_config.DB_PORT,
                database='postgres',
                user='postgres',
                password='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS {test_config.DB_NAME}")
                cursor.execute(f"CREATE DATABASE {test_config.DB_NAME}")
                
                # Create test user
                cursor.execute(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT FROM pg_user WHERE usename = '{test_config.DB_USER}') THEN
                            CREATE USER {test_config.DB_USER} WITH PASSWORD '{test_config.DB_PASSWORD}';
                        END IF;
                    END
                    $$;
                """)
                
                cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {test_config.DB_NAME} TO {test_config.DB_USER}")
            
            conn.close()
            yield test_config
            
            # Cleanup
            conn = psycopg2.connect(
                host=test_config.DB_HOST,
                port=test_config.DB_PORT,
                database='postgres',
                user='postgres',
                password='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS {test_config.DB_NAME}")
                cursor.execute(f"DROP USER IF EXISTS {test_config.DB_USER}")
            
            conn.close()
            
        except Exception as e:
            pytest.skip(f"Could not create test database: {e}")

class TestDatabaseManager:
    """Test database operations."""
    
    def test_database_connection(self, test_database):
        """Test database connection."""
        db_manager = DatabaseManager()
        db_manager.config = test_database
        db_manager.connection_params.update({
            'database': test_database.DB_NAME,
            'user': test_database.DB_USER,
            'password': test_database.DB_PASSWORD
        })
        
        assert db_manager.test_connection() == True
    
    def test_database_setup(self, test_database):
        """Test database schema setup."""
        db_manager = DatabaseManager()
        db_manager.config = test_database
        db_manager.connection_params.update({
            'database': test_database.DB_NAME,
            'user': test_database.DB_USER,
            'password': test_database.DB_PASSWORD
        })
        
        assert db_manager.setup_database() == True
        
        # Verify tables exist
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                tables = [row[0] for row in cursor.fetchall()]
                
                assert 'forex_data' in tables
                assert 'forex_metadata' in tables
    
    def test_insert_forex_data(self, test_database):
        """Test forex data insertion."""
        db_manager = DatabaseManager()
        db_manager.config = test_database
        db_manager.connection_params.update({
            'database': test_database.DB_NAME,
            'user': test_database.DB_USER,
            'password': test_database.DB_PASSWORD
        })
        
        # Setup database first
        db_manager.setup_database()
        
        # Create test data
        test_data = [
            {
                'timestamp': datetime.now(),
                'pair': 'EURUSD',
                'open': 1.1000,
                'high': 1.1010,
                'low': 1.0990,
                'close': 1.1005,
                'volume': 1000,
                'spread': 0.0001,
                'tick_volume': 50,
                'source': 'test'
            }
        ]
        
        assert db_manager.insert_forex_data(test_data) == True
        
        # Verify data was inserted
        data = db_manager.get_forex_data(
            'EURUSD',
            datetime.now() - timedelta(hours=1),
            datetime.now() + timedelta(hours=1)
        )
        
        assert len(data) == 1
        assert data.iloc[0]['pair'] == 'EURUSD'
        assert data.iloc[0]['close'] == 1.1005
    
    def test_get_latest_timestamp(self, test_database):
        """Test getting latest timestamp."""
        db_manager = DatabaseManager()
        db_manager.config = test_database
        db_manager.connection_params.update({
            'database': test_database.DB_NAME,
            'user': test_database.DB_USER,
            'password': test_database.DB_PASSWORD
        })
        
        db_manager.setup_database()
        
        # Should return None for empty table
        latest = db_manager.get_latest_timestamp('EURUSD', 'test')
        assert latest is None
        
        # Insert test data
        test_timestamp = datetime.now()
        test_data = [{
            'timestamp': test_timestamp,
            'pair': 'EURUSD',
            'open': 1.1000,
            'high': 1.1010,
            'low': 1.0990,
            'close': 1.1005,
            'source': 'test'
        }]
        
        db_manager.insert_forex_data(test_data)
        
        # Should return the timestamp
        latest = db_manager.get_latest_timestamp('EURUSD', 'test')
        assert latest is not None
        assert abs((latest - test_timestamp).total_seconds()) < 1
    
    def test_data_statistics(self, test_database):
        """Test data statistics generation."""
        db_manager = DatabaseManager()
        db_manager.config = test_database
        db_manager.connection_params.update({
            'database': test_database.DB_NAME,
            'user': test_database.DB_USER,
            'password': test_database.DB_PASSWORD
        })
        
        db_manager.setup_database()
        
        # Insert test data
        test_data = []
        for i in range(10):
            test_data.append({
                'timestamp': datetime.now() - timedelta(minutes=i),
                'pair': 'EURUSD',
                'open': 1.1000 + i * 0.0001,
                'high': 1.1010 + i * 0.0001,
                'low': 1.0990 + i * 0.0001,
                'close': 1.1005 + i * 0.0001,
                'source': 'test'
            })
        
        db_manager.insert_forex_data(test_data)
        
        stats = db_manager.get_data_statistics()
        assert stats is not None
        assert 'pair_statistics' in stats
        assert 'total_pairs' in stats
        assert 'total_records' in stats
        assert stats['total_records'] == 10

class TestDataCollector:
    """Test data collection functionality."""
    
    @pytest.fixture
    def mock_yfinance_data(self):
        """Mock yfinance data."""
        mock_data = pd.DataFrame({
            'Open': [1.1000, 1.1005, 1.1010],
            'High': [1.1010, 1.1015, 1.1020],
            'Low': [1.0990, 1.0995, 1.1000],
            'Close': [1.1005, 1.1010, 1.1015],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1min'))
        
        return mock_data
    
    @pytest.fixture
    def mock_alpha_vantage_data(self):
        """Mock Alpha Vantage data."""
        return {
            '2024-01-01 09:00:00': {
                '1. open': '1.1000',
                '2. high': '1.1010',
                '3. low': '1.0990',
                '4. close': '1.1005'
            },
            '2024-01-01 09:01:00': {
                '1. open': '1.1005',
                '2. high': '1.1015',
                '3. low': '1.0995',
                '4. close': '1.1010'
            }
        }
    
    @patch('yfinance.Ticker')
    def test_yahoo_finance_collection(self, mock_ticker, mock_yfinance_data):
        """Test Yahoo Finance data collection."""
        # Setup mock
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_yfinance_data
        mock_ticker.return_value = mock_ticker_instance
        
        collector = ForexDataCollector()
        data = collector.get_historical_data_yfinance('EURUSD', '1d', '1m')
        
        assert len(data) == 3
        assert data[0]['pair'] == 'EURUSD'
        assert data[0]['source'] == 'yahoo_finance'
        assert data[0]['open'] == 1.1000
        assert data[0]['close'] == 1.1005
    
    @patch('alpha_vantage.foreignexchange.ForeignExchange.get_currency_exchange_intraday')
    def test_alpha_vantage_collection(self, mock_av, mock_alpha_vantage_data):
        """Test Alpha Vantage data collection."""
        # Setup mock
        mock_av.return_value = (mock_alpha_vantage_data, {})
        
        collector = ForexDataCollector()
        collector.alpha_vantage = Mock()
        collector.alpha_vantage.get_currency_exchange_intraday = mock_av
        
        data = collector.get_historical_data_alpha_vantage('EURUSD', '1min')
        
        assert len(data) == 2
        assert data[0]['pair'] == 'EURUSD'
        assert data[0]['source'] == 'alpha_vantage'
        assert data[0]['open'] == 1.1000
    
    def test_data_validation(self):
        """Test data validation."""
        collector = ForexDataCollector()
        
        # Valid data
        valid_data = [{
            'timestamp': datetime.now(),
            'pair': 'EURUSD',
            'open': 1.1000,
            'high': 1.1010,
            'low': 1.0990,
            'close': 1.1005,
            'source': 'test'
        }]
        
        validated = collector.validate_data(valid_data)
        assert len(validated) == 1
        
        # Invalid data - high < close
        invalid_data = [{
            'timestamp': datetime.now(),
            'pair': 'EURUSD',
            'open': 1.1000,
            'high': 1.1000,  # Invalid: high should be >= close
            'low': 1.0990,
            'close': 1.1005,
            'source': 'test'
        }]
        
        validated = collector.validate_data(invalid_data)
        assert len(validated) == 0
        
        # Invalid data - negative price
        invalid_data2 = [{
            'timestamp': datetime.now(),
            'pair': 'EURUSD',
            'open': -1.1000,  # Invalid: negative price
            'high': 1.1010,
            'low': 1.0990,
            'close': 1.1005,
            'source': 'test'
        }]
        
        validated = collector.validate_data(invalid_data2)
        assert len(validated) == 0

class TestDataEnricher:
    """Test data enrichment functionality."""
    
    def test_technical_indicators(self):
        """Test technical indicator calculations."""
        # Create test data
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        test_data = []
        
        for i, date in enumerate(dates):
            price = 1.1000 + (i * 0.0001) + (0.0005 * (i % 10 - 5))  # Trending with noise
            test_data.append({
                'timestamp': date,
                'pair': 'EURUSD',
                'open': price,
                'high': price + 0.0002,
                'low': price - 0.0002,
                'close': price + 0.0001,
                'source': 'test'
            })
        
        enricher = DataEnricher()
        enriched_data = enricher.calculate_technical_indicators(test_data)
        
        assert len(enriched_data) == 100
        
        # Check that indicators were calculated
        last_record = enriched_data[-1]
        assert 'sma_10' in last_record
        assert 'sma_20' in last_record
        assert 'sma_50' in last_record
        assert 'volatility' in last_record
        
        # SMA values should be reasonable
        assert last_record['sma_10'] is not None
        assert last_record['sma_20'] is not None
    
    def test_spread_calculation(self):
        """Test spread calculation."""
        test_data = [{
            'timestamp': datetime.now(),
            'pair': 'EURUSD',
            'open': 1.1000,
            'high': 1.1010,
            'low': 1.0990,
            'close': 1.1005,
            'spread': None,
            'source': 'test'
        }]
        
        enricher = DataEnricher()
        enriched_data = enricher.calculate_spreads(test_data)
        
        assert enriched_data[0]['spread'] is not None
        assert enriched_data[0]['spread'] > 0

class TestForexSystem:
    """Test the main forex system."""
    
    @pytest.fixture
    def mock_system(self, test_database):
        """Create a mock forex system for testing."""
        system = ForexDataSystem()
        system.config = test_database
        system.db_manager.config = test_database
        system.db_manager.connection_params.update({
            'database': test_database.DB_NAME,
            'user': test_database.DB_USER,
            'password': test_database.DB_PASSWORD
        })
        return system
    
    def test_system_initialization(self, mock_system):
        """Test system initialization."""
        assert mock_system.initialize_system() == True
    
    @patch('data_collector.ForexDataCollector.get_forex_data_with_fallback')
    def test_historical_data_collection(self, mock_collector, mock_system):
        """Test historical data collection."""
        # Mock data collection
        mock_data = [{
            'timestamp': datetime.now(),
            'pair': 'EURUSD',
            'open': 1.1000,
            'high': 1.1010,
            'low': 1.0990,
            'close': 1.1005,
            'source': 'test'
        }]
        
        mock_collector.return_value = mock_data
        
        # Initialize system first
        mock_system.initialize_system()
        
        # Test collection
        result = asyncio.run(mock_system.collect_historical_data(['EURUSD']))
        assert result == True
    
    def test_data_validation(self, mock_system):
        """Test system data validation."""
        mock_system.initialize_system()
        
        # Should not fail even with no data
        result = mock_system.validate_data_integrity()
        assert result == True

class TestPerformance:
    """Performance and stress tests."""
    
    def test_large_data_insertion(self, test_database):
        """Test insertion of large datasets."""
        db_manager = DatabaseManager()
        db_manager.config = test_database
        db_manager.connection_params.update({
            'database': test_database.DB_NAME,
            'user': test_database.DB_USER,
            'password': test_database.DB_PASSWORD
        })
        
        db_manager.setup_database()
        
        # Create large dataset (1000 records)
        large_dataset = []
        base_time = datetime.now()
        
        for i in range(1000):
            large_dataset.append({
                'timestamp': base_time + timedelta(minutes=i),
                'pair': 'EURUSD',
                'open': 1.1000 + (i * 0.000001),
                'high': 1.1010 + (i * 0.000001),
                'low': 1.0990 + (i * 0.000001),
                'close': 1.1005 + (i * 0.000001),
                'source': 'performance_test'
            })
        
        # Measure insertion time
        import time
        start_time = time.time()
        result = db_manager.insert_forex_data(large_dataset)
        end_time = time.time()
        
        assert result == True
        assert (end_time - start_time) < 10  # Should complete within 10 seconds
        
        # Verify all data was inserted
        stats = db_manager.get_data_statistics()
        assert stats['total_records'] >= 1000
    
    def test_concurrent_access(self, test_database):
        """Test concurrent database access."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def worker():
            db_manager = DatabaseManager()
            db_manager.config = test_database
            db_manager.connection_params.update({
                'database': test_database.DB_NAME,
                'user': test_database.DB_USER,
                'password': test_database.DB_PASSWORD
            })
            
            try:
                # Test connection
                connection_result = db_manager.test_connection()
                results.put(connection_result)
            except Exception as e:
                results.put(False)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check results
        success_count = 0
        while not results.empty():
            if results.get():
                success_count += 1
        
        assert success_count == 5  # All connections should succeed

class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.integration
    def test_full_system_workflow(self, test_database):
        """Test complete system workflow."""
        # This test requires actual network access and may be slow
        # Skip if running in CI environment
        if os.getenv('CI'):
            pytest.skip("Skipping integration test in CI environment")
        
        system = ForexDataSystem()
        system.config = test_database
        system.db_manager.config = test_database
        system.db_manager.connection_params.update({
            'database': test_database.DB_NAME,
            'user': test_database.DB_USER,
            'password': test_database.DB_PASSWORD
        })
        
        # Test system initialization
        assert system.initialize_system() == True
        
        # Test real data collection (limited)
        # Note: This will make actual API calls
        result = asyncio.run(system.collect_historical_data(['EURUSD']))
        
        # Even if collection fails due to API limits, system should handle gracefully
        assert isinstance(result, bool)
        
        # Test data validation
        validation_result = system.validate_data_integrity()
        assert validation_result == True

# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    if config.getoption("--runslow"):
        # Don't skip slow tests
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_slow)

# Fixtures for test data
@pytest.fixture
def sample_forex_data():
    """Sample forex data for testing."""
    return [
        {
            'timestamp': datetime(2024, 1, 1, 9, 0, 0),
            'pair': 'EURUSD',
            'open': 1.1000,
            'high': 1.1010,
            'low': 1.0990,
            'close': 1.1005,
            'volume': 1000,
            'source': 'test'
        },
        {
            'timestamp': datetime(2024, 1, 1, 9, 1, 0),
            'pair': 'EURUSD',
            'open': 1.1005,
            'high': 1.1015,
            'low': 1.0995,
            'close': 1.1010,
            'volume': 1100,
            'source': 'test'
        }
    ]

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])