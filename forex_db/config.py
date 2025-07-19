import os
from dotenv import load_dotenv
from typing import Dict, List
import logging

# Load environment variables
load_dotenv()

class ForexConfig:
    """Configuration management for the forex data system."""
    
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'forex_data')
    DB_USER = os.getenv('DB_USER', 'forex_user')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'forex_password')
    
    # API Configuration
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    # Major Forex Pairs
    MAJOR_PAIRS = [
        'AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'
    ]
    
    # Data Configuration
    HISTORICAL_YEARS = 10
    UPDATE_INTERVAL_MINUTES = 1
    BATCH_SIZE = 1000
    MAX_RETRIES = 3
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'forex_system.log')
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get the complete database URL."""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
    
    @classmethod
    def setup_logging(cls):
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        )
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present."""
        required_vars = [
            'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            logging.error(f"Missing required configuration: {missing_vars}")
            return False
        
        if not cls.ALPHA_VANTAGE_API_KEY:
            logging.warning("ALPHA_VANTAGE_API_KEY not found. Some data sources may not work.")
        
        return True

# API Endpoints and Configuration
API_ENDPOINTS = {
    'alpha_vantage': {
        'base_url': 'https://www.alphavantage.co/query',
        'rate_limit': 5,  # requests per minute
        'functions': {
            'intraday': 'FX_INTRADAY',
            'daily': 'FX_DAILY',
            'weekly': 'FX_WEEKLY',
            'monthly': 'FX_MONTHLY'
        }
    }
}

# Database Schema Configuration
DB_SCHEMA = {
    'forex_data': {
        'columns': [
            'timestamp TIMESTAMPTZ NOT NULL',
            'pair VARCHAR(10) NOT NULL',
            'open DECIMAL(20,8) NOT NULL',
            'high DECIMAL(20,8) NOT NULL',
            'low DECIMAL(20,8) NOT NULL',
            'close DECIMAL(20,8) NOT NULL',
            'volume BIGINT',
            'spread DECIMAL(10,8)',
            'tick_volume INTEGER',
            'source VARCHAR(50) NOT NULL',
            'created_at TIMESTAMPTZ DEFAULT NOW()'
        ],
        'indexes': [
            'CREATE INDEX IF NOT EXISTS idx_forex_data_timestamp ON forex_data (timestamp DESC)',
            'CREATE INDEX IF NOT EXISTS idx_forex_data_pair ON forex_data (pair)',
            'CREATE INDEX IF NOT EXISTS idx_forex_data_pair_timestamp ON forex_data (pair, timestamp DESC)'
        ],
        'hypertable': 'SELECT create_hypertable(\'forex_data\', \'timestamp\', if_not_exists => TRUE)'
    },
    'forex_metadata': {
        'columns': [
            'pair VARCHAR(10) PRIMARY KEY',
            'base_currency VARCHAR(3) NOT NULL',
            'quote_currency VARCHAR(3) NOT NULL',
            'pip_location INTEGER DEFAULT 4',
            'margin_rate DECIMAL(5,4)',
            'last_updated TIMESTAMPTZ DEFAULT NOW()',
            'is_active BOOLEAN DEFAULT TRUE'
        ]
    }
}