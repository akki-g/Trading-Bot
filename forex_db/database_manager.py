import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
from contextlib import contextmanager
from config import ForexConfig, DB_SCHEMA

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database operations for the forex data system."""
    
    def __init__(self):
        self.config = ForexConfig()
        self.connection_params = {
            'host': self.config.DB_HOST,
            'port': self.config.DB_PORT,
            'database': self.config.DB_NAME,
            'user': self.config.DB_USER,
            'password': self.config.DB_PASSWORD
        }
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def setup_database(self) -> bool:
        """Set up the database schema and TimescaleDB hypertables."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Enable TimescaleDB extension
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                    
                    # Create tables
                    for table_name, schema in DB_SCHEMA.items():
                        self._create_table(cursor, table_name, schema)
                    
                    # Create indexes
                    self._create_indexes(cursor)
                    
                    # Insert initial metadata
                    self._insert_initial_metadata(cursor)
                    
                    conn.commit()
                    logger.info("Database setup completed successfully")
                    return True
                    
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def _create_table(self, cursor, table_name: str, schema: Dict):
        """Create a table with the given schema."""
        columns_sql = ', '.join(schema['columns'])
        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});"
        
        cursor.execute(create_sql)
        logger.info(f"Created table: {table_name}")
        
        # Create hypertable if specified
        if 'hypertable' in schema:
            try:
                cursor.execute(schema['hypertable'])
                logger.info(f"Created hypertable: {table_name}")
            except psycopg2.Error as e:
                if "already a hypertable" not in str(e):
                    logger.warning(f"Hypertable creation warning for {table_name}: {e}")
    
    def _create_indexes(self, cursor):
        """Create all indexes for the database."""
        for table_name, schema in DB_SCHEMA.items():
            if 'indexes' in schema:
                for index_sql in schema['indexes']:
                    try:
                        cursor.execute(index_sql)
                        logger.info(f"Created index for {table_name}")
                    except psycopg2.Error as e:
                        logger.warning(f"Index creation warning: {e}")
    
    def _insert_initial_metadata(self, cursor):
        """Insert initial forex pair metadata."""
        metadata_insert = """
        INSERT INTO forex_metadata (pair, base_currency, quote_currency, pip_location)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (pair) DO NOTHING;
        """
        
        metadata_data = []
        for pair in self.config.MAJOR_PAIRS:
            base = pair[:3]
            quote = pair[3:]
            pip_location = 4 if quote != 'JPY' else 2
            metadata_data.append((pair, base, quote, pip_location))
        
        execute_batch(cursor, metadata_insert, metadata_data)
        logger.info(f"Inserted metadata for {len(metadata_data)} forex pairs")
    
    def insert_forex_data(self, data: List[Dict], batch_size: int = 1000) -> bool:
        """Insert forex data in batches."""
        if not data:
            return True
            
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    insert_sql = """
                    INSERT INTO forex_data 
                    (timestamp, pair, open, high, low, close, volume, spread, tick_volume, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp, pair, source) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        spread = EXCLUDED.spread,
                        tick_volume = EXCLUDED.tick_volume;
                    """
                    
                    # Process data in batches
                    for i in range(0, len(data), batch_size):
                        batch = data[i:i + batch_size]
                        batch_values = [
                            (
                                item['timestamp'],
                                item['pair'],
                                item['open'],
                                item['high'],
                                item['low'],
                                item['close'],
                                item.get('volume'),
                                item.get('spread'),
                                item.get('tick_volume'),
                                item['source']
                            )
                            for item in batch
                        ]
                        
                        execute_batch(cursor, insert_sql, batch_values)
                        logger.info(f"Inserted batch of {len(batch)} records")
                    
                    conn.commit()
                    logger.info(f"Successfully inserted {len(data)} forex data records")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to insert forex data: {e}")
            return False
    
    def get_latest_timestamp(self, pair: str, source: str = None) -> Optional[datetime]:
        """Get the latest timestamp for a given pair and source."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    if source:
                        query = """
                        SELECT MAX(timestamp) FROM forex_data 
                        WHERE pair = %s AND source = %s
                        """
                        cursor.execute(query, (pair, source))
                    else:
                        query = """
                        SELECT MAX(timestamp) FROM forex_data 
                        WHERE pair = %s
                        """
                        cursor.execute(query, (pair,))
                    
                    result = cursor.fetchone()
                    return result[0] if result[0] else None
                    
        except Exception as e:
            logger.error(f"Failed to get latest timestamp: {e}")
            return None
    
    def get_forex_data(self, pair: str, start_time: datetime, 
                      end_time: datetime, source: str = None) -> pd.DataFrame:
        """Retrieve forex data for analysis."""
        try:
            with self.get_connection() as conn:
                if source:
                    query = """
                    SELECT timestamp, pair, open, high, low, close, volume, spread, tick_volume, source
                    FROM forex_data
                    WHERE pair = %s AND timestamp BETWEEN %s AND %s AND source = %s
                    ORDER BY timestamp ASC
                    """
                    params = (pair, start_time, end_time, source)
                else:
                    query = """
                    SELECT timestamp, pair, open, high, low, close, volume, spread, tick_volume, source
                    FROM forex_data
                    WHERE pair = %s AND timestamp BETWEEN %s AND %s
                    ORDER BY timestamp ASC
                    """
                    params = (pair, start_time, end_time)
                
                df = pd.read_sql_query(query, conn, params=params)
                logger.info(f"Retrieved {len(df)} records for {pair}")
                return df
                
        except Exception as e:
            logger.error(f"Failed to retrieve forex data: {e}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> bool:
        """Clean up data older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    delete_query = """
                    DELETE FROM forex_data 
                    WHERE timestamp < %s
                    """
                    cursor.execute(delete_query, (cutoff_date,))
                    deleted_count = cursor.rowcount
                    
                    conn.commit()
                    logger.info(f"Cleaned up {deleted_count} old records")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False
    
    def get_data_statistics(self) -> Dict:
        """Get statistics about the stored data."""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    stats_query = """
                    SELECT 
                        pair,
                        source,
                        COUNT(*) as record_count,
                        MIN(timestamp) as earliest_data,
                        MAX(timestamp) as latest_data,
                        AVG(close) as avg_close,
                        MIN(close) as min_close,
                        MAX(close) as max_close
                    FROM forex_data
                    GROUP BY pair, source
                    ORDER BY pair, source
                    """
                    
                    cursor.execute(stats_query)
                    results = cursor.fetchall()
                    
                    return {
                        'pair_statistics': [dict(row) for row in results],
                        'total_pairs': len(set(row['pair'] for row in results)),
                        'total_records': sum(row['record_count'] for row in results)
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get data statistics: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    if result[0] == 1:
                        logger.info("Database connection test successful")
                        return True
            return False
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False