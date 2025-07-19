#!/usr/bin/env python3
"""
CSV Data Manager for Forex Trading System
Handles bulk loading of historical forex data from CSV files and incremental updates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import psycopg2
from psycopg2.extras import execute_values
from database_manager import DatabaseManager
from config import ForexConfig

class CSVDataManager:
    """Manages CSV data import and incremental updates for forex data"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def validate_csv_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and clean CSV data"""
        errors = []
        original_count = len(df)
        
        # Check required columns
        required_cols = ['datetime', 'symbol', 'interval', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return df, errors
        
        # Convert datetime to pandas datetime64 first, then to Python datetime
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
            # Convert to Python datetime objects to avoid numpy.datetime64 issues with psycopg2
            df['datetime'] = [dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else dt for dt in df['datetime']]
        except Exception as e:
            errors.append(f"Error parsing datetime: {e}")
            return df, errors
        
        # Remove rows with invalid OHLC data
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            self.logger.warning(f"Removing {invalid_count} rows with invalid OHLC data")
            df = df[~invalid_ohlc]
        
        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0
        df['volume'] = df['volume'].fillna(0)
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['datetime', 'symbol', 'interval'])
        
        cleaned_count = len(df)
        if cleaned_count < original_count:
            self.logger.info(f"Cleaned data: {original_count} -> {cleaned_count} rows")
        
        return df, errors
    
    def load_csv_file(self, csv_path: str) -> pd.DataFrame:
        """Load and validate a single CSV file"""
        self.logger.info(f"Loading CSV file: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            df, errors = self.validate_csv_data(df)
            
            if errors:
                for error in errors:
                    self.logger.error(error)
                raise ValueError(f"CSV validation failed: {errors}")
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV {csv_path}: {e}")
            raise
    
    def load_all_forex_csvs(self, data_dir: str = ".") -> pd.DataFrame:
        """Load all forex CSV files from directory"""
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("fx_data_*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No forex CSV files found in {data_dir}")
        
        self.logger.info(f"Found {len(csv_files)} forex CSV files")
        
        all_data = []
        for csv_file in csv_files:
            try:
                df = self.load_csv_file(csv_file)
                all_data.append(df)
                self.logger.info(f"Loaded {len(df)} records from {csv_file.name}")
            except Exception as e:
                self.logger.error(f"Failed to load {csv_file}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid CSV data could be loaded")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure datetime column is consistently converted to Python datetime objects
        # This fixes the numpy.datetime64 issue that occurs during database insertion
        combined_df['datetime'] = [dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else dt for dt in pd.to_datetime(combined_df['datetime'])]
        
        combined_df = combined_df.sort_values(['symbol', 'interval', 'datetime'])
        
        self.logger.info(f"Combined dataset: {len(combined_df)} total records")
        return combined_df
    
    def create_enhanced_schema(self):
        """Create enhanced database schema for mixed resolution data"""
        # Split schema creation into main schema and optional compression policy
        main_schema_sql = """
        -- Drop existing table if it exists
        DROP TABLE IF EXISTS forex_data CASCADE;
        
        -- Create main forex data table
        CREATE TABLE forex_data (
            timestamp       TIMESTAMPTZ NOT NULL,
            pair           VARCHAR(10) NOT NULL,
            interval_type  VARCHAR(5) NOT NULL,  -- '1m', '1d', etc.
            open           DECIMAL(20,8) NOT NULL,
            high           DECIMAL(20,8) NOT NULL,
            low            DECIMAL(20,8) NOT NULL,
            close          DECIMAL(20,8) NOT NULL,
            volume         BIGINT DEFAULT 0,
            spread         DECIMAL(10,8),
            tick_volume    INTEGER,
            source         VARCHAR(50) DEFAULT 'CSV_IMPORT',
            created_at     TIMESTAMPTZ DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT valid_ohlc CHECK (
                high >= low AND 
                high >= open AND 
                high >= close AND 
                low <= open AND 
                low <= close
            ),
            CONSTRAINT valid_interval CHECK (interval_type IN ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'))
        );
        
        -- Create hypertable (TimescaleDB)
        SELECT create_hypertable('forex_data', 'timestamp', if_not_exists => TRUE);
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_forex_data_pair_interval 
            ON forex_data (pair, interval_type, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_forex_data_timestamp 
            ON forex_data (timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_forex_data_pair 
            ON forex_data (pair);
        CREATE INDEX IF NOT EXISTS idx_forex_data_source 
            ON forex_data (source);
        
        -- Create unique constraint to prevent duplicates
        CREATE UNIQUE INDEX IF NOT EXISTS idx_forex_data_unique 
            ON forex_data (timestamp, pair, interval_type);
        
        -- Create materialized view for latest data
        CREATE MATERIALIZED VIEW IF NOT EXISTS forex_latest AS
        SELECT DISTINCT ON (pair, interval_type)
            pair,
            interval_type,
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM forex_data
        ORDER BY pair, interval_type, timestamp DESC;
        
        -- Create index on materialized view
        CREATE UNIQUE INDEX IF NOT EXISTS idx_forex_latest_pair_interval 
            ON forex_latest (pair, interval_type);
        """
        
        # Compression policy SQL (optional, may fail if compression not enabled)
        compression_sql = """
        -- Add compression policy (compress data older than 7 days)
        SELECT add_compression_policy('forex_data', INTERVAL '7 days', if_not_exists => TRUE);
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Execute main schema creation
                    cursor.execute(main_schema_sql)
                    
                    # Try to add compression policy with error handling
                    try:
                        cursor.execute(compression_sql)
                        self.logger.info("Compression policy added successfully")
                    except Exception as compression_error:
                        self.logger.warning(f"Could not add compression policy (compression may not be enabled): {compression_error}")
                        # Continue without compression - this is not a critical failure
                    
                conn.commit()
            self.logger.info("Enhanced database schema created successfully")
        except Exception as e:
            self.logger.error(f"Error creating schema: {e}")
            raise
    
    def check_table_schema(self):
        """Check if the table has the enhanced schema with interval_type column"""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'forex_data' AND column_name = 'interval_type'
                    """)
                    result = cursor.fetchone()
                    return result is not None
        except Exception as e:
            self.logger.error(f"Error checking table schema: {e}")
            return False

    def bulk_insert_data(self, df: pd.DataFrame, batch_size: int = 10000):
        """Bulk insert CSV data into database"""
        if df.empty:
            self.logger.warning("No data to insert")
            return
        
        # Check if table has enhanced schema
        has_interval_type = self.check_table_schema()
        
        # Prepare data for insertion
        df_clean = df.copy()
        df_clean['pair'] = df_clean['symbol']
        
        # Ensure datetime is converted to Python datetime objects
        # Handle both pandas datetime64 and numpy datetime64 types
        if df_clean['datetime'].dtype.name.startswith('datetime64'):
            df_clean['timestamp'] = [dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else dt for dt in pd.to_datetime(df_clean['datetime'])]
        else:
            # If already Python datetime objects, just copy
            df_clean['timestamp'] = df_clean['datetime']
        
        # Select columns based on table schema
        if has_interval_type:
            df_clean['interval_type'] = df_clean['interval']
            insert_cols = ['timestamp', 'pair', 'interval_type', 'open', 'high', 'low', 'close', 'volume']
            insert_sql = """
            INSERT INTO forex_data (timestamp, pair, interval_type, open, high, low, close, volume, source)
            VALUES %s
            ON CONFLICT (timestamp, pair, interval_type) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source
            """
        else:
            # Legacy schema without interval_type
            insert_cols = ['timestamp', 'pair', 'open', 'high', 'low', 'close', 'volume']
            insert_sql = """
            INSERT INTO forex_data (timestamp, pair, open, high, low, close, volume, source)
            VALUES %s
            ON CONFLICT (timestamp, pair) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source
            """
            self.logger.warning("Using legacy schema without interval_type column")
        
        df_insert = df_clean[insert_cols].copy()
        
        # Convert to records and manually handle datetime conversion
        # to_records() converts back to numpy.datetime64, so we need to handle this manually
        records_list = []
        for _, row in df_insert.iterrows():
            timestamp = row['timestamp']
            # Ensure timestamp is Python datetime
            if hasattr(timestamp, 'to_pydatetime'):
                timestamp = timestamp.to_pydatetime()
            elif isinstance(timestamp, np.datetime64):
                timestamp = pd.to_datetime(timestamp).to_pydatetime()
            
            if has_interval_type:
                record = (
                    timestamp,
                    row['pair'],
                    row['interval_type'],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume'])
                )
            else:
                record = (
                    timestamp,
                    row['pair'],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume'])
                )
            records_list.append(record)
        
        # Debug: Check the record types
        self.logger.info(f"Record tuple example: {records_list[0]}")
        self.logger.info(f"Timestamp in record type: {type(records_list[0][0])}")
        self.logger.info(f"Using schema with interval_type: {has_interval_type}")
        
        records = records_list
        
        total_records = len(records)
        self.logger.info(f"Inserting {total_records} records in batches of {batch_size}")
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Process in batches
                    for i in range(0, total_records, batch_size):
                        batch = records[i:i + batch_size]
                        # Add source column to each record
                        batch_with_source = [tuple(list(record) + ['CSV_IMPORT']) for record in batch]
                        
                        execute_values(
                            cursor,
                            insert_sql,
                            batch_with_source,
                            template=None,
                            page_size=batch_size
                        )
                        
                        if i % (batch_size * 10) == 0:  # Progress update every 10 batches
                            progress = (i + len(batch)) / total_records * 100
                            self.logger.info(f"Progress: {progress:.1f}% ({i + len(batch)}/{total_records})")
                
                conn.commit()
            
            self.logger.info(f"Successfully inserted {total_records} records")
            
            # Refresh materialized view
            self.refresh_latest_view()
            
        except Exception as e:
            self.logger.error(f"Error during bulk insert: {e}")
            raise
    
    def refresh_latest_view(self):
        """Refresh the materialized view with latest data"""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY forex_latest;")
                conn.commit()
            self.logger.info("Refreshed forex_latest materialized view")
        except Exception as e:
            self.logger.warning(f"Could not refresh materialized view: {e}")
    
    def get_last_timestamp(self, pair: str, interval: str) -> Optional[datetime]:
        """Get the last timestamp for a specific pair and interval"""
        query = """
        SELECT MAX(timestamp) as last_timestamp
        FROM forex_data
        WHERE pair = %s AND interval_type = %s
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (pair, interval))
                    result = cursor.fetchone()
                    return result[0] if result and result[0] else None
        except Exception as e:
            self.logger.error(f"Error getting last timestamp: {e}")
            return None
    
    def get_data_gaps(self, pair: str, interval: str) -> List[Tuple[datetime, datetime]]:
        """Identify gaps in the data for a specific pair and interval"""
        if interval == '1d':
            expected_interval = timedelta(days=1)
        elif interval == '1m':
            expected_interval = timedelta(minutes=1)
        else:
            self.logger.warning(f"Gap detection not implemented for interval: {interval}")
            return []
        
        query = """
        SELECT timestamp
        FROM forex_data
        WHERE pair = %s AND interval_type = %s
        ORDER BY timestamp
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(query, conn, params=(pair, interval))
                
            if df.empty:
                return []
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            timestamps = df['timestamp'].sort_values()
            
            gaps = []
            for i in range(1, len(timestamps)):
                gap = timestamps.iloc[i] - timestamps.iloc[i-1]
                if gap > expected_interval * 1.5:  # Allow 50% tolerance
                    gaps.append((timestamps.iloc[i-1], timestamps.iloc[i]))
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error detecting gaps: {e}")
            return []
    
    def import_csv_data(self, data_dir: str = ".", recreate_schema: bool = True):
        """Complete CSV data import process"""
        self.logger.info("Starting CSV data import process")
        
        try:
            # Step 1: Create/recreate schema
            if recreate_schema:
                self.logger.info("Creating database schema...")
                self.create_enhanced_schema()
            
            # Step 2: Load CSV data
            self.logger.info("Loading CSV files...")
            df = self.load_all_forex_csvs(data_dir)
            
            # Step 3: Bulk insert
            self.logger.info("Bulk inserting data...")
            self.bulk_insert_data(df)
            
            # Step 4: Generate summary report
            self.generate_import_summary()
            
            self.logger.info("CSV data import completed successfully")
            
        except Exception as e:
            self.logger.error(f"CSV import failed: {e}")
            raise
    
    def generate_import_summary(self):
        """Generate summary report of imported data"""
        summary_query = """
        SELECT 
            pair,
            interval_type,
            COUNT(*) as record_count,
            MIN(timestamp) as first_timestamp,
            MAX(timestamp) as last_timestamp,
            AVG(volume) as avg_volume
        FROM forex_data
        WHERE source = 'CSV_IMPORT'
        GROUP BY pair, interval_type
        ORDER BY pair, interval_type;
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(summary_query, conn)
            
            self.logger.info("=== CSV IMPORT SUMMARY ===")
            for _, row in df.iterrows():
                self.logger.info(
                    f"{row['pair']} ({row['interval_type']}): "
                    f"{row['record_count']:,} records from "
                    f"{row['first_timestamp']} to {row['last_timestamp']}"
                )
            
            # Total statistics
            total_records = df['record_count'].sum()
            self.logger.info(f"Total records imported: {total_records:,}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return None


def main():
    """Main function for CSV import"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import forex CSV data')
    parser.add_argument('--data-dir', default='.', help='Directory containing CSV files')
    parser.add_argument('--no-recreate', action='store_true', help='Do not recreate database schema')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for insertion')
    
    args = parser.parse_args()
    
    try:
        manager = CSVDataManager()
        manager.import_csv_data(
            data_dir=args.data_dir,
            recreate_schema=not args.no_recreate
        )
        print("CSV import completed successfully!")
        
    except Exception as e:
        print(f"CSV import failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())