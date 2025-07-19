#!/usr/bin/env python3
"""
Load forex CSV data into TimescaleDB database
Handles both daily and minute interval data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import sys
import os

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forex_db.database_manager import DatabaseManager
from forex_db.config import ForexConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForexCSVLoader:
    """Load forex data from CSV files into the database"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.config = ForexConfig()
        self.batch_size = 5000  # Larger batches for CSV loading
        
    def load_csv_file(self, filepath: str, source: str = 'csv_import'):
        """Load a single CSV file into the database"""
        logger.info(f"Loading CSV file: {filepath}")
        
        try:
            # Read CSV with proper parsing
            df = pd.read_csv(filepath, parse_dates=['datetime'])
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            
            # Convert to database format
            db_records = self._convert_to_db_format(df, source)
            
            # Insert in batches
            total_inserted = 0
            for i in range(0, len(db_records), self.batch_size):
                batch = db_records[i:i + self.batch_size]
                if self.db_manager.insert_forex_data(batch):
                    total_inserted += len(batch)
                    logger.info(f"Inserted batch {i//self.batch_size + 1}: {len(batch)} records")
                else:
                    logger.error(f"Failed to insert batch starting at index {i}")
            
            logger.info(f"Successfully inserted {total_inserted} records from {filepath}")
            return total_inserted
            
        except Exception as e:
            logger.error(f"Error loading CSV file {filepath}: {e}")
            return 0
    
    def _convert_to_db_format(self, df: pd.DataFrame, source: str) -> list:
        """Convert CSV dataframe to database format"""
        records = []
        
        for _, row in df.iterrows():
            # Handle symbol to pair conversion (remove =X suffix if present)
            pair = row['symbol'].replace('=X', '')
            
            # Determine if this is minute or daily data based on interval
            interval = row.get('interval', '1d')
            
            record = {
                'timestamp': row['datetime'],
                'pair': pair,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume']) if pd.notna(row['volume']) and row['volume'] > 0 else None,
                'spread': None,  # Will be calculated later
                'tick_volume': None,
                'source': f"{source}_{interval}"  # Include interval in source
            }
            
            records.append(record)
        
        return records
    
    def load_all_csv_files(self, directory: str, pattern: str = "fx_data_*.csv"):
        """Load all CSV files matching pattern from directory"""
        directory_path = Path(directory)
        csv_files = list(directory_path.glob(pattern))
        
        logger.info(f"Found {len(csv_files)} CSV files to load")
        
        total_records = 0
        for csv_file in csv_files:
            records = self.load_csv_file(str(csv_file))
            total_records += records
        
        logger.info(f"Total records loaded: {total_records}")
        return total_records
    
    def validate_and_clean_data(self):
        """Validate and clean loaded data"""
        logger.info("Validating and cleaning data...")
        
        # Get statistics
        stats = self.db_manager.get_data_statistics()
        
        if stats:
            logger.info("Data statistics after loading:")
            logger.info(f"Total pairs: {stats.get('total_pairs', 0)}")
            logger.info(f"Total records: {stats.get('total_records', 0)}")
            
            for pair_stat in stats.get('pair_statistics', []):
                logger.info(
                    f"{pair_stat['pair']} ({pair_stat['source']}): "
                    f"{pair_stat['record_count']} records, "
                    f"Date range: {pair_stat['earliest_data']} to {pair_stat['latest_data']}"
                )
    
    def add_interval_column(self):
        """Add interval column to database if needed"""
        logger.info("Checking if interval column needs to be added...")
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Check if interval column exists
                    cursor.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'forex_data' 
                        AND column_name = 'interval';
                    """)
                    
                    if not cursor.fetchone():
                        logger.info("Adding interval column to forex_data table...")
                        cursor.execute("""
                            ALTER TABLE forex_data 
                            ADD COLUMN interval VARCHAR(10);
                        """)
                        
                        # Update existing data based on source
                        cursor.execute("""
                            UPDATE forex_data 
                            SET interval = 
                                CASE 
                                    WHEN source LIKE '%_1m' THEN '1m'
                                    WHEN source LIKE '%_1d' THEN '1d'
                                    ELSE '1d'
                                END
                            WHERE interval IS NULL;
                        """)
                        
                        conn.commit()
                        logger.info("Interval column added successfully")
                    else:
                        logger.info("Interval column already exists")
                        
        except Exception as e:
            logger.error(f"Error adding interval column: {e}")
    
    def calculate_spreads(self):
        """Calculate estimated spreads based on high-low"""
        logger.info("Calculating spreads for loaded data...")
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Update spreads based on high-low difference
                    cursor.execute("""
                        UPDATE forex_data 
                        SET spread = 
                            CASE 
                                WHEN pair LIKE '%JPY' THEN (high - low) * 100  -- Pips for JPY pairs
                                ELSE (high - low) * 10000  -- Pips for other pairs
                            END
                        WHERE spread IS NULL 
                        AND source LIKE 'csv_import%';
                    """)
                    
                    updated = cursor.rowcount
                    conn.commit()
                    logger.info(f"Updated spreads for {updated} records")
                    
        except Exception as e:
            logger.error(f"Error calculating spreads: {e}")
    
    def deduplicate_data(self):
        """Remove duplicate entries keeping the most recent"""
        logger.info("Checking for duplicate data...")
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Find duplicates
                    cursor.execute("""
                        SELECT pair, timestamp, source, COUNT(*) as count
                        FROM forex_data
                        GROUP BY pair, timestamp, source
                        HAVING COUNT(*) > 1;
                    """)
                    
                    duplicates = cursor.fetchall()
                    
                    if duplicates:
                        logger.warning(f"Found {len(duplicates)} duplicate entries")
                        
                        # Remove duplicates keeping the one with highest created_at
                        cursor.execute("""
                            DELETE FROM forex_data a
                            USING forex_data b
                            WHERE a.pair = b.pair 
                            AND a.timestamp = b.timestamp 
                            AND a.source = b.source
                            AND a.created_at < b.created_at;
                        """)
                        
                        deleted = cursor.rowcount
                        conn.commit()
                        logger.info(f"Removed {deleted} duplicate records")
                    else:
                        logger.info("No duplicates found")
                        
        except Exception as e:
            logger.error(f"Error deduplicating data: {e}")


def main():
    """Main function to load CSV data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load forex CSV data into database')
    parser.add_argument('--path', default='csv_data', help='Path to CSV file or directory containing CSV files')
    parser.add_argument('--source', default='csv_import', help='Source identifier for the data')
    parser.add_argument('--pattern', default='fx_data_*.csv', help='Pattern to match CSV files')
    parser.add_argument('--add-interval', action='store_true', help='Add interval column to database')
    parser.add_argument('--calculate-spreads', action='store_true', help='Calculate spreads from high-low')
    parser.add_argument('--deduplicate', action='store_true', help='Remove duplicate entries')
    parser.add_argument('--validate', action='store_true', help='Validate data after loading')
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = ForexCSVLoader()
    
    # Add interval column if requested
    if args.add_interval:
        loader.add_interval_column()
    
    # Check if path is file or directory
    path = Path(args.path)
    
    if path.is_file():
        # Load single file
        loader.load_csv_file(str(path), args.source)
    elif path.is_dir():
        # Load all matching files from directory
        loader.load_all_csv_files(str(path), args.pattern)
    else:
        logger.error(f"Path {path} does not exist")
        return 1
    
    # Post-processing
    if args.calculate_spreads:
        loader.calculate_spreads()
    
    if args.deduplicate:
        loader.deduplicate_data()
    
    if args.validate:
        loader.validate_and_clean_data()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())