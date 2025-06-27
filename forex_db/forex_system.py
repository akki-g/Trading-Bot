import asyncio
import schedule
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import ForexConfig
from database_manager import DatabaseManager
from data_collector import ForexDataCollector, DataEnricher

logger = logging.getLogger(__name__)

class ForexDataSystem:
    """Main system orchestrator for forex data collection and management."""
    
    def __init__(self):
        self.config = ForexConfig()
        self.db_manager = DatabaseManager()
        self.data_enricher = DataEnricher()
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.is_running = False
        self.executor.shutdown(wait=True)
        sys.exit(0)
    
    def initialize_system(self) -> bool:
        """Initialize the entire forex data system."""
        logger.info("Initializing Forex Data System...")
        
        # Validate configuration
        if not self.config.validate_config():
            logger.error("Configuration validation failed")
            return False
        
        # Test database connection
        if not self.db_manager.test_connection():
            logger.error("Database connection test failed")
            return False
        
        # Setup database schema
        if not self.db_manager.setup_database():
            logger.error("Database setup failed")
            return False
        
        logger.info("System initialization completed successfully")
        return True
    
    async def collect_historical_data(self, pairs: List[str] = None) -> bool:
        """Collect historical data for all or specified forex pairs."""
        if pairs is None:
            pairs = self.config.MAJOR_PAIRS
        
        logger.info(f"Starting historical data collection for {len(pairs)} pairs")
        
        async with ForexDataCollector() as collector:
            for pair in pairs:
                try:
                    logger.info(f"Collecting historical data for {pair}")
                    
                    # Check what data we already have
                    latest_timestamp = self.db_manager.get_latest_timestamp(pair)
                    
                    if latest_timestamp:
                        logger.info(f"Latest data for {pair}: {latest_timestamp}")
                        
                        # If we have recent data, only collect what's missing
                        if (datetime.now() - latest_timestamp).days < 1:
                            logger.info(f"Recent data exists for {pair}, skipping full collection")
                            continue
                    
                    # Collect bulk historical data
                    bulk_data = collector.get_historical_data_bulk([pair], self.config.HISTORICAL_YEARS)
                    pair_data = bulk_data.get(pair, [])
                    
                    if not pair_data:
                        logger.warning(f"No data collected for {pair}")
                        continue
                    
                    # Validate data
                    validated_data = collector.validate_data(pair_data)
                    
                    # Enrich data with technical indicators
                    enriched_data = self.data_enricher.calculate_technical_indicators(validated_data)
                    enriched_data = self.data_enricher.calculate_spreads(enriched_data)
                    
                    # Insert into database
                    if self.db_manager.insert_forex_data(enriched_data):
                        logger.info(f"Successfully stored {len(enriched_data)} records for {pair}")
                    else:
                        logger.error(f"Failed to store data for {pair}")
                    
                    # Rate limiting between pairs
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error collecting historical data for {pair}: {e}")
                    continue
        
        logger.info("Historical data collection completed")
        return True
    
    async def update_real_time_data(self) -> bool:
        """Update with the latest real-time data."""
        logger.info("Starting real-time data update")
        
        try:
            async with ForexDataCollector() as collector:
                # Get real-time data for all pairs
                real_time_data = await collector.get_real_time_data(self.config.MAJOR_PAIRS)
                
                if not real_time_data:
                    logger.warning("No real-time data collected")
                    return False
                
                # Validate and enrich data
                validated_data = collector.validate_data(real_time_data)
                enriched_data = self.data_enricher.calculate_spreads(validated_data)
                
                # Insert into database
                if self.db_manager.insert_forex_data(enriched_data):
                    logger.info(f"Successfully updated {len(enriched_data)} real-time records")
                    return True
                else:
                    logger.error("Failed to insert real-time data")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating real-time data: {e}")
            return False
    
    def schedule_updates(self):
        """Schedule periodic data updates."""
        logger.info("Setting up scheduled data updates")
        
        # Schedule real-time updates every minute
        schedule.every(self.config.UPDATE_INTERVAL_MINUTES).minutes.do(
            lambda: asyncio.run(self.update_real_time_data())
        )
        
        # Schedule daily cleanup at 2 AM
        schedule.every().day.at("02:00").do(
            lambda: self.db_manager.cleanup_old_data(days_to_keep=365)
        )
        
        # Schedule weekly data validation
        schedule.every().sunday.at("03:00").do(self.validate_data_integrity)
        
        # Schedule monthly system statistics
        schedule.every().month.do(self.print_system_statistics)
        
        logger.info("Scheduled updates configured")
    
    def validate_data_integrity(self) -> bool:
        """Validate data integrity and consistency."""
        logger.info("Starting data integrity validation")
        
        try:
            stats = self.db_manager.get_data_statistics()
            
            if not stats:
                logger.error("Failed to get data statistics")
                return False
            
            # Check for gaps in data
            for pair_stat in stats.get('pair_statistics', []):
                pair = pair_stat['pair']
                earliest = pair_stat['earliest_data']
                latest = pair_stat['latest_data']
                record_count = pair_stat['record_count']
                
                # Calculate expected records (assuming 1-minute data)
                if earliest and latest:
                    expected_minutes = int((latest - earliest).total_seconds() / 60)
                    data_completeness = (record_count / expected_minutes) * 100 if expected_minutes > 0 else 0
                    
                    logger.info(f"{pair}: {data_completeness:.1f}% data completeness")
                    
                    if data_completeness < 50:  # Less than 50% completeness
                        logger.warning(f"Low data completeness for {pair}: {data_completeness:.1f}%")
            
            logger.info("Data integrity validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Data integrity validation failed: {e}")
            return False
    
    def print_system_statistics(self):
        """Print comprehensive system statistics."""
        logger.info("Generating system statistics")
        
        try:
            stats = self.db_manager.get_data_statistics()
            
            if stats:
                logger.info("=== FOREX DATA SYSTEM STATISTICS ===")
                logger.info(f"Total pairs tracked: {stats.get('total_pairs', 0)}")
                logger.info(f"Total records stored: {stats.get('total_records', 0):,}")
                
                logger.info("\nPer-pair statistics:")
                for pair_stat in stats.get('pair_statistics', []):
                    logger.info(
                        f"  {pair_stat['pair']} ({pair_stat['source']}): "
                        f"{pair_stat['record_count']:,} records, "
                        f"Range: {pair_stat['earliest_data']} to {pair_stat['latest_data']}"
                    )
                
                logger.info("====================================")
            
        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
    
    async def backfill_missing_data(self, pair: str, start_date: datetime, 
                                  end_date: datetime) -> bool:
        """Backfill missing data for a specific pair and date range."""
        logger.info(f"Backfilling data for {pair} from {start_date} to {end_date}")
        
        try:
            async with ForexDataCollector() as collector:
                # This would need to be implemented based on API capabilities
                # Most free APIs don't allow arbitrary historical date ranges
                
                # For now, use the available historical data methods
                data = collector.get_forex_data_with_fallback(pair, 'historical')
                
                if data:
                    # Filter data to the specified date range
                    filtered_data = [
                        record for record in data
                        if start_date <= record['timestamp'] <= end_date
                    ]
                    
                    if filtered_data:
                        validated_data = collector.validate_data(filtered_data)
                        enriched_data = self.data_enricher.calculate_technical_indicators(validated_data)
                        enriched_data = self.data_enricher.calculate_spreads(enriched_data)
                        
                        if self.db_manager.insert_forex_data(enriched_data):
                            logger.info(f"Backfilled {len(enriched_data)} records for {pair}")
                            return True
                
                logger.warning(f"No data available for backfill: {pair}")
                return False
                
        except Exception as e:
            logger.error(f"Error backfilling data for {pair}: {e}")
            return False
    
    def run_scheduler(self):
        """Run the scheduled task system."""
        logger.info("Starting scheduled task runner")
        self.is_running = True
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                logger.info("Scheduler interrupted by user")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Wait longer on error
        
        logger.info("Scheduled task runner stopped")
    
    async def run_system(self, collect_historical: bool = True, 
                       run_scheduler: bool = True):
        """Run the complete forex data system."""
        logger.info("Starting Forex Data System")
        
        # Initialize system
        if not self.initialize_system():
            logger.error("System initialization failed")
            return False
        
        # Collect historical data if requested
        if collect_historical:
            await self.collect_historical_data()
        
        # Setup and run scheduler if requested
        if run_scheduler:
            self.schedule_updates()
            
            # Run one initial real-time update
            await self.update_real_time_data()
            
            # Print initial statistics
            self.print_system_statistics()
            
            # Start the scheduler in a separate thread
            scheduler_future = asyncio.get_event_loop().run_in_executor(
                self.executor, self.run_scheduler
            )
            
            try:
                await scheduler_future
            except KeyboardInterrupt:
                logger.info("System interrupted by user")
            finally:
                self.is_running = False
        
        logger.info("Forex Data System stopped")
        return True

# Command-line interface
async def main():
    """Main entry point for the forex data system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Forex Data Collection System')
    parser.add_argument('--no-historical', action='store_true',
                       help='Skip historical data collection')
    parser.add_argument('--no-scheduler', action='store_true',
                       help='Skip running the scheduler')
    parser.add_argument('--pairs', nargs='+',
                       help='Specific pairs to collect (default: all major pairs)')
    parser.add_argument('--validate', action='store_true',
                       help='Run data validation only')
    parser.add_argument('--stats', action='store_true',
                       help='Print system statistics only')
    
    args = parser.parse_args()
    
    # Setup logging
    ForexConfig.setup_logging()
    
    # Create system instance
    system = ForexDataSystem()
    
    try:
        if args.validate:
            # Run validation only
            if system.initialize_system():
                system.validate_data_integrity()
        elif args.stats:
            # Print statistics only
            if system.initialize_system():
                system.print_system_statistics()
        else:
            # Run full system
            collect_historical = not args.no_historical
            run_scheduler = not args.no_scheduler
            
            if args.pairs:
                # Override pairs if specified
                system.config.MAJOR_PAIRS = args.pairs
            
            await system.run_system(collect_historical, run_scheduler)
            
    except Exception as e:
        logger.error(f"System error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main())