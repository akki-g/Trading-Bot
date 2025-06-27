#!/usr/bin/env python3
"""Fix database constraints for data insertion."""

from database_manager import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_database_constraints():
    """Add missing database constraints."""
    db_manager = DatabaseManager()
    
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if constraint exists
                cursor.execute("""
                    SELECT constraint_name 
                    FROM information_schema.table_constraints 
                    WHERE table_name = 'forex_data' 
                    AND constraint_type = 'UNIQUE'
                    AND constraint_name = 'forex_data_unique';
                """)
                
                result = cursor.fetchone()
                
                if not result:
                    logger.info("Adding missing unique constraint...")
                    
                    # Add the unique constraint
                    cursor.execute("""
                        ALTER TABLE forex_data 
                        ADD CONSTRAINT forex_data_unique 
                        UNIQUE (timestamp, pair, source);
                    """)
                    
                    conn.commit()
                    logger.info("✅ Unique constraint added successfully!")
                else:
                    logger.info("✅ Unique constraint already exists")
                
                # Verify the constraint exists
                cursor.execute("""
                    SELECT constraint_name 
                    FROM information_schema.table_constraints 
                    WHERE table_name = 'forex_data' 
                    AND constraint_type = 'UNIQUE';
                """)
                
                constraints = cursor.fetchall()
                logger.info(f"Current unique constraints: {[c[0] for c in constraints]}")
                
    except Exception as e:
        logger.error(f"Error fixing database constraints: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if fix_database_constraints():
        print("🎉 Database constraints fixed successfully!")
    else:
        print("❌ Failed to fix database constraints")