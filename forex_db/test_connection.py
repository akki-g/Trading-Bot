#!/usr/bin/env python3
from database_manager import DatabaseManager
from config import ForexConfig

def test_connection():
    """Test database connection and setup."""
    print("Testing database connection...")
    
    config = ForexConfig()
    db_manager = DatabaseManager()
    
    # Test connection
    if db_manager.test_connection():
        print("✅ Database connection successful!")
    else:
        print("❌ Database connection failed!")
        return False
    
    # Test database setup
    if db_manager.setup_database():
        print("✅ Database schema setup successful!")
    else:
        print("❌ Database schema setup failed!")
        return False
    
    print("🎉 Database is ready for data collection!")
    return True

if __name__ == "__main__":
    test_connection()