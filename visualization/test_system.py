#!/usr/bin/env python3
"""
Test script to verify the visualization system components.
"""
import sys
import os
import subprocess
import time
import requests

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_database_connection():
    """Test database connectivity."""
    print("Testing database connection...")
    try:
        from forex_db.config import ForexConfig
        import psycopg2
        
        conn = psycopg2.connect(
            host=ForexConfig.DB_HOST,
            port=ForexConfig.DB_PORT,
            database=ForexConfig.DB_NAME,
            user=ForexConfig.DB_USER,
            password=ForexConfig.DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM forex_data LIMIT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        print(f"✓ Database connection successful. Records available: {result[0] if result else 'Unknown'}")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_backend_import():
    """Test backend module imports."""
    print("Testing backend imports...")
    try:
        from backend import app
        print("✓ Backend imports successful")
        return True
    except Exception as e:
        print(f"✗ Backend import failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints (requires running server)."""
    print("Testing API endpoints...")
    base_url = "http://localhost:8000"
    
    endpoints = [
        "/health",
        "/pairs",
        "/intervals"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✓ {endpoint} - OK")
            else:
                print(f"✗ {endpoint} - Status: {response.status_code}")
        except requests.ConnectionError:
            print(f"✗ {endpoint} - Connection failed (server not running?)")
        except Exception as e:
            print(f"✗ {endpoint} - Error: {e}")

def test_frontend_files():
    """Test that frontend files exist."""
    print("Testing frontend files...")
    
    frontend_files = [
        "frontend/package.json",
        "frontend/src/App.js",
        "frontend/src/components/DataFilters.js",
        "frontend/src/components/DataTable.js",
        "frontend/src/components/DataChart.js",
        "frontend/src/components/StatsSummary.js"
    ]
    
    for file_path in frontend_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path} - Exists")
        else:
            print(f"✗ {file_path} - Missing")

def main():
    """Run all tests."""
    print("=== Forex Data Visualization System Test ===\n")
    
    print("1. Database Connection Test")
    db_ok = test_database_connection()
    print()
    
    print("2. Backend Import Test")
    backend_ok = test_backend_import()
    print()
    
    print("3. Frontend Files Test")
    test_frontend_files()
    print()
    
    print("4. API Endpoints Test (requires running server)")
    print("   Start the backend with: python start_backend.py")
    print("   Then run this test again or test manually:")
    test_api_endpoints()
    print()
    
    if db_ok and backend_ok:
        print("✓ Core system components are ready!")
        print("\nNext steps:")
        print("1. Start backend: python start_backend.py")
        print("2. Start frontend: ./start_frontend.sh")
        print("3. Open browser to: http://localhost:3000")
    else:
        print("✗ Some components need attention before running the system.")

if __name__ == "__main__":
    main()