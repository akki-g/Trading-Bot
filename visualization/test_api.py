#!/usr/bin/env python3
"""
Simple API test script for the visualization backend.
"""
import requests
import time
import sys

def test_api_endpoints():
    """Test API endpoints with actual HTTP requests."""
    base_url = "http://localhost:8000"
    
    endpoints_to_test = [
        ("/health", "Health check"),
        ("/pairs", "Available currency pairs"),
        ("/intervals", "Available intervals"),
    ]
    
    print("Testing API endpoints...")
    print("Make sure the backend server is running first!")
    print("=" * 50)
    
    for endpoint, description in endpoints_to_test:
        try:
            print(f"Testing {endpoint} - {description}")
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Success (200): {len(str(data))} chars returned")
                
                # Show some sample data
                if endpoint == "/pairs" and "pairs" in data:
                    pairs = data["pairs"]
                    print(f"    Found {len(pairs)} pairs: {pairs[:3]}{'...' if len(pairs) > 3 else ''}")
                elif endpoint == "/intervals" and "intervals" in data:
                    intervals = data["intervals"]
                    print(f"    Found intervals: {intervals}")
                elif endpoint == "/health":
                    print(f"    Status: {data.get('status', 'Unknown')}")
                    print(f"    Database: {data.get('database', 'Unknown')}")
            else:
                print(f"  ✗ Failed ({response.status_code}): {response.text[:100]}")
                
        except requests.ConnectionError:
            print(f"  ✗ Connection failed - Is the server running on {base_url}?")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
    
    # Test a data endpoint if basic endpoints work
    try:
        print("Testing data endpoint with EURUSD...")
        response = requests.get(f"{base_url}/data?pair=EURUSD&interval_type=1d&limit=5", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get("data", [])
            print(f"  ✓ Data endpoint works: {len(records)} records returned")
            if records:
                sample = records[0]
                print(f"    Sample record: {sample.get('timestamp', 'N/A')} - Close: {sample.get('close', 'N/A')}")
        else:
            print(f"  ✗ Data endpoint failed ({response.status_code}): {response.text[:200]}")
            
    except Exception as e:
        print(f"  ✗ Data endpoint error: {e}")

if __name__ == "__main__":
    test_api_endpoints()