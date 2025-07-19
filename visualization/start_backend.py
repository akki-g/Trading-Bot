#!/usr/bin/env python3
"""
Script to start the FastAPI backend server for forex data visualization.
"""
import os
import sys
import subprocess

def main():
    print("Starting Forex Data Visualization Backend...")
    
    # Check if we're in the correct directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Check if required packages are installed
    try:
        import fastapi
        import uvicorn
        import pandas
        import psycopg2
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Start the server
    try:
        cmd = [sys.executable, "-m", "uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down backend server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()