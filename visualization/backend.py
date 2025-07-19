from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from typing import Optional, List
import json
import sys
import os
from sqlalchemy import create_engine

# Add the parent directory to the path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from forex_db.config import ForexConfig

app = FastAPI(title="Forex Data Visualization API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    """Get database connection using the config."""
    try:
        conn = psycopg2.connect(
            host=ForexConfig.DB_HOST,
            port=ForexConfig.DB_PORT,
            database=ForexConfig.DB_NAME,
            user=ForexConfig.DB_USER,
            password=ForexConfig.DB_PASSWORD
        )
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

def get_sqlalchemy_engine():
    """Get SQLAlchemy engine for pandas operations."""
    return create_engine(ForexConfig.get_database_url())

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Forex Data Visualization API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "database": "disconnected", "error": str(e)}
        )

@app.get("/pairs")
async def get_available_pairs():
    """Get all available currency pairs."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT pair FROM forex_data ORDER BY pair")
        pairs = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return {"pairs": pairs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/intervals")
async def get_available_intervals():
    """Get all available data intervals."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN interval IS NULL THEN 'daily' 
                    ELSE interval 
                END as interval_clean,
                COUNT(*) as record_count
            FROM forex_data 
            GROUP BY interval 
            ORDER BY interval_clean
        """)
        results = cursor.fetchall()
        intervals = []
        for interval_name, count in results:
            intervals.append({
                "value": interval_name,
                "label": f"{interval_name} ({count:,} records)",
                "count": count
            })
        cursor.close()
        conn.close()
        return {"intervals": intervals}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-range")
async def get_data_range(pair: Optional[str] = None):
    """Get the date range of available data."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if pair:
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM forex_data 
                WHERE pair = %s
            """, (pair,))
        else:
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM forex_data")
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return {
            "min_date": result[0].isoformat() if result[0] else None,
            "max_date": result[1].isoformat() if result[1] else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data")
async def get_forex_data(
    pair: str = Query(..., description="Currency pair (e.g., EURUSD)"),
    interval_type: str = Query("daily", description="Data interval (1m, daily, etc.)"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(1000, description="Maximum number of records"),
    page: int = Query(1, description="Page number for pagination")
):
    """Get forex data with filtering options."""
    try:
        engine = get_sqlalchemy_engine()
        
        # Handle NULL intervals by treating them as 'daily'
        if interval_type == "daily":
            interval_condition = "interval IS NULL"
        else:
            interval_condition = "interval = %(interval)s"
            
        # Build the query
        base_query = f"""
            SELECT timestamp, pair, 
                   CASE WHEN interval IS NULL THEN 'daily' ELSE interval END as interval,
                   open, high, low, close, volume, source
            FROM forex_data
            WHERE pair = %(pair)s AND {interval_condition}
        """
        params = {"pair": pair}
        
        if interval_type != "daily":
            params["interval"] = interval_type
        
        # Add date filters
        if start_date:
            base_query += " AND timestamp >= %(start_date)s"
            params["start_date"] = start_date
        
        if end_date:
            base_query += " AND timestamp <= %(end_date)s"
            params["end_date"] = end_date
        
        # Add ordering and pagination
        base_query += " ORDER BY timestamp DESC"
        offset = (page - 1) * limit
        base_query += f" LIMIT {limit} OFFSET {offset}"
        
        # Execute query
        df = pd.read_sql(base_query, engine, params=params)
        
        # Handle NaN values and convert to JSON-serializable format
        df = df.fillna(0)  # Replace NaN with 0 for numeric columns
        data = df.to_dict('records')
        for record in data:
            if 'timestamp' in record:
                record['timestamp'] = record['timestamp'].isoformat()
            # Rename interval to interval_type for frontend compatibility
            if 'interval' in record:
                record['interval_type'] = record['interval']
                del record['interval']
            # Ensure numeric fields are properly formatted
            for field in ['open', 'high', 'low', 'close']:
                if field in record and record[field] is not None:
                    try:
                        record[field] = float(record[field])
                    except (ValueError, TypeError):
                        record[field] = 0.0
        
        return {
            "data": data,
            "count": len(data),
            "page": page,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_data_statistics(
    pair: Optional[str] = None,
    interval_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get statistical summary of the data."""
    try:
        conn = get_db_connection()
        
        # Build query for statistics
        query = """
            SELECT 
                COUNT(*) as record_count,
                MIN(timestamp) as earliest_date,
                MAX(timestamp) as latest_date,
                AVG(close) as avg_price,
                MIN(low) as min_price,
                MAX(high) as max_price,
                STDDEV(close) as price_stddev
            FROM forex_data
            WHERE 1=1
        """
        params = []
        
        if pair:
            query += " AND pair = %s"
            params.append(pair)
        
        if interval_type:
            if interval_type == "daily":
                query += " AND interval IS NULL"
            else:
                query += " AND interval = %s"
                params.append(interval_type)
        
        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return {
            "record_count": result[0] if result[0] is not None else 0,
            "earliest_date": result[1].isoformat() if result[1] else None,
            "latest_date": result[2].isoformat() if result[2] else None,
            "avg_price": float(result[3]) if result[3] is not None and not pd.isna(result[3]) else None,
            "min_price": float(result[4]) if result[4] is not None and not pd.isna(result[4]) else None,
            "max_price": float(result[5]) if result[5] is not None and not pd.isna(result[5]) else None,
            "price_stddev": float(result[6]) if result[6] is not None and not pd.isna(result[6]) else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chart-data")
async def get_chart_data(
    pair: str = Query(..., description="Currency pair"),
    interval_type: str = Query("daily", description="Data interval"),
    start_date: Optional[str] = Query(None, description="Start date"),
    end_date: Optional[str] = Query(None, description="End date"),
    limit: int = Query(500, description="Maximum number of records for chart")
):
    """Get data formatted for chart visualization."""
    try:
        engine = get_sqlalchemy_engine()
        
        # Handle NULL intervals by treating them as 'daily'
        if interval_type == "daily":
            interval_condition = "interval IS NULL"
            params = {"pair": pair}
        else:
            interval_condition = "interval = %(interval)s"
            params = {"pair": pair, "interval": interval_type}
            
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM forex_data
            WHERE pair = %(pair)s AND {interval_condition}
        """
        
        if start_date:
            query += " AND timestamp >= %(start_date)s"
            params["start_date"] = start_date
        
        if end_date:
            query += " AND timestamp <= %(end_date)s"
            params["end_date"] = end_date
        
        query += f" ORDER BY timestamp ASC LIMIT {limit}"
        
        df = pd.read_sql(query, engine, params=params)
        
        # Handle NaN values
        df = df.fillna(0)
        
        # Format for candlestick chart
        chart_data = []
        for _, row in df.iterrows():
            try:
                chart_data.append({
                    "time": row['timestamp'].isoformat(),
                    "open": float(row['open']) if row['open'] is not None else 0.0,
                    "high": float(row['high']) if row['high'] is not None else 0.0,
                    "low": float(row['low']) if row['low'] is not None else 0.0,
                    "close": float(row['close']) if row['close'] is not None else 0.0,
                    "volume": int(row['volume']) if row['volume'] is not None else 0
                })
            except (ValueError, TypeError) as e:
                # Skip invalid records
                continue
        
        return {
            "chart_data": chart_data,
            "pair": pair,
            "interval": interval_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)