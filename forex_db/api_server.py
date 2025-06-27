#!/usr/bin/env python3
"""
REST API and Web Dashboard for the Forex Data System.

Provides:
- RESTful API endpoints for accessing forex data
- Real-time data streaming via WebSocket
- Web dashboard for monitoring and visualization
- API documentation with Swagger/OpenAPI
- Authentication and rate limiting
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Pydantic models for request/response validation
from pydantic import BaseModel, Field, validator
from pydantic.types import confloat, conint

# Redis for caching and session management
import redis.asyncio as redis

# System components
from config import ForexConfig
from database_manager import DatabaseManager
from forex_system import ForexDataSystem
from examples import ForexAnalyzer, SystemMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ForexDataPoint(BaseModel):
    """Single forex data point model."""
    timestamp: datetime
    pair: str
    open: confloat(gt=0)
    high: confloat(gt=0)
    low: confloat(gt=0)
    close: confloat(gt=0)
    volume: Optional[int] = None
    spread: Optional[confloat(ge=0)] = None
    tick_volume: Optional[int] = None
    source: str

class ForexDataResponse(BaseModel):
    """Response model for forex data queries."""
    pair: str
    start_time: datetime
    end_time: datetime
    data_points: int
    data: List[ForexDataPoint]
    metadata: Dict = {}

class SystemStatsResponse(BaseModel):
    """System statistics response model."""
    total_pairs: int
    total_records: int
    uptime_hours: float
    data_freshness: Dict[str, Dict]
    performance_metrics: Dict = {}

class PairAnalysisResponse(BaseModel):
    """Pair analysis response model."""
    pair: str
    period_days: int
    total_return: float
    volatility: float
    trend_direction: str
    technical_indicators: Dict = {}
    data_quality: Dict = {}

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    database_connected: bool
    redis_connected: bool
    services: Dict[str, str] = {}

# Request models
class DataQueryRequest(BaseModel):
    """Request model for data queries."""
    pair: str = Field(..., regex="^[A-Z]{6}$", description="6-letter currency pair code")
    start_time: datetime
    end_time: datetime
    interval: Optional[str] = Field("1m", description="Data interval (1m, 5m, 15m, 1h, 1d)")
    limit: Optional[conint(ge=1, le=10000)] = 1000

    @validator('end_time')
    def end_time_must_be_after_start_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v

class PairListResponse(BaseModel):
    """Available forex pairs response."""
    pairs: List[str]
    total_count: int
    supported_intervals: List[str] = ["1m", "5m", "15m", "1h", "1d"]

# FastAPI app initialization
app = FastAPI(
    title="Forex Data API",
    description="RESTful API for accessing forex market data with real-time updates",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global variables
config = ForexConfig()
db_manager = DatabaseManager()
system_monitor = SystemMonitor()
forex_analyzer = ForexAnalyzer()
redis_client: Optional[redis.Redis] = None
websocket_connections: List[WebSocket] = []

# Security
security = HTTPBearer(auto_error=False)

class ConnectionManager:
    """WebSocket connection manager for real-time data streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, List[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = []
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str, pairs: List[str] = None):
        disconnected_connections = []
        
        for connection in self.active_connections:
            try:
                # Send to all if no pairs specified, or if connection is subscribed to any of the pairs
                if not pairs or any(pair in self.subscriptions.get(connection, []) for pair in pairs):
                    await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting WebSocket message: {e}")
                disconnected_connections.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected_connections:
            self.disconnect(connection)
    
    def subscribe(self, websocket: WebSocket, pairs: List[str]):
        if websocket in self.subscriptions:
            self.subscriptions[websocket].extend(pairs)
            # Remove duplicates
            self.subscriptions[websocket] = list(set(self.subscriptions[websocket]))

manager = ConnectionManager()

# Dependency functions
async def get_redis():
    """Get Redis connection dependency."""
    return redis_client

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from token (placeholder for authentication)."""
    # Implement actual authentication logic here
    # For now, return a default user
    return {"username": "api_user", "permissions": ["read", "write"]}

async def rate_limit(request_key: str, max_requests: int = 100, window_seconds: int = 3600):
    """Rate limiting using Redis."""
    if not redis_client:
        return True  # Skip rate limiting if Redis not available
    
    try:
        current_requests = await redis_client.incr(request_key)
        if current_requests == 1:
            await redis_client.expire(request_key, window_seconds)
        
        return current_requests <= max_requests
    except Exception as e:
        logger.warning(f"Rate limiting error: {e}")
        return True  # Allow request if rate limiting fails

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize connections and services on startup."""
    global redis_client
    
    logger.info("Starting Forex Data API server...")
    
    # Initialize Redis connection
    try:
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Test database connection
    if not db_manager.test_connection():
        logger.error("Database connection failed!")
    else:
        logger.info("Database connection established")
    
    # Start background tasks
    asyncio.create_task(real_time_data_broadcaster())
    
    logger.info("API server startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down API server...")
    
    if redis_client:
        await redis_client.close()
    
    # Close all WebSocket connections
    for connection in manager.active_connections:
        await connection.close()

# Background tasks
async def real_time_data_broadcaster():
    """Background task to broadcast real-time data to WebSocket clients."""
    logger.info("Starting real-time data broadcaster")
    
    while True:
        try:
            if manager.active_connections:
                # Get fresh data for subscribed pairs
                all_subscribed_pairs = set()
                for pairs in manager.subscriptions.values():
                    all_subscribed_pairs.update(pairs)
                
                if all_subscribed_pairs:
                    # Simulate getting latest data (in real implementation, 
                    # this would get actual latest data from the database)
                    for pair in list(all_subscribed_pairs)[:5]:  # Limit to avoid overload
                        try:
                            latest_data = db_manager.get_forex_data(
                                pair,
                                datetime.now() - timedelta(minutes=5),
                                datetime.now()
                            )
                            
                            if not latest_data.empty:
                                latest_record = latest_data.iloc[-1]
                                message = {
                                    "type": "price_update",
                                    "pair": pair,
                                    "timestamp": latest_record['timestamp'].isoformat(),
                                    "price": float(latest_record['close']),
                                    "change": 0.0  # Calculate actual change
                                }
                                
                                await manager.broadcast(json.dumps(message), [pair])
                        
                        except Exception as e:
                            logger.error(f"Error broadcasting data for {pair}: {e}")
            
            await asyncio.sleep(60)  # Broadcast every minute
            
        except Exception as e:
            logger.error(f"Error in real-time broadcaster: {e}")
            await asyncio.sleep(60)

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Forex Data Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .widget { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .pair-list { list-style: none; padding: 0; }
            .pair-item { padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }
            .price { font-weight: bold; color: #27ae60; }
            .chart { height: 400px; }
            .status { padding: 5px 10px; border-radius: 4px; color: white; font-size: 12px; }
            .status.online { background: #27ae60; }
            .status.offline { background: #e74c3c; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏦 Forex Data Dashboard</h1>
            <p>Real-time forex market data collection and analysis system</p>
            <span id="status" class="status offline">Connecting...</span>
        </div>
        
        <div class="dashboard">
            <div class="widget">
                <h3>📊 Live Prices</h3>
                <ul id="price-list" class="pair-list">
                    <li class="pair-item">Loading...</li>
                </ul>
            </div>
            
            <div class="widget">
                <h3>📈 Price Chart</h3>
                <div id="chart" class="chart"></div>
            </div>
            
            <div class="widget">
                <h3>📋 System Status</h3>
                <div id="system-stats">Loading...</div>
            </div>
            
            <div class="widget">
                <h3>🔧 API Endpoints</h3>
                <ul>
                    <li><a href="/docs">📚 API Documentation</a></li>
                    <li><a href="/api/v1/pairs">📋 Available Pairs</a></li>
                    <li><a href="/api/v1/health">💚 Health Check</a></li>
                    <li><a href="/api/v1/stats">📊 System Statistics</a></li>
                </ul>
            </div>
        </div>
        
        <script>
            // WebSocket connection for real-time updates
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            const statusEl = document.getElementById('status');
            const priceListEl = document.getElementById('price-list');
            const systemStatsEl = document.getElementById('system-stats');
            
            ws.onopen = function(event) {
                statusEl.textContent = 'Connected';
                statusEl.className = 'status online';
                
                // Subscribe to major pairs
                ws.send(JSON.stringify({
                    type: 'subscribe',
                    pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
                }));
            };
            
            ws.onclose = function(event) {
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'status offline';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'price_update') {
                    updatePriceList(data);
                    updateChart(data);
                }
            };
            
            function updatePriceList(data) {
                // Update price in the list
                const existingItem = document.querySelector(`[data-pair="${data.pair}"]`);
                const priceHtml = `
                    <span>${data.pair}</span>
                    <span class="price">${data.price.toFixed(5)}</span>
                `;
                
                if (existingItem) {
                    existingItem.innerHTML = priceHtml;
                } else {
                    const li = document.createElement('li');
                    li.className = 'pair-item';
                    li.setAttribute('data-pair', data.pair);
                    li.innerHTML = priceHtml;
                    priceListEl.appendChild(li);
                }
            }
            
            function updateChart(data) {
                // Simple chart update (in real implementation, maintain historical data)
                const trace = {
                    x: [new Date(data.timestamp)],
                    y: [data.price],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: data.pair
                };
                
                Plotly.react('chart', [trace], {
                    title: `${data.pair} Price`,
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Price' }
                });
            }
            
            // Load initial data
            fetch('/api/v1/pairs')
                .then(response => response.json())
                .then(data => {
                    priceListEl.innerHTML = '';
                    data.pairs.slice(0, 8).forEach(pair => {
                        const li = document.createElement('li');
                        li.className = 'pair-item';
                        li.setAttribute('data-pair', pair);
                        li.innerHTML = `<span>${pair}</span><span class="price">Loading...</span>`;
                        priceListEl.appendChild(li);
                    });
                });
            
            // Load system stats
            fetch('/api/v1/stats')
                .then(response => response.json())
                .then(data => {
                    systemStatsEl.innerHTML = `
                        <p><strong>Total Pairs:</strong> ${data.total_pairs}</p>
                        <p><strong>Total Records:</strong> ${data.total_records.toLocaleString()}</p>
                        <p><strong>Uptime:</strong> ${data.uptime_hours.toFixed(1)} hours</p>
                    `;
                });
        </script>
    </body>
    </html>
    """

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    database_connected = db_manager.test_connection()
    redis_connected = False
    
    if redis_client:
        try:
            await redis_client.ping()
            redis_connected = True
        except:
            pass
    
    status = "healthy" if database_connected else "unhealthy"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now(),
        database_connected=database_connected,
        redis_connected=redis_connected,
        services={
            "database": "online" if database_connected else "offline",
            "redis": "online" if redis_connected else "offline",
            "websocket": f"{len(manager.active_connections)} connections"
        }
    )

@app.get("/api/v1/pairs", response_model=PairListResponse)
async def get_available_pairs():
    """Get list of available forex pairs."""
    # Check rate limiting
    if not await rate_limit("get_pairs", 60, 3600):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return PairListResponse(
        pairs=config.MAJOR_PAIRS,
        total_count=len(config.MAJOR_PAIRS)
    )

@app.get("/api/v1/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """Get system statistics."""
    if not await rate_limit("get_stats", 30, 3600):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        stats = db_manager.get_data_statistics()
        freshness = system_monitor.check_data_freshness()
        
        return SystemStatsResponse(
            total_pairs=stats.get('total_pairs', 0),
            total_records=stats.get('total_records', 0),
            uptime_hours=24.0,  # Placeholder - implement actual uptime tracking
            data_freshness=freshness
        )
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/data", response_model=ForexDataResponse)
async def get_forex_data(request: DataQueryRequest):
    """Get forex data for a specific pair and time range."""
    if not await rate_limit(f"get_data_{request.pair}", 100, 3600):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # Validate pair
        if request.pair not in config.MAJOR_PAIRS:
            raise HTTPException(status_code=400, detail="Unsupported currency pair")
        
        # Get data from database
        data = db_manager.get_forex_data(request.pair, request.start_time, request.end_time)
        
        if data.empty:
            return ForexDataResponse(
                pair=request.pair,
                start_time=request.start_time,
                end_time=request.end_time,
                data_points=0,
                data=[]
            )
        
        # Apply limit
        if len(data) > request.limit:
            data = data.tail(request.limit)
        
        # Convert to response format
        data_points = []
        for _, row in data.iterrows():
            data_points.append(ForexDataPoint(
                timestamp=row['timestamp'],
                pair=row['pair'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row.get('volume'),
                spread=row.get('spread'),
                tick_volume=row.get('tick_volume'),
                source=row['source']
            ))
        
        return ForexDataResponse(
            pair=request.pair,
            start_time=request.start_time,
            end_time=request.end_time,
            data_points=len(data_points),
            data=data_points,
            metadata={
                "interval": request.interval,
                "source_count": data['source'].nunique(),
                "time_range_hours": (request.end_time - request.start_time).total_seconds() / 3600
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting forex data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/analysis/{pair}", response_model=PairAnalysisResponse)
async def get_pair_analysis(
    pair: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """Get technical analysis for a forex pair."""
    if not await rate_limit(f"analysis_{pair}", 20, 3600):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        pair = pair.upper()
        if pair not in config.MAJOR_PAIRS:
            raise HTTPException(status_code=400, detail="Unsupported currency pair")
        
        # Get analysis from analyzer
        analysis = forex_analyzer.analyze_pair_performance(pair, days)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="No data available for analysis")
        
        return PairAnalysisResponse(
            pair=analysis['pair'],
            period_days=analysis['period_days'],
            total_return=analysis['total_return'],
            volatility=analysis['price_volatility'],
            trend_direction=analysis['trend_direction'],
            technical_indicators={
                "current_rsi": analysis.get('current_rsi'),
                "current_macd": analysis.get('current_macd'),
                "bb_position": analysis.get('bb_position')
            },
            data_quality=analysis['data_quality']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pair analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming."""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'subscribe':
                pairs = message.get('pairs', [])
                manager.subscribe(websocket, pairs)
                await manager.send_personal_message(
                    json.dumps({"type": "subscribed", "pairs": pairs}),
                    websocket
                )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Additional utility endpoints
@app.get("/api/v1/latest/{pair}")
async def get_latest_price(pair: str):
    """Get the latest price for a forex pair."""
    if not await rate_limit(f"latest_{pair}", 200, 3600):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        pair = pair.upper()
        if pair not in config.MAJOR_PAIRS:
            raise HTTPException(status_code=400, detail="Unsupported currency pair")
        
        latest_timestamp = db_manager.get_latest_timestamp(pair)
        if not latest_timestamp:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Get latest data point
        data = db_manager.get_forex_data(
            pair,
            latest_timestamp - timedelta(minutes=1),
            latest_timestamp + timedelta(minutes=1)
        )
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        latest = data.iloc[-1]
        
        return {
            "pair": pair,
            "timestamp": latest['timestamp'].isoformat(),
            "price": float(latest['close']),
            "change_24h": 0.0,  # Calculate actual 24h change
            "volume": int(latest['volume']) if latest.get('volume') else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest price: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Development and testing endpoints
if os.getenv('DEBUG'):
    @app.post("/api/v1/test/insert")
    async def test_insert_data(data: List[ForexDataPoint]):
        """Test endpoint for inserting data (development only)."""
        try:
            # Convert to database format
            db_data = []
            for point in data:
                db_data.append({
                    'timestamp': point.timestamp,
                    'pair': point.pair,
                    'open': point.open,
                    'high': point.high,
                    'low': point.low,
                    'close': point.close,
                    'volume': point.volume,
                    'spread': point.spread,
                    'tick_volume': point.tick_volume,
                    'source': point.source
                })
            
            success = db_manager.insert_forex_data(db_data)
            
            return {"success": success, "inserted_count": len(db_data)}
            
        except Exception as e:
            logger.error(f"Error in test insert: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

# Main function for running the server
def main():
    """Main function to run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Forex Data API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Setup logging
    ForexConfig.setup_logging()
    
    logger.info(f"Starting Forex Data API server on {args.host}:{args.port}")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )

if __name__ == "__main__":
    main()