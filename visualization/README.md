# Forex Data Visualization Dashboard

A comprehensive web-based dashboard for visualizing and analyzing forex trading data from your TimescaleDB database. Now optimized to handle 3.8+ million records with full historical data from 2020-2025!

## Features

- **Interactive Data Filtering**: Filter by currency pair, time interval, date range, and record limits
- **Full Historical Coverage**: Access all 3.8M records from 2020-2025 
- **Multiple Data Intervals**: 
  - **Daily Data**: 2.6M records (2020-2025) for long-term analysis
  - **1-Minute Data**: 1.2M records (2025) for high-frequency analysis
- **Quick Date Presets**: Last month, last year, or full historical range (2020-2025)
- **Large Dataset Support**: Handle up to 100,000 records per query
- **Multiple Views**: Switch between table and chart visualizations
- **Real-time Statistics**: View summary statistics for your filtered data
- **Advanced Charts**: OHLC line charts, candlestick representations, and volume analysis
- **Responsive Design**: Works on desktop and mobile devices
- **Robust Backend**: Handles NaN values, optimized queries, and large datasets

## Architecture

### Backend (Python FastAPI)
- **File**: `backend.py`
- **Port**: 8000
- **Features**:
  - RESTful API endpoints for data access
  - Database connection to TimescaleDB
  - Filtering and pagination
  - Statistical analysis
  - CORS enabled for frontend communication

### Frontend (React)
- **Port**: 3000 (development)
- **Features**:
  - Interactive data filters
  - Data table with pagination
  - Multiple chart types using Recharts
  - Statistics dashboard
  - Responsive UI design

## Setup Instructions

### Prerequisites
1. **Database**: Ensure your TimescaleDB is running with forex data loaded
2. **Python**: Python 3.8+ with virtual environment
3. **Node.js**: Node.js 14+ and npm

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   cd visualization
   pip install -r requirements.txt
   ```

2. **Configure database connection**:
   - The backend uses the same configuration as your main forex system
   - Ensure your `.env` file or environment variables are set with database credentials

3. **Start the backend server**:
   ```bash
   python start_backend.py
   ```
   
   Or manually:
   ```bash
   uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend Setup

1. **Install Node.js dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server**:
   ```bash
   ./start_frontend.sh
   ```
   
   Or manually:
   ```bash
   cd frontend
   npm start
   ```

### Quick Start

1. **Start both servers**:
   ```bash
   # Terminal 1 - Backend
   cd visualization
   python start_backend.py
   
   # Terminal 2 - Frontend
   cd visualization
   ./start_frontend.sh
   ```

2. **Access the dashboard**:
   - Open your browser to `http://localhost:3000`
   - The backend API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /health` - Check API and database connectivity

### Data Endpoints
- `GET /pairs` - Get available currency pairs
- `GET /intervals` - Get available time intervals
- `GET /data-range` - Get date range of available data
- `GET /data` - Get forex data with filtering
- `GET /statistics` - Get statistical summary
- `GET /chart-data` - Get data formatted for charts

### Example API Usage

```bash
# Get available pairs
curl http://localhost:8000/pairs

# Get EURUSD daily data for last month
curl "http://localhost:8000/data?pair=EURUSD&interval_type=1d&limit=30"

# Get statistics for EURUSD
curl "http://localhost:8000/statistics?pair=EURUSD&interval_type=1d"
```

## Usage Guide

### Filtering Data
1. **Currency Pair**: Select from available pairs in your database
2. **Interval**: Choose time interval (1m, 1d, etc.)
3. **Date Range**: Set start and end dates for filtering
4. **Record Limit**: Control number of records returned (100-10,000)

### Viewing Data
1. **Table View**: 
   - Paginated table with all data columns
   - Sortable and scrollable
   - Pagination controls

2. **Chart View**:
   - OHLC line charts showing price movements
   - Volume overlay charts
   - High-Low range visualizations
   - Interactive tooltips with detailed information

### Statistics Dashboard
- Total record count
- Date range coverage
- Price statistics (min, max, average, standard deviation)
- Real-time updates based on current filters

## Database Schema Support

The dashboard works with the forex_data table schema:
```sql
CREATE TABLE forex_data (
    timestamp       TIMESTAMPTZ NOT NULL,
    pair           VARCHAR(10) NOT NULL,
    interval_type  VARCHAR(5) NOT NULL,
    open           DECIMAL(20,8) NOT NULL,
    high           DECIMAL(20,8) NOT NULL,
    low            DECIMAL(20,8) NOT NULL,
    close          DECIMAL(20,8) NOT NULL,
    volume         BIGINT DEFAULT 0,
    source         VARCHAR(50),
    created_at     TIMESTAMPTZ DEFAULT NOW()
);
```

## Troubleshooting

### Backend Issues
- **Database Connection**: Check your database credentials and connectivity
- **Missing Dependencies**: Run `pip install -r requirements.txt`
- **Port Conflicts**: Change port in `start_backend.py` if 8000 is in use

### Frontend Issues
- **Dependencies**: Run `npm install` in the frontend directory
- **Port Conflicts**: React will automatically use port 3001 if 3000 is busy
- **API Connection**: Ensure backend is running on port 8000

### Common Errors
- **CORS Issues**: Backend has CORS enabled for localhost:3000
- **No Data**: Check that your database has data and filters are not too restrictive
- **Performance**: Use smaller date ranges and limits for better performance

## Development

### Adding New Features
1. **Backend**: Add new endpoints in `backend.py`
2. **Frontend**: Create new components in `src/components/`
3. **Styling**: Modify CSS in `src/index.css` or component files

### Database Optimization
- Indexes are automatically used for pair + timestamp queries
- Use appropriate date ranges to limit data transfer
- Consider adding more specific indexes for frequent query patterns

## Production Deployment

### Backend
```bash
# Use gunicorn for production
pip install gunicorn
gunicorn backend:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
```bash
# Build for production
cd frontend
npm run build

# Serve with a web server (nginx, apache, etc.)
```

### Environment Variables
Set these for production:
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- Consider using a reverse proxy (nginx) for serving both frontend and API