# Forex Data Collection System

A comprehensive, production-ready system for collecting, storing, and managing forex market data with 1-minute precision using TimescaleDB.

## 🏗️ System Architecture

### Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Data Collector  │───▶│  TimescaleDB    │
│                 │    │                  │    │                 │
│ • Alpha Vantage │    │ • Validation     │    │ • Time-series   │
│ • Yahoo Finance │    │ • Enrichment     │    │ • Hypertables   │
│ • OANDA API     │    │ • Rate Limiting  │    │ • Compression   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │   Scheduler      │
                       │                  │
                       │ • Real-time      │
                       │ • Data Updates   │
                       │ • Maintenance    │
                       └──────────────────┘
```

### Core Components

#### 1. **Configuration Management** (`config.py`)
- Centralized configuration using environment variables
- API key management and validation
- Database connection parameters
- System-wide constants and settings

#### 2. **Database Layer** (`database_manager.py`)
- **TimescaleDB Integration**: Optimized for time-series data
- **Hypertables**: Automatic partitioning by timestamp
- **Batch Operations**: Efficient bulk data insertion
- **Data Integrity**: Constraints and validation
- **Connection Pooling**: Managed database connections

#### 3. **Data Acquisition** (`data_collector.py`)
- **Multi-Source Support**: Alpha Vantage, Yahoo Finance, OANDA
- **Fallback Mechanisms**: Automatic source switching on failure
- **Rate Limiting**: Respects API limitations
- **Data Validation**: OHLC validation and cleaning
- **Async Operations**: Non-blocking data collection

#### 4. **System Orchestrator** (`forex_system.py`)
- **Scheduled Updates**: Automated real-time data collection
- **Historical Backfill**: Initial data population
- **Error Handling**: Robust error recovery
- **Monitoring**: System health and statistics
- **Graceful Shutdown**: Clean system termination

#### 5. **Setup & Deployment** (`setup.py`)
- **Automated Installation**: Complete system setup
- **Database Initialization**: Schema creation and configuration
- **Service Management**: Systemd integration for Linux

## 🚀 Quick Start

### 1. System Requirements
- **Python**: 3.8 or higher
- **PostgreSQL**: 12 or higher
- **TimescaleDB**: 2.0 or higher (recommended)
- **Memory**: 4GB RAM minimum
- **Storage**: 100GB+ for 10 years of data

### 2. Installation

```bash
# Clone the repository
git clone <repository_url>
cd forex-data-system

# Run the setup script
python setup.py
```

The setup script will:
- Install Python dependencies
- Configure PostgreSQL database
- Set up TimescaleDB extension
- Create database schema
- Configure API keys
- Initialize the system

### 3. Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=forex_data
DB_USER=forex_user
DB_PASSWORD=your_password

# API Keys (get free keys from providers)
ALPHA_VANTAGE_API_KEY=your_key_here
OANDA_API_KEY=your_key_here
```

### 4. Running the System

```bash
# Full system with historical data collection
python forex_system.py

# Skip historical data collection
python forex_system.py --no-historical

# Collect specific pairs only
python forex_system.py --pairs EURUSD GBPUSD

# Run validation only
python forex_system.py --validate

# Show statistics
python forex_system.py --stats
```

## 📊 Data Schema

### Forex Data Table
```sql
CREATE TABLE forex_data (
    timestamp       TIMESTAMPTZ NOT NULL,
    pair           VARCHAR(10) NOT NULL,
    open           DECIMAL(20,8) NOT NULL,
    high           DECIMAL(20,8) NOT NULL,
    low            DECIMAL(20,8) NOT NULL,
    close          DECIMAL(20,8) NOT NULL,
    volume         BIGINT,
    spread         DECIMAL(10,8),
    tick_volume    INTEGER,
    source         VARCHAR(50) NOT NULL,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- TimescaleDB hypertable partitioned by timestamp
SELECT create_hypertable('forex_data', 'timestamp');
```

### Supported Forex Pairs
- **Major Pairs**: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD
- **Cross Pairs**: EUR/GBP, EUR/JPY, GBP/JPY, etc.
- **Configurable**: Easy to add/remove pairs

## 🔧 System Features

### Data Collection
- **1-Minute Intervals**: High-resolution market data
- **10-Year History**: Comprehensive historical data
- **Real-Time Updates**: Continuous data streaming
- **Multiple Sources**: Redundancy and reliability
- **Data Validation**: Quality assurance and cleaning

### Performance Optimization
- **TimescaleDB**: Purpose-built for time-series data
- **Batch Processing**: Efficient bulk operations
- **Compression**: Automatic data compression
- **Indexing**: Optimized query performance
- **Connection Pooling**: Managed database connections

### Monitoring & Maintenance
- **Automated Cleanup**: Configurable data retention
- **Health Checks**: System monitoring and alerts
- **Statistics**: Comprehensive reporting
- **Error Recovery**: Robust error handling
- **Logging**: Detailed system logs

### Scalability
- **Horizontal Scaling**: Multiple collector instances
- **Load Balancing**: Distributed data collection
- **Cloud Ready**: Docker and Kubernetes support
- **High Availability**: Redundant components

## 🔄 Data Flow

### Historical Data Collection
1. **Initialization**: System startup and configuration validation
2. **Source Selection**: Choose optimal data provider
3. **Bulk Download**: Fetch historical data in batches
4. **Validation**: Clean and validate OHLC data
5. **Enrichment**: Add technical indicators and metadata
6. **Storage**: Batch insert into TimescaleDB

### Real-Time Updates
1. **Scheduled Trigger**: Every minute (configurable)
2. **Multi-Source Query**: Fetch latest data from APIs
3. **Data Validation**: Ensure data quality
4. **Deduplication**: Prevent duplicate entries
5. **Storage**: Insert new records
6. **Monitoring**: Log success/failure rates

## 📈 Usage Examples

### Basic Data Retrieval
```python
from database_manager import DatabaseManager
from datetime import datetime, timedelta

db = DatabaseManager()

# Get last 24 hours of EUR/USD data
end_time = datetime.now()
start_time = end_time - timedelta(days=1)

data = db.get_forex_data('EURUSD', start_time, end_time)
print(f"Retrieved {len(data)} records")
```

### System Statistics
```python
from forex_system import ForexDataSystem

system = ForexDataSystem()
system.initialize_system()
system.print_system_statistics()
```

### Custom Data Collection
```python
from data_collector import ForexDataCollector
import asyncio

async def collect_custom_data():
    async with ForexDataCollector() as collector:
        data = await collector.get_real_time_data(['EURUSD', 'GBPUSD'])
        return data

# Run the collection
data = asyncio.run(collect_custom_data())
```

## 🛠️ Advanced Configuration

### Custom Pairs
```python
# In config.py, modify MAJOR_PAIRS
MAJOR_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY',  # Major pairs
    'EURJPY', 'GBPJPY',            # Cross pairs
    'XAUUSD', 'XAGUSD'             # Commodities (if supported)
]
```

### Performance Tuning
```python
# Adjust batch sizes for your system
BATCH_SIZE = 5000          # Larger batches for better performance
UPDATE_INTERVAL_MINUTES = 1  # More frequent updates
MAX_RETRIES = 5           # More retry attempts
```

### Database Optimization
```sql
-- Additional indexes for specific queries
CREATE INDEX idx_forex_data_close ON forex_data (close);
CREATE INDEX idx_forex_data_volume ON forex_data (volume) WHERE volume IS NOT NULL;

-- Compression settings (TimescaleDB)
SELECT add_compression_policy('forex_data', INTERVAL '7 days');
```

## 🔒 Security Considerations

### API Key Management
- Store API keys in environment variables
- Use separate keys for development/production
- Monitor API usage and rate limits
- Implement key rotation procedures

### Database Security
- Use dedicated database user with minimal privileges
- Enable SSL connections for remote databases
- Regular security updates and patches
- Monitor access logs

### Network Security
- Firewall configuration for database access
- VPN for remote connections
- Rate limiting for API endpoints
- DDoS protection considerations

## 📊 Monitoring & Alerting

### Key Metrics
- **Data Freshness**: Time since last update per pair
- **Collection Success Rate**: Percentage of successful API calls
- **Database Performance**: Query execution times
- **Storage Usage**: Disk space and growth trends
- **Error Rates**: Failed collections and reasons

### Alerting Setup
```python
# Example Slack/email alerting
def send_alert(message, severity='warning'):
    if severity == 'critical':
        # Send immediate notification
        send_slack_message(f"🚨 CRITICAL: {message}")
    elif severity == 'warning':
        # Log and send daily summary
        logger.warning(message)
```

## 🐳 Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "forex_system.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: forex_data
      POSTGRES_USER: forex_user
      POSTGRES_PASSWORD: forex_password
    volumes:
      - forex_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  forex-system:
    build: .
    depends_on:
      - timescaledb
    environment:
      - DB_HOST=timescaledb
    volumes:
      - ./logs:/app/logs

volumes:
  forex_data:
```

## 🧪 Testing

### Unit Tests
```bash
# Run test suite
python -m pytest tests/

# Test specific component
python -m pytest tests/test_data_collector.py

# Test with coverage
python -m pytest --cov=. tests/
```

### Integration Tests
```bash
# Test database connectivity
python forex_system.py --validate

# Test data collection
python -c "
from data_collector import ForexDataCollector
import asyncio

async def test():
    async with ForexDataCollector() as collector:
        data = await collector.get_real_time_data(['EURUSD'])
        print(f'Collected {len(data)} records')

asyncio.run(test())
"
```

## 🔧 Troubleshooting

### Common Issues

#### Database Connection Errors
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection manually
psql -h localhost -U forex_user -d forex_data
```

#### API Rate Limiting
- **Alpha Vantage**: 5 requests/minute (free tier)
- **Yahoo Finance**: Informal limits, use sparingly
- **OANDA**: 120 requests/minute (practice account)

#### Memory Issues
- Reduce batch sizes in configuration
- Implement data archival for old records
- Monitor TimescaleDB compression

#### Data Quality Issues
```python
# Manual data validation
from forex_system import ForexDataSystem

system = ForexDataSystem()
system.validate_data_integrity()
```

### Performance Optimization

#### Database Tuning
```sql
-- PostgreSQL configuration
shared_preload_libraries = 'timescaledb'
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
```

#### System Resources
- **CPU**: Multi-core for parallel processing
- **Memory**: 8GB+ for large datasets
- **Storage**: SSD recommended for database
- **Network**: Stable internet for API calls

## 📚 API Documentation

### Free API Sources

#### Alpha Vantage
- **Free Tier**: 5 requests/minute, 500 requests/day
- **Registration**: https://www.alphavantage.co/support/#api-key
- **Documentation**: https://www.alphavantage.co/documentation/

#### Yahoo Finance
- **Limitations**: Unofficial API, rate limits vary
- **Data**: Limited historical intraday data
- **Reliability**: Good for major pairs, less reliable for exotics

#### OANDA
- **Practice Account**: Free with registration
- **Rate Limits**: 120 requests/minute
- **Data Quality**: Professional-grade, real-time
- **Registration**: https://www.oanda.com/account/

### Paid API Alternatives

For production systems, consider:
- **Bloomberg API**: Professional-grade data
- **Reuters Eikon**: Comprehensive market data
- **IEX Cloud**: Cost-effective with good coverage
- **Polygon.io**: Real-time forex data

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository_url>
cd forex-data-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions
- Include unit tests for new features

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request with description

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

### Documentation
- System architecture diagrams
- API reference documentation
- Configuration examples
- Troubleshooting guides

### Community
- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and community support
- Wiki: Extended documentation and examples

### Professional Support
For production deployments and enterprise features:
- Custom data sources integration
- High-availability configurations
- Performance optimization consulting
- 24/7 monitoring and support

---

**Built with ❤️ for forex traders and quantitative analysts**