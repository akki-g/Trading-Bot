#!/usr/bin/env python3
"""
Setup script for the Forex Data Collection System.

This script handles:
- Database installation and configuration
- Python dependencies
- System initialization
- Configuration validation
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import getpass
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ForexSystemSetup:
    """Handles complete system setup for the forex data collection system."""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.env_file = self.project_dir / '.env'
        self.config = {}
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        logger.info(f"Python version {sys.version} is compatible")
        return True
    
    def install_python_dependencies(self) -> bool:
        """Install required Python packages."""
        logger.info("Installing Python dependencies...")
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            
            # Install requirements
            requirements_file = self.project_dir / 'requirements.txt'
            if requirements_file.exists():
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)], 
                             check=True, capture_output=True)
                logger.info("Python dependencies installed successfully")
                return True
            else:
                logger.error("requirements.txt not found")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Python dependencies: {e}")
            return False
    
    def check_postgresql_installation(self) -> bool:
        """Check if PostgreSQL is installed and accessible."""
        try:
            result = subprocess.run(['psql', '--version'], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"PostgreSQL found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("PostgreSQL not found. Please install PostgreSQL first.")
            logger.info("Installation instructions:")
            logger.info("  Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib")
            logger.info("  CentOS/RHEL: sudo yum install postgresql postgresql-server")
            logger.info("  macOS: brew install postgresql")
            logger.info("  Windows: Download from https://www.postgresql.org/download/windows/")
            return False
    
    def setup_timescaledb(self) -> bool:
        """Setup TimescaleDB extension."""
        logger.info("Setting up TimescaleDB...")
        
        try:
            # Check if TimescaleDB is available
            with psycopg2.connect(
                host=self.config['DB_HOST'],
                port=self.config['DB_PORT'],
                database='postgres',
                user=self.config['DB_USER'],
                password=self.config['DB_PASSWORD']
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM pg_extension WHERE extname = 'timescaledb';")
                    if cursor.fetchone():
                        logger.info("TimescaleDB extension already available")
                        return True
            
            # If not available, provide installation instructions
            logger.warning("TimescaleDB extension not found")
            logger.info("TimescaleDB installation instructions:")
            logger.info("1. Visit: https://docs.timescale.com/install/latest/")
            logger.info("2. Follow the installation guide for your OS")
            logger.info("3. Re-run this setup script")
            
            # Ask if user wants to continue without TimescaleDB
            response = input("Continue without TimescaleDB? (y/N): ").lower()
            return response == 'y'
            
        except Exception as e:
            logger.warning(f"Could not verify TimescaleDB installation: {e}")
            return True  # Continue anyway
    
    def create_database_user(self) -> bool:
        """Create database user for the forex system."""
        logger.info("Creating database user...")
        
        try:
            # Connect as superuser to create user and database
            admin_password = getpass.getpass("Enter PostgreSQL admin password: ")
            
            with psycopg2.connect(
                host=self.config['DB_HOST'],
                port=self.config['DB_PORT'],
                database='postgres',
                user='akshatguduru',
                password=admin_password
            ) as conn:
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                with conn.cursor() as cursor:
                    # Create user
                    cursor.execute(f"""
                        DO $$
                        BEGIN
                            IF NOT EXISTS (SELECT FROM pg_user WHERE usename = '{self.config['DB_USER']}') THEN
                                CREATE USER {self.config['DB_USER']} WITH PASSWORD '{self.config['DB_PASSWORD']}';
                            END IF;
                        END
                        $$;
                    """)
                    
                    # Create database
                    cursor.execute(f"""
                        SELECT 1 FROM pg_database WHERE datname = '{self.config['DB_NAME']}'
                    """)
                    
                    if not cursor.fetchone():
                        cursor.execute(f"CREATE DATABASE {self.config['DB_NAME']} OWNER {self.config['DB_USER']}")
                        logger.info(f"Created database: {self.config['DB_NAME']}")
                    else:
                        logger.info(f"Database {self.config['DB_NAME']} already exists")
                    
                    # Grant permissions
                    cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {self.config['DB_NAME']} TO {self.config['DB_USER']}")
            
            logger.info("Database user and database created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database user: {e}")
            return False
    
    def collect_configuration(self) -> bool:
        """Collect configuration from user input."""
        logger.info("Collecting system configuration...")
        
        # Database configuration
        self.config['DB_HOST'] = input("Database host [localhost]: ").strip() or 'localhost'
        self.config['DB_PORT'] = input("Database port [5432]: ").strip() or '5432'
        self.config['DB_NAME'] = input("Database name [forex_data]: ").strip() or 'forex_data'
        self.config['DB_USER'] = input("Database user [forex_user]: ").strip() or 'forex_user'
        self.config['DB_PASSWORD'] = getpass.getpass("Database password: ")
        
        # API keys
        print("\nAPI Configuration (optional, but recommended):")
        print("You can get a free Alpha Vantage API key at: https://www.alphavantage.co/support/#api-key")
        self.config['ALPHA_VANTAGE_API_KEY'] = input("Alpha Vantage API key (optional): ").strip()
        
        print("For OANDA API access, sign up at: https://www.oanda.com/account/")
        self.config['OANDA_API_KEY'] = input("OANDA API key (optional): ").strip()
        self.config['OANDA_ACCOUNT_ID'] = input("OANDA Account ID (optional): ").strip()
        
        # System configuration
        self.config['LOG_LEVEL'] = input("Log level [INFO]: ").strip() or 'INFO'
        self.config['LOG_FILE'] = input("Log file [forex_system.log]: ").strip() or 'forex_system.log'
        
        return True
    
    def save_configuration(self) -> bool:
        """Save configuration to .env file."""
        logger.info("Saving configuration...")
        
        try:
            with open(self.env_file, 'w') as f:
                for key, value in self.config.items():
                    f.write(f"{key}={value}\n")
            
            logger.info(f"Configuration saved to {self.env_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def test_database_connection(self) -> bool:
        """Test database connection with the configured settings."""
        logger.info("Testing database connection...")
        
        try:
            with psycopg2.connect(
                host=self.config['DB_HOST'],
                port=self.config['DB_PORT'],
                database=self.config['DB_NAME'],
                user=self.config['DB_USER'],
                password=self.config['DB_PASSWORD']
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    if result[0] == 1:
                        logger.info("Database connection successful")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def initialize_forex_system(self) -> bool:
        """Initialize the forex data system."""
        logger.info("Initializing forex data system...")
        
        try:
            # Import after dependencies are installed
            from forex_system import ForexDataSystem
            
            system = ForexDataSystem()
            if system.initialize_system():
                logger.info("Forex data system initialized successfully")
                return True
            else:
                logger.error("Forex data system initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize forex system: {e}")
            return False
    
    def create_systemd_service(self) -> bool:
        """Create a systemd service file for automatic startup (Linux only)."""
        if os.name != 'posix':
            logger.info("Systemd service creation is only available on Linux")
            return True
        
        service_content = f"""[Unit]
Description=Forex Data Collection System
After=network.target postgresql.service

[Service]
Type=simple
User={os.getenv('USER', 'forex')}
WorkingDirectory={self.project_dir}
Environment=PATH={sys.executable}
ExecStart={sys.executable} forex_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_file = '/etc/systemd/system/forex-data-system.service'
        
        try:
            response = input("Create systemd service for automatic startup? (y/N): ").lower()
            if response == 'y':
                with open('/tmp/forex-data-system.service', 'w') as f:
                    f.write(service_content)
                
                logger.info(f"Service file created at /tmp/forex-data-system.service")
                logger.info(f"To install, run: sudo mv /tmp/forex-data-system.service {service_file}")
                logger.info("Then run: sudo systemctl enable forex-data-system")
                logger.info("And: sudo systemctl start forex-data-system")
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not create systemd service: {e}")
            return True  # Not critical
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        logger.info("Starting Forex Data System Setup")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check for existing configuration
        if self.env_file.exists():
            response = input("Configuration file exists. Overwrite? (y/N): ").lower()
            if response != 'y':
                logger.info("Setup cancelled")
                return False
        
        # Collect configuration
        if not self.collect_configuration():
            return False
        
        # Save configuration
        if not self.save_configuration():
            return False
        
        # Install Python dependencies
        if not self.install_python_dependencies():
            return False
        
        # Check PostgreSQL
        if not self.check_postgresql_installation():
            return False
        
        # Create database user and database
        if not self.create_database_user():
            return False
        
        # Test database connection
        if not self.test_database_connection():
            return False
        
        # Setup TimescaleDB
        if not self.setup_timescaledb():
            return False
        
        # Initialize forex system
        if not self.initialize_forex_system():
            return False
        
        # Create systemd service (optional)
        self.create_systemd_service()
        
        logger.info("Setup completed successfully!")
        logger.info(f"You can now run the system with: python {self.project_dir}/forex_system.py")
        
        return True

def main():
    """Main setup function."""
    setup = ForexSystemSetup()
    
    try:
        success = setup.run_setup()
        if success:
            print("\n🎉 Forex Data System setup completed successfully!")
            print("\nNext steps:")
            print("1. Run the system: python forex_system.py")
            print("2. Check logs: tail -f forex_system.log")
            print("3. Monitor data: python forex_system.py --stats")
            return 0
        else:
            print("\n❌ Setup failed. Please check the logs and try again.")
            return 1
            
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())