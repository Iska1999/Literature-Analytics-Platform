# utils/db_manager.py
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
import pandas as pd
from contextlib import contextmanager
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class DatabaseManager:
    """Manages database connections using environment variables"""
    
    def __init__(self):
        # Validate configuration
        Config.validate()
        
        self.host = Config.DB_HOST
        self.username = Config.DB_USER
        self.password = Config.DB_PASSWORD
        self.database = Config.DB_NAME
        self.engine = create_engine(self.get_database_url())
        self._server_engine = None
    
    def get_server_url(self):
        """Get URL for server connection (without database)"""
        return URL.create(
            drivername="mysql+pymysql",
            host=self.host,
            port=3306,
            username=self.username,
            password=self.password
        )
    
    def get_database_url(self):
        """Get URL for database connection"""
        return URL.create(
            drivername="mysql+pymysql",
            host=self.host,
            port=3306,
            username=self.username,
            password=self.password,
            database=self.database
        )
    
    def create_database_if_not_exists(self):
        """Create database if it doesn't exist"""
        if not self._server_engine:
            self._server_engine = create_engine(self.get_server_url())
        
        with self._server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {self.database}"))
            conn.commit()
    
    def get_engine(self):
        """Get or create database engine"""
        if not self.engine:
            self.engine = create_engine(self.get_database_url())
        return self.engine
    
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        engine = self.get_engine()
        connection = engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def read_table(self, table_name):
        """Read a table from the database"""
        query = f"SELECT * FROM {self.database}.{table_name}"
        eng = self.get_engine()
        return pd.read_sql(query,con=eng )
    
    def write_table(self, df, table_name, if_exists='replace'):
        """Write a DataFrame to the database"""
        df.to_sql(con=self.get_engine(), name=table_name, if_exists=if_exists, index=False)
    
    def dispose(self):
        """Properly dispose of the engines"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
        if self._server_engine:
            self._server_engine.dispose()
            self._server_engine = None

# Create a singleton instance
db_manager = DatabaseManager()
db_manager.read_table('monthly_metadata')