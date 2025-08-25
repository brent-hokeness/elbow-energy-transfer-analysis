"""
Database connection module for MariaDB/MySQL database.
Handles connection establishment and basic database operations.
"""

import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

class DatabaseConnection:
    """Handles database connection and basic operations."""
    
    def __init__(self):
        """Initialize database connection using environment variables."""
        # Load environment variables from .env file
        load_dotenv()
        
        self.connection = None
        self.cursor = None
        
        # Database configuration from environment variables
        self.config = {
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'port': int(os.getenv('DB_PORT', 3306))
        }
        
        # Validate that all required environment variables are present
        required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        self._connect()
    
    def _connect(self):
        """Establish connection to the database."""
        try:
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor(dictionary=True)
            print(f"Connected to MariaDB database: {self.config['database']}")
            
        except Error as e:
            print(f"Error connecting to MariaDB: {e}")
            raise
    
    def test_connection(self):
        """Test if the database connection is working."""
        try:
            if self.connection and self.connection.is_connected():
                self.cursor.execute("SELECT 1")
                result = self.cursor.fetchone()
                return result is not None
            return False
        except Error as e:
            print(f"Connection test failed: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """Execute a SELECT query and return results."""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchall()
        except Error as e:
            print(f"Error executing query: {e}")
            return None
    
    def get_table_info(self, table_name):
        """Get column information for a table."""
        query = f"DESCRIBE {table_name}"
        return self.execute_query(query)
    
    def get_table_sample(self, table_name, n_rows=5):
        """Get a sample of rows from a table."""
        query = f"SELECT * FROM {table_name} LIMIT {n_rows}"
        return self.execute_query(query)
    
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed.")