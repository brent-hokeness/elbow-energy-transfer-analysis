"""
Data exploration utilities for examining database tables and structure.
"""

import pandas as pd
from typing import List, Dict, Any

class DataExplorer:
    """Utility class for exploring database structure and data."""
    
    def __init__(self, db_connection):
        """Initialize with database connection."""
        self.db = db_connection
    
    def describe_table(self, table_name: str) -> None:
        """Print detailed information about a table structure."""
        try:
            # Get table structure
            table_info = self.db.get_table_info(table_name)
            
            if table_info:
                print(f"   Columns in {table_name}:")
                for col in table_info:
                    col_name = col['Field']
                    col_type = col['Type']
                    is_null = col['Null']
                    key_info = col['Key']
                    default = col['Default']
                    
                    key_str = f" ({key_info})" if key_info else ""
                    null_str = "NULL" if is_null == "YES" else "NOT NULL"
                    default_str = f" DEFAULT: {default}" if default else ""
                    
                    print(f"     • {col_name}: {col_type}{key_str} - {null_str}{default_str}")
                
                # Get row count
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_result = self.db.execute_query(count_query)
                if count_result:
                    row_count = count_result[0]['count']
                    print(f"   Total rows: {row_count:,}")
                    
            else:
                print(f"   ❌ Could not retrieve structure for table: {table_name}")
                
        except Exception as e:
            print(f"   ❌ Error describing table {table_name}: {e}")
    
    def show_sample_data(self, table_name: str, n_rows: int = 3) -> None:
        """Display sample data from a table."""
        try:
            sample_data = self.db.get_table_sample(table_name, n_rows)
            
            if sample_data:
                print(f"   Sample data from {table_name}:")
                
                # Convert to DataFrame for better display
                df = pd.DataFrame(sample_data)
                
                # Display with proper formatting
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', 50)
                
                print(f"   {df.to_string(index=False)}")
                print()
                
            else:
                print(f"   ❌ Could not retrieve sample data from {table_name}")
                
        except Exception as e:
            print(f"   ❌ Error getting sample data from {table_name}: {e}")