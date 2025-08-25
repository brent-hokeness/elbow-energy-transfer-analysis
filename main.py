"""
Main script for elbow energy transfer kinematic analysis project.
This script coordinates the analysis of kinematic variables related to elbow energy transfer.
"""

import sys
import os
from database.db_connection import DatabaseConnection
from utils.data_explorer import DataExplorer

def main():
    """Main function to run the analysis pipeline."""
    print("=== Elbow Energy Transfer Kinematic Analysis ===")
    print("Step 1: Connecting to database and exploring data structure...\n")
    
    try:
        # Initialize database connection
        db = DatabaseConnection()
        
        # Test connection
        if db.test_connection():
            print("✅ Database connection successful!")
        else:
            print("❌ Database connection failed!")
            return
        
        # Initialize data explorer
        explorer = DataExplorer(db)
        
        # Explore table structures
        print("\n--- Exploring Table Structures ---")
        tables = ['events', 'poi', 'joint_angles']
        
        for table in tables:
            print(f"\n📊 Exploring {table} table:")
            explorer.describe_table(table)
            explorer.show_sample_data(table, n_rows=3)
        
        # Close connection
        db.close()
        print("\n✅ Step 1 completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in Step 1: {str(e)}")
        return

if __name__ == "__main__":
    main()