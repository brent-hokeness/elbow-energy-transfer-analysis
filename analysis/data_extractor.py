"""
Data extraction module for extracting kinematic variables at specific time points.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np

class KinematicExtractor:
    """Extracts kinematic variables at specific time points for each pitch."""
    
    def __init__(self, db_connection):
        """Initialize with database connection."""
        self.db = db_connection
        
    def get_energy_transfer_data(self) -> pd.DataFrame:
        """Get energy transfer metrics from poi table for pitchers over 85 mph."""
        query = """
        SELECT session_trial, elbow_transfer_fp_br, pitch_speed_mph
        FROM poi 
        WHERE elbow_transfer_fp_br IS NOT NULL 
        AND pitch_speed_mph >= 85.0
        """
        
        results = self.db.execute_query(query)
        if results:
            df = pd.DataFrame(results)
            print(f"Retrieved energy transfer data for {len(df)} pitches over 85 mph")
            return df
        else:
            print("No energy transfer data found for pitches over 85 mph")
            return pd.DataFrame()
    
    def get_event_times(self, session_trials: List[str]) -> pd.DataFrame:
        """Get event timing data from events table for specific session_trials."""
        if not session_trials:
            return pd.DataFrame()
            
        # Create placeholder string for IN clause
        placeholders = ','.join(['%s'] * len(session_trials))
        query = f"""
        SELECT session_trial, FP_v6_time, MER_time, MIR_time, BR_time
        FROM events 
        WHERE session_trial IN ({placeholders})
        AND FP_v6_time IS NOT NULL AND MER_time IS NOT NULL 
        AND MIR_time IS NOT NULL AND BR_time IS NOT NULL
        """
        
        results = self.db.execute_query(query, session_trials)
        if results:
            df = pd.DataFrame(results)
            print(f"Retrieved event timing data for {len(df)} pitches")
            return df
        else:
            print("No event timing data found")
            return pd.DataFrame()

    def get_kinematic_at_timepoint(self, session_trial: str, target_time: float, 
                                    kinematic_columns: List[str]) -> Dict[str, float]:
            """Extract kinematic values at a specific time point for a given pitch."""
            # Create placeholder string for column selection
            column_list = ', '.join(kinematic_columns)
            
            query = f"""
            SELECT time, {column_list}
            FROM joint_angles 
            WHERE session_trial = %s 
            AND time BETWEEN %s AND %s
            ORDER BY ABS(time - %s)
            LIMIT 1
            """
            
            # Search within 0.1 seconds of target time
            time_window = 0.1
           # Convert decimal to float for math operations
            target_time_float = float(target_time)
            params = [session_trial, target_time_float - time_window, target_time_float + time_window, target_time_float]
            
            result = self.db.execute_query(query, params)
            
            if result and len(result) > 0:
                # Return the kinematic values (excluding time)
                row = result[0]
                return {col: row[col] for col in kinematic_columns if row[col] is not None}
            else:
                return {}
    
    def extract_features_for_pitch(self, session_trial: str, event_times: Dict[str, float]) -> Dict[str, float]:
        """Extract all kinematic features for a single pitch at all time points."""
        from .kinematic_variables import ALL_KINEMATIC_VARIABLES, TIME_POINTS
        
        features = {'session_trial': session_trial}
        
        # Extract kinematic variables at each time point
        for time_point_name, time_column in TIME_POINTS.items():
            if time_column in event_times:
                target_time = event_times[time_column]
                
                # Get kinematic values at this time point
                kinematic_values = self.get_kinematic_at_timepoint(
                    session_trial, target_time, ALL_KINEMATIC_VARIABLES
                )
                
                # Add to features with naming convention: variable_timepoint
                for variable, value in kinematic_values.items():
                    feature_name = f"{variable}_{time_point_name}"
                    features[feature_name] = value
        
        return features

    def extract_all_features(self) -> pd.DataFrame:
        """Extract kinematic features for all pitches and merge with energy transfer data."""
        print("Starting feature extraction for all pitches...")
        
        # Get energy transfer data
        energy_df = self.get_energy_transfer_data()
        if energy_df.empty:
            return pd.DataFrame()
        
        # Get event times
        session_trials = energy_df['session_trial'].tolist()
        events_df = self.get_event_times(session_trials)
        if events_df.empty:
            return pd.DataFrame()
        
        # Merge energy and event data
        merged_df = energy_df.merge(events_df, on='session_trial', how='inner')
        print(f"Processing {len(merged_df)} pitches...")
        
        # Extract features for each pitch
        all_features = []
        for idx, row in merged_df.iterrows():
            session_trial = row['session_trial']
            event_times = {
                'FP_v6_time': row['FP_v6_time'],
                'MER_time': row['MER_time'],
                'MIR_time': row['MIR_time'],
                'BR_time': row['BR_time']
            }
            
            # Extract features for this pitch
            features = self.extract_features_for_pitch(session_trial, event_times)
            features['elbow_transfer_fp_br'] = row['elbow_transfer_fp_br']
            features['pitch_speed_mph'] = row['pitch_speed_mph']
            
            all_features.append(features)
            
            # Progress indicator
            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{len(merged_df)} pitches...")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        print(f"Feature extraction complete! Shape: {features_df.shape}")
        
        return features_df    