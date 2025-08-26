#!/usr/bin/env python3
"""
Statistical Parametric Mapping (SPM) Regression Analysis
Shoulder Angle Y vs Elbow Energy Transfer from Foot Plant to Ball Release

This script extracts time-series data from the joint_angles table and performs
SPM regression to identify time windows where shoulder abduction significantly
predicts elbow energy transfer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from database.db_connection import DatabaseConnection
import warnings
warnings.filterwarnings('ignore')

class SPMRegression:
    def __init__(self):
        self.db = None
        self.energy_data = None
        self.time_series_data = {}
        self.normalized_data = None
        self.spm_results = None
        
    def load_energy_transfer_data(self):
        """Load elbow energy transfer data and event times."""
        print("Loading energy transfer and event timing data...")
        
        query = """
        SELECT p.session_trial, p.elbow_transfer_fp_br, 
               e.FP_v6_time, e.BR_time
        FROM poi p
        JOIN events e ON p.session_trial = e.session_trial
        WHERE p.elbow_transfer_fp_br IS NOT NULL 
        AND p.pitch_speed_mph >= 85.0
        AND e.FP_v6_time IS NOT NULL 
        AND e.BR_time IS NOT NULL
        ORDER BY p.session_trial
        """
        
        results = self.db.execute_query(query)
        if not results:
            raise ValueError("No energy transfer data found")
        
        self.energy_data = pd.DataFrame(results)
        
        # Convert to numeric
        for col in ['elbow_transfer_fp_br', 'FP_v6_time', 'BR_time']:
            self.energy_data[col] = pd.to_numeric(self.energy_data[col])
        
        # Filter out invalid time ranges
        valid_mask = (self.energy_data['BR_time'] > self.energy_data['FP_v6_time']) & \
                    ((self.energy_data['BR_time'] - self.energy_data['FP_v6_time']) > 0.05) & \
                    ((self.energy_data['BR_time'] - self.energy_data['FP_v6_time']) < 0.5)
        
        self.energy_data = self.energy_data[valid_mask].reset_index(drop=True)
        
        print(f"Loaded {len(self.energy_data)} pitches with valid energy transfer and timing data")
        return self.energy_data

    def extract_time_series_data(self, max_pitches=1000):
        """Extract shoulder angle Y time series for each pitch from foot plant to ball release."""
        print(f"Extracting shoulder angle Y time series data...")
        
        # Limit to manageable number for processing
        sample_data = self.energy_data.head(max_pitches).copy()
        
        valid_series = []
        
        for idx, row in sample_data.iterrows():
            session_trial = row['session_trial']
            fp_time = float(row['FP_v6_time'])
            br_time = float(row['BR_time'])
            
            # Query time series data for this pitch
            query = """
            SELECT time, shoulder_angle_y
            FROM joint_angles 
            WHERE session_trial = %s 
            AND time BETWEEN %s AND %s
            AND shoulder_angle_y IS NOT NULL
            ORDER BY time
            """
            
            time_series = self.db.execute_query(query, [session_trial, fp_time, br_time])
            
            if time_series and len(time_series) >= 10:  # Need minimum time points
                ts_df = pd.DataFrame(time_series)
                ts_df['time'] = pd.to_numeric(ts_df['time'])
                ts_df['shoulder_angle_y'] = pd.to_numeric(ts_df['shoulder_angle_y'])
                
                # Store time series data
                self.time_series_data[session_trial] = {
                    'time': ts_df['time'].values,
                    'shoulder_angle_y': ts_df['shoulder_angle_y'].values,
                    'energy_transfer': row['elbow_transfer_fp_br'],
                    'duration': br_time - fp_time
                }
                valid_series.append(row)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(sample_data)} pitches...")
        
        print(f"Successfully extracted time series for {len(self.time_series_data)} pitches")
        
        # Update energy data to only include pitches with time series
        valid_trials = list(self.time_series_data.keys())
        self.energy_data = pd.DataFrame(valid_series)
        
        return len(self.time_series_data)

    def normalize_time_series(self, n_points=101):
        """Time-normalize all series to same number of points (0-100% of phase)."""
        print(f"Time-normalizing all series to {n_points} points...")
        
        normalized_shoulder_data = []
        energy_values = []
        
        for session_trial, data in self.time_series_data.items():
            time_original = data['time']
            shoulder_original = data['shoulder_angle_y']
            
            # Normalize time from 0 to 100%
            time_normalized = np.linspace(0, 100, n_points)
            time_percent = (time_original - time_original[0]) / (time_original[-1] - time_original[0]) * 100
            
            # Interpolate shoulder angle to normalized time points
            try:
                f_interp = interpolate.interp1d(time_percent, shoulder_original, 
                                              kind='linear', fill_value='extrapolate')
                shoulder_normalized = f_interp(time_normalized)
                
                normalized_shoulder_data.append(shoulder_normalized)
                energy_values.append(data['energy_transfer'])
                
            except Exception as e:
                print(f"Interpolation failed for {session_trial}: {e}")
                continue
        
        # Convert to arrays
        self.normalized_data = {
            'shoulder_angle_y': np.array(normalized_shoulder_data),  # Shape: (n_subjects, n_timepoints)
            'energy_transfer': np.array(energy_values),  # Shape: (n_subjects,)
            'time_normalized': np.linspace(0, 100, n_points)  # 0-100% of delivery phase
        }
        
        print(f"Time normalization complete: {self.normalized_data['shoulder_angle_y'].shape[0]} subjects, "
              f"{self.normalized_data['shoulder_angle_y'].shape[1]} time points")
        
        return self.normalized_data

    def run_spm_regression(self, alpha=0.05):
        """Run SPM regression analysis."""
        print("Running SPM regression analysis...")
        
        shoulder_data = self.normalized_data['shoulder_angle_y']  # (n_subjects, n_timepoints)
        energy_data = self.normalized_data['energy_transfer']     # (n_subjects,)
        time_points = self.normalized_data['time_normalized']
        
        n_subjects, n_timepoints = shoulder_data.shape
        
        # Initialize results
        t_stats = np.zeros(n_timepoints)
        p_values = np.zeros(n_timepoints)
        correlations = np.zeros(n_timepoints)
        
        # Run regression at each time point
        for t in range(n_timepoints):
            shoulder_at_t = shoulder_data[:, t]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(shoulder_at_t) | np.isnan(energy_data))
            if np.sum(valid_mask) < 10:  # Need minimum subjects
                t_stats[t] = 0
                p_values[t] = 1.0
                correlations[t] = 0
                continue
            
            shoulder_valid = shoulder_at_t[valid_mask]
            energy_valid = energy_data[valid_mask]
            
            # Calculate correlation
            try:
                r, p = stats.pearsonr(shoulder_valid, energy_valid)
                correlations[t] = r
                
                # Convert to t-statistic for SPM
                n_valid = len(shoulder_valid)
                t_stat = r * np.sqrt((n_valid - 2) / (1 - r**2)) if abs(r) < 0.999 else 0
                t_stats[t] = t_stat
                p_values[t] = p
                
            except Exception:
                t_stats[t] = 0
                p_values[t] = 1.0
                correlations[t] = 0
        
        # Apply multiple comparison correction (Bonferroni)
        p_corrected = np.minimum(p_values * n_timepoints, 1.0)
        
        # Identify significant clusters
        significant_mask = p_corrected < alpha
        
        self.spm_results = {
            'time_normalized': time_points,
            't_statistics': t_stats,
            'p_values': p_values,
            'p_corrected': p_corrected,
            'correlations': correlations,
            'significant_mask': significant_mask,
            'alpha': alpha,
            'n_subjects': n_subjects
        }
        
        # Print summary
        sig_timepoints = np.sum(significant_mask)
        max_correlation = np.max(np.abs(correlations))
        max_corr_time = time_points[np.argmax(np.abs(correlations))]
        
        print(f"SPM Regression Results:")
        print(f"  Significant time points: {sig_timepoints}/{n_timepoints} ({sig_timepoints/n_timepoints*100:.1f}%)")
        print(f"  Maximum |correlation|: {max_correlation:.3f} at {max_corr_time:.1f}% of delivery")
        
        return self.spm_results

    def plot_spm_results(self, save_plot=False):
        """Create comprehensive SPM results visualization."""
        if self.spm_results is None:
            print("No SPM results to plot. Run SPM analysis first.")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        time_norm = self.spm_results['time_normalized']
        
        # Plot 1: Correlation time series
        correlations = self.spm_results['correlations']
        sig_mask = self.spm_results['significant_mask']
        
        axes[0].plot(time_norm, correlations, 'b-', linewidth=2, label='Correlation')
        axes[0].fill_between(time_norm, correlations, 0, where=sig_mask, 
                           alpha=0.3, color='red', label='Significant')
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Correlation (r)', fontsize=11)
        axes[0].set_title('SPM Regression: Shoulder Angle Y vs Elbow Energy Transfer', 
                         fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: t-statistics
        t_stats = self.spm_results['t_statistics']
        axes[1].plot(time_norm, t_stats, 'g-', linewidth=2)
        axes[1].fill_between(time_norm, t_stats, 0, where=sig_mask, 
                           alpha=0.3, color='red')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('t-statistic', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: p-values (log scale)
        p_corrected = self.spm_results['p_corrected']
        axes[2].semilogy(time_norm, p_corrected, 'r-', linewidth=2, label='Corrected p-values')
        axes[2].axhline(y=self.spm_results['alpha'], color='black', linestyle='--', 
                       label=f'α = {self.spm_results["alpha"]}')
        axes[2].set_xlabel('Delivery Phase (% from Foot Plant to Ball Release)', fontsize=11)
        axes[2].set_ylabel('p-value (log scale)', fontsize=11)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('spm_regression_results.png', dpi=300, bbox_inches='tight')
            print("SPM plot saved as 'spm_regression_results.png'")
        
        plt.show()

    def run_complete_analysis(self, max_pitches=5000, n_points=101, alpha=0.05, save_plot=False):
        """Run complete SPM regression pipeline."""
        print("=== SPM Regression Analysis: Shoulder Angle Y vs Elbow Energy Transfer ===")
        
        try:
            # Connect to database
            self.db = DatabaseConnection()
            if not self.db.test_connection():
                raise ConnectionError("Database connection failed")
            print("Database connection successful")
            
            # Load data
            self.load_energy_transfer_data()
            
            # Extract time series
            n_series = self.extract_time_series_data(max_pitches)
            if n_series < 50:
                raise ValueError(f"Insufficient time series data: only {n_series} valid series")
            
            # Normalize time series
            self.normalize_time_series(n_points)
            
            # Run SPM regression
            self.run_spm_regression(alpha)
            
            # Plot results
            self.plot_spm_results(save_plot)
            
            # Close database
            self.db.close()
            
            print("SPM regression analysis completed successfully!")
            
        except Exception as e:
            print(f"Error during SPM analysis: {str(e)}")
            if self.db:
                self.db.close()
            raise

def main():
    """Run SPM regression analysis."""
    spm = SPMRegression()
    spm.run_complete_analysis(
        max_pitches=5000,
        n_points=101,     # Standard time normalization
        alpha=0.05,       # Significance level
        save_plot=True    # Save the results plot
    )

if __name__ == "__main__":
    main()