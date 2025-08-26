#!/usr/bin/env python3
"""
SPM Group Comparison Analysis
Plot average shoulder angle Y time series for high vs low elbow energy transfer groups
from foot plant to ball release using continuous time-series data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from database.db_connection import DatabaseConnection
import warnings
warnings.filterwarnings('ignore')

class SPMGroupComparison:
    def __init__(self):
        self.db = None
        self.energy_data = None
        self.time_series_data = {}
        self.high_group_data = None
        self.low_group_data = None
        self.group_averages = None
        
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

    def create_energy_transfer_groups(self, method='percentiles'):
        """Split data into high and low energy transfer groups."""
        if method == 'percentiles':
            # Use 25th and 75th percentiles for clearer separation
            low_threshold = self.energy_data['elbow_transfer_fp_br'].quantile(0.25)
            high_threshold = self.energy_data['elbow_transfer_fp_br'].quantile(0.75)
            
            low_group = self.energy_data[self.energy_data['elbow_transfer_fp_br'] <= low_threshold].copy()
            high_group = self.energy_data[self.energy_data['elbow_transfer_fp_br'] >= high_threshold].copy()
            
            print(f"\nPercentile split:")
            print(f"Low group (≤{low_threshold:.1f} J): n = {len(low_group)}")
            print(f"High group (≥{high_threshold:.1f} J): n = {len(high_group)}")
        
        elif method == 'tertiles':
            # Use top and bottom tertiles
            low_threshold = self.energy_data['elbow_transfer_fp_br'].quantile(0.33)
            high_threshold = self.energy_data['elbow_transfer_fp_br'].quantile(0.67)
            
            low_group = self.energy_data[self.energy_data['elbow_transfer_fp_br'] <= low_threshold].copy()
            high_group = self.energy_data[self.energy_data['elbow_transfer_fp_br'] >= high_threshold].copy()
            
            print(f"\nTertile split:")
            print(f"Low group (≤{low_threshold:.1f} J): n = {len(low_group)}")
            print(f"High group (≥{high_threshold:.1f} J): n = {len(high_group)}")
        
        # Store group information
        self.low_group_data = {
            'trials': low_group['session_trial'].tolist(),
            'energy_values': low_group['elbow_transfer_fp_br'].values,
            'mean_energy': low_group['elbow_transfer_fp_br'].mean(),
            'std_energy': low_group['elbow_transfer_fp_br'].std()
        }
        
        self.high_group_data = {
            'trials': high_group['session_trial'].tolist(),
            'energy_values': high_group['elbow_transfer_fp_br'].values,
            'mean_energy': high_group['elbow_transfer_fp_br'].mean(),
            'std_energy': high_group['elbow_transfer_fp_br'].std()
        }
        
        return low_group, high_group

    def extract_group_time_series(self, max_per_group=500):
        """Extract time series for both groups."""
        print(f"Extracting time series for both groups (max {max_per_group} per group)...")
        
        groups = {
            'low': {
                'trials': self.low_group_data['trials'][:max_per_group],
                'series': []
            },
            'high': {
                'trials': self.high_group_data['trials'][:max_per_group],
                'series': []
            }
        }
        
        # Get corresponding energy data for the groups
        energy_lookup = dict(zip(self.energy_data['session_trial'], 
                                zip(self.energy_data['FP_v6_time'], self.energy_data['BR_time'])))
        
        for group_name, group_info in groups.items():
            print(f"  Processing {group_name} group...")
            
            for i, session_trial in enumerate(group_info['trials']):
                if session_trial not in energy_lookup:
                    continue
                    
                fp_time, br_time = energy_lookup[session_trial]
                fp_time, br_time = float(fp_time), float(br_time)
                
                # Query time series data
                query = """
                SELECT time, shoulder_angle_y
                FROM joint_angles 
                WHERE session_trial = %s 
                AND time BETWEEN %s AND %s
                AND shoulder_angle_y IS NOT NULL
                ORDER BY time
                """
                
                time_series = self.db.execute_query(query, [session_trial, fp_time, br_time])
                
                if time_series and len(time_series) >= 10:
                    ts_df = pd.DataFrame(time_series)
                    ts_df['time'] = pd.to_numeric(ts_df['time'])
                    ts_df['shoulder_angle_y'] = pd.to_numeric(ts_df['shoulder_angle_y'])
                    
                    group_info['series'].append({
                        'session_trial': session_trial,
                        'time': ts_df['time'].values,
                        'shoulder_angle_y': ts_df['shoulder_angle_y'].values,
                        'duration': br_time - fp_time
                    })
                
                if (i + 1) % 50 == 0:
                    print(f"    Processed {i + 1}/{len(group_info['trials'])} trials...")
            
            print(f"  {group_name} group: {len(group_info['series'])} valid time series")
        
        return groups

    def normalize_and_average_groups(self, groups, n_points=101):
        """Time-normalize and calculate group averages."""
        print(f"Time-normalizing and calculating group averages...")
        
        time_normalized = np.linspace(0, 100, n_points)
        group_averages = {}
        
        for group_name, group_info in groups.items():
            normalized_series = []
            
            for series in group_info['series']:
                time_original = series['time']
                shoulder_original = series['shoulder_angle_y']
                
                # Convert to percentage of delivery phase
                time_percent = (time_original - time_original[0]) / (time_original[-1] - time_original[0]) * 100
                
                # Interpolate to normalized time points
                try:
                    f_interp = interpolate.interp1d(time_percent, shoulder_original, 
                                                  kind='linear', fill_value='extrapolate')
                    shoulder_normalized = f_interp(time_normalized)
                    normalized_series.append(shoulder_normalized)
                except Exception:
                    continue
            
            if len(normalized_series) > 0:
                normalized_array = np.array(normalized_series)
                
                group_averages[group_name] = {
                    'mean': np.mean(normalized_array, axis=0),
                    'std': np.std(normalized_array, axis=0),
                    'sem': np.std(normalized_array, axis=0) / np.sqrt(len(normalized_series)),
                    'n_subjects': len(normalized_series),
                    'individual_series': normalized_array
                }
                
                print(f"  {group_name} group: {len(normalized_series)} normalized time series")
            else:
                print(f"  Warning: No valid time series for {group_name} group")
        
        self.group_averages = {
            'time_normalized': time_normalized,
            'groups': group_averages
        }
        
        return self.group_averages

    def perform_statistical_comparison(self):
        """Perform point-wise t-tests between groups."""
        print("Performing statistical comparison between groups...")
        
        if not self.group_averages or len(self.group_averages['groups']) != 2:
            print("Need exactly 2 groups for comparison")
            return None
        
        low_data = self.group_averages['groups']['low']['individual_series']
        high_data = self.group_averages['groups']['high']['individual_series']
        n_points = len(self.group_averages['time_normalized'])
        
        t_stats = np.zeros(n_points)
        p_values = np.zeros(n_points)
        effect_sizes = np.zeros(n_points)
        
        for t in range(n_points):
            low_t = low_data[:, t]
            high_t = high_data[:, t]
            
            # Remove NaN values
            low_valid = low_t[~np.isnan(low_t)]
            high_valid = high_t[~np.isnan(high_t)]
            
            if len(low_valid) > 5 and len(high_valid) > 5:
                t_stat, p_val = stats.ttest_ind(high_valid, low_valid)
                
                # Calculate Cohen's d
                pooled_std = np.sqrt(((len(low_valid)-1)*np.var(low_valid, ddof=1) + 
                                    (len(high_valid)-1)*np.var(high_valid, ddof=1)) / 
                                   (len(low_valid) + len(high_valid) - 2))
                
                effect_size = (np.mean(high_valid) - np.mean(low_valid)) / pooled_std if pooled_std > 0 else 0
                
                t_stats[t] = t_stat
                p_values[t] = p_val
                effect_sizes[t] = effect_size
            else:
                t_stats[t] = 0
                p_values[t] = 1.0
                effect_sizes[t] = 0
        
        # Apply Bonferroni correction
        p_corrected = np.minimum(p_values * n_points, 1.0)
        significant_mask = p_corrected < 0.05
        
        stats_results = {
            't_statistics': t_stats,
            'p_values': p_values,
            'p_corrected': p_corrected,
            'effect_sizes': effect_sizes,
            'significant_mask': significant_mask,
            'n_significant': np.sum(significant_mask)
        }
        
        print(f"Statistical comparison complete:")
        print(f"  Significant time points: {np.sum(significant_mask)}/{n_points} ({np.sum(significant_mask)/n_points*100:.1f}%)")
        
        return stats_results

    def plot_group_comparison(self, stats_results=None, save_plot=False):
        """Create comprehensive group comparison visualization."""
        if not self.group_averages:
            print("No group averages to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        time_norm = self.group_averages['time_normalized']
        groups = self.group_averages['groups']
        
        # Plot 1: Group averages with confidence intervals
        colors = {'low': 'red', 'high': 'blue'}
        labels = {
            'low': f"Low Energy Transfer (n={groups['low']['n_subjects']})", 
            'high': f"High Energy Transfer (n={groups['high']['n_subjects']})"
        }
        
        for group_name, group_data in groups.items():
            mean_vals = group_data['mean']
            sem_vals = group_data['sem']
            color = colors[group_name]
            
            axes[0].plot(time_norm, mean_vals, color=color, linewidth=3, 
                        label=labels[group_name])
            axes[0].fill_between(time_norm, mean_vals - sem_vals, mean_vals + sem_vals, 
                               color=color, alpha=0.2)
        
        # Add significance markers if available
        if stats_results is not None:
            sig_mask = stats_results['significant_mask']
            if np.any(sig_mask):
                y_max = max([np.max(groups[g]['mean'] + groups[g]['sem']) for g in groups])
                sig_y = y_max + 2
                
                # Mark significant regions
                axes[0].fill_between(time_norm, sig_y - 0.5, sig_y + 0.5, where=sig_mask, 
                                   alpha=0.3, color='black', label='Significant difference')
        
        axes[0].set_ylabel('Shoulder Angle Y (degrees)', fontsize=12)
        axes[0].set_title('Group Comparison: Shoulder Abduction During Delivery Phase', 
                         fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Effect sizes and statistical results
        if stats_results is not None:
            effect_sizes = stats_results['effect_sizes']
            sig_mask = stats_results['significant_mask']
            
            # Plot effect sizes
            axes[1].plot(time_norm, effect_sizes, 'purple', linewidth=2, label='Effect Size (Cohen\'s d)')
            axes[1].fill_between(time_norm, effect_sizes, 0, where=sig_mask, 
                               alpha=0.3, color='red', label='Significant')
            
            # Add reference lines for effect size interpretation
            axes[1].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect (d=0.2)')
            axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect (d=0.5)')
            axes[1].axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
            axes[1].axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            axes[1].set_ylabel('Effect Size (Cohen\'s d)', fontsize=12)
            axes[1].set_title('Statistical Comparison: High vs Low Energy Transfer', fontsize=12, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No statistical comparison performed', 
                        transform=axes[1].transAxes, ha='center', va='center', fontsize=14)
            axes[1].set_title('Statistical Comparison', fontsize=12, fontweight='bold')
        
        axes[1].set_xlabel('Delivery Phase (% from Foot Plant to Ball Release)', fontsize=12)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('spm_group_comparison.png', dpi=300, bbox_inches='tight')
            print("Group comparison plot saved as 'spm_group_comparison.png'")
        
        plt.show()

    def run_complete_analysis(self, max_per_group=500, split_method='percentiles', n_points=101, save_plot=False):
        """Run complete group comparison analysis."""
        print("=== SPM Group Comparison: High vs Low Energy Transfer ===")
        
        try:
            # Connect to database
            self.db = DatabaseConnection()
            if not self.db.test_connection():
                raise ConnectionError("Database connection failed")
            print("Database connection successful")
            
            # Load energy transfer data
            self.load_energy_transfer_data()
            
            # Create groups
            self.create_energy_transfer_groups(method=split_method)
            
            # Extract time series for both groups
            groups = self.extract_group_time_series(max_per_group)
            
            # Normalize and calculate averages
            self.normalize_and_average_groups(groups, n_points)
            
            # Perform statistical comparison
            stats_results = self.perform_statistical_comparison()
            
            # Plot results
            self.plot_group_comparison(stats_results, save_plot)
            
            # Print summary
            print(f"\nSummary:")
            print(f"Low energy transfer group: {self.low_group_data['mean_energy']:.1f} ± {self.low_group_data['std_energy']:.1f} J")
            print(f"High energy transfer group: {self.high_group_data['mean_energy']:.1f} ± {self.high_group_data['std_energy']:.1f} J")
            
            # Close database
            self.db.close()
            
            print("Group comparison analysis completed successfully!")
            
        except Exception as e:
            print(f"Error during group comparison analysis: {str(e)}")
            if self.db:
                self.db.close()
            raise

def main():
    """Run SPM group comparison analysis."""
    spm = SPMGroupComparison()
    spm.run_complete_analysis(
        max_per_group=1000,     # Number of pitches per group
        split_method='percentiles',  # 'percentiles' or 'tertiles'
        n_points=101,           # Time normalization points
        save_plot=True          # Save the results plot
    )

if __name__ == "__main__":
    main()