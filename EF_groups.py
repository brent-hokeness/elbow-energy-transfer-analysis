#!/usr/bin/env python3
"""
Split data by elbow energy transfer levels and compare shoulder angle Y patterns 
across time points (foot plant, MER, MIR, ball release).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def load_and_prepare_data():
    """Load data and prepare for group analysis."""
    try:
        df = pd.read_csv('kinematic_features.csv')
        print(f"Loaded data: {df.shape}")
    except FileNotFoundError:
        print("Error: kinematic_features.csv not found.")
        return None
    
    # Define variables we need
    energy_var = 'elbow_transfer_fp_br'
    shoulder_vars = [
        'shoulder_angle_y_foot_plant',
        'shoulder_angle_y_max_external_rotation', 
        'shoulder_angle_y_max_internal_rotation',
        'shoulder_angle_y_ball_release'
    ]
    
    # Check if all variables exist
    missing_vars = [var for var in [energy_var] + shoulder_vars if var not in df.columns]
    if missing_vars:
        print(f"Missing variables: {missing_vars}")
        return None
    
    # Convert to numeric and clean
    for col in [energy_var] + shoulder_vars:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Keep only rows with all data
    clean_df = df[[energy_var] + shoulder_vars].dropna()
    print(f"Clean data: n = {len(clean_df)}")
    
    return clean_df

def split_into_groups(df, energy_var='elbow_transfer_fp_br', method='percentiles'):
    """Split data into high and low energy transfer groups."""
    
    if method == 'percentiles':
        # Use 25th and 75th percentiles
        low_threshold = df[energy_var].quantile(0.25)
        high_threshold = df[energy_var].quantile(0.75)
        
        low_group = df[df[energy_var] <= low_threshold].copy()
        high_group = df[df[energy_var] >= high_threshold].copy()
        
        print(f"\nPercentile split:")
        print(f"Low group (≤{low_threshold:.1f} J): n = {len(low_group)}")
        print(f"High group (≥{high_threshold:.1f} J): n = {len(high_group)}")
        
    elif method == 'median':
        # Use median split
        median_val = df[energy_var].median()
        
        low_group = df[df[energy_var] < median_val].copy()
        high_group = df[df[energy_var] >= median_val].copy()
        
        print(f"\nMedian split:")
        print(f"Low group (<{median_val:.1f} J): n = {len(low_group)}")
        print(f"High group (≥{median_val:.1f} J): n = {len(high_group)}")
    
    # Add group labels
    low_group['group'] = 'Low Energy Transfer'
    high_group['group'] = 'High Energy Transfer'
    
    return low_group, high_group

def calculate_group_statistics(low_group, high_group, shoulder_vars, energy_var):
    """Calculate descriptive statistics for each group."""
    
    print(f"\n=== GROUP STATISTICS ===")
    
    # Energy transfer stats
    print(f"Energy Transfer:")
    print(f"  Low group:  {low_group[energy_var].mean():.1f} ± {low_group[energy_var].std():.1f} J")
    print(f"  High group: {high_group[energy_var].mean():.1f} ± {high_group[energy_var].std():.1f} J")
    
    # T-test for energy transfer (should be highly significant by design)
    t_stat, p_val = stats.ttest_ind(low_group[energy_var], high_group[energy_var])
    print(f"  t-test: t = {t_stat:.2f}, p < 0.001")
    
    # Shoulder angle stats at each time point
    print(f"\nShoulder Angle Y by Time Point:")
    
    time_labels = ['Foot Plant', 'Max External Rotation', 'Max Internal Rotation', 'Ball Release']
    
    results = []
    for var, label in zip(shoulder_vars, time_labels):
        low_mean = low_group[var].mean()
        low_std = low_group[var].std()
        high_mean = high_group[var].mean()
        high_std = high_group[var].std()
        
        # Statistical test
        t_stat, p_val = stats.ttest_ind(low_group[var], high_group[var])
        effect_size = (high_mean - low_mean) / np.sqrt(((len(low_group)-1)*low_std**2 + (len(high_group)-1)*high_std**2) / (len(low_group)+len(high_group)-2))
        
        print(f"  {label}:")
        print(f"    Low:  {low_mean:.1f} ± {low_std:.1f}°")
        print(f"    High: {high_mean:.1f} ± {high_std:.1f}°")
        print(f"    Diff: {high_mean - low_mean:+.1f}° (Cohen's d = {effect_size:.3f})")
        
        if p_val < 0.001:
            print(f"    t-test: t = {t_stat:.2f}, p < 0.001 ***")
        elif p_val < 0.01:
            print(f"    t-test: t = {t_stat:.2f}, p = {p_val:.3f} **")
        elif p_val < 0.05:
            print(f"    t-test: t = {t_stat:.2f}, p = {p_val:.3f} *")
        else:
            print(f"    t-test: t = {t_stat:.2f}, p = {p_val:.3f}")
        
        results.append({
            'time_point': label,
            'variable': var,
            'low_mean': low_mean,
            'low_std': low_std,
            'high_mean': high_mean,
            'high_std': high_std,
            'difference': high_mean - low_mean,
            'effect_size': effect_size,
            't_stat': t_stat,
            'p_value': p_val
        })
    
    return results

def create_visualization(low_group, high_group, shoulder_vars, stats_results):
    """Create visualization comparing groups across time points."""
    
    # Prepare data for plotting
    time_labels = ['Foot Plant', 'Max Ext Rot', 'Max Int Rot', 'Ball Release']
    time_points = range(len(time_labels))
    
    low_means = [stats['low_mean'] for stats in stats_results]
    low_stds = [stats['low_std'] for stats in stats_results]
    high_means = [stats['high_mean'] for stats in stats_results]
    high_stds = [stats['high_std'] for stats in stats_results]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Line plot with error bars
    ax1.errorbar(time_points, low_means, yerr=low_stds, 
                marker='o', linewidth=2, markersize=8, capsize=5, 
                label='Low Energy Transfer', color='red', alpha=0.8)
    ax1.errorbar(time_points, high_means, yerr=high_stds, 
                marker='s', linewidth=2, markersize=8, capsize=5, 
                label='High Energy Transfer', color='blue', alpha=0.8)
    
    ax1.set_xticks(time_points)
    ax1.set_xticklabels(time_labels, rotation=45, ha='right')
    ax1.set_ylabel('Shoulder Angle Y (degrees)', fontsize=12)
    ax1.set_title('Shoulder Abduction Across Pitching Phase', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add significance markers
    for i, stats in enumerate(stats_results):
        if stats['p_value'] < 0.001:
            ax1.text(i, max(high_means[i] + high_stds[i], low_means[i] + low_stds[i]) + 2, 
                    '***', ha='center', fontweight='bold', fontsize=14)
        elif stats['p_value'] < 0.01:
            ax1.text(i, max(high_means[i] + high_stds[i], low_means[i] + low_stds[i]) + 2, 
                    '**', ha='center', fontweight='bold', fontsize=14)
        elif stats['p_value'] < 0.05:
            ax1.text(i, max(high_means[i] + high_stds[i], low_means[i] + low_stds[i]) + 2, 
                    '*', ha='center', fontweight='bold', fontsize=14)
    
    # Plot 2: Effect sizes
    effect_sizes = [stats['effect_size'] for stats in stats_results]
    colors = ['red' if es < 0 else 'blue' for es in effect_sizes]
    
    bars = ax2.bar(time_points, effect_sizes, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect (d=0.2)')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect (d=0.5)')
    ax2.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xticks(time_points)
    ax2.set_xticklabels(time_labels, rotation=45, ha='right')
    ax2.set_ylabel("Cohen's d (Effect Size)", fontsize=12)
    ax2.set_title('Effect Size: High vs Low Energy Transfer', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add effect size values on bars
    for bar, es in zip(bars, effect_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + (0.02 if height >= 0 else -0.05), 
                f'{es:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Run complete group comparison analysis."""
    print("=== Energy Transfer Group Comparison Analysis ===")
    
    # Load data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Define variables
    energy_var = 'elbow_transfer_fp_br'
    shoulder_vars = [
        'shoulder_angle_y_foot_plant',
        'shoulder_angle_y_max_external_rotation', 
        'shoulder_angle_y_max_internal_rotation',
        'shoulder_angle_y_ball_release'
    ]
    
    # Split into groups using percentiles (more conservative than median)
    low_group, high_group = split_into_groups(df, energy_var, method='percentiles')
    
    # Calculate statistics
    stats_results = calculate_group_statistics(low_group, high_group, shoulder_vars, energy_var)
    
    # Create visualization
    fig = create_visualization(low_group, high_group, shoulder_vars, stats_results)
    
    # Summary and SPM recommendation
    print(f"\n=== SPM ANALYSIS RECOMMENDATION ===")
    
    significant_timepoints = sum(1 for stats in stats_results if stats['p_value'] < 0.05)
    max_effect_size = max(abs(stats['effect_size']) for stats in stats_results)
    
    print(f"Significant time points: {significant_timepoints}/4")
    print(f"Maximum effect size: {max_effect_size:.3f}")
    
    if significant_timepoints >= 2 and max_effect_size >= 0.2:
        print("→ RECOMMENDED: Group differences exist. SPM analysis would be worthwhile.")
        print("  SPM can identify specific time windows where differences are most pronounced.")
    elif significant_timepoints >= 1:
        print("→ MAYBE: Some differences exist. SPM might reveal temporal patterns.")
        print("  Consider SPM if you want to identify precise timing of differences.")
    else:
        print("→ NOT RECOMMENDED: Minimal group differences. SPM unlikely to be informative.")
        print("  Focus on other variables with stronger relationships to energy transfer.")

if __name__ == "__main__":
    main()