#!/usr/bin/env python3
"""
Standalone Scatter Plot Analysis for Shoulder Angle Y vs Elbow Energy Transfer and Pitch Velocity

Usage:
    python scatter_analysis.py [--save-plots]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def create_shoulder_scatter_plots(save_plots=False):
    """Create scatter plots for shoulder angle Y relationships."""
    
    # Load the features data
    try:
        features_df = pd.read_csv('kinematic_features.csv')
        print(f"Loaded kinematic features: {features_df.shape}")
    except FileNotFoundError:
        print("Error: kinematic_features.csv not found. Run the main analysis first.")
        return
    
    # Define variables
    shoulder_y_br = 'shoulder_angle_y_ball_release'
    elbow_transfer = 'elbow_transfer_fp_br'
    pitch_velocity = 'pitch_speed_mph'
    
    # Check if variables exist
    missing_vars = [var for var in [shoulder_y_br, elbow_transfer, pitch_velocity] 
                   if var not in features_df.columns]
    if missing_vars:
        print(f"Error: Missing variables in dataset: {missing_vars}")
        return
    
    # Convert to numeric and clean data
    for col in [shoulder_y_br, elbow_transfer, pitch_velocity]:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Shoulder Angle Y vs Elbow Energy Transfer
    valid_data1 = features_df[[shoulder_y_br, elbow_transfer]].dropna()
    
    if len(valid_data1) > 0:
        ax1.scatter(valid_data1[shoulder_y_br], valid_data1[elbow_transfer], 
                   alpha=0.6, s=20, color='blue')
        
        # Add trend line
        z1 = np.polyfit(valid_data1[shoulder_y_br], valid_data1[elbow_transfer], 1)
        p1 = np.poly1d(z1)
        x_trend1 = np.linspace(valid_data1[shoulder_y_br].min(), 
                              valid_data1[shoulder_y_br].max(), 100)
        ax1.plot(x_trend1, p1(x_trend1), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr1, p_val1 = pearsonr(valid_data1[shoulder_y_br], valid_data1[elbow_transfer])
        
        # Add correlation text
        p_text1 = 'p < 0.001' if p_val1 < 0.001 else f'p = {p_val1:.3f}'
        ax1.text(0.05, 0.95, f'r = {corr1:.3f}\n{p_text1}\nn = {len(valid_data1):,}', 
                transform=ax1.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        ax1.set_xlabel('Shoulder Angle Y at Ball Release (degrees)', fontsize=12)
        ax1.set_ylabel('Elbow Energy Transfer (Joules)', fontsize=12)
        ax1.set_title('Shoulder Abduction vs Elbow Energy Transfer', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        print(f"Shoulder Y vs Elbow Energy Transfer:")
        print(f"  Correlation: r = {corr1:.3f}, p = {p_val1:.6f}")
        print(f"  Sample size: n = {len(valid_data1):,}")
    else:
        ax1.text(0.5, 0.5, 'No valid data available', transform=ax1.transAxes, 
                ha='center', va='center', fontsize=12)
        ax1.set_title('Shoulder Abduction vs Elbow Energy Transfer', fontsize=13, fontweight='bold')
    
    # Plot 2: Shoulder Angle Y vs Pitch Velocity
    valid_data2 = features_df[[shoulder_y_br, pitch_velocity]].dropna()
    
    if len(valid_data2) > 0:
        ax2.scatter(valid_data2[shoulder_y_br], valid_data2[pitch_velocity], 
                   alpha=0.6, s=20, color='green')
        
        # Add trend line
        z2 = np.polyfit(valid_data2[shoulder_y_br], valid_data2[pitch_velocity], 1)
        p2 = np.poly1d(z2)
        x_trend2 = np.linspace(valid_data2[shoulder_y_br].min(), 
                              valid_data2[shoulder_y_br].max(), 100)
        ax2.plot(x_trend2, p2(x_trend2), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr2, p_val2 = pearsonr(valid_data2[shoulder_y_br], valid_data2[pitch_velocity])
        
        # Add correlation text
        p_text2 = 'p < 0.001' if p_val2 < 0.001 else f'p = {p_val2:.3f}'
        ax2.text(0.05, 0.95, f'r = {corr2:.3f}\n{p_text2}\nn = {len(valid_data2):,}', 
                transform=ax2.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        ax2.set_xlabel('Shoulder Angle Y at Ball Release (degrees)', fontsize=12)
        ax2.set_ylabel('Pitch Velocity (mph)', fontsize=12)
        ax2.set_title('Shoulder Abduction vs Pitch Velocity', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        print(f"\nShoulder Y vs Pitch Velocity:")
        print(f"  Correlation: r = {corr2:.3f}, p = {p_val2:.6f}")
        print(f"  Sample size: n = {len(valid_data2):,}")
    else:
        ax2.text(0.5, 0.5, 'No valid data available', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Shoulder Abduction vs Pitch Velocity', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if requested
    if save_plots:
        plt.savefig('shoulder_angle_scatter_plots.png', dpi=300, bbox_inches='tight')
        print(f"\nPlots saved as 'shoulder_angle_scatter_plots.png'")
    
    plt.show()
    
    # Print summary statistics
    if len(valid_data1) > 0:
        print(f"\nShoulder Angle Y at Ball Release Summary Statistics:")
        print(f"  Mean: {valid_data1[shoulder_y_br].mean():.1f}°")
        print(f"  Median: {valid_data1[shoulder_y_br].median():.1f}°")
        print(f"  Std Dev: {valid_data1[shoulder_y_br].std():.1f}°")
        print(f"  Range: {valid_data1[shoulder_y_br].min():.1f}° to {valid_data1[shoulder_y_br].max():.1f}°")
        
        print(f"\nElbow Energy Transfer Summary Statistics:")
        print(f"  Mean: {valid_data1[elbow_transfer].mean():.1f} J")
        print(f"  Median: {valid_data1[elbow_transfer].median():.1f} J")
        print(f"  Std Dev: {valid_data1[elbow_transfer].std():.1f} J")
        print(f"  Range: {valid_data1[elbow_transfer].min():.1f} to {valid_data1[elbow_transfer].max():.1f} J")
        
        if len(valid_data2) > 0:
            print(f"\nPitch Velocity Summary Statistics:")
            print(f"  Mean: {valid_data2[pitch_velocity].mean():.1f} mph")
            print(f"  Median: {valid_data2[pitch_velocity].median():.1f} mph")
            print(f"  Std Dev: {valid_data2[pitch_velocity].std():.1f} mph")
            print(f"  Range: {valid_data2[pitch_velocity].min():.1f} to {valid_data2[pitch_velocity].max():.1f} mph")

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='Shoulder Angle Y Scatter Plot Analysis')
    parser.add_argument('--save-plots', action='store_true', 
                       help='Save scatter plots as PNG file')
    
    args = parser.parse_args()
    
    print("=== Shoulder Angle Y Scatter Plot Analysis ===")
    create_shoulder_scatter_plots(save_plots=args.save_plots)

if __name__ == "__main__":
    main()