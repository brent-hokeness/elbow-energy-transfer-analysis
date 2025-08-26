#!/usr/bin/env python3
"""
Query shoulder abduction and elbow varus moment data from database
and examine correlations at discrete time points.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from database.db_connection import DatabaseConnection

def query_shoulder_elbow_data():
    """Query shoulder abduction and elbow varus moment data from database."""
    print("Querying shoulder abduction and elbow varus moment data from database...")
    
    db = DatabaseConnection()
    
    if not db.test_connection():
        raise ConnectionError("Database connection failed!")
    
    query = """
    SELECT p.session_trial, p.pitch_speed_mph, p.elbow_varus_moment,
           e.FP_v6_time, e.MER_time, e.MIR_time, e.BR_time
    FROM poi p
    JOIN events e ON p.session_trial = e.session_trial
    WHERE p.elbow_varus_moment IS NOT NULL 
    AND p.pitch_speed_mph >= 85.0
    AND e.FP_v6_time IS NOT NULL 
    AND e.MER_time IS NOT NULL
    AND e.MIR_time IS NOT NULL
    AND e.BR_time IS NOT NULL
    ORDER BY p.session_trial
    """
    
    results = db.execute_query(query)
    
    if not results:
        db.close()
        raise ValueError("No elbow varus moment data found")
    
    df = pd.DataFrame(results)
    
    # Convert to numeric
    for col in ['pitch_speed_mph', 'elbow_varus_moment', 'FP_v6_time', 'MER_time', 'MIR_time', 'BR_time']:
        df[col] = pd.to_numeric(df[col])
    
    print(f"Retrieved elbow varus data for {len(df)} pitches over 85 mph")
    
    # Now query shoulder angle data for these specific trials
    session_trials = df['session_trial'].tolist()
    
    if len(session_trials) > 5000:
        print(f"Large dataset ({len(session_trials)} trials). Taking first 5000 for analysis.")
        session_trials = session_trials[:5000]
        df = df.head(5000).copy()
    
    shoulder_data = query_shoulder_angles(db, session_trials, df)
    
    db.close()
    
    return shoulder_data

def query_shoulder_angles(db, session_trials, base_df):
    """Query shoulder angle Y at specific time points for given trials."""
    print("Querying shoulder angle Y data at discrete time points...")
    
    shoulder_data = []
    
    for _, row in base_df.iterrows():
        session_trial = row['session_trial']
        
        # Get time points
        fp_time = float(row['FP_v6_time'])
        mer_time = float(row['MER_time'])  
        mir_time = float(row['MIR_time'])
        br_time = float(row['BR_time'])
        
        times = {
            'foot_plant': fp_time,
            'max_external_rotation': mer_time,
            'max_internal_rotation': mir_time, 
            'ball_release': br_time
        }
        
        # Initialize data for this trial
        trial_data = {
            'session_trial': session_trial,
            'elbow_varus_moment': row['elbow_varus_moment'],
            'pitch_speed_mph': row['pitch_speed_mph']
        }
        
        # Query shoulder angle at each time point
        for time_name, target_time in times.items():
            query = """
            SELECT shoulder_angle_y
            FROM joint_angles
            WHERE session_trial = %s
            AND ABS(time - %s) <= 0.02
            AND shoulder_angle_y IS NOT NULL
            ORDER BY ABS(time - %s)
            LIMIT 1
            """
            
            result = db.execute_query(query, [session_trial, target_time, target_time])
            
            if result and len(result) > 0:
                trial_data[f'shoulder_angle_y_{time_name}'] = float(result[0]['shoulder_angle_y'])
            else:
                trial_data[f'shoulder_angle_y_{time_name}'] = None
        
        shoulder_data.append(trial_data)
    
    # Convert to DataFrame
    df_shoulder = pd.DataFrame(shoulder_data)
    
    # Remove trials with missing shoulder data
    initial_count = len(df_shoulder)
    shoulder_cols = [col for col in df_shoulder.columns if 'shoulder_angle_y' in col]
    df_shoulder = df_shoulder.dropna(subset=shoulder_cols + ['elbow_varus_moment'])
    
    print(f"Complete data for {len(df_shoulder)}/{initial_count} trials")
    
    return df_shoulder

def analyze_shoulder_elbow_correlations(df):
    """Analyze correlations between shoulder abduction and elbow varus moment."""
    
    # Define variables with updated column names
    shoulder_vars = {
        'Foot Plant': 'shoulder_angle_y_foot_plant',
        'Max External Rotation': 'shoulder_angle_y_max_external_rotation',
        'Max Internal Rotation': 'shoulder_angle_y_max_internal_rotation', 
        'Ball Release': 'shoulder_angle_y_ball_release'
    }
    
    elbow_var = 'elbow_varus_moment'
    
    # Check if elbow varus moment exists
    if elbow_var not in df.columns:
        print(f"Error: {elbow_var} not found in dataset")
        print("Available columns containing 'elbow':")
        elbow_cols = [col for col in df.columns if 'elbow' in col.lower()]
        for col in elbow_cols:
            print(f"  {col}")
        return None
    
    # Convert to numeric
    for var in list(shoulder_vars.values()) + [elbow_var]:
        df[var] = pd.to_numeric(df[var], errors='coerce')
    
    print(f"\n=== Shoulder Abduction vs Elbow Varus Moment Correlations ===")
    print(f"Analyzing correlations at 4 discrete time points")
    
    results = []
    
    for time_point, shoulder_var in shoulder_vars.items():
        # Check if variables exist
        if shoulder_var not in df.columns:
            print(f"Warning: {shoulder_var} not found")
            continue
            
        # Get clean data
        clean_data = df[[shoulder_var, elbow_var]].dropna()
        
        if len(clean_data) < 10:
            print(f"Insufficient data for {time_point}: n = {len(clean_data)}")
            continue
        
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(clean_data[shoulder_var], clean_data[elbow_var])
        spearman_r, spearman_p = spearmanr(clean_data[shoulder_var], clean_data[elbow_var])
        
        # Store results
        results.append({
            'time_point': time_point,
            'variable': shoulder_var,
            'n_samples': len(clean_data),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'shoulder_mean': clean_data[shoulder_var].mean(),
            'shoulder_std': clean_data[shoulder_var].std(),
            'elbow_mean': clean_data[elbow_var].mean(),
            'elbow_std': clean_data[elbow_var].std()
        })
        
        # Print results
        print(f"\n{time_point}:")
        print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.6f})")
        print(f"  Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.6f})")
        print(f"  Sample size: n = {len(clean_data):,}")
        print(f"  Shoulder abduction: {clean_data[shoulder_var].mean():.1f} ± {clean_data[shoulder_var].std():.1f}°")
        print(f"  Elbow varus moment: {clean_data[elbow_var].mean():.1f} ± {clean_data[elbow_var].std():.1f} N⋅m")
    
    return results

def create_correlation_visualization(df, results):
    """Create visualization of correlations."""
    if not results:
        print("No results to visualize")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    shoulder_vars = {
        'Foot Plant': 'shoulder_angle_y_foot_plant',
        'Max External Rotation': 'shoulder_angle_y_max_external_rotation',
        'Max Internal Rotation': 'shoulder_angle_y_max_internal_rotation', 
        'Ball Release': 'shoulder_angle_y_ball_release'
    }
    
    elbow_var = 'elbow_varus_moment'
    
    for i, (time_point, shoulder_var) in enumerate(shoulder_vars.items()):
        if shoulder_var not in df.columns:
            continue
            
        # Get clean data
        clean_data = df[[shoulder_var, elbow_var]].dropna()
        
        if len(clean_data) < 10:
            continue
        
        # Find corresponding result
        result = next((r for r in results if r['time_point'] == time_point), None)
        if not result:
            continue
        
        # Create scatter plot
        axes[i].scatter(clean_data[shoulder_var], clean_data[elbow_var], 
                       alpha=0.5, s=15, color='blue')
        
        # Add trend line
        z = np.polyfit(clean_data[shoulder_var], clean_data[elbow_var], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(clean_data[shoulder_var].min(), clean_data[shoulder_var].max(), 100)
        axes[i].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        
        # Add correlation information
        r = result['pearson_r']
        p_val = result['pearson_p']
        n = result['n_samples']
        
        # Determine significance
        if p_val < 0.001:
            sig_text = 'p < 0.001 ***'
        elif p_val < 0.01:
            sig_text = f'p = {p_val:.3f} **'
        elif p_val < 0.05:
            sig_text = f'p = {p_val:.3f} *'
        else:
            sig_text = f'p = {p_val:.3f}'
        
        axes[i].text(0.05, 0.95, f'r = {r:.3f}\n{sig_text}\nn = {n:,}', 
                    transform=axes[i].transAxes, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[i].set_xlabel('Shoulder Angle Y (degrees)', fontsize=11)
        axes[i].set_ylabel('Elbow Varus Moment (N⋅m)', fontsize=11)
        axes[i].set_title(f'Shoulder Abduction vs Elbow Varus\nat {time_point}', 
                         fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_summary_comparison(results):
    """Create summary comparison of correlations across time points."""
    if not results:
        return
    
    print(f"\n=== Summary Comparison ===")
    print("Time Point".ljust(25) + "Pearson r".ljust(12) + "p-value".ljust(12) + "Sample Size")
    print("-" * 60)
    
    # Sort by absolute correlation strength
    sorted_results = sorted(results, key=lambda x: abs(x['pearson_r']), reverse=True)
    
    for result in sorted_results:
        time_point = result['time_point']
        r = result['pearson_r']
        p = result['pearson_p']
        n = result['n_samples']
        
        # Format p-value
        if p < 0.001:
            p_str = "< 0.001 ***"
        elif p < 0.01:
            p_str = f"{p:.3f} **"
        elif p < 0.05:
            p_str = f"{p:.3f} *"
        else:
            p_str = f"{p:.3f}"
        
        print(f"{time_point:<25}{r:>8.3f}{p_str:>12}{n:>12,}")
    
    # Find strongest correlation
    strongest = max(results, key=lambda x: abs(x['pearson_r']))
    print(f"\nStrongest correlation: {strongest['time_point']} (r = {strongest['pearson_r']:.3f})")
    
    # Check for significant correlations
    significant = [r for r in results if r['pearson_p'] < 0.05]
    if significant:
        print(f"Significant correlations: {len(significant)}/{len(results)} time points")
    else:
        print("No significant correlations found at α = 0.05")

def create_correlation_comparison_plot(results):
    """Create bar plot comparing correlation strengths."""
    if not results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data for plotting
    time_points = [r['time_point'] for r in results]
    pearson_rs = [r['pearson_r'] for r in results]
    p_values = [r['pearson_p'] for r in results]
    
    # Plot 1: Correlation coefficients
    colors = ['red' if p >= 0.05 else 'blue' for p in p_values]
    bars = ax1.bar(range(len(time_points)), pearson_rs, color=colors, alpha=0.7)
    
    ax1.set_xticks(range(len(time_points)))
    ax1.set_xticklabels([tp.replace(' ', '\n') for tp in time_points], fontsize=10)
    ax1.set_ylabel('Pearson Correlation Coefficient', fontsize=11)
    ax1.set_title('Shoulder Abduction vs Elbow Varus Moment\nCorrelations by Time Point', 
                  fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add correlation values on bars
    for bar, r in zip(bars, pearson_rs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + (0.005 if height >= 0 else -0.015), 
                f'{r:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    # Legend for colors
    ax1.text(0.02, 0.98, 'Blue = Significant (p<0.05)\nRed = Non-significant (p≥0.05)', 
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: P-values (log scale)
    ax2.bar(range(len(time_points)), p_values, color=colors, alpha=0.7)
    ax2.set_yscale('log')
    ax2.set_xticks(range(len(time_points)))
    ax2.set_xticklabels([tp.replace(' ', '\n') for tp in time_points], fontsize=10)
    ax2.set_ylabel('p-value (log scale)', fontsize=11)
    ax2.set_title('Statistical Significance\nof Correlations', fontsize=12, fontweight='bold')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run shoulder abduction vs elbow varus moment correlation analysis."""
    print("=== Shoulder Abduction vs Elbow Varus Moment Analysis ===")
    
    try:
        # Query data from database
        df = query_shoulder_elbow_data()
        
        if df is None or len(df) == 0:
            print("No data available for analysis")
            return
        
        # Analyze correlations
        results = analyze_shoulder_elbow_correlations(df)
        if not results:
            return
        
        # Create visualizations
        create_correlation_visualization(df, results)
        create_summary_comparison(results)
        create_correlation_comparison_plot(results)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()