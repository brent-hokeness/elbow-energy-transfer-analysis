"""
Consolidated Elbow Energy Transfer Kinematic Analysis

This script performs the complete analysis pipeline:
1. Extract kinematic features for all pitches over 85 mph
2. Calculate correlations with elbow energy transfer
3. Generate visualization of results
4. Save results to CSV (optional)

Usage:
    python kinematic_analysis.py [--skip-extraction] [--save-csv] [--top-n 20]
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from database.db_connection import DatabaseConnection
from analysis.data_extractor import KinematicExtractor

class KinematicAnalysis:
    """Complete kinematic analysis pipeline."""
    
    def __init__(self):
        self.features_df = None
        self.corr_df = None
        self.csv_filename = 'kinematic_features.csv'
    
    def extract_features(self, force_reextract=False):
        """Extract kinematic features from database or load from CSV."""
        if not force_reextract and os.path.exists(self.csv_filename):
            print(f"Loading existing features from {self.csv_filename}...")
            self.features_df = pd.read_csv(self.csv_filename)
            print(f"Loaded features: {self.features_df.shape}")
            return
        
        print("Extracting features from database...")
        db = DatabaseConnection()
        
        if not db.test_connection():
            raise ConnectionError("Database connection failed!")
        
        extractor = KinematicExtractor(db)
        self.features_df = extractor.extract_all_features()
        
        db.close()
        
        if self.features_df.empty:
            raise ValueError("No features extracted!")
        
        print(f"Extracted features: {self.features_df.shape}")
    
    def calculate_correlations(self):
        """Calculate correlations between kinematic variables and elbow energy transfer."""
        if self.features_df is None:
            raise ValueError("Features not loaded. Run extract_features() first.")
        
        print("Calculating correlations...")
        
        # Get kinematic columns
        exclude_cols = ['session_trial', 'elbow_transfer_fp_br', 'pitch_speed_mph']
        kinematic_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        # Convert to numeric
        for col in kinematic_cols + ['elbow_transfer_fp_br']:
            self.features_df[col] = pd.to_numeric(self.features_df[col], errors='coerce')
        
        # Calculate correlations
        correlations = []
        for col in kinematic_cols:
            valid_data = self.features_df[[col, 'elbow_transfer_fp_br']].dropna()
            
            if len(valid_data) > 10:
                try:
                    corr_coef, p_value = pearsonr(
                        valid_data[col].astype(float), 
                        valid_data['elbow_transfer_fp_br'].astype(float)
                    )
                    correlations.append({
                        'variable': col,
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'abs_correlation': abs(corr_coef),
                        'n_samples': len(valid_data)
                    })
                except Exception as e:
                    print(f"Error calculating correlation for {col}: {e}")
        
        self.corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
        print(f"Calculated correlations for {len(self.corr_df)} variables")
    
    def print_results(self, top_n=15):
        """Print top correlation results."""
        if self.corr_df is None:
            raise ValueError("Correlations not calculated. Run calculate_correlations() first.")
        
        print(f"\nTop {top_n} kinematic variables most correlated with elbow energy transfer:")
        print("=" * 80)
        
        for i, (_, row) in enumerate(self.corr_df.head(top_n).iterrows()):
            significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"{i+1:2d}. {row['variable']:<40} r={row['correlation']:6.3f} {significance} (n={row['n_samples']})")
        
        # Summary statistics
        significant = self.corr_df[self.corr_df['p_value'] < 0.001]
        strong = significant[significant['abs_correlation'] >= 0.3]
        moderate = significant[(significant['abs_correlation'] >= 0.1) & (significant['abs_correlation'] < 0.3)]
        
        print(f"\nSummary:")
        print(f"Total variables analyzed: {len(self.corr_df)}")
        print(f"Significant correlations (p<0.001): {len(significant)}")
        print(f"Strong correlations (|r|≥0.3, p<0.001): {len(strong)}")
        print(f"Moderate correlations (0.1≤|r|<0.3, p<0.001): {len(moderate)}")
    
    def create_visualization(self, top_n=20, save_plot=False):
        """Create correlation visualization."""
        if self.corr_df is None:
            raise ValueError("Correlations not calculated. Run calculate_correlations() first.")
        
        print(f"Creating visualization for top {top_n} variables...")
        
        top_vars = self.corr_df.head(top_n)
        
        # Color mapping function
        def get_color(corr, p_val):
            if p_val >= 0.001:
                return 'lightgray'
            elif abs(corr) >= 0.3:
                return 'darkred' if corr < 0 else 'darkblue'
            elif abs(corr) >= 0.1:
                return 'red' if corr < 0 else 'blue'
            else:
                return 'lightcoral' if corr < 0 else 'lightblue'
        
        # Prepare data for plotting (reversed for top-to-bottom display)
        correlations_reversed = top_vars['correlation'].tolist()[::-1]
        variables_reversed = top_vars['variable'].tolist()[::-1]
        colors = [get_color(corr, p_val) for corr, p_val in zip(top_vars['correlation'], top_vars['p_value'])]
        colors_reversed = colors[::-1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create shorter, cleaner variable names
        clean_labels = []
        for var in variables_reversed:
            # Simplify variable names for better readability
            parts = var.split('_')
            if len(parts) >= 3:
                joint = parts[0].title()
                angle = parts[1].title() 
                timepoint = ' '.join(parts[2:]).replace('_', ' ').title()
                clean_labels.append(f"{joint} {angle} @ {timepoint}")
            else:
                clean_labels.append(var.replace('_', ' ').title())
        
        bars = ax.barh(range(len(top_vars)), correlations_reversed, color=colors_reversed, height=0.6)
        
        # Customize plot
        ax.set_yticks(range(len(top_vars)))
        ax.set_yticklabels(clean_labels, fontsize=9)
        ax.set_xlabel('Correlation with Elbow Energy Transfer', fontsize=12)
        ax.set_title(f'Top {top_n} Kinematic Variables Related to Elbow Energy Transfer', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Customize plot
        ax.set_yticks(range(len(top_vars)))
        ax.set_yticklabels(clean_labels, fontsize=9)
        ax.set_xlabel('Correlation with Elbow Energy Transfer', fontsize=12)
        ax.set_title(f'Top {top_n} Kinematic Variables Related to Elbow Energy Transfer', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add correlation values and p-values on top of bars
        for i, row_idx in enumerate(reversed(range(len(top_vars)))):
            row = top_vars.iloc[row_idx]
            corr = row['correlation']
            p_val = row['p_value']
            
            # Position text in the center of each bar
            x_pos = corr / 2  # Middle of the bar
            
            p_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            
            # Use white text on dark bars, black text on light bars
            text_color = 'white' if abs(corr) >= 0.1 else 'black'
            
            ax.text(x_pos, i, f'r={corr:.3f}{p_text}', 
                   va='center', ha='center', fontsize=9, fontweight='bold',
                   color=text_color)
        
        # Legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, color='darkblue', label='Strong Positive (|r|≥0.3, p<0.001)'),
            plt.Rectangle((0,0),1,1, color='blue', label='Moderate Positive (0.1≤|r|<0.3, p<0.001)'),
            plt.Rectangle((0,0),1,1, color='lightblue', label='Weak Positive (|r|<0.1, p<0.001)'),
            plt.Rectangle((0,0),1,1, color='darkred', label='Strong Negative (|r|≥0.3, p<0.001)'),
            plt.Rectangle((0,0),1,1, color='red', label='Moderate Negative (0.1≤|r|<0.3, p<0.001)'),
            plt.Rectangle((0,0),1,1, color='lightcoral', label='Weak Negative (|r|<0.1, p<0.001)'),
            plt.Rectangle((0,0),1,1, color='lightgray', label='Not Significant (p≥0.001)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('kinematic_correlations.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'kinematic_correlations.png'")
        
        plt.show()
    
    def save_results(self, save_features=True, save_correlations=True):
        """Save results to CSV files."""
        if save_features and self.features_df is not None:
            self.features_df.to_csv(self.csv_filename, index=False)
            print(f"Features saved to '{self.csv_filename}'")
        
        if save_correlations and self.corr_df is not None:
            corr_filename = 'correlation_results.csv'
            self.corr_df.to_csv(corr_filename, index=False)
            print(f"Correlation results saved to '{corr_filename}'")
    
    def run_complete_analysis(self, force_reextract=False, save_csv=False, top_n=20, save_plot=False):
        """Run the complete analysis pipeline."""
        start_time = time.time()
        
        print("=== Elbow Energy Transfer Kinematic Analysis ===")
        
        try:
            # Extract features
            self.extract_features(force_reextract)
            
            # Calculate correlations
            self.calculate_correlations()
            
            # Print results
            self.print_results(top_n)
            
            # Create visualization
            self.create_visualization(top_n, save_plot)
            
            # Save results if requested
            if save_csv:
                self.save_results()
            
            elapsed_time = time.time() - start_time
            print(f"\nAnalysis completed in {elapsed_time:.1f} seconds!")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Elbow Energy Transfer Kinematic Analysis')
    parser.add_argument('--force-reextract', action='store_true', 
                       help='Force re-extraction from database even if CSV exists')
    parser.add_argument('--save-csv', action='store_true', 
                       help='Save results to CSV files')
    parser.add_argument('--save-plot', action='store_true', 
                       help='Save visualization plot as PNG')
    parser.add_argument('--top-n', type=int, default=20, 
                       help='Number of top variables to display (default: 20)')
    
    args = parser.parse_args()
    
    # Run analysis
    analysis = KinematicAnalysis()
    analysis.run_complete_analysis(
        force_reextract=args.force_reextract,
        save_csv=args.save_csv,
        top_n=args.top_n,
        save_plot=args.save_plot
    )

if __name__ == "__main__":
    main()