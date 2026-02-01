#!/usr/bin/env python3
"""
Script to compare metrics between boltz_results_70W5 and boltz_results_70W5_with_MSA
to analyze the impact of using MSA on model performance.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def load_json_file(file_path):
    """Load JSON file and return the data."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_percentage_difference(value1, value2):
    """Calculate percentage difference between two values."""
    if value1 == 0:
        return float('inf') if value2 != 0 else 0
    return ((value2 - value1) / value1) * 100

def compare_metrics(metrics1, metrics2, metric_name=""):
    """Compare metrics and return percentage differences."""
    differences = {}
    
    # Handle nested dictionaries (like chains_ptm and pair_chains_iptm)
    if isinstance(metrics1, dict) and isinstance(metrics2, dict):
        for key in metrics1:
            if key in metrics2:
                if isinstance(metrics1[key], dict):
                    # Recursively handle nested dictionaries
                    nested_diff = compare_metrics(metrics1[key], metrics2[key], f"{metric_name}.{key}" if metric_name else key)
                    differences.update(nested_diff)
                else:
                    # Calculate percentage difference for numeric values
                    diff = calculate_percentage_difference(metrics1[key], metrics2[key])
                    differences[f"{metric_name}.{key}" if metric_name else key] = diff
    else:
        # Handle simple numeric values
        diff = calculate_percentage_difference(metrics1, metrics2)
        differences[metric_name if metric_name else "value"] = diff
    
    return differences

def main():
    # Define file paths
    base_dir = Path(__file__).resolve().parents[2] / "outputs" / "boltz_out"
    
    file1_path = base_dir / "boltz_results_70W5" / "predictions" / "70W5" / "confidence_70W5_model_0.json"
    file2_path = base_dir / "boltz_results_70W5_with_MSA" / "predictions" / "70W5" / "confidence_70W5_model_0_with_MSA.json"
    
    # Load the JSON files
    print("Loading JSON files...")
    data1 = load_json_file(file1_path)
    data2 = load_json_file(file2_path)
    
    print(f"File 1 (without MSA): {file1_path}")
    print(f"File 2 (with MSA): {file2_path}")
    print()
    
    # Compare all metrics
    print("Calculating percentage differences...")
    differences = compare_metrics(data1, data2)
    
    # Create a DataFrame for better visualization
    df = pd.DataFrame([
        {
            'Metric': metric,
            'Without_MSA': data1.get(metric.split('.')[0], 'N/A'),
            'With_MSA': data2.get(metric.split('.')[0], 'N/A'),
            'Percentage_Difference': diff
        }
        for metric, diff in differences.items()
        if '.' not in metric  # Only top-level metrics for this comparison
    ])
    
    # Add nested metrics separately
    nested_metrics = []
    for metric, diff in differences.items():
        if '.' in metric:
            parts = metric.split('.')
            if len(parts) == 2:  # chains_ptm.0, chains_ptm.1, etc.
                parent_key, child_key = parts
                try:
                    without_msa_val = data1[parent_key].get(child_key, 'N/A')
                    with_msa_val = data2[parent_key].get(child_key, 'N/A')
                    nested_metrics.append({
                        'Metric': metric,
                        'Parent_Metric': parent_key,
                        'Child_Key': child_key,
                        'Without_MSA': without_msa_val,
                        'With_MSA': with_msa_val,
                        'Percentage_Difference': diff
                    })
                except (KeyError, TypeError):
                    continue
            elif len(parts) == 3:  # pair_chains_iptm.0.0, pair_chains_iptm.0.1, etc.
                parent_key, child_key1, child_key2 = parts
                try:
                    without_msa_val = data1[parent_key][child_key1].get(child_key2, 'N/A')
                    with_msa_val = data2[parent_key][child_key1].get(child_key2, 'N/A')
                    nested_metrics.append({
                        'Metric': metric,
                        'Parent_Metric': parent_key,
                        'Child_Key': f"{child_key1}.{child_key2}",
                        'Without_MSA': without_msa_val,
                        'With_MSA': with_msa_val,
                        'Percentage_Difference': diff
                    })
                except (KeyError, TypeError):
                    continue
    
    # Debug: Print the differences to see what's being captured
    print(f"\nDebug: Total differences found: {len(differences)}")
    print(f"Debug: Differences with dots: {[k for k in differences.keys() if '.' in k]}")
    print(f"Debug: Differences without dots: {[k for k in differences.keys() if '.' not in k]}")
    
    nested_df = pd.DataFrame(nested_metrics)
    
    # Debug information
    print(f"\nDebug: nested_df shape: {nested_df.shape}")
    if not nested_df.empty:
        print(f"Debug: nested_df columns: {list(nested_df.columns)}")
        print(f"Debug: First few rows:")
        print(nested_df.head())
    
    # Display results
    print("\n" + "="*80)
    print("IMPACT OF MSA ON MODEL METRICS")
    print("="*80)
    
    print("\nTOP-LEVEL METRICS:")
    print("-" * 50)
    for _, row in df.iterrows():
        print(f"{row['Metric']:25} | {row['Percentage_Difference']:8.2f}%")
    
    print("\nCHAIN-SPECIFIC METRICS (chains_ptm):")
    print("-" * 50)
    if not nested_df.empty and 'Parent_Metric' in nested_df.columns:
        chain_metrics = nested_df[nested_df['Parent_Metric'] == 'chains_ptm']
        for _, row in chain_metrics.iterrows():
            print(f"Chain {row['Child_Key']:15} | {row['Percentage_Difference']:8.2f}%")
    else:
        print("No chain-specific metrics found")
    
    print("\nPAIR-CHAIN METRICS (pair_chains_iptm):")
    print("-" * 50)
    if not nested_df.empty and 'Parent_Metric' in nested_df.columns:
        pair_metrics = nested_df[nested_df['Parent_Metric'] == 'pair_chains_iptm']
        for _, row in pair_metrics.iterrows():
            print(f"Pair {row['Child_Key']:15} | {row['Percentage_Difference']:8.2f}%")
    else:
        print("No pair-chain metrics found")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Top-level metrics summary
    top_level_diffs = df['Percentage_Difference'].values
    print(f"Top-level metrics:")
    print(f"  Mean change: {np.mean(top_level_diffs):.2f}%")
    print(f"  Median change: {np.median(top_level_diffs):.2f}%")
    print(f"  Max improvement: {np.max(top_level_diffs):.2f}%")
    print(f"  Max decline: {np.min(top_level_diffs):.2f}%")
    
    # Overall summary
    all_diffs = [diff for diff in differences.values() if not np.isinf(diff)]
    print(f"\nAll metrics:")
    print(f"  Mean change: {np.mean(all_diffs):.2f}%")
    print(f"  Median change: {np.median(all_diffs):.2f}%")
    print(f"  Max improvement: {np.max(all_diffs):.2f}%")
    print(f"  Max decline: {np.min(all_diffs):.2f}%")
    
    # Save results to CSV
    output_dir = base_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / "msa_impact_top_level.csv", index=False)
    nested_df.to_csv(output_dir / "msa_impact_detailed.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"  {output_dir / 'msa_impact_top_level.csv'}")
    print(f"  {output_dir / 'msa_impact_detailed.csv'}")

if __name__ == "__main__":
    main() 