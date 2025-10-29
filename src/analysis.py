# analysis.py

"""Script to aggregate results, perform stats tests, and generate final tables/plots."""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from typing import Dict, List, Tuple
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.config import METRICS_PATH, NUM_SPLITS, COVERAGE_LEVELS, STATION_COLUMNS, RESULTS_PATH
from src.visualization import (
    plot_coverage_calibration_point_cp,
    plot_coverage_calibration_interval_cqr,
    plot_winkler_vs_coverage,
    plot_conditional_coverage_comparison
)

def method_mapping() -> dict:
    """Generates a mapping from full method names to shorter, presentation-friendly names."""
    base_shortcuts = {
        'Naive': 'Naive', 'HCI': 'EPI', 'Lasso': 'LASSO', 'XGBoost': 'XGBoost',
        'LightGBM': 'LightGBM', 'CatBoost': 'CatBoost', 'MLP': 'DNN',
        'XGBQuantile': 'QR', 'QRF': 'QRF', 'QuantDNN': 'QuantDNN',
        'MCDropout': 'Dropout', 'GaussianDeepEnsembles': 'DE'
    }
    cp_methods = ['Naive', 'Lasso', 'XGBoost', 'LightGBM', 'CatBoost', 'MLP']
    cqr_methods = ['XGBQuantile', 'QRF', 'QuantDNN', 'MCDropout', 'GaussianDeepEnsembles']
    
    mapping = {**base_shortcuts} # Start with base methods
    
    for method in cp_methods:
        mapping[f"{method}+CP"] = f"C-{base_shortcuts.get(method, method)}"
        mapping[f"{method}+MondrianCP"] = f"M-{base_shortcuts.get(method, method)}"
        
    for method in cqr_methods:
        mapping[f"{method}+CQR"] = f"C-{base_shortcuts.get(method, method)}"
        mapping[f"{method}+MondrianCP"] = f"M-{base_shortcuts.get(method, method)}"

    # Add base HCI method if it exists
    if 'HCI' in base_shortcuts:
        mapping['HCI'] = 'EPI'
        
    return mapping

point_predictors_base = ['Naive', 'Lasso', 'XGBoost', 'LightGBM', 'CatBoost', 'MLP']
interval_predictors_base = ['HCI', 'XGBQuantile', 'QRF', 'QuantDNN', 'MCDropout', 'GaussianDeepEnsembles']

def load_all_results() -> pd.DataFrame:
    """Load all results from the metrics directory."""
    all_results = []
    for filename in os.listdir(METRICS_PATH):
        if filename.endswith('_metrics.csv') and filename.startswith('split_'): # Ensure we only load split results
            filepath = os.path.join(METRICS_PATH, filename)
            df = pd.read_csv(filepath)
            # Extract split number from filename
            try:
                split_num = int(filename.split('_')[1])
                df['split'] = split_num
                all_results.append(df)
            except (ValueError, IndexError):
                print(f"Warning: Could not parse split number from filename: {filename}")

    if not all_results:
        raise ValueError(f"No valid split result files found in {METRICS_PATH}. Ensure files are named 'split_<num>_metrics.csv'.")
    
    combined_df = pd.concat(all_results, ignore_index=True)

    # NAME STANDARDIZATION FIX
    print("Standardizing method names: replacing '+MondrianCQR' with '+MondrianCP'...")
    initial_methods = combined_df['method'].unique()
    combined_df['method'] = combined_df['method'].str.replace('+MondrianCQR', '+MondrianCP', regex=False)
    final_methods = combined_df['method'].unique()
    
    changed_methods = set(initial_methods) - set(final_methods)
    if changed_methods:
        print(f"  - Renamed methods containing '+MondrianCQR'. Found and corrected in: {changed_methods}")
    else:
        print("  - No inconsistent '+MondrianCQR' names found.")

    return combined_df

def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across splits."""
    results_df = results_df.copy()
    point_metrics = ['mae', 'rmse', 'r2']
    interval_metrics = ['coverage_rate', 'mean_width', 'winkler_score']
    group_metrics = [col for col in results_df.columns if col.startswith('coverage_group_')]
    available_metrics = [m for m in point_metrics + interval_metrics + group_metrics if m in results_df.columns]
    
    groupby_cols = ['method']
    if 'coverage_level' in results_df.columns:
        results_df['coverage_level'] = results_df['coverage_level'].fillna(-1)
        groupby_cols.append('coverage_level')

    agg_dict = {m: ['mean', 'std'] for m in available_metrics}
    if not agg_dict:
        return pd.DataFrame()
        
    summary = results_df.groupby(groupby_cols).agg(agg_dict).reset_index()
    summary.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in summary.columns]
    
    for metric in available_metrics:
        if f'{metric}_mean' in summary.columns:
            summary.rename(columns={f'{metric}_mean': metric, f'{metric}_std': f'{metric}_std'}, inplace=True)
            
    summary.loc[summary['coverage_level'] == -1, 'coverage_level'] = np.nan
    return summary

def _nan_result() -> Dict[str, float]:
    """Return a result dict with NaN values as Python floats."""
    return {'statistic': float('nan'), 'p_value': float('nan'), 'mean_diff': float('nan'), 'std_diff': float('nan')}


def perform_wilcoxon_tests(results_df: pd.DataFrame, 
                          method1: str, 
                          method2: str,
                          metric: str = 'coverage_rate',
                          coverage_level: float = 0.95) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test between two methods by merging results on 'split'.
    This approach is robust to cases where one method failed on a particular split.
    """
    # Base query for each method
    base_query = "method == '{}'"
    
    # Filter for method1
    df1_query = base_query.format(method1)
    if pd.notna(coverage_level):
        df1_query += f" and coverage_level == {coverage_level}"
    else:
        df1_query += " and coverage_level.isnull()"
    df1 = results_df.query(df1_query)[['split', metric]].dropna()

    # Filter for method2
    df2_query = base_query.format(method2)
    if pd.notna(coverage_level):
        df2_query += f" and coverage_level == {coverage_level}"
    else:
        df2_query += " and coverage_level.isnull()"
    df2 = results_df.query(df2_query)[['split', metric]].dropna()

    # Merge results on the 'split' column to get paired samples
    merged_data = pd.merge(df1, df2, on='split', suffixes=('_1', '_2'))

    if len(merged_data) < 2:
        # Not enough paired samples to perform the test
        return _nan_result()

    values1 = merged_data[f'{metric}_1'].values
    values2 = merged_data[f'{metric}_2'].values

    # The difference between paired samples
    diff = values1 - values2
    
    # Wilcoxon test requires non-zero differences
    if np.all(diff == 0):
        return {'statistic': 0.0, 'p_value': 1.0, 'mean_diff': 0.0, 'std_diff': 0.0}

    try:
        # Perform the test on non-zero differences
        statistic, p_value = wilcoxon(diff, alternative='two-sided')
    except ValueError:
        # This can happen if all differences are the same, etc.
        return _nan_result()
        
    return {'statistic': float(statistic), 'p_value': float(p_value),
            'mean_diff': float(np.mean(diff)), 'std_diff': float(np.std(diff))}




def analyze_conditional_coverage(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze conditional coverage for Mondrian methods."""
    group_cols = [col for col in results_df.columns if col.startswith('coverage_group_')]
    if not group_cols: return pd.DataFrame()
    
    cond_coverage = []
    relevant_methods = results_df[results_df.method.str.contains('CP|CQR', na=False)]['method'].unique()
    for conf_level in COVERAGE_LEVELS:
        level_df = results_df[results_df['coverage_level'] == conf_level]
        for method in relevant_methods:
            method_df = level_df[level_df['method'] == method].dropna(subset=group_cols, how='all')
            for group_col in group_cols:
                group_id = int(group_col.split('_')[-1])
                coverage_values = method_df[group_col].dropna()
                if len(coverage_values) > 0:
                    cond_coverage.append({'method': method, 'coverage_level': conf_level, 'group': group_id,
                                          'coverage_mean': coverage_values.mean(), 'coverage_std': coverage_values.std(),
                                          'n_samples': len(coverage_values)})
    return pd.DataFrame(cond_coverage)



def generate_statistical_ranking_report(results_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: str, method_map: Dict[str, str]):
    """
    Performs pairwise Wilcoxon tests on sorted models and generates a report.
    Uses method_map for readable names and fixes ranking index.
    """
    report_lines = ["=" * 80, "AUTOMATED STATISTICAL RANKING REPORT", "=" * 80, ""]
    
    # --- Test 1: Point Predictors by R2 Score ---
    report_lines.append("Point Predictor Ranking (by R2 Score, higher is better)")
    report_lines.append("-" * 60)
    
    point_summary = summary_df[summary_df['method'].isin(point_predictors_base)].dropna(subset=['r2'])
    sorted_point = point_summary.sort_values('r2', ascending=False).reset_index(drop=True)
    point_methods_sorted = sorted_point['method'].tolist()
    
    report_lines.append("Ranking:")
    for i, row in sorted_point.iterrows():
        method_name = method_map.get(row['method'], row['method'])
        report_lines.append(f"  {i+1}. {method_name} (R2: {row['r2']:.4f} ± {row['r2_std']:.4f})")
    report_lines.append("\nPairwise Wilcoxon Tests (p-value < 0.05 indicates significant difference):")
    
    for i in range(len(point_methods_sorted) - 1):
        method1 = point_methods_sorted[i]
        method2 = point_methods_sorted[i+1]
        
        test_result = perform_wilcoxon_tests(results_df, method1, method2, metric='r2', coverage_level=np.nan)
        p_value = test_result['p_value']
        significance = "Yes" if p_value < 0.05 else "No"
        
        method1_name = method_map.get(method1, method1)
        method2_name = method_map.get(method2, method2)
        report_lines.append(f"  - {method1_name} vs. {method2_name}: p-value = {p_value:.4f} (Significant: {significance})")

    report_lines.append("\n" * 2)

    # --- Test 2: Interval Predictors by Winkler Score at 90% CL ---
    report_lines.append("Interval Predictor Ranking (by Winkler Score at 90% CL, lower is better)")
    report_lines.append("-" * 70)
    
    interval_methods_all = [m for m in summary_df['method'].unique() 
                            if (not m in point_predictors_base) and ('Mondrian' not in m)]
    
    interval_summary_90 = summary_df[
        (summary_df['method'].isin(interval_methods_all)) &
        (summary_df['coverage_level'] == 0.9)
    ].dropna(subset=['winkler_score'])
    
    sorted_interval = interval_summary_90.sort_values('winkler_score', ascending=True).reset_index(drop=True)
    interval_methods_sorted = sorted_interval['method'].tolist()
    
    report_lines.append("Ranking (at 90% CL):")
    for i, row in sorted_interval.iterrows():
        method_name = method_map.get(row['method'], row['method'])
        report_lines.append(f"  {i+1}. {method_name} (Winkler: {row['winkler_score']:.2f} ± {row['winkler_score_std']:.2f})")
    report_lines.append("\nPairwise Wilcoxon Tests (p-value < 0.05 indicates significant difference):")

    for i in range(len(interval_methods_sorted) - 1):
        method1 = interval_methods_sorted[i]
        method2 = interval_methods_sorted[i+1]
        
        test_result = perform_wilcoxon_tests(results_df, method1, method2, metric='winkler_score', coverage_level=0.90)
        p_value = test_result['p_value']
        significance = "Yes" if p_value < 0.05 else "No"
        
        method1_name = method_map.get(method1, method1)
        method2_name = method_map.get(method2, method2)
        report_lines.append(f"  - {method1_name} vs. {method2_name}: p-value = {p_value:.4f} (Significant: {significance})")

    # Save and print report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("\n" + "\n".join(report_lines))
    print(f"\nStatistical ranking report saved to {output_path}")



def analyze_conformal_winkler_improvement(results_df: pd.DataFrame, method_map: Dict[str, str], output_path: str):
    """
    Performs a focused statistical analysis to test if conformal prediction significantly
    reduces the Winkler Score for interval prediction methods.

    - Compares each base interval model to its conformalized version (e.g., QR vs. C-QR).
    - Uses a one-sided, paired Wilcoxon signed-rank test.
    - Applies Holm-Bonferroni correction for multiple comparisons.
    - Excludes Mondrian, EPI, and pure point predictors.
    """
    print("\n" + "=" * 80)
    print("Analyzing Winkler Score Improvement from Conformal Prediction")
    print("=" * 80)
    print("Hypothesis: Conformalized version has a significantly LOWER Winkler Score.")
    
    # Define the pairs of (base_method, conformalized_method) to compare
    # These are the full names used in the results CSV files.
    model_pairs_full_names = [
        # CQR methods (base model is already an interval predictor)
        ('XGBQuantile', 'XGBQuantile+CQR'),
        ('QRF', 'QRF+CQR'),
        ('QuantDNN', 'QuantDNN+CQR'),
        ('MCDropout', 'MCDropout+CQR'),
        ('GaussianDeepEnsembles', 'GaussianDeepEnsembles+CQR'),
    ]

    analysis_records = []
    
    # Loop through each coverage level to perform tests
    for conf_level in COVERAGE_LEVELS:
        print(f"\n--- Testing for Coverage Level: {conf_level:.2f} ---")
        
        for base_method, cp_method in model_pairs_full_names:
            # Filter data for the current pair and coverage level
            base_scores_df = results_df[(results_df['method'] == base_method) & (results_df['coverage_level'] == conf_level)]
            cp_scores_df = results_df[(results_df['method'] == cp_method) & (results_df['coverage_level'] == conf_level)]

            # Merge to ensure we have paired data from the same splits
            merged_data = pd.merge(
                base_scores_df[['split', 'winkler_score']],
                cp_scores_df[['split', 'winkler_score']],
                on='split',
                suffixes=('_base', '_cp')
            )

            # Check if we have enough data to perform the test
            if len(merged_data) < 2:
                print(f"  - Skipping {method_map.get(base_method, base_method)} vs {method_map.get(cp_method, cp_method)}: Not enough paired data.")
                continue

            base_scores = merged_data['winkler_score_base'].values
            cp_scores = merged_data['winkler_score_cp'].values
            
            # The difference: base_score - cp_score. We expect this to be positive.
            diff = base_scores - cp_scores

            # Perform one-sided Wilcoxon signed-rank test
            # H0: The median of the differences is zero.
            # H1 (alternative='greater'): The median of the differences is greater than zero.
            # This tests if base_scores are systematically greater than cp_scores.
            try:
                if np.all(diff == 0):
                    p_value = 1.0
                else:
                    _, p_value = wilcoxon(diff, alternative='greater')
            except ValueError:
                # This can happen if all differences are identical and non-zero, etc.
                p_value = np.nan

            analysis_records.append({
                'coverage_level': conf_level,
                'base_model': method_map.get(base_method, base_method),
                'cp_model': method_map.get(cp_method, cp_method),
                'base_mean_winkler': np.mean(base_scores),
                'cp_mean_winkler': np.mean(cp_scores),
                'mean_improvement': np.mean(diff),
                'p_value_raw': p_value,
                'n_pairs': len(merged_data)
            })

    if not analysis_records:
        print("No valid pairs found for analysis. Check model names and result files.")
        return

    # Create a DataFrame from the collected records
    results_table = pd.DataFrame(analysis_records)
    
    # Apply Holm-Bonferroni correction across all tests performed
    raw_p_values = results_table['p_value_raw'].dropna()
    if len(raw_p_values) > 0:
        reject, p_values_corrected, _, _ = multipletests(raw_p_values, alpha=0.05, method='holm')
        results_table.loc[results_table['p_value_raw'].notna(), 'p_value_corrected'] = p_values_corrected
        results_table.loc[results_table['p_value_raw'].notna(), 'significant_holm'] = reject
    else:
        results_table['p_value_corrected'] = np.nan
        results_table['significant_holm'] = False

    # Format for printing and saving
    results_table['significant_holm'] = results_table['significant_holm'].map({True: 'Yes', False: 'No'})
    
    # Print results to console
    for conf_level in COVERAGE_LEVELS:
        print(f"\n--- Corrected Results for Coverage Level: {conf_level:.2f} ---")
        display_df = results_table[results_table['coverage_level'] == conf_level].copy()
        if display_df.empty:
            print("No results for this level.")
            continue
        
        display_df.drop(columns=['coverage_level'], inplace=True)
        print(display_df.to_string(index=False, float_format="%.4f"))

    # Save detailed results to CSV
    results_table.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\nDetailed Winkler score improvement analysis saved to: {output_path}")



def main():
    """Main analysis function."""
    print("Starting analysis...")
    
    method_map = method_mapping()
    tables_path = os.path.join(RESULTS_PATH, 'tables')
    plots_path = os.path.join(RESULTS_PATH, 'plots')
    os.makedirs(tables_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    
    print("Loading and aggregating results...")
    results_df = load_all_results()
    summary_df = aggregate_results(results_df)
    
    print("Generating summary tables...")

    summary_df_mapped = summary_df.copy()
    summary_df_mapped['method'] = summary_df_mapped['method'].map(method_map)
    summary_path = os.path.join(tables_path, 'summary_all_methods.csv')
    summary_df_mapped.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"Full summary table saved to {summary_path}")

    point_methods_full = [m for m in point_predictors_base if m in summary_df['method'].unique()]
    point_summary = summary_df[summary_df['method'].isin(point_methods_full)][['method', 'mae', 'mae_std', 'rmse', 'rmse_std', 'r2', 'r2_std']]
    point_summary['method'] = point_summary['method'].map(method_map)
    point_summary.to_csv(os.path.join(tables_path, 'summary_point_predictors.csv'), index=False, float_format='%.4f')
    print(f"Point predictors summary saved.")

    interval_methods_full = [m for m in summary_df['method'].unique() if (not m in point_predictors_base) and ('Mondrian' not in m)]
    interval_summary = summary_df[summary_df['method'].isin(interval_methods_full)].dropna(subset=['coverage_level'])
    interval_cols = ['method', 'coverage_level', 'coverage_rate', 'coverage_rate_std', 'mean_width', 'mean_width_std', 'winkler_score', 'winkler_score_std']
    interval_summary = interval_summary[interval_cols]
    interval_summary['method'] = interval_summary['method'].map(method_map)
    interval_summary.to_csv(os.path.join(tables_path, 'summary_interval_predictors.csv'), index=False, float_format='%.4f')
    print(f"Interval predictors summary saved.")

    cond_methods = [m for m in summary_df['method'].unique() if 'CP' in m or 'CQR' in m]
    cond_summary = summary_df[summary_df['method'].isin(cond_methods)].dropna(subset=['coverage_level'])
    group_cols = sorted([col for col in cond_summary.columns if col.startswith('coverage_group_')])
    cond_summary_cols = ['method', 'coverage_level'] + [col for col in group_cols if col in cond_summary.columns]
    cond_summary = cond_summary[cond_summary_cols]
    cond_summary['method'] = cond_summary['method'].map(method_map)
    cond_summary.to_csv(os.path.join(tables_path, 'summary_conditional_coverage.csv'), index=False, float_format='%.4f')
    print(f"Conditional coverage summary saved.")
    
    print("Generating plots...")
    
    point_cp_methods = [f"{m}+CP" for m in point_predictors_base]
    plot_coverage_calibration_point_cp(
        results_df,
        os.path.join(plots_path, 'coverage_calibration_point_cp.pdf'),
        methods=point_cp_methods,
        method_map=method_map
    )

    interval_cqr_methods = interval_predictors_base + [f"{m}+CQR" for m in interval_predictors_base]
    plot_coverage_calibration_interval_cqr(
        results_df,
        os.path.join(plots_path, 'coverage_calibration_interval_cqr.pdf'),
        methods=interval_cqr_methods,
        method_map=method_map
    )
    
    interval_methods_all = [m for m in results_df['method'].unique() if (not m in point_predictors_base) and ('Mondrian' not in m)]
    plot_winkler_vs_coverage(
        results_df[results_df['method'].isin(interval_methods_all)],
        os.path.join(plots_path, 'winkler_vs_coverage.pdf'),
        method_map=method_map
    )
    
    cond_cov_df = analyze_conditional_coverage(results_df)
    if not cond_cov_df.empty:
        # Include all conformal methods for this plot
        all_conformal_methods = [m for m in results_df['method'].unique() if 'CP' in m or 'CQR' in m]
        plot_data = cond_cov_df[cond_cov_df['method'].isin(all_conformal_methods)]
        if not plot_data.empty:
            plot_conditional_coverage_comparison(
                plot_data,
                os.path.join(plots_path, 'conditional_coverage_comparison.pdf'),
                group_names=['CHRX', 'LNDNBDE', 'SVNOAKS', 'TONBDG', 'WLOE'],
                coverage_levels=COVERAGE_LEVELS,
                method_map=method_map
            )

    # Generate and save the statistical ranking report
    print("\nGenerating statistical ranking report...")
    report_path = os.path.join(RESULTS_PATH, 'statistical_ranking_report.txt')
    generate_statistical_ranking_report(results_df, summary_df, report_path, method_map)

    # Perform the focused analysis on Winkler Score improvement from conformal prediction.
    winkler_analysis_path = os.path.join(tables_path, 'winkler_improvement_analysis.csv')
    analyze_conformal_winkler_improvement(results_df, method_map, winkler_analysis_path)

    print("\nAnalysis complete!")
    print(f"Tables, plots, and reports saved in: {RESULTS_PATH}")

if __name__ == "__main__":
    main()