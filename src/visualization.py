# visualization.py

"""Functions for plotting results."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
from src.config import COVERAGE_LEVELS

warnings.filterwarnings("ignore", category=UserWarning)

def set_plot_style():
    """Set consistent plot style for all visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['figure.autolayout'] = True

def plot_coverage_calibration_point_cp(results_df: pd.DataFrame, 
                                       output_path: str,
                                       methods: List[str],
                                       method_map: Optional[Dict[str, str]] = None) -> None:
    """Plot coverage calibration for point predictors with +CP."""
    set_plot_style()
    
    plot_data = results_df[results_df['method'].isin(methods)].copy()
    if plot_data.empty:
        print(f"Warning: No data for plot_coverage_calibration_point_cp.")
        return
        
    if method_map:
        plot_data['method'] = plot_data['method'].map(method_map).fillna(plot_data['method'])

    plt.figure(figsize=(8, 5))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=4, label='Perfect Calibration', alpha=0.7)
    
    unique_methods = sorted(plot_data['method'].unique())
    palette = sns.color_palette("husl", len(unique_methods))

    for i, method in enumerate(unique_methods):
        method_data = plot_data[plot_data['method'] == method]
        agg_data = method_data.groupby('coverage_level')['coverage_rate'].mean().reset_index()
        plt.plot(agg_data['coverage_level'], agg_data['coverage_rate'],
                 marker='o', markersize=12, linestyle='-', linewidth=3, alpha=0.9,
                 color=palette[i], label=method)
    
    plt.xlabel('Desired Coverage Level')
    plt.ylabel('Coverage Rate')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.grid(True, alpha=0.5)
    if COVERAGE_LEVELS and len(COVERAGE_LEVELS) > 0:
        plt.xlim(min(COVERAGE_LEVELS)-0.02, max(COVERAGE_LEVELS)+0.02)
        plt.ylim(min(COVERAGE_LEVELS)-0.02, max(COVERAGE_LEVELS)+0.02)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_coverage_calibration_interval_cqr(results_df: pd.DataFrame, 
                                           output_path: str,
                                           methods: List[str],
                                           method_map: Optional[Dict[str, str]] = None) -> None:
    """Plot coverage calibration for interval predictors (+CQR)."""
    set_plot_style()
    
    plot_data = results_df[results_df['method'].isin(methods)].copy()
    if plot_data.empty:
        print(f"Warning: No data for plot_coverage_calibration_interval_cqr.")
        return

    base_methods = sorted(list(set([m.split('+')[0] for m in methods])))
    colors = {base: color for base, color in zip(base_methods, sns.color_palette("husl", len(base_methods)))}

    plt.figure(figsize=(8, 5))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=4, label='Perfect Calibration', alpha=0.7)

    # Use mapped names for sorting and labeling
    mapped_methods = {m: method_map.get(m, m) for m in methods}
    
    for method_name in sorted(methods, key=lambda m: mapped_methods[m]):
        method_data = plot_data[plot_data['method'] == method_name]
        if method_data.empty: continue
        agg_data = method_data.groupby('coverage_level')['coverage_rate'].mean().reset_index()
        base_name = method_name.split('+')[0]
        color = colors.get(base_name, 'gray')
        line_style = '-' if ('+CQR' in method_name or '+CP' in method_name) else '--'
        label = mapped_methods[method_name]
        plt.plot(agg_data['coverage_level'], agg_data['coverage_rate'],
                 marker='o', markersize=12, linestyle=line_style, linewidth=3, alpha=0.9,
                 color=color, label=label)
    
    plt.xlabel('Desired Coverage Level')
    plt.ylabel('Coverage Rate')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.grid(True, alpha=0.5)
    if COVERAGE_LEVELS and len(COVERAGE_LEVELS) > 0:
        plt.xlim(min(COVERAGE_LEVELS)-0.02, max(COVERAGE_LEVELS)+0.02)
        plt.ylim(0.13, max(COVERAGE_LEVELS)+0.05)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_winkler_vs_coverage(results_df: pd.DataFrame,
                             output_path: str,
                             method_map: Optional[Dict[str, str]] = None) -> None:
    """Plot Winkler Score vs. Coverage Level for all interval methods as a bar chart."""
    set_plot_style()
    
    plot_data = results_df.copy()
    if plot_data.empty or 'winkler_score' not in plot_data.columns:
        print("Warning: No data for Winkler score plot.")
        return
        
    if method_map:
        plot_data['method'] = plot_data['method'].map(method_map).fillna(plot_data['method'])

    sorted_methods = sorted(plot_data['method'].unique())
    plot_data['log_winkler_score'] = np.log(plot_data['winkler_score'] + 1e-8)  # Avoid log(0)
    
    plt.figure(figsize=(15, 6))
    ax = sns.barplot(
        data=plot_data,
        x='coverage_level',
        y='log_winkler_score',
        hue='method',
        hue_order=sorted_methods,
        errorbar='sd',
        palette='husl',
        capsize=0.1
    )
    
    ax.set_xlabel("Desired Coverage Level")
    ax.set_ylabel(r"$\log(\text{Winkler\ Score})$")
    # ax.set_title('Efficiency and Performance Comparison', y=1.03)
    
    plt.legend(title='Methods', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
    plt.ylim(5, 9.5)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_conditional_coverage_comparison(cond_cov_df: pd.DataFrame,
                                         output_path: str,
                                         group_names: List[str],
                                         coverage_levels: List[float],
                                         method_map: Optional[Dict[str, str]] = None) -> None:
    """
    Plot conditional coverage comparison using bars for CQR/CP and points for Mondrian.
    """
    set_plot_style()

    if cond_cov_df.empty:
        print("Warning: No data for conditional coverage comparison plot.")
        return

    plot_data = cond_cov_df.copy()
    plot_data['base_method'] = plot_data['method'].apply(lambda x: x.split('+')[0])
    
    # Use method_map to get short names for base methods
    if method_map:
        base_method_map = {k: v for k, v in method_map.items() if '+' not in k and k in plot_data['base_method'].unique()}
        plot_data['base_method_short'] = plot_data['base_method'].map(base_method_map)
    else:
        plot_data['base_method_short'] = plot_data['base_method']
        
    plot_data.dropna(subset=['base_method_short'], inplace=True)
    plot_data['group_name'] = plot_data['group'].map(dict(enumerate(group_names)))

    unique_base_methods = sorted(plot_data['base_method_short'].unique())
    colors = {method: color for method, color in zip(unique_base_methods, sns.color_palette("husl", len(unique_base_methods)))}

    n_methods = len(unique_base_methods)
    n_groups = len(group_names)
    bar_width = 0.8 / n_methods
    
    fig, axes = plt.subplots(len(coverage_levels), 1, figsize=(11, 2.2 * len(coverage_levels)), sharex=True)
    if len(coverage_levels) == 1: axes = [axes]

    for i, conf_level in enumerate(coverage_levels):
        ax = axes[i]
        level_data = plot_data[plot_data['coverage_level'] == conf_level]
        
        for j, base_method_short in enumerate(unique_base_methods):
            method_data = level_data[level_data['base_method_short'] == base_method_short]
            
            # Data for standard CQR/CP (bars)
            cqr_data = method_data[method_data['method'].str.contains(r'\+CQR|\+CP$')]
            # Data for Mondrian CQR/CP (points)
            mondrian_data = method_data[method_data['method'].str.contains(r'\+MondrianCP$')]

            if not cqr_data.empty:
                # Align data with group names for plotting
                cqr_plot_df = pd.DataFrame({'group_name': group_names}).merge(cqr_data, on='group_name', how='left')
                x_pos = np.arange(n_groups) + j * bar_width - (n_methods - 1) * bar_width / 2
                
                ax.bar(x_pos, cqr_plot_df['coverage_mean'], width=bar_width, yerr=cqr_plot_df['coverage_std'],
                       color=colors[base_method_short], capsize=4, alpha=0.8)

            if not mondrian_data.empty:
                # Align data with group names for plotting
                mondrian_plot_df = pd.DataFrame({'group_name': group_names}).merge(mondrian_data, on='group_name', how='left')
                x_pos = np.arange(n_groups) + j * bar_width - (n_methods - 1) * bar_width / 2

                ax.errorbar(x_pos, mondrian_plot_df['coverage_mean'], yerr=mondrian_plot_df['coverage_std'],
                            fmt='o', color='black', mfc=colors[base_method_short], mec='black',
                            ms=8, elinewidth=2, capsize=4, mew=1.5)

        ax.axhline(y=conf_level, color='red', linestyle='--', alpha=0.9, label=f'Target ({conf_level:.2f})')
        ax.set_ylabel("Coverage Rate", fontsize=16)
        ax.set_title(f"Desired Coverage Level = {conf_level}", fontsize=16)
        ax.set_xticks(np.arange(n_groups))
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xticklabels(group_names, rotation=0, fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Compute overall min and max coverage in this subplot
        all_coverages = []
        for base_method_short in unique_base_methods:
            method_data = level_data[level_data['base_method_short'] == base_method_short]
            cqr_data = method_data[method_data['method'].str.contains(r'\+CQR|\+CP$')]
            mondrian_data = method_data[method_data['method'].str.contains(r'\+MondrianCP$')]
            if not cqr_data.empty:
                all_coverages.extend(cqr_data['coverage_mean'].dropna().tolist())
            if not mondrian_data.empty:
                all_coverages.extend(mondrian_data['coverage_mean'].dropna().tolist())

        if all_coverages:
            min_cov = min(all_coverages)
            max_cov = max(all_coverages)
            bottom = max(0, min_cov - 0.05)
            top = min(1, max_cov + 0.05)
        else:
            # Fallback if no data
            bottom = max(0, conf_level - 0.2)
            top = min(1, conf_level + 0.2)

        ax.set_ylim(bottom=bottom, top=top)
        # --- Add Subplot Label (a), (b), etc. in italic ---
        label = f"({chr(97 + i)})"
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', style='italic',
                va='baseline', ha='left')

    # Create a comprehensive legend
    legend_elements = [mpatches.Patch(color=colors[m], label=m) for m in unique_base_methods]
    legend_elements.append(Line2D([0], [0], color='red', lw=2, linestyle='--', label='Target Level')) # type: ignore
    legend_elements.append(mpatches.Patch(facecolor='grey', edgecolor='black', alpha=0.8, label='Standard CP (Bar)')) # type: ignore
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Mondrian CP (Point)',
                          markerfacecolor='grey', markeredgecolor='black', markersize=10, mew=1.5)) # type: ignore

    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5), title="Methods & Variants", fontsize=16, title_fontsize=16)
    # fig.suptitle('Conditional Coverage: Standard vs. Mondrian Conformalization', y=1.01, fontsize=20)
    plt.xlabel("Target Station Group", fontsize=16)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")
