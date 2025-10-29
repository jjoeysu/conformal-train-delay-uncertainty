# analysis.py

"""
Enhanced Exploratory Data Analysis (EDA) for train operation data.
Features:
1. Load processed train data (timestamps in seconds).
2. Improved journey identification using origin station detection within the same day.
3. Corrected intra-journey stop ordering.
4. In-depth statistical analysis with detailed report output.
5. Network analysis module: directed station graph, centrality metrics.
6. High-quality visualizations with English labels, optimized for academic use.
7. Configurable parameters (e.g., Top-N).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from upsetplot import from_contents, UpSet

# --- 1. Setup & Initialization ---
matplotlib.use('Agg')  # Save plots directly to files
sns.set_theme(style="whitegrid", palette="colorblind")
print("Matplotlib backend set to 'Agg'. Plots will be saved to files without being displayed.")

# --- 2. Constants, Paths, and Output Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'statistical_analysis')
VIS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'dataset_production_one-year_epochtime_oneday.csv')

TOP_N = 20  # Number of top entries for analysis
DELAY_CLIP_SECONDS = 3600  # Clip delays to ±1 hour for visualization

def create_output_directories():
    """Create directories for reports and visualizations."""
    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Output directories are ready:\n  - Reports: {REPORT_DIR}\n  - Visualizations: {VIS_DIR}")

# --- 3. Data Loading & Preprocessing ---
def load_and_preprocess_data(file_path):
    """
    Load CSV data and preprocess it.
    Key logic:
    1. Flag start-of-journey records by checking BookedArrival == -10000.
    2. Within each group (Headcode, UnitNumber, Days), cumulatively sum flags to get journey instance number.
    3. Combine group keys with instance number to form unique JourneyID.
    4. Sort by JourneyID and arrival time to ensure correct order.
    5. Assign stop sequence numbers within each journey.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, dtype={'UnitNumber': str})
    except FileNotFoundError:
        print(f"Error: Data file not found. Please ensure '{file_path}' is correct.")
        return None

    print("Data loaded. Starting preprocessing...")

    # Step 1: Identify journey starts
    df['is_start_station'] = df['BookedArrival'] == -10000
    print("Step 1: Flagged all start-of-journey records.")

    # Step 2: Count journey instances within each group
    journey_instance_num = df.groupby(['Headcode', 'UnitNumber', 'Days'])['is_start_station'].cumsum()
    print("Step 2: Created journey instance number (e.g., 1st run, 2nd run).")

    # Step 3: Build unique JourneyID
    df['JourneyID'] = (df['Headcode'].astype(str) + '_' +
                       df['UnitNumber'].astype(str) + '_' +
                       df['Days'].astype(str) + '_' +
                       journey_instance_num.astype(str))
    print("Step 3: Assembled final, unique JourneyID.")

    # Step 4: Sort by JourneyID and arrival time
    df = df.sort_values(by=['JourneyID', 'BookedArrival']).reset_index(drop=True)
    print("Step 4: Sorted data by JourneyID and booked arrival time.")

    # Step 5: Add stop sequence index
    df['StopSequence'] = df.groupby('JourneyID').cumcount()
    print("Step 5: Created stop sequence numbers within each journey.")

    # Cleanup and add derived columns
    df = df.drop(columns=['is_start_station'])
    df['Date'] = pd.to_datetime('1970-01-01') + pd.to_timedelta(df['Days'], unit='d')
    df['DayOfWeek'] = df['Date'].dt.day_name()

    print("Preprocessing complete. Final 'JourneyID' and 'StopSequence' created.")
    return df

# --- 4. Network Construction & Topological Analysis ---
def build_and_analyze_network(df):
    """
    Construct a directed station graph from journey data.
    Requires correctly assigned JourneyID and StopSequence.
    Computes degree, betweenness, and eigenvector centrality.
    """
    print("\nBuilding and analyzing the station network...")
    G = nx.DiGraph()

    # Sort to ensure correct stop order
    df_sorted = df.sort_values(['JourneyID', 'StopSequence'])
    # Use shift to get next station per journey
    df_sorted['NextTiploc'] = df_sorted.groupby('JourneyID')['Tiploc'].shift(-1)

    # Remove last stops (no next station)
    edge_df = df_sorted.dropna(subset=['NextTiploc'])
    # Get unique edges
    unique_edges = edge_df[['Tiploc', 'NextTiploc']].drop_duplicates()
    G.add_edges_from(unique_edges.values)

    print(f"Network built. It has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Compute centrality measures
    print("Calculating network centrality measures...")
    in_degree = {node: G.in_degree(node) for node in G.nodes()}
    out_degree = {node: G.out_degree(node) for node in G.nodes()}
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-8)

    # Store results in DataFrame
    centrality_df = pd.DataFrame({
        'Tiploc': list(G.nodes()),
        'Degree': [G.degree(node) for node in G.nodes()],
        'InDegree': list(in_degree.values()),
        'OutDegree': list(out_degree.values()),
        'BetweennessCentrality': list(betweenness_centrality.values()),
        'EigenvectorCentrality': list(eigenvector_centrality.values())
    }).set_index('Tiploc')

    print("Network analysis complete.")
    return G, centrality_df

# --- 5. Statistical Analysis ---
def perform_statistical_analysis(df, centrality_df, report_path):
    """Perform comprehensive statistical analysis and save results to a text report."""
    print("Performing statistical analysis...")

    report_content = []
    report_content.append("="*60)
    report_content.append("Enhanced Train Operation EDA Report")
    report_content.append("="*60 + "\n")

    # Data overview
    report_content.append("--- 1. Dataset Overview ---")
    num_records = len(df)
    unique_journeys = df['JourneyID'].nunique()
    unique_headcodes = df['Headcode'].nunique()
    unique_tiplocs = df['Tiploc'].nunique()
    unique_units = df['UnitNumber'].nunique()
    min_date = df['Date'].min().strftime('%Y-%m-%d')
    max_date = df['Date'].max().strftime('%Y-%m-%d')

    report_content.append(f"Total records (stop events): {num_records:,}")
    report_content.append(f"Unique journey instances: {unique_journeys:,}")
    report_content.append(f"Unique Headcodes: {unique_headcodes:,}")
    report_content.append(f"Unique Tiplocs: {unique_tiplocs:,}")
    report_content.append(f"Unique UnitNumbers: {unique_units:,}")
    report_content.append(f"Date range: {min_date} to {max_date}\n")

    # Special value analysis (-10000)
    report_content.append("--- 2. Special Values (-10000) ---")
    start_points = (df['BookedArrival'] == -10000).sum()
    end_points = (df['BookedDeparture'] == -10000).sum()
    report_content.append("Value -10000 indicates journey start or end:")
    report_content.append(f"  - Journey starts (no arrival): {start_points:,} records")
    report_content.append(f"  - Journey ends (no departure): {end_points:,} records\n")

    # Descriptive statistics
    report_content.append("--- 3. Descriptive Statistics ---")
    numeric_cols = ['DepartureDiff', 'ArrivalDiff', 'DwellBooked', 'DwellActual', 'DwellDiff',
                    'UntilNextLocationBookedTime', 'UntilNextLocationActualTime', 'UntilNextLocationTimeDiff']
    valid_arrivals = df['ArrivalDiff'] > -10000
    report_content.append(df[numeric_cols][valid_arrivals].describe().to_string() + "\n")

    # Top N categorical features
    report_content.append(f"--- 4. Top Categorical Features (Top {TOP_N}) ---")
    top_headcodes = df['Headcode'].value_counts().nlargest(TOP_N)
    report_content.append(f"Top {TOP_N} Headcodes by frequency:\n" + top_headcodes.to_string() + "\n")
    top_tiplocs = df['Tiploc'].value_counts().nlargest(TOP_N)
    report_content.append(f"Top {TOP_N} Tiplocs by frequency:\n" + top_tiplocs.to_string() + "\n")
    top_units = df['UnitNumber'].value_counts().nlargest(TOP_N)
    report_content.append(f"Top {TOP_N} UnitNumbers by usage:\n" + top_units.to_string() + "\n")

    # Delay analysis
    report_content.append(f"--- 5. Delay Analysis (Top {TOP_N}) ---")
    df_with_arrival = df[df['ArrivalDiff'] > -10000]
    avg_delay_by_headcode = df_with_arrival.groupby('Headcode')['ArrivalDiff'].mean().nlargest(TOP_N)
    report_content.append(f"Top {TOP_N} Headcodes by average arrival delay:\n" + avg_delay_by_headcode.to_string() + "\n")
    avg_delay_by_tiploc = df_with_arrival.groupby('Tiploc')['ArrivalDiff'].mean().nlargest(TOP_N)
    report_content.append(f"Top {TOP_N} Tiplocs by average arrival delay:\n" + avg_delay_by_tiploc.to_string() + "\n")

    # Network centrality
    report_content.append(f"--- 6. Network Centrality Analysis (Top {TOP_N}) ---")
    report_content.append(f"Top {TOP_N} Tiplocs by Degree (total connections):\n" + centrality_df['Degree'].nlargest(TOP_N).to_string() + "\n")
    report_content.append(f"Top {TOP_N} Tiplocs by In-Degree (inbound traffic):\n" + centrality_df['InDegree'].nlargest(TOP_N).to_string() + "\n")
    report_content.append(f"Top {TOP_N} Tiplocs by Out-Degree (outbound traffic):\n" + centrality_df['OutDegree'].nlargest(TOP_N).to_string() + "\n")
    report_content.append(f"Top {TOP_N} Tiplocs by Betweenness Centrality (bottleneck role):\n" + centrality_df['BetweennessCentrality'].nlargest(TOP_N).to_string() + "\n")
    report_content.append(f"Top {TOP_N} Tiplocs by Eigenvector Centrality (influential neighbors):\n" + centrality_df['EigenvectorCentrality'].nlargest(TOP_N).to_string() + "\n")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))

    print(f"Statistical analysis report saved to: {report_path}")

# --- 6. Visualization Functions ---
def save_plot(fig, filename, sub_dir=None):
    """Save figure to specified directory."""
    save_path = VIS_DIR
    if sub_dir:
        save_path = os.path.join(VIS_DIR, sub_dir)
        os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    fig.tight_layout(pad=2.0)
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  - Saved plot: {filename}")

def plot_distributions(df):
    print("  - Plotting delay distributions...")
    df_filtered_arrival = df[df['ArrivalDiff'] > -10000]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_filtered_arrival['ArrivalDiff'].clip(-DELAY_CLIP_SECONDS, DELAY_CLIP_SECONDS), bins=100, kde=True, ax=ax)
    ax.set_xlabel('Arrival Delay (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.axvline(df_filtered_arrival['ArrivalDiff'].mean(), color='r', linestyle='--', label=f'Mean: {df_filtered_arrival["ArrivalDiff"].mean():.2f}s')
    ax.axvline(df_filtered_arrival['ArrivalDiff'].median(), color='g', linestyle='-', label=f'Median: {df_filtered_arrival["ArrivalDiff"].median():.2f}s')
    ax.legend()
    save_plot(fig, 'distribution_arrival_delay.png')

    df_filtered_departure = df[df['DepartureDiff'] > -10000]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df_filtered_departure['DepartureDiff'].clip(-DELAY_CLIP_SECONDS, DELAY_CLIP_SECONDS), bins=100, kde=True, ax=ax)
    ax.set_title(f'Distribution of Departure Delay (Clipped to ±{int(DELAY_CLIP_SECONDS/60)} Minutes)', fontsize=16)
    ax.set_xlabel('Departure Delay (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.axvline(df_filtered_departure['DepartureDiff'].mean(), color='r', linestyle='--', label=f'Mean: {df_filtered_departure["DepartureDiff"].mean():.2f}s')
    ax.axvline(df_filtered_departure['DepartureDiff'].median(), color='g', linestyle='-', label=f'Median: {df_filtered_departure["DepartureDiff"].median():.2f}s')
    ax.legend()
    save_plot(fig, 'distribution_departure_delay.png')

def plot_top_n_charts(df):
    print(f"  - Plotting Top {TOP_N} bar charts...")
    def create_bar(data_series, title, xlabel, ylabel, filename):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(y=data_series.index, x=data_series.values, orient='h', ax=ax)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        save_plot(fig, filename)

    create_bar(df['Headcode'].value_counts().nlargest(TOP_N), f'Top {TOP_N} Headcodes by Frequency', 'Number of Records', 'Headcode', 'top_n_headcodes.png')
    create_bar(df['Tiploc'].value_counts().nlargest(TOP_N), f'Top {TOP_N} Tiplocs by Frequency', 'Number of Stops', 'Tiploc', 'top_n_tiplocs.png')
    create_bar(df['UnitNumber'].value_counts().nlargest(TOP_N), f'Top {TOP_N} Unit Numbers by Frequency', 'Number of Uses', 'Unit Number', 'top_n_unit_numbers.png')

def plot_correlation_heatmap(df):
    print("  - Plotting correlation heatmap...")
    corr_cols = ['DepartureDiff', 'ArrivalDiff', 'DwellBooked', 'DwellActual', 'UntilNextLocationBookedTime', 'UntilNextLocationActualTime']
    df_corr = df[corr_cols][(df['ArrivalDiff'] > -10000) & (df['DepartureDiff'] > -10000)].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, annot_kws={"size": 10})
    ax.set_title('Correlation Matrix of Numerical Features', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    save_plot(fig, 'correlation_heatmap.png')

def plot_delay_over_time(df):
    print("  - Plotting delay over time of day and day of week...")
    df_filtered = df[df['ArrivalDiff'] > -10000].copy()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='DayOfWeek', y='ArrivalDiff', data=df_filtered, order=weekday_order, showfliers=False, ax=ax)
    ax.set_title('Arrival Delay Distribution by Day of the Week', fontsize=16)
    ax.set_xlabel('Day of the Week', fontsize=12)
    ax.set_ylabel('Arrival Delay (seconds)', fontsize=12)
    ax.set_ylim(df_filtered['ArrivalDiff'].clip(-600, 1200).quantile(0.01), df_filtered['ArrivalDiff'].clip(-600, 1200).quantile(0.99))
    save_plot(fig, 'delay_by_day_of_week.png')

    bins = [-1, 6*3600, 12*3600, 18*3600, 24*3600]
    labels = ['Early Morning (00-06)', 'Morning (06-12)', 'Afternoon (12-18)', 'Evening (18-24)']
    df_filtered['TimeOfDay'] = pd.cut(df_filtered['BookedDeparture'], bins=bins, labels=labels, right=False)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='TimeOfDay', y='ArrivalDiff', data=df_filtered, showfliers=False, ax=ax)
    ax.set_title('Arrival Delay Distribution by Time of Day', fontsize=16)
    ax.set_xlabel('Time of Day (based on Booked Departure)', fontsize=12)
    ax.set_ylabel('Arrival Delay (seconds)', fontsize=12)
    ax.set_ylim(df_filtered['ArrivalDiff'].clip(-600, 1200).quantile(0.01), df_filtered['ArrivalDiff'].clip(-600, 1200).quantile(0.99))
    save_plot(fig, 'delay_by_time_of_day.png')

def plot_single_journey(df):
    """Plot a single journey's delay profile. Uses corrected JourneyID and StopSequence."""
    print("  - Plotting a single journey for illustration...")
    journey_counts = df['JourneyID'].value_counts()
    example_journey_id = journey_counts[(journey_counts > 15) & (journey_counts < 25)].index[0]
    journey_df = df[df['JourneyID'] == example_journey_id].sort_values(by='StopSequence').reset_index()
    journey_df_plot = journey_df[(journey_df['BookedArrival'] > -10000) & (journey_df['ActualArrival'] > -10000)]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(journey_df_plot['Tiploc'], journey_df_plot['BookedArrival'], 'o-', label='Booked Arrival', color='gray', markerfacecolor='none')
    ax.plot(journey_df_plot['Tiploc'], journey_df_plot['ActualArrival'], 'o-', label='Actual Arrival', color='royalblue')
    ax.fill_between(journey_df_plot['Tiploc'], journey_df_plot['BookedArrival'], journey_df_plot['ActualArrival'], where=journey_df_plot['ActualArrival'] > journey_df_plot['BookedArrival'], facecolor='red', alpha=0.3, interpolate=True, label='Delay')
    ax.fill_between(journey_df_plot['Tiploc'], journey_df_plot['BookedArrival'], journey_df_plot['ActualArrival'], where=journey_df_plot['ActualArrival'] < journey_df_plot['BookedArrival'], facecolor='green', alpha=0.3, interpolate=True, label='Early Arrival')
    ax.set_title(f'Profile for a Single Journey (ID: {example_journey_id})', fontsize=16)
    ax.set_xlabel('Station (Tiploc)', fontsize=12)
    ax.set_ylabel('Time of Day (seconds since midnight)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    save_plot(fig, 'single_journey_profile.png')

def plot_feature_relationships(df):
    print("  - Plotting feature relationship scatter plots...")
    df_filtered = df[(df['ArrivalDiff'] > -10000) & (df['DepartureDiff'] > -10000)].sample(n=min(50000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='UntilNextLocationBookedTime', y='UntilNextLocationTimeDiff', data=df_filtered, alpha=0.3, ax=ax, edgecolor=None)
    ax.set_title('Booked Travel Time to Next Station vs. Actual Time Difference', fontsize=14)
    ax.set_xlabel('Booked Travel Time to Next Station (seconds)', fontsize=12)
    ax.set_ylabel('Travel Time Difference (Actual - Booked)', fontsize=12)
    save_plot(fig, 'relationship_travel_time_vs_diff.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='DwellActual', y='UntilNextLocationTimeDiff', data=df_filtered, alpha=0.3, ax=ax, edgecolor=None)
    ax.set_title('Actual Dwell Time vs. Next Leg Travel Time Difference', fontsize=14)
    ax.set_xlabel('Actual Dwell Time at Station (seconds)', fontsize=12)
    ax.set_ylabel('Travel Time Difference (Actual - Booked)', fontsize=12)
    ax.set_xlim(0, df_filtered['DwellActual'].quantile(0.99))
    save_plot(fig, 'relationship_dwell_time_vs_diff.png')

# --- 7. Additional Visualization Functions ---
def plot_network_graph(G, centrality_df):
    print("  - Plotting the full network graph...")
    fig, ax = plt.subplots(figsize=(31, 20))
    pos = nx.kamada_kawai_layout(G)
    node_sizes = centrality_df['Degree'] * 50
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes.values, node_color=centrality_df['BetweennessCentrality'], cmap=plt.cm.viridis, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.3, arrows=True, arrowstyle='-|>', arrowsize=20, ax=ax)
    max_degree_node_name = centrality_df['Degree'].idxmax()
    node_position = pos[max_degree_node_name]
    ax.annotate(text=max_degree_node_name, xy=node_position, xytext=(100, 30), textcoords='offset points', arrowprops=dict(arrowstyle="->", color='black', linewidth=4, shrinkA=2, shrinkB=2), fontsize=40, fontweight='bold', color='black', bbox=dict(boxstyle="round,pad=0.5", fc='yellow', alpha=0.7))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(centrality_df['BetweennessCentrality']), vmax=max(centrality_df['BetweennessCentrality'])))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.ax.tick_params(labelsize=40, pad=5)
    cbar.ax.set_ylabel('Betweenness Centrality', rotation=270, labelpad=75, fontsize=40)
    plt.axis('off')
    save_plot(fig, 'network_topology.png')

def plot_degree_analysis(centrality_df):
    print("  - Plotting network degree analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(centrality_df['InDegree'], bins=30, ax=axes[0], kde=True)
    axes[0].set_title('In-Degree Distribution', fontsize=14)
    axes[0].set_xlabel('In-Degree', fontsize=12)
    axes[0].set_ylabel('Number of Stations', fontsize=12)
    sns.histplot(centrality_df['OutDegree'], bins=30, ax=axes[1], kde=True)
    axes[1].set_title('Out-Degree Distribution', fontsize=14)
    axes[1].set_xlabel('Out-Degree', fontsize=12)
    fig.suptitle('Station Degree Distributions', fontsize=18)
    save_plot(fig, 'network_degree_distribution.png')

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x='InDegree', y='OutDegree', data=centrality_df, ax=ax, alpha=0.7)
    ax.set_title('In-Degree vs. Out-Degree of Stations', fontsize=16)
    ax.set_xlabel('In-Degree', fontsize=12)
    ax.set_ylabel('Out-Degree', fontsize=12)
    max_degree = max(centrality_df['InDegree'].max(), centrality_df['OutDegree'].max())
    ax.plot([0, max_degree], [0, max_degree], ls="--", c=".3", label='In-Degree = Out-Degree')
    ax.legend()
    save_plot(fig, 'network_indegree_vs_outdegree.png')

def plot_centrality_vs_delay(df, centrality_df):
    print("  - Plotting centrality vs. delay relationship...")
    df_with_arrival = df[df['ArrivalDiff'] > -10000]
    avg_delay_by_tiploc = df_with_arrival.groupby('Tiploc')['ArrivalDiff'].mean().reset_index()
    merged_df = pd.merge(centrality_df.reset_index(), avg_delay_by_tiploc, on='Tiploc')
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.regplot(x='BetweennessCentrality', y='ArrivalDiff', data=merged_df, ax=axes[0], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[0].set_title('Betweenness Centrality vs. Average Arrival Delay', fontsize=14)
    axes[0].set_xlabel('Betweenness Centrality', fontsize=12)
    axes[0].set_ylabel('Average Arrival Delay (seconds)', fontsize=12)
    sns.regplot(x='Degree', y='ArrivalDiff', data=merged_df, ax=axes[1], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[1].set_title('Degree Centrality vs. Average Arrival Delay', fontsize=14)
    axes[1].set_xlabel('Total Degree', fontsize=12)
    axes[1].set_ylabel('Average Arrival Delay (seconds)', fontsize=12)
    save_plot(fig, 'network_centrality_vs_delay.png')

def plot_delay_propagation(df):
    """Plot how delay evolves along journey stops. Uses corrected JourneyID."""
    print("  - Plotting delay propagation along journeys...")
    df_filtered = df[df['ArrivalDiff'] > -10000].copy()
    delay_by_stop = df_filtered.groupby('StopSequence')['ArrivalDiff'].agg(['mean', 'std', 'count'])
    delay_by_stop = delay_by_stop[delay_by_stop['count'] > 100].reset_index()
    delay_by_stop['ci_upper'] = delay_by_stop['mean'] + 1.96 * delay_by_stop['std'] / np.sqrt(delay_by_stop['count'])
    delay_by_stop['ci_lower'] = delay_by_stop['mean'] - 1.96 * delay_by_stop['std'] / np.sqrt(delay_by_stop['count'])
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(delay_by_stop['StopSequence'], delay_by_stop['mean'], marker='o', linestyle='-', label='Mean Arrival Delay')
    ax.fill_between(delay_by_stop['StopSequence'], delay_by_stop['ci_lower'], delay_by_stop['ci_upper'], color='b', alpha=0.2, label='95% Confidence Interval')
    ax.axhline(0, color='r', linestyle='--', label='On Time')
    ax.set_title('Average Arrival Delay by Stop Sequence in Journey', fontsize=16)
    ax.set_xlabel('Stop Sequence Number (0 = Origin)', fontsize=12)
    ax.set_ylabel('Average Arrival Delay (seconds)', fontsize=12)
    ax.set_xlim(left=0, right=min(50, delay_by_stop['StopSequence'].max()))
    ax.legend()
    ax.grid(True)
    save_plot(fig, 'delay_propagation.png')

def plot_route_station_overlap(df):
    print("  - Plotting route-station overlap with UpSet plot...")
    top_routes = df['Headcode'].value_counts().nlargest(10).index
    top_stations = df['Tiploc'].value_counts().nlargest(15).index
    filtered_df = df[df['Headcode'].isin(top_routes) & df['Tiploc'].isin(top_stations)]
    station_route_sets = filtered_df.groupby('Tiploc')['Headcode'].unique().apply(list)
    upset_data = from_contents(station_route_sets)
    fig = plt.figure(figsize=(15, 8))
    upset = UpSet(upset_data, subset_size='count', show_counts=True, sort_by='cardinality')
    upset.plot(fig=fig)
    plt.suptitle('Overlap of Top 10 Routes across Top 15 Stations', fontsize=16)
    save_plot(fig, 'route_station_overlap_upset.png')

def generate_visualizations(df, G, centrality_df):
    """Generate and save all visualizations."""
    print("\nGenerating visualizations with English labels...")
    plot_distributions(df)
    plot_network_graph(G, centrality_df)
    # plot_top_n_charts(df)
    # plot_correlation_heatmap(df)
    # plot_delay_over_time(df)
    # plot_feature_relationships(df)
    # plot_delay_propagation(df)
    # try:
    #     plot_route_station_overlap(df)
    # except Exception as e:
    #     print(f"  - Could not generate UpSet plot: {e}")
    # plot_degree_analysis(centrality_df)
    # plot_centrality_vs_delay(df, centrality_df)
    # try:
    #     plot_single_journey(df)
    # except IndexError:
    #     print("  - Could not generate single journey plot: No suitable journey found for illustration.")
    print("\nAll visualizations have been generated.")

# --- 8. Main Function ---
def main():
    """Main entry point."""
    print("Starting data analysis workflow...")
    create_output_directories()
    df = load_and_preprocess_data(DATA_PATH)
    if df is not None:
        G, centrality_df = build_and_analyze_network(df)
        report_file = os.path.join(REPORT_DIR, 'statistical_report.txt')
        perform_statistical_analysis(df, centrality_df, report_file)
        generate_visualizations(df, G, centrality_df)
        print("\nAnalysis workflow completed successfully!")
        print(f"Please check the results in:\n  - Reports: {REPORT_DIR}\n  - Visualizations: {VIS_DIR}")
    else:
        print("\nAnalysis workflow terminated due to data loading failure.")

if __name__ == '__main__':
    main()