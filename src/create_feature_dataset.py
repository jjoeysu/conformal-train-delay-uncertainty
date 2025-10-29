# create_feature_dataset.py

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import os


def create_feature_dataset():
    """
    Creates a machine learning-ready feature dataset by:
    1. Loading preprocessed data and production journey logs.
    2. Building a directed graph of train movements (Tiploc network).
    3. Extracting topological features (in/out degrees, route complexity).
    4. Encoding categorical, cyclical, and frequency-based features.
    5. Saving the final feature set to CSV.
    """
    # --- 1. Environment Setup & File Paths ---
    print("Step 1: Setting up file paths...")

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prod_data_path = os.path.join(base_path, 'data', 'dataset_production_one-year_epochtime_oneday.csv')
    preprocessed_data_path = os.path.join(base_path, 'data', 'dataset_preprocessed.csv')
    output_path = os.path.join(base_path, 'data', 'delay_feature_dataset.csv')

    print(f"Loading production data from: {prod_data_path}")
    print(f"Loading preprocessed data from: {preprocessed_data_path}")

    # Load with explicit string dtype for UnitNumber to avoid mixed-type warnings
    df_prod = pd.read_csv(prod_data_path, dtype={'UnitNumber': str})
    df_preprocessed = pd.read_csv(preprocessed_data_path, dtype={'UnitNumber': str})

    # Work on a copy for feature engineering
    df_features = df_preprocessed.copy()

    # --- 2. Build Train Traffic Network (Directed Graph) ---
    print("\nStep 2: Building the train traffic network graph...")
    G = nx.DiGraph()

    # Corrected journey identification: multiple journeys may exist per (Headcode, UnitNumber, Days)
    # Use cumulative sum of start stations to distinguish instances
    print("Applying corrected journey identification logic for graph building...")
    df_prod['is_start_station'] = df_prod['BookedArrival'] == -10000
    journey_instance_num = df_prod.groupby(['Headcode', 'UnitNumber', 'Days'])['is_start_station'].cumsum()
    df_prod['CorrectedJourneyID'] = (
        df_prod['Headcode'].astype(str) + '_' +
        df_prod['UnitNumber'].astype(str) + '_' +
        df_prod['Days'].astype(str) + '_' +
        journey_instance_num.astype(str)
    )
    df_prod = df_prod.drop(columns=['is_start_station'])
    print("Correct JourneyIDs generated for graph construction.")

    # Build graph: add directed edges between consecutive Tiplocs in each journey
    journeys = df_prod.groupby('CorrectedJourneyID')
    for _, journey_df in tqdm(journeys, desc="Processing journeys for graph"):
        journey_df = journey_df.sort_values(by='BookedArrival')  # Ensure correct order
        tiplocs = journey_df['Tiploc'].tolist()
        for i in range(len(tiplocs) - 1):
            u, v = tiplocs[i], tiplocs[i + 1]
            if u != v:
                G.add_edge(u, v)

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- 3. Extract Graph-Based Topological Features ---
    print("\nStep 3: Calculating graph-based topological features...")

    # 3.1 Station-level features: in-degree and out-degree
    in_degrees = {node: G.in_degree(node) for node in G.nodes()}
    out_degrees = {node: G.out_degree(node) for node in G.nodes()}

    df_features['station_current_in_degree'] = df_features['Tiploc_current'].map(in_degrees).fillna(0)
    df_features['station_current_out_degree'] = df_features['Tiploc_current'].map(out_degrees).fillna(0)
    print("Calculated in/out degrees for current stations.")

    # 3.2 Route-level features: total stations and aggregated degrees along the route
    headcode_to_stations = df_prod.groupby('Headcode')['Tiploc'].unique().to_dict()
    headcode_features = {}

    for headcode, stations in tqdm(headcode_to_stations.items(), desc="Calculating headcode features"):
        num_stations = len(stations)
        total_in_degree = sum(in_degrees.get(station, 0) for station in stations)
        total_out_degree = sum(out_degrees.get(station, 0) for station in stations)
        headcode_features[headcode] = {
            'headcode_num_stations': num_stations,
            'headcode_total_in_degree': total_in_degree,
            'headcode_total_out_degree': total_out_degree
        }

    # Map headcode features to the dataset
    headcode_features_df = df_features['Headcode'].map(headcode_features).apply(pd.Series)
    df_features = pd.concat([df_features, headcode_features_df], axis=1)
    print("Calculated and mapped headcode features (station count, total degrees).")

    # --- 4. Encode Categorical and Cyclical Features ---
    print("\nStep 4: Encoding categorical and cyclical features...")

    # 4.1 One-Hot Encoding
    target_dummies = pd.get_dummies(df_features['Tiploc_target'], prefix='target')
    time_of_day_dummies = pd.get_dummies(df_features['TimeOfDay_Category'], prefix='time_of_day')

    df_features['UnitNumber_prefix'] = df_features['UnitNumber'].astype(str).str[:3]
    unit_prefix_dummies = pd.get_dummies(df_features['UnitNumber_prefix'], prefix='unit_prefix')
    print("Performed one-hot encoding for Tiploc_target, TimeOfDay_Category, and UnitNumber prefix.")

    # 4.2 Frequency Encoding for UnitNumber (based on full production data)
    unit_freq = df_prod['UnitNumber'].value_counts(normalize=True)
    df_features['UnitNumber_freq'] = df_features['UnitNumber'].map(unit_freq).fillna(0)
    print("Performed frequency encoding for UnitNumber.")

    # 4.3 Cyclical Encoding for DayOfWeek and Month
    df_features['DayOfWeek_sin'] = np.sin(2 * np.pi * df_features['DayOfWeek'] / 7)
    df_features['DayOfWeek_cos'] = np.cos(2 * np.pi * df_features['DayOfWeek'] / 7)
    df_features['Month_sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
    df_features['Month_cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
    print("Performed sine/cosine transformation for DayOfWeek and Month.")

    # --- 5. Finalize and Save Dataset ---
    print("\nStep 5: Integrating all features and preparing the final dataset...")

    # 5.1 Combine all feature types
    numeric_features = [
        'ArrivalDiff_current', 'DepartureDiff_current', 'DwellDiff_current',
        'Num_stops_between', 'Booked_travel_time_segment', 'Booked_dwell_time_segment',
        'Stop_number_current', 'Trip_progress_percentage', 'ArrivalDiff_mean_so_far',
        'ArrivalDiff_max_so_far', 'ArrivalDiff_trend', 'DwellDiff_mean_so_far'
    ]

    df_final = pd.concat([
        df_features[numeric_features],
        df_features[['station_current_in_degree', 'station_current_out_degree',
                    'headcode_num_stations', 'headcode_total_in_degree', 'headcode_total_out_degree']],
        target_dummies,
        time_of_day_dummies,
        unit_prefix_dummies,
        df_features[['UnitNumber_freq', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos']]
    ], axis=1)

    # 5.2 Add target variable: Arrival delay in seconds
    target_column = df_preprocessed['Segment_Actual_Travel_Time_Seconds']
    df_final['Segment_Actual_Travel_Time_Seconds'] = target_column
    df_final['Arrival_Delay_Seconds'] = (
        df_final['DepartureDiff_current'] +
        df_final['Segment_Actual_Travel_Time_Seconds'] -
        df_final['Booked_travel_time_segment']
    )
    df_final = df_final.drop(columns=['Segment_Actual_Travel_Time_Seconds'])

    # 5.3 Handle missing values
    if df_final.isnull().sum().sum() > 0:
        print("\nWarning: Missing values detected after feature engineering. Filling with 0.")
        print(df_final.isnull().sum()[df_final.isnull().sum() > 0])
        df_final.fillna(0, inplace=True)

    # 5.4 Save to file
    df_final.to_csv(output_path, index=False)

    print(f"\nSuccess! Final feature dataset created with {df_final.shape[0]} samples and {df_final.shape[1]} columns.")
    print(f"Dataset saved to: {output_path}")
    print("\nFinal columns:", df_final.columns.tolist())


if __name__ == '__main__':
    create_feature_dataset()