# data_preprocessing.py

"""
Train Running Data Preprocessing Script

Purpose:
1. Load processed train schedule and operational data.
2. Identify the Top-N busiest stations by traffic volume.
3. Construct predictive "segments" from upstream stations to each target station within a journey.
4. Engineer rich features for each segment, including static, dynamic, temporal, cumulative, and trend-based attributes.
5. Save the final feature-engineered dataset as a CSV for downstream machine learning.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm


# --- 1. Configuration & Initialization ---

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
INPUT_DATA_PATH = os.path.join(DATA_DIR, 'dataset_production_one-year_epochtime_oneday.csv')
OUTPUT_DATA_PATH = os.path.join(DATA_DIR, 'dataset_preprocessed.csv')

# Configurable parameters
TOP_N_STATIONS = 5  # Number of top stations to use as prediction targets


# --- 2. Utility Function ---

def get_time_of_day_category(seconds_since_midnight):
    """
    Categorizes time of day based on seconds since midnight.
    
    Bins:
        00:00–05:59 → Night_Early_Morning
        06:00–11:59 → Morning
        12:00–17:59 → Afternoon
        18:00–23:59 → Evening
    """
    if not isinstance(seconds_since_midnight, (int, float)) or seconds_since_midnight < 0:
        return "Unknown"

    hour = seconds_since_midnight // 3600

    if 0 <= hour < 6:
        return "Night_Early_Morning"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    else:
        return "Evening"


# --- 3. Data Loading & Preparation ---

def load_and_prepare_data(file_path):
    """
    Loads raw data and prepares base features including JourneyID and temporal attributes.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, dtype={'UnitNumber': str})
    except FileNotFoundError:
        print(f"Error: Data file not found at '{file_path}'. Exiting.")
        return None

    print("Data loaded. Building unique JourneyID using corrected logic...")

    # Flag start-of-journey records (where BookedArrival == -10000)
    df['is_start_station'] = df['BookedArrival'] == -10000

    # Within each (Headcode, UnitNumber, Days) group, cumulatively sum start flags
    # to distinguish multiple runs on the same day
    journey_instance_num = df.groupby(['Headcode', 'UnitNumber', 'Days'])['is_start_station'].cumsum()

    # Assemble unique JourneyID: Headcode_UnitNumber_Days_InstanceNumber
    df['JourneyID'] = (
        df['Headcode'].astype(str) + '_' +
        df['UnitNumber'].astype(str) + '_' +
        df['Days'].astype(str) + '_' +
        journey_instance_num.astype(str)
    )

    # Drop temporary flag column
    df = df.drop(columns=['is_start_station'])

    # Derive date and time-based features
    df['Date'] = pd.to_datetime('1970-01-01') + pd.to_timedelta(df['Days'], unit='d')
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
    df['Month'] = df['Date'].dt.month
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)

    print("Initial preparation complete with corrected JourneyID.")
    return df


# --- 4. Core Logic: Segment Creation & Feature Engineering ---

def create_prediction_segments(df, top_n):
    """
    Builds prediction segments for the top-n busiest stations.
    Each segment represents travel from an upstream station to a target station.
    Rich features are extracted for each segment.
    """
    # Identify top N busiest stations
    print(f"Identifying Top {top_n} busiest stations...")
    top_stations = df['Tiploc'].value_counts().nlargest(top_n).index.tolist()
    print(f"Top {top_n} stations: {top_stations}")

    # Filter journeys that pass through any of the top stations
    journey_ids = df[df['Tiploc'].isin(top_stations)]['JourneyID'].unique()
    relevant_df = df[df['JourneyID'].isin(journey_ids)].copy()
    print(f"Found {len(journey_ids)} journeys passing through these stations.")

    # Group by journey for processing
    grouped = relevant_df.groupby('JourneyID')
    all_segments = []

    print("Processing journeys to generate prediction segments...")
    for journey_id, journey_df in tqdm(grouped):
        # Sort by scheduled arrival to ensure correct sequence (start station first)
        journey_df = journey_df.sort_values(by='BookedArrival').reset_index(drop=True)

        # Get indices of stops that are in the top-station list
        target_indices = journey_df[journey_df['Tiploc'].isin(top_stations)].index
        valid_target_indices = target_indices[target_indices > 0]  # Exclude first stop

        # Generate segments for each valid target station
        for target_idx in valid_target_indices:
            target_row = journey_df.loc[target_idx]

            # Create a segment from each prior stop to the current target
            for current_idx in range(target_idx):
                current_row = journey_df.loc[current_idx]

                # Skip if actual departure/arrival times are invalid
                if current_row['ActualDeparture'] <= 0 or target_row['ActualArrival'] <= 0:
                    continue

                segment = {}

                # Target variable: actual travel time between current and target
                segment['Segment_Actual_Travel_Time_Seconds'] = \
                    target_row['ActualArrival'] - current_row['ActualDeparture']

                # Static and context features
                segment['Tiploc_current'] = current_row['Tiploc']
                segment['Tiploc_target'] = target_row['Tiploc']
                segment['UnitNumber'] = current_row['UnitNumber']
                segment['Headcode'] = current_row['Headcode']
                segment['JourneyID'] = journey_id

                # Temporal features (based on current stop's scheduled departure)
                segment['TimeOfDay_Category'] = get_time_of_day_category(current_row['BookedDeparture'])
                segment['DayOfWeek'] = current_row['DayOfWeek']
                segment['Month'] = current_row['Month']
                segment['Is_Weekend'] = current_row['Is_Weekend']

                # Real-time deviation features at current stop
                segment['ArrivalDiff_current'] = current_row['ArrivalDiff'] if current_row['BookedArrival'] != -10000 else 0
                segment['DepartureDiff_current'] = current_row['DepartureDiff']
                segment['DwellDiff_current'] = current_row['DwellDiff']

                # Segment-level characteristics
                segment['Num_stops_between'] = target_idx - current_idx - 1
                segment['Booked_travel_time_segment'] = target_row['BookedArrival'] - current_row['BookedDeparture']
                # Sum booked dwell times between current and target (excluding endpoints)
                intermediate_dwell_sum = journey_df.iloc[current_idx+1:target_idx]['DwellBooked'].sum() \
                    if segment['Num_stops_between'] > 0 else 0
                segment['Booked_dwell_time_segment'] = intermediate_dwell_sum

                # Cumulative journey progress and historical deviations up to current stop
                history_df = journey_df.iloc[:current_idx+1]
                segment['Stop_number_current'] = current_idx + 1
                segment['Trip_progress_percentage'] = (current_idx + 1) / len(journey_df)

                # Mean and max arrival delay so far (excluding start marker)
                valid_arrivals = history_df[history_df['BookedArrival'] != -10000]['ArrivalDiff']
                segment['ArrivalDiff_mean_so_far'] = valid_arrivals.mean() if not valid_arrivals.empty else 0
                segment['ArrivalDiff_max_so_far'] = valid_arrivals.max() if not valid_arrivals.empty else 0

                # Arrival delay trend: change from previous stop
                if current_idx > 0:
                    prev_row = journey_df.loc[current_idx - 1]
                    has_valid_prev = prev_row['BookedArrival'] != -10000
                    has_valid_curr = current_row['BookedArrival'] != -10000
                    segment['ArrivalDiff_trend'] = \
                        current_row['ArrivalDiff'] - prev_row['ArrivalDiff'] if has_valid_prev and has_valid_curr else 0
                else:
                    segment['ArrivalDiff_trend'] = 0

                # Average dwell time deviation so far
                segment['DwellDiff_mean_so_far'] = history_df['DwellDiff'].mean()

                all_segments.append(segment)

    print(f"Processing complete. Generated {len(all_segments)} prediction segments.")
    if not all_segments:
        print("Warning: No segments were generated. Check TOP_N_STATIONS or data integrity.")
        return None

    return pd.DataFrame(all_segments)


# --- 5. Main Execution ---

def main():
    """Main entry point."""
    print("Starting data preprocessing for prediction task (v2, corrected)...")

    # Load and prepare data
    df = load_and_prepare_data(INPUT_DATA_PATH)
    if df is None:
        print("Preprocessing terminated due to data loading failure.")
        return

    # Build feature-engineered dataset
    feature_df = create_prediction_segments(df, TOP_N_STATIONS)
    if feature_df is None:
        print("Preprocessing failed to generate a dataset.")
        return

    # Save results
    print(f"Saving feature-engineered dataset to {OUTPUT_DATA_PATH}...")
    feature_df.to_csv(OUTPUT_DATA_PATH, index=False)
    print("Dataset saved successfully.")

    print("\n--- Feature Overview ---")
    print(feature_df.info())

    print("\n--- Sample Data ---")
    print(feature_df.head())

    print("\nPreprocessing workflow completed successfully!")


if __name__ == '__main__':
    main()