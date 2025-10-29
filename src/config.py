# config.py

"""Central configuration file for all parameters."""

import os
from typing import Dict, List, Any

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'delay_feature_dataset.csv')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
MODELS_PATH = os.path.join(RESULTS_PATH, 'models')
PARAMS_PATH = os.path.join(RESULTS_PATH, 'params')
METRICS_PATH = os.path.join(RESULTS_PATH, 'metrics')


# Create directories if they don't exist
for path in [RESULTS_PATH, MODELS_PATH, PARAMS_PATH, METRICS_PATH]:
    os.makedirs(path, exist_ok=True)

# Added data sampling parameter for faster experiments
DATA_SAMPLING_RATE = 1.0

# Models to run
MODEL_LIST = [
    'Naive', 'HCI', 'Lasso', 'XGBoost', 'CatBoost', 'MLP', 'MCDropout', 
    'XGBQuantile', 'QuantDNN', 'LightGBM', 'QRF', 'GaussianDeepEnsembles'
]

# Experiment parameters
NUM_SPLITS = 10
TEST_SIZE = 0.2
# For example, if TEST_SIZE=0.2, the remaining 80% is split.
# If CALIBRATION_SIZE=0.25, it takes 25% of that 80%, which is 20% of the total data.
CALIBRATION_SIZE = 0.25 
# If VALIDATION_SIZE=1/6, it takes 1/6 of the remaining (80% - 20% = 60%), which is 10% of total.
VALIDATION_SIZE = 1/6  
# MODIFICATION: Set a fixed, larger batch size for all NN models
BATCH_SIZE = 1024
SUBSET_RATIO = 0.1  # Ratio of training data to use for hyperparameter tuning

# Coverage levels for evaluation
COVERAGE_LEVELS = [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

N_ENSEMBLE_MODELS = 5
DROPOUT_SAMPLESIZE = 100  # Number of MC Dropout samples
# Random seed
RANDOM_SEED = 42

# Target column
# TARGET_COLUMN = 'Segment_Actual_Travel_Time_Seconds'
TARGET_COLUMN = 'Arrival_Delay_Seconds'

# Feature columns
FEATURE_COLUMNS = [
    'ArrivalDiff_current', 'DepartureDiff_current', 'DwellDiff_current',
    'Num_stops_between', 'Booked_travel_time_segment', 'Booked_dwell_time_segment',
    'Stop_number_current', 'Trip_progress_percentage', 'ArrivalDiff_mean_so_far',
    'ArrivalDiff_max_so_far', 'ArrivalDiff_trend', 'DwellDiff_mean_so_far',
    'station_current_in_degree', 'station_current_out_degree', 'headcode_num_stations',
    'headcode_total_in_degree', 'headcode_total_out_degree', 'target_CHRX',
    'target_LNDNBDE', 'target_SVNOAKS', 'target_TONBDG', 'target_WLOE',
    'time_of_day_Afternoon', 'time_of_day_Evening', 'time_of_day_Morning',
    'time_of_day_Night_Early_Morning', 'unit_prefix_375', 'unit_prefix_376',
    'unit_prefix_377', 'UnitNumber_freq', 'DayOfWeek_sin', 'DayOfWeek_cos',
    'Month_sin', 'Month_cos'
]

# Station group columns for Mondrian CP
STATION_COLUMNS = ['target_CHRX', 'target_LNDNBDE', 'target_SVNOAKS', 'target_TONBDG', 'target_WLOE']

# Hyperparameter search spaces
HYPERPARAMETER_SPACES: Dict[str, Dict[str, Any]] = {
    'Lasso': {
        'alpha': [1e-5, 1e-1],
    },
    'XGBoost': {
        'n_estimators': (100, 1000),
        'max_depth': (4, 16),
        'learning_rate': (0.01, 0.2),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'reg_alpha': (0.0, 10.0),
        'reg_lambda': (0.0, 10.0),
    },
    'LightGBM': {
        'n_estimators': (100, 1000),
        'max_depth': (4, 16),
        'num_leaves': (50, 150),
        'learning_rate': (0.01, 0.2),
        'feature_fraction': (0.6, 1.0),
        'bagging_fraction': (0.6, 1.0),
        'bagging_freq': (1, 7),
        'reg_alpha': (0.0, 10.0),
        'reg_lambda': (0.0, 10.0),
    },
    'CatBoost': {
        'iterations': (100, 1000),
        'depth': (4, 16),
        'learning_rate': (0.01, 0.2),
        'l2_leaf_reg': (1.0, 10.0),
        'bagging_temperature': (0.0, 1.0),
    },
    'MLP': {
        'hidden_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'learning_rate': (0.0001, 0.01),
        'dropout_rate': (0.1, 0.5),
        'weight_decay': (0.0, 0.01),
    },
    'XGBQuantile': {
        'n_estimators': (100, 1000),
        'max_depth': (4, 16),
        'learning_rate': (0.01, 0.2),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'reg_alpha': (0.0, 10.0),
        'reg_lambda': (0.0, 10.0),
    },
    'QRF': {
        'n_estimators': (100, 1000),
        'max_depth': (4, 16),
        'min_samples_split': (20, 200),
        'min_samples_leaf': (10, 100),
        'max_features': (0.6, 1.0),
    },
    'QuantDNN': {
        'hidden_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'learning_rate': (0.0001, 0.01),
        'dropout_rate': (0.0, 0.3),
        'weight_decay': (0.0, 0.01),
    },
    'MCDropout': {
        'hidden_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'learning_rate': (0.0001, 0.01),
        'dropout_rate': (0.1, 0.5),
        'weight_decay': (0.0, 0.01),
    },
    'GaussianDeepEnsembles': {
        'hidden_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'learning_rate': (1e-5, 1e-3),
        'dropout_rate': (0.1, 0.5),
        'weight_decay': (0.0, 0.01),
    },
}

# Training parameters
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 3600*6  # 6 hour
EARLY_STOPPING_ROUNDS = 50
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 20