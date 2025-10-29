# data_loader.py

"""Handles data loading, splitting, and preprocessing."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
from src.config import (
    DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_SEED,
    TEST_SIZE, VALIDATION_SIZE, CALIBRATION_SIZE, STATION_COLUMNS,
    DATA_SAMPLING_RATE  
)


class DataLoader:
    """Class for loading and splitting the dataset."""
    
    def __init__(self, data_path: str = DATA_PATH):
        """Initialize DataLoader with data path.
        
        Args:
            data_path: Path to the CSV file containing the dataset.
        """
        self.data_path = data_path
        self.data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        MODIFICATION: If DATA_SAMPLING_RATE < 1.0, it samples the data.
        
        Returns:
            DataFrame containing the loaded (and possibly sampled) data.
        """
        if self.data is None:
            print(f"Loading data from {self.data_path}...")
            self.data = pd.read_csv(self.data_path)
            print(f"Full data loaded: {self.data.shape[0]} samples, {self.data.shape[1]} columns")

            if DATA_SAMPLING_RATE < 1.0:
                print(f"Applying sampling with rate: {DATA_SAMPLING_RATE}")
                self.data = self.data.sample(
                    frac=DATA_SAMPLING_RATE, 
                    random_state=RANDOM_SEED
                ).reset_index(drop=True)
                print(f"Data sampled: {self.data.shape[0]} samples, {self.data.shape[1]} columns")

        return self.data
    
    def get_split(
        self, 
        split_index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        It splits data into train, validation, calibration, and test sets once.
        This ensures all models are trained on the exact same training set for fair comparison.

        1.  Data is split into (Train + Val + Cal) and Test.
        2.  (Train + Val + Cal) is split into (Train + Val) and Calibration.
        3.  (Train + Val) is split into Train and Validation.
        
        Args:
            split_index: Index of the split (for varying random seeds).
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_cal, y_cal, X_test, y_test).
        """
        data = self.load_data()
        X = data[FEATURE_COLUMNS].astype(np.float32).values
        y = data[TARGET_COLUMN].astype(np.float32).values
        
        random_state = RANDOM_SEED + split_index
        
        # First split: (train + val + cal) vs test
        X_train_val_cal, X_test, y_train_val_cal, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=random_state
        )
        
        # Second split: (train + val) vs cal
        # CALIBRATION_SIZE is relative to the remaining data after test split
        X_train_val, X_cal, y_train_val, y_cal = train_test_split(
            X_train_val_cal, y_train_val_cal, test_size=CALIBRATION_SIZE, random_state=random_state
        )

        # Third split: train vs val
        # VALIDATION_SIZE is relative to the remaining data after test and cal splits
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=VALIDATION_SIZE, random_state=random_state
        )
        
        print(f"Data split {split_index}:")
        print(f"  - Train set size: {len(X_train)}")
        print(f"  - Validation set size: {len(X_val)}")
        print(f"  - Calibration set size: {len(X_cal)}")
        print(f"  - Test set size: {len(X_test)}")

        return X_train, y_train, X_val, y_val, X_cal, y_cal, X_test, y_test

    def get_station_groups(self, X: np.ndarray) -> np.ndarray:
        """Extract station group labels from features."""
        station_indices = [FEATURE_COLUMNS.index(col) for col in STATION_COLUMNS]
        station_data = X[:, station_indices]
        groups = np.argmax(station_data, axis=1)
        return groups
    