# models.py

"""Implementation of all prediction models."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
import joblib
import json
import os
import traceback

from scipy import stats
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from quantile_forest import RandomForestQuantileRegressor

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Specify GPU device
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split

from src.config import (
    HYPERPARAMETER_SPACES, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT,
    EARLY_STOPPING_ROUNDS, MLP_MAX_EPOCHS, MLP_PATIENCE, RANDOM_SEED, 
    N_ENSEMBLE_MODELS, DROPOUT_SAMPLESIZE, COVERAGE_LEVELS, BATCH_SIZE,
    FEATURE_COLUMNS, PARAMS_PATH, SUBSET_RATIO 
)


class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, model_name: str):
        """Initialize base model.
        
        Args:
            model_name: Name of the model.
        """
        self.model_name = model_name
        self.model = None
        self.best_params = None
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
        """
        pass
    
    @abstractmethod
    def predict(self, X_test: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Make predictions.
        
        Args:
            X_test: Test features.
            
        Returns:
            Predictions (point predictions or intervals).
        """
        pass
    
    def save(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save the model.
        """
        joblib.dump(self.model, path)
        
    def load(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Path to load the model from.
        """
        self.model = joblib.load(path)
        
    def save_params(self, path: str) -> None:
        """Save best parameters to disk.
        
        Args:
            path: Path to save the parameters.
        """
        if self.best_params is not None:
            # Convert tuple hidden_sizes to list for JSON serialization
            if 'hidden_sizes' in self.best_params and isinstance(self.best_params['hidden_sizes'], tuple):
                self.best_params['hidden_sizes'] = list(self.best_params['hidden_sizes'])
            with open(path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
                
    def load_params(self, path: str) -> None:
        """Load parameters from disk.
        
        Args:
            path: Path to load the parameters from.
        """
        with open(path, 'r') as f:
            self.best_params = json.load(f)
            # Convert list back to tuple for consistency
            if 'hidden_sizes' in self.best_params and isinstance(self.best_params['hidden_sizes'], list):
                self.best_params['hidden_sizes'] = tuple(self.best_params['hidden_sizes'])

class HistoricalAverage(BaseModel):
    """Historical Average model - predicts the mean of training data."""
    
    def __init__(self):
        super().__init__("HA")
        self.mean_value = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train by calculating mean of training targets."""
        self.mean_value = np.mean(y_train)
        self.model = self.mean_value  # For consistency with save/load
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the mean value for all samples."""
        return np.full(len(X_test), self.model)


class HistoricalConfidenceInterval(BaseModel):
    """Historical Confidence Interval - empirical quantiles from training data."""
    
    def __init__(self):
        super().__init__("HCI")
        self.quantiles = {}
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Calculate empirical quantiles from training data."""
        # Calculate quantiles for different coverage levels
        coverage_levels = COVERAGE_LEVELS
        for conf in coverage_levels:
            alpha = 1 - conf
            self.quantiles[conf] = {
                'lower': np.quantile(y_train, alpha / 2),
                'upper': np.quantile(y_train, 1 - alpha / 2)
            }
        self.model = self.quantiles
        
    def predict(self, X_test: np.ndarray, coverage_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Predict intervals based on empirical quantiles."""
        n_samples = len(X_test)
        lower = np.full(n_samples, self.model[coverage_level]['lower'])
        upper = np.full(n_samples, self.model[coverage_level]['upper'])
        return lower, upper
    

class NaiveModel(BaseModel):
    """
    Naive baseline: predicts arrival delay equals current departure delay.
    """
    def __init__(self):
        super().__init__("Naive")
        try:
            # Get index of 'DepartureDiff_current' feature
            self.feature_index = FEATURE_COLUMNS.index('DepartureDiff_current')
        except ValueError:
            raise ValueError("'DepartureDiff_current' not found in FEATURE_COLUMNS from config.py")
        # Dummy model attribute for save/load compatibility
        self.model = "NaivePredictor"

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """No training needed."""
        print("NaiveModel does not require training. Skipping.")
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return 'DepartureDiff_current' column as predictions."""
        return X_test[:, self.feature_index]

    def save(self, path: str) -> None:
        """No state to save; implemented for compatibility."""
        joblib.dump(self.model, path)
        
    def load(self, path: str) -> None:
        """No state to load; implemented for compatibility."""
        self.model = joblib.load(path)


class LassoModel(BaseModel):
    """Lasso regression with Optuna hyperparameter tuning and feature scaling."""
    
    def __init__(self):
        super().__init__("Lasso")

    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optuna objective: minimize validation RMSE."""
        alpha = trial.suggest_float('alpha', *HYPERPARAMETER_SPACES['Lasso']['alpha'], log=True)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=alpha, random_state=RANDOM_SEED))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred_val = pipeline.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        
        return rmse
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train Lasso model. If no preloaded params, use Optuna to tune alpha on data subset.
        Uses pipeline with StandardScaler and Lasso.
        """
        if self.best_params is None:
            print(f"Running hyperparameter optimization for {self.model_name}...")
            
            # Sample subsets for HPO
            n_train_subset = int(len(X_train) * SUBSET_RATIO)
            n_val_subset = int(len(X_val) * SUBSET_RATIO)

            train_indices = np.random.choice(len(X_train), n_train_subset, replace=False)
            val_indices = np.random.choice(len(X_val), n_val_subset, replace=False)

            X_train_subset, y_train_subset = X_train[train_indices], y_train[train_indices]
            X_val_subset, y_val_subset = X_val[val_indices], y_val[val_indices]

            print(f"Using subset for HPO: {n_train_subset} train, {n_val_subset} val samples.")

            # Run Optuna study
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=RANDOM_SEED)
            )
            study.optimize(
                lambda trial: self._objective(trial, X_train_subset, y_train_subset, X_val_subset, y_val_subset),
                n_trials=OPTUNA_N_TRIALS,
                timeout=OPTUNA_TIMEOUT,
                n_jobs=1
            )
            self.best_params = study.best_params
            print(f"Best params for Lasso: {self.best_params}, RMSE: {study.best_value}")
        else:
            print(f"Using pre-loaded hyperparameters for {self.model_name}.")

        # Train final model with best alpha
        final_alpha = self.best_params['alpha']
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=final_alpha, random_state=RANDOM_SEED))
        ])
        
        print("Training final Lasso model on full training data...")
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using trained pipeline (scaling included)."""
        return self.model.predict(X_test) # type: ignore


class XGBoostModel(BaseModel):
    """XGBoost regression model with hyperparameter tuning."""
    
    def __init__(self):
        super().__init__("XGBoost")
        
    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optuna objective function for hyperparameter tuning."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *HYPERPARAMETER_SPACES['XGBoost']['n_estimators']),
            'max_depth': trial.suggest_int('max_depth', *HYPERPARAMETER_SPACES['XGBoost']['max_depth']),
            'learning_rate': trial.suggest_float('learning_rate', *HYPERPARAMETER_SPACES['XGBoost']['learning_rate']),
            'subsample': trial.suggest_float('subsample', *HYPERPARAMETER_SPACES['XGBoost']['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *HYPERPARAMETER_SPACES['XGBoost']['colsample_bytree']),
            'reg_alpha': trial.suggest_float('reg_alpha', *HYPERPARAMETER_SPACES['XGBoost']['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *HYPERPARAMETER_SPACES['XGBoost']['reg_lambda']),
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }

        if torch.cuda.is_available():
            params["device"] = "cuda"
            params["tree_method"] = "hist"
        
        model = xgb.XGBRegressor(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        return rmse
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train XGBoost. If hyperparameters are not loaded, run Optuna tuning.
        Otherwise, use the pre-loaded hyperparameters.
        """

        if self.best_params is None:
            print(f"Running hyperparameter optimization for {self.model_name}...")
            # for example, use 10% of data for HPO
            n_train_subset = int(len(X_train) * SUBSET_RATIO)
            n_val_subset = int(len(X_val) * SUBSET_RATIO)

            # random sampling for representativeness
            train_indices = np.random.choice(len(X_train), n_train_subset, replace=False)
            val_indices = np.random.choice(len(X_val), n_val_subset, replace=False)

            X_train_subset, y_train_subset = X_train[train_indices], y_train[train_indices]
            X_val_subset, y_val_subset = X_val[val_indices], y_val[val_indices]

            print(f"Using subset for HPO: {n_train_subset} train samples, {n_val_subset} validation samples.")

            # Run hyperparameter optimization
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=RANDOM_SEED)
            )
            study.optimize(
                lambda trial: self._objective(trial, X_train_subset, y_train_subset, X_val_subset, y_val_subset),
                n_trials=OPTUNA_N_TRIALS,
                timeout=OPTUNA_TIMEOUT,
                n_jobs=1
            )
            self.best_params = study.best_params
        else:
            print(f"Using pre-loaded hyperparameters for {self.model_name}.")

        # Ensure fixed parameters are set
        params = self.best_params.copy()
        if torch.cuda.is_available():
            params["device"] = "cuda"
            params["tree_method"] = "hist"
        params['random_state'] = RANDOM_SEED
        params['n_jobs'] = -1
        
        # Train final model with best parameters
        self.model = xgb.XGBRegressor(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make point predictions."""
        # Ensure the returned prediction is a NumPy ndarray (handles sparse matrix returns)
        y_pred = self.model.predict(X_test)
        try:
            from scipy import sparse
            # Use getattr to safely obtain a toarray-like callable without static attribute access
            toarray_fn = getattr(y_pred, "toarray", None)
            if callable(toarray_fn):
                y_pred = toarray_fn()
            # Handle lists of elements where some items might be sparse matrices
            elif isinstance(y_pred, list):
                converted = []
                for item in y_pred:
                    item_toarray = getattr(item, "toarray", None)
                    if callable(item_toarray):
                        converted.append(item_toarray())
                    else:
                        converted.append(np.asarray(item))
                y_pred = np.asarray(converted)
        except Exception:
            # If scipy isn't available or something unexpected occurs, fall back to asarray
            pass
        return np.asarray(y_pred).squeeze()


class LightGBMModel(BaseModel):
    """LightGBM regression model with hyperparameter tuning."""
    
    def __init__(self):
        super().__init__("LightGBM")
        
    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optuna objective function for hyperparameter tuning."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *HYPERPARAMETER_SPACES['LightGBM']['n_estimators']),
            'max_depth': trial.suggest_int('max_depth', *HYPERPARAMETER_SPACES['LightGBM']['max_depth']),
            'num_leaves': trial.suggest_int('num_leaves', *HYPERPARAMETER_SPACES['LightGBM']['num_leaves']),
            'learning_rate': trial.suggest_float('learning_rate', *HYPERPARAMETER_SPACES['LightGBM']['learning_rate']),
            'feature_fraction': trial.suggest_float('feature_fraction', *HYPERPARAMETER_SPACES['LightGBM']['feature_fraction']),
            'bagging_fraction': trial.suggest_float('bagging_fraction', *HYPERPARAMETER_SPACES['LightGBM']['bagging_fraction']),
            'bagging_freq': trial.suggest_int('bagging_freq', *HYPERPARAMETER_SPACES['LightGBM']['bagging_freq']),
            'reg_alpha': trial.suggest_float('reg_alpha', *HYPERPARAMETER_SPACES['LightGBM']['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *HYPERPARAMETER_SPACES['LightGBM']['reg_lambda']),
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbose': 1
        }

        if torch.cuda.is_available():
            params['device'] = 'gpu'
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
        )
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        return rmse
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train LightGBM with hyperparameter tuning. If hyperparameters are not loaded, run Optuna tuning.
        Otherwise, use the pre-loaded hyperparameters."""

        if self.best_params is None:
            print(f"Running hyperparameter optimization for {self.model_name}...")
            n_train_subset = int(len(X_train) * SUBSET_RATIO)
            n_val_subset = int(len(X_val) * SUBSET_RATIO)

            train_indices = np.random.choice(len(X_train), n_train_subset, replace=False)
            val_indices = np.random.choice(len(X_val), n_val_subset, replace=False)

            X_train_subset, y_train_subset = X_train[train_indices], y_train[train_indices]
            X_val_subset, y_val_subset = X_val[val_indices], y_val[val_indices]

            print(f"Using subset for HPO: {n_train_subset} train samples, {n_val_subset} validation samples.")

            # Run hyperparameter optimization
            study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=RANDOM_SEED)
            )
            study.optimize(
                lambda trial: self._objective(trial, X_train_subset, y_train_subset, X_val_subset, y_val_subset),
                n_trials=OPTUNA_N_TRIALS,
                timeout=OPTUNA_TIMEOUT,
                n_jobs=1
            )
            self.best_params = study.best_params
        else:
            print(f"Using pre-loaded hyperparameters for {self.model_name}.")

        # Ensure fixed parameters are set
        params = self.best_params.copy()
        if torch.cuda.is_available():
            params['device'] = 'gpu'
        params['random_state'] = RANDOM_SEED
        params['n_jobs'] = -1
        params['verbose'] = -1
        
        
        # Train final model
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
        )
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make point predictions."""
        y_pred = self.model.predict(X_test)
        # Convert sparse matrix to dense numpy array if needed
        try:
            from scipy import sparse
            # If the prediction is a sparse matrix, call its toarray method safely via getattr
            if sparse.issparse(y_pred):
                toarray_fn = getattr(y_pred, "toarray", None)
                if callable(toarray_fn):
                    y_pred = toarray_fn()
                else:
                    # Fallback if toarray is not available for some reason
                    y_pred = np.asarray(y_pred)
            # If the prediction is a list (e.g., list of sparse matrices), convert each element safely
            elif isinstance(y_pred, list):
                converted = []
                for item in y_pred:
                    item_toarray = getattr(item, "toarray", None)
                    if callable(item_toarray):
                        converted.append(item_toarray())
                    else:
                        converted.append(np.asarray(item))
                y_pred = np.asarray(converted)
        except Exception:
            # If scipy isn't available or something unexpected occurs, fall back to asarray
            pass
        return np.asarray(y_pred).squeeze()


class CatBoostModel(BaseModel):
    """CatBoost regression model with hyperparameter tuning."""
    
    def __init__(self):
        super().__init__("CatBoost")
        
    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optuna objective function for hyperparameter tuning."""
        params = {
            'iterations': trial.suggest_int('iterations', *HYPERPARAMETER_SPACES['CatBoost']['iterations']),
            'depth': trial.suggest_int('depth', *HYPERPARAMETER_SPACES['CatBoost']['depth']),
            'learning_rate': trial.suggest_float('learning_rate', *HYPERPARAMETER_SPACES['CatBoost']['learning_rate']),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *HYPERPARAMETER_SPACES['CatBoost']['l2_leaf_reg']),
            'bagging_temperature': trial.suggest_float('bagging_temperature', *HYPERPARAMETER_SPACES['CatBoost']['bagging_temperature']),
            'random_seed': RANDOM_SEED,
            'verbose': False
        }

        if torch.cuda.is_available():
            params['task_type'] = 'GPU'
            params['devices'] = '0'
        
        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False
        )
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        return rmse
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train CatBoost with hyperparameter tuning. If hyperparameters are not loaded, run Optuna tuning.
        Otherwise, use the pre-loaded hyperparameters."""
        if self.best_params is None:
            print(f"Running hyperparameter optimization for {self.model_name}...")
            # for example, use 10% of data for HPO
            n_train_subset = int(len(X_train) * SUBSET_RATIO)
            n_val_subset = int(len(X_val) * SUBSET_RATIO)

            # random sampling for representativeness
            train_indices = np.random.choice(len(X_train), n_train_subset, replace=False)
            val_indices = np.random.choice(len(X_val), n_val_subset, replace=False)

            X_train_subset, y_train_subset = X_train[train_indices], y_train[train_indices]
            X_val_subset, y_val_subset = X_val[val_indices], y_val[val_indices]

            print(f"Using subset for HPO: {n_train_subset} train samples, {n_val_subset} validation samples.")

            # Run hyperparameter optimization
            study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=RANDOM_SEED)
            )
            study.optimize(
                lambda trial: self._objective(trial, X_train_subset, y_train_subset, X_val_subset, y_val_subset),
                n_trials=OPTUNA_N_TRIALS,
                timeout=OPTUNA_TIMEOUT,
                n_jobs=1
            )
            self.best_params = study.best_params
        else:
            print(f"Using pre-loaded hyperparameters for {self.model_name}.")

        # Ensure fixed parameters are set
        params = self.best_params.copy()
        if torch.cuda.is_available():
            params['task_type'] = 'GPU'
            params['devices'] = '0'
        params['random_state'] = RANDOM_SEED
        params['verbose'] = False
        
        # Train final model
        self.model = cb.CatBoostRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False
        )
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make point predictions."""
        y_pred = self.model.predict(X_test)
        # Normalize outputs to a 1D numpy array: handle arrays, sparse matrices and tuples
        try:
            from scipy import sparse
            # if prediction is a sparse matrix
            if sparse.issparse(y_pred):
                toarray_fn = getattr(y_pred, "toarray", None)
                if callable(toarray_fn):
                    y_pred = toarray_fn()
                else:
                    y_pred = np.asarray(y_pred)
            # if prediction is a tuple of arrays (e.g., some models might return (lower, upper))
            elif isinstance(y_pred, tuple):
                # choose the first element for point prediction
                y_pred = np.asarray(y_pred[0])
            # handle list of items (e.g., list of sparse matrices)
            elif isinstance(y_pred, list):
                converted = []
                for item in y_pred:
                    item_toarray = getattr(item, "toarray", None)
                    if callable(item_toarray):
                        converted.append(item_toarray())
                    else:
                        converted.append(np.asarray(item))
                y_pred = np.asarray(converted)
        except Exception:
            # If scipy isn't available or something unexpected occurs, fall back to asarray
            pass
        return np.asarray(y_pred).squeeze()


class MLPNet(nn.Module):
    """PyTorch MLP network."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], dropout_rate: float = 0.0):
        super(MLPNet, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x).squeeze(-1)

class MLPModel(BaseModel):
    """MLP regression model with PyTorch."""
    
    def __init__(self):
        super().__init__("MLP")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = None
        
    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optuna objective function for hyperparameter tuning."""

        # Suggest hyperparameters
        hidden_sizes = trial.suggest_categorical('hidden_sizes', HYPERPARAMETER_SPACES['MLP']['hidden_sizes'])
        learning_rate = trial.suggest_float('learning_rate', *HYPERPARAMETER_SPACES['MLP']['learning_rate'], log=True)
        dropout_rate = trial.suggest_float('dropout_rate', *HYPERPARAMETER_SPACES['MLP']['dropout_rate'])
        weight_decay = trial.suggest_float('weight_decay', *HYPERPARAMETER_SPACES['MLP']['weight_decay'])
        
        batch_size = BATCH_SIZE  # Use fixed batch size from config

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = MLPNet(X_train.shape[1], hidden_sizes, dropout_rate).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(MLP_MAX_EPOCHS):
            # Training
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    val_loss += criterion(outputs, batch_y).item() * len(batch_x)
            
            val_loss /= len(val_dataset)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= MLP_PATIENCE:
                break
                
            # Report intermediate value for pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.sqrt(best_val_loss)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train MLP with hyperparameter tuning. If hyperparameters are not loaded, run Optuna tuning.
        Otherwise, use the pre-loaded hyperparameters."""
        self.input_size = X_train.shape[1]

        if self.best_params is None:
            print(f"Running hyperparameter optimization for {self.model_name}...")

            # for example, use 10% of data for HPO
            n_train_subset = int(len(X_train) * SUBSET_RATIO)
            n_val_subset = int(len(X_val) * SUBSET_RATIO)

            # random sampling for representativeness
            train_indices = np.random.choice(len(X_train), n_train_subset, replace=False)
            val_indices = np.random.choice(len(X_val), n_val_subset, replace=False)

            X_train_subset, y_train_subset = X_train[train_indices], y_train[train_indices]
            X_val_subset, y_val_subset = X_val[val_indices], y_val[val_indices]

            print(f"Using subset for HPO: {n_train_subset} train samples, {n_val_subset} validation samples.")

            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=RANDOM_SEED)
            )
            study.optimize(
                lambda trial: self._objective(trial, X_train_subset, y_train_subset, X_val_subset, y_val_subset),
                n_trials=OPTUNA_N_TRIALS,
                timeout=OPTUNA_TIMEOUT,
                n_jobs=1
            )
            self.best_params = study.best_params
        else:
            print(f"Using pre-loaded hyperparameters for {self.model_name}.")
        
        
        # Train final model with best parameters
        hidden_sizes = self.best_params['hidden_sizes']
        learning_rate = self.best_params['learning_rate']
        dropout_rate = self.best_params['dropout_rate']
        weight_decay = self.best_params['weight_decay']
        batch_size = BATCH_SIZE
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create and train model
        self.model = MLPNet(X_train.shape[1], hidden_sizes, dropout_rate).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(MLP_MAX_EPOCHS):
            # Training
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    val_loss += criterion(outputs, batch_y).item() * len(batch_x)
            
            val_loss /= len(val_dataset)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= MLP_PATIENCE:
                break
        
        # Load best state
        if best_state:
            self.model.load_state_dict(best_state)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make point predictions."""
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = self.model(X_test_tensor).cpu().numpy()
        return predictions
    
    # Add a new prediction method for MC Dropout
    def predict_with_mc_dropout(self, X_test: np.ndarray, coverage_level: float = 0.95, n_samples: int = DROPOUT_SAMPLESIZE) -> Tuple[np.ndarray, np.ndarray]:
        """Predict intervals using MC Dropout by enabling train mode."""
        if self.best_params is None or self.best_params.get('dropout_rate', 0.0) == 0.0:
            raise ValueError("MC Dropout requires a model trained with a non-zero dropout rate.")

        
        # Set model to train mode to activate dropout layers
        self.model.train()
        
        predictions = []
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            for _ in range(n_samples):# Get n_samples predictions
                pred = self.model(X_test_tensor).cpu().numpy()
                predictions.append(pred)
        self.model.eval()  # Reset to eval mode
        
        predictions = np.array(predictions)
        
        alpha = 1 - coverage_level
        lower = np.quantile(predictions, alpha / 2, axis=0)
        upper = np.quantile(predictions, 1 - alpha / 2, axis=0)
        
        return lower, upper
    
    # Save model and architecture params together
    def save(self, path: str) -> None:
        """Save model state and architecture parameters to a single file."""
        if self.model is not None and self.best_params is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_size': self.model.model[0].in_features,
                'hidden_sizes': self.best_params['hidden_sizes'],
                'dropout_rate': self.best_params['dropout_rate'],
            }, path)
        
    # Load model from the self-contained file
    def load(self, path: str) -> None:
        """Load model from a self-contained checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_size = checkpoint['input_size']
        
        # Recreate model with saved architecture
        self.model = MLPNet(
            input_size=checkpoint['input_size'],
            hidden_sizes=checkpoint['hidden_sizes'],
            dropout_rate=checkpoint['dropout_rate']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.best_params is None:
            self.best_params = {}
        self.best_params['hidden_sizes'] = checkpoint['hidden_sizes']
        self.best_params['dropout_rate'] = checkpoint['dropout_rate']


class XGBQuantile(BaseModel):
    """
    Quantile Regression using XGBoost.
    This model REUSES the hyperparameters found by the XGBoostModel for efficiency and consistency.
    """
    
    def __init__(self):
        super().__init__("XGBQuantile")
        self.models = {}  # Store models for different quantiles
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train XGBoost quantile regression models.
        This method loads the pre-tuned parameters from the standard XGBoost model.
        It requires a modern version of XGBoost (>= 1.6).
        """
        # --- Load hyperparameters from XGBoost ---
        if self.best_params is None:
            print(f"Hyperparameters for {self.model_name} not loaded. Attempting to use parameters from XGBoost.")
            
            # Construct the path to the canonical XGBoost parameter file
            xgb_params_path = os.path.join(PARAMS_PATH, "XGBoost_best_params.json")

            if not os.path.exists(xgb_params_path):
                raise FileNotFoundError(
                    f"Required hyperparameter file not found at {xgb_params_path}. "
                    "Please ensure the 'XGBoost' model is tuned and its parameters are saved before running 'XGBQuantile'."
                )
            
            print(f"Loading hyperparameters from {xgb_params_path}.")
            self.load_params(xgb_params_path) # Use the base class loader method

            if self.best_params is None:
                raise ValueError("Failed to load hyperparameters from LightGBM.")
            
        else:
            print(f"Using pre-loaded hyperparameters for {self.model_name}.")
        
        # --- Logic for training quantile models ---
        coverage_levels = COVERAGE_LEVELS
        quantiles_to_train = set()
        for conf in coverage_levels:
            alpha = 1 - conf
            quantiles_to_train.add(round(alpha / 2, 4))
            quantiles_to_train.add(round(1 - alpha / 2, 4))

        trained_models = {}

        for quantile in sorted(list(quantiles_to_train)):
            print(f"Training final XGBoost model for quantile {quantile}...")
            
            # Start with the best parameters from XGBoost
            params = self.best_params.copy()
            
            # Set XGBoost-specific parameters for quantile regression
            params.update({
                'objective': 'reg:quantileerror',
                'quantile_alpha': quantile, # XGBoost uses 'quantile_alpha'
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
            })
            
            # Ensure GPU is used if available, consistent with XGBoostModel
            if torch.cuda.is_available():
                params["device"] = "cuda"
                params["tree_method"] = "hist"

            # Instantiate an XGBoost regressor with the quantile parameters
            model = xgb.XGBRegressor(**params)
            
            # The number of estimators is taken from the tuned XGBoost parameters.
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            trained_models[quantile] = model

        # Map trained quantile models to coverage levels
        for conf in coverage_levels:
            alpha_lower = round((1 - conf) / 2, 4)
            alpha_upper = round(1 - alpha_lower, 4)
            
            self.models[conf] = {
                'lower': trained_models[alpha_lower],
                'upper': trained_models[alpha_upper]
            }
        
        self.model = self.models # For compatibility with save/load
        
    
    def predict(self, X_test: np.ndarray, coverage_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Predict quantile intervals."""
        if coverage_level not in self.models:
            available_levels = sorted(self.models.keys())
            closest_level = min(available_levels, key=lambda x:abs(x-coverage_level))
            print(f"Warning: Model not trained for coverage level {coverage_level}. Using closest: {closest_level}")
            coverage_level = closest_level
            
        lower_model = self.models[coverage_level]['lower']
        upper_model = self.models[coverage_level]['upper']
        
        lower = lower_model.predict(X_test)
        upper = upper_model.predict(X_test)
        
        # Ensure lower <= upper
        return np.minimum(lower, upper), np.maximum(lower, upper)
    
    def save(self, path: str) -> None:
        """Save all quantile models."""
        joblib.dump(self.models, path)

    def load(self, path: str) -> None:
        """Load all quantile models."""
        self.models = joblib.load(path)
        self.model = self.models



# Use the quantile-forest library for a much cleaner and more robust implementation
class QuantileRandomForest(BaseModel):
    """
    Quantile Random Forest model using the quantile-forest package.
    """
    
    def __init__(self):
        super().__init__("QRF")

    @staticmethod
    def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """
        Calculates the pinball loss for a given quantile.

        Args:
            y_true: True target values.
            y_pred: Predicted quantile values.
            quantile: The quantile level (e.g., 0.05, 0.5, 0.95).

        Returns:
            The pinball loss score.
        """
        error = y_true - y_pred
        # Loss for under-prediction (error > 0) is quantile * error
        # Loss for over-prediction (error < 0) is (1 - quantile) * -error
        loss = np.mean(np.maximum(quantile * error, (quantile - 1) * error))
        return float(loss)

    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Optuna objective function for hyperparameter tuning.
        The objective is to minimize the average pinball loss on the validation set
        across all required quantiles defined in COVERAGE_LEVELS.
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *HYPERPARAMETER_SPACES['QRF']['n_estimators']),
            'max_depth': trial.suggest_int('max_depth', *HYPERPARAMETER_SPACES['QRF']['max_depth']),
            'min_samples_split': trial.suggest_int('min_samples_split', *HYPERPARAMETER_SPACES['QRF']['min_samples_split']),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', *HYPERPARAMETER_SPACES['QRF']['min_samples_leaf']),
            'max_features': trial.suggest_float('max_features', *HYPERPARAMETER_SPACES['QRF']['max_features']),
            'random_state': RANDOM_SEED,
            'n_jobs': -1  # Use all available CPU cores. GPU is not supported by this library.
        }
        
        # Instantiate and train the model with the trial's hyperparameters
        model = RandomForestQuantileRegressor(**params)
        model.fit(X_train, y_train)
        
        # --- Correct Evaluation Metric: Average Pinball Loss ---
        # 1. Determine all unique quantiles needed for prediction based on config
        quantiles_to_evaluate = set()
        for conf in COVERAGE_LEVELS:
            alpha = 1 - conf
            quantiles_to_evaluate.add(round(alpha / 2, 4))
            quantiles_to_evaluate.add(round(1 - alpha / 2, 4))
        
        quantiles_list = sorted(list(quantiles_to_evaluate))
        
        # 2. Predict all quantiles at once on the validation set
        y_pred_quantiles = model.predict(X_val, quantiles=quantiles_list)
        
        # 3. Calculate the pinball loss for each quantile and average them
        total_pinball_loss = 0
        for i, quantile in enumerate(quantiles_list):
            quantile_predictions = y_pred_quantiles[:, i]
            loss = self._pinball_loss(y_val, quantile_predictions, quantile)
            total_pinball_loss += loss
            
        average_pinball_loss = total_pinball_loss / len(quantiles_list)
        
        # Optuna will minimize this returned value
        return average_pinball_loss
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train Quantile Random Forest. If hyperparameters are not loaded, run Optuna tuning.
        Otherwise, use the pre-loaded hyperparameters.
        Note: Early stopping is not applicable to Random Forest models. The number of estimators
        is a hyperparameter determined by the tuning process.
        """
        if self.best_params is None:
            print(f"Finding best structural hyperparameters for {self.model_name} using average pinball loss...")
            # for example, use 10% of data for HPO
            n_train_subset = int(len(X_train) * SUBSET_RATIO)
            n_val_subset = int(len(X_val) * SUBSET_RATIO)

            # random sampling for representativeness
            train_indices = np.random.choice(len(X_train), n_train_subset, replace=False)
            val_indices = np.random.choice(len(X_val), n_val_subset, replace=False)

            X_train_subset, y_train_subset = X_train[train_indices], y_train[train_indices]
            X_val_subset, y_val_subset = X_val[val_indices], y_val[val_indices]

            print(f"Using subset for HPO: {n_train_subset} train samples, {n_val_subset} validation samples.")

            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=RANDOM_SEED)
            )
            study.optimize(
                lambda trial: self._objective(trial, X_train_subset, y_train_subset, X_val_subset, y_val_subset),
                n_trials=OPTUNA_N_TRIALS,
                timeout=OPTUNA_TIMEOUT,
                n_jobs=1
            )
            self.best_params = study.best_params
            print(f"Best params found for {self.model_name}: {self.best_params}")
        else:
            print(f"Using pre-loaded hyperparameters for {self.model_name}.")
        
        # Prepare final model parameters
        params = self.best_params.copy()
        params['random_state'] = RANDOM_SEED
        params['n_jobs'] = -1
        
        # Train the final model on the full training data
        print("Training final QRF model on train data...")
        self.model = RandomForestQuantileRegressor(**params)
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test: np.ndarray, coverage_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Predict quantile intervals using the dedicated library method."""
        alpha = 1 - coverage_level
        quantiles = [alpha / 2, 1 - alpha / 2]
        predictions = self.model.predict(X_test, quantiles=quantiles)
        
        lower = predictions[:, 0]
        upper = predictions[:, 1]
        
        return lower, upper
    



class MCDropout(BaseModel):
    """
    MC Dropout wrapper for a pre-trained MLP model.
    Inherits from BaseModel for framework compatibility.
    """
    def __init__(self, base_model: MLPModel):
        """
        Args:
            base_model: Pre-trained MLPModel instance.
        """
        super().__init__(model_name="MCDropout")
        self.base_model_instance = base_model
        self.model = self.base_model_instance.model
        self.best_params = self.base_model_instance.best_params

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """No training needed."""
        print("MCDropoutModel does not require training. Skipping.")
        pass

    def predict(self, X_test: np.ndarray, coverage_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Delegate prediction to the base MLP's MC Dropout method."""
        return self.base_model_instance.predict_with_mc_dropout(X_test, coverage_level)

    def save(self, path: str) -> None:
        """No state to save."""
        print(f"MCDropoutModel does not save its own model file. Skipping save for path: {path}")
        pass

    def load(self, path: str) -> None:
        """Cannot be loaded from file; must be initialized with a trained MLP."""
        print(f"MCDropoutModel cannot be loaded from a file.")
        pass

    def save_params(self, path: str) -> None:
        """Save parameters of the underlying MLP model."""
        print(f"Saving parameters of the underlying MLP model for reference.")
        self.base_model_instance.save_params(path)

    def load_params(self, path: str) -> None:
        """Load parameters into the underlying MLP model."""
        print(f"Loading parameters into the underlying MLP model.")
        self.base_model_instance.load_params(path)
        self.best_params = self.base_model_instance.best_params


class QuantDNNNet(nn.Module):
    """
    Minimal network for QuantDNN: takes a single point prediction and outputs two quantiles.
    """
    def __init__(self):
        super(QuantDNNNet, self).__init__()
        self.quantile_layer = nn.Linear(1, 2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (batch_size, 1), point predictions from MLP.
        Returns:
            lower, upper bounds (ensured via min/max).
        """
        quantiles = self.quantile_layer(x)
        lower, upper = quantiles[:, 0], quantiles[:, 1]
        return torch.min(lower, upper), torch.max(lower, upper)
    

class QuantDNN(BaseModel):
    """
    Quantile DNN that trains on top of a frozen pre-trained base model (e.g., MLP).
    Uses the base model's point prediction as input.
    """
    def __init__(self, base_model: MLPModel):
        super().__init__("QuantDNN")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = base_model
        self.models: Dict[float, QuantDNNNet] = {} 

    def _pinball_loss(self, y_true, y_lower, y_upper, alpha_lower, alpha_upper):
        """Compute pinball loss for quantile regression."""
        errors_lower = y_true - y_lower
        errors_upper = y_true - y_upper
        loss_lower = torch.mean(torch.max(alpha_lower * errors_lower, (alpha_lower - 1) * errors_lower))
        loss_upper = torch.mean(torch.max(alpha_upper * errors_upper, (alpha_upper - 1) * errors_upper))
        return loss_lower + loss_upper

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train quantile heads on frozen base model's predictions."""
        # Freeze base model
        self.base_model.model.eval()
        for param in self.base_model.model.parameters():
            param.requires_grad = False

        # Generate point predictions from base model
        print("Generating point predictions from the base MLP model...")
        with torch.no_grad():
            train_inputs = self.base_model.predict(X_train).reshape(-1, 1)
            val_inputs = self.base_model.predict(X_val).reshape(-1, 1)

        # Convert to tensors
        train_inputs_tensor = torch.FloatTensor(train_inputs).to(self.device)
        val_inputs_tensor = torch.FloatTensor(val_inputs).to(self.device)

        # Use base model's hyperparameters
        params = getattr(self.base_model, 'best_params', {'learning_rate': 0.001, 'weight_decay': 0.0})
        batch_size = BATCH_SIZE
        coverage_levels = COVERAGE_LEVELS

        for conf_level in coverage_levels:
            print(f"Training QuantDNN head for coverage level {conf_level}...")
            
            # Create and train head model
            head_model = QuantDNNNet().to(self.device)
            optimizer = optim.Adam(head_model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
            
            # Prepare data loaders
            train_dataset = TensorDataset(train_inputs_tensor, torch.FloatTensor(y_train))
            val_dataset = TensorDataset(val_inputs_tensor, torch.FloatTensor(y_val))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            alpha_lower = (1 - conf_level) / 2
            alpha_upper = 1 - alpha_lower

            # Training loop with early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None
            
            for epoch in range(MLP_MAX_EPOCHS):
                head_model.train()
                for batch_x_pred, batch_y in train_loader:
                    batch_x_pred, batch_y = batch_x_pred.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    y_lower_pred, y_upper_pred = head_model(batch_x_pred)
                    loss = self._pinball_loss(batch_y, y_lower_pred, y_upper_pred, alpha_lower, alpha_upper)
                    loss.backward()
                    optimizer.step()

                # Validation
                head_model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x_pred_val, batch_y_val in val_loader:
                        batch_x_pred_val, batch_y_val = batch_x_pred_val.to(self.device), batch_y_val.to(self.device)
                        y_lower_val, y_upper_val = head_model(batch_x_pred_val)
                        loss = self._pinball_loss(batch_y_val, y_lower_val, y_upper_val, alpha_lower, alpha_upper)
                        val_loss += loss.item() * len(batch_x_pred_val)
                val_loss /= len(val_dataset)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = head_model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= MLP_PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break

            if best_state:
                head_model.load_state_dict(best_state)
            
            self.models[conf_level] = head_model
        self.model = self.models

    def predict(self, X_test: np.ndarray, coverage_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Predict prediction interval using base model and quantile head."""
        if coverage_level not in self.models:
            available_levels = sorted(self.models.keys())
            closest_level = min(available_levels, key=lambda x: abs(x - coverage_level))
            print(f"Warning: No head for {coverage_level}. Using closest: {closest_level}")
            coverage_level = closest_level

        head_model = self.models[coverage_level]
        head_model.eval()
        self.base_model.model.eval()
        
        with torch.no_grad():
            point_predictions = self.base_model.predict(X_test).reshape(-1, 1)
            inputs_tensor = torch.FloatTensor(point_predictions).to(self.device)
            y_lower, y_upper = head_model(inputs_tensor)
            lower = y_lower.cpu().numpy()
            upper = y_upper.cpu().numpy()
        return lower, upper

    def save(self, path: str) -> None:
        """Save all quantile head models."""
        state_dicts = {conf: model.state_dict() for conf, model in self.models.items()}
        torch.save({'state_dicts': state_dicts}, path)

    def load(self, path: str) -> None:
        """Load all quantile head models."""
        checkpoint = torch.load(path, map_location=self.device)
        self.models = {}
        for conf, state_dict in checkpoint['state_dicts'].items():
            model = QuantDNNNet().to(self.device)
            model.load_state_dict(state_dict)
            self.models[float(conf)] = model
        self.model = self.models




# Network for predicting mean and log-variance
class GaussianMLPNet(nn.Module):
    """MLP that outputs mean and log-variance."""
    def __init__(self, input_size: int, hidden_sizes: list, dropout_rate: float):
        super(GaussianMLPNet, self).__init__()
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_size = size
        layers.append(nn.Linear(last_size, 2))  # Output: mu and log_var
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Negative log-likelihood loss for Gaussian output
class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu, log_var = pred.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        variance = torch.exp(log_var)
        loss = 0.5 * (log_var + (target - mu)**2 / variance)
        return loss.mean()


class GaussianDeepEnsembles(BaseModel):
    """Deep Ensembles model with Optuna hyperparameter tuning."""
    
    def __init__(self):
        super().__init__("GaussianDeepEnsembles")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: List[nn.Module] = []
        self.n_models = N_ENSEMBLE_MODELS
        self.input_size = None

    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optuna objective: train one model and return validation loss."""
        params = {
            'hidden_sizes': trial.suggest_categorical('hidden_sizes', HYPERPARAMETER_SPACES['GaussianDeepEnsembles']['hidden_sizes']),
            'learning_rate': trial.suggest_float('learning_rate', *HYPERPARAMETER_SPACES['GaussianDeepEnsembles']['learning_rate'], log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', *HYPERPARAMETER_SPACES['GaussianDeepEnsembles']['dropout_rate']),
            'weight_decay': trial.suggest_float('weight_decay', *HYPERPARAMETER_SPACES['GaussianDeepEnsembles']['weight_decay'])
        }
        model = self._train_single_model(X_train, y_train, X_val, y_val, params, seed=trial.number)

        model.eval()
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val.reshape(-1, 1)))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        criterion = GaussianNLLLoss()

        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item() * len(batch_x)
        final_val_loss = val_loss / len(val_dataset)
        return final_val_loss

    def _train_single_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           params: Dict[str, Any], seed: int) -> nn.Module:
        """Train a single ensemble member with given hyperparameters."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        hidden_sizes = params['hidden_sizes']
        learning_rate = params['learning_rate']
        dropout_rate = params['dropout_rate']
        weight_decay = params['weight_decay']
        batch_size = BATCH_SIZE

        if y_train.ndim == 1: y_train = y_train.reshape(-1, 1)
        if y_val.ndim == 1: y_val = y_val.reshape(-1, 1)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = GaussianMLPNet(X_train.shape[1], hidden_sizes, dropout_rate).to(self.device)
        criterion = GaussianNLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(MLP_MAX_EPOCHS):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    val_loss += criterion(outputs, batch_y).item() * len(batch_x)
            val_loss /= len(val_dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            if patience_counter >= MLP_PATIENCE:
                break

        if best_state:
            model.load_state_dict(best_state)
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train the ensemble using optimized hyperparameters."""
        self.input_size = X_train.shape[1]

        # Standardize inputs
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        y_train_reshaped = y_train.reshape(-1, 1)
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        y_train_scaled = self.y_scaler.fit_transform(y_train_reshaped).flatten()

        X_val_scaled = self.x_scaler.transform(X_val)
        y_val_scaled = self.y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        if self.best_params is None:
            print(f"Running hyperparameter optimization for {self.model_name}...")

            n_train_subset = int(len(X_train_scaled) * SUBSET_RATIO)
            n_val_subset = int(len(X_val_scaled) * SUBSET_RATIO)
            train_indices = np.random.choice(len(X_train_scaled), n_train_subset, replace=False)
            val_indices = np.random.choice(len(X_val_scaled), n_val_subset, replace=False)
            X_train_scaled_subset = X_train_scaled[train_indices]
            y_train_scaled_subset = y_train_scaled[train_indices]
            X_val_scaled_subset = X_val_scaled[val_indices]
            y_val_scaled_subset = y_val_scaled[val_indices]

            print(f"Using subset for HPO: {n_train_subset} train, {n_val_subset} val samples.")

            study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_SEED))
            study.optimize(
                lambda trial: self._objective(trial, X_train_scaled_subset, y_train_scaled_subset,
                                              X_val_scaled_subset, y_val_scaled_subset),
                n_trials=OPTUNA_N_TRIALS,
                timeout=OPTUNA_TIMEOUT,
                n_jobs=1
            )
            self.best_params = study.best_params
            print(f"Best params found: {self.best_params}")
        else:
            print(f"Using pre-loaded hyperparameters.")

        # Train final ensemble
        self.models = []
        print("Training final ensemble models with best hyperparameters...")
        for i in range(self.n_models):
            print(f"Training ensemble member {i+1}/{self.n_models}...")
            model = self._train_single_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                                           self.best_params, seed=RANDOM_SEED + i)
            self.models.append(model)

        self.model = self.models
        print("Deep Ensembles training complete.")

    def predict(self, X_test: np.ndarray, coverage_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Predict coverage intervals using the trained ensemble."""
        if not self.models:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        X_test_scaled = self.x_scaler.transform(X_test)
        all_preds = self._get_all_predictions(X_test_scaled)

        mus = all_preds[:, :, 0]          # (n_models, n_samples)
        log_vars = all_preds[:, :, 1]
        variances = np.exp(log_vars)

        mean_pred_scaled = np.mean(mus, axis=0)
        aleatoric_variance_scaled = np.mean(variances, axis=0)
        epistemic_variance_scaled = np.var(mus, axis=0)
        total_variance_scaled = aleatoric_variance_scaled + epistemic_variance_scaled

        mean_pred = self.y_scaler.inverse_transform(mean_pred_scaled.reshape(-1, 1)).flatten()
        total_std = np.sqrt(total_variance_scaled) * self.y_scaler.scale_

        alpha = 1.0 - coverage_level
        z = stats.norm.ppf(1 - alpha / 2)
        lower_bound = mean_pred - z * total_std
        upper_bound = mean_pred + z * total_std

        return lower_bound, upper_bound

    def _get_all_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """Get raw predictions from all ensemble members."""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                pred = model(X_test_tensor).cpu().numpy()
                predictions.append(pred)
        return np.array(predictions)

    def save(self, path: str) -> None:
        """Save model states, architecture, and scalers."""
        if self.models and self.best_params:
            torch.save({
                'models_state_dicts': [model.state_dict() for model in self.models],
                'input_size': self.input_size,
                'hidden_sizes': self.best_params['hidden_sizes'],
                'dropout_rate': self.best_params['dropout_rate'],
                'x_scaler': self.x_scaler,
                'y_scaler': self.y_scaler,
            }, path)

    def load(self, path: str, X_train=None, y_train=None) -> None:
        """Load model and scalers. Re-fit scalers if not saved."""
        checkpoint = torch.load(path, map_location=self.device)
        self.models = []

        if 'x_scaler' in checkpoint and 'y_scaler' in checkpoint:
            self.x_scaler = checkpoint['x_scaler']
            self.y_scaler = checkpoint['y_scaler']
            print("Scalers loaded from checkpoint.")
        else:
            if X_train is None or y_train is None:
                raise ValueError("Scalers not found. Provide X_train and y_train to re-fit.")
            print("Warning: Scalers not found. Re-fitting using provided data...")
            self.x_scaler = StandardScaler().fit(X_train)
            self.y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

        self.input_size = checkpoint['input_size']
        hidden_sizes = checkpoint['hidden_sizes']
        dropout_rate = checkpoint['dropout_rate']

        if self.best_params is None:
            self.best_params = {}
        self.best_params['hidden_sizes'] = hidden_sizes
        self.best_params['dropout_rate'] = dropout_rate

        for state_dict in checkpoint['models_state_dicts']:
            model = GaussianMLPNet(input_size=self.input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
            model.load_state_dict(state_dict)
            self.models.append(model.to(self.device))

        self.model = self.models