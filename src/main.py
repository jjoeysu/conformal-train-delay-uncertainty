# main.py

"""Main script to run all experiments."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

warnings.filterwarnings('ignore')

from src.config import (
    MODEL_LIST, NUM_SPLITS, COVERAGE_LEVELS,
    MODELS_PATH, PARAMS_PATH, METRICS_PATH,
    DROPOUT_SAMPLESIZE, RANDOM_SEED
)
from src.data_loader import DataLoader
from src.models import (
    HistoricalAverage, HistoricalConfidenceInterval,
    NaiveModel, LassoModel,
    XGBoostModel, LightGBMModel, CatBoostModel, MLPModel, XGBQuantile, 
    QuantileRandomForest, QuantDNN,
    MCDropout, GaussianDeepEnsembles
)
from src.conformal import SplitConformal, CQR, MondrianCP
from src.evaluation import evaluate_predictions, evaluate_conditional_coverage


def save_predictions(
    y_true: np.ndarray,
    method: str,
    split_index: int,
    y_pred_point: Optional[np.ndarray] = None,
    y_pred_lower: Optional[np.ndarray] = None,
    y_pred_upper: Optional[np.ndarray] = None
) -> None:
    """Save raw predictions to CSV."""
    pred_data = {'y_true': y_true}
    if y_pred_point is not None:
        pred_data['y_pred_point'] = y_pred_point
    if y_pred_lower is not None:
        pred_data['y_pred_lower'] = y_pred_lower
    if y_pred_upper is not None:
        pred_data['y_pred_upper'] = y_pred_upper
    df = pd.DataFrame(pred_data)
    filename = f"{method}_split{split_index}_predictions.csv"
    path = os.path.join(METRICS_PATH, filename)
    df.to_csv(path, index=False)


def save_metrics_per_method(metrics_list: list, method: str, split_index: int):
    """Save evaluation metrics for a method and split to CSV."""
    df = pd.DataFrame(metrics_list)
    filename = f"{method}_split{split_index}_metrics.csv"
    path = os.path.join(METRICS_PATH, filename)
    df.to_csv(path, index=False)
    print(f"Saved metrics for {method} to {path}")


def get_model_instance(model_name: str, split_index: int, **kwargs):
    """Create model instance; handles dependencies like MCDropout requiring a trained MLP."""
    model_mapping = {
        'HA': HistoricalAverage, 'HCI': HistoricalConfidenceInterval,
        'Naive': NaiveModel, 'Lasso': LassoModel,
        'XGBoost': XGBoostModel, 'LightGBM': LightGBMModel,
        'CatBoost': CatBoostModel, 'MLP': MLPModel,
        'XGBQuantile': XGBQuantile, 'QRF': QuantileRandomForest,
        'QuantDNN': QuantDNN, 'MCDropout': MCDropout,
        'GaussianDeepEnsembles': GaussianDeepEnsembles
    }
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model: {model_name}")

    # Models that depend on a pre-trained base model (e.g., MLP)
    dependent_models = {
        'MCDropout': 'MLP',
        'QuantDNN': 'MLP'
    }

    if model_name in dependent_models:
        base_model_name = dependent_models[model_name]
        print(f"Initializing '{model_name}' with pre-trained '{base_model_name}' for split {split_index}.")

        base_model_instance = model_mapping[base_model_name]()
        base_model_path = os.path.join(MODELS_PATH, f"{base_model_name}_split{split_index}.pkl")
        base_params_path = os.path.join(PARAMS_PATH, f"{base_model_name}_best_params.json")

        if not os.path.exists(base_model_path):
            raise FileNotFoundError(
                f"Missing required model file: {base_model_path}. "
                f"Train '{base_model_name}' first for split {split_index}."
            )

        print(f"Loading base model from: {base_model_path}")
        base_model_instance.load(base_model_path)

        if os.path.exists(base_params_path):
            print(f"Loading base model params from: {base_params_path}")
            base_model_instance.load_params(base_params_path)
        else:
            print(f"Warning: Base model params not found at {base_params_path}.")

        return model_mapping[model_name](base_model=base_model_instance)

    return model_mapping[model_name](**kwargs)


def is_point_predictor(model_name: str) -> bool:
    """Check if model only predicts point estimates."""
    return model_name in ['HA', 'Naive', 'Lasso', 'XGBoost', 'LightGBM', 'CatBoost', 'MLP']


def is_interval_predictor(model_name: str) -> bool:
    """Check if model natively outputs prediction intervals."""
    return model_name in ['HCI', 'XGBQuantile', 'QRF', 'QuantDNN', 'MCDropout', 'DeepEnsembles', 'GaussianDeepEnsembles']


def can_be_calibrated(model_name: str) -> bool:
    """Check if model supports conformal calibration."""
    return model_name not in ['HA', 'HCI']


def run_experiment(split_index: int) -> None:
    """Run full experiment for one data split."""
    print(f"\n{'='*60}\nRunning Split {split_index + 1}/{NUM_SPLITS}\n{'='*60}")

    data_loader = DataLoader()
    X_train, y_train, X_val, y_val, X_cal, y_cal, X_test, y_test = data_loader.get_split(split_index)
    groups_cal = data_loader.get_station_groups(X_cal)
    groups_test = data_loader.get_station_groups(X_test)

    all_results = []

    for model_name in MODEL_LIST:
        print(f"\n--- Processing Model: {model_name} ---")

        try:
            model = get_model_instance(model_name, split_index=split_index)
        except FileNotFoundError as e:
            print(f"Skipping '{model_name}': {e}")
            continue
        except Exception as e:
            print(f"Error initializing '{model_name}': {e}")
            continue

        canonical_params_path = os.path.join(PARAMS_PATH, f"{model_name}_best_params.json")

        if os.path.exists(canonical_params_path):
            print(f"Loading hyperparameters for {model_name} from {canonical_params_path}.")
            model.load_params(canonical_params_path)
        else:
            print(f"No hyperparameters found for {model_name}. HPO will run if needed.")

        # Train or load model
        model_path = os.path.join(MODELS_PATH, f"{model_name}_split{split_index}.pkl")

        if os.path.exists(model_path):
            print(f"Loading trained {model_name} model.")
            if model_name in ['GaussianDeepEnsembles']:
                model.load(model_path, X_train=X_train, y_train=y_train)
            else:
                model.load(model_path)
        else:
            print(f"Training {model_name} for split {split_index}...")
            model.train(X_train, y_train, X_val, y_val)
            model.save(model_path)
            if not os.path.exists(canonical_params_path) and model.best_params is not None:
                print(f"Saving hyperparameters for {model_name} to {canonical_params_path}.")
                model.save_params(canonical_params_path)

        # Evaluate baseline model
        print(f"Evaluating baseline {model_name}...")
        if is_point_predictor(model_name):
            y_pred_point = model.predict(X_test)
            point_metrics = evaluate_predictions(y_test, y_pred_point=y_pred_point)
            result = {
                'model': model_name, 'method': model_name, 'split': split_index,
                'coverage_level': np.nan, **point_metrics
            }
            all_results.append(result)
            save_metrics_per_method([result], method=model_name, split_index=split_index)

        if is_interval_predictor(model_name):
            interval_results = []
            for conf_level in COVERAGE_LEVELS:
                y_lower, y_upper = model.predict(X_test, coverage_level=conf_level)
                metrics = evaluate_predictions(y_test, y_pred_lower=y_lower, y_pred_upper=y_upper, coverage_levels=[conf_level])
                result = {
                    'model': model_name, 'method': model_name, 'split': split_index,
                    'coverage_level': conf_level, **metrics
                }
                all_results.append(result)
                interval_results.append(result)
            save_metrics_per_method(interval_results, method=model_name, split_index=split_index)

        # Apply conformal calibration if supported
        if can_be_calibrated(model_name):
            print(f"Calibrating {model_name} with conformal methods...")

            if is_point_predictor(model_name):
                y_pred = model.predict(X_test)
                cp_results = []
                for conf_level in COVERAGE_LEVELS:
                    # Standard CP
                    split_cp = SplitConformal(model)
                    split_cp.calibrate(X_cal, y_cal)
                    y_lower, y_upper = split_cp.predict(X_test, conf_level)
                    metrics = evaluate_predictions(y_test, y_pred, y_lower, y_upper, coverage_levels=[conf_level])
                    group_cov = evaluate_conditional_coverage(y_test, y_lower, y_upper, groups_test)
                    result = {
                        'model': model_name, 'method': f'{model_name}+CP', 'split': split_index,
                        'coverage_level': conf_level, **metrics,
                        **{f'coverage_group_{g}': cov for g, cov in group_cov.items()}
                    }
                    all_results.append(result)
                    cp_results.append(result)
                save_metrics_per_method(cp_results, method=f'{model_name}+CP', split_index=split_index)

                # Mondrian CP
                mondrian_cp = MondrianCP(model, base_conformalizer=SplitConformal(model))
                mondrian_cp.calibrate(X_cal, y_cal, groups_cal=groups_cal)
                mondrian_results = []
                for conf_level in COVERAGE_LEVELS:
                    y_lower, y_upper = mondrian_cp.predict(X_test, coverage_level=conf_level, groups_test=groups_test)
                    metrics = evaluate_predictions(y_test, y_pred, y_lower, y_upper, coverage_levels=[conf_level])
                    group_cov = evaluate_conditional_coverage(y_test, y_lower, y_upper, groups_test)
                    result = {
                        'model': model_name, 'method': f'{model_name}+MondrianCP', 'split': split_index,
                        'coverage_level': conf_level, **metrics,
                        **{f'coverage_group_{g}': cov for g, cov in group_cov.items()}
                    }
                    all_results.append(result)
                    mondrian_results.append(result)
                save_metrics_per_method(mondrian_results, method=f'{model_name}+MondrianCP', split_index=split_index)

            elif is_interval_predictor(model_name):
                cqr_results = []
                for conf_level in COVERAGE_LEVELS:
                    # Standard CQR
                    cqr = CQR(model)
                    cqr.calibrate(X_cal, y_cal, coverage_level=conf_level)
                    y_lower, y_upper = cqr.predict(X_test, conf_level)
                    metrics = evaluate_predictions(y_test, y_pred_lower=y_lower, y_pred_upper=y_upper, coverage_levels=[conf_level])
                    group_cov = evaluate_conditional_coverage(y_test, y_lower, y_upper, groups_test)
                    result = {
                        'model': model_name, 'method': f'{model_name}+CQR', 'split': split_index,
                        'coverage_level': conf_level, **metrics,
                        **{f'coverage_group_{g}': cov for g, cov in group_cov.items()}
                    }
                    all_results.append(result)
                    cqr_results.append(result)
                save_metrics_per_method(cqr_results, method=f'{model_name}+CQR', split_index=split_index)

                # Mondrian CQR
                mondrian_cqr = MondrianCP(model, base_conformalizer=CQR(model))
                mondrian_cqr_results = []
                for conf_level in COVERAGE_LEVELS:
                    mondrian_cqr.calibrate(X_cal, y_cal, groups_cal=groups_cal, coverage_level=conf_level)
                    y_lower, y_upper = mondrian_cqr.predict(X_test, coverage_level=conf_level, groups_test=groups_test)
                    metrics = evaluate_predictions(y_test, y_pred_lower=y_lower, y_pred_upper=y_upper, coverage_levels=[conf_level])
                    group_cov = evaluate_conditional_coverage(y_test, y_lower, y_upper, groups_test)
                    result = {
                        'model': model_name, 'method': f'{model_name}+MondrianCP', 'split': split_index,
                        'coverage_level': conf_level, **metrics,
                        **{f'coverage_group_{g}': cov for g, cov in group_cov.items()}
                    }
                    all_results.append(result)
                    mondrian_cqr_results.append(result)
                save_metrics_per_method(mondrian_cqr_results, method=f'{model_name}+MondrianCP', split_index=split_index)

    # Save all results for this split
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(METRICS_PATH, f"split_{split_index}_metrics.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nAll metrics for split {split_index} saved to {results_path}")


def main():
    """Run all experiment splits."""
    print("Starting experiments...")
    for split_idx in range(NUM_SPLITS):
        run_experiment(split_idx)
    print("\nAll experiments completed!")
    print(f"Metrics and predictions saved in: {METRICS_PATH}")


if __name__ == "__main__":
    main()