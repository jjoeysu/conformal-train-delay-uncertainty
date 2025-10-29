# evaluation.py

"""Functions for calculating all evaluation metrics."""

import numpy as np
from typing import Dict, List, Tuple


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        R² value. Can be negative if predictions are worse than constant mean.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Avoid division by zero
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0  # Perfect prediction or not
    
    r2 = 1 - (ss_res / ss_tot)
    return float(r2)



def calculate_coverage_rate(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Calculate coverage rate of prediction intervals.
    
    Args:
        y_true: True values.
        y_lower: Lower bounds of intervals.
        y_upper: Upper bounds of intervals.
        
    Returns:
        Coverage rate (percentage of true values within intervals).
    """
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(covered))


def calculate_mean_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Calculate mean width of prediction intervals.
    
    Args:
        y_lower: Lower bounds of intervals.
        y_upper: Upper bounds of intervals.
        
    Returns:
        Mean interval width.
    """
    return float(np.mean(y_upper - y_lower))


def calculate_winkler_score(y_true: np.ndarray, y_lower: np.ndarray, 
                           y_upper: np.ndarray, coverage_level: float) -> float:
    """Calculate Winkler score for interval predictions.
    
    The Winkler score penalizes both wide intervals and non-coverage.
    
    Args:
        y_true: True values.
        y_lower: Lower bounds of intervals.
        y_upper: Upper bounds of intervals.
        coverage_level: Coverage level used for intervals.
        
    Returns:
        Winkler score.
    """
    alpha = 1 - coverage_level
    width = y_upper - y_lower
    
    # Penalty for values outside interval
    lower_penalty = 2 / alpha * (y_lower - y_true) * (y_true < y_lower)
    upper_penalty = 2 / alpha * (y_true - y_upper) * (y_true > y_upper)
    
    score = width + lower_penalty + upper_penalty
    return float(np.mean(score))


from typing import Optional

def evaluate_predictions(y_true: np.ndarray, 
                        y_pred_point: Optional[np.ndarray] = None,
                        y_pred_lower: Optional[np.ndarray] = None, 
                        y_pred_upper: Optional[np.ndarray] = None,
                        coverage_levels: Optional[List[float]] = None) -> Dict[str, float]:
    """Evaluate all predictions and return metrics.
    
    Args:
        y_true: True values.
        y_pred_point: Point predictions (optional).
        y_pred_lower: Lower bounds of intervals (optional).
        y_pred_upper: Upper bounds of intervals (optional).
        coverage_levels: List of coverage levels used (optional).
        
    Returns:
        Dictionary of metric names and values.
    """
    metrics = {}
    
    # Point prediction metrics
    if y_pred_point is not None:
        metrics['mae'] = calculate_mae(y_true, y_pred_point)
        metrics['rmse'] = calculate_rmse(y_true, y_pred_point)
        metrics['r2'] = calculate_r2(y_true, y_pred_point)
    
    # Interval prediction metrics
    if y_pred_lower is not None and y_pred_upper is not None:
        lower, upper = np.minimum(y_pred_lower, y_pred_upper), np.maximum(y_pred_lower, y_pred_upper)
        metrics['coverage_rate'] = calculate_coverage_rate(y_true, lower, upper)
        metrics['mean_width'] = calculate_mean_width(lower, upper)
        
        if coverage_levels is not None and len(coverage_levels) == 1:
            metrics['winkler_score'] = calculate_winkler_score(
                y_true, lower, upper, coverage_levels[0]
            )
    
    return metrics


def evaluate_conditional_coverage(y_true: np.ndarray,
                                 y_pred_lower: np.ndarray,
                                 y_pred_upper: np.ndarray,
                                 groups: np.ndarray) -> Dict[int, float]:
    """Evaluate coverage rate for each group.
    
    Args:
        y_true: True values.
        y_pred_lower: Lower bounds of intervals.
        y_pred_upper: Upper bounds of intervals.
        groups: Group labels.
        
    Returns:
        Dictionary mapping group to coverage rate.
    """
    group_coverage = {}
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        mask = groups == group
        if np.sum(mask) > 0:
            coverage = calculate_coverage_rate(
                y_true[mask], 
                y_pred_lower[mask], 
                y_pred_upper[mask]
            )
            group_coverage[group] = coverage
    
    return group_coverage