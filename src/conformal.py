# conformal.py

"""Implements conformal prediction methods."""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from abc import ABC, abstractmethod

class BaseConformal(ABC):
    """Base class for conformal prediction methods."""
    
    def __init__(self, model):
        """Initialize with a trained model.
        
        Args:
            model: A trained prediction model.
        """
        self.model = model
        self.scores = None
        
    @abstractmethod
    def calculate_scores(self, X_cal: np.ndarray, y_cal: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate non-conformity scores on calibration set.
        
        Args:
            X_cal: Calibration features.
            y_cal: Calibration targets.
            
        Returns:
            Array of non-conformity scores.
        """
        pass
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray, **kwargs) -> None:
        """Calibrate the conformal predictor.
        
        Args:
            X_cal: Calibration features.
            y_cal: Calibration targets.
            **kwargs: Additional keyword arguments for score calculation.
        """
        self.scores = self.calculate_scores(X_cal, y_cal, **kwargs)
        
    def get_quantile(self, coverage_level: float) -> float:
        """Get the quantile of non-conformity scores.
        
        Args:
            coverage_level: Desired coverage level (e.g., 0.95).
            
        Returns:
            Quantile value.
        """

        if self.scores is None or len(self.scores) == 0:
            raise RuntimeError("Predictor has not been calibrated. Call calibrate() first.")
        n = len(self.scores)
        q_val = np.ceil((n + 1) * coverage_level) / n
        q_val = np.clip(q_val, 0, 1)
        return np.quantile(self.scores, q_val, method="higher")
    
    @abstractmethod
    def predict(self, X_test: np.ndarray, coverage_level: float = 0.95, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Produce prediction intervals.
        
        Args:
            X_test: Test features.
            coverage_level: Desired coverage level.
            
        Returns:
            Tuple of (lower_bounds, upper_bounds).
        """
        pass


class SplitConformal(BaseConformal):
    """Split Conformal Prediction for point prediction models."""
    
    def calculate_scores(self, X_cal: np.ndarray, y_cal: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate absolute residuals as non-conformity scores."""
        y_pred = self.model.predict(X_cal)
        scores = np.abs(y_cal - y_pred)
        return scores
    
    def predict(self, X_test: np.ndarray, coverage_level: float = 0.95, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Produce symmetric prediction intervals."""
        y_pred = self.model.predict(X_test)
        q_hat = self.get_quantile(coverage_level)
        
        lower = y_pred - q_hat
        upper = y_pred + q_hat
        
        return lower, upper


class CQR(BaseConformal):
    """Conformalized Quantile Regression."""
    
    def calculate_scores(self, X_cal: np.ndarray, y_cal: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate non-conformity scores for quantile regression.
        Scores are specific to the coverage level of the underlying quantile predictor.
        """
        coverage_level = kwargs.get('coverage_level')
        if coverage_level is None:
            raise ValueError("CQR.calculate_scores requires 'coverage_level' keyword argument.")
            
        y_lower, y_upper = self.model.predict(X_cal, coverage_level=coverage_level)
        scores = np.maximum(y_lower - y_cal, y_cal - y_upper)
        return scores
    
    def predict(self, X_test: np.ndarray, coverage_level: float = 0.95, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce calibrated prediction intervals.
        IMPORTANT: The conformalizer must have been calibrated using the *same coverage_level*
        before calling this method. The main experiment loop must handle this.
        """
        y_lower, y_upper = self.model.predict(X_test, coverage_level=coverage_level)
        
        q_hat = self.get_quantile(coverage_level)
        
        lower = y_lower - q_hat
        upper = y_upper + q_hat
        
        return lower, upper

class MondrianCP(BaseConformal):
    """Mondrian Conformal Prediction for conditional coverage."""
    
    def __init__(self, model, base_conformalizer: BaseConformal):
        super().__init__(model)
        self.base_conformalizer = base_conformalizer
        self.group_scores: Dict[int, np.ndarray] = {}
        self.overall_scores: Optional[np.ndarray] = None

    def calculate_scores(self, X_cal: np.ndarray, y_cal: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError("Use calibrate() for MondrianCP")

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray, groups_cal: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Calibrate with group information.
        It's crucial that **kwargs contains any parameters needed by the base conformalizer's
        calculate_scores method (e.g., 'coverage_level' for CQR).
        """
        if groups_cal is None:
            raise ValueError("groups_cal must be provided for MondrianCP calibration.")
        
        unique_groups = np.unique(groups_cal)
        all_scores_list = []
        self.group_scores = {} # Reset scores
        
        for group in unique_groups:
            mask = (groups_cal == group)
            if not np.any(mask):
                continue
            
            # Use the base conformalizer to calculate scores for this group
            group_scores = self.base_conformalizer.calculate_scores(
                X_cal[mask], y_cal[mask], **kwargs
            )
            self.group_scores[group] = group_scores
            all_scores_list.append(group_scores)
        
        if all_scores_list:
            self.overall_scores = np.concatenate(all_scores_list)

    def get_group_quantile(self, group: int, coverage_level: float) -> float:
        """Get quantile for a specific group."""
        scores_to_use = self.group_scores.get(group)
        
        if scores_to_use is None or len(scores_to_use) == 0:
            if self.overall_scores is None or len(self.overall_scores) == 0:
                raise RuntimeError("MondrianCP has no scores to compute quantiles.")
            scores_to_use = self.overall_scores

        n = len(scores_to_use)
        q_val = np.ceil((n + 1) * coverage_level) / n
        q_val = np.clip(q_val, 0, 1)
        return np.quantile(scores_to_use, q_val, method="higher")

    def predict(self, X_test: np.ndarray, coverage_level: float = 0.95, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Produce group-conditional prediction intervals."""
        groups_test = kwargs.get("groups_test")
        if groups_test is None:
            raise ValueError("groups_test must be provided as a keyword argument for MondrianCP prediction.")

        if isinstance(self.base_conformalizer, CQR):
            y_lower_base, y_upper_base = self.model.predict(X_test, coverage_level=coverage_level)
        else:
            y_pred = self.model.predict(X_test)

        lower_bounds = np.zeros(len(X_test))
        upper_bounds = np.zeros(len(X_test))
        
        unique_test_groups = np.unique(groups_test)
        for group in unique_test_groups:
            mask = (groups_test == group)
            q_hat = self.get_group_quantile(group, coverage_level)
            
            if isinstance(self.base_conformalizer, CQR):
                lower_bounds[mask] = y_lower_base[mask] - q_hat
                upper_bounds[mask] = y_upper_base[mask] + q_hat
            else: # Assumes SplitConformal
                lower_bounds[mask] = y_pred[mask] - q_hat
                upper_bounds[mask] = y_pred[mask] + q_hat
        
        return lower_bounds, upper_bounds