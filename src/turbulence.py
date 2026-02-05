"""
Market Turbulence calculation module.

Implements Mahalanobis distance-based turbulence measurement following
Kritzman & Li (2010) methodology.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import TurbulenceConfig

logger = logging.getLogger(__name__)


class TurbulenceCalculator:
    """
    Calculates market turbulence using Mahalanobis distance.

    The turbulence metric measures how unusual the current return vector is
    relative to the historical distribution of returns across assets.

    Turbulence_t = (r_t - μ)' Σ^(-1) (r_t - μ)

    Where:
    - r_t is the return vector at time t
    - μ is the mean return vector from the reference period
    - Σ is the covariance matrix from the reference period
    """

    def __init__(self, config: TurbulenceConfig):
        self.config = config
        self.warning_threshold: Optional[float] = None
        self.extreme_threshold: Optional[float] = None

    def estimate_covariance(
        self,
        returns: pd.DataFrame,
        method: str = 'ledoit_wolf'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate mean and covariance matrix from returns.

        Args:
            returns: DataFrame of returns (rows=dates, cols=assets)
            method: 'ledoit_wolf' for shrinkage, 'sample' for plain sample cov

        Returns:
            Tuple of (mean_vector, covariance_matrix)
        """
        returns_clean = returns.dropna()

        if len(returns_clean) < self.config.min_observations:
            raise ValueError(
                f"Insufficient observations ({len(returns_clean)}) for covariance estimation. "
                f"Need at least {self.config.min_observations}."
            )

        mean_vec = returns_clean.mean().values

        if method == 'ledoit_wolf':
            try:
                lw = LedoitWolf()
                lw.fit(returns_clean.values)
                cov_mat = lw.covariance_
                logger.debug(f"Ledoit-Wolf shrinkage coefficient: {lw.shrinkage_:.4f}")
            except Exception as e:
                logger.warning(f"Ledoit-Wolf failed: {e}. Falling back to sample covariance.")
                cov_mat = returns_clean.cov().values
        else:
            cov_mat = returns_clean.cov().values

        return mean_vec, cov_mat

    def compute_mahalanobis_distance(
        self,
        return_vec: np.ndarray,
        mean_vec: np.ndarray,
        cov_mat: np.ndarray
    ) -> float:
        """
        Compute Mahalanobis distance for a single return vector.

        Args:
            return_vec: Current return vector
            mean_vec: Historical mean return vector
            cov_mat: Historical covariance matrix

        Returns:
            Mahalanobis distance (turbulence score)
        """
        diff = return_vec - mean_vec

        try:
            # Use pseudo-inverse for numerical stability
            cov_inv = np.linalg.pinv(cov_mat)
            distance = np.sqrt(diff @ cov_inv @ diff)
        except np.linalg.LinAlgError as e:
            logger.warning(f"Covariance inversion failed: {e}. Using regularization.")
            # Add small regularization
            reg_cov = cov_mat + np.eye(cov_mat.shape[0]) * 1e-6
            cov_inv = np.linalg.pinv(reg_cov)
            distance = np.sqrt(diff @ cov_inv @ diff)

        return distance

    def compute_rolling_turbulence(
        self,
        returns: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Compute rolling turbulence for each date.

        For each date t, uses the previous `lookback` days to estimate
        the reference distribution, then computes Mahalanobis distance
        of the current day's returns.

        Args:
            returns: DataFrame of returns (rows=dates, cols=assets)
            lookback: Rolling window size (default: from config)

        Returns:
            Series of turbulence values indexed by date
        """
        if lookback is None:
            lookback = self.config.lookback_window

        # Clean returns - drop any columns with too many NaNs
        valid_columns = returns.columns[returns.notna().sum() > lookback]
        returns_clean = returns[valid_columns].copy()

        # Forward fill any remaining NaNs within columns
        returns_clean = returns_clean.ffill().bfill()

        n_days = len(returns_clean)
        turbulence = pd.Series(index=returns_clean.index, dtype=float)

        logger.info(f"Computing turbulence with {lookback}-day lookback over {n_days} days...")

        for i in range(lookback, n_days):
            # Reference window: lookback days ending yesterday
            ref_start = i - lookback
            ref_end = i

            ref_returns = returns_clean.iloc[ref_start:ref_end]
            current_return = returns_clean.iloc[i].values

            # Skip if current return has NaN
            if np.any(np.isnan(current_return)):
                continue

            try:
                mean_vec, cov_mat = self.estimate_covariance(
                    ref_returns,
                    method=self.config.covariance_method
                )
                turbulence.iloc[i] = self.compute_mahalanobis_distance(
                    current_return, mean_vec, cov_mat
                )
            except Exception as e:
                logger.warning(f"Turbulence calculation failed for index {i}: {e}")
                continue

        # Drop NaN values
        turbulence = turbulence.dropna()

        logger.info(f"Computed turbulence for {len(turbulence)} days")

        return turbulence

    def compute_thresholds(
        self,
        turbulence: pd.Series
    ) -> Tuple[float, float]:
        """
        Compute warning and extreme thresholds from historical turbulence.

        Args:
            turbulence: Series of turbulence values

        Returns:
            Tuple of (warning_threshold, extreme_threshold)
        """
        self.warning_threshold = np.percentile(
            turbulence.dropna(),
            self.config.warning_percentile
        )
        self.extreme_threshold = np.percentile(
            turbulence.dropna(),
            self.config.extreme_percentile
        )

        logger.info(
            f"Thresholds computed - Warning (P{self.config.warning_percentile}): "
            f"{self.warning_threshold:.2f}, Extreme (P{self.config.extreme_percentile}): "
            f"{self.extreme_threshold:.2f}"
        )

        return self.warning_threshold, self.extreme_threshold

    def compute_days_elevated(
        self,
        turbulence: pd.Series,
        threshold: Optional[float] = None
    ) -> pd.Series:
        """
        Compute consecutive days turbulence has been elevated.

        Args:
            turbulence: Series of turbulence values
            threshold: Threshold for "elevated" (default: warning threshold)

        Returns:
            Series of consecutive elevated day counts
        """
        if threshold is None:
            threshold = self.warning_threshold

        if threshold is None:
            raise ValueError("Threshold not set. Run compute_thresholds first.")

        # Create boolean series for elevated days
        is_elevated = turbulence >= threshold

        # Compute consecutive counts
        days_elevated = pd.Series(index=turbulence.index, dtype=int)

        count = 0
        for idx in turbulence.index:
            if is_elevated.loc[idx]:
                count += 1
            else:
                count = 0
            days_elevated.loc[idx] = count

        return days_elevated


def compute_ai_turbulence(
    ai_returns: pd.DataFrame,
    config: TurbulenceConfig
) -> pd.Series:
    """
    Compute turbulence for AI/Tech sector basket.

    Args:
        ai_returns: DataFrame of AI sector returns
        config: Configuration object

    Returns:
        Series of AI sector turbulence values
    """
    if ai_returns.empty:
        return pd.Series(dtype=float)

    calculator = TurbulenceCalculator(config)

    try:
        ai_turbulence = calculator.compute_rolling_turbulence(
            ai_returns,
            lookback=min(config.lookback_window, len(ai_returns) - 1)
        )
        return ai_turbulence
    except Exception as e:
        logger.warning(f"Failed to compute AI turbulence: {e}")
        return pd.Series(dtype=float)
