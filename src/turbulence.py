"""
Market Turbulence calculation module.

Implements squared Mahalanobis distance (d²) turbulence measurement following
Kritzman & Li (2010) methodology, enhanced with a dual-covariance approach
inspired by Jordi Visser's turbulence model.

Key improvement: turbulence is measured against a CALM-PERIOD baseline
covariance, not a rolling window that drifts into crisis data.  This
prevents the desensitisation bug where extreme moves look "normal"
because the rolling covariance has absorbed months of elevated vol.
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
    Calculates market turbulence using squared Mahalanobis distance (d²).

    The turbulence metric measures how unusual the current return vector is
    relative to a *calm-period baseline* distribution of returns across assets.

    Turbulence_t = (r_t - mu_base)' Sigma_base^{-1} (r_t - mu_base)

    This is the squared form (d²) as defined in Kritzman & Li (2010) — no
    square root. The squared form amplifies tail events (a 2x distance move
    becomes 4x in the score), producing the dramatic crisis spikes seen in
    Jordi Visser's model.

    Where:
    - r_t is the return vector at time t
    - mu_base is the mean return vector from the calm baseline period
    - Sigma_base is the covariance matrix from the calm baseline period
    """

    def __init__(self, config: TurbulenceConfig):
        self.config = config
        self.warning_threshold: Optional[float] = None
        self.extreme_threshold: Optional[float] = None
        # Baseline stats (computed once, used for all subsequent scoring)
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_cov: Optional[np.ndarray] = None
        self._baseline_cov_inv: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Covariance estimation helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Baseline computation
    # ------------------------------------------------------------------

    def compute_baseline(
        self,
        returns: pd.DataFrame,
        vix: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calm-period baseline mean and covariance.

        The baseline is the *anchor* the model uses so that when markets
        enter stress the Mahalanobis distance actually increases rather
        than staying flat because the rolling covariance absorbed the crisis.

        Strategy (``baseline_method`` in config):
        - "calm_period": use only days where VIX < baseline_vix_threshold.
          Falls back to lowest-VIX quartile if fewer than min_observations
          days qualify.
        - "first_n_days": use the first ``lookback_window`` days of data.
        - "expanding": use all data (original behaviour, kept for back-compat).

        Returns:
            Tuple of (baseline_mean, baseline_cov)
        """
        method = getattr(self.config, 'baseline_method', 'calm_period')

        if method == 'calm_period' and vix is not None and not vix.empty:
            # Align VIX to returns index
            vix_aligned = vix.reindex(returns.index)

            # Primary filter: VIX < threshold
            threshold = getattr(self.config, 'baseline_vix_threshold', 25.0)
            calm_mask = vix_aligned < threshold
            calm_returns = returns.loc[calm_mask.fillna(False)]

            # Fallback: if not enough calm days, use lowest-VIX quartile
            min_baseline = getattr(self.config, 'min_baseline_days', self.config.min_observations)
            if len(calm_returns) < min_baseline:
                logger.warning(
                    f"Only {len(calm_returns)} days with VIX < {threshold}. "
                    "Falling back to lowest-VIX quartile."
                )
                vix_valid = vix_aligned.dropna()
                if len(vix_valid) > 0:
                    q25 = vix_valid.quantile(0.25)
                    calm_mask = vix_aligned <= q25
                    calm_returns = returns.loc[calm_mask.fillna(False)]

            # Second fallback: first N days
            if len(calm_returns) < min_baseline:
                logger.warning(
                    "Lowest-VIX quartile still insufficient. "
                    "Using first lookback_window days as baseline."
                )
                calm_returns = returns.iloc[:self.config.lookback_window]

        elif method == 'first_n_days':
            calm_returns = returns.iloc[:self.config.lookback_window]
        else:
            # "expanding" or no VIX available — use all data
            calm_returns = returns

        logger.info(
            f"Baseline computed from {len(calm_returns)} calm-period days "
            f"(method={method})"
        )

        mean_vec, cov_mat = self.estimate_covariance(
            calm_returns, method=self.config.covariance_method
        )

        # Cache for re-use
        self._baseline_mean = mean_vec
        self._baseline_cov = cov_mat
        self._baseline_cov_inv = np.linalg.pinv(cov_mat)

        return mean_vec, cov_mat

    # ------------------------------------------------------------------
    # Mahalanobis distance
    # ------------------------------------------------------------------

    def compute_mahalanobis_distance(
        self,
        return_vec: np.ndarray,
        mean_vec: np.ndarray,
        cov_mat: np.ndarray
    ) -> float:
        """
        Compute squared Mahalanobis distance (d²) for a single return vector.

        Args:
            return_vec: Current return vector
            mean_vec: Historical mean return vector
            cov_mat: Historical covariance matrix

        Returns:
            Squared Mahalanobis distance (d²) turbulence score
        """
        diff = return_vec - mean_vec

        try:
            # Use pseudo-inverse for numerical stability
            cov_inv = np.linalg.pinv(cov_mat)
            distance = diff @ cov_inv @ diff
        except np.linalg.LinAlgError as e:
            logger.warning(f"Covariance inversion failed: {e}. Using regularization.")
            # Add small regularization
            reg_cov = cov_mat + np.eye(cov_mat.shape[0]) * 1e-6
            cov_inv = np.linalg.pinv(reg_cov)
            distance = diff @ cov_inv @ diff

        return distance

    # ------------------------------------------------------------------
    # Rolling turbulence (baseline-anchored)
    # ------------------------------------------------------------------

    def compute_rolling_turbulence(
        self,
        returns: pd.DataFrame,
        lookback: Optional[int] = None,
        vix: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute turbulence for each date, scored against the calm-period baseline.

        Unlike the previous implementation that re-estimated covariance from a
        rolling window (which drifts into crisis data), this version:
        1. Computes a fixed baseline from calm-period data.
        2. Measures every day's returns against that fixed baseline.

        Args:
            returns: DataFrame of returns (rows=dates, cols=assets)
            lookback: Minimum warmup period before scoring starts (default: from config)
            vix: Optional VIX series used for calm-period identification

        Returns:
            Series of turbulence values indexed by date
        """
        if lookback is None:
            lookback = self.config.lookback_window

        # Clean returns — drop columns with too many NaNs
        min_valid = min(lookback, len(returns))
        valid_columns = returns.columns[returns.notna().sum() > min_valid]
        if len(valid_columns) == 0:
            # Not enough data for any column — return empty
            logger.warning("No columns have enough valid observations. Returning empty turbulence.")
            return pd.Series(dtype=float)
        returns_clean = returns[valid_columns].copy()
        returns_clean = returns_clean.ffill().bfill()

        # Convert decimal returns to percentage for Kritzman & Li scale
        returns_clean = returns_clean * 100

        n_days = len(returns_clean)
        turbulence = pd.Series(index=returns_clean.index, dtype=float)

        # --- Compute baseline covariance (fixed anchor) ---
        if self._baseline_mean is None or self._baseline_cov is None:
            try:
                self.compute_baseline(returns_clean, vix=vix)
            except ValueError as e:
                logger.warning(f"Cannot compute baseline: {e}. Returning empty turbulence.")
                return pd.Series(dtype=float)

        baseline_mean = self._baseline_mean
        baseline_cov_inv = self._baseline_cov_inv

        logger.info(
            f"Computing baseline-anchored turbulence over {n_days} days "
            f"(warmup={lookback})..."
        )

        # Score every day from ``lookback`` onward against the baseline
        for i in range(lookback, n_days):
            current_return = returns_clean.iloc[i].values

            if np.any(np.isnan(current_return)):
                continue

            try:
                diff = current_return - baseline_mean
                dist = diff @ baseline_cov_inv @ diff
                turbulence.iloc[i] = dist
            except Exception as e:
                logger.warning(f"Turbulence calculation failed for index {i}: {e}")
                continue

        turbulence = turbulence.dropna()
        logger.info(f"Computed turbulence for {len(turbulence)} days")

        return turbulence

    # ------------------------------------------------------------------
    # Rolling-window turbulence (old approach — for comparison only)
    # ------------------------------------------------------------------

    def compute_rolling_turbulence_naive(
        self,
        returns: pd.DataFrame,
        lookback: Optional[int] = None,
    ) -> pd.Series:
        """
        Compute turbulence using a naive rolling-window covariance.

        This is the OLD approach that suffers from desensitisation: as
        stress persists, the rolling window absorbs crisis volatility and
        the Mahalanobis distance shrinks back toward "normal."

        Kept solely so the dashboard can overlay both traces and make
        the desensitisation problem visible.

        Args:
            returns: DataFrame of returns (rows=dates, cols=assets)
            lookback: Rolling window size (default: from config)

        Returns:
            Series of turbulence values indexed by date
        """
        if lookback is None:
            lookback = self.config.lookback_window

        # Clean returns
        min_valid = min(lookback, len(returns))
        valid_columns = returns.columns[returns.notna().sum() > min_valid]
        if len(valid_columns) == 0:
            return pd.Series(dtype=float)
        returns_clean = returns[valid_columns].copy()
        returns_clean = returns_clean.ffill().bfill()

        # Convert decimal returns to percentage for Kritzman & Li scale
        returns_clean = returns_clean * 100

        n_days = len(returns_clean)
        turbulence = pd.Series(index=returns_clean.index, dtype=float)

        logger.info(
            f"Computing ROLLING-WINDOW turbulence (naive) over {n_days} days "
            f"(lookback={lookback})..."
        )

        for i in range(lookback, n_days):
            window = returns_clean.iloc[i - lookback:i]
            current_return = returns_clean.iloc[i].values

            if np.any(np.isnan(current_return)):
                continue

            try:
                mean_vec = window.mean().values
                if self.config.covariance_method == 'ledoit_wolf':
                    try:
                        lw = LedoitWolf()
                        lw.fit(window.values)
                        cov_mat = lw.covariance_
                    except Exception:
                        cov_mat = window.cov().values
                else:
                    cov_mat = window.cov().values

                cov_inv = np.linalg.pinv(cov_mat)
                diff = current_return - mean_vec
                dist = diff @ cov_inv @ diff
                turbulence.iloc[i] = dist
            except Exception as e:
                logger.warning(f"Rolling turbulence failed at index {i}: {e}")
                continue

        turbulence = turbulence.dropna()
        logger.info(f"Computed rolling-window turbulence for {len(turbulence)} days")

        return turbulence

    # ------------------------------------------------------------------
    # Thresholds (static — kept for backward compat; rolling in signals.py)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Days elevated
    # ------------------------------------------------------------------

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

        is_elevated = turbulence >= threshold
        days_elevated = pd.Series(index=turbulence.index, dtype=int)

        count = 0
        for idx in turbulence.index:
            if is_elevated.loc[idx]:
                count += 1
            else:
                count = 0
            days_elevated.loc[idx] = count

        return days_elevated


def compute_sector_turbulence(
    sector_returns: pd.DataFrame,
    config: TurbulenceConfig,
    label: str = "sector",
) -> pd.Series:
    """
    Compute turbulence for an arbitrary sector basket.

    Args:
        sector_returns: DataFrame of sector returns
        config: Configuration object
        label: Human-readable label for log messages

    Returns:
        Series of sector turbulence values
    """
    if sector_returns.empty:
        return pd.Series(dtype=float)

    calculator = TurbulenceCalculator(config)

    try:
        turbulence = calculator.compute_rolling_turbulence(
            sector_returns,
            lookback=min(config.lookback_window, len(sector_returns) - 1)
        )
        return turbulence
    except Exception as e:
        logger.warning(f"Failed to compute {label} turbulence: {e}")
        return pd.Series(dtype=float)


# Backward-compat alias
compute_ai_turbulence = compute_sector_turbulence
