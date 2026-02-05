"""
Unit tests for turbulence calculation module.

Tests Mahalanobis distance computation for sanity:
- Non-negative values
- Higher turbulence during known stress periods
- Correct threshold computation
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TurbulenceConfig
from src.turbulence import TurbulenceCalculator


class TestMahalanobisDistance:
    """Tests for Mahalanobis distance computation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            covariance_method='ledoit_wolf'
        )

    @pytest.fixture
    def calculator(self, config):
        """Create turbulence calculator instance."""
        return TurbulenceCalculator(config)

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns data for testing."""
        np.random.seed(42)
        n_days = 300
        n_assets = 5

        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

        # Normal returns
        normal_returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=np.eye(n_assets) * 0.0001,  # Low correlation
            size=n_days
        )

        # Add stress period (days 200-220) with higher volatility and correlation
        stress_cov = np.ones((n_assets, n_assets)) * 0.0008 + np.eye(n_assets) * 0.0002
        stress_returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=stress_cov,
            size=21
        )
        normal_returns[200:221] = stress_returns

        returns_df = pd.DataFrame(
            normal_returns,
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )

        return returns_df

    def test_mahalanobis_non_negative(self, calculator, sample_returns):
        """Test that Mahalanobis distance is always non-negative."""
        turbulence = calculator.compute_rolling_turbulence(sample_returns)

        assert turbulence.min() >= 0, "Mahalanobis distance should be non-negative"

    def test_mahalanobis_stress_higher(self, calculator, sample_returns):
        """Test that turbulence is higher during stress periods."""
        turbulence = calculator.compute_rolling_turbulence(sample_returns)

        # Get stress period (days 200-220)
        stress_dates = sample_returns.index[200:221]
        normal_dates = sample_returns.index[100:121]  # Normal period

        # Filter turbulence for available dates
        stress_turb = turbulence.reindex(stress_dates).dropna()
        normal_turb = turbulence.reindex(normal_dates).dropna()

        if len(stress_turb) > 0 and len(normal_turb) > 0:
            assert stress_turb.mean() > normal_turb.mean(), \
                "Stress period should have higher average turbulence"

    def test_threshold_computation(self, calculator, sample_returns):
        """Test that thresholds are computed correctly."""
        turbulence = calculator.compute_rolling_turbulence(sample_returns)
        warning, extreme = calculator.compute_thresholds(turbulence)

        # Warning should be at 95th percentile, extreme at 99th
        assert warning < extreme, "Warning threshold should be less than extreme"
        assert warning == pytest.approx(np.percentile(turbulence.dropna(), 95), rel=0.01)
        assert extreme == pytest.approx(np.percentile(turbulence.dropna(), 99), rel=0.01)

    def test_days_elevated_counting(self, calculator, sample_returns):
        """Test consecutive elevated days counting."""
        turbulence = calculator.compute_rolling_turbulence(sample_returns)
        warning, _ = calculator.compute_thresholds(turbulence)

        days_elevated = calculator.compute_days_elevated(turbulence, warning)

        # Days elevated should be non-negative integers
        assert (days_elevated >= 0).all(), "Days elevated should be non-negative"
        # Check that all values are whole numbers (can be float representation of int)
        assert (days_elevated == days_elevated.astype(int)).all(), "Days elevated should be whole numbers"

    def test_covariance_estimation_methods(self, config, sample_returns):
        """Test that both covariance methods work."""
        # Ledoit-Wolf
        calc_lw = TurbulenceCalculator(config)
        config.covariance_method = 'ledoit_wolf'
        turb_lw = calc_lw.compute_rolling_turbulence(sample_returns)

        # Sample covariance
        config_sample = TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            covariance_method='sample'
        )
        calc_sample = TurbulenceCalculator(config_sample)
        turb_sample = calc_sample.compute_rolling_turbulence(sample_returns)

        # Both should produce valid (non-negative) results
        assert turb_lw.min() >= 0
        assert turb_sample.min() >= 0

        # They should be correlated but not identical
        common_idx = turb_lw.index.intersection(turb_sample.index)
        corr = turb_lw.loc[common_idx].corr(turb_sample.loc[common_idx])
        assert corr > 0.8, "Different covariance methods should produce correlated results"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        config = TurbulenceConfig(
            lookback_window=100,
            min_observations=50
        )
        calculator = TurbulenceCalculator(config)

        # Only 30 days of data - less than min_observations
        dates = pd.date_range(start='2023-01-01', periods=30, freq='B')
        returns = pd.DataFrame(
            np.random.randn(30, 3) * 0.01,
            index=dates,
            columns=['A', 'B', 'C']
        )

        # Should handle gracefully (may return empty or partial series)
        turbulence = calculator.compute_rolling_turbulence(returns)
        # Either empty or all valid (no NaN after dropna)
        assert len(turbulence.dropna()) == len(turbulence)

    def test_single_asset(self):
        """Test with single asset (edge case)."""
        config = TurbulenceConfig(
            lookback_window=60,
            min_observations=30
        )
        calculator = TurbulenceCalculator(config)

        dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        returns = pd.DataFrame(
            np.random.randn(100, 1) * 0.01,
            index=dates,
            columns=['SingleAsset']
        )

        turbulence = calculator.compute_rolling_turbulence(returns)
        assert turbulence.min() >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
