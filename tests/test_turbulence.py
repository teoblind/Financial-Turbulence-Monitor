"""
Unit tests for turbulence calculation module.

Tests Mahalanobis distance computation for sanity:
- Non-negative values
- Higher turbulence during known stress periods
- Correct threshold computation
- Baseline-anchored computation does not desensitise during crisis
- Expanding thresholds change over time
- VIX override circuit breaker
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
from src.signals import SignalGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns_with_stress(n_days=300, n_assets=5, stress_start=200, stress_end=221):
    """Generate sample returns with a known stress period."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

    normal_returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=np.eye(n_assets) * 0.0001,
        size=n_days,
    )

    stress_cov = np.ones((n_assets, n_assets)) * 0.0008 + np.eye(n_assets) * 0.0002
    stress_returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=stress_cov,
        size=stress_end - stress_start,
    )
    normal_returns[stress_start:stress_end] = stress_returns

    return pd.DataFrame(
        normal_returns,
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)],
    )


def _make_vix_series(returns_index, calm_level=18.0, stress_level=70.0,
                     stress_start=200, stress_end=221):
    """Generate a synthetic VIX series aligned to a returns index."""
    vix = pd.Series(calm_level, index=returns_index)
    vix.iloc[stress_start:stress_end] = stress_level
    return vix


# ---------------------------------------------------------------------------
# Mahalanobis distance tests
# ---------------------------------------------------------------------------

class TestMahalanobisDistance:
    """Tests for Mahalanobis distance computation."""

    @pytest.fixture
    def config(self):
        return TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            covariance_method='ledoit_wolf',
            baseline_method='expanding',  # use all data for simple baseline
        )

    @pytest.fixture
    def calculator(self, config):
        return TurbulenceCalculator(config)

    @pytest.fixture
    def sample_returns(self):
        return _make_returns_with_stress()

    def test_mahalanobis_non_negative(self, calculator, sample_returns):
        turbulence = calculator.compute_rolling_turbulence(sample_returns)
        assert turbulence.min() >= 0, "Mahalanobis distance should be non-negative"

    def test_mahalanobis_stress_higher(self, calculator, sample_returns):
        turbulence = calculator.compute_rolling_turbulence(sample_returns)

        stress_dates = sample_returns.index[200:221]
        normal_dates = sample_returns.index[100:121]

        stress_turb = turbulence.reindex(stress_dates).dropna()
        normal_turb = turbulence.reindex(normal_dates).dropna()

        if len(stress_turb) > 0 and len(normal_turb) > 0:
            assert stress_turb.mean() > normal_turb.mean(), \
                "Stress period should have higher average turbulence"

    def test_threshold_computation(self, calculator, sample_returns):
        turbulence = calculator.compute_rolling_turbulence(sample_returns)
        warning, extreme = calculator.compute_thresholds(turbulence)

        assert warning < extreme, "Warning threshold should be less than extreme"
        assert warning == pytest.approx(np.percentile(turbulence.dropna(), 95), rel=0.01)
        assert extreme == pytest.approx(np.percentile(turbulence.dropna(), 99), rel=0.01)

    def test_days_elevated_counting(self, calculator, sample_returns):
        turbulence = calculator.compute_rolling_turbulence(sample_returns)
        warning, _ = calculator.compute_thresholds(turbulence)

        days_elevated = calculator.compute_days_elevated(turbulence, warning)

        assert (days_elevated >= 0).all(), "Days elevated should be non-negative"
        assert (days_elevated == days_elevated.astype(int)).all(), "Days elevated should be whole numbers"

    def test_covariance_estimation_methods(self, config, sample_returns):
        calc_lw = TurbulenceCalculator(config)
        config.covariance_method = 'ledoit_wolf'
        turb_lw = calc_lw.compute_rolling_turbulence(sample_returns)

        config_sample = TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            covariance_method='sample',
            baseline_method='expanding',
        )
        calc_sample = TurbulenceCalculator(config_sample)
        turb_sample = calc_sample.compute_rolling_turbulence(sample_returns)

        assert turb_lw.min() >= 0
        assert turb_sample.min() >= 0

        common_idx = turb_lw.index.intersection(turb_sample.index)
        corr = turb_lw.loc[common_idx].corr(turb_sample.loc[common_idx])
        assert corr > 0.8, "Different covariance methods should produce correlated results"


# ---------------------------------------------------------------------------
# Baseline / dual-covariance tests (Fix 1)
# ---------------------------------------------------------------------------

class TestBaselineCovariance:
    """Verify that the calm-period baseline prevents desensitisation."""

    def test_calm_period_baseline_uses_low_vix_days(self):
        """Baseline should be derived from low-VIX days."""
        config = TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            baseline_method='calm_period',
            baseline_vix_threshold=25.0,
        )
        returns = _make_returns_with_stress(n_days=300)
        vix = _make_vix_series(returns.index)

        calc = TurbulenceCalculator(config)
        mean_vec, cov_mat = calc.compute_baseline(returns, vix=vix)

        assert mean_vec is not None
        assert cov_mat.shape == (5, 5)
        # The baseline should NOT include stress period rows
        # We can't directly check which rows, but we can verify cov is
        # smaller than if we used all data (stress inflates it).
        all_cov = returns.cov().values
        baseline_trace = np.trace(cov_mat)
        all_trace = np.trace(all_cov)
        assert baseline_trace <= all_trace, \
            "Calm-period baseline covariance should be smaller than full-sample cov"

    def test_baseline_anchored_turbulence_rises_during_stress(self):
        """
        With a calm baseline, turbulence during stress should be MUCH higher
        than the rolling-window approach (which compresses during extended stress).
        """
        config = TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            baseline_method='calm_period',
            baseline_vix_threshold=25.0,
        )
        returns = _make_returns_with_stress(n_days=300)
        vix = _make_vix_series(returns.index)

        calc = TurbulenceCalculator(config)
        turbulence = calc.compute_rolling_turbulence(returns, vix=vix)

        stress_turb = turbulence.reindex(returns.index[200:221]).dropna()
        normal_turb = turbulence.reindex(returns.index[100:121]).dropna()

        assert len(stress_turb) > 0
        assert len(normal_turb) > 0
        # Stress turbulence should be significantly higher
        assert stress_turb.mean() > normal_turb.mean() * 1.5, \
            "Baseline-anchored stress turbulence should be well above normal"

    def test_first_n_days_baseline(self):
        config = TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            baseline_method='first_n_days',
        )
        returns = _make_returns_with_stress(n_days=300)
        calc = TurbulenceCalculator(config)
        mean_vec, cov_mat = calc.compute_baseline(returns)

        assert mean_vec is not None
        assert cov_mat.shape == (5, 5)


# ---------------------------------------------------------------------------
# Expanding thresholds tests (Fix 2)
# ---------------------------------------------------------------------------

class TestExpandingThresholds:
    """Verify that thresholds update over time."""

    def test_expanding_thresholds_change_over_time(self):
        config = TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            min_threshold_observations=50,
            baseline_method='expanding',
        )
        returns = _make_returns_with_stress(n_days=300)
        calc = TurbulenceCalculator(config)
        turbulence = calc.compute_rolling_turbulence(returns)

        signal_gen = SignalGenerator(config)
        warn_s, ext_s = signal_gen.compute_expanding_thresholds(turbulence)

        valid_warn = warn_s.dropna()
        assert len(valid_warn) > 0, "Should have some valid thresholds"

        # Thresholds should NOT all be the same value
        assert valid_warn.nunique() > 1, "Expanding thresholds should vary over time"

    def test_expanding_thresholds_respects_min_obs(self):
        config = TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            min_threshold_observations=100,
            baseline_method='expanding',
        )
        returns = _make_returns_with_stress(n_days=300)
        calc = TurbulenceCalculator(config)
        turbulence = calc.compute_rolling_turbulence(returns)

        signal_gen = SignalGenerator(config)
        warn_s, _ = signal_gen.compute_expanding_thresholds(turbulence)

        # First 99 values should be NaN
        assert warn_s.iloc[:99].isna().all(), \
            "Thresholds before min_threshold_observations should be NaN"


# ---------------------------------------------------------------------------
# VIX override tests (Fix 3)
# ---------------------------------------------------------------------------

class TestVIXOverride:
    """Verify VIX circuit breaker behaviour."""

    def test_vix_override_flags_high_vix_healthy(self):
        """When VIX > 40 and regime is HEALTHY, override flag should be True."""
        config = TurbulenceConfig(vix_warning_level=40.0, vix_critical_level=60.0)
        signal_gen = SignalGenerator(config)

        dates = pd.date_range('2025-01-01', periods=5, freq='B')
        regime = pd.Series(['HEALTHY', 'HEALTHY', 'ELEVATED', 'HEALTHY', 'CRISIS'],
                           index=dates)
        vix = pd.Series([20, 45, 50, 65, 80], index=dates, dtype=float)

        adj_regime, override = signal_gen.apply_vix_override(regime, vix)

        # idx 1: VIX=45 > 40, HEALTHY → flag only (not forced)
        assert override.iloc[1] == True
        assert adj_regime.iloc[1] == 'HEALTHY'  # stays HEALTHY (warning only)

        # idx 3: VIX=65 > 60, HEALTHY → forced to ELEVATED
        assert override.iloc[3] == True
        assert adj_regime.iloc[3] == 'ELEVATED'

        # idx 2: already ELEVATED, VIX=50 → no override needed
        assert override.iloc[2] == False

        # idx 4: already CRISIS → no override
        assert override.iloc[4] == False

    def test_vix_override_no_false_positives_low_vix(self):
        config = TurbulenceConfig(vix_warning_level=40.0, vix_critical_level=60.0)
        signal_gen = SignalGenerator(config)

        dates = pd.date_range('2025-01-01', periods=3, freq='B')
        regime = pd.Series(['HEALTHY', 'HEALTHY', 'HEALTHY'], index=dates)
        vix = pd.Series([15, 20, 25], index=dates, dtype=float)

        _, override = signal_gen.apply_vix_override(regime, vix)
        assert not override.any(), "No overrides when VIX is low"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_insufficient_data(self):
        config = TurbulenceConfig(
            lookback_window=100,
            min_observations=50,
            baseline_method='expanding',
        )
        calculator = TurbulenceCalculator(config)

        dates = pd.date_range(start='2023-01-01', periods=30, freq='B')
        returns = pd.DataFrame(
            np.random.randn(30, 3) * 0.01,
            index=dates,
            columns=['A', 'B', 'C']
        )

        turbulence = calculator.compute_rolling_turbulence(returns)
        assert len(turbulence.dropna()) == len(turbulence)

    def test_single_asset(self):
        config = TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            baseline_method='expanding',
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
