"""
Unit tests for turbulence calculation and Jordi Visser regime framework.

Tests:
- Mahalanobis distance sanity (non-negative, stress > normal)
- Baseline covariance excludes high-VIX periods
- Expanding thresholds change over time
- Visser three-regime classification (NORMAL, GREEN_BAR, STORM)
- Contagion detection via HYG/IEF slope
- Dispersion computation
- Ratio pair computation
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


def _make_spx_series(returns_index, base=450.0, uptrend=True):
    """Generate a synthetic SPX series."""
    n = len(returns_index)
    if uptrend:
        prices = base + np.linspace(0, 50, n)
    else:
        prices = base - np.linspace(0, 50, n)
    return pd.Series(prices, index=returns_index)


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
            baseline_method='expanding',
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
# Baseline / dual-covariance tests
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
        all_cov = returns.cov().values
        baseline_trace = np.trace(cov_mat)
        all_trace = np.trace(all_cov)
        assert baseline_trace <= all_trace, \
            "Calm-period baseline covariance should be smaller than full-sample cov"

    def test_baseline_excludes_high_vix_periods(self):
        """Explicit test that high-VIX days are excluded from baseline."""
        config = TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            baseline_method='calm_period',
            baseline_vix_threshold=25.0,
        )
        returns = _make_returns_with_stress(n_days=300)
        vix = _make_vix_series(returns.index, calm_level=18.0, stress_level=70.0)

        calc = TurbulenceCalculator(config)
        calc.compute_baseline(returns, vix=vix)

        # Compute what baseline WOULD look like with only calm days
        calm_mask = vix < 25.0
        calm_returns = returns.loc[calm_mask]
        calm_cov = calm_returns.cov().values

        # The baseline cov trace should be similar to the calm-only cov trace
        # (not inflated by stress periods)
        baseline_trace = np.trace(calc._baseline_cov)
        calm_trace = np.trace(calm_cov)
        # Allow some tolerance due to Ledoit-Wolf shrinkage
        assert baseline_trace < calm_trace * 2.0, \
            "Baseline should be close to calm-period covariance"

    def test_baseline_anchored_turbulence_rises_during_stress(self):
        """With a calm baseline, turbulence during stress should be significantly higher."""
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
# Expanding thresholds tests
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
# Visser regime classification tests
# ---------------------------------------------------------------------------

class TestVisserRegime:
    """Test Jordi Visser's three-regime framework."""

    @pytest.fixture
    def config(self):
        return TurbulenceConfig(
            lookback_window=60,
            min_observations=30,
            vix_calm_ceiling=25.0,
        )

    @pytest.fixture
    def signal_gen(self, config):
        return SignalGenerator(config)

    def test_green_bar_triggers(self, signal_gen):
        """GREEN_BAR when turbulence > P95, VIX < 25, SPX > 50DMA."""
        regime = signal_gen.classify_regime_visser(
            turbulence_value=10.0,
            spx_close=500.0,
            spx_50dma=480.0,   # SPX above 50DMA
            vix=20.0,          # VIX below 25
            warning_threshold=5.0,  # turbulence above P95
        )
        assert regime == "GREEN_BAR"

    def test_storm_triggers_high_vix(self, signal_gen):
        """STORM when turbulence > P95 and VIX > 25."""
        regime = signal_gen.classify_regime_visser(
            turbulence_value=10.0,
            spx_close=500.0,
            spx_50dma=480.0,
            vix=30.0,          # VIX above 25
            warning_threshold=5.0,
        )
        assert regime == "STORM"

    def test_storm_triggers_spx_below_50dma(self, signal_gen):
        """STORM when turbulence > P95, even if VIX < 25 but SPX below 50DMA."""
        regime = signal_gen.classify_regime_visser(
            turbulence_value=10.0,
            spx_close=460.0,   # SPX below 50DMA
            spx_50dma=480.0,
            vix=20.0,          # VIX low, but SPX below MA
            warning_threshold=5.0,
        )
        assert regime == "STORM"

    def test_normal_when_turbulence_below_threshold(self, signal_gen):
        """NORMAL when turbulence is below P95."""
        regime = signal_gen.classify_regime_visser(
            turbulence_value=3.0,
            spx_close=500.0,
            spx_50dma=480.0,
            vix=20.0,
            warning_threshold=5.0,
        )
        assert regime == "NORMAL"

    def test_regime_series_with_spx_and_vix(self, signal_gen):
        """compute_regime_series uses Visser logic when SPX and VIX are provided."""
        dates = pd.date_range('2025-01-01', periods=5, freq='B')
        turb = pd.Series([3.0, 10.0, 10.0, 10.0, 3.0], index=dates)
        warn = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0], index=dates)
        ext = pd.Series([15.0, 15.0, 15.0, 15.0, 15.0], index=dates)
        # SPX needs to be clearly above the rolling MA for GREEN_BAR
        # With min_periods=1, the 50DMA at idx 1 = mean([480,510]) = 495, so 510 > 495
        spx = pd.Series([480, 510, 510, 460, 510], index=dates, dtype=float)
        vix = pd.Series([18, 20, 30, 30, 18], index=dates, dtype=float)

        regimes = signal_gen.compute_regime_series(
            turb, warn, ext, spx_prices=spx, vix=vix
        )

        assert regimes.iloc[0] == "NORMAL"      # turb < threshold
        assert regimes.iloc[1] == "GREEN_BAR"    # turb high, VIX low, SPX above MA
        assert regimes.iloc[2] == "STORM"        # turb high, VIX > 25
        assert regimes.iloc[3] == "STORM"        # turb high, SPX < 50DMA
        assert regimes.iloc[4] == "NORMAL"       # turb below threshold


# ---------------------------------------------------------------------------
# Contagion detection tests
# ---------------------------------------------------------------------------

class TestContagionDetection:
    """Test HYG/IEF contagion sub-signal."""

    @pytest.fixture
    def config(self):
        return TurbulenceConfig(
            contagion_lookback=20,
            contagion_slope_threshold=0.0,
        )

    @pytest.fixture
    def signal_gen(self, config):
        return SignalGenerator(config)

    def test_contagion_with_declining_ratio(self, signal_gen):
        """Contagion should be True when HYG/IEF ratio is declining."""
        dates = pd.date_range('2025-01-01', periods=50, freq='B')
        # Declining ratio
        ratio = pd.Series(np.linspace(1.0, 0.8, 50), index=dates)

        slope, contagion = signal_gen.check_contagion(ratio, lookback=20)

        # After lookback period, slope should be negative → contagion True
        valid_contagion = contagion.iloc[20:]
        assert valid_contagion.any(), "Declining HYG/IEF should trigger contagion"

    def test_no_contagion_with_rising_ratio(self, signal_gen):
        """No contagion when HYG/IEF ratio is rising."""
        dates = pd.date_range('2025-01-01', periods=50, freq='B')
        # Rising ratio
        ratio = pd.Series(np.linspace(0.8, 1.0, 50), index=dates)

        slope, contagion = signal_gen.check_contagion(ratio, lookback=20)

        valid_contagion = contagion.iloc[20:]
        assert not valid_contagion.any(), "Rising HYG/IEF should not trigger contagion"

    def test_storm_contagion_upgrade(self, signal_gen):
        """STORM days should upgrade to STORM_CONTAGION when contagion is active."""
        dates = pd.date_range('2025-01-01', periods=5, freq='B')
        regime = pd.Series(['NORMAL', 'STORM', 'STORM', 'NORMAL', 'GREEN_BAR'], index=dates)
        contagion = pd.Series([True, True, False, True, True], index=dates)

        adjusted = signal_gen.apply_contagion_to_regime(regime, contagion)

        assert adjusted.iloc[0] == 'NORMAL'          # NORMAL stays NORMAL even with contagion
        assert adjusted.iloc[1] == 'STORM_CONTAGION'  # STORM + contagion → STORM_CONTAGION
        assert adjusted.iloc[2] == 'STORM'            # STORM without contagion stays STORM
        assert adjusted.iloc[3] == 'NORMAL'           # NORMAL stays
        assert adjusted.iloc[4] == 'GREEN_BAR'        # GREEN_BAR stays (contagion only upgrades STORM)


# ---------------------------------------------------------------------------
# Dispersion tests
# ---------------------------------------------------------------------------

class TestDispersion:
    """Test cross-sectional dispersion computation."""

    def test_dispersion_computed(self):
        config = TurbulenceConfig(min_threshold_observations=10)
        signal_gen = SignalGenerator(config)

        dates = pd.date_range('2025-01-01', periods=50, freq='B')
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(50, 5) * 0.01,
            index=dates,
            columns=[f'A{i}' for i in range(5)]
        )

        dispersion, disp_pctile = signal_gen.compute_dispersion(returns)

        assert len(dispersion) == 50
        assert (dispersion >= 0).all(), "Dispersion should be non-negative"
        # Percentile should be valid after min_obs
        valid_pctile = disp_pctile.dropna()
        assert len(valid_pctile) > 0
        assert (valid_pctile >= 0).all() and (valid_pctile <= 100).all()


# ---------------------------------------------------------------------------
# Ratio pair tests
# ---------------------------------------------------------------------------

class TestRatioPairs:
    """Test ratio pair computation in DataFetcher."""

    def test_ratio_pair_columns_added(self):
        from src.data import DataFetcher

        config = TurbulenceConfig(include_ratio_pairs=True)
        fetcher = DataFetcher(config)

        dates = pd.date_range('2025-01-01', periods=50, freq='B')
        prices = pd.DataFrame({
            'SPY': np.linspace(450, 460, 50),
            'QQQ': np.linspace(380, 390, 50),
            'IWM': np.linspace(200, 210, 50),
            'IEF': np.linspace(105, 106, 50),
            'HYG': np.linspace(78, 77, 50),
        }, index=dates)

        returns = fetcher.compute_returns(prices)
        augmented, hyg_ief_ratio = fetcher.add_ratio_pairs(prices, returns)

        assert 'IWM_QQQ_ratio_ret' in augmented.columns
        assert 'HYG_IEF_ratio_ret' in augmented.columns
        assert not hyg_ief_ratio.empty
        assert len(hyg_ief_ratio) == 50


# ---------------------------------------------------------------------------
# VIX override tests
# ---------------------------------------------------------------------------

class TestVIXOverride:
    """Verify VIX circuit breaker behaviour with new regime names."""

    def test_vix_override_flags_high_vix_normal(self):
        """When VIX > 40 and regime is NORMAL, override flag should be True."""
        config = TurbulenceConfig(vix_warning_level=40.0, vix_critical_level=60.0)
        signal_gen = SignalGenerator(config)

        dates = pd.date_range('2025-01-01', periods=5, freq='B')
        regime = pd.Series(['NORMAL', 'NORMAL', 'STORM', 'NORMAL', 'STORM_CONTAGION'],
                           index=dates)
        vix = pd.Series([20, 45, 50, 65, 80], index=dates, dtype=float)

        adj_regime, override = signal_gen.apply_vix_override(regime, vix)

        # idx 1: VIX=45 > 40, NORMAL → flag only
        assert override.iloc[1] == True
        assert adj_regime.iloc[1] == 'NORMAL'

        # idx 3: VIX=65 > 60, NORMAL → forced to STORM
        assert override.iloc[3] == True
        assert adj_regime.iloc[3] == 'STORM'

        # idx 2: already STORM → no override
        assert override.iloc[2] == False

        # idx 4: already STORM_CONTAGION → no override
        assert override.iloc[4] == False

    def test_vix_override_no_false_positives_low_vix(self):
        config = TurbulenceConfig(vix_warning_level=40.0, vix_critical_level=60.0)
        signal_gen = SignalGenerator(config)

        dates = pd.date_range('2025-01-01', periods=3, freq='B')
        regime = pd.Series(['NORMAL', 'NORMAL', 'NORMAL'], index=dates)
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
