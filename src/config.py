"""
Configuration module for Market Turbulence Monitoring System.

Contains all configurable parameters for data fetching, turbulence calculation,
regime classification, contagion detection, and visualization.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Literal


@dataclass
class TurbulenceConfig:
    """Configuration for the Market Turbulence Monitoring System."""

    # === Date Range ===
    # Default: last 3 years of data
    start_date: str = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
    end_date: str = datetime.now().strftime("%Y-%m-%d")

    # === Ticker Configuration ===
    # Primary index proxy
    spx_ticker: str = "SPY"  # Use SPY as proxy for S&P 500
    vix_ticker: str = "^VIX"

    # Multi-asset basket for turbulence calculation
    turbulence_tickers: List[str] = field(default_factory=lambda: [
        # Equities
        "SPY",   # S&P 500
        "QQQ",   # Nasdaq 100
        "IWM",   # Russell 2000
        "EFA",   # Developed Markets ex-US
        "EEM",   # Emerging Markets
        # Rates
        "TLT",   # Long-term Treasuries
        "IEF",   # Intermediate Treasuries
        # Credit
        "HYG",   # High Yield Corporate
        # Commodities
        "GLD",   # Gold
        "USO",   # Oil (alternative to DBC)
        # Dollar
        "UUP",   # US Dollar Index
    ])

    # AI/Tech basket for sector context
    ai_tickers: List[str] = field(default_factory=lambda: [
        "NVDA",  # NVIDIA
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet
        "AMZN",  # Amazon
        "SMH",   # Semiconductor ETF
    ])

    # === Turbulence Calculation ===
    # Rolling window for covariance estimation (trading days)
    lookback_window: int = 252  # 1 year

    # Minimum observations required for covariance calculation
    min_observations: int = 60

    # Covariance estimation method: 'ledoit_wolf' or 'sample'
    covariance_method: Literal['ledoit_wolf', 'sample'] = 'ledoit_wolf'

    # === Baseline Covariance ===
    # Method: "calm_period", "first_n_days", "expanding"
    baseline_method: str = "calm_period"

    # VIX ceiling used to identify "calm" days for the baseline covariance
    baseline_vix_threshold: float = 25.0

    # Minimum days needed for baseline computation
    min_baseline_days: int = 252

    # === Regime Classification (Jordi Visser's framework) ===
    # VIX threshold separating GREEN_BAR from STORM
    vix_calm_ceiling: float = 25.0

    # === Contagion Detection ===
    # Lookback window for HYG/IEF slope computation (trading days)
    contagion_lookback: int = 20

    # Slope threshold: negative slope = contagion
    contagion_slope_threshold: float = 0.0

    # === Ratio Pairs ===
    # Whether to include IWM/QQQ and HYG/IEF ratio returns in the covariance matrix
    include_ratio_pairs: bool = True

    # === VIX Override Settings (backward compat) ===
    vix_warning_level: float = 40.0
    vix_critical_level: float = 60.0

    # === Threshold Percentiles ===
    warning_percentile: float = 95.0
    extreme_percentile: float = 99.0

    # Minimum observations before computing expanding thresholds
    min_threshold_observations: int = 126

    # === Divergence Detection ===
    divergence_rule: Literal['ma50', 'ma20_slope'] = 'ma50'

    # Moving average periods
    ma_short: int = 20
    ma_long: int = 50

    # === Data Source ===
    use_simulated: bool = False  # Use simulated data instead of yfinance

    # === Output Configuration ===
    output_dir: str = "output"
    dashboard_filename: str = "dashboard.png"
    features_filename: str = "features.csv"

    # === Visualization ===
    figure_width: int = 16
    figure_height: int = 12
    dpi: int = 150

    # Color scheme
    color_spx: str = "#1f77b4"  # Blue
    color_turbulence: str = "#d62728"  # Red
    color_warning: str = "#ff7f0e"  # Orange
    color_extreme: str = "#9467bd"  # Purple
    color_divergence: str = "#2ca02c"  # Green (with alpha)
    color_ma50: str = "#7f7f7f"  # Gray

    # Regime colors (Jordi's framework)
    color_normal: str = "#28a745"       # Green
    color_green_bar: str = "#ffc107"    # Amber/Yellow
    color_storm: str = "#fd7e14"        # Orange
    color_storm_contagion: str = "#dc3545"  # Red

    # Status box colors (backward compat aliases)
    status_healthy: str = "#28a745"  # Green
    status_elevated: str = "#ffc107"  # Yellow/Amber
    status_crisis: str = "#dc3545"  # Red


# Default configuration instance
DEFAULT_CONFIG = TurbulenceConfig()


def get_config(**kwargs) -> TurbulenceConfig:
    """
    Create a configuration with optional overrides.

    Args:
        **kwargs: Any TurbulenceConfig field to override

    Returns:
        TurbulenceConfig instance with applied overrides
    """
    config = TurbulenceConfig()

    # Always compute dates dynamically if not explicitly provided
    if 'end_date' not in kwargs:
        config.end_date = datetime.now().strftime("%Y-%m-%d")
    if 'start_date' not in kwargs:
        config.start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    return config
