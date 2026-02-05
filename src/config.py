"""
Configuration module for Market Turbulence Monitoring System.

Contains all configurable parameters for data fetching, turbulence calculation,
divergence detection, and visualization.
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

    # === Threshold Percentiles ===
    warning_percentile: float = 95.0
    extreme_percentile: float = 99.0

    # === Divergence Detection ===
    # Rule for determining "SPX rising": 'ma50' or 'ma20_slope'
    # 'ma50': SPX close > 50-day moving average
    # 'ma20_slope': 20-day MA has positive slope
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

    # Status box colors
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
    # This fixes the issue where dates are cached at module import time
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
