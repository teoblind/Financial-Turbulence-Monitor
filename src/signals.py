"""
Signal generation module.

Handles divergence detection, regime classification, and status interpretations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from .config import TurbulenceConfig

logger = logging.getLogger(__name__)


# Regime definitions
Regime = Literal['HEALTHY', 'ELEVATED', 'CRISIS']


@dataclass
class MarketStatus:
    """Container for current market status information."""
    date: datetime
    regime: Regime
    turbulence: float
    turbulence_percentile: float
    days_elevated: int
    spx_level: float
    spx_vs_ma50: str
    spx_ma50_ratio: float
    vix_level: float
    divergence_active: bool
    ai_turbulence: Optional[float] = None
    warning_threshold: float = 0.0
    extreme_threshold: float = 0.0


class SignalGenerator:
    """Generates trading signals and regime classifications."""

    def __init__(self, config: TurbulenceConfig):
        self.config = config

    def detect_spx_rising(
        self,
        spx_prices: pd.Series,
        rule: Optional[str] = None
    ) -> pd.Series:
        """
        Detect when SPX is "rising" based on configured rule.

        Args:
            spx_prices: SPX/SPY price series
            rule: 'ma50' or 'ma20_slope' (default from config)

        Returns:
            Boolean series indicating rising market
        """
        if rule is None:
            rule = self.config.divergence_rule

        if rule == 'ma50':
            # SPX close > 50-day moving average
            ma50 = spx_prices.rolling(window=self.config.ma_long).mean()
            is_rising = spx_prices > ma50
            logger.info("Using MA50 rule: SPX > 50-day MA")

        elif rule == 'ma20_slope':
            # 20-day MA has positive slope
            ma20 = spx_prices.rolling(window=self.config.ma_short).mean()
            # Slope: compare current MA to MA from 5 days ago
            ma20_slope = ma20 - ma20.shift(5)
            is_rising = ma20_slope > 0
            logger.info("Using MA20 slope rule: 20-day MA has positive slope")

        else:
            raise ValueError(f"Unknown divergence rule: {rule}")

        return is_rising.fillna(False)

    def detect_divergence(
        self,
        turbulence: pd.Series,
        spx_prices: pd.Series,
        warning_threshold: float,
        rule: Optional[str] = None
    ) -> pd.Series:
        """
        Detect divergence: high turbulence while SPX is rising.

        This is the key warning signal - markets appear calm (rising)
        but underlying stress is elevated.

        Args:
            turbulence: Turbulence series
            spx_prices: SPX price series
            warning_threshold: Threshold for elevated turbulence
            rule: Divergence detection rule

        Returns:
            Boolean series indicating divergence
        """
        is_rising = self.detect_spx_rising(spx_prices, rule)
        is_elevated = turbulence >= warning_threshold

        # Align indices
        common_idx = turbulence.index.intersection(is_rising.index)
        divergence = is_elevated.loc[common_idx] & is_rising.loc[common_idx]

        logger.info(f"Divergence detected on {divergence.sum()} days")

        return divergence

    def classify_regime(
        self,
        turbulence_value: float,
        warning_threshold: float,
        extreme_threshold: float
    ) -> Regime:
        """
        Classify current market regime based on turbulence level.

        Args:
            turbulence_value: Current turbulence value
            warning_threshold: 95th percentile threshold
            extreme_threshold: 99th percentile threshold

        Returns:
            Regime classification
        """
        if turbulence_value >= extreme_threshold:
            return 'CRISIS'
        elif turbulence_value >= warning_threshold:
            return 'ELEVATED'
        else:
            return 'HEALTHY'

    def compute_regime_series(
        self,
        turbulence: pd.Series,
        warning_threshold: float,
        extreme_threshold: float
    ) -> pd.Series:
        """
        Compute regime classification for all dates.

        Args:
            turbulence: Turbulence series
            warning_threshold: 95th percentile threshold
            extreme_threshold: 99th percentile threshold

        Returns:
            Series of regime classifications
        """
        def classify(t):
            if pd.isna(t):
                return None
            return self.classify_regime(t, warning_threshold, extreme_threshold)

        return turbulence.apply(classify)

    def get_interpretation(self, regime: Regime) -> List[str]:
        """
        Get interpretation text for current regime.

        Args:
            regime: Current market regime

        Returns:
            List of interpretation lines
        """
        interpretations = {
            'HEALTHY': [
                "Market showing normal stress levels",
                "Cross-asset correlations stable",
                "No concerning correlation breakdowns",
                "Risk-on environment favorable"
            ],
            'ELEVATED': [
                "Cross-asset correlations showing instability",
                "Hidden stress building beneath surface",
                "Watch for rapid regime shifts",
                "Consider reducing leverage and risk exposure"
            ],
            'CRISIS': [
                "Systemic stress significantly elevated",
                "Correlation regime shift likely underway",
                "Historical relationships breaking down",
                "Risk management action urgently required"
            ]
        }
        return interpretations.get(regime, [])

    def get_recommended_actions(self, regime: Regime) -> List[str]:
        """
        Get recommended actions for current regime.

        Args:
            regime: Current market regime

        Returns:
            List of recommended action lines
        """
        actions = {
            'HEALTHY': [
                "Maintain current positioning",
                "Monitor for regime transitions",
                "Standard risk protocols apply"
            ],
            'ELEVATED': [
                "Review portfolio correlations",
                "Consider de-risking by 10-25%",
                "Increase hedge ratios",
                "Prepare contingency plans"
            ],
            'CRISIS': [
                "Reduce gross exposure significantly",
                "Maximize defensive positioning",
                "Activate crisis playbook",
                "Daily monitoring required"
            ]
        }
        return actions.get(regime, [])

    def generate_status(
        self,
        turbulence: pd.Series,
        spx_prices: pd.Series,
        vix: pd.Series,
        warning_threshold: float,
        extreme_threshold: float,
        days_elevated_series: pd.Series,
        divergence: pd.Series,
        ai_turbulence: Optional[pd.Series] = None
    ) -> MarketStatus:
        """
        Generate complete market status for the latest date.

        Args:
            turbulence: Turbulence series
            spx_prices: SPX price series
            vix: VIX series
            warning_threshold: 95th percentile threshold
            extreme_threshold: 99th percentile threshold
            days_elevated_series: Consecutive elevated days series
            divergence: Divergence boolean series
            ai_turbulence: Optional AI sector turbulence series

        Returns:
            MarketStatus object with all current metrics
        """
        # Get latest values
        latest_date = turbulence.index[-1]
        latest_turbulence = turbulence.iloc[-1]

        # Compute percentile rank
        turbulence_pct = (turbulence <= latest_turbulence).mean() * 100

        # SPX metrics
        latest_spx = spx_prices.loc[latest_date] if latest_date in spx_prices.index else spx_prices.iloc[-1]
        ma50 = spx_prices.rolling(window=self.config.ma_long).mean()
        latest_ma50 = ma50.loc[latest_date] if latest_date in ma50.index else ma50.iloc[-1]
        spx_ratio = latest_spx / latest_ma50 if latest_ma50 > 0 else 1.0
        spx_vs_ma50 = "ABOVE" if latest_spx > latest_ma50 else "BELOW"

        # VIX
        latest_vix = vix.loc[latest_date] if latest_date in vix.index else (vix.iloc[-1] if not vix.empty else np.nan)

        # Days elevated
        days_elevated = int(days_elevated_series.iloc[-1]) if not days_elevated_series.empty else 0

        # Divergence
        div_active = bool(divergence.iloc[-1]) if not divergence.empty and latest_date in divergence.index else False

        # Regime
        regime = self.classify_regime(latest_turbulence, warning_threshold, extreme_threshold)

        # AI turbulence
        ai_turb = None
        if ai_turbulence is not None and not ai_turbulence.empty:
            if latest_date in ai_turbulence.index:
                ai_turb = ai_turbulence.loc[latest_date]
            elif len(ai_turbulence) > 0:
                ai_turb = ai_turbulence.iloc[-1]

        return MarketStatus(
            date=latest_date.to_pydatetime() if hasattr(latest_date, 'to_pydatetime') else latest_date,
            regime=regime,
            turbulence=float(latest_turbulence),
            turbulence_percentile=float(turbulence_pct),
            days_elevated=days_elevated,
            spx_level=float(latest_spx),
            spx_vs_ma50=spx_vs_ma50,
            spx_ma50_ratio=float(spx_ratio),
            vix_level=float(latest_vix) if not np.isnan(latest_vix) else 0.0,
            divergence_active=div_active,
            ai_turbulence=float(ai_turb) if ai_turb is not None else None,
            warning_threshold=float(warning_threshold),
            extreme_threshold=float(extreme_threshold)
        )


def format_status_text(status: MarketStatus, signal_gen: SignalGenerator) -> str:
    """
    Format the status into a printable text panel.

    Args:
        status: MarketStatus object
        signal_gen: SignalGenerator for interpretations

    Returns:
        Formatted status text string
    """
    lines = []

    # Header
    date_str = status.date.strftime("%Y-%m-%d")
    lines.append("=" * 60)
    lines.append(f"  CURRENT IMMUNE SYSTEM STATUS — {date_str}")
    lines.append("=" * 60)
    lines.append("")

    # Warning level with color indicator
    level_indicator = {
        'HEALTHY': "🟢 HEALTHY",
        'ELEVATED': "🟡 ELEVATED",
        'CRISIS': "🔴 CRISIS"
    }
    lines.append(f"  WARNING LEVEL: {level_indicator.get(status.regime, status.regime)}")
    lines.append("")

    # Current metrics
    lines.append("  CURRENT METRICS:")
    lines.append(f"    • Market Turbulence: {status.turbulence:.1f} (P{status.turbulence_percentile:.0f})")
    lines.append(f"    • Days Elevated: {status.days_elevated}")
    lines.append(f"    • SPX Level: {status.spx_level:,.2f}")
    lines.append(f"    • SPX vs 50-day MA: {status.spx_vs_ma50} ({status.spx_ma50_ratio:.2%})")
    lines.append(f"    • VIX Level: {status.vix_level:.1f}")
    lines.append(f"    • Divergence Active: {'YES ⚠️' if status.divergence_active else 'NO'}")
    lines.append("")

    # AI Sector Context
    lines.append("  AI SECTOR CONTEXT:")
    if status.ai_turbulence is not None:
        lines.append(f"    • AI Basket Turbulence: {status.ai_turbulence:.1f}")
    else:
        lines.append("    • AI Basket Turbulence: N/A")
    lines.append("")

    # Interpretation
    lines.append("  INTERPRETATION:")
    for interp in signal_gen.get_interpretation(status.regime):
        lines.append(f"    • {interp}")
    lines.append("")

    # Recommended Actions
    lines.append("  RECOMMENDED ACTIONS:")
    for action in signal_gen.get_recommended_actions(status.regime):
        lines.append(f"    • {action}")
    lines.append("")

    # Thresholds
    lines.append("  THRESHOLDS:")
    lines.append(f"    • Warning (P95): {status.warning_threshold:.1f}")
    lines.append(f"    • Extreme (P99): {status.extreme_threshold:.1f}")

    lines.append("=" * 60)

    return "\n".join(lines)
