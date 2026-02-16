"""
Signal generation module.

Handles divergence detection, regime classification, VIX override circuit
breaker, and expanding-window threshold computation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional, Tuple

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
    # New fields for VIX override
    vix_override: bool = False
    model_regime: Optional[str] = None  # regime before override


class SignalGenerator:
    """Generates trading signals and regime classifications."""

    def __init__(self, config: TurbulenceConfig):
        self.config = config

    # ------------------------------------------------------------------
    # SPX / divergence detection (unchanged)
    # ------------------------------------------------------------------

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
            ma50 = spx_prices.rolling(window=self.config.ma_long).mean()
            is_rising = spx_prices > ma50
            logger.info("Using MA50 rule: SPX > 50-day MA")

        elif rule == 'ma20_slope':
            ma20 = spx_prices.rolling(window=self.config.ma_short).mean()
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

        common_idx = turbulence.index.intersection(is_rising.index)
        divergence = is_elevated.loc[common_idx] & is_rising.loc[common_idx]

        logger.info(f"Divergence detected on {divergence.sum()} days")
        return divergence

    # ------------------------------------------------------------------
    # Expanding-window rolling thresholds (Fix 2)
    # ------------------------------------------------------------------

    def compute_expanding_thresholds(
        self,
        turbulence: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute rolling P95 / P99 thresholds using an expanding window.

        Instead of a single static threshold computed once from the full
        sample, this computes thresholds at each point in time using only
        data available up to that date.  A minimum of
        ``config.min_threshold_observations`` days is required before the
        first threshold is emitted.

        Returns:
            Tuple of (warning_threshold_series, extreme_threshold_series)
        """
        min_obs = getattr(self.config, 'min_threshold_observations', 126)
        warn_pct = self.config.warning_percentile
        ext_pct = self.config.extreme_percentile

        warning_series = pd.Series(index=turbulence.index, dtype=float)
        extreme_series = pd.Series(index=turbulence.index, dtype=float)

        values = turbulence.values
        for i in range(len(values)):
            if i + 1 < min_obs:
                continue
            window = values[:i + 1]
            warning_series.iloc[i] = np.percentile(window, warn_pct)
            extreme_series.iloc[i] = np.percentile(window, ext_pct)

        logger.info(
            f"Expanding thresholds computed (min_obs={min_obs}, "
            f"first valid at index {min_obs})"
        )

        return warning_series, extreme_series

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

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
        warning_threshold,
        extreme_threshold,
    ) -> pd.Series:
        """
        Compute regime classification for all dates.

        ``warning_threshold`` and ``extreme_threshold`` can each be either
        a scalar (backward-compatible) or a pd.Series of the same index
        (for expanding thresholds).
        """
        # Normalise to Series
        if np.isscalar(warning_threshold):
            warn_s = pd.Series(warning_threshold, index=turbulence.index)
        else:
            warn_s = warning_threshold.reindex(turbulence.index)

        if np.isscalar(extreme_threshold):
            ext_s = pd.Series(extreme_threshold, index=turbulence.index)
        else:
            ext_s = extreme_threshold.reindex(turbulence.index)

        def _classify(idx):
            t = turbulence.loc[idx]
            w = warn_s.loc[idx]
            e = ext_s.loc[idx]
            if pd.isna(t) or pd.isna(w) or pd.isna(e):
                return None
            return self.classify_regime(t, w, e)

        return pd.Series(
            [_classify(idx) for idx in turbulence.index],
            index=turbulence.index,
        )

    # ------------------------------------------------------------------
    # VIX override circuit breaker (Fix 3)
    # ------------------------------------------------------------------

    def apply_vix_override(
        self,
        regime: pd.Series,
        vix: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Apply VIX-based sanity-check overrides to regime classifications.

        Rules:
        - VIX > vix_warning_level AND regime == HEALTHY  →  regime stays
          but ``vix_override`` is flagged as OVERRIDE_WARNING.
        - VIX > vix_critical_level  →  minimum regime forced to ELEVATED.

        Args:
            regime: Series of regime strings
            vix: VIX level series

        Returns:
            Tuple of (adjusted_regime, vix_override_bool)
        """
        warn_level = getattr(self.config, 'vix_warning_level', 40.0)
        crit_level = getattr(self.config, 'vix_critical_level', 60.0)

        vix_aligned = vix.reindex(regime.index)
        adjusted = regime.copy()
        override_flag = pd.Series(False, index=regime.index)

        for idx in regime.index:
            v = vix_aligned.get(idx, np.nan)
            if pd.isna(v):
                continue

            r = regime.loc[idx]

            # Critical override: force at least ELEVATED
            if v > crit_level and r == 'HEALTHY':
                adjusted.loc[idx] = 'ELEVATED'
                override_flag.loc[idx] = True
                continue

            # Warning flag: model says HEALTHY but VIX is high
            if v > warn_level and r == 'HEALTHY':
                override_flag.loc[idx] = True

        n_overrides = override_flag.sum()
        if n_overrides > 0:
            logger.info(f"VIX override triggered on {n_overrides} days")

        return adjusted, override_flag

    # ------------------------------------------------------------------
    # Interpretation helpers (unchanged)
    # ------------------------------------------------------------------

    def get_interpretation(self, regime: Regime) -> List[str]:
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

    # ------------------------------------------------------------------
    # Status generation
    # ------------------------------------------------------------------

    def generate_status(
        self,
        turbulence: pd.Series,
        spx_prices: pd.Series,
        vix: pd.Series,
        warning_threshold,
        extreme_threshold,
        days_elevated_series: pd.Series,
        divergence: pd.Series,
        ai_turbulence: Optional[pd.Series] = None,
        regime_series: Optional[pd.Series] = None,
        vix_override_series: Optional[pd.Series] = None,
    ) -> MarketStatus:
        """
        Generate complete market status for the latest date.
        """
        latest_date = turbulence.index[-1]
        latest_turbulence = turbulence.iloc[-1]

        # Compute percentile rank
        turbulence_pct = (turbulence <= latest_turbulence).mean() * 100

        # Resolve scalar / Series thresholds to latest value
        if isinstance(warning_threshold, pd.Series):
            w_thresh = warning_threshold.loc[latest_date] if latest_date in warning_threshold.index else warning_threshold.dropna().iloc[-1]
        else:
            w_thresh = warning_threshold

        if isinstance(extreme_threshold, pd.Series):
            e_thresh = extreme_threshold.loc[latest_date] if latest_date in extreme_threshold.index else extreme_threshold.dropna().iloc[-1]
        else:
            e_thresh = extreme_threshold

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

        # Regime (use pre-computed series if available, else classify)
        if regime_series is not None and latest_date in regime_series.index:
            regime = regime_series.loc[latest_date]
        else:
            regime = self.classify_regime(latest_turbulence, w_thresh, e_thresh)

        # Model regime before override
        model_regime = self.classify_regime(latest_turbulence, w_thresh, e_thresh)

        # VIX override
        vix_override = False
        if vix_override_series is not None and latest_date in vix_override_series.index:
            vix_override = bool(vix_override_series.loc[latest_date])

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
            warning_threshold=float(w_thresh),
            extreme_threshold=float(e_thresh),
            vix_override=vix_override,
            model_regime=model_regime if vix_override else None,
        )


def format_status_text(status: MarketStatus, signal_gen: SignalGenerator) -> str:
    """
    Format the status into a printable text panel.
    """
    lines = []

    date_str = status.date.strftime("%Y-%m-%d")
    lines.append("=" * 60)
    lines.append(f"  CURRENT IMMUNE SYSTEM STATUS — {date_str}")
    lines.append("=" * 60)
    lines.append("")

    level_indicator = {
        'HEALTHY': "🟢 HEALTHY",
        'ELEVATED': "🟡 ELEVATED",
        'CRISIS': "🔴 CRISIS"
    }
    regime_text = level_indicator.get(status.regime, status.regime)
    if status.vix_override and status.model_regime:
        regime_text += f"  [VIX OVERRIDE — model said {status.model_regime}]"
    lines.append(f"  WARNING LEVEL: {regime_text}")
    lines.append("")

    lines.append("  CURRENT METRICS:")
    lines.append(f"    • Market Turbulence: {status.turbulence:.1f} (P{status.turbulence_percentile:.0f})")
    lines.append(f"    • Days Elevated: {status.days_elevated}")
    lines.append(f"    • SPX Level: {status.spx_level:,.2f}")
    lines.append(f"    • SPX vs 50-day MA: {status.spx_vs_ma50} ({status.spx_ma50_ratio:.2%})")
    lines.append(f"    • VIX Level: {status.vix_level:.1f}")
    lines.append(f"    • VIX Override Active: {'YES ⚠️' if status.vix_override else 'NO'}")
    lines.append(f"    • Divergence Active: {'YES ⚠️' if status.divergence_active else 'NO'}")
    lines.append("")

    lines.append("  AI SECTOR CONTEXT:")
    if status.ai_turbulence is not None:
        lines.append(f"    • AI Basket Turbulence: {status.ai_turbulence:.1f}")
    else:
        lines.append("    • AI Basket Turbulence: N/A")
    lines.append("")

    lines.append("  INTERPRETATION:")
    for interp in signal_gen.get_interpretation(status.regime):
        lines.append(f"    • {interp}")
    lines.append("")

    lines.append("  RECOMMENDED ACTIONS:")
    for action in signal_gen.get_recommended_actions(status.regime):
        lines.append(f"    • {action}")
    lines.append("")

    lines.append("  THRESHOLDS:")
    lines.append(f"    • Warning (P95): {status.warning_threshold:.1f}")
    lines.append(f"    • Extreme (P99): {status.extreme_threshold:.1f}")

    lines.append("=" * 60)

    return "\n".join(lines)
