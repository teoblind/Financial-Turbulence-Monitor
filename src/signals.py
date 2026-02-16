"""
Signal generation module.

Implements Jordi Visser's three-regime framework:
- NORMAL: no unusual cross-asset stress
- GREEN_BAR: turbulence high + VIX low + SPX above 50DMA (complacency warning)
- STORM: turbulence high + VIX confirms stress (risk realized)
- STORM_CONTAGION: STORM + HYG/IEF ratio declining (credit contagion)

Also provides expanding-window thresholds, divergence detection,
contagion detection via HYG/IEF slope, and cross-sectional dispersion.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .config import TurbulenceConfig

logger = logging.getLogger(__name__)


# Regime definitions (Jordi Visser's framework)
Regime = Literal['NORMAL', 'GREEN_BAR', 'STORM', 'STORM_CONTAGION']

# Backward-compat aliases used only in get_interpretation / get_recommended_actions
_OLD_REGIMES = {'HEALTHY', 'ELEVATED', 'CRISIS'}


@dataclass
class MarketStatus:
    """Container for current market status information."""
    date: datetime
    regime: str
    turbulence: float
    turbulence_percentile: float
    days_elevated: int
    spx_level: float
    spx_vs_ma50: str
    spx_ma50_ratio: float
    vix_level: float
    divergence_active: bool
    ai_infra_turbulence: Optional[float] = None
    saas_turbulence: Optional[float] = None
    warning_threshold: float = 0.0
    extreme_threshold: float = 0.0
    # Contagion / credit
    hyg_ief_trend: Optional[str] = None   # "Credit Stable" or "Credit Weakening"
    contagion_active: bool = False
    # Dispersion
    dispersion: Optional[float] = None
    dispersion_percentile: Optional[float] = None
    # VIX override (backward compat)
    vix_override: bool = False
    model_regime: Optional[str] = None


class SignalGenerator:
    """Generates trading signals and regime classifications."""

    def __init__(self, config: TurbulenceConfig):
        self.config = config

    # ------------------------------------------------------------------
    # SPX / divergence detection
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
    # Expanding-window rolling thresholds
    # ------------------------------------------------------------------

    def compute_expanding_thresholds(
        self,
        turbulence: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute rolling P95 / P99 thresholds using an expanding window.

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
    # Jordi Visser's three-regime classification
    # ------------------------------------------------------------------

    def classify_regime_visser(
        self,
        turbulence_value: float,
        spx_close: float,
        spx_50dma: float,
        vix: float,
        warning_threshold: float,
    ) -> str:
        """
        Jordi Visser's three-regime classification.

        GREEN_BAR: Turbulence high + markets calm (complacency warning)
        STORM: Turbulence high + VIX confirms stress (risk realized)
        NORMAL: No unusual stress

        Args:
            turbulence_value: Current turbulence score
            spx_close: Current SPX/SPY close price
            spx_50dma: Current 50-day moving average of SPX
            vix: Current VIX level
            warning_threshold: P95 turbulence threshold

        Returns:
            Regime string: "NORMAL", "GREEN_BAR", or "STORM"
        """
        vix_ceiling = getattr(self.config, 'vix_calm_ceiling', 25.0)

        if turbulence_value > warning_threshold:
            if vix < vix_ceiling and spx_close > spx_50dma:
                return "GREEN_BAR"
            else:
                return "STORM"
        else:
            return "NORMAL"

    def compute_regime_series(
        self,
        turbulence: pd.Series,
        warning_threshold,
        extreme_threshold,
        spx_prices: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute regime classification for all dates using Jordi Visser's framework.

        If ``spx_prices`` and ``vix`` are provided, uses the three-regime model.
        Otherwise falls back to threshold-only classification for backward compat.

        Args:
            turbulence: Turbulence series
            warning_threshold: Scalar or Series of P95 thresholds
            extreme_threshold: Scalar or Series of P99 thresholds (unused in Visser model,
                               kept for backward compat signature)
            spx_prices: SPX/SPY price series (optional)
            vix: VIX level series (optional)

        Returns:
            Series of regime strings
        """
        # Normalise thresholds to Series
        if np.isscalar(warning_threshold):
            warn_s = pd.Series(warning_threshold, index=turbulence.index)
        else:
            warn_s = warning_threshold.reindex(turbulence.index)

        # If we have SPX and VIX, use Visser's model
        if spx_prices is not None and vix is not None:
            spx_aligned = spx_prices.reindex(turbulence.index).ffill()
            vix_aligned = vix.reindex(turbulence.index).ffill()
            spx_50dma = spx_aligned.rolling(window=self.config.ma_long, min_periods=1).mean()

            regimes = []
            for idx in turbulence.index:
                t = turbulence.loc[idx]
                w = warn_s.loc[idx]
                s = spx_aligned.get(idx, np.nan)
                s50 = spx_50dma.get(idx, np.nan)
                v = vix_aligned.get(idx, np.nan)

                if pd.isna(t) or pd.isna(w) or pd.isna(s) or pd.isna(v):
                    regimes.append(None)
                    continue

                regimes.append(self.classify_regime_visser(t, s, s50, v, w))

            return pd.Series(regimes, index=turbulence.index)

        # Fallback: threshold-only (backward compat)
        if np.isscalar(extreme_threshold):
            ext_s = pd.Series(extreme_threshold, index=turbulence.index)
        else:
            ext_s = extreme_threshold.reindex(turbulence.index)

        def _classify_legacy(idx):
            t = turbulence.loc[idx]
            w = warn_s.loc[idx]
            e = ext_s.loc[idx]
            if pd.isna(t) or pd.isna(w) or pd.isna(e):
                return None
            if t >= e:
                return 'STORM'
            elif t >= w:
                return 'GREEN_BAR'
            else:
                return 'NORMAL'

        return pd.Series(
            [_classify_legacy(idx) for idx in turbulence.index],
            index=turbulence.index,
        )

    # ------------------------------------------------------------------
    # Contagion detection (HYG/IEF slope)
    # ------------------------------------------------------------------

    def check_contagion(
        self,
        hyg_ief_ratio: pd.Series,
        lookback: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Check if HYG/IEF ratio is declining (credit contagion spreading).

        Uses rolling linear regression slope of the ratio over ``lookback`` days.
        Negative slope → credit weakening → contagion risk.

        Args:
            hyg_ief_ratio: Daily HYG/IEF price ratio series
            lookback: Days for slope computation (default from config)

        Returns:
            Tuple of (slope_series, contagion_bool_series)
        """
        if lookback is None:
            lookback = getattr(self.config, 'contagion_lookback', 20)

        slope_threshold = getattr(self.config, 'contagion_slope_threshold', 0.0)

        slope_series = pd.Series(np.nan, index=hyg_ief_ratio.index)

        ratio_vals = hyg_ief_ratio.values
        x = np.arange(lookback, dtype=float)

        for i in range(lookback, len(ratio_vals)):
            window = ratio_vals[i - lookback:i]
            if np.any(np.isnan(window)):
                continue
            slope, _, _, _, _ = sp_stats.linregress(x, window)
            slope_series.iloc[i] = slope

        contagion = slope_series < slope_threshold

        logger.info(f"Contagion detection: {contagion.sum()} days with declining HYG/IEF")

        return slope_series, contagion

    def apply_contagion_to_regime(
        self,
        regime: pd.Series,
        contagion: pd.Series,
    ) -> pd.Series:
        """
        Upgrade STORM days to STORM_CONTAGION if contagion is active.

        Args:
            regime: Series of regime strings
            contagion: Boolean series indicating credit contagion

        Returns:
            Updated regime series
        """
        adjusted = regime.copy()
        contagion_aligned = contagion.reindex(regime.index).fillna(False)

        storm_mask = (regime == 'STORM')
        contagion_storm = storm_mask & contagion_aligned
        adjusted[contagion_storm] = 'STORM_CONTAGION'

        n_contagion = contagion_storm.sum()
        if n_contagion > 0:
            logger.info(f"STORM_CONTAGION flagged on {n_contagion} days")

        return adjusted

    # ------------------------------------------------------------------
    # Dispersion metric
    # ------------------------------------------------------------------

    def compute_dispersion(
        self,
        returns: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute cross-sectional dispersion of returns.

        For each day, the cross-sectional standard deviation of returns across
        basket assets. Normalized to a rolling percentile.

        Args:
            returns: DataFrame of asset returns (cols = assets)

        Returns:
            Tuple of (dispersion, dispersion_percentile)
        """
        # Cross-sectional std per day
        dispersion = returns.std(axis=1)

        # Rolling percentile (expanding window)
        min_obs = getattr(self.config, 'min_threshold_observations', 126)
        disp_pctile = pd.Series(np.nan, index=dispersion.index)
        vals = dispersion.values
        for i in range(min_obs, len(vals)):
            window = vals[:i + 1]
            rank = (window <= vals[i]).sum()
            disp_pctile.iloc[i] = (rank / len(window)) * 100.0

        logger.info(f"Dispersion computed for {len(dispersion)} days")

        return dispersion, disp_pctile

    # ------------------------------------------------------------------
    # VIX override circuit breaker (backward compat)
    # ------------------------------------------------------------------

    def apply_vix_override(
        self,
        regime: pd.Series,
        vix: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Apply VIX-based sanity-check overrides to regime classifications.

        In the new Visser model VIX is already incorporated into the regime
        classification, so this mainly catches edge cases where the model
        says NORMAL but VIX is extremely elevated.

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

            # Critical override: force at least STORM if VIX > critical and regime is NORMAL
            if v > crit_level and r == 'NORMAL':
                adjusted.loc[idx] = 'STORM'
                override_flag.loc[idx] = True
                continue

            # Warning flag: model says NORMAL but VIX is high
            if v > warn_level and r == 'NORMAL':
                override_flag.loc[idx] = True

        n_overrides = override_flag.sum()
        if n_overrides > 0:
            logger.info(f"VIX override triggered on {n_overrides} days")

        return adjusted, override_flag

    # ------------------------------------------------------------------
    # Interpretation helpers
    # ------------------------------------------------------------------

    def get_interpretation(self, regime: str) -> List[str]:
        interpretations = {
            'NORMAL': [
                "Market showing normal stress levels",
                "Cross-asset correlations stable",
                "No concerning correlation breakdowns",
                "Risk-on environment favorable"
            ],
            'GREEN_BAR': [
                "Covariance matrix shocked but surface looks calm",
                "VIX low and markets trending up — complacency warning",
                "Market immune system detecting hidden fever",
                "This is the early warning — most dangerous state"
            ],
            'STORM': [
                "Turbulence high AND VIX confirms stress",
                "Risk is no longer hidden — it's realized",
                "Monitor HYG/IEF for credit contagion",
                "Consider reducing leverage and risk exposure"
            ],
            'STORM_CONTAGION': [
                "STORM with credit contagion spreading",
                "HYG/IEF ratio declining — high yield weakening vs Treasuries",
                "Waterfall risk elevated — correlations breaking down",
                "Risk management action urgently required"
            ],
            # Backward compat
            'HEALTHY': [
                "Market showing normal stress levels",
                "Cross-asset correlations stable",
            ],
            'ELEVATED': [
                "Cross-asset correlations showing instability",
                "Hidden stress building beneath surface",
            ],
            'CRISIS': [
                "Systemic stress significantly elevated",
                "Risk management action urgently required",
            ],
        }
        return interpretations.get(regime, [])

    def get_recommended_actions(self, regime: str) -> List[str]:
        actions = {
            'NORMAL': [
                "Maintain current positioning",
                "Monitor for regime transitions",
                "Standard risk protocols apply"
            ],
            'GREEN_BAR': [
                "Heightened alertness — complacency is the risk",
                "Review portfolio tail hedges",
                "Increase hedge ratios prophylactically",
                "Prepare for rapid regime shift to STORM"
            ],
            'STORM': [
                "Monitor HYG/IEF ratio for credit contagion",
                "Consider de-risking by 10-25%",
                "Review portfolio correlations",
                "Prepare contingency plans"
            ],
            'STORM_CONTAGION': [
                "Reduce gross exposure significantly",
                "Maximize defensive positioning",
                "Activate crisis playbook",
                "Daily monitoring required"
            ],
            # Backward compat
            'HEALTHY': ["Maintain current positioning"],
            'ELEVATED': ["Review portfolio correlations", "Consider de-risking"],
            'CRISIS': ["Reduce exposure", "Activate crisis playbook"],
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
        ai_infra_turbulence: Optional[pd.Series] = None,
        saas_turbulence: Optional[pd.Series] = None,
        regime_series: Optional[pd.Series] = None,
        vix_override_series: Optional[pd.Series] = None,
        hyg_ief_slope: Optional[pd.Series] = None,
        contagion_series: Optional[pd.Series] = None,
        dispersion: Optional[pd.Series] = None,
        dispersion_pctile: Optional[pd.Series] = None,
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

        # Regime
        if regime_series is not None and latest_date in regime_series.index:
            regime = regime_series.loc[latest_date]
        else:
            regime = self.classify_regime_visser(latest_turbulence, latest_spx, latest_ma50, latest_vix, w_thresh)

        # VIX override
        vix_override = False
        if vix_override_series is not None and latest_date in vix_override_series.index:
            vix_override = bool(vix_override_series.loc[latest_date])

        # AI Infrastructure turbulence
        ai_infra_turb = None
        if ai_infra_turbulence is not None and not ai_infra_turbulence.empty:
            if latest_date in ai_infra_turbulence.index:
                ai_infra_turb = ai_infra_turbulence.loc[latest_date]
            elif len(ai_infra_turbulence) > 0:
                ai_infra_turb = ai_infra_turbulence.iloc[-1]

        # SaaS turbulence
        saas_turb = None
        if saas_turbulence is not None and not saas_turbulence.empty:
            if latest_date in saas_turbulence.index:
                saas_turb = saas_turbulence.loc[latest_date]
            elif len(saas_turbulence) > 0:
                saas_turb = saas_turbulence.iloc[-1]

        # Contagion / credit
        hyg_ief_trend = None
        contagion_active = False
        if contagion_series is not None and latest_date in contagion_series.index:
            contagion_active = bool(contagion_series.loc[latest_date])
        if hyg_ief_slope is not None and latest_date in hyg_ief_slope.index:
            slope_val = hyg_ief_slope.loc[latest_date]
            if not pd.isna(slope_val):
                hyg_ief_trend = "Credit Weakening" if slope_val < 0 else "Credit Stable"

        # Dispersion
        disp_val = None
        disp_pctile_val = None
        if dispersion is not None and latest_date in dispersion.index:
            disp_val = float(dispersion.loc[latest_date])
        if dispersion_pctile is not None and latest_date in dispersion_pctile.index:
            dp = dispersion_pctile.loc[latest_date]
            if not pd.isna(dp):
                disp_pctile_val = float(dp)

        return MarketStatus(
            date=latest_date.to_pydatetime() if hasattr(latest_date, 'to_pydatetime') else latest_date,
            regime=regime if regime is not None else 'NORMAL',
            turbulence=float(latest_turbulence),
            turbulence_percentile=float(turbulence_pct),
            days_elevated=days_elevated,
            spx_level=float(latest_spx),
            spx_vs_ma50=spx_vs_ma50,
            spx_ma50_ratio=float(spx_ratio),
            vix_level=float(latest_vix) if not np.isnan(latest_vix) else 0.0,
            divergence_active=div_active,
            ai_infra_turbulence=float(ai_infra_turb) if ai_infra_turb is not None else None,
            saas_turbulence=float(saas_turb) if saas_turb is not None else None,
            warning_threshold=float(w_thresh),
            extreme_threshold=float(e_thresh),
            hyg_ief_trend=hyg_ief_trend,
            contagion_active=contagion_active,
            dispersion=disp_val,
            dispersion_percentile=disp_pctile_val,
            vix_override=vix_override,
            model_regime=None,
        )


def format_status_text(status: MarketStatus, signal_gen: SignalGenerator) -> str:
    """
    Format the status into a printable text panel.
    """
    lines = []

    date_str = status.date.strftime("%Y-%m-%d")
    lines.append("=" * 60)
    lines.append(f"  MARKET IMMUNE SYSTEM STATUS — {date_str}")
    lines.append("=" * 60)
    lines.append("")

    regime_indicators = {
        'NORMAL': "NORMAL",
        'GREEN_BAR': "GREEN BAR (Complacency Warning)",
        'STORM': "STORM (Risk Realized)",
        'STORM_CONTAGION': "STORM + CONTAGION (Credit Waterfall)",
        # Backward compat
        'HEALTHY': "HEALTHY",
        'ELEVATED': "ELEVATED",
        'CRISIS': "CRISIS",
    }
    regime_text = regime_indicators.get(status.regime, status.regime)
    if status.vix_override:
        regime_text += "  [VIX OVERRIDE]"
    lines.append(f"  REGIME: {regime_text}")
    lines.append("")

    lines.append("  CURRENT METRICS:")
    lines.append(f"    Turbulence: {status.turbulence:.1f} (P{status.turbulence_percentile:.0f})")
    lines.append(f"    Days Elevated: {status.days_elevated}")
    lines.append(f"    SPX Level: {status.spx_level:,.2f}")
    lines.append(f"    SPX vs 50DMA: {status.spx_vs_ma50} ({status.spx_ma50_ratio:.2%})")
    lines.append(f"    VIX Level: {status.vix_level:.1f}")
    if status.hyg_ief_trend:
        lines.append(f"    HYG/IEF Trend: {status.hyg_ief_trend}")
    if status.dispersion_percentile is not None:
        lines.append(f"    Dispersion: P{status.dispersion_percentile:.0f}")
    lines.append(f"    Divergence Active: {'YES' if status.divergence_active else 'NO'}")
    lines.append("")

    lines.append("  TECH SECTOR CONTEXT:")
    if status.ai_infra_turbulence is not None:
        lines.append(f"    AI Infra Turbulence: {status.ai_infra_turbulence:.1f}")
    else:
        lines.append("    AI Infra Turbulence: N/A")
    if status.saas_turbulence is not None:
        lines.append(f"    SaaS Turbulence:     {status.saas_turbulence:.1f}")
    else:
        lines.append("    SaaS Turbulence:     N/A")
    lines.append("")

    lines.append("  INTERPRETATION:")
    for interp in signal_gen.get_interpretation(status.regime):
        lines.append(f"    {interp}")
    lines.append("")

    lines.append("  RECOMMENDED ACTIONS:")
    for action in signal_gen.get_recommended_actions(status.regime):
        lines.append(f"    {action}")
    lines.append("")

    lines.append("  THRESHOLDS:")
    lines.append(f"    Warning (P95): {status.warning_threshold:.1f}")
    lines.append(f"    Extreme (P99): {status.extreme_threshold:.1f}")

    lines.append("=" * 60)

    return "\n".join(lines)
