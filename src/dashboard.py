"""
Dashboard visualization module.

Creates the combined market turbulence dashboard with:
- Status panel showing Jordi Visser's regime framework
- Turbulence time series with regime shading
- VIX subplot
- HYG/IEF credit ratio subplot
- Last 90 days regime summary
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .config import TurbulenceConfig
from .signals import MarketStatus, SignalGenerator

logger = logging.getLogger(__name__)


# Regime color mapping
REGIME_COLORS = {
    'NORMAL': '#28a745',          # Green
    'GREEN_BAR': '#ffc107',       # Amber
    'STORM': '#fd7e14',           # Orange
    'STORM_CONTAGION': '#dc3545', # Red
}

REGIME_BG_COLORS = {
    'NORMAL': '#e8f5e9',
    'GREEN_BAR': '#fff8e1',
    'STORM': '#fff3e0',
    'STORM_CONTAGION': '#ffebee',
}


class DashboardRenderer:
    """Renders the market turbulence dashboard."""

    def __init__(self, config: TurbulenceConfig):
        self.config = config
        for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot', 'default']:
            try:
                plt.style.use(style)
                break
            except OSError:
                continue

    def _vix_color(self, vix_level: float) -> str:
        if vix_level > 40:
            return '#dc3545'   # Red
        elif vix_level > 25:
            return '#fd7e14'   # Orange
        elif vix_level > 20:
            return '#ffc107'   # Yellow
        else:
            return '#28a745'   # Green

    def render_status_panel(
        self,
        ax: plt.Axes,
        status: MarketStatus,
        signal_gen: SignalGenerator,
        regime_counts: Optional[dict] = None,
    ) -> None:
        """Render the status panel as a text box with Visser regime info."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        bg_color = REGIME_BG_COLORS.get(status.regime, '#f5f5f5')
        regime_color = REGIME_COLORS.get(status.regime, '#333333')

        rect = mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02",
            facecolor=bg_color,
            edgecolor='#333333',
            linewidth=2
        )
        ax.add_patch(rect)

        date_str = status.date.strftime("%Y-%m-%d")
        ax.text(0.5, 0.94, f"MARKET IMMUNE SYSTEM STATUS — {date_str}",
                ha='center', va='top', fontsize=13, fontweight='bold',
                family='monospace')

        # Regime label
        regime_labels = {
            'NORMAL': 'NORMAL',
            'GREEN_BAR': 'GREEN BAR (Complacency Warning)',
            'STORM': 'STORM (Risk Realized)',
            'STORM_CONTAGION': 'STORM + CONTAGION',
        }
        regime_text = regime_labels.get(status.regime, status.regime)
        if status.vix_override:
            regime_text += "  [VIX OVERRIDE]"

        ax.text(0.5, 0.86, f"REGIME: {regime_text}",
                ha='center', va='top', fontsize=15, fontweight='bold',
                color=regime_color, family='monospace')

        # Left column — metrics
        y_start = 0.76
        line_h = 0.048
        left_x = 0.06

        ax.text(left_x, y_start + line_h, "CURRENT METRICS:",
                ha='left', va='top', fontsize=10, fontweight='bold',
                family='monospace')

        metrics = [
            f"Turbulence: {status.turbulence:.1f} (P{status.turbulence_percentile:.0f})",
            f"Days Elevated: {status.days_elevated}",
            f"SPX: {status.spx_level:,.2f}  vs 50DMA: {status.spx_vs_ma50} ({status.spx_ma50_ratio:.2%})",
        ]
        for i, m in enumerate(metrics):
            ax.text(left_x + 0.02, y_start - i * line_h, f"  {m}",
                    ha='left', va='top', fontsize=9, family='monospace')

        # VIX with color
        vix_y = y_start - len(metrics) * line_h
        vix_color = self._vix_color(status.vix_level)
        ax.text(left_x + 0.02, vix_y, f"  VIX: {status.vix_level:.1f}",
                ha='left', va='top', fontsize=9, family='monospace',
                color=vix_color, fontweight='bold')

        # Credit / contagion
        credit_y = vix_y - line_h
        if status.hyg_ief_trend:
            credit_color = '#dc3545' if status.hyg_ief_trend == "Credit Weakening" else '#28a745'
            ax.text(left_x + 0.02, credit_y, f"  HYG/IEF: {status.hyg_ief_trend}",
                    ha='left', va='top', fontsize=9, family='monospace',
                    color=credit_color, fontweight='bold')
        else:
            ax.text(left_x + 0.02, credit_y, "  HYG/IEF: N/A",
                    ha='left', va='top', fontsize=9, family='monospace')

        # Dispersion
        disp_y = credit_y - line_h
        if status.dispersion_percentile is not None:
            ax.text(left_x + 0.02, disp_y,
                    f"  Dispersion: P{status.dispersion_percentile:.0f}",
                    ha='left', va='top', fontsize=9, family='monospace')

        # Divergence
        div_y = disp_y - line_h
        div_color = '#dc3545' if status.divergence_active else '#333333'
        div_text = "YES" if status.divergence_active else "NO"
        ax.text(left_x + 0.02, div_y, f"  Divergence: {div_text}",
                ha='left', va='top', fontsize=9, family='monospace',
                color=div_color, fontweight='bold' if status.divergence_active else 'normal')

        # Right column — interpretation + actions
        right_x = 0.52

        # Interpretation
        ax.text(right_x, y_start + line_h, "INTERPRETATION:",
                ha='left', va='top', fontsize=10, fontweight='bold',
                family='monospace')
        for i, interp in enumerate(signal_gen.get_interpretation(status.regime)[:4]):
            ax.text(right_x + 0.02, y_start - i * line_h, f"  {interp}",
                    ha='left', va='top', fontsize=8, family='monospace')

        # Actions
        action_y = y_start - 5 * line_h
        ax.text(right_x, action_y, "RECOMMENDED ACTIONS:",
                ha='left', va='top', fontsize=10, fontweight='bold',
                family='monospace')
        for i, action in enumerate(signal_gen.get_recommended_actions(status.regime)[:4]):
            ax.text(right_x + 0.02, action_y - (i + 1) * line_h, f"  {action}",
                    ha='left', va='top', fontsize=8, family='monospace')

        # Bottom: thresholds + 90-day summary
        bottom_y = 0.12
        ax.text(left_x, bottom_y, "THRESHOLDS:",
                ha='left', va='top', fontsize=10, fontweight='bold',
                family='monospace')
        ax.text(left_x + 0.02, bottom_y - 0.04,
                f"  P95: {status.warning_threshold:.1f}   P99: {status.extreme_threshold:.1f}",
                ha='left', va='top', fontsize=9, family='monospace')

        # 90-day regime summary
        if regime_counts:
            ax.text(right_x, bottom_y, "LAST 90 DAYS:",
                    ha='left', va='top', fontsize=10, fontweight='bold',
                    family='monospace')
            parts = []
            for r in ['NORMAL', 'GREEN_BAR', 'STORM', 'STORM_CONTAGION']:
                cnt = regime_counts.get(r, 0)
                if cnt > 0:
                    parts.append(f"{r}: {cnt}d")
            ax.text(right_x + 0.02, bottom_y - 0.04,
                    "  " + "  |  ".join(parts) if parts else "  No data",
                    ha='left', va='top', fontsize=8, family='monospace')

    def render_turbulence_chart(
        self,
        ax: plt.Axes,
        turbulence: pd.Series,
        spx_prices: pd.Series,
        warning_threshold,
        extreme_threshold,
        divergence: pd.Series,
        regime: Optional[pd.Series] = None,
        turbulence_rolling: Optional[pd.Series] = None,
    ) -> None:
        """
        Render the turbulence chart with regime shading and optional
        rolling-window comparison trace.

        GREEN_BAR periods -> yellow shading
        STORM_CONTAGION periods -> red shading
        STORM periods -> orange shading
        """
        common_idx = turbulence.index.intersection(spx_prices.index)
        turb = turbulence.loc[common_idx]
        spx = spx_prices.loc[common_idx]

        # Resolve thresholds
        if isinstance(warning_threshold, pd.Series):
            warn_vals = warning_threshold.reindex(common_idx)
        else:
            warn_vals = pd.Series(warning_threshold, index=common_idx)

        if isinstance(extreme_threshold, pd.Series):
            ext_vals = extreme_threshold.reindex(common_idx)
        else:
            ext_vals = pd.Series(extreme_threshold, index=common_idx)

        # Twin axis for SPX
        ax2 = ax.twinx()

        # Plot baseline-anchored turbulence (the good one)
        ax.plot(turb.index, turb.values, color=self.config.color_turbulence,
                linewidth=1.5, label='Turbulence (Baseline-Anchored)', zorder=4)

        # Plot rolling-window turbulence (the old, desensitised one) if available
        if turbulence_rolling is not None and not turbulence_rolling.empty:
            rolling_aligned = turbulence_rolling.reindex(common_idx)
            ax.plot(rolling_aligned.index, rolling_aligned.values,
                    color='#999999', linewidth=1.0, linestyle='-', alpha=0.6,
                    label='Turbulence (Rolling-Window)', zorder=3)

            # Shade the gap where baseline > rolling (= desensitisation gap)
            both_valid = turb.notna() & rolling_aligned.notna()
            if both_valid.any():
                idx_valid = common_idx[both_valid]
                turb_v = turb.reindex(idx_valid).values
                roll_v = rolling_aligned.reindex(idx_valid).values
                ax.fill_between(
                    idx_valid, roll_v, turb_v,
                    where=turb_v > roll_v,
                    alpha=0.18, color='#d62728',
                    label='Desensitisation Gap', zorder=2,
                    interpolate=True,
                )

        # Plot SPX
        ax2.plot(spx.index, spx.values, color=self.config.color_spx,
                 linewidth=1.5, label='SPX/SPY', zorder=3)

        # 50-day MA
        ma50 = spx.rolling(window=self.config.ma_long).mean()
        ax2.plot(spx.index, ma50.values, color=self.config.color_ma50,
                 linewidth=1, linestyle='--', label='50-day MA', alpha=0.7, zorder=2)

        # Threshold lines
        ax.plot(warn_vals.index, warn_vals.values,
                color=self.config.color_warning, linestyle='--', linewidth=1.5,
                label='Warning (P95)', zorder=2)
        ax.plot(ext_vals.index, ext_vals.values,
                color=self.config.color_extreme, linestyle='--', linewidth=1.5,
                label='Extreme (P99)', zorder=2)

        # Regime shading
        if regime is not None:
            regime_aligned = regime.reindex(common_idx)
            shade_map = {
                'GREEN_BAR': (self.config.color_green_bar, 0.25),
                'STORM': (self.config.color_storm, 0.20),
                'STORM_CONTAGION': (self.config.color_storm_contagion, 0.30),
            }
            for regime_name, (color, alpha) in shade_map.items():
                in_regime = False
                start_date = None
                for date in common_idx:
                    r = regime_aligned.get(date, None)
                    if r == regime_name and not in_regime:
                        start_date = date
                        in_regime = True
                    elif r != regime_name and in_regime:
                        ax.axvspan(start_date, date, alpha=alpha, color=color, zorder=1)
                        in_regime = False
                if in_regime and start_date is not None:
                    ax.axvspan(start_date, common_idx[-1], alpha=alpha, color=color, zorder=1)

        # Formatting
        ax.set_ylabel('Market Turbulence', fontsize=10, color=self.config.color_turbulence)
        ax2.set_ylabel('SPX/SPY Level', fontsize=10, color=self.config.color_spx)

        ax.tick_params(axis='y', labelcolor=self.config.color_turbulence)
        ax2.tick_params(axis='y', labelcolor=self.config.color_spx)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        turb_max = turb.max()
        ext_max = ext_vals.dropna().max() if len(ext_vals.dropna()) > 0 else 0
        ax.set_ylim(0, max(turb_max, ext_max) * 1.2)

        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        legend_patches = []
        for rname, (color, _) in [('GREEN_BAR', (self.config.color_green_bar, 0.3)),
                                    ('STORM', (self.config.color_storm, 0.3)),
                                    ('STORM_CONTAGION', (self.config.color_storm_contagion, 0.3))]:
            legend_patches.append(mpatches.Patch(color=color, alpha=0.3, label=rname))

        ax.legend(lines1 + lines2 + legend_patches,
                  labels1 + labels2 + ['GREEN_BAR', 'STORM', 'STORM_CONTAGION'],
                  loc='upper left', fontsize=7, framealpha=0.9)

        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    def render_vix_subplot(
        self,
        ax: plt.Axes,
        vix: pd.Series,
        turbulence: pd.Series,
    ) -> None:
        """Render VIX subplot with calm ceiling line."""
        common_idx = turbulence.index.intersection(vix.dropna().index)
        if len(common_idx) == 0:
            ax.text(0.5, 0.5, 'VIX data unavailable', ha='center', va='center')
            return

        vix_plot = vix.reindex(common_idx)

        ax.plot(vix_plot.index, vix_plot.values, color='#8B0000',
                linewidth=1.2, label='VIX')
        ax.fill_between(vix_plot.index, 0, vix_plot.values,
                        alpha=0.15, color='#8B0000')

        vix_ceiling = getattr(self.config, 'vix_calm_ceiling', 25.0)
        ax.axhline(y=vix_ceiling, color='#ffc107', linestyle=':', linewidth=1,
                   label=f'Calm Ceiling ({vix_ceiling:.0f})')
        ax.axhline(y=40, color='orange', linestyle=':', linewidth=1,
                   label='VIX Warning (40)')

        ax.set_ylabel('VIX', fontsize=9, color='#8B0000')
        ax.tick_params(axis='y', labelcolor='#8B0000')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    def render_credit_subplot(
        self,
        ax: plt.Axes,
        hyg_ief_ratio: pd.Series,
        turbulence: pd.Series,
        contagion: Optional[pd.Series] = None,
    ) -> None:
        """Render HYG/IEF credit ratio subplot with contagion shading."""
        common_idx = turbulence.index.intersection(hyg_ief_ratio.dropna().index)
        if len(common_idx) == 0:
            ax.text(0.5, 0.5, 'HYG/IEF data unavailable', ha='center', va='center')
            return

        ratio_plot = hyg_ief_ratio.reindex(common_idx)

        ax.plot(ratio_plot.index, ratio_plot.values, color='#1a237e',
                linewidth=1.2, label='HYG/IEF Ratio')

        # Shade contagion periods
        if contagion is not None:
            contagion_aligned = contagion.reindex(common_idx).fillna(False)
            in_contagion = False
            start_date = None
            for date in common_idx:
                if contagion_aligned.get(date, False) and not in_contagion:
                    start_date = date
                    in_contagion = True
                elif not contagion_aligned.get(date, False) and in_contagion:
                    ax.axvspan(start_date, date, alpha=0.2, color='#dc3545', zorder=1)
                    in_contagion = False
            if in_contagion and start_date is not None:
                ax.axvspan(start_date, common_idx[-1], alpha=0.2, color='#dc3545', zorder=1)

        ax.set_ylabel('HYG/IEF', fontsize=9, color='#1a237e')
        ax.tick_params(axis='y', labelcolor='#1a237e')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        contagion_patch = mpatches.Patch(color='#dc3545', alpha=0.2, label='Contagion')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [contagion_patch], labels + ['Contagion'],
                  loc='upper left', fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    def render_dashboard(
        self,
        turbulence: pd.Series,
        spx_prices: pd.Series,
        status: MarketStatus,
        signal_gen: SignalGenerator,
        divergence: pd.Series,
        output_path: Optional[str] = None,
        vix: Optional[pd.Series] = None,
        warning_threshold=None,
        extreme_threshold=None,
        regime: Optional[pd.Series] = None,
        hyg_ief_ratio: Optional[pd.Series] = None,
        contagion: Optional[pd.Series] = None,
        regime_counts: Optional[dict] = None,
        turbulence_rolling: Optional[pd.Series] = None,
    ) -> plt.Figure:
        """
        Render the complete dashboard with status panel, turbulence chart,
        VIX subplot, and HYG/IEF credit subplot.
        """
        has_vix = vix is not None and not vix.empty
        has_credit = hyg_ief_ratio is not None and not hyg_ief_ratio.empty

        # Layout: status (25%), chart (35%), VIX (20%), credit (20%)
        n_rows = 2  # status + chart
        ratios = [0.30, 0.40]
        if has_vix:
            n_rows += 1
            ratios.append(0.15)
        if has_credit:
            n_rows += 1
            ratios.append(0.15)

        total_height = self.config.figure_height + (n_rows - 2) * 2
        fig = plt.figure(figsize=(self.config.figure_width, total_height))
        gs = fig.add_gridspec(n_rows, 1, height_ratios=ratios, hspace=0.25)

        # Status panel
        ax_status = fig.add_subplot(gs[0])
        self.render_status_panel(ax_status, status, signal_gen, regime_counts)

        # Resolve thresholds
        w_thresh = warning_threshold if warning_threshold is not None else status.warning_threshold
        e_thresh = extreme_threshold if extreme_threshold is not None else status.extreme_threshold

        # Turbulence chart
        ax_chart = fig.add_subplot(gs[1])
        self.render_turbulence_chart(
            ax_chart, turbulence, spx_prices,
            w_thresh, e_thresh, divergence,
            regime=regime,
            turbulence_rolling=turbulence_rolling,
        )

        # VIX subplot
        subplot_idx = 2
        if has_vix:
            ax_vix = fig.add_subplot(gs[subplot_idx])
            self.render_vix_subplot(ax_vix, vix, turbulence)
            subplot_idx += 1

        # Credit subplot
        if has_credit:
            ax_credit = fig.add_subplot(gs[subplot_idx])
            self.render_credit_subplot(ax_credit, hyg_ief_ratio, turbulence, contagion)

        fig.suptitle(
            "MARKET TURBULENCE MONITOR — Jordi Visser Framework\n"
            "(Red line=Baseline-Anchored  Grey line=Rolling-Window  "
            "Shaded gap=Desensitisation fix)",
            fontsize=12, fontweight='bold', y=0.99
        )

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            logger.info(f"Dashboard saved to {output_file}")

        return fig


def save_features_csv(
    turbulence: pd.Series,
    spx_prices: pd.Series,
    vix: pd.Series,
    divergence: pd.Series,
    days_elevated: pd.Series,
    regime: pd.Series,
    warning_threshold,
    extreme_threshold,
    output_path: str,
    vix_override: Optional[pd.Series] = None,
    hyg_ief_ratio: Optional[pd.Series] = None,
    hyg_ief_slope: Optional[pd.Series] = None,
    contagion: Optional[pd.Series] = None,
    dispersion: Optional[pd.Series] = None,
    dispersion_pctile: Optional[pd.Series] = None,
    turbulence_rolling: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Save computed features to CSV.
    """
    spx_ma50 = spx_prices.rolling(window=50).mean()
    common_idx = turbulence.index

    # Resolve thresholds
    if isinstance(warning_threshold, pd.Series):
        warn_col = warning_threshold.reindex(common_idx)
    else:
        warn_col = pd.Series(warning_threshold, index=common_idx)

    if isinstance(extreme_threshold, pd.Series):
        ext_col = extreme_threshold.reindex(common_idx)
    else:
        ext_col = pd.Series(extreme_threshold, index=common_idx)

    features = pd.DataFrame({
        'spx_close': spx_prices.reindex(common_idx),
        'spx_50dma': spx_ma50.reindex(common_idx),
        'spx_above_50dma': (spx_prices > spx_ma50).reindex(common_idx),
        'vix': vix.reindex(common_idx),
        'turbulence': turbulence,
        'warning_threshold': warn_col,
        'extreme_threshold': ext_col,
        'regime': regime.reindex(common_idx),
        'divergence': divergence.reindex(common_idx),
        'days_elevated': days_elevated.reindex(common_idx),
    })

    if vix_override is not None:
        features['vix_override'] = vix_override.reindex(common_idx)

    if hyg_ief_ratio is not None and not hyg_ief_ratio.empty:
        features['hyg_ief_ratio'] = hyg_ief_ratio.reindex(common_idx)

    if hyg_ief_slope is not None:
        features['hyg_ief_slope'] = hyg_ief_slope.reindex(common_idx)

    if contagion is not None:
        features['contagion'] = contagion.reindex(common_idx)

    if dispersion is not None:
        features['dispersion'] = dispersion.reindex(common_idx)

    if dispersion_pctile is not None:
        features['dispersion_pctile'] = dispersion_pctile.reindex(common_idx)

    if turbulence_rolling is not None and not turbulence_rolling.empty:
        features['turbulence_rolling'] = turbulence_rolling.reindex(common_idx)
        # Desensitisation gap: how much the rolling approach underestimates
        features['desensitisation_gap'] = (
            features['turbulence'] - features['turbulence_rolling']
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_file)
    logger.info(f"Features saved to {output_file}")

    return features
