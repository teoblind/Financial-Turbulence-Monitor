"""
Dashboard visualization module.

Creates the combined market turbulence dashboard with status panel,
divergence detector chart, and VIX subplot using Matplotlib.
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

    def render_status_panel(
        self,
        ax: plt.Axes,
        status: MarketStatus,
        signal_gen: SignalGenerator
    ) -> None:
        """Render the status panel as a text box."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        bg_colors = {
            'HEALTHY': '#e8f5e9',
            'ELEVATED': '#fff3e0',
            'CRISIS': '#ffebee'
        }
        bg_color = bg_colors.get(status.regime, '#f5f5f5')

        rect = mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02",
            facecolor=bg_color,
            edgecolor='#333333',
            linewidth=2
        )
        ax.add_patch(rect)

        date_str = status.date.strftime("%Y-%m-%d")
        ax.text(0.5, 0.92, f"CURRENT IMMUNE SYSTEM STATUS — {date_str}",
                ha='center', va='top', fontsize=14, fontweight='bold',
                family='monospace')

        regime_colors = {
            'HEALTHY': self.config.status_healthy,
            'ELEVATED': self.config.status_elevated,
            'CRISIS': self.config.status_crisis
        }

        regime_label = status.regime
        if status.vix_override and status.model_regime:
            regime_label += f"  [VIX OVERRIDE]"

        ax.text(0.5, 0.83, f"WARNING LEVEL: {regime_label}",
                ha='center', va='top', fontsize=16, fontweight='bold',
                color=regime_colors.get(status.regime, 'black'),
                family='monospace')

        # Metrics - left column
        y_start = 0.72
        line_height = 0.050
        left_x = 0.08

        metrics = [
            f"Market Turbulence: {status.turbulence:.1f} (P{status.turbulence_percentile:.0f})",
            f"Days Elevated: {status.days_elevated}",
            f"SPX Level: {status.spx_level:,.2f}",
            f"SPX vs 50-day MA: {status.spx_vs_ma50} ({status.spx_ma50_ratio:.2%})",
            f"VIX Level: {status.vix_level:.1f}",
        ]

        ax.text(left_x, y_start + line_height, "CURRENT METRICS:",
                ha='left', va='top', fontsize=11, fontweight='bold',
                family='monospace')

        for i, metric in enumerate(metrics):
            ax.text(left_x + 0.02, y_start - i * line_height, f"• {metric}",
                    ha='left', va='top', fontsize=10, family='monospace')

        # VIX override indicator
        override_y = y_start - 5 * line_height
        if status.vix_override:
            override_text = "• VIX Override: ACTIVE"
            override_color = self.config.status_crisis
            override_weight = 'bold'
        else:
            override_text = "• VIX Override: inactive"
            override_color = 'black'
            override_weight = 'normal'
        ax.text(left_x + 0.02, override_y, override_text,
                ha='left', va='top', fontsize=10, family='monospace',
                color=override_color, fontweight=override_weight)

        # Divergence indicator
        div_y = override_y - line_height
        div_color = self.config.status_crisis if status.divergence_active else 'black'
        div_text = "YES" if status.divergence_active else "NO"
        ax.text(left_x + 0.02, div_y, f"• Divergence Active: {div_text}",
                ha='left', va='top', fontsize=10, family='monospace',
                color=div_color, fontweight='bold' if status.divergence_active else 'normal')

        # If overridden, show model regime
        if status.vix_override and status.model_regime:
            ax.text(left_x + 0.02, div_y - line_height,
                    f"• Model Regime (pre-override): {status.model_regime}",
                    ha='left', va='top', fontsize=10, family='monospace',
                    color='#666666')

        # AI Sector - right side top
        right_x = 0.55
        ax.text(right_x, y_start + line_height, "AI SECTOR CONTEXT:",
                ha='left', va='top', fontsize=11, fontweight='bold',
                family='monospace')

        if status.ai_turbulence is not None:
            ai_text = f"• AI Basket Turbulence: {status.ai_turbulence:.1f}"
        else:
            ai_text = "• AI Basket Turbulence: N/A"
        ax.text(right_x + 0.02, y_start, ai_text,
                ha='left', va='top', fontsize=10, family='monospace')

        # Interpretation - right side middle
        interp_y = y_start - 2 * line_height
        ax.text(right_x, interp_y + line_height, "INTERPRETATION:",
                ha='left', va='top', fontsize=11, fontweight='bold',
                family='monospace')

        for i, interp in enumerate(signal_gen.get_interpretation(status.regime)[:3]):
            ax.text(right_x + 0.02, interp_y - i * line_height, f"• {interp}",
                    ha='left', va='top', fontsize=9, family='monospace')

        # Recommended Actions - bottom
        action_y = 0.20
        ax.text(0.08, action_y, "RECOMMENDED ACTIONS:",
                ha='left', va='top', fontsize=11, fontweight='bold',
                family='monospace')

        actions = signal_gen.get_recommended_actions(status.regime)
        for i, action in enumerate(actions[:3]):
            ax.text(0.10, action_y - (i + 1) * 0.042, f"• {action}",
                    ha='left', va='top', fontsize=9, family='monospace')

        # Thresholds - bottom right
        ax.text(right_x, action_y, "THRESHOLDS:",
                ha='left', va='top', fontsize=11, fontweight='bold',
                family='monospace')
        ax.text(right_x + 0.02, action_y - 0.042,
                f"• Warning (P95): {status.warning_threshold:.1f}",
                ha='left', va='top', fontsize=9, family='monospace')
        ax.text(right_x + 0.02, action_y - 0.084,
                f"• Extreme (P99): {status.extreme_threshold:.1f}",
                ha='left', va='top', fontsize=9, family='monospace')

    def render_divergence_chart(
        self,
        ax: plt.Axes,
        turbulence: pd.Series,
        spx_prices: pd.Series,
        warning_threshold,
        extreme_threshold,
        divergence: pd.Series,
        vix: Optional[pd.Series] = None,
    ) -> None:
        """
        Render the divergence detector chart with optional VIX subplot.

        ``warning_threshold`` and ``extreme_threshold`` may be scalars or
        pd.Series (expanding thresholds).
        """
        common_idx = turbulence.index.intersection(spx_prices.index)
        turb = turbulence.loc[common_idx]
        spx = spx_prices.loc[common_idx]
        div = divergence.reindex(common_idx).fillna(False)

        # Resolve thresholds to arrays for plotting
        if isinstance(warning_threshold, pd.Series):
            warn_vals = warning_threshold.reindex(common_idx)
        else:
            warn_vals = pd.Series(warning_threshold, index=common_idx)

        if isinstance(extreme_threshold, pd.Series):
            ext_vals = extreme_threshold.reindex(common_idx)
        else:
            ext_vals = pd.Series(extreme_threshold, index=common_idx)

        # Create twin axis for SPX
        ax2 = ax.twinx()

        # Plot turbulence on left axis
        ax.plot(turb.index, turb.values, color=self.config.color_turbulence,
                linewidth=1.2, label='Market Turbulence', zorder=3)

        # Plot SPX on right axis
        ax2.plot(spx.index, spx.values, color=self.config.color_spx,
                 linewidth=1.5, label='SPX/SPY', zorder=3)

        # Plot 50-day MA
        ma50 = spx.rolling(window=self.config.ma_long).mean()
        ax2.plot(spx.index, ma50.values, color=self.config.color_ma50,
                 linewidth=1, linestyle='--', label='50-day MA', alpha=0.7, zorder=2)

        # Draw threshold lines (now rolling)
        ax.plot(warn_vals.index, warn_vals.values,
                color=self.config.color_warning, linestyle='--', linewidth=1.5,
                label=f'Warning (P95)', zorder=2)
        ax.plot(ext_vals.index, ext_vals.values,
                color=self.config.color_extreme, linestyle='--', linewidth=1.5,
                label=f'Extreme (P99)', zorder=2)

        # Green shading for divergence periods
        in_divergence = False
        start_date = None

        for i, (date, is_div) in enumerate(div.items()):
            if is_div and not in_divergence:
                start_date = date
                in_divergence = True
            elif not is_div and in_divergence:
                ax.axvspan(start_date, date, alpha=0.3,
                          color=self.config.color_divergence, zorder=1)
                in_divergence = False

        if in_divergence and start_date is not None:
            ax.axvspan(start_date, div.index[-1], alpha=0.3,
                      color=self.config.color_divergence, zorder=1)

        # Formatting
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Market Turbulence', fontsize=11, color=self.config.color_turbulence)
        ax2.set_ylabel('SPX/SPY Level', fontsize=11, color=self.config.color_spx)

        ax.tick_params(axis='y', labelcolor=self.config.color_turbulence)
        ax2.tick_params(axis='y', labelcolor=self.config.color_spx)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        turb_max = max(turb.max(), ext_vals.dropna().max() * 1.2) if len(ext_vals.dropna()) > 0 else turb.max() * 1.2
        ax.set_ylim(0, turb_max)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        div_patch = mpatches.Patch(color=self.config.color_divergence, alpha=0.3,
                                   label='Divergence (High Turb + SPX Rising)')
        all_handles = lines1 + lines2 + [div_patch]
        all_labels = labels1 + labels2 + ['Divergence (High Turb + SPX Rising)']

        ax.legend(all_handles, all_labels, loc='upper left', fontsize=8,
                  framealpha=0.9)

        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    def render_vix_subplot(
        self,
        ax: plt.Axes,
        vix: pd.Series,
        turbulence: pd.Series,
    ) -> None:
        """Render a VIX subplot below the main divergence chart."""
        common_idx = turbulence.index.intersection(vix.dropna().index)
        if len(common_idx) == 0:
            ax.text(0.5, 0.5, 'VIX data unavailable', ha='center', va='center')
            return

        vix_plot = vix.reindex(common_idx)

        ax.plot(vix_plot.index, vix_plot.values, color='#8B0000',
                linewidth=1.2, label='VIX')
        ax.fill_between(vix_plot.index, 0, vix_plot.values,
                        alpha=0.15, color='#8B0000')

        # Draw override thresholds
        warn_level = getattr(self.config, 'vix_warning_level', 40.0)
        crit_level = getattr(self.config, 'vix_critical_level', 60.0)
        ax.axhline(y=warn_level, color='orange', linestyle=':', linewidth=1,
                   label=f'VIX Warning ({warn_level:.0f})')
        ax.axhline(y=crit_level, color='red', linestyle=':', linewidth=1,
                   label=f'VIX Critical ({crit_level:.0f})')

        ax.set_ylabel('VIX', fontsize=10, color='#8B0000')
        ax.tick_params(axis='y', labelcolor='#8B0000')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
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
    ) -> plt.Figure:
        """
        Render the complete dashboard.

        Now includes a VIX subplot at the bottom when VIX data is available.
        ``warning_threshold`` and ``extreme_threshold`` may be scalars or Series.
        """
        has_vix = vix is not None and not vix.empty

        # Layout: status panel (30%), main chart (50%), VIX subplot (20%)
        if has_vix:
            fig = plt.figure(figsize=(self.config.figure_width, self.config.figure_height + 2))
            gs = fig.add_gridspec(3, 1, height_ratios=[0.30, 0.48, 0.22], hspace=0.20)
        else:
            fig = plt.figure(figsize=(self.config.figure_width, self.config.figure_height))
            gs = fig.add_gridspec(2, 1, height_ratios=[0.35, 0.65], hspace=0.15)

        # Status panel
        ax_status = fig.add_subplot(gs[0])
        self.render_status_panel(ax_status, status, signal_gen)

        # Resolve thresholds for chart
        w_thresh = warning_threshold if warning_threshold is not None else status.warning_threshold
        e_thresh = extreme_threshold if extreme_threshold is not None else status.extreme_threshold

        # Divergence chart
        ax_chart = fig.add_subplot(gs[1])
        self.render_divergence_chart(
            ax_chart, turbulence, spx_prices,
            w_thresh, e_thresh, divergence,
            vix=vix,
        )

        # VIX subplot
        if has_vix:
            ax_vix = fig.add_subplot(gs[2])
            self.render_vix_subplot(ax_vix, vix, turbulence)

        fig.suptitle(
            "DIVERGENCE DETECTOR: Turbulence vs SPX (ENLARGED)\n"
            "(Green shading = High turbulence while SPX rising - YOUR WARNING SIGNAL)",
            fontsize=14, fontweight='bold', y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

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
) -> pd.DataFrame:
    """
    Save computed features to CSV.

    ``warning_threshold`` and ``extreme_threshold`` may be scalars or pd.Series.
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

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_file)
    logger.info(f"Features saved to {output_file}")

    return features
