"""
Dashboard visualization module.

Creates the combined market turbulence dashboard with status panel and
divergence detector chart using Matplotlib.
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
        # Try different style names for compatibility
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
        """
        Render the status panel as a text box.

        Args:
            ax: Matplotlib axes to render on
            status: Current market status
            signal_gen: Signal generator for interpretations
        """
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Background color based on regime
        bg_colors = {
            'HEALTHY': '#e8f5e9',  # Light green
            'ELEVATED': '#fff3e0',  # Light orange
            'CRISIS': '#ffebee'    # Light red
        }
        bg_color = bg_colors.get(status.regime, '#f5f5f5')

        # Draw background rectangle
        rect = mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02",
            facecolor=bg_color,
            edgecolor='#333333',
            linewidth=2
        )
        ax.add_patch(rect)

        # Text formatting
        date_str = status.date.strftime("%Y-%m-%d")

        # Title
        ax.text(0.5, 0.92, f"CURRENT IMMUNE SYSTEM STATUS — {date_str}",
                ha='center', va='top', fontsize=14, fontweight='bold',
                family='monospace')

        # Regime indicator colors
        regime_colors = {
            'HEALTHY': self.config.status_healthy,
            'ELEVATED': self.config.status_elevated,
            'CRISIS': self.config.status_crisis
        }

        # Warning level
        ax.text(0.5, 0.83, f"WARNING LEVEL: {status.regime}",
                ha='center', va='top', fontsize=16, fontweight='bold',
                color=regime_colors.get(status.regime, 'black'),
                family='monospace')

        # Metrics - left column
        y_start = 0.72
        line_height = 0.055
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

        # Divergence indicator
        div_color = self.config.status_crisis if status.divergence_active else 'black'
        div_text = "YES ⚠" if status.divergence_active else "NO"
        ax.text(left_x + 0.02, y_start - 5 * line_height, f"• Divergence Active: {div_text}",
                ha='left', va='top', fontsize=10, family='monospace',
                color=div_color, fontweight='bold' if status.divergence_active else 'normal')

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
        action_y = 0.22
        ax.text(0.08, action_y, "RECOMMENDED ACTIONS:",
                ha='left', va='top', fontsize=11, fontweight='bold',
                family='monospace')

        actions = signal_gen.get_recommended_actions(status.regime)
        for i, action in enumerate(actions[:3]):
            ax.text(0.10, action_y - (i + 1) * 0.045, f"• {action}",
                    ha='left', va='top', fontsize=9, family='monospace')

        # Thresholds - bottom right
        ax.text(right_x, action_y, "THRESHOLDS:",
                ha='left', va='top', fontsize=11, fontweight='bold',
                family='monospace')
        ax.text(right_x + 0.02, action_y - 0.045,
                f"• Warning (P95): {status.warning_threshold:.1f}",
                ha='left', va='top', fontsize=9, family='monospace')
        ax.text(right_x + 0.02, action_y - 0.09,
                f"• Extreme (P99): {status.extreme_threshold:.1f}",
                ha='left', va='top', fontsize=9, family='monospace')

    def render_divergence_chart(
        self,
        ax: plt.Axes,
        turbulence: pd.Series,
        spx_prices: pd.Series,
        warning_threshold: float,
        extreme_threshold: float,
        divergence: pd.Series
    ) -> None:
        """
        Render the divergence detector chart.

        Args:
            ax: Matplotlib axes to render on
            turbulence: Turbulence series
            spx_prices: SPX price series
            warning_threshold: Warning threshold value
            extreme_threshold: Extreme threshold value
            divergence: Boolean series indicating divergence
        """
        # Align data
        common_idx = turbulence.index.intersection(spx_prices.index)
        turb = turbulence.loc[common_idx]
        spx = spx_prices.loc[common_idx]
        div = divergence.reindex(common_idx).fillna(False)

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

        # Draw threshold lines
        ax.axhline(y=warning_threshold, color=self.config.color_warning,
                   linestyle='--', linewidth=1.5, label=f'Warning (P95): {warning_threshold:.1f}',
                   zorder=2)
        ax.axhline(y=extreme_threshold, color=self.config.color_extreme,
                   linestyle='--', linewidth=1.5, label=f'Extreme (P99): {extreme_threshold:.1f}',
                   zorder=2)

        # Add green shading for divergence periods
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

        # Handle if still in divergence at end
        if in_divergence and start_date is not None:
            ax.axvspan(start_date, div.index[-1], alpha=0.3,
                      color=self.config.color_divergence, zorder=1)

        # Formatting
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Market Turbulence', fontsize=11, color=self.config.color_turbulence)
        ax2.set_ylabel('SPX/SPY Level', fontsize=11, color=self.config.color_spx)

        ax.tick_params(axis='y', labelcolor=self.config.color_turbulence)
        ax2.tick_params(axis='y', labelcolor=self.config.color_spx)

        # X-axis date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Set y limits with some padding
        turb_max = max(turb.max(), extreme_threshold * 1.2)
        ax.set_ylim(0, turb_max)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        # Add divergence patch to legend
        div_patch = mpatches.Patch(color=self.config.color_divergence, alpha=0.3,
                                   label='Divergence (High Turb + SPX Rising)')
        all_handles = lines1 + lines2 + [div_patch]
        all_labels = labels1 + labels2 + ['Divergence (High Turb + SPX Rising)']

        ax.legend(all_handles, all_labels, loc='upper left', fontsize=8,
                  framealpha=0.9)

        # Grid
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    def render_dashboard(
        self,
        turbulence: pd.Series,
        spx_prices: pd.Series,
        status: MarketStatus,
        signal_gen: SignalGenerator,
        divergence: pd.Series,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Render the complete dashboard.

        Args:
            turbulence: Turbulence series
            spx_prices: SPX price series
            status: Current market status
            signal_gen: Signal generator
            divergence: Divergence boolean series
            output_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        # Create figure with GridSpec
        fig = plt.figure(figsize=(self.config.figure_width, self.config.figure_height))

        # Use GridSpec for layout: status panel on top (30%), chart below (70%)
        gs = fig.add_gridspec(2, 1, height_ratios=[0.35, 0.65], hspace=0.15)

        # Status panel
        ax_status = fig.add_subplot(gs[0])
        self.render_status_panel(ax_status, status, signal_gen)

        # Divergence chart
        ax_chart = fig.add_subplot(gs[1])
        self.render_divergence_chart(
            ax_chart,
            turbulence,
            spx_prices,
            status.warning_threshold,
            status.extreme_threshold,
            divergence
        )

        # Main title
        fig.suptitle(
            "DIVERGENCE DETECTOR: Turbulence vs SPX (ENLARGED)\n"
            "(Green shading = High turbulence while SPX rising - YOUR WARNING SIGNAL)",
            fontsize=14, fontweight='bold', y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save if path provided
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
    warning_threshold: float,
    extreme_threshold: float,
    output_path: str
) -> pd.DataFrame:
    """
    Save computed features to CSV.

    Args:
        turbulence: Turbulence series
        spx_prices: SPX price series
        vix: VIX series
        divergence: Divergence boolean series
        days_elevated: Days elevated series
        regime: Regime classification series
        warning_threshold: Warning threshold
        extreme_threshold: Extreme threshold
        output_path: Path to save CSV

    Returns:
        DataFrame of features
    """
    # Compute 50-day MA
    spx_ma50 = spx_prices.rolling(window=50).mean()

    # Align all series to common index
    common_idx = turbulence.index

    features = pd.DataFrame({
        'spx_close': spx_prices.reindex(common_idx),
        'spx_50dma': spx_ma50.reindex(common_idx),
        'spx_above_50dma': (spx_prices > spx_ma50).reindex(common_idx),
        'vix': vix.reindex(common_idx),
        'turbulence': turbulence,
        'warning_threshold': warning_threshold,
        'extreme_threshold': extreme_threshold,
        'regime': regime.reindex(common_idx),
        'divergence': divergence.reindex(common_idx),
        'days_elevated': days_elevated.reindex(common_idx)
    })

    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_file)
    logger.info(f"Features saved to {output_file}")

    return features
