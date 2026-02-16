"""
Main orchestration module for Market Turbulence Monitoring System.

This module ties together all components and provides CLI interface.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from .config import TurbulenceConfig, get_config
from .data import DataFetcher, clean_data
from .turbulence import TurbulenceCalculator, compute_ai_turbulence
from .signals import SignalGenerator, format_status_text
from .dashboard import DashboardRenderer, save_features_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(config: TurbulenceConfig) -> None:
    """
    Run the complete market turbulence pipeline.

    Args:
        config: Configuration object
    """
    logger.info("=" * 60)
    logger.info("MARKET TURBULENCE MONITORING SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Date range: {config.start_date} to {config.end_date}")
    logger.info(f"Divergence rule: {config.divergence_rule}")
    logger.info(f"Covariance method: {config.covariance_method}")
    logger.info(f"Baseline method: {getattr(config, 'baseline_method', 'calm_period')}")

    # Step 1: Fetch data
    logger.info("\n[1/6] Fetching market data...")
    fetcher = DataFetcher(config)
    turbulence_prices, turbulence_returns, ai_returns, vix = fetcher.fetch_all_data()

    # Clean data
    turbulence_returns_clean = clean_data(turbulence_returns)
    logger.info(f"Using {len(turbulence_returns_clean.columns)} assets for turbulence: "
                f"{list(turbulence_returns_clean.columns)}")

    # Get SPX data
    spx_prices = fetcher.get_spx_data(turbulence_prices)

    # Step 2: Compute turbulence (now baseline-anchored)
    logger.info("\n[2/6] Computing market turbulence (baseline-anchored)...")
    calc = TurbulenceCalculator(config)

    # Compute baseline covariance from calm period, then score
    turbulence = calc.compute_rolling_turbulence(
        turbulence_returns_clean,
        vix=vix,
    )

    # Step 3: Compute expanding thresholds + regimes + VIX override
    logger.info("\n[3/6] Computing expanding thresholds and regimes...")
    signal_gen = SignalGenerator(config)

    # Expanding thresholds (Fix 2)
    warning_thresh_series, extreme_thresh_series = signal_gen.compute_expanding_thresholds(turbulence)

    # Also compute static thresholds (for backward compat / status display)
    warning_thresh_static, extreme_thresh_static = calc.compute_thresholds(turbulence)

    # For days_elevated, use the latest expanding threshold value
    latest_warn = warning_thresh_series.dropna()
    days_elevated_threshold = latest_warn.iloc[-1] if len(latest_warn) > 0 else warning_thresh_static
    days_elevated = calc.compute_days_elevated(turbulence, days_elevated_threshold)

    # Compute regimes using expanding thresholds
    regime = signal_gen.compute_regime_series(
        turbulence, warning_thresh_series, extreme_thresh_series
    )

    # Apply VIX override (Fix 3)
    regime, vix_override = signal_gen.apply_vix_override(regime, vix)

    # Compute AI turbulence
    ai_turbulence = None
    if not ai_returns.empty:
        logger.info("Computing AI sector turbulence...")
        ai_returns_clean = clean_data(ai_returns, min_valid_ratio=0.5)
        if not ai_returns_clean.empty:
            ai_turbulence = compute_ai_turbulence(ai_returns_clean, config)

    # Step 4: Generate signals
    logger.info("\n[4/6] Generating divergence signals...")
    # Use the latest static threshold for divergence detection
    divergence = signal_gen.detect_divergence(
        turbulence, spx_prices, warning_thresh_static, config.divergence_rule
    )

    # Generate current status
    status = signal_gen.generate_status(
        turbulence=turbulence,
        spx_prices=spx_prices,
        vix=vix,
        warning_threshold=warning_thresh_series,
        extreme_threshold=extreme_thresh_series,
        days_elevated_series=days_elevated,
        divergence=divergence,
        ai_turbulence=ai_turbulence,
        regime_series=regime,
        vix_override_series=vix_override,
    )

    # Step 5: Print status to console
    logger.info("\n[5/6] Current Market Status:")
    status_text = format_status_text(status, signal_gen)
    print("\n" + status_text + "\n")

    # Step 6: Render dashboard and save outputs
    logger.info("\n[6/6] Rendering dashboard and saving outputs...")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = DashboardRenderer(config)
    dashboard_path = output_dir / config.dashboard_filename

    fig = renderer.render_dashboard(
        turbulence=turbulence,
        spx_prices=spx_prices,
        status=status,
        signal_gen=signal_gen,
        divergence=divergence,
        output_path=str(dashboard_path),
        vix=vix,
        warning_threshold=warning_thresh_series,
        extreme_threshold=extreme_thresh_series,
    )
    plt.close(fig)

    # Save features CSV
    features_path = output_dir / config.features_filename
    features = save_features_csv(
        turbulence=turbulence,
        spx_prices=spx_prices,
        vix=vix,
        divergence=divergence,
        days_elevated=days_elevated,
        regime=regime,
        warning_threshold=warning_thresh_series,
        extreme_threshold=extreme_thresh_series,
        output_path=str(features_path),
        vix_override=vix_override,
    )

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Dashboard saved to: {dashboard_path.absolute()}")
    logger.info(f"Features CSV saved to: {features_path.absolute()}")
    logger.info(f"Data range: {turbulence.index[0].date()} to {turbulence.index[-1].date()}")
    logger.info(f"Total observations: {len(turbulence)}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Market Turbulence Monitoring System - Mahalanobis Distance Based",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main
  python -m src.main --start 2023-01-01 --end 2026-01-01
  python -m src.main --divergence_rule ma20_slope
  python -m src.main --covariance sample --lookback 126
  python -m src.main --baseline_method calm_period --baseline_vix 25
        """
    )

    parser.add_argument(
        '--start', '-s',
        type=str,
        help='Start date (YYYY-MM-DD). Default: 3 years ago'
    )

    parser.add_argument(
        '--end', '-e',
        type=str,
        help='End date (YYYY-MM-DD). Default: today'
    )

    parser.add_argument(
        '--divergence_rule', '-d',
        type=str,
        choices=['ma50', 'ma20_slope'],
        help='Divergence detection rule. Default: ma50'
    )

    parser.add_argument(
        '--covariance', '-c',
        type=str,
        choices=['ledoit_wolf', 'sample'],
        help='Covariance estimation method. Default: ledoit_wolf'
    )

    parser.add_argument(
        '--lookback', '-l',
        type=int,
        help='Lookback window for covariance estimation (trading days). Default: 252'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory. Default: output'
    )

    parser.add_argument(
        '--warning_pct',
        type=float,
        help='Warning threshold percentile. Default: 95'
    )

    parser.add_argument(
        '--extreme_pct',
        type=float,
        help='Extreme threshold percentile. Default: 99'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (debug) logging'
    )

    parser.add_argument(
        '--simulated',
        action='store_true',
        help='Use simulated data instead of fetching from yfinance'
    )

    parser.add_argument(
        '--baseline_method',
        type=str,
        choices=['calm_period', 'first_n_days', 'expanding'],
        help='Baseline covariance method. Default: calm_period'
    )

    parser.add_argument(
        '--baseline_vix',
        type=float,
        help='VIX ceiling for calm baseline period. Default: 25'
    )

    # Jupyter compatibility: detect if running inside Jupyter kernel
    if "ipykernel" in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build config from arguments
    config_kwargs = {}

    if args.start:
        config_kwargs['start_date'] = args.start
    if args.end:
        config_kwargs['end_date'] = args.end
    if args.divergence_rule:
        config_kwargs['divergence_rule'] = args.divergence_rule
    if args.covariance:
        config_kwargs['covariance_method'] = args.covariance
    if args.lookback:
        config_kwargs['lookback_window'] = args.lookback
    if args.output:
        config_kwargs['output_dir'] = args.output
    if args.warning_pct:
        config_kwargs['warning_percentile'] = args.warning_pct
    if args.extreme_pct:
        config_kwargs['extreme_percentile'] = args.extreme_pct
    if args.simulated:
        config_kwargs['use_simulated'] = True
    if args.baseline_method:
        config_kwargs['baseline_method'] = args.baseline_method
    if args.baseline_vix:
        config_kwargs['baseline_vix_threshold'] = args.baseline_vix

    try:
        config = get_config(**config_kwargs)
        run_pipeline(config)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
