"""
Main orchestration module for Market Turbulence Monitoring System.

This module ties together all components and provides CLI interface.
Implements Jordi Visser's three-regime framework with contagion detection.
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
    logger.info("Jordi Visser Framework — Three-Regime Model")
    logger.info("=" * 60)
    logger.info(f"Date range: {config.start_date} to {config.end_date}")
    logger.info(f"Divergence rule: {config.divergence_rule}")
    logger.info(f"Covariance method: {config.covariance_method}")
    logger.info(f"Baseline method: {getattr(config, 'baseline_method', 'calm_period')}")
    logger.info(f"Include ratio pairs: {getattr(config, 'include_ratio_pairs', True)}")

    # Step 1: Fetch data (now returns 5 items including hyg_ief_ratio)
    logger.info("\n[1/7] Fetching market data...")
    fetcher = DataFetcher(config)
    turbulence_prices, turbulence_returns, ai_returns, vix, hyg_ief_ratio = fetcher.fetch_all_data()

    # Clean data
    turbulence_returns_clean = clean_data(turbulence_returns)
    logger.info(f"Using {len(turbulence_returns_clean.columns)} assets for turbulence: "
                f"{list(turbulence_returns_clean.columns)}")

    # Get SPX data
    spx_prices = fetcher.get_spx_data(turbulence_prices)

    # Step 2: Compute turbulence (baseline-anchored)
    logger.info("\n[2/7] Computing market turbulence (baseline-anchored)...")
    calc = TurbulenceCalculator(config)

    turbulence = calc.compute_rolling_turbulence(
        turbulence_returns_clean,
        vix=vix,
    )

    # Step 3: Compute expanding thresholds
    logger.info("\n[3/7] Computing expanding thresholds...")
    signal_gen = SignalGenerator(config)

    warning_thresh_series, extreme_thresh_series = signal_gen.compute_expanding_thresholds(turbulence)

    # Static thresholds for backward compat
    warning_thresh_static, extreme_thresh_static = calc.compute_thresholds(turbulence)

    # Days elevated
    latest_warn = warning_thresh_series.dropna()
    days_elevated_threshold = latest_warn.iloc[-1] if len(latest_warn) > 0 else warning_thresh_static
    days_elevated = calc.compute_days_elevated(turbulence, days_elevated_threshold)

    # Step 4: Compute Visser regime classification
    logger.info("\n[4/7] Computing Visser regime classification...")

    regime = signal_gen.compute_regime_series(
        turbulence, warning_thresh_series, extreme_thresh_series,
        spx_prices=spx_prices, vix=vix,
    )

    # Contagion detection
    hyg_ief_slope = None
    contagion = None
    if hyg_ief_ratio is not None and not hyg_ief_ratio.empty:
        logger.info("Running contagion detection (HYG/IEF slope)...")
        hyg_ief_slope, contagion = signal_gen.check_contagion(hyg_ief_ratio)
        regime = signal_gen.apply_contagion_to_regime(regime, contagion)

    # VIX override (catches edge cases)
    regime, vix_override = signal_gen.apply_vix_override(regime, vix)

    # Step 5: Compute dispersion
    logger.info("\n[5/7] Computing cross-sectional dispersion...")
    dispersion, dispersion_pctile = signal_gen.compute_dispersion(turbulence_returns_clean)

    # Compute AI turbulence
    ai_turbulence = None
    if not ai_returns.empty:
        logger.info("Computing AI sector turbulence...")
        ai_returns_clean = clean_data(ai_returns, min_valid_ratio=0.5)
        if not ai_returns_clean.empty:
            ai_turbulence = compute_ai_turbulence(ai_returns_clean, config)

    # Step 6: Generate signals and status
    logger.info("\n[6/7] Generating signals and status...")
    divergence = signal_gen.detect_divergence(
        turbulence, spx_prices, warning_thresh_static, config.divergence_rule
    )

    # 90-day regime summary
    regime_last_90 = regime.iloc[-90:] if len(regime) >= 90 else regime
    regime_counts = regime_last_90.value_counts().to_dict()

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
        hyg_ief_slope=hyg_ief_slope,
        contagion_series=contagion,
        dispersion=dispersion,
        dispersion_pctile=dispersion_pctile,
    )

    # Print status to console
    status_text = format_status_text(status, signal_gen)
    print("\n" + status_text + "\n")

    # Step 7: Render dashboard and save outputs
    logger.info("\n[7/7] Rendering dashboard and saving outputs...")

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
        regime=regime,
        hyg_ief_ratio=hyg_ief_ratio,
        contagion=contagion,
        regime_counts=regime_counts,
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
        hyg_ief_ratio=hyg_ief_ratio,
        hyg_ief_slope=hyg_ief_slope,
        contagion=contagion,
        dispersion=dispersion,
        dispersion_pctile=dispersion_pctile,
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

    parser.add_argument(
        '--no_ratio_pairs',
        action='store_true',
        help='Disable IWM/QQQ and HYG/IEF ratio pair columns in the covariance matrix'
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
    if args.no_ratio_pairs:
        config_kwargs['include_ratio_pairs'] = False

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
