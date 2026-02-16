"""
Data fetching and processing module.

Handles downloading market data from yfinance, cleaning, and computing returns.
Includes fallback to simulated data for testing when API access is unavailable.
Also computes ratio pair columns (IWM/QQQ, HYG/IEF) for the turbulence matrix.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from .config import TurbulenceConfig

logger = logging.getLogger(__name__)


def generate_simulated_data(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Generate realistic simulated market data for testing.

    Creates synthetic price data with realistic properties:
    - Trending behavior
    - Volatility clustering
    - Cross-asset correlations
    - Periodic stress events

    Args:
        tickers: List of ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        DataFrame of simulated prices
    """
    np.random.seed(42)  # For reproducibility

    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    n_tickers = len(tickers)

    logger.info(f"Generating simulated data for {n_tickers} tickers over {n_days} days")

    # Base prices for different asset classes
    base_prices = {
        'SPY': 450, 'QQQ': 380, 'IWM': 200, 'EFA': 75, 'EEM': 42,
        'TLT': 100, 'IEF': 105, 'HYG': 78, 'GLD': 180, 'USO': 75,
        'UUP': 28, 'NVDA': 500, 'MSFT': 380, 'GOOGL': 140,
        'AMZN': 170, 'SMH': 200,
        # SaaS basket
        'CRM': 280, 'NOW': 700, 'SNOW': 160, 'DDOG': 120,
        'CRWD': 300, 'WDAY': 250, 'ZS': 200,
        '^VIX': 18,
    }

    # Correlation structure
    ticker_types = {
        'equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'SMH',
                   'CRM', 'NOW', 'SNOW', 'DDOG', 'CRWD', 'WDAY', 'ZS'],
        'bond': ['TLT', 'IEF'],
        'credit': ['HYG'],
        'commodity': ['GLD', 'USO'],
        'dollar': ['UUP'],
        'volatility': ['^VIX']
    }

    # Create correlation matrix
    corr_matrix = np.eye(n_tickers)
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i == j:
                continue
            for type_name, type_tickers in ticker_types.items():
                if t1 in type_tickers and t2 in type_tickers:
                    corr_matrix[i, j] = 0.7 + np.random.uniform(-0.1, 0.1)
            if (t1 in ticker_types['equity'] and t2 in ticker_types['bond']) or \
               (t2 in ticker_types['equity'] and t1 in ticker_types['bond']):
                corr_matrix[i, j] = -0.3 + np.random.uniform(-0.1, 0.1)
            if (t1 in ticker_types['equity'] and t2 in ticker_types['volatility']) or \
               (t2 in ticker_types['equity'] and t1 in ticker_types['volatility']):
                corr_matrix[i, j] = -0.6 + np.random.uniform(-0.1, 0.1)

    # Make symmetric and positive definite
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)

    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    eigvals = np.maximum(eigvals, 0.01)
    corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

    base_vol = np.array([0.01 if t not in ['^VIX'] else 0.05 for t in tickers])

    L = np.linalg.cholesky(corr_matrix)

    prices = np.zeros((n_days, n_tickers))
    prices[0] = [base_prices.get(t, 100) for t in tickers]

    vol_state = np.ones(n_tickers)

    for t in range(1, n_days):
        vol_state = 0.9 * vol_state + 0.1 * np.abs(np.random.randn(n_tickers))
        vol_state = np.clip(vol_state, 0.5, 3.0)

        if t in [int(n_days * 0.3), int(n_days * 0.6), int(n_days * 0.85)]:
            vol_state *= 2.5

        z = np.random.randn(n_tickers)
        correlated_z = L @ z

        daily_vol = base_vol * vol_state
        returns = 0.0003 + daily_vol * correlated_z

        vix_idx = [i for i, t2 in enumerate(tickers) if t2 == '^VIX']
        spy_idx = [i for i, t2 in enumerate(tickers) if t2 == 'SPY']
        if vix_idx and spy_idx:
            returns[vix_idx[0]] = -returns[spy_idx[0]] * 3 + np.random.randn() * 0.02

        prices[t] = prices[t-1] * np.exp(returns)

    df = pd.DataFrame(prices, index=dates, columns=tickers)

    logger.info(f"Generated simulated data: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


class DataFetcher:
    """Handles fetching and processing market data."""

    def __init__(self, config: TurbulenceConfig):
        self.config = config

    def fetch_ticker_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_simulated: bool = False
    ) -> pd.DataFrame:
        """
        Fetch adjusted close prices for a list of tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            use_simulated: Force use of simulated data

        Returns:
            DataFrame with adjusted close prices, tickers as columns
        """
        if not YFINANCE_AVAILABLE or use_simulated:
            logger.info("Using simulated data (yfinance unavailable or simulated mode requested)")
            return generate_simulated_data(tickers, start_date, end_date)

        successful_tickers = []
        all_data = {}

        for ticker in tickers:
            try:
                logger.info(f"Fetching data for {ticker}...")
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )

                if data.empty:
                    logger.warning(f"No data returned for {ticker}, skipping")
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    close_col = ('Close', ticker)
                    if close_col in data.columns:
                        all_data[ticker] = data[close_col]
                    else:
                        all_data[ticker] = data['Close'].iloc[:, 0]
                else:
                    all_data[ticker] = data['Close']

                successful_tickers.append(ticker)
                logger.info(f"Successfully fetched {len(all_data[ticker])} rows for {ticker}")

            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
                continue

        if not all_data:
            logger.warning("Failed to fetch any live data. Falling back to simulated data.")
            return generate_simulated_data(tickers, start_date, end_date)

        prices_df = pd.DataFrame(all_data)
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.sort_index()

        logger.info(f"Successfully fetched data for {len(successful_tickers)} tickers: {successful_tickers}")

        return prices_df

    def compute_returns(
        self,
        prices: pd.DataFrame,
        method: str = 'log'
    ) -> pd.DataFrame:
        """
        Compute returns from price data.

        Args:
            prices: DataFrame of prices
            method: 'log' for log returns, 'simple' for simple returns

        Returns:
            DataFrame of returns
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        returns = returns.dropna(how='all')

        return returns

    def compute_moving_averages(
        self,
        prices: pd.Series,
        windows: List[int]
    ) -> Dict[int, pd.Series]:
        """
        Compute moving averages for a price series.

        Args:
            prices: Price series
            windows: List of window sizes

        Returns:
            Dictionary mapping window size to MA series
        """
        mas = {}
        for window in windows:
            mas[window] = prices.rolling(window=window).mean()
        return mas

    def add_ratio_pairs(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Add ratio pair log-return columns to the returns matrix.

        Adds:
        - IWM_QQQ_ratio_ret: daily log-change of IWM/QQQ ratio
        - HYG_IEF_ratio_ret: daily log-change of HYG/IEF ratio

        Also returns the raw HYG/IEF price ratio for contagion detection.

        Args:
            prices: Price DataFrame (must contain IWM, QQQ, HYG, IEF)
            returns: Returns DataFrame to augment

        Returns:
            Tuple of (augmented_returns, hyg_ief_price_ratio)
        """
        augmented = returns.copy()
        hyg_ief_ratio = pd.Series(dtype=float)

        # IWM/QQQ ratio returns
        if 'IWM' in prices.columns and 'QQQ' in prices.columns:
            iwm_qqq_ratio = prices['IWM'] / prices['QQQ']
            iwm_qqq_ret = np.log(iwm_qqq_ratio / iwm_qqq_ratio.shift(1))
            augmented['IWM_QQQ_ratio_ret'] = iwm_qqq_ret.reindex(augmented.index)
            logger.info("Added IWM/QQQ ratio returns to turbulence matrix")
        else:
            logger.warning("IWM or QQQ not in prices — skipping IWM/QQQ ratio")

        # HYG/IEF ratio returns
        if 'HYG' in prices.columns and 'IEF' in prices.columns:
            hyg_ief_ratio = prices['HYG'] / prices['IEF']
            hyg_ief_ret = np.log(hyg_ief_ratio / hyg_ief_ratio.shift(1))
            augmented['HYG_IEF_ratio_ret'] = hyg_ief_ret.reindex(augmented.index)
            logger.info("Added HYG/IEF ratio returns to turbulence matrix")
        else:
            logger.warning("HYG or IEF not in prices — skipping HYG/IEF ratio")

        return augmented, hyg_ief_ratio

    def fetch_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Fetch all required data for the turbulence system.

        Returns:
            Tuple of:
            - turbulence_prices: Prices for turbulence basket
            - turbulence_returns: Returns for turbulence basket (with ratio pairs if enabled)
            - ai_infra_returns: Returns for AI infrastructure basket
            - saas_returns: Returns for SaaS basket
            - vix: VIX series
            - hyg_ief_ratio: Raw HYG/IEF price ratio for contagion detection
        """
        use_sim = getattr(self.config, 'use_simulated', False)

        # Fetch turbulence basket
        logger.info("Fetching turbulence basket data...")
        turbulence_prices = self.fetch_ticker_data(
            self.config.turbulence_tickers,
            self.config.start_date,
            self.config.end_date,
            use_simulated=use_sim
        )

        # Fetch VIX
        logger.info("Fetching VIX data...")
        try:
            vix_data = self.fetch_ticker_data(
                [self.config.vix_ticker],
                self.config.start_date,
                self.config.end_date,
                use_simulated=use_sim
            )
            vix = vix_data[self.config.vix_ticker] if self.config.vix_ticker in vix_data.columns else vix_data.iloc[:, 0]
        except Exception as e:
            logger.warning(f"Failed to fetch VIX: {e}. Using NaN.")
            vix = pd.Series(np.nan, index=turbulence_prices.index)

        # Fetch AI Infrastructure basket (hyperscalers / semis)
        logger.info("Fetching AI Infrastructure basket data...")
        try:
            ai_infra_prices = self.fetch_ticker_data(
                self.config.ai_infra_tickers,
                self.config.start_date,
                self.config.end_date,
                use_simulated=use_sim
            )
            ai_infra_returns = self.compute_returns(ai_infra_prices)
        except Exception as e:
            logger.warning(f"Failed to fetch AI Infra basket: {e}. Using empty DataFrame.")
            ai_infra_returns = pd.DataFrame()

        # Fetch SaaS basket (application-layer names)
        logger.info("Fetching SaaS basket data...")
        try:
            saas_prices = self.fetch_ticker_data(
                self.config.saas_tickers,
                self.config.start_date,
                self.config.end_date,
                use_simulated=use_sim
            )
            saas_returns = self.compute_returns(saas_prices)
        except Exception as e:
            logger.warning(f"Failed to fetch SaaS basket: {e}. Using empty DataFrame.")
            saas_returns = pd.DataFrame()

        # Compute returns
        turbulence_returns = self.compute_returns(turbulence_prices)

        # Add ratio pair columns if configured
        hyg_ief_ratio = pd.Series(dtype=float)
        if getattr(self.config, 'include_ratio_pairs', True):
            turbulence_returns, hyg_ief_ratio = self.add_ratio_pairs(
                turbulence_prices, turbulence_returns
            )

        # Align all data to common dates
        common_index = turbulence_returns.index
        if not vix.empty:
            vix = vix.reindex(common_index)
        if not ai_infra_returns.empty:
            ai_infra_returns = ai_infra_returns.reindex(common_index)
        if not saas_returns.empty:
            saas_returns = saas_returns.reindex(common_index)
        if not hyg_ief_ratio.empty:
            hyg_ief_ratio = hyg_ief_ratio.reindex(common_index)

        return turbulence_prices, turbulence_returns, ai_infra_returns, saas_returns, vix, hyg_ief_ratio

    def get_spx_data(self, prices: pd.DataFrame) -> pd.Series:
        """
        Extract SPX/SPY data from prices DataFrame.

        Args:
            prices: Full prices DataFrame

        Returns:
            SPX price series
        """
        spx_ticker = self.config.spx_ticker
        if spx_ticker in prices.columns:
            return prices[spx_ticker]
        elif 'SPY' in prices.columns:
            return prices['SPY']
        else:
            raise ValueError(f"SPX ticker {spx_ticker} not found in data")


def clean_data(df: pd.DataFrame, min_valid_ratio: float = 0.7) -> pd.DataFrame:
    """
    Clean DataFrame by handling missing values.

    Args:
        df: Input DataFrame
        min_valid_ratio: Minimum ratio of valid (non-NaN) values required per column

    Returns:
        Cleaned DataFrame
    """
    valid_counts = df.notna().sum()
    min_valid = int(len(df) * min_valid_ratio)
    valid_columns = valid_counts[valid_counts >= min_valid].index.tolist()

    if len(valid_columns) < len(df.columns):
        removed = set(df.columns) - set(valid_columns)
        logger.warning(f"Removed columns with insufficient data: {removed}")

    df_clean = df[valid_columns].copy()

    df_clean = df_clean.ffill().bfill()

    df_clean = df_clean.dropna()

    return df_clean
