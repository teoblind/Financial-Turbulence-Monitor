# Market Turbulence Monitoring System

A Python implementation of a "Market Immune System" dashboard inspired by Jordi Visser's market monitoring approach. This system uses **Mahalanobis distance** to measure market turbulence across multiple asset classes and identifies dangerous divergences when markets appear calm but underlying stress is elevated.

## Overview

The system provides:
- **Market Turbulence Score**: Mahalanobis distance-based measurement of cross-asset stress
- **Immune System Status Panel**: Real-time classification (HEALTHY / ELEVATED / CRISIS)
- **Divergence Detection**: Warning signals when turbulence is high but markets are rising
- **Interactive Dashboard**: Combined visualization with status panel and time series chart

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone or navigate to the repository
cd Financial-turbulence-monitoring-system-with-Mahalanobis-distance

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run with default settings (last 3 years of data)
python -m src.main
```

### Command Line Options

```bash
# Custom date range
python -m src.main --start 2023-01-01 --end 2026-01-01

# Use alternative divergence rule (MA20 slope instead of MA50)
python -m src.main --divergence_rule ma20_slope

# Use sample covariance instead of Ledoit-Wolf shrinkage
python -m src.main --covariance sample

# Custom lookback window (126 days = ~6 months)
python -m src.main --lookback 126

# Custom output directory
python -m src.main --output my_output

# Verbose logging
python -m src.main -v

# Full example
python -m src.main --start 2022-01-01 --end 2026-01-01 --divergence_rule ma50 --covariance ledoit_wolf -v
```

### All CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--start` | `-s` | Start date (YYYY-MM-DD) | 3 years ago |
| `--end` | `-e` | End date (YYYY-MM-DD) | Today |
| `--divergence_rule` | `-d` | Divergence rule: `ma50` or `ma20_slope` | `ma50` |
| `--covariance` | `-c` | Covariance method: `ledoit_wolf` or `sample` | `ledoit_wolf` |
| `--lookback` | `-l` | Lookback window in trading days | 252 |
| `--output` | `-o` | Output directory path | `output` |
| `--warning_pct` | | Warning threshold percentile | 95 |
| `--extreme_pct` | | Extreme threshold percentile | 99 |
| `--verbose` | `-v` | Enable debug logging | False |

## Output Files

The system generates the following outputs in the `output/` directory:

1. **`dashboard.png`**: Combined dashboard image with:
   - Status panel showing current regime and metrics
   - Divergence detector chart with turbulence, SPX, and warning signals

2. **`features.csv`**: Daily data with columns:
   - `spx_close`: S&P 500 (SPY) closing price
   - `spx_50dma`: 50-day moving average
   - `spx_above_50dma`: Boolean indicator
   - `vix`: VIX level
   - `turbulence`: Computed turbulence score
   - `warning_threshold`: 95th percentile threshold
   - `extreme_threshold`: 99th percentile threshold
   - `regime`: HEALTHY / ELEVATED / CRISIS
   - `divergence`: Boolean divergence signal
   - `days_elevated`: Consecutive days above warning threshold

3. **Console Output**: Status panel printed to terminal

## Methodology

### Market Turbulence (Mahalanobis Distance)

The turbulence metric measures how unusual the current return vector is relative to historical patterns:

```
Turbulence_t = sqrt((r_t - μ)' Σ^(-1) (r_t - μ))
```

Where:
- `r_t` = Vector of daily log returns across all assets at time t
- `μ` = Mean return vector from the reference window
- `Σ` = Covariance matrix from the reference window (estimated via Ledoit-Wolf shrinkage by default)

**Intuition**: When assets move together in unusual ways (relative to historical covariance), the Mahalanobis distance increases, indicating market stress.

### Multi-Asset Basket

Default turbulence basket includes:
- **Equities**: SPY, QQQ, IWM, EFA, EEM
- **Rates**: TLT, IEF
- **Credit**: HYG
- **Commodities**: GLD, USO
- **Dollar**: UUP

### Regime Classification

| Regime | Condition | Interpretation |
|--------|-----------|----------------|
| HEALTHY | Turbulence < P95 | Normal market conditions |
| ELEVATED | P95 ≤ Turbulence < P99 | Cross-asset stress building |
| CRISIS | Turbulence ≥ P99 | Systemic stress, regime shift likely |

### Divergence Detection

The **divergence signal** (green shading on chart) activates when:
1. Turbulence is above the 95th percentile threshold, AND
2. SPX is "rising" (configurable definition)

**Divergence Rules**:
- `ma50`: SPX close > 50-day moving average
- `ma20_slope`: 20-day moving average has positive slope

**Why It Matters**: This divergence represents a dangerous market condition where:
- Surface appears calm (markets rising)
- Underlying stress is elevated (cross-asset correlations unstable)
- Risk of sudden correction is elevated

## Project Structure

```
Financial-turbulence-monitoring-system-with-Mahalanobis-distance/
├── src/
│   ├── __init__.py      # Package exports
│   ├── config.py        # Configuration and parameters
│   ├── data.py          # Data fetching and processing
│   ├── turbulence.py    # Mahalanobis distance calculation
│   ├── signals.py       # Divergence detection and regime classification
│   ├── dashboard.py     # Matplotlib visualization
│   └── main.py          # CLI and pipeline orchestration
├── tests/
│   └── test_turbulence.py  # Unit tests
├── output/              # Generated outputs
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## References

- Kritzman, M., & Li, Y. (2010). "Skulls, Financial Turbulence, and Risk Management". Financial Analysts Journal.
- Jordi Visser's Market Immune System concept

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or pull request.
