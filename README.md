# Search Arbitrage Backtester

A modular Python backtesting system for analyzing momentum trading setups using the Polygon.io API.

## Features

### Four Distinct Trading Setups

1. **Opening Range Breakout (ORB)** - 3/5/15/60 minute variants
   - Entry on 1m candle close above opening range high
   - Automatic timeframe optimization based on ADR

2. **Episodic Pivots (EP)**
   - Gap > 4% from previous close
   - Price > 1-month high
   - Volume > 3x 50-day average

3. **Delayed Reactions**
   - Monitors stocks 2-5 days post-EP
   - Entry on breach of "Inside Day" consolidation high

4. **Earnings Plays**
   - Post-earnings momentum (gap and go)
   - Post-earnings pullback
   - Pre-earnings drift

### Context Engine ("The Why")

Every trade logs a comprehensive context snapshot:
- **Relative Volume (RVOL)**: Volume / SMA(50)
- **Tightness Factor**: 5-day close range / ATR(20)
- **MA Surfing**: Distance from 10 EMA and 20 EMA
- **Market Regime**: SPY/QQQ vs their 4 EMA and 50 SMA
- **Float & Sector**: Categorized float size and industry group

### Analytics

- **Attribute Correlation**: Find which contexts correlate with Profit Factor > 2.0
- **Timeframe Optimization**: Identify optimal ORB timeframe by ADR
- **Golden Setups**: Discover high-performing condition combinations
- **Market Regime Impact**: Performance breakdown by market conditions

### Exit Strategies

- Trailing stop based on 10 EMA (StockBee "Surfer" style)
- Fixed R:R of 1:3
- ATR-based stop loss

## Installation

```bash
cd SearchArbitrage
pip install -r requirements.txt
```

## Configuration

1. Set your Polygon.io API key in `.env`:
```
POLYGON_API_KEY=your_api_key_here
```

2. Adjust parameters in `config.py` as needed.

## Usage

### Run Demo
```bash
python main.py --demo
```

### Full Backtest
```bash
python main.py --tickers NVDA AMD TSLA META --start 2024-01-01 --end 2024-06-01
```

### Quick Scan (Today's Signals)
```bash
python main.py --scan
```

### Analyze Existing Results
```bash
python main.py --analyze output/master_trade_log.csv
```

### Select Specific Setups
```bash
python main.py --setups orb15 ep dr --tickers NVDA AMD
```

Available setups: `orb3`, `orb5`, `orb15`, `orb60`, `ep`, `dr`, `earnings`

## Output

- `output/master_trade_log.csv` - Complete trade log with context snapshots
- `output/equity_curve.csv` - Daily equity tracking
- `output/dashboard.html` - Interactive Plotly dashboard
- `output/golden_setups.html` - Best-performing combinations
- `output/analysis_report.txt` - Text summary report

## Interpreting Results

### The "Golden Setup"

Look for combinations like:
- **Setup**: 15-min ORB
- **Circumstance**: Top-3 sector, SPY above 4-day EMA, float < 20M
- **Result**: 70%+ win rate vs 40% baseline

### Key Metrics

- **Profit Factor > 2.0**: Strong edge
- **Win Rate > 55%**: Favorable odds
- **Avg R-Multiple > 1.5**: Good risk/reward execution

## Project Structure

```
SearchArbitrage/
├── main.py              # Entry point
├── config.py            # Configuration
├── polygon_fetcher.py   # Polygon.io API wrapper
├── backtester.py        # Main backtesting engine
├── context_engine.py    # Context snapshot generation
├── analyzer.py          # Statistical analysis
├── dashboard.py         # Plotly visualizations
├── setups/
│   ├── base_setup.py    # Base class
│   ├── orb_setup.py     # Opening Range Breakout
│   ├── ep_setup.py      # Episodic Pivots
│   ├── delayed_reaction.py
│   └── earnings_play.py
├── requirements.txt
└── .env                 # API key (not in repo)
```

## API Rate Limits

The Polygon.io free tier allows 5 API calls per minute. The `PolygonFetcher` class handles rate limiting automatically with caching to minimize API usage.

## License

MIT
