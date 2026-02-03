"""
Search Arbitrage Backtester - Main Entry Point.

A modular Python backtester using Polygon.io API to analyze momentum setups:
- 3/5/15/60m Opening Range Breakout (ORB)
- Episodic Pivots (EP)
- Delayed Reactions
- Earnings Plays

Usage:
    python main.py                    # Run with default settings
    python main.py --demo             # Run demo mode with sample tickers
    python main.py --analyze trades.csv  # Analyze existing trade log
"""
import argparse
import os
from datetime import datetime, timedelta

import pandas as pd

from backtester import Backtester
from analyzer import TradeAnalyzer
from dashboard import Dashboard, generate_dashboard
from polygon_fetcher import PolygonFetcher
from setups import ORBSetup, EpisodicPivotSetup, DelayedReactionSetup, EarningsPlaySetup
import config


# Sample watchlist for demo/testing
DEMO_TICKERS = [
    "NVDA", "AMD", "TSLA", "META", "AAPL", "MSFT", "GOOGL", "AMZN",
    "NFLX", "CRM", "SNOW", "PLTR", "COIN", "SQ", "SHOP", "ROKU"
]

MOMENTUM_TICKERS = [
    # Semiconductors
    "NVDA", "AMD", "AVGO", "MU", "MRVL", "QCOM", "INTC", "ASML",
    # Software/Tech
    "MSFT", "CRM", "NOW", "ADBE", "SNOW", "PLTR", "NET", "DDOG",
    # Consumer Tech
    "AAPL", "AMZN", "TSLA", "META", "GOOGL", "NFLX",
    # Fintech
    "SQ", "COIN", "PYPL", "HOOD", "SOFI",
    # Healthcare/Biotech
    "MRNA", "BNTX", "LLY", "NVO"
]


def run_backtest(
    tickers: list = None,
    start_date: str = None,
    end_date: str = None,
    setups: list = None,
    initial_capital: float = config.INITIAL_CAPITAL
) -> pd.DataFrame:
    """
    Run the full backtest.

    Args:
        tickers: List of stock symbols to test
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        setups: List of setup instances (uses defaults if None)
        initial_capital: Starting capital

    Returns:
        DataFrame of completed trades
    """
    # Set defaults
    tickers = tickers or DEMO_TICKERS
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = start_date or (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    # Initialize setups if not provided
    if setups is None:
        setups = [
            ORBSetup(timeframe=3),
            ORBSetup(timeframe=5),
            ORBSetup(timeframe=15),
            ORBSetup(timeframe=60),
            EpisodicPivotSetup(),
            DelayedReactionSetup(),
            EarningsPlaySetup(play_type="post_momentum"),
        ]

    # Create and run backtester
    backtester = Backtester(setups=setups, initial_capital=initial_capital)

    print("\n" + "=" * 60)
    print("SEARCH ARBITRAGE BACKTESTER")
    print("=" * 60)
    print(f"Period: {start_date} to {end_date}")
    print(f"Tickers: {len(tickers)}")
    print(f"Setups: {[s.name for s in setups]}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("=" * 60 + "\n")

    # Run backtest
    trades_df = backtester.run(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        include_minute_data=True
    )

    # Save equity curve
    equity_df = backtester.get_equity_curve()
    if not equity_df.empty:
        equity_path = os.path.join(config.OUTPUT_DIR, "equity_curve.csv")
        equity_df.to_csv(equity_path, index=False)
        print(f"Equity curve saved to {equity_path}")

    return trades_df


def analyze_results(trades_csv: str) -> None:
    """
    Analyze existing trade log and generate reports.

    Args:
        trades_csv: Path to trade log CSV file
    """
    print(f"\nAnalyzing trades from: {trades_csv}")

    trades_df = pd.read_csv(trades_csv)
    analyzer = TradeAnalyzer(trades_df)

    # Generate report
    report_path = os.path.join(config.OUTPUT_DIR, "analysis_report.txt")
    report = analyzer.generate_report(report_path)
    print(report)

    # Generate dashboard
    dashboard_path = os.path.join(config.OUTPUT_DIR, config.DASHBOARD_FILE)
    equity_csv = os.path.join(config.OUTPUT_DIR, "equity_curve.csv")

    equity_df = None
    if os.path.exists(equity_csv):
        equity_df = pd.read_csv(equity_csv)

    dashboard = Dashboard(trades_df, equity_df)
    dashboard.create_full_dashboard(dashboard_path)

    # Generate golden setups chart
    golden_setups = analyzer.find_golden_setups()
    if golden_setups:
        gs_fig = dashboard.create_golden_setup_chart(golden_setups)
        gs_path = os.path.join(config.OUTPUT_DIR, "golden_setups.html")
        gs_fig.write_html(gs_path)
        print(f"Golden setups chart saved to {gs_path}")


def quick_scan(tickers: list = None) -> pd.DataFrame:
    """
    Quick scan for today's signals (no backtest, just screening).

    Args:
        tickers: List of tickers to scan

    Returns:
        DataFrame of current signals
    """
    from datetime import date

    tickers = tickers or MOMENTUM_TICKERS
    fetcher = PolygonFetcher()

    today = date.today().strftime("%Y-%m-%d")
    lookback = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")

    setups = [
        EpisodicPivotSetup(),
        DelayedReactionSetup(),
    ]

    all_signals = []

    print(f"\nScanning {len(tickers)} tickers for today's signals...")

    for ticker in tickers:
        daily_data = fetcher.get_daily_bars(ticker, lookback, today)
        if daily_data.empty:
            continue

        for setup in setups:
            signals = setup.scan_for_signals(
                ticker, daily_data, pd.DataFrame(), datetime.now()
            )
            for signal in signals:
                all_signals.append({
                    'ticker': signal.ticker,
                    'setup': signal.setup_type,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'target': signal.target_price,
                    'strength': signal.signal_strength
                })

    signals_df = pd.DataFrame(all_signals)

    if not signals_df.empty:
        signals_df = signals_df.sort_values('strength', ascending=False)
        print("\nToday's Signals:")
        print(signals_df.to_string(index=False))
    else:
        print("No signals found today.")

    return signals_df


def demo_mode():
    """Run a quick demo with sample data."""
    print("\n" + "=" * 60)
    print("DEMO MODE - Running with sample tickers and short period")
    print("=" * 60)

    # Use a shorter period for demo
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Fewer tickers for faster demo
    demo_tickers = ["NVDA", "AMD", "TSLA", "META", "AAPL"]

    trades_df = run_backtest(
        tickers=demo_tickers,
        start_date=start_date,
        end_date=end_date
    )

    if not trades_df.empty:
        # Run analysis
        trades_path = os.path.join(config.OUTPUT_DIR, config.TRADE_LOG_FILE)
        analyze_results(trades_path)
    else:
        print("No trades were executed in demo period.")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Search Arbitrage Backtester - Momentum Strategy Analysis Tool"
    )

    parser.add_argument(
        '--demo', action='store_true',
        help='Run demo mode with sample tickers'
    )
    parser.add_argument(
        '--scan', action='store_true',
        help='Quick scan for today\'s signals'
    )
    parser.add_argument(
        '--analyze', type=str, metavar='CSV',
        help='Analyze existing trade log CSV'
    )
    parser.add_argument(
        '--tickers', type=str, nargs='+',
        help='List of tickers to backtest'
    )
    parser.add_argument(
        '--start', type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--capital', type=float, default=config.INITIAL_CAPITAL,
        help='Initial capital'
    )
    parser.add_argument(
        '--setups', type=str, nargs='+',
        choices=['orb3', 'orb5', 'orb15', 'orb60', 'ep', 'dr', 'earnings'],
        help='Setups to test'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Demo mode
    if args.demo:
        demo_mode()
        return

    # Quick scan mode
    if args.scan:
        quick_scan(args.tickers)
        return

    # Analyze existing results
    if args.analyze:
        analyze_results(args.analyze)
        return

    # Parse setups if specified
    setups = None
    if args.setups:
        setup_map = {
            'orb3': ORBSetup(timeframe=3),
            'orb5': ORBSetup(timeframe=5),
            'orb15': ORBSetup(timeframe=15),
            'orb60': ORBSetup(timeframe=60),
            'ep': EpisodicPivotSetup(),
            'dr': DelayedReactionSetup(),
            'earnings': EarningsPlaySetup(play_type="post_momentum"),
        }
        setups = [setup_map[s] for s in args.setups]

    # Run full backtest
    trades_df = run_backtest(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        setups=setups,
        initial_capital=args.capital
    )

    if not trades_df.empty:
        # Auto-analyze results
        trades_path = os.path.join(config.OUTPUT_DIR, config.TRADE_LOG_FILE)
        analyze_results(trades_path)


if __name__ == "__main__":
    main()
