"""
Main Backtester Engine.
Orchestrates signal generation, trade execution, and exit management.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass
import uuid
import os

from polygon_fetcher import PolygonFetcher, DataManager
from context_engine import ContextEngine
from setups.base_setup import BaseSetup, Signal, Trade
from setups import ORBSetup, EpisodicPivotSetup, DelayedReactionSetup, EarningsPlaySetup
import config


@dataclass
class Position:
    """Represents an open position."""
    trade: Trade
    shares: int
    entry_value: float


class Portfolio:
    """Manages portfolio state and position sizing."""

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.equity_history: List[Dict] = []
        self.trade_count = 0

    @property
    def equity(self) -> float:
        """Current total equity."""
        position_value = sum(
            p.shares * p.trade.exit_price if p.trade.exit_price else p.entry_value
            for p in self.positions.values()
        )
        return self.cash + position_value

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return len(self.positions) < config.MAX_POSITIONS

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """
        Calculate position size based on risk management.

        Uses fixed percentage of equity with stop-loss based adjustment.
        """
        # Maximum capital to risk on this trade
        max_position_value = self.equity * config.POSITION_SIZE_PCT

        # Risk per share
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return 0

        # Position size based on 1% equity risk
        equity_risk = self.equity * 0.01
        shares_by_risk = int(equity_risk / risk_per_share)

        # Position size based on max allocation
        shares_by_allocation = int(max_position_value / entry_price)

        # Take the smaller of the two
        return min(shares_by_risk, shares_by_allocation)

    def open_position(self, trade: Trade, shares: int) -> bool:
        """Open a new position."""
        if trade.ticker in self.positions:
            return False

        cost = shares * trade.entry_price
        if cost > self.cash:
            return False

        self.cash -= cost
        self.positions[trade.ticker] = Position(
            trade=trade,
            shares=shares,
            entry_value=cost
        )
        self.trade_count += 1
        return True

    def close_position(self, ticker: str, exit_price: float) -> Optional[Trade]:
        """Close a position and return the completed trade."""
        if ticker not in self.positions:
            return None

        position = self.positions.pop(ticker)
        trade = position.trade

        # Calculate PnL
        exit_value = position.shares * exit_price
        pnl = exit_value - position.entry_value
        pnl_pct = (pnl / position.entry_value) * 100

        # Calculate R-multiple
        risk = trade.entry_price - trade.stop_loss
        if risk > 0:
            r_multiple = (exit_price - trade.entry_price) / risk
        else:
            r_multiple = 0

        # Update trade
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.r_multiple = r_multiple
        trade.position_size = position.shares

        # Update cash
        self.cash += exit_value

        return trade

    def record_equity(self, date: datetime):
        """Record equity for a given date."""
        self.equity_history.append({
            "date": date,
            "equity": self.equity,
            "cash": self.cash,
            "positions": len(self.positions)
        })


class Backtester:
    """
    Main backtesting engine.

    Coordinates:
    - Data loading via DataManager
    - Signal generation via Setup classes
    - Trade execution and position management
    - Context capture via ContextEngine
    - Performance tracking
    """

    def __init__(
        self,
        setups: List[BaseSetup] = None,
        initial_capital: float = config.INITIAL_CAPITAL
    ):
        self.fetcher = PolygonFetcher()
        self.data_manager = DataManager(self.fetcher)
        self.context_engine = ContextEngine(self.data_manager)
        self.portfolio = Portfolio(initial_capital)

        # Initialize default setups if none provided
        if setups is None:
            self.setups = [
                ORBSetup(timeframe=3),
                ORBSetup(timeframe=5),
                ORBSetup(timeframe=15),
                ORBSetup(timeframe=60),
                EpisodicPivotSetup(),
                DelayedReactionSetup(),
                EarningsPlaySetup(play_type="post_momentum"),
            ]
        else:
            self.setups = setups

        self.completed_trades: List[Trade] = []
        self.all_signals: List[Signal] = []

    def run(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        include_minute_data: bool = True
    ) -> pd.DataFrame:
        """
        Run the backtest.

        Args:
            tickers: List of stock symbols to test
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            include_minute_data: Whether to load minute data (needed for ORB)

        Returns:
            DataFrame of completed trades
        """
        print(f"Starting backtest from {start_date} to {end_date}")
        print(f"Tickers: {len(tickers)}, Setups: {len(self.setups)}")

        # Preload data
        self.data_manager.preload_tickers(
            tickers, start_date, end_date,
            include_minute=include_minute_data
        )

        # Preload market data for context
        self.context_engine.preload_market_data(start_date, end_date)

        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

        # Main backtest loop
        from tqdm import tqdm
        for date in tqdm(dates, desc="Backtesting"):
            self._process_date(date, tickers)

        # Close any remaining positions at end
        self._close_all_positions(dates[-1])

        # Generate results DataFrame
        results_df = self._generate_results_df()

        # Save results
        self._save_results(results_df)

        return results_df

    def _process_date(self, date: datetime, tickers: List[str]):
        """Process a single trading day."""
        # First, check exits for open positions
        self._check_exits(date)

        # Then, scan for new signals
        for ticker in tickers:
            if ticker in self.portfolio.positions:
                continue  # Already have a position

            if not self.portfolio.can_open_position():
                break  # Max positions reached

            daily_data = self.data_manager.get_daily(ticker)
            minute_data = self.data_manager.get_minute(ticker)

            if daily_data.empty:
                continue

            # Scan each setup for signals
            for setup in self.setups:
                signals = setup.scan_for_signals(ticker, daily_data, minute_data, date)

                for signal in signals:
                    self.all_signals.append(signal)

                    # Execute signal if we can
                    if self.portfolio.can_open_position():
                        self._execute_signal(signal, daily_data, date)

        # Record daily equity
        self.portfolio.record_equity(date)

    def _execute_signal(self, signal: Signal, daily_data: pd.DataFrame, date: datetime):
        """Execute a trading signal."""
        # Calculate position size
        shares = self.portfolio.calculate_position_size(
            signal.entry_price, signal.stop_loss
        )

        if shares <= 0:
            return

        # Get ticker info
        ticker_info = self.data_manager.get_info(signal.ticker)

        # Generate context snapshot
        context = self.context_engine.generate_context_snapshot(
            signal.ticker, date, daily_data, ticker_info
        )

        # Create trade
        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            ticker=signal.ticker,
            setup_type=signal.setup_type,
            entry_time=signal.timestamp,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            position_size=shares,
            context_snapshot=context,
            metadata=signal.metadata
        )

        # Open position
        if self.portfolio.open_position(trade, shares):
            # Add trade to appropriate setup's trade list
            for setup in self.setups:
                if setup.name == signal.setup_type:
                    setup.trades.append(trade)
                    break

    def _check_exits(self, date: datetime):
        """Check exit conditions for all open positions."""
        positions_to_close = []

        for ticker, position in self.portfolio.positions.items():
            daily_data = self.data_manager.get_daily(ticker)
            if daily_data.empty:
                continue

            # Get current bar
            try:
                current_idx = daily_data.index.get_loc(
                    daily_data.index[daily_data.index.date == date.date()][0]
                )
                current_bar = daily_data.iloc[current_idx]
            except (IndexError, KeyError):
                continue

            # Find the setup and check exit
            for setup in self.setups:
                if setup.name == position.trade.setup_type:
                    exit_result = setup.check_exit(
                        position.trade, current_bar, daily_data.iloc[:current_idx + 1]
                    )
                    if exit_result:
                        exit_price, exit_reason = exit_result
                        positions_to_close.append((ticker, exit_price, exit_reason, date))
                    break

        # Close positions
        for ticker, exit_price, exit_reason, date in positions_to_close:
            trade = self.portfolio.close_position(ticker, exit_price)
            if trade:
                trade.exit_time = date
                trade.exit_reason = exit_reason
                self.completed_trades.append(trade)

    def _close_all_positions(self, date: datetime):
        """Close all remaining positions at market close."""
        for ticker in list(self.portfolio.positions.keys()):
            daily_data = self.data_manager.get_daily(ticker)
            if daily_data.empty:
                continue

            try:
                current_bar = daily_data.iloc[-1]
                exit_price = current_bar["close"]
            except (IndexError, KeyError):
                continue

            trade = self.portfolio.close_position(ticker, exit_price)
            if trade:
                trade.exit_time = date
                trade.exit_reason = "backtest_end"
                self.completed_trades.append(trade)

    def _generate_results_df(self) -> pd.DataFrame:
        """Generate results DataFrame from completed trades."""
        if not self.completed_trades:
            return pd.DataFrame()

        return pd.DataFrame([trade.to_dict() for trade in self.completed_trades])

    def _save_results(self, results_df: pd.DataFrame):
        """Save results to CSV and generate summary."""
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        # Save trade log
        output_path = os.path.join(config.OUTPUT_DIR, config.TRADE_LOG_FILE)
        results_df.to_csv(output_path, index=False)
        print(f"Trade log saved to {output_path}")

        # Print summary
        self._print_summary(results_df)

    def _print_summary(self, results_df: pd.DataFrame):
        """Print backtest summary."""
        if results_df.empty:
            print("No trades executed.")
            return

        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)

        print(f"\nTotal Trades: {len(results_df)}")
        print(f"Winning Trades: {(results_df['pnl'] > 0).sum()}")
        print(f"Losing Trades: {(results_df['pnl'] <= 0).sum()}")
        print(f"Win Rate: {(results_df['pnl'] > 0).mean():.1%}")

        total_pnl = results_df['pnl'].sum()
        print(f"\nTotal P&L: ${total_pnl:,.2f}")
        print(f"Return: {(total_pnl / self.portfolio.initial_capital) * 100:.2f}%")

        wins = results_df[results_df['pnl'] > 0]['pnl'].sum()
        losses = abs(results_df[results_df['pnl'] <= 0]['pnl'].sum())
        pf = wins / losses if losses > 0 else float('inf')
        print(f"Profit Factor: {pf:.2f}")

        print(f"\nAvg Win: ${results_df[results_df['pnl'] > 0]['pnl'].mean():,.2f}")
        print(f"Avg Loss: ${results_df[results_df['pnl'] <= 0]['pnl'].mean():,.2f}")
        print(f"Avg R-Multiple: {results_df['r_multiple'].mean():.2f}")

        # Per-setup breakdown
        print("\n" + "-" * 40)
        print("BREAKDOWN BY SETUP")
        print("-" * 40)

        for setup_type in results_df['setup_type'].unique():
            subset = results_df[results_df['setup_type'] == setup_type]
            setup_wins = subset[subset['pnl'] > 0]['pnl'].sum()
            setup_losses = abs(subset[subset['pnl'] <= 0]['pnl'].sum())
            setup_pf = setup_wins / setup_losses if setup_losses > 0 else float('inf')

            print(f"\n{setup_type}:")
            print(f"  Trades: {len(subset)}")
            print(f"  Win Rate: {(subset['pnl'] > 0).mean():.1%}")
            print(f"  Profit Factor: {setup_pf:.2f}")
            print(f"  Total P&L: ${subset['pnl'].sum():,.2f}")

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame(self.portfolio.equity_history)

    def get_setup_stats(self) -> Dict[str, Dict]:
        """Get performance stats for each setup."""
        return {setup.name: setup.get_trade_stats() for setup in self.setups}
