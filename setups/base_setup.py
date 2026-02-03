"""
Base Setup Class for all trading strategies.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np


@dataclass
class Signal:
    """Represents a trading signal/entry opportunity."""
    ticker: str
    timestamp: datetime
    setup_type: str
    entry_price: float
    stop_loss: float
    target_price: float
    signal_strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """Represents a completed trade with full context."""
    trade_id: str
    ticker: str
    setup_type: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    stop_loss: float = 0.0
    target_price: float = 0.0
    position_size: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    exit_reason: str = ""
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert trade to dictionary for DataFrame export."""
        base = {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "setup_type": self.setup_type,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "position_size": self.position_size,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "r_multiple": self.r_multiple,
            "exit_reason": self.exit_reason,
        }
        # Flatten context snapshot
        for key, value in self.context_snapshot.items():
            base[f"ctx_{key}"] = value
        # Flatten metadata
        for key, value in self.metadata.items():
            base[f"meta_{key}"] = value
        return base


class BaseSetup(ABC):
    """
    Abstract base class for all trading setups.
    Each setup implements its own signal generation and exit logic.
    """

    def __init__(self, name: str):
        self.name = name
        self.signals: List[Signal] = []
        self.trades: List[Trade] = []

    @abstractmethod
    def scan_for_signals(
        self,
        ticker: str,
        daily_data: pd.DataFrame,
        minute_data: pd.DataFrame,
        date: datetime
    ) -> List[Signal]:
        """
        Scan for entry signals on a given date.

        Args:
            ticker: Stock symbol
            daily_data: Daily OHLCV DataFrame
            minute_data: Minute OHLCV DataFrame
            date: Date to scan

        Returns:
            List of Signal objects
        """
        pass

    @abstractmethod
    def check_exit(
        self,
        trade: Trade,
        current_bar: pd.Series,
        daily_data: pd.DataFrame
    ) -> Optional[tuple]:
        """
        Check if exit conditions are met.

        Args:
            trade: Active trade to check
            current_bar: Current price bar
            daily_data: Full daily data for indicators

        Returns:
            (exit_price, exit_reason) tuple if exit triggered, None otherwise
        """
        pass

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        multiplier: float = 1.5
    ) -> float:
        """Calculate ATR-based stop loss."""
        return entry_price - (atr * multiplier)

    def calculate_target(
        self,
        entry_price: float,
        stop_loss: float,
        rr_ratio: float = 3.0
    ) -> float:
        """Calculate target based on R:R ratio."""
        risk = entry_price - stop_loss
        return entry_price + (risk * rr_ratio)

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_adr(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Average Daily Range (as percentage)."""
        daily_range = (df["high"] - df["low"]) / df["close"] * 100
        return daily_range.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period).mean()

    @staticmethod
    def calculate_rvol(volume: pd.Series, period: int = 50) -> pd.Series:
        """Calculate Relative Volume."""
        avg_volume = volume.rolling(window=period).mean()
        return volume / avg_volume

    def get_trade_stats(self) -> Dict[str, Any]:
        """Calculate performance statistics for this setup."""
        if not self.trades:
            return {"total_trades": 0}

        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]

        total_wins = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing))

        return {
            "setup_name": self.name,
            "total_trades": len(self.trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(self.trades) if self.trades else 0,
            "total_pnl": sum(t.pnl for t in self.trades),
            "avg_win": total_wins / len(winning) if winning else 0,
            "avg_loss": total_losses / len(losing) if losing else 0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
            "avg_r_multiple": np.mean([t.r_multiple for t in self.trades]),
            "max_win": max([t.pnl for t in winning]) if winning else 0,
            "max_loss": min([t.pnl for t in losing]) if losing else 0,
        }
