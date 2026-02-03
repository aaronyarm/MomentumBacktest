"""
Opening Range Breakout (ORB) Setup.
Entry on 1m candle close above the High of the specified range (3, 5, 15, or 60m).
"""
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import List, Optional, Dict, Any
from .base_setup import BaseSetup, Signal, Trade
import config


class ORBSetup(BaseSetup):
    """
    Opening Range Breakout Strategy.

    Identifies breakouts above the high of the opening range
    for various timeframes (3, 5, 15, 60 minutes).
    """

    def __init__(self, timeframe: int = 15):
        """
        Initialize ORB setup.

        Args:
            timeframe: Opening range duration in minutes (3, 5, 15, or 60)
        """
        super().__init__(f"ORB_{timeframe}m")
        self.timeframe = timeframe
        self.market_open = time(9, 30)  # US market open

    def get_opening_range(
        self,
        minute_data: pd.DataFrame,
        date: datetime
    ) -> Optional[Dict[str, float]]:
        """
        Calculate the opening range high/low for a given date.

        Args:
            minute_data: Minute-level OHLCV data
            date: The trading date

        Returns:
            Dictionary with 'high', 'low', 'range' or None if insufficient data
        """
        # Filter to the specific date (handle both DatetimeIndex and RangeIndex)
        if minute_data.empty:
            return None

        if hasattr(minute_data.index, 'date'):
            day_data = minute_data[minute_data.index.date == date.date()]
        else:
            # No minute data with proper datetime index
            return None

        if day_data.empty:
            return None

        # Get market open time
        market_open_dt = datetime.combine(date.date(), self.market_open)
        range_end_dt = market_open_dt + timedelta(minutes=self.timeframe)

        # Filter to opening range period
        or_data = day_data[
            (day_data.index >= market_open_dt) &
            (day_data.index < range_end_dt)
        ]

        if or_data.empty or len(or_data) < self.timeframe:
            return None

        return {
            "high": or_data["high"].max(),
            "low": or_data["low"].min(),
            "range": or_data["high"].max() - or_data["low"].min(),
            "open": or_data.iloc[0]["open"],
            "close": or_data.iloc[-1]["close"],
            "volume": or_data["volume"].sum(),
            "range_end": range_end_dt
        }

    def scan_for_signals(
        self,
        ticker: str,
        daily_data: pd.DataFrame,
        minute_data: pd.DataFrame,
        date: datetime
    ) -> List[Signal]:
        """
        Scan for ORB breakout signals.

        Entry criteria:
        - 1-minute candle closes above the opening range high
        - Must occur after the opening range period ends
        """
        signals = []

        opening_range = self.get_opening_range(minute_data, date)
        if opening_range is None:
            return signals

        or_high = opening_range["high"]
        or_low = opening_range["low"]
        range_end = opening_range["range_end"]

        # Get minute data after the opening range
        if hasattr(minute_data.index, 'date'):
            day_data = minute_data[minute_data.index.date == date.date()]
        else:
            return signals  # No valid minute data
        post_or_data = day_data[day_data.index >= range_end]

        if post_or_data.empty:
            return signals

        # Look for the first candle that closes above the OR high
        for idx, row in post_or_data.iterrows():
            if row["close"] > or_high + config.ORB_ENTRY_BUFFER:
                # Calculate ATR for stop loss
                if not daily_data.empty:
                    atr = self.calculate_atr(daily_data).iloc[-1]
                else:
                    atr = opening_range["range"]

                entry_price = row["close"]
                stop_loss = max(or_low, entry_price - (1.5 * atr))
                target = self.calculate_target(entry_price, stop_loss, config.FIXED_RR_RATIO)

                signal = Signal(
                    ticker=ticker,
                    timestamp=idx,
                    setup_type=self.name,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_price=target,
                    signal_strength=self._calculate_signal_strength(opening_range, daily_data),
                    metadata={
                        "or_high": or_high,
                        "or_low": or_low,
                        "or_range": opening_range["range"],
                        "or_volume": opening_range["volume"],
                        "timeframe": self.timeframe,
                        "breakout_bar_volume": row["volume"]
                    }
                )
                signals.append(signal)
                break  # Only take the first breakout

        return signals

    def _calculate_signal_strength(
        self,
        opening_range: Dict[str, float],
        daily_data: pd.DataFrame
    ) -> float:
        """
        Calculate signal strength based on range tightness and volume.

        Returns a score from 0 to 1.
        """
        strength = 0.5  # Base strength

        if daily_data.empty:
            return strength

        # Tight range relative to ADR is stronger
        adr = self.calculate_adr(daily_data).iloc[-1] if len(daily_data) > 20 else 2.0
        last_close = daily_data.iloc[-1]["close"]
        or_range_pct = (opening_range["range"] / last_close) * 100

        if or_range_pct < adr * 0.5:
            strength += 0.2  # Tight range bonus

        # Higher relative volume is stronger
        avg_volume = daily_data["volume"].rolling(20).mean().iloc[-1]
        if avg_volume > 0:
            rvol = opening_range["volume"] / (avg_volume * (self.timeframe / 390))
            if rvol > 2:
                strength += 0.2
            if rvol > 3:
                strength += 0.1

        return min(strength, 1.0)

    def check_exit(
        self,
        trade: Trade,
        current_bar: pd.Series,
        daily_data: pd.DataFrame
    ) -> Optional[tuple]:
        """
        Check exit conditions for ORB trade.

        Exit conditions:
        1. Stop loss hit (below OR low or ATR stop)
        2. Target hit (fixed R:R)
        3. Trailing stop (10 EMA on higher timeframe)
        4. End of day (close position)
        """
        current_price = current_bar["close"]
        current_low = current_bar["low"]
        current_high = current_bar["high"]

        # Check stop loss
        if current_low <= trade.stop_loss:
            return (trade.stop_loss, "stop_loss")

        # Check target
        if current_high >= trade.target_price:
            return (trade.target_price, "target_hit")

        # Check trailing stop (10 EMA)
        if not daily_data.empty and len(daily_data) >= 10:
            ema_10 = self.calculate_ema(daily_data["close"], 10).iloc[-1]
            if current_low < ema_10:
                return (current_price, "trailing_stop_ema")

        # End of day exit (check if near market close)
        if hasattr(current_bar.name, 'time'):
            bar_time = current_bar.name.time()
            if bar_time >= time(15, 55):  # Exit 5 min before close
                return (current_price, "end_of_day")

        return None

    def get_optimal_timeframe_for_adr(
        self,
        adr: float
    ) -> int:
        """
        Suggest optimal ORB timeframe based on ADR.

        Higher ADR stocks may need wider (longer) opening ranges.
        """
        if adr < 3:
            return 5  # Low volatility - use tight range
        elif adr < 5:
            return 15  # Medium volatility
        elif adr < 8:
            return 30  # Higher volatility
        else:
            return 60  # High volatility - need wider range
