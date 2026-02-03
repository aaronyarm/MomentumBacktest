"""
Episodic Pivot (EP) Setup.
Gap > 4%, Price > 1-month high, Volume > 3x average.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from .base_setup import BaseSetup, Signal, Trade
import config


class EpisodicPivotSetup(BaseSetup):
    """
    Episodic Pivot Strategy.

    Identifies stocks with significant catalyst-driven gaps that
    break to new highs on extraordinary volume.

    Criteria:
    - Gap up > 4% from previous close
    - Price > 1-month (20-day) high
    - Volume > 3x 50-day average
    """

    def __init__(self):
        super().__init__("Episodic_Pivot")
        self.gap_threshold = config.EP_GAP_THRESHOLD
        self.volume_multiplier = config.EP_VOLUME_MULTIPLIER
        self.lookback_days = config.EP_LOOKBACK_DAYS

    def scan_for_signals(
        self,
        ticker: str,
        daily_data: pd.DataFrame,
        minute_data: pd.DataFrame,
        date: datetime
    ) -> List[Signal]:
        """
        Scan for Episodic Pivot signals.

        Entry criteria:
        - Gap > 4% from previous close
        - Current price > 20-day high
        - Volume > 3x 50-day average
        """
        signals = []

        if daily_data.empty or len(daily_data) < self.lookback_days + 1:
            return signals

        # Get the current day's data
        try:
            current_idx = daily_data.index.get_loc(
                daily_data.index[daily_data.index.date == date.date()][0]
            )
        except (IndexError, KeyError):
            return signals

        if current_idx < self.lookback_days:
            return signals

        current_bar = daily_data.iloc[current_idx]
        prev_bar = daily_data.iloc[current_idx - 1]
        lookback_data = daily_data.iloc[current_idx - self.lookback_days:current_idx]

        # Calculate metrics
        gap_pct = (current_bar["open"] - prev_bar["close"]) / prev_bar["close"]
        one_month_high = lookback_data["high"].max()
        avg_volume = daily_data.iloc[max(0, current_idx - 50):current_idx]["volume"].mean()
        current_volume = current_bar["volume"]

        # Check EP criteria
        is_gap = gap_pct >= self.gap_threshold
        is_new_high = current_bar["high"] > one_month_high
        is_volume_surge = current_volume > (avg_volume * self.volume_multiplier)

        if is_gap and is_new_high and is_volume_surge:
            atr = self.calculate_atr(daily_data.iloc[:current_idx + 1]).iloc[-1]
            entry_price = current_bar["close"]

            # EP entry is typically on pullback to VWAP or gap fill support
            # For simplicity, we use gap low as stop
            stop_loss = min(current_bar["low"], current_bar["open"] - atr)
            target = self.calculate_target(entry_price, stop_loss, config.FIXED_RR_RATIO)

            rvol = current_volume / avg_volume if avg_volume > 0 else 0

            signal = Signal(
                ticker=ticker,
                timestamp=date,
                setup_type=self.name,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target,
                signal_strength=self._calculate_signal_strength(gap_pct, rvol, current_bar, lookback_data),
                metadata={
                    "gap_pct": gap_pct * 100,
                    "prev_close": prev_bar["close"],
                    "one_month_high": one_month_high,
                    "rvol": rvol,
                    "avg_volume": avg_volume,
                    "current_volume": current_volume,
                    "gap_size": current_bar["open"] - prev_bar["close"]
                }
            )
            signals.append(signal)

        return signals

    def _calculate_signal_strength(
        self,
        gap_pct: float,
        rvol: float,
        current_bar: pd.Series,
        lookback_data: pd.DataFrame
    ) -> float:
        """
        Calculate signal strength for EP.

        Factors:
        - Larger gaps = stronger (up to a point)
        - Higher RVOL = stronger
        - Clean breakout (close near high) = stronger
        """
        strength = 0.3  # Base strength

        # Gap size contribution (4-10% is sweet spot)
        if 0.04 <= gap_pct <= 0.10:
            strength += 0.2
        elif gap_pct > 0.10:
            strength += 0.15  # Very large gaps can be riskier

        # RVOL contribution
        if rvol >= 5:
            strength += 0.25
        elif rvol >= 3:
            strength += 0.15

        # Close near high of day (strong close)
        day_range = current_bar["high"] - current_bar["low"]
        if day_range > 0:
            close_position = (current_bar["close"] - current_bar["low"]) / day_range
            if close_position > 0.7:
                strength += 0.15

        # Breakout magnitude (how far above prior high)
        prior_high = lookback_data["high"].max()
        if prior_high > 0:
            breakout_pct = (current_bar["high"] - prior_high) / prior_high
            if breakout_pct > 0.05:
                strength += 0.1

        return min(strength, 1.0)

    def check_exit(
        self,
        trade: Trade,
        current_bar: pd.Series,
        daily_data: pd.DataFrame
    ) -> Optional[tuple]:
        """
        Check exit conditions for EP trade.

        Exit conditions:
        1. Stop loss (below entry day low or ATR stop)
        2. Fixed R:R target
        3. 10 EMA trailing stop (Surfer style)
        4. Close below 20 EMA (momentum loss)
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

        # Trailing stop using 10 EMA
        if not daily_data.empty and len(daily_data) >= 10:
            ema_10 = self.calculate_ema(daily_data["close"], 10).iloc[-1]
            ema_20 = self.calculate_ema(daily_data["close"], 20).iloc[-1]

            # StockBee Surfer style: exit on close below 10 EMA
            if current_price < ema_10:
                return (current_price, "trailing_stop_ema10")

            # Momentum loss: exit on close below 20 EMA
            if current_price < ema_20:
                return (current_price, "momentum_loss_ema20")

        return None

    def identify_catalyst_type(self, metadata: Dict) -> str:
        """
        Attempt to classify the catalyst type based on gap characteristics.

        This is a simplified heuristic - real implementation would use news data.
        """
        gap_pct = metadata.get("gap_pct", 0)
        rvol = metadata.get("rvol", 0)

        if gap_pct > 20 and rvol > 10:
            return "Major_Catalyst"  # Could be M&A, FDA approval, etc.
        elif gap_pct > 10 and rvol > 5:
            return "Earnings_Beat"
        elif gap_pct > 4 and rvol > 3:
            return "Momentum_Break"
        else:
            return "Standard_Gap"

    def get_ep_follow_through_score(
        self,
        daily_data: pd.DataFrame,
        ep_date: datetime,
        days_after: int = 3
    ) -> float:
        """
        Calculate follow-through score after EP signal.

        Higher score = stronger continuation.
        """
        try:
            ep_idx = daily_data.index.get_loc(
                daily_data.index[daily_data.index.date == ep_date.date()][0]
            )
        except (IndexError, KeyError):
            return 0.0

        follow_data = daily_data.iloc[ep_idx + 1:ep_idx + 1 + days_after]
        if follow_data.empty:
            return 0.0

        ep_high = daily_data.iloc[ep_idx]["high"]
        ep_close = daily_data.iloc[ep_idx]["close"]

        # Score based on holding above EP close and making new highs
        days_above_ep_close = (follow_data["close"] > ep_close).sum()
        new_high_days = (follow_data["high"] > ep_high).sum()

        score = (days_above_ep_close / days_after * 0.5) + (new_high_days / days_after * 0.5)
        return score
