"""
Delayed Reaction Setup.
Monitor stocks 2-5 days post-EP for "Inside Day" consolidation
and entry on breach of consolidation high.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from .base_setup import BaseSetup, Signal, Trade
from .ep_setup import EpisodicPivotSetup
import config


class DelayedReactionSetup(BaseSetup):
    """
    Delayed Reaction Strategy.

    Identifies stocks that have had an EP (Episodic Pivot) and are now
    consolidating in a tight range, waiting for continuation.

    Criteria:
    - Stock had an EP signal within the last 2-5 days
    - Currently forming an "Inside Day" pattern (tighter range)
    - Entry on break above consolidation high
    """

    def __init__(self):
        super().__init__("Delayed_Reaction")
        self.ep_setup = EpisodicPivotSetup()
        self.monitor_start = config.DR_MONITOR_START
        self.monitor_end = config.DR_MONITOR_END
        self.inside_day_tolerance = config.DR_INSIDE_DAY_TOLERANCE

    def find_recent_ep(
        self,
        ticker: str,
        daily_data: pd.DataFrame,
        current_date: datetime
    ) -> Optional[Tuple[datetime, Dict]]:
        """
        Find if there was an EP signal in the last 2-5 days.

        Returns:
            Tuple of (EP date, EP metadata) or None
        """
        if daily_data.empty or len(daily_data) < config.EP_LOOKBACK_DAYS + self.monitor_end:
            return None

        # Look back through the monitoring window
        for days_ago in range(self.monitor_start, self.monitor_end + 1):
            check_date = current_date - timedelta(days=days_ago)

            # Skip weekends (simplified - real implementation needs market calendar)
            if check_date.weekday() >= 5:
                continue

            # Check if EP signal existed on that date
            signals = self.ep_setup.scan_for_signals(
                ticker, daily_data, pd.DataFrame(), check_date
            )

            if signals:
                return (check_date, signals[0].metadata)

        return None

    def is_inside_day(
        self,
        current_bar: pd.Series,
        reference_bar: pd.Series
    ) -> bool:
        """
        Check if current day is an inside day relative to reference.

        Inside day: High < Reference High AND Low > Reference Low
        """
        tolerance = reference_bar["close"] * self.inside_day_tolerance

        is_high_inside = current_bar["high"] <= reference_bar["high"] + tolerance
        is_low_inside = current_bar["low"] >= reference_bar["low"] - tolerance

        return is_high_inside and is_low_inside

    def find_consolidation_range(
        self,
        daily_data: pd.DataFrame,
        ep_date: datetime,
        current_date: datetime
    ) -> Optional[Dict[str, float]]:
        """
        Find the consolidation range after an EP.

        Returns:
            Dictionary with consolidation high, low, and tightness
        """
        try:
            ep_idx = daily_data.index.get_loc(
                daily_data.index[daily_data.index.date == ep_date.date()][0]
            )
            current_idx = daily_data.index.get_loc(
                daily_data.index[daily_data.index.date == current_date.date()][0]
            )
        except (IndexError, KeyError):
            return None

        if current_idx <= ep_idx:
            return None

        # Get data from day after EP to current
        consol_data = daily_data.iloc[ep_idx + 1:current_idx + 1]

        if consol_data.empty:
            return None

        ep_bar = daily_data.iloc[ep_idx]
        consol_high = consol_data["high"].max()
        consol_low = consol_data["low"].min()
        consol_range = consol_high - consol_low

        # Calculate tightness relative to EP day range
        ep_range = ep_bar["high"] - ep_bar["low"]
        tightness = consol_range / ep_range if ep_range > 0 else 1.0

        # Count inside days
        inside_days = 0
        for i in range(1, len(consol_data)):
            if self.is_inside_day(consol_data.iloc[i], consol_data.iloc[i - 1]):
                inside_days += 1

        return {
            "high": consol_high,
            "low": consol_low,
            "range": consol_range,
            "tightness": tightness,
            "inside_days": inside_days,
            "days_consolidated": len(consol_data),
            "ep_high": ep_bar["high"],
            "ep_close": ep_bar["close"]
        }

    def scan_for_signals(
        self,
        ticker: str,
        daily_data: pd.DataFrame,
        minute_data: pd.DataFrame,
        date: datetime
    ) -> List[Signal]:
        """
        Scan for Delayed Reaction signals.

        Entry criteria:
        - Recent EP (2-5 days ago)
        - Tight consolidation (inside days)
        - Breakout above consolidation high
        """
        signals = []

        # Find recent EP
        ep_result = self.find_recent_ep(ticker, daily_data, date)
        if ep_result is None:
            return signals

        ep_date, ep_metadata = ep_result

        # Find consolidation range
        consolidation = self.find_consolidation_range(daily_data, ep_date, date)
        if consolidation is None:
            return signals

        # Check for tightness and inside days
        if consolidation["tightness"] > 0.75:  # Not tight enough
            return signals

        if consolidation["inside_days"] < 1:  # Need at least one inside day
            return signals

        # Get current bar
        try:
            current_idx = daily_data.index.get_loc(
                daily_data.index[daily_data.index.date == date.date()][0]
            )
        except (IndexError, KeyError):
            return signals

        current_bar = daily_data.iloc[current_idx]

        # Check for breakout above consolidation high
        if current_bar["high"] > consolidation["high"]:
            atr = self.calculate_atr(daily_data.iloc[:current_idx + 1]).iloc[-1]

            entry_price = consolidation["high"] + 0.01  # Entry just above breakout
            stop_loss = consolidation["low"] - (atr * 0.5)
            target = self.calculate_target(entry_price, stop_loss, config.FIXED_RR_RATIO)

            signal = Signal(
                ticker=ticker,
                timestamp=date,
                setup_type=self.name,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target,
                signal_strength=self._calculate_signal_strength(consolidation, ep_metadata),
                metadata={
                    "ep_date": ep_date,
                    "ep_gap_pct": ep_metadata.get("gap_pct", 0),
                    "ep_rvol": ep_metadata.get("rvol", 0),
                    "consolidation_high": consolidation["high"],
                    "consolidation_low": consolidation["low"],
                    "consolidation_tightness": consolidation["tightness"],
                    "inside_days": consolidation["inside_days"],
                    "days_since_ep": consolidation["days_consolidated"]
                }
            )
            signals.append(signal)

        return signals

    def _calculate_signal_strength(
        self,
        consolidation: Dict[str, float],
        ep_metadata: Dict
    ) -> float:
        """
        Calculate signal strength for Delayed Reaction.

        Factors:
        - Tighter consolidation = stronger
        - More inside days = stronger
        - Stronger original EP = stronger
        - Holding above EP close = stronger
        """
        strength = 0.3

        # Tightness contribution
        if consolidation["tightness"] < 0.3:
            strength += 0.25
        elif consolidation["tightness"] < 0.5:
            strength += 0.15

        # Inside days contribution
        inside_days = consolidation["inside_days"]
        if inside_days >= 3:
            strength += 0.2
        elif inside_days >= 2:
            strength += 0.15
        elif inside_days >= 1:
            strength += 0.1

        # Original EP strength
        ep_rvol = ep_metadata.get("rvol", 0)
        if ep_rvol > 5:
            strength += 0.15
        elif ep_rvol > 3:
            strength += 0.1

        # Holding above EP close
        if consolidation["low"] > consolidation["ep_close"]:
            strength += 0.1

        return min(strength, 1.0)

    def check_exit(
        self,
        trade: Trade,
        current_bar: pd.Series,
        daily_data: pd.DataFrame
    ) -> Optional[tuple]:
        """
        Check exit conditions for Delayed Reaction trade.

        Exit conditions:
        1. Stop loss (below consolidation low)
        2. Fixed R:R target
        3. 10 EMA trailing stop
        4. Close back inside consolidation range (failed breakout)
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
            if current_price < ema_10:
                return (current_price, "trailing_stop_ema10")

        # Failed breakout - close back inside consolidation
        consol_high = trade.metadata.get("consolidation_high", 0)
        if consol_high > 0 and current_price < consol_high:
            return (current_price, "failed_breakout")

        return None
