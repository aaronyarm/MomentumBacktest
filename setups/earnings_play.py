"""
Earnings Play Setup.
Pre/post earnings momentum strategies based on historical patterns.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from .base_setup import BaseSetup, Signal, Trade
import config


class EarningsPlaySetup(BaseSetup):
    """
    Earnings Play Strategy.

    Identifies momentum opportunities around earnings announcements.
    Can be configured for:
    - Pre-earnings drift (momentum into earnings)
    - Post-earnings momentum (gap and go)
    - Post-earnings pullback (buy the dip on beat)
    """

    def __init__(self, play_type: str = "post_momentum"):
        """
        Initialize Earnings Play setup.

        Args:
            play_type: 'pre_drift', 'post_momentum', or 'post_pullback'
        """
        super().__init__(f"Earnings_{play_type}")
        self.play_type = play_type

        # Earnings-specific thresholds
        self.min_gap = 0.03  # 3% minimum gap for post-earnings
        self.min_rvol = 2.0  # Minimum RVOL
        self.pre_drift_days = 5  # Days before earnings to enter

    def detect_earnings_gap(
        self,
        daily_data: pd.DataFrame,
        date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if there's an earnings-related gap on the given date.

        Note: This is a heuristic approach. Real implementation would
        integrate with an earnings calendar API.
        """
        if daily_data.empty or len(daily_data) < 2:
            return None

        try:
            if hasattr(daily_data.index, 'date'):
                current_idx = daily_data.index.get_loc(
                    daily_data.index[daily_data.index.date == date.date()][0]
                )
            else:
                return None
        except (IndexError, KeyError, AttributeError):
            return None

        if current_idx < 1:
            return None

        current_bar = daily_data.iloc[current_idx]
        prev_bar = daily_data.iloc[current_idx - 1]

        # Calculate gap
        gap = (current_bar["open"] - prev_bar["close"]) / prev_bar["close"]

        # Calculate RVOL
        lookback = min(50, current_idx)
        avg_volume = daily_data.iloc[current_idx - lookback:current_idx]["volume"].mean()
        rvol = current_bar["volume"] / avg_volume if avg_volume > 0 else 0

        # Heuristic: Large gap + high volume often indicates earnings
        # Real implementation would check earnings calendar
        if abs(gap) >= self.min_gap and rvol >= self.min_rvol * 1.5:
            return {
                "gap_pct": gap * 100,
                "gap_direction": "up" if gap > 0 else "down",
                "rvol": rvol,
                "avg_volume": avg_volume,
                "prev_close": prev_bar["close"],
                "open": current_bar["open"],
                "is_likely_earnings": True
            }

        return None

    def scan_for_signals(
        self,
        ticker: str,
        daily_data: pd.DataFrame,
        minute_data: pd.DataFrame,
        date: datetime
    ) -> List[Signal]:
        """
        Scan for Earnings Play signals based on play type.
        """
        if self.play_type == "post_momentum":
            return self._scan_post_momentum(ticker, daily_data, date)
        elif self.play_type == "post_pullback":
            return self._scan_post_pullback(ticker, daily_data, date)
        elif self.play_type == "pre_drift":
            return self._scan_pre_drift(ticker, daily_data, date)
        else:
            return []

    def _scan_post_momentum(
        self,
        ticker: str,
        daily_data: pd.DataFrame,
        date: datetime
    ) -> List[Signal]:
        """
        Post-earnings momentum (gap and go on earnings beat).

        Entry criteria:
        - Gap up > 3% on likely earnings
        - RVOL > 2x
        - Price holds above gap (green candle preferred)
        """
        signals = []

        earnings_gap = self.detect_earnings_gap(daily_data, date)
        if earnings_gap is None:
            return signals

        # Only trade gap ups for momentum
        if earnings_gap["gap_direction"] != "up":
            return signals

        try:
            if hasattr(daily_data.index, 'date'):
                current_idx = daily_data.index.get_loc(
                    daily_data.index[daily_data.index.date == date.date()][0]
                )
            else:
                return signals
        except (IndexError, KeyError, AttributeError):
            return signals

        current_bar = daily_data.iloc[current_idx]

        # Check for strong close (holding the gap)
        day_range = current_bar["high"] - current_bar["low"]
        if day_range > 0:
            close_position = (current_bar["close"] - current_bar["low"]) / day_range
            if close_position < 0.4:  # Weak close, skip
                return signals

        atr = self.calculate_atr(daily_data.iloc[:current_idx + 1]).iloc[-1]

        entry_price = current_bar["close"]
        stop_loss = max(current_bar["low"], earnings_gap["prev_close"]) - (atr * 0.5)
        target = self.calculate_target(entry_price, stop_loss, config.FIXED_RR_RATIO)

        signal = Signal(
            ticker=ticker,
            timestamp=date,
            setup_type=self.name,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target,
            signal_strength=self._calculate_signal_strength(earnings_gap, current_bar),
            metadata={
                **earnings_gap,
                "play_type": "post_momentum",
                "close_position": close_position,
                "entry_day_range": day_range
            }
        )
        signals.append(signal)

        return signals

    def _scan_post_pullback(
        self,
        ticker: str,
        daily_data: pd.DataFrame,
        date: datetime
    ) -> List[Signal]:
        """
        Post-earnings pullback (buy dip after initial gap reaction).

        Entry criteria:
        - Had earnings gap 1-3 days ago
        - Pulled back to support (gap fill area or 10 EMA)
        - Showing signs of holding support
        """
        signals = []

        if daily_data.empty or len(daily_data) < 5:
            return signals

        # Look for earnings gap in the last 3 days
        for days_ago in range(1, 4):
            check_date = date - timedelta(days=days_ago)
            earnings_gap = self.detect_earnings_gap(daily_data, check_date)
            if earnings_gap and earnings_gap["gap_direction"] == "up":
                break
        else:
            return signals

        try:
            if hasattr(daily_data.index, 'date'):
                current_idx = daily_data.index.get_loc(
                    daily_data.index[daily_data.index.date == date.date()][0]
                )
            else:
                return signals
        except (IndexError, KeyError, AttributeError):
            return signals

        current_bar = daily_data.iloc[current_idx]

        # Calculate pullback support levels
        gap_fill_level = earnings_gap["prev_close"] * 1.02  # 2% above prior close
        ema_10 = self.calculate_ema(daily_data.iloc[:current_idx + 1]["close"], 10).iloc[-1]

        support_level = max(gap_fill_level, ema_10)

        # Check if price has pulled back to support
        if current_bar["low"] <= support_level * 1.01:  # Within 1% of support
            # Check for bounce (close above support)
            if current_bar["close"] > support_level:
                atr = self.calculate_atr(daily_data.iloc[:current_idx + 1]).iloc[-1]

                entry_price = current_bar["close"]
                stop_loss = support_level - (atr * 0.5)
                target = self.calculate_target(entry_price, stop_loss, config.FIXED_RR_RATIO)

                signal = Signal(
                    ticker=ticker,
                    timestamp=date,
                    setup_type=self.name,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_price=target,
                    signal_strength=0.6,  # Moderate strength for pullback plays
                    metadata={
                        **earnings_gap,
                        "play_type": "post_pullback",
                        "support_level": support_level,
                        "days_since_earnings": days_ago
                    }
                )
                signals.append(signal)

        return signals

    def _scan_pre_drift(
        self,
        ticker: str,
        daily_data: pd.DataFrame,
        date: datetime
    ) -> List[Signal]:
        """
        Pre-earnings drift (momentum into anticipated earnings).

        Entry criteria:
        - Strong recent momentum (10-day performance > 5%)
        - Above key moving averages
        - Increasing volume trend

        Note: Real implementation needs earnings calendar integration.
        """
        signals = []

        if daily_data.empty or len(daily_data) < 20:
            return signals

        try:
            if hasattr(daily_data.index, 'date'):
                current_idx = daily_data.index.get_loc(
                    daily_data.index[daily_data.index.date == date.date()][0]
                )
            else:
                return signals
        except (IndexError, KeyError, AttributeError):
            return signals

        if current_idx < 20:
            return signals

        current_bar = daily_data.iloc[current_idx]
        ten_days_ago = daily_data.iloc[current_idx - 10]

        # Calculate momentum
        momentum_10d = (current_bar["close"] - ten_days_ago["close"]) / ten_days_ago["close"]

        if momentum_10d < 0.05:  # Need at least 5% gain in 10 days
            return signals

        # Check MA alignment
        ema_10 = self.calculate_ema(daily_data.iloc[:current_idx + 1]["close"], 10).iloc[-1]
        ema_20 = self.calculate_ema(daily_data.iloc[:current_idx + 1]["close"], 20).iloc[-1]

        if current_bar["close"] < ema_10 or ema_10 < ema_20:
            return signals  # Not in uptrend

        # Check volume trend
        recent_vol = daily_data.iloc[current_idx - 5:current_idx + 1]["volume"].mean()
        prior_vol = daily_data.iloc[current_idx - 15:current_idx - 5]["volume"].mean()
        vol_expansion = recent_vol / prior_vol if prior_vol > 0 else 0

        if vol_expansion < 1.2:  # Need volume expansion
            return signals

        atr = self.calculate_atr(daily_data.iloc[:current_idx + 1]).iloc[-1]

        entry_price = current_bar["close"]
        stop_loss = ema_10 - (atr * 0.5)
        target = self.calculate_target(entry_price, stop_loss, config.FIXED_RR_RATIO)

        signal = Signal(
            ticker=ticker,
            timestamp=date,
            setup_type=self.name,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target,
            signal_strength=0.5,  # Lower strength for anticipatory trades
            metadata={
                "play_type": "pre_drift",
                "momentum_10d": momentum_10d * 100,
                "vol_expansion": vol_expansion,
                "ema_10": ema_10,
                "ema_20": ema_20
            }
        )
        signals.append(signal)

        return signals

    def _calculate_signal_strength(
        self,
        earnings_gap: Dict,
        current_bar: pd.Series
    ) -> float:
        """Calculate signal strength for earnings plays."""
        strength = 0.4

        # Gap size contribution
        gap_pct = abs(earnings_gap["gap_pct"])
        if gap_pct >= 10:
            strength += 0.2
        elif gap_pct >= 5:
            strength += 0.15

        # RVOL contribution
        rvol = earnings_gap["rvol"]
        if rvol >= 5:
            strength += 0.2
        elif rvol >= 3:
            strength += 0.15

        # Candle quality
        day_range = current_bar["high"] - current_bar["low"]
        if day_range > 0:
            close_position = (current_bar["close"] - current_bar["low"]) / day_range
            if close_position > 0.7:
                strength += 0.1

        return min(strength, 1.0)

    def check_exit(
        self,
        trade: Trade,
        current_bar: pd.Series,
        daily_data: pd.DataFrame
    ) -> Optional[tuple]:
        """Check exit conditions for Earnings Play trade."""
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

        return None
