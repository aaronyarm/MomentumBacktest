"""
Context Engine - The "Why" behind every trade.
Captures market conditions, relative strength, and contextual factors.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from polygon_fetcher import PolygonFetcher, DataManager
import config


class ContextEngine:
    """
    Generates context snapshots for trades.

    Captures:
    - Relative Volume (RVOL)
    - Tightness Factor
    - MA Surfing (distance from EMAs)
    - Market Regime (SPY/QQQ vs their MAs)
    - Float & Sector information
    """

    def __init__(self, data_manager: DataManager = None):
        self.data_manager = data_manager or DataManager()
        self.market_data: Dict[str, pd.DataFrame] = {}

    def preload_market_data(self, from_date: str, to_date: str):
        """Preload SPY and QQQ data for market regime analysis."""
        for ticker in config.MARKET_TICKERS:
            df = self.data_manager.fetcher.get_daily_bars(ticker, from_date, to_date)
            if not df.empty:
                self.market_data[ticker] = df
                # Pre-calculate indicators
                self.market_data[f"{ticker}_ema4"] = self._calculate_ema(df["close"], 4)
                self.market_data[f"{ticker}_sma50"] = self._calculate_sma(df["close"], 50)

    def generate_context_snapshot(
        self,
        ticker: str,
        entry_date: datetime,
        daily_data: pd.DataFrame,
        ticker_info: Dict = None
    ) -> Dict[str, Any]:
        """
        Generate a complete context snapshot for a trade entry.

        Args:
            ticker: Stock symbol
            entry_date: Date of trade entry
            daily_data: Daily OHLCV data for the ticker
            ticker_info: Dict with float, sector info

        Returns:
            Dictionary containing all context variables
        """
        context = {
            "entry_date": entry_date,
            "ticker": ticker
        }

        # Get the index for entry date
        try:
            entry_idx = daily_data.index.get_loc(
                daily_data.index[daily_data.index.date == entry_date.date()][0]
            )
        except (IndexError, KeyError):
            return context

        # Slice data up to entry
        data_to_entry = daily_data.iloc[:entry_idx + 1]

        # 1. Relative Volume
        context["rvol"] = self._calculate_rvol(data_to_entry)
        context["rvol_category"] = self._categorize_rvol(context["rvol"])

        # 2. Tightness Factor
        context["tightness"] = self._calculate_tightness(data_to_entry)
        context["tightness_category"] = self._categorize_tightness(context["tightness"])

        # 3. MA Surfing (distance from EMAs)
        ma_data = self._calculate_ma_surfing(data_to_entry)
        context.update(ma_data)

        # 4. Market Regime
        market_regime = self._get_market_regime(entry_date)
        context.update(market_regime)

        # 5. Float & Sector
        if ticker_info:
            context["float"] = ticker_info.get("float")
            context["float_category"] = self._categorize_float(context["float"])
            context["sector"] = ticker_info.get("sector", "Unknown")
        else:
            context["float"] = None
            context["float_category"] = "Unknown"
            context["sector"] = self._get_sector_from_map(ticker)

        # 6. Price Action Context
        price_context = self._calculate_price_context(data_to_entry)
        context.update(price_context)

        return context

    def _calculate_rvol(self, data: pd.DataFrame) -> float:
        """Calculate Relative Volume (Volume / SMA50)."""
        if data.empty or len(data) < config.RVOL_SMA_PERIOD:
            return 1.0

        current_volume = data.iloc[-1]["volume"]
        avg_volume = data["volume"].rolling(config.RVOL_SMA_PERIOD).mean().iloc[-1]

        if avg_volume > 0:
            return current_volume / avg_volume
        return 1.0

    def _categorize_rvol(self, rvol: float) -> str:
        """Categorize RVOL into buckets."""
        if rvol >= 5:
            return "5x+"
        elif rvol >= 3:
            return "3x-5x"
        elif rvol >= 2:
            return "2x-3x"
        elif rvol >= 1.5:
            return "1.5x-2x"
        else:
            return "<1.5x"

    def _calculate_tightness(self, data: pd.DataFrame) -> float:
        """
        Calculate Tightness Factor.
        (Max(Close, 5 days) - Min(Close, 5 days)) / ATR(20)
        """
        if data.empty or len(data) < max(config.TIGHTNESS_LOOKBACK, config.ATR_PERIOD):
            return 1.0

        # Last 5 days close range
        recent_closes = data["close"].tail(config.TIGHTNESS_LOOKBACK)
        close_range = recent_closes.max() - recent_closes.min()

        # ATR calculation
        high = data["high"]
        low = data["low"]
        close = data["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(config.ATR_PERIOD).mean().iloc[-1]

        if atr > 0:
            return close_range / atr
        return 1.0

    def _categorize_tightness(self, tightness: float) -> str:
        """Categorize tightness into buckets."""
        if tightness <= 0.5:
            return "Very_Tight"
        elif tightness <= 1.0:
            return "Tight"
        elif tightness <= 1.5:
            return "Normal"
        elif tightness <= 2.0:
            return "Loose"
        else:
            return "Very_Loose"

    def _calculate_ma_surfing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate distance from 10 EMA and 20 EMA."""
        result = {
            "dist_from_ema10": None,
            "dist_from_ema20": None,
            "ema10_above_ema20": None,
            "ma_alignment": "Unknown"
        }

        if data.empty or len(data) < 20:
            return result

        close = data["close"]
        current_price = close.iloc[-1]

        ema10 = self._calculate_ema(close, 10).iloc[-1]
        ema20 = self._calculate_ema(close, 20).iloc[-1]

        # Distance as percentage
        result["dist_from_ema10"] = ((current_price - ema10) / ema10) * 100
        result["dist_from_ema20"] = ((current_price - ema20) / ema20) * 100
        result["ema10_above_ema20"] = ema10 > ema20

        # MA Alignment categories
        if current_price > ema10 > ema20:
            result["ma_alignment"] = "Bullish_Stacked"
        elif current_price < ema10 < ema20:
            result["ma_alignment"] = "Bearish_Stacked"
        elif ema10 > ema20:
            result["ma_alignment"] = "Bullish_Mixed"
        else:
            result["ma_alignment"] = "Bearish_Mixed"

        return result

    def _get_market_regime(self, date: datetime) -> Dict[str, Any]:
        """
        Get market regime based on SPY and QQQ position relative to their MAs.
        """
        regime = {
            "spy_above_ema4": None,
            "spy_above_sma50": None,
            "qqq_above_ema4": None,
            "qqq_above_sma50": None,
            "market_regime": "Unknown"
        }

        for ticker in ["SPY", "QQQ"]:
            if ticker not in self.market_data:
                continue

            df = self.market_data[ticker]
            ema4 = self.market_data.get(f"{ticker}_ema4")
            sma50 = self.market_data.get(f"{ticker}_sma50")

            if df is None or ema4 is None or sma50 is None:
                continue

            try:
                idx = df.index.get_loc(
                    df.index[df.index.date == date.date()][0]
                )
                current_close = df.iloc[idx]["close"]
                current_ema4 = ema4.iloc[idx]
                current_sma50 = sma50.iloc[idx]

                prefix = ticker.lower()
                regime[f"{prefix}_above_ema4"] = current_close > current_ema4
                regime[f"{prefix}_above_sma50"] = current_close > current_sma50

            except (IndexError, KeyError):
                continue

        # Determine overall market regime
        spy_bullish = regime.get("spy_above_ema4") and regime.get("spy_above_sma50")
        qqq_bullish = regime.get("qqq_above_ema4") and regime.get("qqq_above_sma50")

        if spy_bullish and qqq_bullish:
            regime["market_regime"] = "Risk_On"
        elif not regime.get("spy_above_ema4") and not regime.get("qqq_above_ema4"):
            regime["market_regime"] = "Risk_Off"
        else:
            regime["market_regime"] = "Mixed"

        return regime

    def _categorize_float(self, float_shares: Optional[float]) -> str:
        """Categorize float size."""
        if float_shares is None:
            return "Unknown"

        # Convert to millions
        float_m = float_shares / 1_000_000

        if float_m < 10:
            return "Micro_Float"
        elif float_m < 20:
            return "Low_Float"
        elif float_m < 50:
            return "Small_Float"
        elif float_m < 200:
            return "Mid_Float"
        else:
            return "Large_Float"

    def _get_sector_from_map(self, ticker: str) -> str:
        """Get sector from hardcoded map."""
        for sector, tickers in config.SECTORS.items():
            if ticker in tickers:
                return sector
        return "Unknown"

    def _calculate_price_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate additional price action context."""
        context = {
            "adr_pct": None,
            "days_from_52w_high": None,
            "pct_from_52w_high": None,
            "trend_20d": None
        }

        if data.empty or len(data) < 20:
            return context

        # ADR (Average Daily Range as %)
        daily_range_pct = ((data["high"] - data["low"]) / data["close"]) * 100
        context["adr_pct"] = daily_range_pct.rolling(20).mean().iloc[-1]

        # Distance from 52-week high (if we have enough data)
        if len(data) >= 252:
            high_52w = data["high"].tail(252).max()
            current_price = data.iloc[-1]["close"]
            context["pct_from_52w_high"] = ((current_price - high_52w) / high_52w) * 100

            # Days since 52w high
            high_idx = data["high"].tail(252).idxmax()
            context["days_from_52w_high"] = (data.index[-1] - high_idx).days

        # 20-day trend (simple: up, down, sideways)
        price_20d_ago = data.iloc[-20]["close"]
        current_price = data.iloc[-1]["close"]
        change_20d = ((current_price - price_20d_ago) / price_20d_ago) * 100

        if change_20d > 5:
            context["trend_20d"] = "Uptrend"
        elif change_20d < -5:
            context["trend_20d"] = "Downtrend"
        else:
            context["trend_20d"] = "Sideways"

        return context

    @staticmethod
    def _calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Calculate SMA."""
        return series.rolling(window=period).mean()


class ContextAnalyzer:
    """
    Analyzes context data to find correlations with trade performance.
    """

    def __init__(self, trades_df: pd.DataFrame):
        """
        Initialize with trades DataFrame.

        Args:
            trades_df: DataFrame with trade data including context columns
        """
        self.trades_df = trades_df

    def find_best_contexts(
        self,
        min_profit_factor: float = 2.0,
        min_trades: int = 10
    ) -> pd.DataFrame:
        """
        Find context combinations correlated with high profit factors.
        """
        results = []

        # Context columns to analyze
        context_cols = [col for col in self.trades_df.columns if col.startswith("ctx_")]

        for col in context_cols:
            if self.trades_df[col].dtype == 'object':
                # Categorical column
                for value in self.trades_df[col].unique():
                    subset = self.trades_df[self.trades_df[col] == value]
                    if len(subset) >= min_trades:
                        stats = self._calculate_stats(subset)
                        if stats["profit_factor"] >= min_profit_factor:
                            results.append({
                                "context": col.replace("ctx_", ""),
                                "value": value,
                                **stats
                            })
            else:
                # Numeric column - create buckets
                try:
                    buckets = pd.qcut(self.trades_df[col], q=4, duplicates='drop')
                    for bucket in buckets.unique():
                        subset = self.trades_df[buckets == bucket]
                        if len(subset) >= min_trades:
                            stats = self._calculate_stats(subset)
                            if stats["profit_factor"] >= min_profit_factor:
                                results.append({
                                    "context": col.replace("ctx_", ""),
                                    "value": str(bucket),
                                    **stats
                                })
                except (ValueError, TypeError):
                    continue

        return pd.DataFrame(results).sort_values("profit_factor", ascending=False)

    def _calculate_stats(self, subset: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance stats for a subset of trades."""
        wins = subset[subset["pnl"] > 0]["pnl"].sum()
        losses = abs(subset[subset["pnl"] <= 0]["pnl"].sum())

        return {
            "trades": len(subset),
            "win_rate": (subset["pnl"] > 0).mean(),
            "profit_factor": wins / losses if losses > 0 else float("inf"),
            "avg_pnl": subset["pnl"].mean(),
            "total_pnl": subset["pnl"].sum()
        }

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Generate correlation matrix between context variables and PnL.
        """
        # Get numeric context columns
        numeric_cols = [
            col for col in self.trades_df.columns
            if col.startswith("ctx_") and self.trades_df[col].dtype in ['float64', 'int64']
        ]

        if not numeric_cols:
            return pd.DataFrame()

        # Add PnL for correlation
        cols_for_corr = numeric_cols + ["pnl", "pnl_pct"]
        return self.trades_df[cols_for_corr].corr()
