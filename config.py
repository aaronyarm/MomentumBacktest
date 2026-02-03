"""
Configuration settings for the Search Arbitrage Backtester.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Polygon.io API Configuration
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "YOUR_API_KEY_HERE")
POLYGON_BASE_URL = "https://api.polygon.io"
RATE_LIMIT_CALLS = 5  # Free tier: 5 calls/minute
RATE_LIMIT_PERIOD = 60  # seconds

# Backtest Parameters
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2024-01-01"
INITIAL_CAPITAL = 100000
POSITION_SIZE_PCT = 0.10  # 10% of capital per trade
MAX_POSITIONS = 5

# ORB Setup Parameters
ORB_TIMEFRAMES = [3, 5, 15, 60]  # minutes
ORB_ENTRY_BUFFER = 0.01  # 1 cent above high

# Episodic Pivot (EP) Parameters
EP_GAP_THRESHOLD = 0.04  # 4% gap
EP_VOLUME_MULTIPLIER = 3.0  # 3x average volume
EP_LOOKBACK_DAYS = 20  # For 1-month high calculation

# Delayed Reaction Parameters
DR_MONITOR_START = 2  # Days after EP
DR_MONITOR_END = 5  # Days after EP
DR_INSIDE_DAY_TOLERANCE = 0.001  # 0.1% tolerance for inside day

# Context Engine Parameters
RVOL_SMA_PERIOD = 50
TIGHTNESS_LOOKBACK = 5
ATR_PERIOD = 20
EMA_FAST = 10
EMA_SLOW = 20
MARKET_EMA = 4
MARKET_SMA = 50

# Exit Strategy Parameters
TRAILING_STOP_EMA = 10
FIXED_RR_RATIO = 3.0  # 1:3 Risk:Reward

# Market Regime Tickers
MARKET_TICKERS = ["SPY", "QQQ"]

# Output Paths
OUTPUT_DIR = "output"
TRADE_LOG_FILE = "master_trade_log.csv"
DASHBOARD_FILE = "dashboard.html"

# Sector Classifications (simplified GICS-like)
SECTORS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "AVGO", "QCOM"],
    "Semiconductors": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "MRVL"],
    "Software": ["MSFT", "CRM", "ADBE", "NOW", "SNOW", "PLTR"],
    "Consumer": ["AMZN", "TSLA", "HD", "NKE", "SBUX", "MCD"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY"],
    "Financials": ["JPM", "BAC", "GS", "MS", "WFC", "C"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY"],
    "Communication": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "T"],
}
