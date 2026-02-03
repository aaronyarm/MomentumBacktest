"""
Trading Setup Classes for Momentum Strategies.
"""
from .base_setup import BaseSetup, Trade, Signal
from .orb_setup import ORBSetup
from .ep_setup import EpisodicPivotSetup
from .delayed_reaction import DelayedReactionSetup
from .earnings_play import EarningsPlaySetup

__all__ = [
    "BaseSetup",
    "Trade",
    "Signal",
    "ORBSetup",
    "EpisodicPivotSetup",
    "DelayedReactionSetup",
    "EarningsPlaySetup"
]
