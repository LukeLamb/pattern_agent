"""
Models Package - Core data structures and schemas
"""

# Import working models
from .market_data import MarketData, OHLCV, MarketDataType, MarketSession

# Try to import other models if they're working
try:
    from .pattern import Pattern, PatternType

    PATTERN_AVAILABLE = True
except ImportError:
    PATTERN_AVAILABLE = False

# Temporarily disable signal import until validators are fixed
SIGNAL_AVAILABLE = False

__all__ = [
    # Market data models (always available)
    "MarketData",
    "OHLCV",
    "MarketDataType",
    "MarketSession",
]

# Add imports based on availability
if PATTERN_AVAILABLE:
    __all__.extend(["Pattern", "PatternType"])
