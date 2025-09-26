"""
Simple test to verify pattern detection functionality.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pattern_detection.pattern_engine import PatternDetectionEngine
from src.models.market_data import MarketData, OHLCV
import numpy as np
from datetime import datetime, timedelta


def create_test_data():
    """Create simple test data."""
    base_time = datetime(2024, 1, 1)
    ohlcv_data = []

    # Simple trending data
    for i in range(50):
        price = 100 + i * 0.5 + np.sin(i * 0.1) * 2
        ohlcv = OHLCV(
            timestamp=base_time + timedelta(hours=i),
            open=float(price),
            high=float(price + 0.5),
            low=float(price - 0.5),
            close=float(price),
            volume=1000 + i * 10,
        )
        ohlcv_data.append(ohlcv)

    return MarketData(symbol="TEST", timeframe="1h", ohlcv_data=ohlcv_data)


def test_pattern_engine():
    """Test basic pattern engine functionality."""
    print("Testing Pattern Detection Engine...")

    engine = PatternDetectionEngine()
    market_data = create_test_data()

    # Test full pattern detection
    patterns = engine.detect_patterns(market_data)
    print(f"✓ Detected {len(patterns)} patterns")

    for pattern in patterns:
        print(
            f"  - {pattern.pattern_type.value} (confidence: {pattern.confidence_score:.2f})"
        )

    print("🎉 Pattern detection engine test passed!")


if __name__ == "__main__":
    test_pattern_engine()
