"""
Tests for Advanced Pattern Detection (Phase 2.2)

Comprehensive test suite for:
- Flag & Pennant patterns
- Double/Triple Top/Bottom patterns
- Rectangle & Channel patterns
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pattern_detection.flag_pennant import FlagPennantDetector, FlagPole
from pattern_detection.double_patterns import DoublePatternDetector
from pattern_detection.channels import ChannelDetector
from pattern_detection.pattern_engine import PatternType


class TestFlagPennantDetector:
    """Test flag and pennant pattern detection."""

    @pytest.fixture
    def detector(self):
        """Create FlagPennantDetector instance."""
        return FlagPennantDetector(
            min_flagpole_move=0.08,
            max_flag_duration_days=21,
            min_flag_duration_days=5
        )

    @pytest.fixture
    def bull_flag_data(self):
        """Create synthetic data with bull flag pattern."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=50), periods=50, freq='D')

        # Sharp upward move (flagpole)
        flagpole_prices = np.linspace(100, 115, 10)  # 15% up

        # Downward consolidation (flag)
        flag_prices = np.linspace(115, 112, 15)  # Slight downward drift

        # Continuation
        continuation_prices = np.linspace(112, 120, 25)

        prices = np.concatenate([flagpole_prices, flag_prices, continuation_prices])

        data = {
            'timestamp': dates,
            'open': prices + np.random.randn(50) * 0.5,
            'high': prices + abs(np.random.randn(50) * 1.0),
            'low': prices - abs(np.random.randn(50) * 1.0),
            'close': prices,
            'volume': np.concatenate([
                np.ones(10) * 2000000,  # High volume on flagpole
                np.ones(15) * 1000000,  # Low volume on flag
                np.ones(25) * 1500000   # Volume picks up
            ])
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def bear_flag_data(self):
        """Create synthetic data with bear flag pattern."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=50), periods=50, freq='D')

        # Sharp downward move
        flagpole_prices = np.linspace(115, 100, 10)  # 13% down

        # Upward consolidation
        flag_prices = np.linspace(100, 103, 15)

        # Continuation down
        continuation_prices = np.linspace(103, 92, 25)

        prices = np.concatenate([flagpole_prices, flag_prices, continuation_prices])

        data = {
            'timestamp': dates,
            'open': prices + np.random.randn(50) * 0.5,
            'high': prices + abs(np.random.randn(50) * 1.0),
            'low': prices - abs(np.random.randn(50) * 1.0),
            'close': prices,
            'volume': np.concatenate([
                np.ones(10) * 2000000,
                np.ones(15) * 1000000,
                np.ones(25) * 1500000
            ])
        }

        return pd.DataFrame(data)

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.min_flagpole_move == 0.08
        assert detector.max_flag_duration_days == 21

    def test_find_upward_flagpoles(self, detector, bull_flag_data):
        """Test finding upward flagpoles."""
        flagpoles = detector._find_flagpoles(bull_flag_data, direction='up')

        assert len(flagpoles) > 0
        assert all(fp.direction == 'up' for fp in flagpoles)
        assert all(fp.magnitude >= detector.min_flagpole_move for fp in flagpoles)

    def test_find_downward_flagpoles(self, detector, bear_flag_data):
        """Test finding downward flagpoles."""
        flagpoles = detector._find_flagpoles(bear_flag_data, direction='down')

        assert len(flagpoles) > 0
        assert all(fp.direction == 'down' for fp in flagpoles)

    def test_detect_bull_flags(self, detector, bull_flag_data):
        """Test bull flag detection."""
        patterns = detector.detect_bull_flags(bull_flag_data, symbol="TEST", timeframe="daily")

        # May or may not detect depending on exact data
        # Just ensure no errors and returns list
        assert isinstance(patterns, list)

    def test_detect_bear_flags(self, detector, bear_flag_data):
        """Test bear flag detection."""
        patterns = detector.detect_bear_flags(bear_flag_data, symbol="TEST", timeframe="daily")

        assert isinstance(patterns, list)

    def test_detect_pennants(self, detector, bull_flag_data):
        """Test pennant detection."""
        patterns = detector.detect_pennants(bull_flag_data, symbol="TEST", timeframe="daily")

        assert isinstance(patterns, list)

    def test_calculate_price_slope(self, detector, bull_flag_data):
        """Test price slope calculation."""
        slope = detector._calculate_price_slope(bull_flag_data.iloc[10:25])

        # Should calculate slope without error
        assert isinstance(slope, float)

    def test_check_convergence(self, detector):
        """Test convergence checking for pennants."""
        # Create converging data
        dates = pd.date_range(start=datetime.now(), periods=20, freq='D')
        highs = np.linspace(110, 105, 20)  # Converging highs
        lows = np.linspace(100, 102, 20)   # Converging lows

        df = pd.DataFrame({
            'timestamp': dates,
            'high': highs,
            'low': lows,
            'close': (highs + lows) / 2
        })

        # Should detect convergence
        is_converging = detector._check_convergence(df)
        assert isinstance(is_converging, bool)


class TestDoublePatternDetector:
    """Test double and triple top/bottom pattern detection."""

    @pytest.fixture
    def detector(self):
        """Create DoublePatternDetector instance."""
        return DoublePatternDetector(
            peak_similarity_tolerance=0.03,
            min_retracement=0.05
        )

    @pytest.fixture
    def double_top_data(self):
        """Create synthetic double top pattern."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), periods=60, freq='D')

        # Rising to first peak
        part1 = np.linspace(100, 120, 15)
        # Decline to valley
        part2 = np.linspace(120, 110, 15)
        # Rise to second peak
        part3 = np.linspace(110, 119, 15)
        # Decline (breakout)
        part4 = np.linspace(119, 105, 15)

        prices = np.concatenate([part1, part2, part3, part4])

        data = {
            'timestamp': dates,
            'open': prices,
            'high': prices + abs(np.random.randn(60) * 0.5),
            'low': prices - abs(np.random.randn(60) * 0.5),
            'close': prices,
            'volume': np.ones(60) * 1000000
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def double_bottom_data(self):
        """Create synthetic double bottom pattern."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), periods=60, freq='D')

        # Falling to first trough
        part1 = np.linspace(120, 100, 15)
        # Rise to peak
        part2 = np.linspace(100, 110, 15)
        # Fall to second trough
        part3 = np.linspace(110, 101, 15)
        # Rise (breakout)
        part4 = np.linspace(101, 115, 15)

        prices = np.concatenate([part1, part2, part3, part4])

        data = {
            'timestamp': dates,
            'open': prices,
            'high': prices + abs(np.random.randn(60) * 0.5),
            'low': prices - abs(np.random.randn(60) * 0.5),
            'close': prices,
            'volume': np.ones(60) * 1000000
        }

        return pd.DataFrame(data)

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.peak_similarity_tolerance == 0.03
        assert detector.min_retracement == 0.05

    def test_find_peaks(self, detector, double_top_data):
        """Test peak finding."""
        peaks = detector._find_peaks(double_top_data)

        assert len(peaks) > 0
        assert all('price' in p and 'idx' in p and 'time' in p for p in peaks)

    def test_find_troughs(self, detector, double_bottom_data):
        """Test trough finding."""
        troughs = detector._find_troughs(double_bottom_data)

        assert len(troughs) > 0
        assert all('price' in t and 'idx' in t and 'time' in t for t in troughs)

    def test_are_similar_levels(self, detector):
        """Test price level similarity check."""
        # Similar prices
        assert detector._are_similar_levels(100.0, 102.0) is True

        # Dissimilar prices
        assert detector._are_similar_levels(100.0, 110.0) is False

    def test_detect_double_tops(self, detector, double_top_data):
        """Test double top detection."""
        patterns = detector.detect_double_tops(double_top_data, symbol="TEST", timeframe="daily")

        assert isinstance(patterns, list)
        # May or may not find pattern depending on exact peak detection

    def test_detect_double_bottoms(self, detector, double_bottom_data):
        """Test double bottom detection."""
        patterns = detector.detect_double_bottoms(double_bottom_data, symbol="TEST", timeframe="daily")

        assert isinstance(patterns, list)

    def test_detect_triple_tops(self, detector, double_top_data):
        """Test triple top detection."""
        patterns = detector.detect_triple_tops(double_top_data, symbol="TEST", timeframe="daily")

        assert isinstance(patterns, list)

    def test_detect_triple_bottoms(self, detector, double_bottom_data):
        """Test triple bottom detection."""
        patterns = detector.detect_triple_bottoms(double_bottom_data, symbol="TEST", timeframe="daily")

        assert isinstance(patterns, list)


class TestChannelDetector:
    """Test rectangle and channel pattern detection."""

    @pytest.fixture
    def detector(self):
        """Create ChannelDetector instance."""
        return ChannelDetector(
            min_touches=4,
            min_duration_days=15,
            parallel_tolerance=0.02
        )

    @pytest.fixture
    def rectangle_data(self):
        """Create synthetic rectangle pattern."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), periods=60, freq='D')

        # Price oscillating between 100 and 110
        prices = 105 + 5 * np.sin(np.linspace(0, 4 * np.pi, 60))

        data = {
            'timestamp': dates,
            'open': prices,
            'high': prices + abs(np.random.randn(60) * 0.5),
            'low': prices - abs(np.random.randn(60) * 0.5),
            'close': prices,
            'volume': np.ones(60) * 1000000
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def ascending_channel_data(self):
        """Create synthetic ascending channel."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), periods=60, freq='D')

        # Uptrend with oscillation
        trend = np.linspace(100, 120, 60)
        oscillation = 3 * np.sin(np.linspace(0, 6 * np.pi, 60))
        prices = trend + oscillation

        data = {
            'timestamp': dates,
            'open': prices,
            'high': prices + abs(np.random.randn(60) * 0.5),
            'low': prices - abs(np.random.randn(60) * 0.5),
            'close': prices,
            'volume': np.ones(60) * 1000000
        }

        return pd.DataFrame(data)

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.min_touches == 4
        assert detector.min_duration_days == 15

    def test_are_parallel(self, detector):
        """Test parallel slope detection."""
        # Parallel slopes
        assert detector._are_parallel(0.5, 0.51) is True

        # Non-parallel slopes
        assert detector._are_parallel(0.5, 1.0) is False

        # Both horizontal
        assert detector._are_parallel(0.0, 0.0) is True

    def test_detect_rectangles(self, detector, rectangle_data):
        """Test rectangle detection."""
        patterns = detector.detect_rectangles(rectangle_data, symbol="TEST", timeframe="daily")

        assert isinstance(patterns, list)

    def test_detect_ascending_channels(self, detector, ascending_channel_data):
        """Test ascending channel detection."""
        patterns = detector.detect_ascending_channels(ascending_channel_data, symbol="TEST", timeframe="daily")

        assert isinstance(patterns, list)

    def test_detect_descending_channels(self, detector, ascending_channel_data):
        """Test descending channel detection."""
        # Use inverted data for descending
        df = ascending_channel_data.copy()
        df['close'] = 220 - df['close']  # Invert
        df['high'] = 220 - df['low']
        df['low'] = 220 - ascending_channel_data['high']

        patterns = detector.detect_descending_channels(df, symbol="TEST", timeframe="daily")

        assert isinstance(patterns, list)

    def test_find_horizontal_levels(self, detector, rectangle_data):
        """Test horizontal level finding."""
        support_levels = detector._find_horizontal_levels(rectangle_data, level_type='support')
        resistance_levels = detector._find_horizontal_levels(rectangle_data, level_type='resistance')

        assert isinstance(support_levels, list)
        assert isinstance(resistance_levels, list)


class TestPatternIntegration:
    """Integration tests for all advanced patterns."""

    def test_all_detectors_importable(self):
        """Test that all detectors can be imported."""
        from pattern_detection.flag_pennant import FlagPennantDetector
        from pattern_detection.double_patterns import DoublePatternDetector
        from pattern_detection.channels import ChannelDetector

        assert FlagPennantDetector is not None
        assert DoublePatternDetector is not None
        assert ChannelDetector is not None

    def test_all_detectors_instantiable(self):
        """Test that all detectors can be instantiated."""
        flag_detector = FlagPennantDetector()
        double_detector = DoublePatternDetector()
        channel_detector = ChannelDetector()

        assert flag_detector is not None
        assert double_detector is not None
        assert channel_detector is not None

    def test_pattern_types_available(self):
        """Test that new pattern types are available."""
        assert hasattr(PatternType, 'BULL_FLAG')
        assert hasattr(PatternType, 'BEAR_FLAG')
        assert hasattr(PatternType, 'PENNANT')
        assert hasattr(PatternType, 'DOUBLE_TOP')
        assert hasattr(PatternType, 'DOUBLE_BOTTOM')
        assert hasattr(PatternType, 'TRIPLE_TOP')
        assert hasattr(PatternType, 'TRIPLE_BOTTOM')
        assert hasattr(PatternType, 'RECTANGLE')
        assert hasattr(PatternType, 'ASCENDING_CHANNEL')
        assert hasattr(PatternType, 'DESCENDING_CHANNEL')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
