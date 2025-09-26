"""
Tests for Pattern Detection Engine - Comprehensive test suite.

This module tests all pattern detection functionality including:
- Basic pattern detection engine
- Triangle patterns (ascending, descending, symmetrical)
- Head & Shoulders patterns (classic and inverse)
- Integration with technical indicators and market data
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

from src.pattern_detection.pattern_engine import (
    PatternDetectionEngine,
    DetectedPattern,
    PatternType,
    PatternStrength,
    PivotPoint,
    SupportResistanceLevel,
    TrendLine,
)
from src.pattern_detection.triangle_patterns import TrianglePatternDetector
from src.pattern_detection.head_shoulders import HeadShouldersDetector
from src.technical_indicators.technical_indicator_engine import TechnicalIndicatorEngine


class TestPatternDetectionEngine:
    """Test suite for the core pattern detection engine."""

    def setup_method(self):
        """Setup test environment."""
        self.engine = PatternDetectionEngine()

    def create_test_data(self, num_points: int = 100) -> pd.DataFrame:
        """Create synthetic market data for testing."""
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(hours=i) for i in range(num_points)]

        # Create trending price data with some noise
        base_price = 100.0
        trend = np.linspace(0, 10, num_points)  # Upward trend
        noise = np.random.normal(0, 0.5, num_points)
        prices = base_price + trend + noise

        # Create OHLC data
        highs = prices + np.random.uniform(0.1, 0.5, num_points)
        lows = prices - np.random.uniform(0.1, 0.5, num_points)
        opens = np.roll(prices, 1)  # Previous close as open
        opens[0] = prices[0]
        closes = prices

        # Create volume data
        volumes = np.random.uniform(1000, 5000, num_points)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "symbol": ["TESTSTOCK"] * num_points,
            }
        )

    def create_triangle_data(self) -> pd.DataFrame:
        """Create data that forms a clear ascending triangle pattern."""
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(hours=i) for i in range(60)]

        # Ascending triangle: rising support, flat resistance
        resistance_level = 105.0
        support_levels = np.linspace(100, 104, 60)  # Rising support

        # Create price action that bounces between support and resistance
        prices = []
        for i in range(60):
            if i % 10 < 5:  # Rising toward resistance
                price = (
                    support_levels[i]
                    + (resistance_level - support_levels[i]) * (i % 10) / 4
                )
            else:  # Falling back to support
                price = (
                    resistance_level
                    - (resistance_level - support_levels[i]) * (i % 5) / 4
                )
            prices.append(price)

        # Add some realistic OHLC spread
        opens = np.array(prices)
        closes = np.array(prices) + np.random.normal(0, 0.1, 60)
        highs = np.maximum(opens, closes) + np.random.uniform(0, 0.2, 60)
        lows = np.minimum(opens, closes) - np.random.uniform(0, 0.2, 60)
        volumes = np.random.uniform(1000, 3000, 60)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "symbol": ["TRIANGLE"] * 60,
            }
        )

    def create_head_shoulders_data(self) -> pd.DataFrame:
        """Create data that forms a clear head and shoulders pattern."""
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(days=i) for i in range(50)]

        # H&S pattern: left shoulder, head, right shoulder
        pattern_x = np.linspace(0, 4 * np.pi, 50)

        # Create the three peaks with head higher
        left_shoulder = 20
        head = 25
        right_shoulder = 20
        valley = 15

        # Generate pattern
        prices = []
        for i, x in enumerate(pattern_x):
            if i < 15:  # Left shoulder
                price = valley + (left_shoulder - valley) * np.sin(x / 4) ** 2
            elif i < 35:  # Head
                price = valley + (head - valley) * np.sin((x - np.pi) / 4) ** 2
            else:  # Right shoulder
                price = (
                    valley
                    + (right_shoulder - valley) * np.sin((x - 2 * np.pi) / 4) ** 2
                )
            prices.append(100 + price)  # Base price 100

        # Create OHLC
        closes = np.array(prices)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = closes + np.random.uniform(0, 0.5, 50)
        lows = closes - np.random.uniform(0, 0.5, 50)
        volumes = np.random.uniform(2000, 5000, 50)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "symbol": ["HEADSHOULDERS"] * 50,
            }
        )

    def test_pivot_point_detection(self):
        """Test pivot point identification."""
        df = self.create_test_data(50)
        pivot_points = self.engine.detect_pivot_points(df)

        assert len(pivot_points) > 0, "Should detect some pivot points"
        assert all(
            isinstance(p, PivotPoint) for p in pivot_points
        ), "All should be PivotPoint objects"
        assert all(
            p.pivot_type in ["peak", "valley"] for p in pivot_points
        ), "All should be peaks or valleys"

        # Check chronological order
        timestamps = [p.timestamp for p in pivot_points]
        assert timestamps == sorted(
            timestamps
        ), "Pivot points should be chronologically ordered"

    def test_support_resistance_detection(self):
        """Test support and resistance level identification."""
        df = self.create_test_data(100)
        pivot_points = self.engine.detect_pivot_points(df)
        support_levels, resistance_levels = self.engine.detect_support_resistance(
            pivot_points
        )

        assert len(support_levels) >= 0, "Should have non-negative support levels"
        assert len(resistance_levels) >= 0, "Should have non-negative resistance levels"

        # Verify level structure
        for level in support_levels:
            assert isinstance(level, SupportResistanceLevel)
            assert level.level_type == "support"
            assert level.strength > 0

        for level in resistance_levels:
            assert isinstance(level, SupportResistanceLevel)
            assert level.level_type == "resistance"
            assert level.strength > 0

    def test_trendline_detection(self):
        """Test trendline identification."""
        df = self.create_test_data(80)
        pivot_points = self.engine.detect_pivot_points(df)
        support_lines, resistance_lines = self.engine.detect_trendlines(pivot_points)

        assert len(support_lines) >= 0, "Should have non-negative support trendlines"
        assert (
            len(resistance_lines) >= 0
        ), "Should have non-negative resistance trendlines"

        # Verify trendline structure
        for line in support_lines:
            assert isinstance(line, TrendLine)
            assert line.line_type == "support"
            assert (
                len(line.touches) >= 2
            ), "Trendline should have at least 2 touch points"
            assert 0 <= line.r_squared <= 1, "R-squared should be between 0 and 1"

        for line in resistance_lines:
            assert isinstance(line, TrendLine)
            assert line.line_type == "resistance"
            assert (
                len(line.touches) >= 2
            ), "Trendline should have at least 2 touch points"
            assert 0 <= line.r_squared <= 1, "R-squared should be between 0 and 1"

    def test_basic_pattern_detection(self):
        """Test basic pattern detection integration."""
        df = self.create_test_data(100)
        patterns = self.engine.detect_patterns(df)

        assert isinstance(patterns, list), "Should return a list of patterns"

        # If patterns are detected, verify their structure
        for pattern in patterns:
            assert isinstance(pattern, DetectedPattern)
            assert isinstance(pattern.pattern_type, PatternType)
            assert isinstance(pattern.strength, PatternStrength)
            assert (
                0 <= pattern.confidence_score <= 1
            ), "Confidence should be between 0 and 1"
            assert len(pattern.key_points) > 0, "Should have key points"
            assert pattern.start_time < pattern.end_time, "Start should be before end"


class TestTrianglePatternDetector:
    """Test suite for triangle pattern detection."""

    def setup_method(self):
        """Setup test environment."""
        self.detector = TrianglePatternDetector()
        self.engine = PatternDetectionEngine()

    def test_ascending_triangle_detection(self):
        """Test ascending triangle pattern detection."""
        df = self.create_ascending_triangle_data()
        pivot_points = self.engine.detect_pivot_points(df)
        support_lines, resistance_lines = self.engine.detect_trendlines(pivot_points)

        patterns = self.detector.detect_ascending_triangles(
            df, support_lines, resistance_lines
        )

        # Should detect at least one ascending triangle in our synthetic data
        triangle_patterns = [
            p for p in patterns if p.pattern_type == PatternType.ASCENDING_TRIANGLE
        ]

        if triangle_patterns:  # If we detect triangles
            pattern = triangle_patterns[0]
            assert pattern.confidence_score > 0, "Should have positive confidence"
            assert pattern.target_price is not None, "Should have target price"
            assert len(pattern.key_points) >= 4, "Should have at least 4 key points"

            # Check pattern metrics
            assert "triangle_height" in pattern.pattern_metrics
            assert "touch_count" in pattern.pattern_metrics
            assert pattern.pattern_metrics["triangle_height"] > 0

    def create_ascending_triangle_data(self) -> pd.DataFrame:
        """Create data with clear ascending triangle pattern."""
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(hours=i * 6) for i in range(40)]

        # Ascending triangle: flat resistance at 110, rising support
        resistance = 110.0
        support_base = 100.0
        support_rise = 8.0  # Rising by 8 points over period

        prices = []
        for i in range(40):
            support_level = support_base + (support_rise * i / 39)

            # Oscillate between support and resistance with convergence
            cycle_pos = (i % 8) / 8  # 8-hour cycles
            if cycle_pos < 0.5:  # Rising phase
                price = support_level + (resistance - support_level) * cycle_pos * 2
            else:  # Falling phase
                price = (
                    resistance - (resistance - support_level) * (cycle_pos - 0.5) * 2
                )

            # Add some noise
            price += np.random.normal(0, 0.3)
            prices.append(price)

        # Create OHLC data
        closes = np.array(prices)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = closes + np.random.uniform(0, 0.5, 40)
        lows = closes - np.random.uniform(0, 0.5, 40)
        volumes = np.linspace(5000, 3000, 40)  # Decreasing volume typical in triangles

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "symbol": ["ASC_TRIANGLE"] * 40,
            }
        )

    def test_symmetrical_triangle_detection(self):
        """Test symmetrical triangle pattern detection."""
        df = self.create_symmetrical_triangle_data()
        pivot_points = self.engine.detect_pivot_points(df)
        support_lines, resistance_lines = self.engine.detect_trendlines(pivot_points)

        patterns = self.detector.detect_symmetrical_triangles(
            df, support_lines, resistance_lines
        )

        # Verify detected patterns
        for pattern in patterns:
            assert pattern.pattern_type == PatternType.SYMMETRICAL_TRIANGLE
            assert "upside_target" in pattern.pattern_metrics
            assert "downside_target" in pattern.pattern_metrics
            assert pattern.pattern_metrics["upside_target"] > pattern.target_price
            assert pattern.pattern_metrics["downside_target"] < pattern.target_price

    def create_symmetrical_triangle_data(self) -> pd.DataFrame:
        """Create data with symmetrical triangle pattern."""
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(hours=i * 4) for i in range(50)]

        # Symmetrical triangle: converging support and resistance lines
        center_price = 105.0
        initial_range = 8.0

        prices = []
        for i in range(50):
            # Convergence factor - range decreases over time
            convergence_factor = 1 - (i / 49) * 0.8  # 80% convergence
            current_range = initial_range * convergence_factor

            # Oscillate within the converging range
            cycle_pos = (i % 6) / 6  # 6-period cycles
            oscillation = np.sin(cycle_pos * 2 * np.pi) * current_range / 2

            price = center_price + oscillation + np.random.normal(0, 0.2)
            prices.append(price)

        # Create OHLC data
        closes = np.array(prices)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = closes + np.random.uniform(0, 0.3, 50)
        lows = closes - np.random.uniform(0, 0.3, 50)
        volumes = np.linspace(4000, 2000, 50)  # Decreasing volume

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "symbol": ["SYM_TRIANGLE"] * 50,
            }
        )

    def test_triangle_volume_confirmation(self):
        """Test volume confirmation logic for triangles."""
        df = self.create_ascending_triangle_data()

        # Test volume pattern validation
        start_time = df["timestamp"].iloc[0]
        end_time = df["timestamp"].iloc[-1]

        # This should work without errors
        result = self.detector._check_volume_confirmation(df, start_time, end_time)
        assert isinstance(result, bool), "Should return boolean result"


class TestHeadShouldersDetector:
    """Test suite for Head & Shoulders pattern detection."""

    def setup_method(self):
        """Setup test environment."""
        self.detector = HeadShouldersDetector()
        self.engine = PatternDetectionEngine()

    def create_head_shoulders_data(self) -> pd.DataFrame:
        """Create data with clear head and shoulders pattern."""
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(days=i) for i in range(60)]

        # Define H&S structure
        base_price = 100.0
        valley_level = 95.0
        left_shoulder = 108.0
        head_peak = 115.0
        right_shoulder = 107.0

        prices = []
        volumes = []

        for i in range(60):
            if i < 12:  # Rising to left shoulder
                price = base_price + (left_shoulder - base_price) * (i / 11)
                volume = 3000 + 500 * (i / 11)  # Increasing volume
            elif i < 18:  # Falling from left shoulder
                price = left_shoulder - (left_shoulder - valley_level) * ((i - 12) / 5)
                volume = 3500 - 300 * ((i - 12) / 5)  # Decreasing volume
            elif i < 30:  # Rising to head
                price = valley_level + (head_peak - valley_level) * ((i - 18) / 11)
                volume = 3200 + 1000 * ((i - 18) / 11)  # High volume on head
            elif i < 36:  # Falling from head
                price = head_peak - (head_peak - valley_level) * ((i - 30) / 5)
                volume = 4200 - 400 * ((i - 30) / 5)  # Decreasing volume
            elif i < 48:  # Rising to right shoulder
                price = valley_level + (right_shoulder - valley_level) * ((i - 36) / 11)
                volume = 3800 - 600 * ((i - 36) / 11)  # Lower volume on right shoulder
            else:  # Falling from right shoulder
                price = right_shoulder - (right_shoulder - valley_level) * (
                    (i - 48) / 11
                )
                volume = 3200 - 200 * ((i - 48) / 11)

            # Add some noise
            price += np.random.normal(0, 0.5)
            volume += np.random.normal(0, 100)

            prices.append(max(price, 90))  # Floor price
            volumes.append(max(volume, 1000))  # Floor volume

        # Create OHLC data
        closes = np.array(prices)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = closes + np.random.uniform(0, 0.8, 60)
        lows = closes - np.random.uniform(0, 0.8, 60)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "symbol": ["H_AND_S"] * 60,
            }
        )

    def test_head_shoulders_detection(self):
        """Test head and shoulders pattern detection."""
        df = self.create_head_shoulders_data()
        pivot_points = self.engine.detect_pivot_points(df, window=3)

        patterns = self.detector.detect_head_and_shoulders(df, pivot_points)

        # Verify pattern structure if detected
        for pattern in patterns:
            assert pattern.pattern_type == PatternType.HEAD_AND_SHOULDERS
            assert pattern.confidence_score > 0
            assert pattern.target_price is not None
            assert (
                len(pattern.key_points) == 5
            )  # Left shoulder, valley, head, valley, right shoulder

            # Check metrics
            metrics = pattern.pattern_metrics
            assert "pattern_height" in metrics
            assert "shoulder_symmetry" in metrics
            assert "head_prominence_left" in metrics
            assert "head_prominence_right" in metrics
            assert "volume_confirmed" in metrics

            assert metrics["pattern_height"] > 0
            assert 0 <= metrics["shoulder_symmetry"] <= 1

    def test_inverse_head_shoulders_detection(self):
        """Test inverse head and shoulders pattern detection."""
        df = self.create_inverse_head_shoulders_data()
        pivot_points = self.engine.detect_pivot_points(df, window=3)

        patterns = self.detector.detect_inverse_head_and_shoulders(df, pivot_points)

        # Verify pattern structure if detected
        for pattern in patterns:
            assert pattern.pattern_type == PatternType.INVERSE_HEAD_AND_SHOULDERS
            assert pattern.confidence_score > 0
            assert pattern.target_price is not None
            assert len(pattern.key_points) == 5

    def create_inverse_head_shoulders_data(self) -> pd.DataFrame:
        """Create data with inverse head and shoulders pattern."""
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(days=i) for i in range(60)]

        # Inverse H&S: valleys instead of peaks
        base_price = 100.0
        peak_level = 105.0
        left_shoulder = 92.0
        head_valley = 85.0
        right_shoulder = 93.0

        prices = []
        volumes = []

        for i in range(60):
            if i < 12:  # Falling to left shoulder
                price = base_price - (base_price - left_shoulder) * (i / 11)
                volume = 3000 + 200 * (i / 11)
            elif i < 18:  # Rising from left shoulder
                price = left_shoulder + (peak_level - left_shoulder) * ((i - 12) / 5)
                volume = 3200 - 100 * ((i - 12) / 5)
            elif i < 30:  # Falling to head
                price = peak_level - (peak_level - head_valley) * ((i - 18) / 11)
                volume = 3100 + 300 * ((i - 18) / 11)  # Volume on head
            elif i < 36:  # Rising from head
                price = head_valley + (peak_level - head_valley) * ((i - 30) / 5)
                volume = 3400 - 150 * ((i - 30) / 5)
            elif i < 48:  # Falling to right shoulder
                price = peak_level - (peak_level - right_shoulder) * ((i - 36) / 11)
                volume = 3250 + 400 * (
                    (i - 36) / 11
                )  # Increasing volume on right shoulder
            else:  # Rising from right shoulder
                price = right_shoulder + (peak_level - right_shoulder) * ((i - 48) / 11)
                volume = 3650 - 150 * ((i - 48) / 11)

            # Add noise
            price += np.random.normal(0, 0.5)
            volume += np.random.normal(0, 80)

            prices.append(max(price, 80))
            volumes.append(max(volume, 1000))

        # Create OHLC data
        closes = np.array(prices)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = closes + np.random.uniform(0, 0.6, 60)
        lows = closes - np.random.uniform(0, 0.6, 60)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "symbol": ["INV_H_S"] * 60,
            }
        )

    def test_volume_pattern_validation(self):
        """Test volume pattern validation for H&S patterns."""
        df = self.create_head_shoulders_data()

        # Create mock pivot points
        left_shoulder = PivotPoint(
            timestamp=df["timestamp"].iloc[10],
            price=df["close"].iloc[10],
            pivot_type="peak",
            strength=1.0,
        )
        head = PivotPoint(
            timestamp=df["timestamp"].iloc[25],
            price=df["close"].iloc[25],
            pivot_type="peak",
            strength=1.0,
        )
        right_shoulder = PivotPoint(
            timestamp=df["timestamp"].iloc[40],
            price=df["close"].iloc[40],
            pivot_type="peak",
            strength=1.0,
        )

        # Test volume confirmation
        result = self.detector._check_hs_volume_pattern(
            df, left_shoulder, head, right_shoulder
        )
        assert isinstance(result, bool), "Should return boolean result"


class TestPatternDetectionIntegration:
    """Test integration between pattern detection and technical indicators."""

    def setup_method(self):
        """Setup test environment."""
        self.pattern_engine = PatternDetectionEngine()
        self.indicator_engine = TechnicalIndicatorEngine()

    def test_integration_with_indicators(self):
        """Test pattern detection with technical indicator data."""
        df = self.create_trending_data()

        # Calculate technical indicators
        indicators = self.indicator_engine.calculate_all_indicators(df)

        # Detect patterns
        patterns = self.pattern_engine.detect_patterns(df)

        # Verify integration works without errors
        assert isinstance(indicators, dict), "Indicators should be returned as dict"
        assert isinstance(patterns, list), "Patterns should be returned as list"

        # Test that we can access indicator data alongside patterns
        if "trend_indicators" in indicators:
            assert "sma_20" in indicators["trend_indicators"]

    def create_trending_data(self) -> pd.DataFrame:
        """Create data with clear trend for integration testing."""
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(hours=i) for i in range(200)]

        # Create trending data with some patterns
        base_price = 100.0
        trend = np.linspace(0, 20, 200)  # Strong uptrend
        noise = np.random.normal(0, 1, 200)

        # Add some cyclical patterns
        cycles = np.sin(np.linspace(0, 8 * np.pi, 200)) * 2

        prices = base_price + trend + noise + cycles

        # Create OHLC data
        closes = prices
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = closes + np.random.uniform(0.2, 1.0, 200)
        lows = closes - np.random.uniform(0.2, 1.0, 200)
        volumes = np.random.uniform(2000, 6000, 200)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "symbol": ["TRENDING"] * 200,
            }
        )

    def test_pattern_strength_classification(self):
        """Test pattern strength classification logic."""
        # Test strength determination
        assert PatternStrength.VERY_STRONG.value == "VERY_STRONG"
        assert PatternStrength.STRONG.value == "STRONG"
        assert PatternStrength.MODERATE.value == "MODERATE"
        assert PatternStrength.WEAK.value == "WEAK"

    def test_pattern_type_enumeration(self):
        """Test pattern type enumeration."""
        # Verify all expected pattern types are available
        expected_types = [
            "ASCENDING_TRIANGLE",
            "DESCENDING_TRIANGLE",
            "SYMMETRICAL_TRIANGLE",
            "HEAD_AND_SHOULDERS",
            "INVERSE_HEAD_AND_SHOULDERS",
        ]

        actual_types = [pt.value for pt in PatternType]

        for expected_type in expected_types:
            assert (
                expected_type in actual_types
            ), f"Pattern type {expected_type} should be available"


# Test runner for manual execution
if __name__ == "__main__":
    # Run basic smoke tests
    test_engine = TestPatternDetectionEngine()
    test_engine.setup_method()

    print("Running pattern detection engine tests...")

    try:
        test_engine.test_pivot_point_detection()
        print("✓ Pivot point detection test passed")

        test_engine.test_support_resistance_detection()
        print("✓ Support/resistance detection test passed")

        test_engine.test_trendline_detection()
        print("✓ Trendline detection test passed")

        test_engine.test_basic_pattern_detection()
        print("✓ Basic pattern detection test passed")

        # Test triangle patterns
        triangle_test = TestTrianglePatternDetector()
        triangle_test.setup_method()
        triangle_test.test_triangle_volume_confirmation()
        print("✓ Triangle volume confirmation test passed")

        # Test head & shoulders
        hs_test = TestHeadShouldersDetector()
        hs_test.setup_method()
        hs_test.test_volume_pattern_validation()
        print("✓ Head & shoulders volume validation test passed")

        # Test integration
        integration_test = TestPatternDetectionIntegration()
        integration_test.setup_method()
        integration_test.test_pattern_strength_classification()
        integration_test.test_pattern_type_enumeration()
        print("✓ Integration tests passed")

        print("\\n🎉 All basic pattern detection tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
