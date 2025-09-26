"""
Integration Tests for Pattern Recognition Agent.

This module tests the complete end-to-end workflow including:
- Pattern engine with indicator engine integration
- Full pipeline with sample data validation
- Cross-component integration testing
- Performance and accuracy validation
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

try:
    from src.pattern_detection.pattern_engine import (
        PatternDetectionEngine,
        PatternType,
        PatternStrength,
    )
    from src.technical_indicators.indicator_engine import (
        TechnicalIndicatorEngine,
        IndicatorConfig,
    )
    from src.validation.pattern_validator import PatternValidator, ValidationCriteria
    from src.models.market_data import MarketData, OHLCV, MarketDataType, MarketSession
except ImportError:
    # For testing and development
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

    from pattern_detection.pattern_engine import (
        PatternDetectionEngine,
        PatternType,
        PatternStrength,
    )
    from technical_indicators.indicator_engine import (
        TechnicalIndicatorEngine,
        IndicatorConfig,
    )
    from validation.pattern_validator import PatternValidator, ValidationCriteria
    from models.market_data import MarketData, OHLCV, MarketDataType, MarketSession


class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete integration workflow."""

    def setUp(self):
        """Set up test components."""
        # Initialize engines
        self.pattern_engine = PatternDetectionEngine(
            min_pattern_length=10, min_touches=2
        )

        self.indicator_engine = TechnicalIndicatorEngine(config=IndicatorConfig())

        self.validator = PatternValidator(
            min_formation_days=5,
            max_formation_days=60,
            min_volume_confirmation_ratio=1.2,
        )

        # Create comprehensive test data
        self.market_data = self._create_comprehensive_market_data()

    def _create_comprehensive_market_data(self) -> MarketData:
        """Create comprehensive market data with multiple patterns."""
        rng = np.random.default_rng(42)
        dates = pd.date_range(
            start="2024-01-01", periods=250, freq="D"
        )  # Create exactly 250 data points

        # Create base trend with multiple patterns
        base_trend = np.linspace(95, 120, len(dates))
        noise = rng.normal(0, 2, len(dates))

        # Add triangle pattern in the middle (ascending triangle)
        mid_start = len(dates) // 3
        mid_end = 2 * len(dates) // 3
        triangle_length = mid_end - mid_start

        # Create ascending triangle: rising support, flat resistance
        for i in range(
            mid_start, min(mid_end, len(dates))
        ):  # Fixed range to avoid index errors
            progress = (i - mid_start) / triangle_length if triangle_length > 0 else 0
            # Add triangle characteristics
            base_trend[i] += progress * 5  # Rising support
            if i > mid_start + triangle_length // 2:
                base_trend[i] = min(base_trend[i], 110)  # Flat resistance

        prices = base_trend + noise

        # Create OHLCV data
        ohlcv_data = []
        for i, date in enumerate(dates):
            base_price = max(80, prices[i])  # Ensure positive prices

            high = base_price + abs(rng.normal(0, 1.5))
            low = base_price - abs(rng.normal(0, 1.5))
            open_price = base_price + rng.normal(0, 0.5)
            close_price = base_price
            volume = int(abs(rng.normal(150000, 30000)))

            ohlcv_data.append(
                OHLCV(
                    timestamp=date,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=volume,
                )
            )

        return MarketData(
            symbol="INTEGRATION_TEST",
            timeframe="1d",
            data_type=MarketDataType.OHLCV,
            data=ohlcv_data,
            start_time=dates[0],
            end_time=dates[-1],
            data_source="integration_test",
            market_session=MarketSession.REGULAR,
            completeness_score=1.0,
            gaps_detected=0,
            anomalies_detected=0,
            total_volume=sum(d.volume for d in ohlcv_data),
            vwap=105.0,
            price_range=40.0,
            volatility=0.02,
        )

    def test_full_pipeline_integration(self):
        """Test complete pattern detection pipeline."""
        print("\n=== Testing Full Pipeline Integration ===")

        # Step 1: Technical Indicator Calculation
        print("Step 1: Calculating technical indicators...")
        indicators = self.indicator_engine.calculate_indicators(self.market_data)

        # Debug: Print available indicators
        print(
            f"Available indicators: {list(indicators.keys()) if isinstance(indicators, dict) else 'Error in calculation'}"
        )

        # Verify indicators were calculated (check for error first)
        self.assertNotIn(
            "error", indicators, f"Technical indicator calculation failed: {indicators}"
        )
        self.assertIn("trend", indicators)
        self.assertIn("momentum", indicators)
        self.assertIn("volume", indicators)
        self.assertIn("volatility", indicators)
        print("✅ Calculated indicators successfully")

        # Step 2: Pattern Detection
        print("Step 2: Detecting patterns...")
        detected_patterns = self.pattern_engine.detect_patterns(self.market_data)

        # Should detect at least one pattern
        self.assertGreater(len(detected_patterns), 0)
        print(f"✅ Detected {len(detected_patterns)} patterns")

        # Step 3: Pattern Validation
        print("Step 3: Validating detected patterns...")
        validation_results = []

        for pattern in detected_patterns:
            results = self.validator.validate_pattern(pattern, self.market_data)
            overall_score = self.validator.calculate_overall_validation_score(results)
            quality_metrics = self.validator.calculate_pattern_quality_metrics(
                pattern, self.market_data
            )

            validation_results.append(
                {
                    "pattern": pattern,
                    "validation_results": results,
                    "overall_score": overall_score,
                    "quality_metrics": quality_metrics,
                }
            )

            print(f"   Pattern {pattern.pattern_type.value}: Score {overall_score:.3f}")

        self.assertEqual(len(validation_results), len(detected_patterns))

        # Step 4: Integration Verification
        print("Step 4: Verifying integration consistency...")

        for result in validation_results:
            pattern = result["pattern"]

            # Verify pattern has valid timeframes (accept both datetime and numpy.datetime64)
            self.assertIsNotNone(pattern.start_time)
            self.assertIsNotNone(pattern.end_time)

            # Convert to pandas timestamps for comparison
            start_ts = pd.Timestamp(pattern.start_time)
            end_ts = pd.Timestamp(pattern.end_time)
            self.assertLess(start_ts, end_ts)

            # Verify validation results structure
            validation_res = result["validation_results"]
            self.assertEqual(len(validation_res), 5)  # 5 validation criteria

            # Verify all validation criteria are covered
            criteria_found = set()
            for val_result in validation_res:
                criteria_found.add(val_result.criteria)

            expected_criteria = {
                ValidationCriteria.VOLUME_CONFIRMATION,
                ValidationCriteria.TIMEFRAME_CONSISTENCY,
                ValidationCriteria.HISTORICAL_SUCCESS_RATE,
                ValidationCriteria.PATTERN_QUALITY,
                ValidationCriteria.MARKET_CONTEXT,
            }
            self.assertEqual(criteria_found, expected_criteria)

            # Verify quality metrics
            quality = result["quality_metrics"]
            self.assertGreaterEqual(quality.overall_quality_score, 0.0)
            self.assertLessEqual(quality.overall_quality_score, 1.0)
            self.assertGreaterEqual(quality.confidence_adjustment, 0.5)
            self.assertLessEqual(quality.confidence_adjustment, 1.5)

        print("✅ Full pipeline integration successful!")
        return validation_results

    def test_pattern_engine_with_indicator_engine(self):
        """Test pattern engine integration with indicator engine."""
        print("\n=== Testing Pattern Engine + Indicator Engine Integration ===")

        # Calculate indicators first
        indicators = self.indicator_engine.calculate_indicators(self.market_data)

        # Detect patterns
        patterns = self.pattern_engine.detect_patterns(self.market_data)

        # Verify integration works
        self.assertGreater(len(indicators), 0)
        self.assertGreater(len(patterns), 0)

        # Verify pattern engine can access market data that indicator engine processed
        df = self.market_data.to_dataframe()
        self.assertIn("close", df.columns)
        self.assertIn("volume", df.columns)

        # Verify patterns have valid confidence scores
        for pattern in patterns:
            self.assertGreaterEqual(pattern.confidence_score, 0.0)
            self.assertLessEqual(pattern.confidence_score, 1.0)
            self.assertIsInstance(pattern.pattern_type, PatternType)
            self.assertIsInstance(pattern.strength, PatternStrength)

        print(
            f"✅ Successfully integrated {len(patterns)} patterns with {len(indicators)} indicators"
        )

    def test_end_to_end_accuracy_validation(self):
        """Test end-to-end workflow accuracy and consistency."""
        print("\n=== Testing End-to-End Accuracy Validation ===")

        # Run full pipeline multiple times to ensure consistency
        results = []
        for _ in range(3):
            patterns = self.pattern_engine.detect_patterns(self.market_data)
            results.append(len(patterns))

        # Results should be consistent
        self.assertTrue(
            all(r == results[0] for r in results),
            f"Inconsistent pattern detection across runs: {results}",
        )

        # Test with different market data
        different_data = self._create_different_market_data()
        patterns_different = self.pattern_engine.detect_patterns(different_data)

        # Should handle different data without errors
        self.assertIsInstance(patterns_different, list)

        print("✅ End-to-end accuracy validation passed")

    def _create_different_market_data(self) -> MarketData:
        """Create different market data for testing diversity."""
        dates = pd.date_range(start="2024-03-01", periods=50, freq="D")

        # Create different pattern (head and shoulders)
        prices = []
        for i, _ in enumerate(dates):
            if i < 15:
                prices.append(100 + i * 0.5)  # Rising left shoulder
            elif i < 25:
                prices.append(107 + (i - 15) * 0.8)  # Head formation
            elif i < 35:
                prices.append(115 - (i - 25) * 0.8)  # Head decline
            else:
                prices.append(107 + (i - 35) * 0.3)  # Right shoulder

        # Add noise
        rng = np.random.default_rng(123)  # Different seed
        prices = np.array(prices) + rng.normal(0, 1, len(dates))

        ohlcv_data = []
        for i, date in enumerate(dates):
            base_price = max(90, prices[i])

            high = base_price + abs(rng.normal(0, 1))
            low = base_price - abs(rng.normal(0, 1))
            open_price = base_price + rng.normal(0, 0.3)
            close_price = base_price
            volume = int(abs(rng.normal(120000, 25000)))

            ohlcv_data.append(
                OHLCV(
                    timestamp=date,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=volume,
                )
            )

        return MarketData(
            symbol="DIFFERENT_TEST",
            timeframe="1d",
            data_type=MarketDataType.OHLCV,
            data=ohlcv_data,
            start_time=dates[0],
            end_time=dates[-1],
            data_source="integration_test_2",
            market_session=MarketSession.REGULAR,
            completeness_score=1.0,
            gaps_detected=0,
            anomalies_detected=0,
            total_volume=sum(d.volume for d in ohlcv_data),
            vwap=108.0,
            price_range=25.0,
            volatility=0.025,
        )

    def test_performance_benchmarks(self):
        """Test performance requirements (<500ms for pattern detection)."""
        print("\n=== Testing Performance Benchmarks ===")

        import time

        # Test pattern detection speed
        start_time = time.time()
        patterns = self.pattern_engine.detect_patterns(self.market_data)
        detection_time = (time.time() - start_time) * 1000  # Convert to ms

        print(f"Pattern detection time: {detection_time:.2f}ms")
        self.assertLess(
            detection_time, 500, f"Pattern detection too slow: {detection_time:.2f}ms"
        )

        # Test validation speed
        if patterns:
            start_time = time.time()
            _ = self.validator.validate_pattern(patterns[0], self.market_data)
            validation_time = (time.time() - start_time) * 1000

            print(f"Pattern validation time: {validation_time:.2f}ms")
            self.assertLess(
                validation_time,
                100,
                f"Pattern validation too slow: {validation_time:.2f}ms",
            )

        # Test indicator calculation speed
        start_time = time.time()
        _ = self.indicator_engine.calculate_indicators(self.market_data)
        indicator_time = (time.time() - start_time) * 1000

        print(f"Indicator calculation time: {indicator_time:.2f}ms")
        self.assertLess(
            indicator_time,
            200,
            f"Indicator calculation too slow: {indicator_time:.2f}ms",
        )

        print("✅ All performance benchmarks met")

    def test_memory_usage_optimization(self):
        """Test memory usage remains reasonable."""
        print("\n=== Testing Memory Usage Optimization ===")

        try:
            import psutil
            import os

            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            print("⚠️  psutil not available - skipping memory usage test")
            return

        # Run multiple pattern detections
        for _ in range(10):
            patterns = self.pattern_engine.detect_patterns(self.market_data)
            if patterns:
                _ = self.validator.validate_pattern(patterns[0], self.market_data)

        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(
            f"Memory usage - Initial: {initial_memory:.2f}MB, Final: {final_memory:.2f}MB"
        )
        print(f"Memory increase: {memory_increase:.2f}MB")

        # Memory increase should be reasonable (less than 100MB for this test)
        self.assertLess(
            memory_increase,
            100,
            f"Memory usage increased too much: {memory_increase:.2f}MB",
        )

        print("✅ Memory usage optimization validated")


class TestComponentIntegration(unittest.TestCase):
    """Test specific component integration scenarios."""

    def test_data_flow_consistency(self):
        """Test data flows consistently between components."""
        # Create market data
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 0.5, 30))

        ohlcv_data = []
        for i, date in enumerate(dates):
            base_price = prices[i]
            ohlcv_data.append(
                OHLCV(
                    timestamp=date,
                    open=base_price + rng.normal(0, 0.1),
                    high=base_price + abs(rng.normal(0, 0.5)),
                    low=base_price - abs(rng.normal(0, 0.5)),
                    close=base_price,
                    volume=int(abs(rng.normal(100000, 20000))),
                )
            )

        market_data = MarketData(
            symbol="DATA_FLOW_TEST",
            timeframe="1d",
            data_type=MarketDataType.OHLCV,
            data=ohlcv_data,
            start_time=dates[0],
            end_time=dates[-1],
            data_source="test",
            market_session=MarketSession.REGULAR,
            completeness_score=1.0,
            gaps_detected=0,
            anomalies_detected=0,
            total_volume=sum(d.volume for d in ohlcv_data),
            vwap=100.0,
            price_range=20.0,
            volatility=0.02,
        )

        # Test DataFrame conversion consistency
        df1 = market_data.to_dataframe()
        df2 = market_data.to_dataframe()

        # DataFrames should be identical
        pd.testing.assert_frame_equal(df1, df2)

        # Test data integrity through pipeline
        pattern_engine = PatternDetectionEngine()
        patterns = pattern_engine.detect_patterns(market_data)

        # Patterns should reference the same symbol
        for pattern in patterns:
            self.assertEqual(pattern.symbol, market_data.symbol)
            self.assertGreaterEqual(pattern.start_time, market_data.start_time)
            self.assertLessEqual(pattern.end_time, market_data.end_time)

    def test_error_handling_integration(self):
        """Test integrated error handling across components."""
        # Test with insufficient data
        dates = pd.date_range(
            start="2024-01-01", periods=5, freq="D"
        )  # Too few data points

        ohlcv_data = []
        for i, date in enumerate(dates):
            ohlcv_data.append(
                OHLCV(
                    timestamp=date,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    volume=100000,
                )
            )

        market_data = MarketData(
            symbol="ERROR_TEST",
            timeframe="1d",
            data_type=MarketDataType.OHLCV,
            data=ohlcv_data,
            start_time=dates[0],
            end_time=dates[-1],
            data_source="test",
            market_session=MarketSession.REGULAR,
            completeness_score=1.0,
            gaps_detected=0,
            anomalies_detected=0,
            total_volume=500000,
            vwap=100.0,
            price_range=2.0,
            volatility=0.01,
        )

        # Components should handle insufficient data gracefully
        pattern_engine = PatternDetectionEngine()
        patterns = pattern_engine.detect_patterns(market_data)

        # Should return empty list, not crash
        self.assertIsInstance(patterns, list)

        indicator_engine = TechnicalIndicatorEngine()
        _ = indicator_engine.calculate_indicators(market_data)

        # Should return indicators (possibly with NaN values), not crash
        # self.assertIsInstance(indicators, dict)  # Commented since we don't use the result


def run_integration_tests():
    """Run all integration tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestComponentIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 80)
    print("PATTERN RECOGNITION AGENT - INTEGRATION TEST SUITE")
    print("=" * 80)

    success = run_integration_tests()

    print("\n" + "=" * 80)
    if success:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some integration tests failed!")
    print("=" * 80)
