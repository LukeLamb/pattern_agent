"""
Test suite for Pattern Validation Engine.

This module provides comprehensive tests for the PatternValidator including:
- Volume confirmation validation
- Timeframe consistency checks
- Historical success rate evaluation
- Pattern quality metrics calculation
- Market context validation
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

try:
    from .pattern_validator import (
        PatternValidator, ValidationCriteria, ValidationResult, 
        PatternQualityMetrics
    )
    from ..pattern_detection.pattern_engine import (
        PatternType, PatternStrength
    )
    from ..models.market_data import MarketData, OHLCV, MarketDataType, MarketSession
except ImportError:
    # For testing and development
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from validation.pattern_validator import (
        PatternValidator, ValidationCriteria, ValidationResult, 
        PatternQualityMetrics
    )
    from pattern_detection.pattern_engine import (
        PatternType, PatternStrength
    )
    from models.market_data import MarketData, OHLCV, MarketDataType, MarketSession


class MockDetectedPattern:
    """Mock DetectedPattern for testing - matches actual structure."""
    
    def __init__(self):
        self.pattern_type = PatternType.ASCENDING_TRIANGLE
        self.symbol = "AAPL"
        self.timeframe = "1d"
        self.start_time = datetime(2024, 1, 15)
        self.end_time = datetime(2024, 2, 15)
        self.confidence_score = 0.75
        self.strength = PatternStrength.STRONG
        self.key_points = [(datetime(2024, 1, 15), 102.0), (datetime(2024, 2, 15), 108.0)]
        self.pattern_metrics = {"height": 6.0, "width": 31}
        self.target_price: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.volume_confirmation = False


class TestPatternValidator(unittest.TestCase):
    """Test suite for PatternValidator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = PatternValidator(
            min_formation_days=5,
            max_formation_days=60,
            min_volume_confirmation_ratio=1.2,
            historical_lookback_days=365
        )
        
        # Create sample market data with proper structure
        rng = np.random.default_rng(42)  # For reproducible tests
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
        base_prices = 100 + np.cumsum(rng.normal(0, 0.02, len(dates)))
        
        # Create OHLCV data points
        ohlcv_data = []
        for i, date in enumerate(dates):
            base_price = base_prices[i]
            high = base_price + abs(rng.normal(0, 0.5))
            low = base_price - abs(rng.normal(0, 0.5))
            open_price = base_price + rng.normal(0, 0.1)
            close_price = base_price
            volume = int(rng.lognormal(10, 0.5))
            
            ohlcv_data.append(OHLCV(
                timestamp=date,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume
            ))
        
        self.market_data = MarketData(
            symbol="AAPL",
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
            price_range=10.0,
            volatility=0.02
        )
        
        # Create sample pattern
        self.sample_pattern = MockDetectedPattern()
    
    def test_validator_initialization(self):
        """Test PatternValidator initialization."""
        self.assertEqual(self.validator.min_formation_days, 5)
        self.assertEqual(self.validator.max_formation_days, 60)
        self.assertEqual(self.validator.min_volume_confirmation_ratio, 1.2)
        self.assertEqual(self.validator.historical_lookback_days, 365)
        
        # Test validation weights sum to 1.0
        total_weight = sum(self.validator.validation_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_volume_confirmation_validation(self):
        """Test volume confirmation validation."""
        result = self.validator._validate_volume_confirmation(
            self.sample_pattern, self.market_data
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.criteria, ValidationCriteria.VOLUME_CONFIRMATION)
        self.assertEqual(result.pattern_id, "AAPL")
        self.assertIsInstance(result.score, float)
        self.assertIsInstance(result.passed, bool)
        self.assertIn("volume_ratio", result.details)
        self.assertIn("threshold", result.details)
    
    def test_timeframe_consistency_validation(self):
        """Test timeframe consistency validation."""
        result = self.validator._validate_timeframe_consistency(
            self.sample_pattern, self.market_data
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.criteria, ValidationCriteria.TIMEFRAME_CONSISTENCY)
        self.assertEqual(result.pattern_id, "AAPL")
        self.assertTrue(result.passed)  # 31 days should be within 5-60 range
        self.assertIn("formation_days", result.details)
        self.assertEqual(result.details["formation_days"], 31)
    
    def test_historical_success_rate_validation(self):
        """Test historical success rate validation."""
        result = self.validator._validate_historical_success_rate(self.sample_pattern)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.criteria, ValidationCriteria.HISTORICAL_SUCCESS_RATE)
        self.assertEqual(result.pattern_id, "AAPL")
        self.assertIn("historical_count", result.details)
        self.assertIn("success_rate", result.details)
        # Should pass with moderate score due to insufficient history
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.6)
    
    def test_pattern_quality_validation(self):
        """Test pattern quality validation."""
        result = self.validator._validate_pattern_quality(
            self.sample_pattern, self.market_data
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.criteria, ValidationCriteria.PATTERN_QUALITY)
        self.assertEqual(result.pattern_id, "AAPL")
        self.assertIn("formation_time_score", result.details)
        self.assertIn("symmetry_score", result.details)
        self.assertIn("volume_pattern_score", result.details)
        self.assertIn("technical_strength_score", result.details)
        self.assertIn("market_context_score", result.details)
        self.assertIn("confidence_adjustment", result.details)
    
    def test_market_context_validation(self):
        """Test market context validation."""
        result = self.validator._validate_market_context(
            self.sample_pattern, self.market_data
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.criteria, ValidationCriteria.MARKET_CONTEXT)
        self.assertEqual(result.pattern_id, "AAPL")
        self.assertIn("volatility", result.details)
        self.assertIn("trend_strength", result.details)
        self.assertIn("volatility_score", result.details)
        self.assertIn("trend_score", result.details)
        self.assertIn("market_regime", result.details)
    
    def test_comprehensive_validation(self):
        """Test comprehensive pattern validation."""
        results = self.validator.validate_pattern(
            self.sample_pattern, self.market_data
        )
        
        self.assertEqual(len(results), 5)  # All validation criteria
        
        # Check all validation criteria are covered
        criteria_covered = {result.criteria for result in results}
        expected_criteria = {
            ValidationCriteria.VOLUME_CONFIRMATION,
            ValidationCriteria.TIMEFRAME_CONSISTENCY,
            ValidationCriteria.HISTORICAL_SUCCESS_RATE,
            ValidationCriteria.PATTERN_QUALITY,
            ValidationCriteria.MARKET_CONTEXT
        }
        self.assertEqual(criteria_covered, expected_criteria)
        
        # All results should be ValidationResult instances
        for result in results:
            self.assertIsInstance(result, ValidationResult)
            self.assertEqual(result.pattern_id, "AAPL")
            self.assertIsInstance(result.score, float)
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)
    
    def test_overall_validation_score_calculation(self):
        """Test overall validation score calculation."""
        results = self.validator.validate_pattern(
            self.sample_pattern, self.market_data
        )
        
        overall_score = self.validator.calculate_overall_validation_score(results)
        
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 1.0)
        
        # Calculate expected weighted score manually
        expected_score = 0.0
        for result in results:
            if result.criteria in self.validator.validation_weights:
                weight = self.validator.validation_weights[result.criteria]
                expected_score += result.score * weight
        
        self.assertAlmostEqual(overall_score, expected_score, places=5)
    
    def test_pattern_quality_metrics_calculation(self):
        """Test pattern quality metrics calculation."""
        metrics = self.validator.calculate_pattern_quality_metrics(
            self.sample_pattern, self.market_data
        )
        
        self.assertIsInstance(metrics, PatternQualityMetrics)
        
        # Check all metrics are present and within valid ranges
        self.assertGreaterEqual(metrics.formation_time_score, 0.0)
        self.assertLessEqual(metrics.formation_time_score, 1.0)
        
        self.assertGreaterEqual(metrics.symmetry_score, 0.0)
        self.assertLessEqual(metrics.symmetry_score, 1.0)
        
        self.assertGreaterEqual(metrics.volume_pattern_score, 0.0)
        self.assertLessEqual(metrics.volume_pattern_score, 1.0)
        
        self.assertGreaterEqual(metrics.technical_strength_score, 0.0)
        self.assertLessEqual(metrics.technical_strength_score, 1.0)
        
        self.assertGreaterEqual(metrics.market_context_score, 0.0)
        self.assertLessEqual(metrics.market_context_score, 1.0)
        
        self.assertGreaterEqual(metrics.overall_quality_score, 0.0)
        self.assertLessEqual(metrics.overall_quality_score, 1.0)
        
        self.assertGreaterEqual(metrics.confidence_adjustment, 0.5)
        self.assertLessEqual(metrics.confidence_adjustment, 1.5)
    
    def test_formation_time_scoring(self):
        """Test formation time scoring logic."""
        # Test optimal formation time (31 days - within optimal range)
        score = self.validator._calculate_formation_time_score(self.sample_pattern)
        self.assertEqual(score, 1.0)  # Should be optimal
        
        # Test too fast formation
        fast_pattern = MockDetectedPattern()
        fast_pattern.start_time = datetime(2024, 1, 15)
        fast_pattern.end_time = datetime(2024, 1, 18)  # Only 3 days
        
        fast_score = self.validator._calculate_formation_time_score(fast_pattern)
        self.assertEqual(fast_score, 0.4)  # Too fast
        
        # Test too slow formation
        slow_pattern = MockDetectedPattern()
        slow_pattern.start_time = datetime(2024, 1, 15)
        slow_pattern.end_time = datetime(2024, 4, 15)  # 90 days
        
        slow_score = self.validator._calculate_formation_time_score(slow_pattern)
        self.assertEqual(slow_score, 0.2)  # Too slow
    
    def test_rsi_calculation(self):
        """Test RSI calculation helper method."""
        # Create test price series
        prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0])
        
        rsi = self.validator._calculate_rsi(prices, period=5)
        
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(prices))
        
        # RSI values should be between 0 and 100 (excluding NaN)
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            self.assertTrue(all(0 <= val <= 100 for val in valid_rsi))


class TestValidationDataClasses(unittest.TestCase):
    """Test validation data classes."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            pattern_id="AAPL",
            criteria=ValidationCriteria.VOLUME_CONFIRMATION,
            score=0.8,
            passed=True,
            details={"test": "value"},
            validation_timestamp=datetime.now()
        )
        
        self.assertEqual(result.pattern_id, "AAPL")
        self.assertEqual(result.criteria, ValidationCriteria.VOLUME_CONFIRMATION)
        self.assertEqual(result.score, 0.8)
        self.assertTrue(result.passed)
        self.assertEqual(result.details["test"], "value")
    
    def test_pattern_quality_metrics_creation(self):
        """Test PatternQualityMetrics creation."""
        metrics = PatternQualityMetrics(
            formation_time_score=0.8,
            symmetry_score=0.9,
            volume_pattern_score=0.7,
            technical_strength_score=0.6,
            market_context_score=0.8,
            overall_quality_score=0.76,
            confidence_adjustment=1.1
        )
        
        self.assertEqual(metrics.formation_time_score, 0.8)
        self.assertEqual(metrics.symmetry_score, 0.9)
        self.assertEqual(metrics.volume_pattern_score, 0.7)
        self.assertEqual(metrics.technical_strength_score, 0.6)
        self.assertEqual(metrics.market_context_score, 0.8)
        self.assertEqual(metrics.overall_quality_score, 0.76)
        self.assertEqual(metrics.confidence_adjustment, 1.1)


def run_validation_tests():
    """Run all pattern validation tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPatternValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationDataClasses))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("Pattern Validation Engine Test Suite")
    print("=" * 60)
    
    success = run_validation_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All pattern validation tests passed!")
    else:
        print("❌ Some pattern validation tests failed!")
    print("=" * 60)