"""
Comprehensive tests for working data models.
"""

import pytest
from datetime import datetime, timedelta
from typing import List
import pandas as pd

# Import working models
from src.models.market_data import MarketData, OHLCV, MarketDataType, MarketSession
from src.models.pattern import Pattern, PatternType


class TestIntegratedModels:
    """Test integration between working models."""
    
    def create_sample_market_data(self) -> MarketData:
        """Create sample market data for testing."""
        # Create 10 OHLCV data points
        base_time = datetime.now() - timedelta(days=10)
        data = []
        
        for i in range(10):
            base_price = 100.0 + i + (i % 3)  # Add some variation
            ohlcv = OHLCV(
                timestamp=base_time + timedelta(days=i),
                open=base_price,
                high=base_price + 3,
                low=base_price - 2,
                close=base_price + 1,
                volume=1000 * (i + 1)
            )
            data.append(ohlcv)
        
        return MarketData(
            symbol="TEST",
            timeframe="daily",
            data=data,
            start_time=data[0].timestamp,
            end_time=data[-1].timestamp
        )
    
    def test_market_data_comprehensive(self):
        """Test comprehensive MarketData functionality."""
        market_data = self.create_sample_market_data()
        
        # Basic properties
        assert market_data.symbol == "TEST"
        assert market_data.timeframe == "daily"
        assert len(market_data) == 10
        assert market_data.data_type == MarketDataType.OHLCV
        
        # Price series
        closes = market_data.get_price_series('close')
        assert len(closes) == 10
        assert all(isinstance(price, float) for price in closes)
        
        # Volume series
        volumes = market_data.get_volume_series()
        assert len(volumes) == 10
        assert all(isinstance(vol, int) for vol in volumes)
        
        # VWAP calculation
        vwap = market_data.calculate_vwap()
        assert isinstance(vwap, float)
        assert vwap > 0
        
        # Summary statistics
        stats = market_data.get_summary_stats()
        assert stats['symbol'] == "TEST"
        assert stats['periods'] == 10
        assert 'total_volume' in stats
        assert 'vwap' in stats
        
        # DataFrame conversion
        df = market_data.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_pattern_creation_basic(self):
        """Test basic Pattern creation and functionality."""
        pattern = Pattern(
            pattern_type=PatternType.ASCENDING_TRIANGLE,
            symbol="TEST",
            timeframe="daily",
            start_time=datetime.now() - timedelta(days=5),
            end_time=datetime.now(),
            confidence_score=0.75
        )
        
        # Basic properties
        assert pattern.pattern_type == PatternType.ASCENDING_TRIANGLE
        assert pattern.symbol == "TEST"
        assert pattern.timeframe == "daily"
        assert abs(pattern.confidence_score - 0.75) < 0.01
        
        # Check default values
        assert pattern.support_levels == []
        assert pattern.resistance_levels == []
        assert pattern.target_prices == []
        
        # String representations
        pattern_str = str(pattern)
        assert "Pattern" in pattern_str
        assert "ascending_triangle" in pattern_str
        
        pattern_repr = repr(pattern)
        assert "Pattern" in pattern_repr
    
    def test_pattern_validation(self):
        """Test pattern validation logic."""
        # Test with valid pattern
        pattern = Pattern(
            pattern_type=PatternType.DOUBLE_BOTTOM,
            symbol="VALID",
            timeframe="1hr", 
            start_time=datetime.now() - timedelta(hours=24),
            end_time=datetime.now() - timedelta(hours=1),  # End before now
            confidence_score=0.80
        )
        
        # Should not raise any validation errors
        assert pattern.confidence_score == 0.80
        assert pattern.pattern_type == PatternType.DOUBLE_BOTTOM
    
    def test_model_serialization(self):
        """Test model serialization capabilities."""
        market_data = self.create_sample_market_data()
        
        # Test dict conversion
        data_dict = market_data.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict['symbol'] == "TEST"
        assert data_dict['timeframe'] == "daily"
        assert len(data_dict['data']) == 10
        
        # Test JSON serialization
        json_str = market_data.model_dump_json()
        assert isinstance(json_str, str)
        assert "TEST" in json_str
        
        # Test reconstruction from dict
        reconstructed = MarketData.model_validate(data_dict)
        assert reconstructed.symbol == market_data.symbol
        assert len(reconstructed.data) == len(market_data.data)
    
    def test_data_quality_validation(self):
        """Test data quality validation features."""
        market_data = self.create_sample_market_data()
        
        # Test quality validation
        quality_metrics = market_data.validate_data_quality()
        assert isinstance(quality_metrics, dict)
        assert 'total_periods' in quality_metrics
        assert 'completeness_score' in quality_metrics
        assert 'quality_score' in quality_metrics
        
        # Test gap detection
        gaps = market_data.detect_gaps(gap_threshold_percent=1.0)
        assert isinstance(gaps, list)
        # With our test data, there shouldn't be significant gaps
        
        # Test price range
        price_range = market_data.get_price_range()
        assert isinstance(price_range, float)
        assert price_range > 0
    
    def test_time_based_operations(self):
        """Test time-based operations on market data."""
        market_data = self.create_sample_market_data()
        
        # Test latest price
        latest_close = market_data.get_latest_price('close')
        assert isinstance(latest_close, float)
        
        # Test returns calculation
        returns = market_data.calculate_returns()
        assert isinstance(returns, list)
        assert len(returns) == 9  # One less than data points
        
        # Test typical price
        typical_prices = market_data.get_typical_price_series()
        assert len(typical_prices) == 10
        assert all(isinstance(price, float) for price in typical_prices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])