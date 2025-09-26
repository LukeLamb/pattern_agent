"""
Basic tests for core data models - simplified version.
"""

import pytest
from datetime import datetime, timedelta
from typing import List
import pandas as pd

# Import the basic model classes we need directly
from src.models.market_data import MarketData, OHLCV, MarketDataType, MarketSession


class TestBasicOHLCV:
    """Test basic OHLCV functionality."""
    
    def test_ohlcv_creation(self):
        """Test basic OHLCV creation."""
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=98.0,
            close=102.0,
            volume=10000
        )
        assert abs(ohlcv.open - 100.0) < 0.01
        assert abs(ohlcv.high - 105.0) < 0.01
        assert abs(ohlcv.low - 98.0) < 0.01
        assert abs(ohlcv.close - 102.0) < 0.01
        assert ohlcv.volume == 10000


class TestBasicMarketData:
    """Test basic MarketData functionality."""
    
    def create_sample_data(self, count: int = 3) -> List[OHLCV]:
        """Create simple sample OHLCV data."""
        base_time = datetime.now()
        data = []
        
        for i in range(count):
            base_price = 100.0 + i
            ohlcv = OHLCV(
                timestamp=base_time + timedelta(hours=i),
                open=base_price,
                high=base_price + 2,
                low=base_price - 1,
                close=base_price + 1,
                volume=1000 * (i + 1)
            )
            data.append(ohlcv)
        
        return data
    
    def test_market_data_creation(self):
        """Test basic MarketData creation."""
        data = self.create_sample_data()
        market_data = MarketData(
            symbol="AAPL",
            timeframe="1hr",
            data=data,
            start_time=data[0].timestamp,
            end_time=data[-1].timestamp
        )
        
        assert market_data.symbol == "AAPL"
        assert market_data.timeframe == "1hr"
        assert len(market_data.data) == 3
        assert market_data.data_type == MarketDataType.OHLCV
    
    def test_to_dataframe_conversion(self):
        """Test conversion to pandas DataFrame."""
        data = self.create_sample_data()
        market_data = MarketData(
            symbol="AAPL",
            timeframe="1hr",
            data=data,
            start_time=data[0].timestamp,
            end_time=data[-1].timestamp
        )
        
        df = market_data.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        assert all(col in df.columns for col in required_cols)
    
    def test_price_series_extraction(self):
        """Test price series extraction."""
        data = self.create_sample_data()
        market_data = MarketData(
            symbol="AAPL",
            timeframe="1hr",
            data=data,
            start_time=data[0].timestamp,
            end_time=data[-1].timestamp
        )
        
        closes = market_data.get_price_series('close')
        assert len(closes) == 3
        assert abs(closes[0] - 101.0) < 0.01  # First close price
        assert abs(closes[-1] - 103.0) < 0.01  # Last close price
    
    def test_vwap_calculation(self):
        """Test basic VWAP calculation."""
        data = self.create_sample_data()
        market_data = MarketData(
            symbol="AAPL",
            timeframe="1hr",
            data=data,
            start_time=data[0].timestamp,
            end_time=data[-1].timestamp
        )
        
        vwap = market_data.calculate_vwap()
        assert isinstance(vwap, float)
        assert vwap > 0
        # Should be somewhere around the average price
        assert 100.0 < vwap < 105.0
    
    def test_summary_statistics(self):
        """Test summary statistics generation."""
        data = self.create_sample_data()
        market_data = MarketData(
            symbol="AAPL",
            timeframe="1hr",
            data=data,
            start_time=data[0].timestamp,
            end_time=data[-1].timestamp
        )
        
        stats = market_data.get_summary_stats()
        assert stats['symbol'] == "AAPL"
        assert stats['periods'] == 3
        assert abs(stats['first_price'] - 101.0) < 0.01
        assert abs(stats['last_price'] - 103.0) < 0.01
        assert stats['total_volume'] == 6000  # 1000 + 2000 + 3000
    
    def test_dataframe_roundtrip(self):
        """Test DataFrame creation and conversion back."""
        # Create sample DataFrame
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(3)]
        df_data = {
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000, 1500, 2000]
        }
        df = pd.DataFrame(df_data, index=timestamps)
        
        # Create MarketData from DataFrame
        market_data = MarketData.from_dataframe(df, "TEST", "1hr")
        
        assert market_data.symbol == "TEST"
        assert market_data.timeframe == "1hr"
        assert len(market_data.data) == 3
        
        # Convert back to DataFrame
        df_out = market_data.to_dataframe()
        assert len(df_out) == 3
        assert all(col in df_out.columns for col in ['open', 'high', 'low', 'close', 'volume'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])