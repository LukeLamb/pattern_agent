"""
Tests for core data models.
"""

import pytest
from datetime import datetime, timedelta
from typing import List
import pandas as pd

# Import all model classes
from src.models import (
    Pattern, PatternType, PatternMetrics, ValidationCriteria, SupportResistanceLevel,
    TradingSignal, SignalType, SignalStatus, SignalPriority, PositionSizing, 
    RiskManagement, TargetPrices, ExecutionParameters, SignalMetrics,
    MarketData, OHLCV, MarketDataType, MarketSession
)


class TestOHLCV:
    """Test OHLCV data model."""
    
    def test_valid_ohlcv(self):
        """Test valid OHLCV creation."""
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=98.0,
            close=102.0,
            volume=10000
        )
        assert ohlcv.open == 100.0
        assert ohlcv.high == 105.0
        assert ohlcv.low == 98.0
        assert ohlcv.close == 102.0
        assert ohlcv.volume == 10000
    
    def test_invalid_high_validation(self):
        """Test high price validation."""
        with pytest.raises(ValueError, match="High must be >= close"):
            OHLCV(
                timestamp=datetime.now(),
                open=100.0,
                high=99.0,  # High less than close
                low=98.0,
                close=102.0,
                volume=10000
            )
    
    def test_invalid_low_validation(self):
        """Test low price validation."""
        with pytest.raises(ValueError, match="Low must be <= close"):
            OHLCV(
                timestamp=datetime.now(),
                open=100.0,
                high=105.0,
                low=103.0,  # Low greater than close
                close=102.0,
                volume=10000
            )


class TestMarketData:
    """Test MarketData model."""
    
    def create_sample_ohlcv_data(self, count: int = 5) -> List[OHLCV]:
        """Create sample OHLCV data."""
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
        """Test MarketData creation."""
        data = self.create_sample_ohlcv_data()
        market_data = MarketData(
            symbol="AAPL",
            timeframe="1hr",
            data=data,
            start_time=data[0].timestamp,
            end_time=data[-1].timestamp
        )
        
        assert market_data.symbol == "AAPL"
        assert market_data.timeframe == "1hr"
        assert len(market_data.data) == 5
        assert market_data.data_type == MarketDataType.OHLCV
    
    def test_to_dataframe(self):
        """Test conversion to pandas DataFrame."""
        data = self.create_sample_ohlcv_data()
        market_data = MarketData(
            symbol="AAPL",
            timeframe="1hr",
            data=data,
            start_time=data[0].timestamp,
            end_time=data[-1].timestamp
        )
        
        df = market_data.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    def test_price_series(self):
        """Test price series extraction."""
        data = self.create_sample_ohlcv_data()
        market_data = MarketData(
            symbol="AAPL",
            timeframe="1hr",
            data=data,
            start_time=data[0].timestamp,
            end_time=data[-1].timestamp
        )
        
        closes = market_data.get_price_series('close')
        assert len(closes) == 5
        assert closes[0] == 101.0  # First close price
        assert closes[-1] == 105.0  # Last close price
    
    def test_calculate_vwap(self):
        """Test VWAP calculation."""
        data = self.create_sample_ohlcv_data()
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
    
    def test_summary_stats(self):
        """Test summary statistics."""
        data = self.create_sample_ohlcv_data()
        market_data = MarketData(
            symbol="AAPL",
            timeframe="1hr",
            data=data,
            start_time=data[0].timestamp,
            end_time=data[-1].timestamp
        )
        
        stats = market_data.get_summary_stats()
        assert stats['symbol'] == "AAPL"
        assert stats['periods'] == 5
        assert stats['first_price'] == 101.0
        assert stats['last_price'] == 105.0


class TestPattern:
    """Test Pattern model."""
    
    def test_pattern_creation(self):
        """Test Pattern creation."""
        support_levels = [
            SupportResistanceLevel(price=100.0, touches=3, strength=0.8),
            SupportResistanceLevel(price=105.0, touches=2, strength=0.6)
        ]
        
        metrics = PatternMetrics(
            height=5.0,
            width_periods=10,
            volume_confirmation=True,
            breakout_volume_ratio=1.5,
            reliability_score=0.85
        )
        
        validation = ValidationCriteria(
            min_touches=2,
            min_periods=5,
            max_periods=50,
            volume_confirmation=True,
            price_action_confirmation=True
        )
        
        pattern = Pattern(
            pattern_type=PatternType.DOUBLE_BOTTOM,
            symbol="AAPL",
            timeframe="daily",
            start_time=datetime.now() - timedelta(days=10),
            end_time=datetime.now(),
            confidence_score=0.85,
            support_levels=support_levels,
            resistance_levels=[],
            pattern_metrics=metrics,
            validation_criteria=validation
        )
        
        assert pattern.pattern_type == PatternType.DOUBLE_BOTTOM
        assert pattern.symbol == "AAPL"
        assert pattern.confidence_score == 0.85
        assert len(pattern.support_levels) == 2
        assert pattern.is_valid()
    
    def test_pattern_price_targets(self):
        """Test pattern price target calculation."""
        support_levels = [
            SupportResistanceLevel(price=100.0, touches=3, strength=0.8)
        ]
        
        resistance_levels = [
            SupportResistanceLevel(price=110.0, touches=2, strength=0.7)
        ]
        
        pattern = Pattern(
            pattern_type=PatternType.RECTANGLE,
            symbol="AAPL",
            timeframe="daily",
            start_time=datetime.now() - timedelta(days=10),
            end_time=datetime.now(),
            confidence_score=0.75,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            pattern_metrics=PatternMetrics(
                height=10.0,
                width_periods=10,
                volume_confirmation=True,
                breakout_volume_ratio=1.2,
                reliability_score=0.75
            ),
            validation_criteria=ValidationCriteria(
                min_touches=2,
                min_periods=5,
                max_periods=50,
                volume_confirmation=True,
                price_action_confirmation=True
            )
        )
        
        targets = pattern.calculate_price_targets(current_price=105.0)
        assert 'upside_target' in targets
        assert 'downside_target' in targets
        assert targets['upside_target'] > 105.0
        assert targets['downside_target'] < 105.0


class TestTradingSignal:
    """Test TradingSignal model."""
    
    def test_signal_creation(self):
        """Test TradingSignal creation."""
        pattern = Pattern(
            pattern_type=PatternType.BREAKOUT,
            symbol="AAPL",
            timeframe="daily",
            start_time=datetime.now() - timedelta(days=5),
            end_time=datetime.now(),
            confidence_score=0.80,
            support_levels=[],
            resistance_levels=[],
            pattern_metrics=PatternMetrics(
                height=5.0,
                width_periods=5,
                volume_confirmation=True,
                breakout_volume_ratio=1.3,
                reliability_score=0.80
            ),
            validation_criteria=ValidationCriteria(
                min_touches=1,
                min_periods=3,
                max_periods=20,
                volume_confirmation=True,
                price_action_confirmation=True
            )
        )
        
        position_sizing = PositionSizing(
            position_size_usd=10000,
            position_size_percent=2.0,
            max_position_size=50000,
            risk_per_trade_percent=1.0
        )
        
        risk_mgmt = RiskManagement(
            stop_loss_price=95.0,
            stop_loss_percent=5.0,
            risk_reward_ratio=3.0,
            max_risk_usd=500
        )
        
        target_prices = TargetPrices(
            primary_target=110.0,
            secondary_target=115.0,
            stop_loss=95.0
        )
        
        execution = ExecutionParameters(
            entry_price=100.0,
            entry_type="market",
            time_in_force="GTC",
            execution_window_minutes=30
        )
        
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            symbol="AAPL",
            timeframe="daily",
            signal_strength=0.85,
            pattern_source=pattern,
            position_sizing=position_sizing,
            risk_management=risk_mgmt,
            target_prices=target_prices,
            execution_parameters=execution
        )
        
        assert signal.signal_type == SignalType.BUY
        assert signal.symbol == "AAPL"
        assert signal.signal_strength == 0.85
        assert signal.status == SignalStatus.PENDING
        assert signal.is_valid()
    
    def test_signal_risk_reward(self):
        """Test risk-reward calculation."""
        pattern = Pattern(
            pattern_type=PatternType.SUPPORT_RESISTANCE,
            symbol="AAPL",
            timeframe="1hr",
            start_time=datetime.now() - timedelta(hours=5),
            end_time=datetime.now(),
            confidence_score=0.70,
            support_levels=[],
            resistance_levels=[],
            pattern_metrics=PatternMetrics(
                height=3.0,
                width_periods=10,
                volume_confirmation=False,
                breakout_volume_ratio=1.0,
                reliability_score=0.70
            ),
            validation_criteria=ValidationCriteria(
                min_touches=2,
                min_periods=5,
                max_periods=30,
                volume_confirmation=False,
                price_action_confirmation=True
            )
        )
        
        signal = TradingSignal(
            signal_type=SignalType.SELL,
            symbol="AAPL",
            timeframe="1hr",
            signal_strength=0.75,
            pattern_source=pattern,
            position_sizing=PositionSizing(
                position_size_usd=5000,
                position_size_percent=1.0,
                max_position_size=25000,
                risk_per_trade_percent=0.5
            ),
            risk_management=RiskManagement(
                stop_loss_price=105.0,
                stop_loss_percent=5.0,
                risk_reward_ratio=2.0,
                max_risk_usd=250
            ),
            target_prices=TargetPrices(
                primary_target=90.0,
                secondary_target=85.0,
                stop_loss=105.0
            ),
            execution_parameters=ExecutionParameters(
                entry_price=100.0,
                entry_type="limit",
                time_in_force="DAY",
                execution_window_minutes=60
            )
        )
        
        rr_ratio = signal.calculate_risk_reward_ratio()
        assert isinstance(rr_ratio, float)
        assert rr_ratio > 0


if __name__ == "__main__":
    pytest.main([__file__])