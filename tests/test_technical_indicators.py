"""
Test suite for Technical Indicator Engine.

Tests all technical indicators for correctness and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from technical_indicators.indicator_engine import (
    TechnicalIndicatorEngine,
    IndicatorConfig,
)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Create synthetic OHLCV data with some realistic patterns
    rng = np.random.default_rng(42)  # For reproducible tests

    close_prices = 100 + np.cumsum(rng.normal(0, 0.5, 100))  # Random walk
    high_prices = close_prices + rng.uniform(0.5, 2.0, 100)
    low_prices = close_prices - rng.uniform(0.5, 2.0, 100)
    open_prices = close_prices + rng.uniform(-1.0, 1.0, 100)
    volume = rng.integers(1000000, 5000000, 100)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }
    )


@pytest.fixture
def indicator_engine():
    """Create a technical indicator engine with default config."""
    return TechnicalIndicatorEngine()


@pytest.fixture
def custom_config():
    """Create a custom indicator configuration for testing."""
    return IndicatorConfig(
        sma_periods=[5, 10, 20],
        ema_periods=[12, 26],
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        rsi_period=14,
        stoch_k_period=14,
        stoch_d_period=3,
        bb_period=20,
        bb_std_dev=2.0,
        atr_period=14,
        adx_period=14,
        williams_r_period=14,
        roc_period=12,
        cmf_period=21,
    )


class TestTechnicalIndicatorEngine:
    """Test suite for the Technical Indicator Engine."""

    def test_engine_initialization(self):
        """Test engine initializes with correct default configuration."""
        engine = TechnicalIndicatorEngine()

        assert engine.config.sma_periods == [10, 20, 50, 200]
        assert engine.config.ema_periods == [12, 26, 50]
        assert engine.config.rsi_period == 14
        assert engine.config.bb_period == 20
        assert engine.config.atr_period == 14

    def test_engine_custom_config(self, custom_config):
        """Test engine initializes with custom configuration."""
        engine = TechnicalIndicatorEngine(custom_config)

        assert engine.config.sma_periods == [5, 10, 20]
        assert engine.config.ema_periods == [12, 26]
        assert engine.config.rsi_period == 14

    def test_data_validation_sufficient_data(self, sample_market_data):
        """Test validation passes with sufficient data."""
        # Create engine with smaller periods that fit our 100-point dataset
        config = IndicatorConfig(
            sma_periods=[10, 20, 50],  # Remove 200-period SMA
            ema_periods=[12, 26],
            bb_period=20,
        )
        engine = TechnicalIndicatorEngine(config)

        result = engine._validate_data(sample_market_data)
        assert result == True

    def test_data_validation_insufficient_data(self, indicator_engine):
        """Test validation fails with insufficient data."""
        small_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10),
                "open": range(10),
                "high": range(1, 11),
                "low": range(10),
                "close": range(10),
                "volume": range(1000, 1010),
            }
        )

        result = indicator_engine._validate_data(small_data)
        assert result == False

    def test_calculate_all_indicators(self, indicator_engine, sample_market_data):
        """Test calculation of indicators with DataFrame directly."""
        # Use direct DataFrame calculation for testing
        # This tests the internal calculation methods

        result_trend = indicator_engine._calculate_trend_indicators(sample_market_data)
        result_momentum = indicator_engine._calculate_momentum_indicators(
            sample_market_data
        )
        result_volume = indicator_engine._calculate_volume_indicators(
            sample_market_data
        )
        result_volatility = indicator_engine._calculate_volatility_indicators(
            sample_market_data
        )

        # Check main categories exist and have content
        assert len(result_trend) > 0
        assert len(result_momentum) > 0
        assert len(result_volume) > 0
        assert len(result_volatility) > 0

        # Check specific indicators exist in trend
        assert any("sma" in key for key in result_trend.keys())
        assert any("ema" in key for key in result_trend.keys())

        # Check specific indicators exist in momentum
        assert "rsi" in result_momentum

        # Check specific indicators exist in volume
        assert "obv" in result_volume

        # Check specific indicators exist in volatility
        assert any("bollinger" in key for key in result_volatility.keys())

    def test_sma_calculation(self, indicator_engine, sample_market_data):
        """Test Simple Moving Average calculation."""
        close_prices = sample_market_data["close"]
        sma_20 = indicator_engine._calculate_sma(close_prices, 20)

        # Check length matches input
        assert len(sma_20) == len(close_prices)

        # Check first 19 values are NaN (not enough data)
        assert pd.isna(sma_20.iloc[:19]).all()

        # Check 20th value equals manual calculation
        expected_20th = close_prices.iloc[:20].mean()
        assert abs(sma_20.iloc[19] - expected_20th) < 1e-6

    def test_ema_calculation(self, indicator_engine, sample_market_data):
        """Test Exponential Moving Average calculation."""
        close_prices = sample_market_data["close"]
        ema_12 = indicator_engine._calculate_ema(close_prices, 12)

        # Check length matches input
        assert len(ema_12) == len(close_prices)

        # EMA should start from first value
        assert not pd.isna(ema_12.iloc[0])

        # EMA should be smoother than price (less volatile)
        price_std = close_prices.std()
        ema_std = ema_12.std()
        assert ema_std < price_std

    def test_rsi_calculation(self, indicator_engine, sample_market_data):
        """Test RSI calculation."""
        close_prices = sample_market_data["close"]
        rsi = indicator_engine._calculate_rsi(close_prices, 14)

        # RSI should be between 0 and 100
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

        # RSI should have reasonable values (not all extreme)
        assert rsi.mean() > 20
        assert rsi.mean() < 80

    def test_bollinger_bands_calculation(self, indicator_engine, sample_market_data):
        """Test Bollinger Bands calculation."""
        close_prices = sample_market_data["close"]
        bb_upper, bb_middle, bb_lower = indicator_engine._calculate_bollinger_bands(
            close_prices, 20, 2.0
        )

        # Upper band should be above middle, middle above lower (excluding NaN values)
        valid_indices = ~(bb_upper.isna() | bb_middle.isna() | bb_lower.isna())
        assert (bb_upper[valid_indices] >= bb_middle[valid_indices]).all()
        assert (bb_middle[valid_indices] >= bb_lower[valid_indices]).all()

        # Middle band should equal SMA
        sma_20 = indicator_engine._calculate_sma(close_prices, 20)
        pd.testing.assert_series_equal(bb_middle, sma_20, check_names=False)

    def test_macd_calculation(self, indicator_engine, sample_market_data):
        """Test MACD calculation."""
        close_prices = sample_market_data["close"]
        macd_line, signal_line, histogram = indicator_engine._calculate_macd(
            close_prices, 12, 26, 9
        )

        # Check all series have same length
        assert len(macd_line) == len(close_prices)
        assert len(signal_line) == len(close_prices)
        assert len(histogram) == len(close_prices)

        # Histogram should equal MACD - Signal
        calculated_histogram = macd_line - signal_line
        pd.testing.assert_series_equal(
            histogram, calculated_histogram, check_names=False
        )

    def test_atr_calculation(self, indicator_engine, sample_market_data):
        """Test Average True Range calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        atr = indicator_engine._calculate_atr(high, low, close, 14)

        # ATR should be positive
        assert (atr > 0).all()

        # ATR should be reasonable relative to price range
        price_range = (high - low).mean()
        assert atr.mean() <= price_range * 2  # Sanity check

    def test_obv_calculation(self, indicator_engine, sample_market_data):
        """Test On-Balance Volume calculation."""
        close_prices = sample_market_data["close"]
        volume = sample_market_data["volume"]

        obv = indicator_engine._calculate_obv(close_prices, volume)

        # OBV should be cumulative
        assert len(obv) == len(close_prices)

        # First OBV should equal first volume (assuming price increased)
        if close_prices.iloc[1] > close_prices.iloc[0]:
            assert obv.iloc[1] == volume.iloc[1]

    def test_stochastic_oscillator(self, indicator_engine, sample_market_data):
        """Test Stochastic Oscillator calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        stoch_k, stoch_d = indicator_engine._calculate_stochastic(
            high, low, close, 14, 3
        )

        # Both should be between 0 and 100
        assert (stoch_k >= 0).all()
        assert (stoch_k <= 100).all()
        assert (stoch_d >= 0).all()
        assert (stoch_d <= 100).all()

    def test_adx_calculation(self, indicator_engine, sample_market_data):
        """Test ADX calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        adx, plus_di, minus_di = indicator_engine._calculate_adx(high, low, close, 14)

        # ADX should be between 0 and 100 (excluding NaN values)
        valid_adx = adx.dropna()
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()

        # DI values should be positive (excluding NaN values)
        valid_plus_di = plus_di.dropna()
        valid_minus_di = minus_di.dropna()
        assert (valid_plus_di >= 0).all()
        assert (valid_minus_di >= 0).all()

    def test_williams_r_calculation(self, indicator_engine, sample_market_data):
        """Test Williams %R calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        williams_r = indicator_engine._calculate_williams_r(high, low, close, 14)

        # Williams %R should be between -100 and 0
        assert (williams_r >= -100).all()
        assert (williams_r <= 0).all()

    def test_roc_calculation(self, indicator_engine, sample_market_data):
        """Test Rate of Change calculation."""
        close_prices = sample_market_data["close"]
        roc = indicator_engine._calculate_roc(close_prices, 12)

        # ROC can be any value, but should be reasonable
        assert len(roc) == len(close_prices)
        assert not roc.isna().all()  # Not all NaN

    def test_cmf_calculation(self, indicator_engine, sample_market_data):
        """Test Chaikin Money Flow calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]
        volume = sample_market_data["volume"]

        cmf = indicator_engine._calculate_cmf(high, low, close, volume, 21)

        # CMF should be between -1 and 1
        assert (cmf >= -1).all()
        assert (cmf <= 1).all()

    def test_empty_data_handling(self, indicator_engine):
        """Test engine handles empty data gracefully."""
        empty_data = pd.DataFrame()

        # Should not be able to create MarketData with empty DataFrame
        # so test validate_data directly instead
        result = indicator_engine._validate_data(empty_data)
        assert result == False

    def test_minimal_data_handling(self, indicator_engine):
        """Test engine handles minimal data gracefully."""
        minimal_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5),
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        result = indicator_engine._validate_data(minimal_data)

        # Should return False due to insufficient data
        assert result == False

    def test_config_with_none_periods(self):
        """Test configuration with None periods."""
        # Create config that should not trigger __post_init__ defaults
        config = IndicatorConfig()
        # Manually set to None after creation to test handling
        config.sma_periods = None
        config.ema_periods = None

        engine = TechnicalIndicatorEngine(config)

        # Should handle None gracefully
        assert engine.config.sma_periods is None
        assert engine.config.ema_periods is None


class TestIndicatorConfig:
    """Test suite for IndicatorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IndicatorConfig()

        assert config.sma_periods == [10, 20, 50, 200]  # Set in __post_init__
        assert config.ema_periods == [12, 26, 50]  # Set in __post_init__
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.macd_signal == 9
        assert config.rsi_period == 14
        assert config.bb_period == 20
        assert abs(config.bb_std_dev - 2.0) < 1e-6

    def test_custom_config(self):
        """Test custom configuration values."""
        config = IndicatorConfig(sma_periods=[5, 10], rsi_period=21, bb_std_dev=2.5)

        assert config.sma_periods == [5, 10]
        assert config.rsi_period == 21
        assert abs(config.bb_std_dev - 2.5) < 1e-6
        # Other values should remain default
        assert config.ema_periods == [12, 26, 50]

    def test_config_validation(self):
        """Test configuration validates reasonable values."""
        # This would be where we add validation logic if needed
        config = IndicatorConfig(rsi_period=14, bb_std_dev=2.0)

        assert config.rsi_period > 0
        assert config.bb_std_dev > 0


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
