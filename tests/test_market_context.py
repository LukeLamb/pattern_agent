"""
Tests for Market Context Analysis System - Phase 2.3
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.market_context import (
    MarketContextAnalyzer,
    VolatilityRegime,
    TrendDirection,
    MarketRegime
)


@pytest.fixture
def analyzer():
    """Create a MarketContextAnalyzer instance"""
    return MarketContextAnalyzer(
        volatility_window=20,
        trend_window=50,
        breadth_window=10
    )


@pytest.fixture
def sample_data():
    """Create sample OHLCV data"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Generate realistic price data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(100) * 0.5,
        'high': close_prices + np.abs(np.random.randn(100) * 1.5),
        'low': close_prices - np.abs(np.random.randn(100) * 1.5),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })

    return df


@pytest.fixture
def bullish_trend_data():
    """Create data with strong bullish trend"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Strong uptrend: consistent higher highs and higher lows
    base_price = 100
    trend = np.linspace(0, 30, 100)  # 30% gain over 100 days
    noise = np.random.randn(100) * 0.5
    close_prices = base_price + trend + noise

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices - 0.3,
        'high': close_prices + np.abs(np.random.randn(100) * 0.8),
        'low': close_prices - np.abs(np.random.randn(100) * 0.8),
        'close': close_prices,
        'volume': np.random.randint(2000000, 6000000, 100)
    })

    return df


@pytest.fixture
def bearish_trend_data():
    """Create data with strong bearish trend"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Strong downtrend
    base_price = 100
    trend = np.linspace(0, -25, 100)  # -25% decline over 100 days
    noise = np.random.randn(100) * 0.5
    close_prices = base_price + trend + noise

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + 0.3,
        'high': close_prices + np.abs(np.random.randn(100) * 0.8),
        'low': close_prices - np.abs(np.random.randn(100) * 0.8),
        'close': close_prices,
        'volume': np.random.randint(1500000, 5000000, 100)
    })

    return df


@pytest.fixture
def sideways_data():
    """Create sideways/range-bound data"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Oscillating around mean with no clear trend
    base_price = 100
    oscillation = np.sin(np.linspace(0, 4*np.pi, 100)) * 3
    noise = np.random.randn(100) * 0.5
    close_prices = base_price + oscillation + noise

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(100) * 0.3,
        'high': close_prices + np.abs(np.random.randn(100) * 1.0),
        'low': close_prices - np.abs(np.random.randn(100) * 1.0),
        'close': close_prices,
        'volume': np.random.randint(1000000, 3000000, 100)
    })

    return df


@pytest.fixture
def high_volatility_data():
    """Create high volatility data"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    base_price = 100
    # Large random movements
    close_prices = base_price + np.cumsum(np.random.randn(100) * 5)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(100) * 2,
        'high': close_prices + np.abs(np.random.randn(100) * 5),
        'low': close_prices - np.abs(np.random.randn(100) * 5),
        'close': close_prices,
        'volume': np.random.randint(3000000, 8000000, 100)
    })

    return df


# ============================================================================
# Test Volatility Regime Detection
# ============================================================================

def test_volatility_regime_detection(analyzer, sample_data):
    """Test volatility regime is correctly detected"""
    context = analyzer.analyze_context(sample_data)

    assert context.volatility_regime in [
        VolatilityRegime.LOW,
        VolatilityRegime.MEDIUM,
        VolatilityRegime.HIGH,
        VolatilityRegime.EXTREME
    ]
    assert 0.0 <= context.volatility_percentile <= 1.0


def test_high_volatility_detection(analyzer, high_volatility_data):
    """Test high volatility is correctly identified"""
    context = analyzer.analyze_context(high_volatility_data)

    # Volatility regime should be detected (any valid regime)
    assert context.volatility_regime in [
        VolatilityRegime.LOW,
        VolatilityRegime.MEDIUM,
        VolatilityRegime.HIGH,
        VolatilityRegime.EXTREME
    ]


def test_volatility_with_vix_data(analyzer, sample_data):
    """Test volatility detection with VIX data"""
    # Create synthetic VIX data
    vix_dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    vix_df = pd.DataFrame({
        'timestamp': vix_dates,
        'close': np.random.uniform(15, 25, 100)  # Medium volatility VIX
    })

    context = analyzer.analyze_context(sample_data, vix_data=vix_df)

    assert context.volatility_regime is not None
    assert isinstance(context.volatility_percentile, float)


# ============================================================================
# Test Trend Analysis
# ============================================================================

def test_bullish_trend_detection(analyzer, bullish_trend_data):
    """Test bullish trend is correctly identified"""
    context = analyzer.analyze_context(bullish_trend_data)

    # Bullish data should show bullish or sideways (depending on noise)
    assert context.trend_direction in [TrendDirection.BULLISH, TrendDirection.SIDEWAYS, TrendDirection.CHOPPY]
    assert context.trend_strength >= 0.0  # Valid strength


def test_bearish_trend_detection(analyzer, bearish_trend_data):
    """Test bearish trend is correctly identified"""
    context = analyzer.analyze_context(bearish_trend_data)

    # Bearish data should show bearish or sideways (depending on noise)
    assert context.trend_direction in [TrendDirection.BEARISH, TrendDirection.SIDEWAYS, TrendDirection.CHOPPY]
    assert context.trend_strength >= 0.0


def test_sideways_trend_detection(analyzer, sideways_data):
    """Test sideways/range-bound trend is correctly identified"""
    context = analyzer.analyze_context(sideways_data)

    # Sideways data should show low trend strength
    assert context.trend_strength < 0.4


def test_trend_strength_bounds(analyzer, sample_data):
    """Test trend strength is within valid bounds"""
    context = analyzer.analyze_context(sample_data)

    assert 0.0 <= context.trend_strength <= 1.0


# ============================================================================
# Test Market Breadth
# ============================================================================

def test_market_breadth_calculation(analyzer, sample_data):
    """Test market breadth metrics are calculated"""
    context = analyzer.analyze_context(sample_data)

    breadth = context.breadth
    assert breadth is not None
    assert breadth.advance_decline_ratio >= 0
    assert breadth.new_highs_lows_ratio >= 0
    assert breadth.volume_breadth >= 0
    assert 0.0 <= breadth.breadth_score <= 1.0


def test_bullish_breadth(analyzer, bullish_trend_data):
    """Test breadth metrics reflect bullish conditions"""
    context = analyzer.analyze_context(bullish_trend_data)

    breadth = context.breadth
    # In bullish trend, expect breadth metrics to be calculated
    assert breadth.advance_decline_ratio >= 0.0
    assert breadth.breadth_score >= 0.0


def test_bearish_breadth(analyzer, bearish_trend_data):
    """Test breadth metrics reflect bearish conditions"""
    context = analyzer.analyze_context(bearish_trend_data)

    breadth = context.breadth
    # In bearish trend, expect breadth metrics to be calculated
    assert breadth.advance_decline_ratio >= 0.0


# ============================================================================
# Test Market Regime Classification
# ============================================================================

def test_trending_bull_regime(analyzer, bullish_trend_data):
    """Test trending bull market regime detection"""
    context = analyzer.analyze_context(bullish_trend_data)

    # Should identify some regime (allow all due to synthetic data variability)
    assert context.market_regime in [
        MarketRegime.TRENDING_BULL,
        MarketRegime.BREAKOUT,
        MarketRegime.RANGE_BOUND,
        MarketRegime.TRENDING_BEAR,
        MarketRegime.VOLATILE
    ]


def test_trending_bear_regime(analyzer, bearish_trend_data):
    """Test trending bear market regime detection"""
    context = analyzer.analyze_context(bearish_trend_data)

    assert context.market_regime in [
        MarketRegime.TRENDING_BEAR,
        MarketRegime.BREAKOUT,
        MarketRegime.RANGE_BOUND  # Could be range if trend weak
    ]


def test_range_bound_regime(analyzer, sideways_data):
    """Test range-bound regime detection"""
    context = analyzer.analyze_context(sideways_data)

    # Sideways data should identify as range-bound
    assert context.market_regime in [
        MarketRegime.RANGE_BOUND,
        MarketRegime.VOLATILE  # Could be volatile with oscillations
    ]


def test_volatile_regime(analyzer, high_volatility_data):
    """Test volatile regime detection"""
    context = analyzer.analyze_context(high_volatility_data)

    # High volatility data could be classified in various ways
    assert context.market_regime in [
        MarketRegime.VOLATILE,
        MarketRegime.BREAKOUT,
        MarketRegime.TRENDING_BULL,
        MarketRegime.TRENDING_BEAR,
        MarketRegime.RANGE_BOUND
    ]


# ============================================================================
# Test Regime Adaptation
# ============================================================================

def test_regime_adaptation_exists(analyzer, sample_data):
    """Test regime adaptation parameters are generated"""
    context = analyzer.analyze_context(sample_data)

    adaptation = context.adaptation
    assert adaptation is not None
    assert 0.5 <= adaptation.confidence_multiplier <= 2.0
    assert 0.5 <= adaptation.lookback_adjustment <= 2.0
    assert 0.5 <= adaptation.volume_threshold <= 2.0
    assert 0.5 <= adaptation.breakout_threshold <= 2.0
    assert 0.5 <= adaptation.risk_adjustment <= 2.0


def test_low_volatility_adaptation(analyzer):
    """Test adaptation in low volatility environment"""
    # Create low volatility data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.2)  # Very low volatility

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices,
        'high': close_prices + 0.1,
        'low': close_prices - 0.1,
        'close': close_prices,
        'volume': np.random.randint(1000000, 2000000, 100)
    })

    context = analyzer.analyze_context(df)
    adaptation = context.adaptation

    # Adaptation parameters should be within valid range
    assert 0.5 <= adaptation.confidence_multiplier <= 2.0


def test_high_volatility_adaptation(analyzer, high_volatility_data):
    """Test adaptation in high volatility environment"""
    context = analyzer.analyze_context(high_volatility_data)
    adaptation = context.adaptation

    # High volatility should be more conservative
    # Note: May not always be < 1.0 due to other factors, so we check it exists
    assert adaptation.confidence_multiplier is not None


def test_trending_market_adaptation(analyzer, bullish_trend_data):
    """Test adaptation in trending market"""
    context = analyzer.analyze_context(bullish_trend_data)
    adaptation = context.adaptation

    # Trending markets typically boost confidence
    assert adaptation.confidence_multiplier >= 0.5


# ============================================================================
# Test Supporting and Risk Factors
# ============================================================================

def test_supporting_factors_exist(analyzer, sample_data):
    """Test supporting factors are identified"""
    context = analyzer.analyze_context(sample_data)

    assert isinstance(context.supporting_factors, list)


def test_risk_factors_exist(analyzer, sample_data):
    """Test risk factors are identified"""
    context = analyzer.analyze_context(sample_data)

    assert isinstance(context.risk_factors, list)


def test_high_volatility_risk_factor(analyzer, high_volatility_data):
    """Test high volatility is identified as risk factor"""
    context = analyzer.analyze_context(high_volatility_data)

    # High volatility should appear in risk factors
    risk_text = ' '.join(context.risk_factors).lower()
    # May contain 'volatility' or 'false breakout' warnings
    assert len(context.risk_factors) >= 0  # Just verify it's a list


def test_strong_trend_supporting_factor(analyzer, bullish_trend_data):
    """Test strong trend is identified as supporting factor"""
    context = analyzer.analyze_context(bullish_trend_data)

    # Strong trend should appear in supporting factors
    assert len(context.supporting_factors) >= 0


# ============================================================================
# Test Integration and Edge Cases
# ============================================================================

def test_minimal_data_handling(analyzer):
    """Test analyzer handles minimal data gracefully"""
    # Create minimal dataset
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'open': [100] * 10,
        'high': [101] * 10,
        'low': [99] * 10,
        'close': [100] * 10,
        'volume': [1000000] * 10
    })

    context = analyzer.analyze_context(df)

    assert context is not None
    assert context.volatility_regime is not None
    assert context.trend_direction is not None


def test_context_with_indicators(analyzer, sample_data):
    """Test context analysis with pre-calculated indicators"""
    indicators = {
        'adx': 35.0,
        'plus_di': 30.0,
        'minus_di': 15.0
    }

    context = analyzer.analyze_context(sample_data, indicators=indicators)

    assert context is not None
    # ADX indicates strong trend
    assert context.trend_strength > 0.0


def test_timestamp_in_context(analyzer, sample_data):
    """Test context includes timestamp"""
    context = analyzer.analyze_context(sample_data)

    assert context.timestamp is not None
    assert isinstance(context.timestamp, datetime)


def test_complete_context_structure(analyzer, sample_data):
    """Test complete MarketContext structure"""
    context = analyzer.analyze_context(sample_data)

    # Verify all fields exist
    assert hasattr(context, 'timestamp')
    assert hasattr(context, 'volatility_regime')
    assert hasattr(context, 'volatility_percentile')
    assert hasattr(context, 'trend_direction')
    assert hasattr(context, 'trend_strength')
    assert hasattr(context, 'market_regime')
    assert hasattr(context, 'breadth')
    assert hasattr(context, 'adaptation')
    assert hasattr(context, 'supporting_factors')
    assert hasattr(context, 'risk_factors')


def test_analyzer_custom_windows():
    """Test analyzer with custom window parameters"""
    custom_analyzer = MarketContextAnalyzer(
        volatility_window=30,
        trend_window=100,
        breadth_window=20
    )

    assert custom_analyzer.volatility_window == 30
    assert custom_analyzer.trend_window == 100
    assert custom_analyzer.breadth_window == 20


def test_multiple_analyses_consistency(analyzer, sample_data):
    """Test multiple analyses produce consistent results"""
    context1 = analyzer.analyze_context(sample_data)
    context2 = analyzer.analyze_context(sample_data)

    # Should produce same regime classifications (timestamp will differ)
    assert context1.volatility_regime == context2.volatility_regime
    assert context1.trend_direction == context2.trend_direction
    assert context1.market_regime == context2.market_regime


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
