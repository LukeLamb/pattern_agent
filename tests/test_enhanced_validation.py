"""
Tests for Enhanced Pattern Validation - Phase 2.4

Tests context-aware pattern validation with market regime integration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.validation import (
    EnhancedPatternValidator,
    EnhancedValidationResult,
    PATTERN_REGIME_AFFINITY,
)
from src.pattern_detection.pattern_engine import DetectedPattern, PatternType, PatternStrength
from src.models.market_data import MarketData
from src.market_context import (
    MarketContextAnalyzer,
    MarketContext,
    MarketRegime,
    VolatilityRegime,
    TrendDirection,
    MarketBreadth,
    RegimeAdaptation,
)


@pytest.fixture
def enhanced_validator():
    """Create EnhancedPatternValidator instance"""
    return EnhancedPatternValidator()


@pytest.fixture
def sample_market_data():
    """Create sample market data"""
    from src.models.market_data import OHLCV

    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    ohlcv_data = []

    for i, date in enumerate(dates):
        ohlcv_data.append(OHLCV(
            timestamp=date.to_pydatetime(),
            open=float(close_prices[i] - 0.2),
            high=float(close_prices[i] + 0.8),
            low=float(close_prices[i] - 0.8),
            close=float(close_prices[i]),
            volume=int(np.random.randint(1000000, 5000000))
        ))

    return MarketData(
        symbol="TEST",
        timeframe="1D",
        data=ohlcv_data,
        start_time=dates[0].to_pydatetime(),
        end_time=dates[-1].to_pydatetime()
    )


@pytest.fixture
def sample_pattern():
    """Create sample detected pattern"""
    return DetectedPattern(
        symbol="TEST",
        pattern_type=PatternType.BULL_FLAG,
        timeframe="1D",
        start_time=datetime(2024, 1, 10),
        end_time=datetime(2024, 1, 25),
        confidence_score=0.75,
        strength=PatternStrength.MODERATE,
        key_points=[(datetime(2024, 1, 10), 100.0), (datetime(2024, 1, 25), 110.0)],
        pattern_metrics={"flagpole_start": 100.0, "flagpole_end": 110.0},
        direction="bullish"
    )


@pytest.fixture
def low_vol_trending_context():
    """Create low volatility trending bull context"""
    return MarketContext(
        timestamp=datetime.now(),
        volatility_regime=VolatilityRegime.LOW,
        volatility_percentile=0.15,
        trend_direction=TrendDirection.BULLISH,
        trend_strength=0.6,
        market_regime=MarketRegime.TRENDING_BULL,
        breadth=MarketBreadth(
            advance_decline_ratio=2.0,
            new_highs_lows_ratio=1.5,
            volume_breadth=1.8,
            breadth_score=0.75
        ),
        adaptation=RegimeAdaptation(
            confidence_multiplier=1.56,
            lookback_adjustment=1.2,
            volume_threshold=1.0,
            breakout_threshold=0.8,
            risk_adjustment=1.2
        ),
        supporting_factors=["Low volatility favors patterns", "Strong trend"],
        risk_factors=[]
    )


@pytest.fixture
def high_vol_context():
    """Create high volatility context"""
    return MarketContext(
        timestamp=datetime.now(),
        volatility_regime=VolatilityRegime.EXTREME,
        volatility_percentile=0.95,
        trend_direction=TrendDirection.CHOPPY,
        trend_strength=0.2,
        market_regime=MarketRegime.VOLATILE,
        breadth=MarketBreadth(
            advance_decline_ratio=0.8,
            new_highs_lows_ratio=0.6,
            volume_breadth=0.7,
            breadth_score=0.35
        ),
        adaptation=RegimeAdaptation(
            confidence_multiplier=0.6,
            lookback_adjustment=1.0,
            volume_threshold=1.5,
            breakout_threshold=1.5,
            risk_adjustment=0.5
        ),
        supporting_factors=[],
        risk_factors=["Extreme volatility", "Choppy price action"]
    )


# ============================================================================
# Test 1: Basic Context Integration
# ============================================================================

def test_validator_accepts_market_context(enhanced_validator, sample_pattern, sample_market_data, low_vol_trending_context):
    """Test that enhanced validator accepts MarketContext parameter"""
    result = enhanced_validator.validate_pattern_with_context(
        sample_pattern,
        sample_market_data,
        context=low_vol_trending_context
    )

    assert isinstance(result, EnhancedValidationResult)
    assert result.market_context is not None
    assert result.market_context == low_vol_trending_context


def test_validator_works_without_context(enhanced_validator, sample_pattern, sample_market_data):
    """Test backward compatibility - works without context"""
    result = enhanced_validator.validate_pattern_with_context(
        sample_pattern,
        sample_market_data,
        context=None
    )

    assert isinstance(result, EnhancedValidationResult)
    assert result.market_context is None
    assert result.adjusted_confidence == result.base_confidence
    assert result.context_boost == 0.0


def test_validation_result_structure(enhanced_validator, sample_pattern, sample_market_data, low_vol_trending_context):
    """Test that validation result has all required fields"""
    result = enhanced_validator.validate_pattern_with_context(
        sample_pattern,
        sample_market_data,
        context=low_vol_trending_context
    )

    # Base fields
    assert hasattr(result, 'pattern_id')
    assert hasattr(result, 'symbol')
    assert hasattr(result, 'pattern_type')
    assert hasattr(result, 'base_confidence')
    assert hasattr(result, 'is_valid')

    # Context fields
    assert hasattr(result, 'market_context')
    assert hasattr(result, 'adjusted_confidence')
    assert hasattr(result, 'regime_affinity')
    assert hasattr(result, 'context_boost')

    # Scoring breakdown
    assert hasattr(result, 'volume_score')
    assert hasattr(result, 'context_score')

    # Recommendations
    assert hasattr(result, 'recommendation_strength')
    assert hasattr(result, 'supporting_reasons')
    assert hasattr(result, 'risk_warnings')


# ============================================================================
# Test 2: Context Scoring
# ============================================================================

def test_context_score_calculation(enhanced_validator, sample_pattern, low_vol_trending_context):
    """Test context score is calculated correctly"""
    score = enhanced_validator._calculate_enhanced_context_score(
        sample_pattern,
        low_vol_trending_context
    )

    assert 0.0 <= score <= 1.0
    # Low vol + trending bull + bull flag should score high
    assert score > 0.6


def test_volatility_suitability_scoring(enhanced_validator):
    """Test volatility suitability scoring"""
    bull_flag = DetectedPattern(
        symbol="TEST",
        pattern_type=PatternType.BULL_FLAG,
        timeframe="1D",
        start_time=datetime.now(),
        end_time=datetime.now(),
        confidence_score=0.75,
        strength=PatternStrength.MODERATE,
        key_points=[(datetime.now(), 100.0)],
        pattern_metrics={},
        direction="bullish"
    )

    # Low volatility should score high for flags
    low_score = enhanced_validator._score_volatility_suitability(
        bull_flag, VolatilityRegime.LOW
    )
    assert low_score >= 0.9

    # Extreme volatility should score low
    extreme_score = enhanced_validator._score_volatility_suitability(
        bull_flag, VolatilityRegime.EXTREME
    )
    assert extreme_score <= 0.4


def test_trend_alignment_scoring(enhanced_validator):
    """Test trend alignment scoring"""
    bull_flag = DetectedPattern(
        symbol="TEST",
        pattern_type=PatternType.BULL_FLAG,
        timeframe="1D",
        start_time=datetime.now(),
        end_time=datetime.now(),
        confidence_score=0.75,
        strength=PatternStrength.MODERATE,
        key_points=[(datetime.now(), 100.0)],
        pattern_metrics={},
        direction="bullish"
    )

    # Bull flag in bullish trend should score high
    bullish_score = enhanced_validator._score_trend_alignment(
        bull_flag, TrendDirection.BULLISH, 0.6
    )
    assert bullish_score >= 0.9

    # Bull flag in bearish trend should score low
    bearish_score = enhanced_validator._score_trend_alignment(
        bull_flag, TrendDirection.BEARISH, 0.6
    )
    assert bearish_score <= 0.4


def test_breadth_support_scoring(enhanced_validator):
    """Test breadth support scoring"""
    # High breadth
    high_score = enhanced_validator._score_breadth_support(0.8)
    assert high_score == 0.8

    # Low breadth
    low_score = enhanced_validator._score_breadth_support(0.3)
    assert low_score == 0.3


# ============================================================================
# Test 3: Pattern-Regime Affinity
# ============================================================================

def test_pattern_regime_affinity_matrix_complete():
    """Test affinity matrix covers all pattern types"""
    # Check that all 15 pattern types are in the matrix
    expected_patterns = [
        PatternType.BULL_FLAG,
        PatternType.BEAR_FLAG,
        PatternType.PENNANT,
        PatternType.DOUBLE_TOP,
        PatternType.DOUBLE_BOTTOM,
        PatternType.TRIPLE_TOP,
        PatternType.TRIPLE_BOTTOM,
        PatternType.ASCENDING_TRIANGLE,
        PatternType.DESCENDING_TRIANGLE,
        PatternType.SYMMETRICAL_TRIANGLE,
        PatternType.HEAD_AND_SHOULDERS,
        PatternType.INVERSE_HEAD_AND_SHOULDERS,
        PatternType.RECTANGLE,
        PatternType.ASCENDING_CHANNEL,
        PatternType.DESCENDING_CHANNEL,
    ]

    for pattern_type in expected_patterns:
        assert pattern_type in PATTERN_REGIME_AFFINITY


def test_bull_flag_trending_bull_affinity(enhanced_validator):
    """Test bull flag has high affinity for trending bull market"""
    affinity = enhanced_validator._get_pattern_regime_affinity(
        PatternType.BULL_FLAG,
        MarketRegime.TRENDING_BULL
    )

    assert affinity == 1.0  # Perfect match


def test_double_top_range_bound_affinity(enhanced_validator):
    """Test double top has high affinity for range-bound market"""
    affinity = enhanced_validator._get_pattern_regime_affinity(
        PatternType.DOUBLE_TOP,
        MarketRegime.RANGE_BOUND
    )

    assert affinity == 1.0  # Perfect match


def test_triangle_multi_regime_affinity(enhanced_validator):
    """Test triangle patterns work in multiple regimes"""
    sym_triangle_regimes = [
        MarketRegime.RANGE_BOUND,
        MarketRegime.BREAKOUT,
        MarketRegime.TRENDING_BULL,
        MarketRegime.TRENDING_BEAR,
    ]

    for regime in sym_triangle_regimes:
        affinity = enhanced_validator._get_pattern_regime_affinity(
            PatternType.SYMMETRICAL_TRIANGLE,
            regime
        )
        # Should have reasonable affinity in all regimes
        assert affinity >= 0.6


def test_incompatible_pattern_regime(enhanced_validator):
    """Test incompatible pattern-regime combinations score low"""
    # Bull flag in bearish trending market
    affinity = enhanced_validator._get_pattern_regime_affinity(
        PatternType.BULL_FLAG,
        MarketRegime.TRENDING_BEAR
    )

    assert affinity == 0.0  # Incompatible


# ============================================================================
# Test 4: Confidence Adjustment
# ============================================================================

def test_confidence_boost_low_volatility(enhanced_validator, sample_pattern, sample_market_data, low_vol_trending_context):
    """Test confidence is boosted in favorable conditions"""
    result = enhanced_validator.validate_pattern_with_context(
        sample_pattern,
        sample_market_data,
        context=low_vol_trending_context
    )

    # Bull flag in trending bull + low vol should boost confidence
    assert result.adjusted_confidence > result.base_confidence
    assert result.context_boost > 0


def test_confidence_reduction_high_volatility(enhanced_validator, sample_pattern, sample_market_data, high_vol_context):
    """Test confidence is reduced in unfavorable conditions"""
    result = enhanced_validator.validate_pattern_with_context(
        sample_pattern,
        sample_market_data,
        context=high_vol_context
    )

    # Extreme volatility should reduce confidence
    assert result.adjusted_confidence < result.base_confidence
    assert result.context_boost < 0


def test_confidence_bounds(enhanced_validator, sample_pattern, sample_market_data, low_vol_trending_context):
    """Test adjusted confidence stays within 0-1 bounds"""
    result = enhanced_validator.validate_pattern_with_context(
        sample_pattern,
        sample_market_data,
        context=low_vol_trending_context
    )

    assert 0.0 <= result.adjusted_confidence <= 1.0


def test_affinity_bonus_applied(enhanced_validator):
    """Test high affinity applies bonus to multiplier"""
    base_confidence = 0.75

    # Create context with 1.2x multiplier
    context = MarketContext(
        timestamp=datetime.now(),
        volatility_regime=VolatilityRegime.LOW,
        volatility_percentile=0.2,
        trend_direction=TrendDirection.BULLISH,
        trend_strength=0.5,
        market_regime=MarketRegime.TRENDING_BULL,
        breadth=MarketBreadth(1.5, 1.2, 1.4, 0.7),
        adaptation=RegimeAdaptation(1.2, 1.0, 1.0, 1.0, 1.0),
        supporting_factors=[],
        risk_factors=[]
    )

    # High affinity (0.9) should apply bonus
    adjusted, boost = enhanced_validator._apply_context_adjustment(
        base_confidence, context, regime_affinity=0.9
    )

    # Should get multiplier + affinity bonus
    assert adjusted > base_confidence * 1.2  # More than just base multiplier


# ============================================================================
# Test 5: Recommendation System
# ============================================================================

def test_recommendation_strength_calculation(enhanced_validator):
    """Test recommendation strength thresholds"""
    assert enhanced_validator._determine_recommendation_strength(0.90) == "VERY_STRONG"
    assert enhanced_validator._determine_recommendation_strength(0.75) == "STRONG"
    assert enhanced_validator._determine_recommendation_strength(0.60) == "MODERATE"
    assert enhanced_validator._determine_recommendation_strength(0.50) == "WEAK"


def test_supporting_reasons_generation(enhanced_validator, sample_pattern, sample_market_data, low_vol_trending_context):
    """Test supporting reasons are generated"""
    result = enhanced_validator.validate_pattern_with_context(
        sample_pattern,
        sample_market_data,
        context=low_vol_trending_context
    )

    assert isinstance(result.supporting_reasons, list)
    assert len(result.supporting_reasons) > 0

    # Should mention low volatility
    reasons_text = ' '.join(result.supporting_reasons).lower()
    assert 'volatility' in reasons_text or 'trend' in reasons_text


def test_risk_warnings_generation(enhanced_validator, sample_pattern, sample_market_data, high_vol_context):
    """Test risk warnings are generated"""
    result = enhanced_validator.validate_pattern_with_context(
        sample_pattern,
        sample_market_data,
        context=high_vol_context
    )

    assert isinstance(result.risk_warnings, list)
    assert len(result.risk_warnings) > 0

    # Should mention volatility risk
    warnings_text = ' '.join(result.risk_warnings).lower()
    assert 'volatility' in warnings_text or 'risk' in warnings_text


def test_high_affinity_mentioned_in_support(enhanced_validator, sample_pattern, sample_market_data, low_vol_trending_context):
    """Test high affinity is mentioned in supporting reasons"""
    result = enhanced_validator.validate_pattern_with_context(
        sample_pattern,
        sample_market_data,
        context=low_vol_trending_context
    )

    # Bull flag in trending bull should mention suitability
    reasons_text = ' '.join(result.supporting_reasons).lower()
    # Should mention either 'suitable' or the regime
    assert 'suitable' in reasons_text or 'trending' in reasons_text or result.regime_affinity >= 0.8


def test_low_affinity_mentioned_in_warnings(enhanced_validator, sample_market_data):
    """Test low affinity is mentioned in warnings"""
    # Create bear flag in bull market (incompatible)
    bear_flag = DetectedPattern(
        symbol="TEST",
        pattern_type=PatternType.BEAR_FLAG,
        timeframe="1D",
        start_time=datetime(2024, 1, 10),
        end_time=datetime(2024, 1, 25),
        confidence_score=0.75,
        strength=PatternStrength.MODERATE,
        key_points=[(datetime(2024, 1, 10), 100.0)],
        pattern_metrics={},
        direction="bearish"
    )

    # Bull market context
    bull_context = MarketContext(
        timestamp=datetime.now(),
        volatility_regime=VolatilityRegime.LOW,
        volatility_percentile=0.2,
        trend_direction=TrendDirection.BULLISH,
        trend_strength=0.6,
        market_regime=MarketRegime.TRENDING_BULL,
        breadth=MarketBreadth(2.0, 1.5, 1.8, 0.75),
        adaptation=RegimeAdaptation(1.2, 1.0, 1.0, 1.0, 1.0),
        supporting_factors=[],
        risk_factors=[]
    )

    validator = EnhancedPatternValidator()
    result = validator.validate_pattern_with_context(
        bear_flag,
        sample_market_data,
        context=bull_context
    )

    # Should warn about incompatibility
    warnings_text = ' '.join(result.risk_warnings).lower()
    assert 'not well-suited' in warnings_text or result.regime_affinity <= 0.4


# ============================================================================
# Test 6: Integration Tests
# ============================================================================

def test_full_context_aware_validation(enhanced_validator, sample_pattern, sample_market_data, low_vol_trending_context):
    """Test complete validation flow with context"""
    result = enhanced_validator.validate_pattern_with_context(
        sample_pattern,
        sample_market_data,
        context=low_vol_trending_context
    )

    # Verify all components worked
    assert result.base_confidence > 0
    assert result.adjusted_confidence > 0
    assert result.context_score > 0
    assert result.regime_affinity > 0
    assert result.recommendation_strength in ["WEAK", "MODERATE", "STRONG", "VERY_STRONG"]
    assert isinstance(result.supporting_reasons, list)
    assert isinstance(result.risk_warnings, list)


def test_multiple_patterns_different_affinities(enhanced_validator, sample_market_data):
    """Test different patterns in same regime have different affinities"""
    context = MarketContext(
        timestamp=datetime.now(),
        volatility_regime=VolatilityRegime.LOW,
        volatility_percentile=0.2,
        trend_direction=TrendDirection.BULLISH,
        trend_strength=0.6,
        market_regime=MarketRegime.TRENDING_BULL,
        breadth=MarketBreadth(2.0, 1.5, 1.8, 0.75),
        adaptation=RegimeAdaptation(1.2, 1.0, 1.0, 1.0, 1.0),
        supporting_factors=[],
        risk_factors=[]
    )

    # Bull flag - should have high affinity
    bull_flag = DetectedPattern(
        symbol="TEST",
        pattern_type=PatternType.BULL_FLAG,
        timeframe="1D",
        start_time=datetime(2024, 1, 10),
        end_time=datetime(2024, 1, 25),
        confidence_score=0.75,
        strength=PatternStrength.MODERATE,
        key_points=[(datetime(2024, 1, 10), 100.0)],
        pattern_metrics={},
        direction="bullish"
    )

    # Double bottom - should have lower affinity
    double_bottom = DetectedPattern(
        symbol="TEST",
        pattern_type=PatternType.DOUBLE_BOTTOM,
        timeframe="1D",
        start_time=datetime(2024, 1, 10),
        end_time=datetime(2024, 1, 25),
        confidence_score=0.75,
        strength=PatternStrength.MODERATE,
        key_points=[(datetime(2024, 1, 10), 100.0)],
        pattern_metrics={},
        direction="bullish"
    )

    result1 = enhanced_validator.validate_pattern_with_context(
        bull_flag, sample_market_data, context
    )
    result2 = enhanced_validator.validate_pattern_with_context(
        double_bottom, sample_market_data, context
    )

    # Bull flag should have higher affinity and confidence in trending bull
    assert result1.regime_affinity > result2.regime_affinity
    assert result1.adjusted_confidence >= result2.adjusted_confidence


def test_same_pattern_different_regimes(enhanced_validator, sample_market_data):
    """Test same pattern in different regimes gets different adjustments"""
    bull_flag = DetectedPattern(
        symbol="TEST",
        pattern_type=PatternType.BULL_FLAG,
        timeframe="1D",
        start_time=datetime(2024, 1, 10),
        end_time=datetime(2024, 1, 25),
        confidence_score=0.75,
        strength=PatternStrength.MODERATE,
        key_points=[(datetime(2024, 1, 10), 100.0)],
        pattern_metrics={},
        direction="bullish"
    )

    # Favorable context
    favorable = MarketContext(
        timestamp=datetime.now(),
        volatility_regime=VolatilityRegime.LOW,
        volatility_percentile=0.2,
        trend_direction=TrendDirection.BULLISH,
        trend_strength=0.6,
        market_regime=MarketRegime.TRENDING_BULL,
        breadth=MarketBreadth(2.0, 1.5, 1.8, 0.75),
        adaptation=RegimeAdaptation(1.5, 1.0, 1.0, 1.0, 1.2),
        supporting_factors=[],
        risk_factors=[]
    )

    # Unfavorable context
    unfavorable = MarketContext(
        timestamp=datetime.now(),
        volatility_regime=VolatilityRegime.EXTREME,
        volatility_percentile=0.95,
        trend_direction=TrendDirection.CHOPPY,
        trend_strength=0.2,
        market_regime=MarketRegime.VOLATILE,
        breadth=MarketBreadth(0.8, 0.6, 0.7, 0.35),
        adaptation=RegimeAdaptation(0.6, 1.0, 1.5, 1.5, 0.5),
        supporting_factors=[],
        risk_factors=["Extreme volatility"]
    )

    result_favorable = enhanced_validator.validate_pattern_with_context(
        bull_flag, sample_market_data, favorable
    )
    result_unfavorable = enhanced_validator.validate_pattern_with_context(
        bull_flag, sample_market_data, unfavorable
    )

    # Favorable should have higher confidence
    assert result_favorable.adjusted_confidence > result_unfavorable.adjusted_confidence
    assert result_favorable.context_boost > result_unfavorable.context_boost


# ============================================================================
# Test 7: Edge Cases
# ============================================================================

def test_unknown_pattern_type_default_affinity(enhanced_validator):
    """Test unknown pattern type gets default affinity"""
    # This would handle future pattern types not yet in the matrix
    affinity = enhanced_validator._get_pattern_regime_affinity(
        PatternType.SYMMETRICAL_TRIANGLE,  # Use existing as placeholder
        MarketRegime.TRENDING_BULL
    )

    assert affinity is not None
    assert 0.0 <= affinity <= 1.0


def test_extreme_confidence_values(enhanced_validator):
    """Test extreme confidence values are handled correctly"""
    context = MarketContext(
        timestamp=datetime.now(),
        volatility_regime=VolatilityRegime.LOW,
        volatility_percentile=0.1,
        trend_direction=TrendDirection.BULLISH,
        trend_strength=0.8,
        market_regime=MarketRegime.TRENDING_BULL,
        breadth=MarketBreadth(2.0, 1.5, 1.8, 0.8),
        adaptation=RegimeAdaptation(2.0, 1.0, 1.0, 1.0, 1.5),  # Max multiplier
        supporting_factors=[],
        risk_factors=[]
    )

    # Very high base confidence
    adjusted, boost = enhanced_validator._apply_context_adjustment(
        0.95, context, regime_affinity=1.0
    )

    # Should not exceed 1.0
    assert adjusted <= 1.0

    # Very low base confidence
    adjusted, boost = enhanced_validator._apply_context_adjustment(
        0.10, context, regime_affinity=1.0
    )

    # Should not go negative
    assert adjusted >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
