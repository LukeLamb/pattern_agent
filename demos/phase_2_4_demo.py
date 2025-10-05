"""
Phase 2.4 Demo: Enhanced Pattern Validation with Market Context

Demonstrates context-aware pattern validation that adjusts confidence
based on market regime, volatility, trend, and breadth.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.market_data import MarketData, OHLCV
from src.pattern_detection import DetectedPattern, PatternType, PatternStrength
from src.validation import EnhancedPatternValidator, PATTERN_REGIME_AFFINITY
from src.market_context import (
    MarketContextAnalyzer,
    MarketRegime,
    VolatilityRegime,
    TrendDirection,
)


def create_market_data(
    symbol: str,
    trend: str = "bullish",
    volatility: str = "low",
    days: int = 100
) -> MarketData:
    """Create synthetic market data with specified characteristics"""
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    ohlcv_data = []

    # Determine trend slope
    if trend == "bullish":
        base_trend = np.linspace(100, 120, days)
    elif trend == "bearish":
        base_trend = np.linspace(120, 100, days)
    else:  # sideways
        base_trend = np.full(days, 110)

    # Determine volatility
    if volatility == "low":
        noise = np.random.randn(days) * 0.5
    elif volatility == "high":
        noise = np.random.randn(days) * 2.0
    else:  # medium
        noise = np.random.randn(days) * 1.0

    prices = base_trend + noise

    for i, date in enumerate(dates):
        ohlcv_data.append(OHLCV(
            timestamp=date.to_pydatetime(),
            open=float(prices[i] - 0.2),
            high=float(prices[i] + 0.5),
            low=float(prices[i] - 0.5),
            close=float(prices[i]),
            volume=int(1000000 + np.random.randint(-200000, 200000))
        ))

    return MarketData(
        symbol=symbol,
        timeframe="1D",
        data=ohlcv_data,
        start_time=dates[0].to_pydatetime(),
        end_time=dates[-1].to_pydatetime()
    )


def create_sample_pattern(
    pattern_type: PatternType,
    base_confidence: float = 0.70
) -> DetectedPattern:
    """Create a sample detected pattern"""
    # Determine direction based on pattern type
    if pattern_type in [PatternType.BULL_FLAG, PatternType.DOUBLE_BOTTOM,
                        PatternType.ASCENDING_TRIANGLE, PatternType.INVERSE_HEAD_AND_SHOULDERS]:
        direction = "bullish"
    elif pattern_type in [PatternType.BEAR_FLAG, PatternType.DOUBLE_TOP,
                          PatternType.DESCENDING_TRIANGLE, PatternType.HEAD_AND_SHOULDERS]:
        direction = "bearish"
    else:
        direction = "neutral"

    return DetectedPattern(
        symbol="TEST",
        pattern_type=pattern_type,
        timeframe="1D",
        start_time=datetime(2024, 1, 10),
        end_time=datetime(2024, 1, 25),
        confidence_score=base_confidence,
        strength=PatternStrength.MODERATE,
        key_points=[(datetime(2024, 1, 10), 100.0), (datetime(2024, 1, 25), 110.0)],
        pattern_metrics={"test": "data"},
        direction=direction
    )


def print_divider(char="=", length=80):
    """Print a divider line"""
    print(char * length)


def print_section_header(title: str):
    """Print a section header"""
    print_divider()
    print(f"  {title}")
    print_divider()
    print()


def print_validation_result(result, scenario_name: str):
    """Print enhanced validation result"""
    print(f"[*] {scenario_name}")
    print("-" * 80)
    print(f"  Pattern: {result.pattern_type.value}")
    print(f"  Symbol: {result.symbol}")
    print()

    # Market Context
    if result.market_context:
        ctx = result.market_context
        print(f"  Market Context:")
        print(f"    - Regime: {ctx.market_regime.value}")
        print(f"    - Volatility: {ctx.volatility_regime.value} ({ctx.volatility_percentile:.1%} percentile)")
        print(f"    - Trend: {ctx.trend_direction.value} (strength: {ctx.trend_strength:.2f})")
        print(f"    - Breadth Score: {ctx.breadth.breadth_score:.2f}")
        print()

    # Confidence Adjustment
    print(f"  Confidence Analysis:")
    print(f"    - Base Confidence: {result.base_confidence:.1%}")
    print(f"    - Adjusted Confidence: {result.adjusted_confidence:.1%}")
    print(f"    - Context Boost: {result.context_boost:+.1%}")
    print(f"    - Regime Affinity: {result.regime_affinity:.2f}")
    print()

    # Scoring Breakdown
    print(f"  Scoring Breakdown:")
    print(f"    - Context Score: {result.context_score:.2f}")
    print(f"    - Volume Score: {result.volume_score:.2f}")
    print(f"    - Quality Score: {result.quality_score:.2f}")
    print()

    # Recommendation
    print(f"  [+] Recommendation: {result.recommendation_strength}")

    if result.supporting_reasons:
        print(f"  [SUPPORTING FACTORS]:")
        for reason in result.supporting_reasons:
            print(f"    - {reason}")

    if result.risk_warnings:
        print(f"  [RISK WARNINGS]:")
        for warning in result.risk_warnings:
            print(f"    - {warning}")

    print()


def demo_scenario_1():
    """Scenario 1: Bull Flag in Trending Bull Market (Ideal Conditions)"""
    print_section_header("SCENARIO 1: Bull Flag in Trending Bull Market")

    # Create favorable market conditions
    market_data = create_market_data("AAPL", trend="bullish", volatility="low")

    # Analyze market context
    analyzer = MarketContextAnalyzer()
    df = market_data.to_dataframe()
    context = analyzer.analyze_context(df)

    # Create bull flag pattern
    pattern = create_sample_pattern(PatternType.BULL_FLAG, base_confidence=0.75)

    # Validate with context
    validator = EnhancedPatternValidator()
    result = validator.validate_pattern_with_context(pattern, market_data, context)

    print_validation_result(result, "Bull Flag in Ideal Conditions")
    print("[!] Analysis: Bull flag in trending bull + low volatility should receive")
    print("   significant confidence boost due to high regime affinity (1.0)")
    print()


def demo_scenario_2():
    """Scenario 2: Bear Flag in Trending Bull Market (Incompatible)"""
    print_section_header("SCENARIO 2: Bear Flag in Trending Bull Market")

    # Create bull market conditions
    market_data = create_market_data("TSLA", trend="bullish", volatility="low")

    # Analyze market context
    analyzer = MarketContextAnalyzer()
    df = market_data.to_dataframe()
    context = analyzer.analyze_context(df)

    # Create bear flag pattern (counter-trend)
    pattern = create_sample_pattern(PatternType.BEAR_FLAG, base_confidence=0.75)

    # Validate with context
    validator = EnhancedPatternValidator()
    result = validator.validate_pattern_with_context(pattern, market_data, context)

    print_validation_result(result, "Bear Flag in Bull Market (Incompatible)")
    print("[!] Analysis: Bear flag in trending bull market should receive confidence")
    print("   penalty due to low regime affinity (0.0) and trend misalignment")
    print()


def demo_scenario_3():
    """Scenario 3: Double Top in Range-Bound Market (Good Match)"""
    print_section_header("SCENARIO 3: Double Top in Range-Bound Market")

    # Create range-bound market
    market_data = create_market_data("SPY", trend="sideways", volatility="medium")

    # Analyze market context
    analyzer = MarketContextAnalyzer()
    df = market_data.to_dataframe()
    context = analyzer.analyze_context(df)

    # Create double top pattern
    pattern = create_sample_pattern(PatternType.DOUBLE_TOP, base_confidence=0.70)

    # Validate with context
    validator = EnhancedPatternValidator()
    result = validator.validate_pattern_with_context(pattern, market_data, context)

    print_validation_result(result, "Double Top in Range-Bound Market")
    print("[!] Analysis: Double tops excel in range-bound markets (affinity: 0.9)")
    print("   where price oscillates between support/resistance levels")
    print()


def demo_scenario_4():
    """Scenario 4: Same Pattern, Different Regimes"""
    print_section_header("SCENARIO 4: Bull Flag Across Different Market Regimes")

    validator = EnhancedPatternValidator()
    pattern = create_sample_pattern(PatternType.BULL_FLAG, base_confidence=0.70)

    scenarios = [
        ("Trending Bull + Low Vol", "bullish", "low"),
        ("Trending Bear + Low Vol", "bearish", "low"),
        ("Sideways + High Vol", "sideways", "high"),
    ]

    for scenario_name, trend, volatility in scenarios:
        market_data = create_market_data("TEST", trend=trend, volatility=volatility)
        analyzer = MarketContextAnalyzer()
        df = market_data.to_dataframe()
        context = analyzer.analyze_context(df)

        result = validator.validate_pattern_with_context(pattern, market_data, context)

        print(f"[*] {scenario_name}:")
        print(f"    Base: {result.base_confidence:.1%} -> Adjusted: {result.adjusted_confidence:.1%} "
              f"(Boost: {result.context_boost:+.1%})")
        print(f"    Regime: {context.market_regime.value}, Affinity: {result.regime_affinity:.2f}")
        print(f"    Recommendation: {result.recommendation_strength}")
        print()

    print("[!] Analysis: Same pattern receives different confidence adjustments")
    print("   based on market regime compatibility and conditions")
    print()


def demo_scenario_5():
    """Scenario 5: Pattern-Regime Affinity Matrix Overview"""
    print_section_header("SCENARIO 5: Pattern-Regime Affinity Matrix")

    print("[LIST] Affinity Matrix (Pattern vs. Market Regime)")
    print("-" * 80)
    print()

    # Show affinity for selected patterns across all regimes
    selected_patterns = [
        PatternType.BULL_FLAG,
        PatternType.BEAR_FLAG,
        PatternType.DOUBLE_TOP,
        PatternType.ASCENDING_TRIANGLE,
    ]

    regimes = [
        MarketRegime.TRENDING_BULL,
        MarketRegime.TRENDING_BEAR,
        MarketRegime.RANGE_BOUND,
        MarketRegime.VOLATILE,
        MarketRegime.BREAKOUT,
    ]

    # Print header
    print(f"{'Pattern':<30}", end="")
    for regime in regimes:
        print(f"{regime.value[:12]:<15}", end="")
    print()
    print("-" * 105)

    # Print affinity scores
    for pattern_type in selected_patterns:
        print(f"{pattern_type.value:<30}", end="")
        if pattern_type in PATTERN_REGIME_AFFINITY:
            for regime in regimes:
                affinity = PATTERN_REGIME_AFFINITY[pattern_type].get(regime, 0.5)
                # Color-code the affinity
                if affinity >= 0.8:
                    marker = "[H]"
                elif affinity >= 0.5:
                    marker = "[M]"
                else:
                    marker = "[L]"
                print(f"{marker} {affinity:.1f}          ", end="")
        print()

    print()
    print("Legend: [H] High (>=0.8)  [M] Medium (>=0.5)  [L] Low (<0.5)")
    print()
    print("[!] Analysis: Affinity matrix maps pattern suitability to market regimes,")
    print("   enabling context-aware confidence adjustments")
    print()


def demo_backward_compatibility():
    """Demo: Backward Compatibility (Validation without Context)"""
    print_section_header("BONUS: Backward Compatibility Test")

    market_data = create_market_data("TEST", trend="bullish", volatility="low")
    pattern = create_sample_pattern(PatternType.BULL_FLAG, base_confidence=0.75)

    validator = EnhancedPatternValidator()

    # Validate WITHOUT context (backward compatible)
    result = validator.validate_pattern_with_context(pattern, market_data, context=None)

    print("[*] Validation Without Market Context (Backward Compatible):")
    print("-" * 80)
    print(f"  Pattern: {result.pattern_type.value}")
    print(f"  Base Confidence: {result.base_confidence:.1%}")
    print(f"  Adjusted Confidence: {result.adjusted_confidence:.1%}")
    print(f"  Context Provided: {'Yes' if result.market_context else 'No'}")
    print(f"  Recommendation: {result.recommendation_strength}")
    print()
    print("[!] Analysis: Enhanced validator works with or without market context,")
    print("   ensuring backward compatibility with existing code")
    print()


def main():
    """Run Phase 2.4 demo scenarios"""
    print()
    print("=" * 80)
    print("  PHASE 2.4: ENHANCED PATTERN VALIDATION WITH MARKET CONTEXT")
    print("  Context-Aware Confidence Adjustment & Pattern-Regime Affinity")
    print("=" * 80)
    print()

    # Run all scenarios
    demo_scenario_1()
    demo_scenario_2()
    demo_scenario_3()
    demo_scenario_4()
    demo_scenario_5()
    demo_backward_compatibility()

    # Summary
    print_divider("=")
    print("  [OK] PHASE 2.4 DEMO COMPLETE")
    print_divider("=")
    print()
    print("Key Takeaways:")
    print("  1. [OK] Context-aware validation adjusts confidence based on market regime")
    print("  2. [OK] Pattern-regime affinity matrix ensures suitable pattern matching")
    print("  3. [OK] Multi-factor scoring: volatility, trend, breadth, affinity")
    print("  4. [OK] Enhanced recommendations with supporting reasons & risk warnings")
    print("  5. [OK] Backward compatible - works with or without market context")
    print()
    print("Next Steps:")
    print("  -> Phase 2.5: Historical Pattern Success Tracking")
    print("  -> Phase 2.6: Advanced Risk Management Integration")
    print()


if __name__ == "__main__":
    main()
