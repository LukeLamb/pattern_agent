"""
Interactive Test 1: Basic Market Context Analysis

This script demonstrates basic usage of the MarketContextAnalyzer
with a simple bullish trend scenario.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.market_context import MarketContextAnalyzer

print("=" * 70)
print("  Interactive Test 1: Basic Market Context Analysis")
print("=" * 70)

# Create analyzer
print("\n1. Creating MarketContextAnalyzer...")
analyzer = MarketContextAnalyzer()
print("   ✓ Analyzer created")

# Generate sample bullish trend data
print("\n2. Generating sample bullish trend data (100 days, +30% gain)...")
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
trend = np.linspace(100, 130, 100)  # 30% uptrend
noise = np.random.randn(100) * 0.5
close_prices = trend + noise

df = pd.DataFrame({
    'timestamp': dates,
    'open': close_prices - 0.2,
    'high': close_prices + 0.8,
    'low': close_prices - 0.8,
    'close': close_prices,
    'volume': np.random.randint(1000000, 5000000, 100)
})
print(f"   ✓ Generated {len(df)} days of data")
print(f"   ✓ Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
print(f"   ✓ Total gain: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.1f}%")

# Analyze context
print("\n3. Analyzing market context...")
context = analyzer.analyze_context(df)
print("   ✓ Analysis complete")

# Print results
print("\n" + "=" * 70)
print("  MARKET CONTEXT ANALYSIS RESULTS")
print("=" * 70)

print(f"\n📊 VOLATILITY ANALYSIS")
print(f"   Regime: {context.volatility_regime.value.upper()}")
print(f"   Percentile: {context.volatility_percentile:.1%}")

print(f"\n📈 TREND ANALYSIS")
print(f"   Direction: {context.trend_direction.value.upper()}")
print(f"   Strength: {context.trend_strength:.2f}/1.00")

if context.trend_strength > 0.6:
    strength_label = "STRONG"
elif context.trend_strength > 0.4:
    strength_label = "MODERATE"
elif context.trend_strength > 0.2:
    strength_label = "WEAK"
else:
    strength_label = "VERY WEAK"
print(f"   Assessment: {strength_label}")

print(f"\n🏛️ MARKET REGIME")
print(f"   Classification: {context.market_regime.value.upper()}")

print(f"\n📉 MARKET BREADTH")
breadth = context.breadth
print(f"   Advance/Decline Ratio: {breadth.advance_decline_ratio:.2f}")
print(f"   New Highs/Lows Ratio: {breadth.new_highs_lows_ratio:.2f}")
print(f"   Volume Breadth: {breadth.volume_breadth:.2f}")
print(f"   Overall Score: {breadth.breadth_score:.2f}/1.00")

print(f"\n⚙️ ADAPTIVE PARAMETERS")
adapt = context.adaptation
print(f"   Confidence Multiplier: {adapt.confidence_multiplier:.2f}x")
print(f"   Lookback Adjustment: {adapt.lookback_adjustment:.2f}x")
print(f"   Volume Threshold: {adapt.volume_threshold:.2f}x")
print(f"   Breakout Threshold: {adapt.breakout_threshold:.2f}x")
print(f"   Risk Adjustment: {adapt.risk_adjustment:.2f}x")

if context.supporting_factors:
    print(f"\n✅ SUPPORTING FACTORS ({len(context.supporting_factors)})")
    for factor in context.supporting_factors:
        print(f"   • {factor}")

if context.risk_factors:
    print(f"\n⚠️ RISK FACTORS ({len(context.risk_factors)})")
    for factor in context.risk_factors:
        print(f"   • {factor}")

# Practical interpretation
print("\n" + "=" * 70)
print("  PRACTICAL INTERPRETATION")
print("=" * 70)

print("\nWhat this means for pattern detection:")
if adapt.confidence_multiplier > 1.1:
    print(f"   ✓ Pattern confidence will be BOOSTED by {((adapt.confidence_multiplier - 1) * 100):.0f}%")
elif adapt.confidence_multiplier < 0.9:
    print(f"   ⚠ Pattern confidence will be REDUCED by {((1 - adapt.confidence_multiplier) * 100):.0f}%")
else:
    print(f"   → Pattern confidence will be NEUTRAL (minimal adjustment)")

print("\nWhat this means for risk management:")
if adapt.risk_adjustment > 1.1:
    print(f"   ✓ Position sizes can be INCREASED by {((adapt.risk_adjustment - 1) * 100):.0f}%")
elif adapt.risk_adjustment < 0.9:
    print(f"   ⚠ Position sizes should be REDUCED by {((1 - adapt.risk_adjustment) * 100):.0f}%")
else:
    print(f"   → Position sizes should be STANDARD (minimal adjustment)")

print("\nExample: Bull flag pattern with 75% base confidence")
base_confidence = 0.75
adjusted_confidence = base_confidence * adapt.confidence_multiplier
print(f"   Base confidence: {base_confidence:.1%}")
print(f"   Adjusted confidence: {adjusted_confidence:.1%}")
print(f"   Change: {((adjusted_confidence - base_confidence) / base_confidence * 100):+.1f}%")

print("\n" + "=" * 70)
print("  Test Complete!")
print("=" * 70)
print()
