"""
Interactive Test 3: Adaptive Parameters

This script demonstrates how market context parameters adapt
across different market scenarios.
"""

import sys
import pandas as pd
import numpy as np

# Configure UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.market_context import MarketContextAnalyzer

print("=" * 80)
print("  Interactive Test 3: Adaptive Parameters Across Market Scenarios")
print("=" * 80)


def create_scenario_data(scenario_type):
    """Create different market scenarios"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    if scenario_type == 'trending_bull':
        # Strong uptrend
        close = np.linspace(100, 125, 100) + np.random.randn(100) * 0.3
        desc = "Strong uptrend (+25% over 100 days)"

    elif scenario_type == 'trending_bear':
        # Strong downtrend
        close = np.linspace(100, 80, 100) + np.random.randn(100) * 0.3
        desc = "Strong downtrend (-20% over 100 days)"

    elif scenario_type == 'volatile':
        # High volatility random walk
        close = 100 + np.cumsum(np.random.randn(100) * 3)
        desc = "High volatility (large random swings)"

    elif scenario_type == 'sideways':
        # Range-bound oscillation
        close = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 3 + np.random.randn(100) * 0.5
        desc = "Sideways/range-bound (±3% oscillation)"

    elif scenario_type == 'low_vol_trend':
        # Low volatility uptrend
        close = np.linspace(100, 115, 100) + np.random.randn(100) * 0.1
        desc = "Low volatility uptrend (+15%, tight)"

    else:
        close = 100 + np.random.randn(100)
        desc = "Random walk"

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close - 0.2,
        'high': close + np.abs(np.random.randn(100) * 0.8),
        'low': close - np.abs(np.random.randn(100) * 0.8),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, 100)
    })

    return df, desc


print("\n1. Creating different market scenarios...")
analyzer = MarketContextAnalyzer()

scenarios = {
    'trending_bull': 'Strong Bull Trend',
    'trending_bear': 'Strong Bear Trend',
    'volatile': 'High Volatility',
    'sideways': 'Sideways/Range',
    'low_vol_trend': 'Low Vol Trend'
}

results = []

for scenario_key, scenario_name in scenarios.items():
    df, description = create_scenario_data(scenario_key)
    context = analyzer.analyze_context(df)

    results.append({
        'name': scenario_name,
        'desc': description,
        'vol_regime': context.volatility_regime.value,
        'trend': context.trend_direction.value,
        'strength': context.trend_strength,
        'market_regime': context.market_regime.value,
        'conf_mult': context.adaptation.confidence_multiplier,
        'lookback': context.adaptation.lookback_adjustment,
        'vol_thresh': context.adaptation.volume_threshold,
        'bo_thresh': context.adaptation.breakout_threshold,
        'risk_adj': context.adaptation.risk_adjustment,
    })

print(f"   ✓ Analyzed {len(results)} different market scenarios")

# Display comprehensive results
print("\n" + "=" * 80)
print("  ADAPTIVE PARAMETER COMPARISON")
print("=" * 80)

print(f"\n{'Scenario':<20} {'Vol':<8} {'Trend':<10} {'Str':<6} {'Conf':<8} {'Risk':<8}")
print("-" * 80)

for r in results:
    print(f"{r['name']:<20} {r['vol_regime']:<8} {r['trend']:<10} "
          f"{r['strength']:>4.2f}  {r['conf_mult']:>6.2f}x {r['risk_adj']:>6.2f}x")

# Detailed parameter breakdown
print("\n" + "=" * 80)
print("  DETAILED PARAMETER BREAKDOWN")
print("=" * 80)

print(f"\n{'Scenario':<20} {'Conf':<8} {'Lookback':<10} {'Vol Req':<9} "
      f"{'BO Req':<8} {'Risk':<8}")
print("-" * 80)

for r in results:
    print(f"{r['name']:<20} {r['conf_mult']:>6.2f}x {r['lookback']:>8.2f}x "
          f"{r['vol_thresh']:>7.2f}x {r['bo_thresh']:>6.2f}x {r['risk_adj']:>6.2f}x")

# Detailed analysis for each scenario
print("\n" + "=" * 80)
print("  SCENARIO-BY-SCENARIO ANALYSIS")
print("=" * 80)

for r in results:
    print(f"\n{r['name']}")
    print(f"  {r['desc']}")
    print(f"  Volatility: {r['vol_regime'].upper()} | "
          f"Trend: {r['trend'].upper()} ({r['strength']:.2f}) | "
          f"Regime: {r['market_regime'].upper()}")
    print(f"\n  Adaptive Parameters:")
    print(f"    • Confidence Multiplier: {r['conf_mult']:.2f}x", end="")
    if r['conf_mult'] > 1.1:
        print(" → BOOST pattern confidence")
    elif r['conf_mult'] < 0.9:
        print(" → REDUCE pattern confidence")
    else:
        print(" → NEUTRAL")

    print(f"    • Lookback Adjustment: {r['lookback']:.2f}x", end="")
    if r['lookback'] > 1.0:
        print(" → Use LONGER formation periods")
    elif r['lookback'] < 1.0:
        print(" → Use SHORTER formation periods")
    else:
        print(" → STANDARD periods")

    print(f"    • Volume Threshold: {r['vol_thresh']:.2f}x", end="")
    if r['vol_thresh'] > 1.1:
        print(" → Require STRONGER volume")
    else:
        print(" → STANDARD volume requirements")

    print(f"    • Breakout Threshold: {r['bo_thresh']:.2f}x", end="")
    if r['bo_thresh'] > 1.1:
        print(" → Require LARGER breakouts")
    elif r['bo_thresh'] < 0.9:
        print(" → Allow SMALLER breakouts")
    else:
        print(" → STANDARD breakouts")

    print(f"    • Risk Adjustment: {r['risk_adj']:.2f}x", end="")
    if r['risk_adj'] > 1.1:
        print(" → INCREASE position sizes")
    elif r['risk_adj'] < 0.9:
        print(" → REDUCE position sizes")
    else:
        print(" → STANDARD position sizes")

# Comparison analysis
print("\n" + "=" * 80)
print("  KEY INSIGHTS: Parameter Adaptation Logic")
print("=" * 80)

print("\n1. CONFIDENCE MULTIPLIER:")
print("   • Trending markets → Higher confidence (favor continuation)")
print("   • Low volatility → Higher confidence (clearer patterns)")
print("   • High volatility → Lower confidence (more false signals)")
print("   • Range: 0.5x (extreme vol) to 1.56x (low vol + trend)")

print("\n2. LOOKBACK ADJUSTMENT:")
print("   • Trending → Longer lookback (1.2x) → larger patterns valid")
print("   • Range-bound → Shorter lookback (0.9x) → quick reversals")

print("\n3. VOLUME THRESHOLD:")
print("   • High volatility → Stricter (1.3x-1.5x) → avoid false breakouts")
print("   • Breakout regime → Very strict (1.4x) → need strong confirmation")
print("   • Low volatility → Standard (1.0x) → normal requirements")

print("\n4. BREAKOUT THRESHOLD:")
print("   • Low volatility → Easier (0.8x) → small moves significant")
print("   • High volatility → Harder (1.3x-1.5x) → need larger moves")

print("\n5. RISK ADJUSTMENT:")
print("   • Low volatility → Larger positions (1.2x) → favorable conditions")
print("   • High volatility → Smaller positions (0.7x) → protect capital")
print("   • Extreme volatility → Much smaller (0.5x) → capital preservation")

# Practical example
print("\n" + "=" * 80)
print("  PRACTICAL EXAMPLE: Bull Flag Pattern Across Scenarios")
print("=" * 80)

base_confidence = 0.75
base_position = 100  # shares

print(f"\nBase pattern: Bull flag with 75% confidence")
print(f"Base position size: {base_position} shares\n")

print(f"{'Scenario':<20} {'Adj Conf':<12} {'Adj Position':<14} {'Recommendation':<20}")
print("-" * 80)

for r in results:
    adj_conf = base_confidence * r['conf_mult']
    adj_pos = int(base_position * r['risk_adj'])

    if adj_conf >= 0.85 and r['risk_adj'] >= 1.0:
        rec = "STRONG BUY"
    elif adj_conf >= 0.70 and r['risk_adj'] >= 0.8:
        rec = "BUY"
    elif adj_conf >= 0.55:
        rec = "WEAK BUY"
    else:
        rec = "AVOID/REDUCE"

    print(f"{r['name']:<20} {adj_conf:>10.0%}  {adj_pos:>12} sh  {rec:<20}")

print("\n" + "=" * 80)
print("  Test Complete!")
print("=" * 80)
print("\nKey Takeaway:")
print("  The same pattern can have vastly different confidence and position sizing")
print("  based on market context. This adaptive approach helps:")
print("    • Boost performance in favorable conditions")
print("    • Reduce losses in unfavorable conditions")
print("    • Automatically adjust risk as markets change")
print()
