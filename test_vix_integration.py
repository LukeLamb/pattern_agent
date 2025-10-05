"""
Interactive Test 2: VIX Integration

This script demonstrates how volatility regime detection works
with VIX data across different volatility environments.
"""

import sys
import pandas as pd
import numpy as np

# Configure UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.market_context import MarketContextAnalyzer

print("=" * 70)
print("  Interactive Test 2: VIX Integration")
print("=" * 70)

analyzer = MarketContextAnalyzer()

# Create neutral market data
print("\n1. Creating neutral market data (sideways movement)...")
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'timestamp': dates,
    'open': 100 + np.random.randn(100) * 0.5,
    'high': 101 + np.random.randn(100) * 0.5,
    'low': 99 + np.random.randn(100) * 0.5,
    'close': 100 + np.random.randn(100) * 0.5,
    'volume': np.random.randint(1000000, 5000000, 100)
})
print("   ✓ Market data created (neutral/sideways)")

# Test different VIX scenarios
print("\n2. Testing different VIX scenarios...")
vix_scenarios = {
    'Low Volatility (VIX=12-15)': 13,
    'Normal Volatility (VIX=15-20)': 18,
    'Elevated Volatility (VIX=20-25)': 22,
    'High Volatility (VIX=25-35)': 30,
    'Extreme Volatility (VIX=35-50)': 42
}

print("\n" + "=" * 70)
print("  VIX INTEGRATION RESULTS")
print("=" * 70)

results = []

for scenario_name, vix_value in vix_scenarios.items():
    # Create VIX data
    vix_df = pd.DataFrame({
        'timestamp': dates,
        'close': [vix_value] * 100
    })

    # Analyze with VIX data
    context = analyzer.analyze_context(df, vix_data=vix_df)

    results.append({
        'scenario': scenario_name,
        'vix': vix_value,
        'regime': context.volatility_regime.value,
        'conf_mult': context.adaptation.confidence_multiplier,
        'risk_adj': context.adaptation.risk_adjustment,
        'vol_thresh': context.adaptation.volume_threshold,
        'bo_thresh': context.adaptation.breakout_threshold
    })

# Display results in table format
print(f"\n{'VIX Scenario':<35} {'VIX':<6} {'Regime':<10} {'Conf':<8} {'Risk':<8}")
print("-" * 70)

for r in results:
    print(f"{r['scenario']:<35} {r['vix']:<6.0f} {r['regime']:<10} "
          f"{r['conf_mult']:>6.2f}x {r['risk_adj']:>6.2f}x")

# Detailed analysis of each scenario
print("\n" + "=" * 70)
print("  DETAILED ANALYSIS BY SCENARIO")
print("=" * 70)

for r in results:
    print(f"\n{r['scenario']} (VIX={r['vix']})")
    print(f"   Detected Regime: {r['regime'].upper()}")
    print(f"   Confidence Multiplier: {r['conf_mult']:.2f}x")
    print(f"   Risk Adjustment: {r['risk_adj']:.2f}x")
    print(f"   Volume Threshold: {r['vol_thresh']:.2f}x")
    print(f"   Breakout Threshold: {r['bo_thresh']:.2f}x")

    # Interpretation
    if r['regime'] == 'low':
        print("   → Low volatility: Pattern clarity is high, can be more aggressive")
    elif r['regime'] == 'medium':
        print("   → Medium volatility: Normal trading conditions, standard parameters")
    elif r['regime'] == 'high':
        print("   → High volatility: Increased risk, require stronger confirmation")
    elif r['regime'] == 'extreme':
        print("   → Extreme volatility: Very high risk, be very conservative")

# Practical example
print("\n" + "=" * 70)
print("  PRACTICAL EXAMPLE: Pattern Confidence Adjustment")
print("=" * 70)

base_confidence = 0.75
print(f"\nAssume a triangle pattern is detected with 75% base confidence.")
print(f"How does confidence change across VIX environments?\n")

print(f"{'VIX Environment':<35} {'Base':<10} {'Adjusted':<12} {'Change':<10}")
print("-" * 70)

for r in results:
    adjusted = base_confidence * r['conf_mult']
    change = ((adjusted - base_confidence) / base_confidence) * 100

    print(f"{r['scenario']:<35} {base_confidence:>8.0%}  {adjusted:>10.0%}  {change:>8.1f}%")

# Key insights
print("\n" + "=" * 70)
print("  KEY INSIGHTS")
print("=" * 70)

print("\n1. Volatility Impact on Confidence:")
print("   • Low VIX (12-15): Boost confidence by ~20% → clearer patterns")
print("   • Normal VIX (15-20): Maintain confidence → standard conditions")
print("   • High VIX (25-35): Reduce confidence by ~20% → more noise")
print("   • Extreme VIX (35+): Reduce confidence by ~40% → very unreliable")

print("\n2. Risk Management Adaptation:")
print("   • Low volatility → Larger positions (1.2x)")
print("   • High volatility → Smaller positions (0.7x)")
print("   • Extreme volatility → Much smaller positions (0.5x)")

print("\n3. Volume Requirements:")
print("   • Low volatility → Standard volume needed")
print("   • High/Extreme volatility → 1.3x-1.5x higher volume needed")
print("   • Reason: Avoid false breakouts in choppy markets")

print("\n" + "=" * 70)
print("  Test Complete!")
print("=" * 70)
print("\nNext: Try test_adaptive_params.py to see how all parameters adjust")
print("      across different market conditions.")
print()
