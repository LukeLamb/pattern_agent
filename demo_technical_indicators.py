#!/usr/bin/env python3
"""
Technical Indicator Engine Demo

Demonstrates the comprehensive technical indicator capabilities
of the Pattern Recognition Agent.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from technical_indicators.indicator_engine import (
    TechnicalIndicatorEngine,
    IndicatorConfig,
)


def generate_sample_data(days=200):
    """Generate realistic sample market data for demonstration."""
    rng = np.random.default_rng(42)  # For reproducible results

    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")

    # Generate price with trend and volatility
    base_price = 100
    trend = np.linspace(0, 20, days)  # Upward trend
    noise = np.cumsum(rng.normal(0, 1.5, days))  # Random walk

    close_prices = base_price + trend + noise

    # Generate OHLC from close prices
    volatility = rng.uniform(0.5, 3.0, days)

    high_prices = close_prices + rng.uniform(0.2, 1.0, days) * volatility
    low_prices = close_prices - rng.uniform(0.2, 1.0, days) * volatility

    # Open prices with some gap behavior
    open_prices = close_prices + rng.normal(0, 0.5, days)

    # Volume with some correlation to price movement
    base_volume = 1000000
    volume_trend = np.abs(np.diff(close_prices, prepend=close_prices[0])) * 500000
    volume_noise = rng.uniform(0.5, 1.5, days)
    volume = (base_volume + volume_trend) * volume_noise

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.round(open_prices, 2),
            "high": np.round(high_prices, 2),
            "low": np.round(low_prices, 2),
            "close": np.round(close_prices, 2),
            "volume": volume.astype(int),
        }
    )


def demonstrate_indicators():
    """Demonstrate the technical indicator engine capabilities."""
    print("🚀 Pattern Recognition Agent - Technical Indicator Engine Demo")
    print("=" * 65)

    # Generate sample data
    print("\n📊 Generating sample market data...")
    sample_data = generate_sample_data(200)
    print(f"✅ Generated {len(sample_data)} days of market data")
    print(
        f"   Price range: ${sample_data['close'].min():.2f} - ${sample_data['close'].max():.2f}"
    )
    print(f"   Average volume: {sample_data['volume'].mean():,.0f}")

    # Create indicator engine with default configuration
    print("\n🔧 Initializing Technical Indicator Engine...")
    engine = TechnicalIndicatorEngine()

    print("   Default configuration:")
    print(f"   • SMA Periods: {engine.config.sma_periods}")
    print(f"   • EMA Periods: {engine.config.ema_periods}")
    print(f"   • RSI Period: {engine.config.rsi_period}")
    print(
        f"   • Bollinger Bands: {engine.config.bb_period} periods, {engine.config.bb_std_dev} std dev"
    )

    # Calculate all indicators
    print("\n🧮 Calculating comprehensive technical indicators...")

    if not engine._validate_data(sample_data):
        print("❌ Insufficient data for comprehensive indicator calculation")
        return

    # Calculate each category
    trend_indicators = engine._calculate_trend_indicators(sample_data)
    momentum_indicators = engine._calculate_momentum_indicators(sample_data)
    volume_indicators = engine._calculate_volume_indicators(sample_data)
    volatility_indicators = engine._calculate_volatility_indicators(sample_data)

    print("✅ All indicators calculated successfully!")

    # Display results
    print("\n📈 TREND INDICATORS:")
    for name, values in trend_indicators.items():
        if isinstance(values, dict):
            print(
                f"   • {name.upper()}: Latest = {list(values.values())[0].iloc[-1]:.3f}"
            )
        else:
            latest_value = (
                values.dropna().iloc[-1] if not values.dropna().empty else "N/A"
            )
            print(f"   • {name.upper()}: Latest = {latest_value}")

    print("\n📊 MOMENTUM INDICATORS:")
    for name, values in momentum_indicators.items():
        if isinstance(values, dict):
            for sub_name, sub_values in values.items():
                latest_value = (
                    sub_values.dropna().iloc[-1]
                    if not sub_values.dropna().empty
                    else "N/A"
                )
                print(f"   • {name.upper()} {sub_name}: Latest = {latest_value}")
        else:
            latest_value = (
                values.dropna().iloc[-1] if not values.dropna().empty else "N/A"
            )
            print(f"   • {name.upper()}: Latest = {latest_value}")

    print("\n📦 VOLUME INDICATORS:")
    for name, values in volume_indicators.items():
        latest_value = values.dropna().iloc[-1] if not values.dropna().empty else "N/A"
        print(f"   • {name.upper()}: Latest = {latest_value}")

    print("\n🌊 VOLATILITY INDICATORS:")
    for name, values in volatility_indicators.items():
        if isinstance(values, dict):
            for sub_name, sub_values in values.items():
                latest_value = (
                    sub_values.dropna().iloc[-1]
                    if not sub_values.dropna().empty
                    else "N/A"
                )
                print(f"   • {name.upper()} {sub_name}: Latest = {latest_value}")
        else:
            latest_value = (
                values.dropna().iloc[-1] if not values.dropna().empty else "N/A"
            )
            print(f"   • {name.upper()}: Latest = {latest_value}")

    # Demonstrate custom configuration
    print("\n🎛️ Demonstrating custom configuration...")
    custom_config = IndicatorConfig(
        sma_periods=[5, 10, 21],
        ema_periods=[8, 21],
        rsi_period=10,
        bb_period=14,
        bb_std_dev=1.5,
    )

    custom_engine = TechnicalIndicatorEngine(custom_config)
    custom_trend = custom_engine._calculate_trend_indicators(sample_data)

    print("   Custom SMA values (last 5 days):")
    for period in custom_config.sma_periods:
        sma_key = f"sma_{period}"
        if sma_key in custom_trend:
            recent_values = custom_trend[sma_key].dropna().tail(5)
            print(f"     SMA-{period}: {[f'{v:.2f}' for v in recent_values]}")

    # Performance insights
    print("\n⚡ Performance Statistics:")
    total_indicators = (
        len(trend_indicators)
        + len(momentum_indicators)
        + len(volume_indicators)
        + len(volatility_indicators)
    )
    print(f"   • Total indicators calculated: {total_indicators}")
    print(f"   • Data points processed: {len(sample_data)}")
    print("   • Engine configuration flexibility: ✅ Full customization support")
    print("   • Error handling: ✅ Robust NaN and edge case management")

    print("\n🎯 Technical Indicator Engine Status: FULLY OPERATIONAL")
    print("🔮 Ready for Pattern Detection Algorithm integration!")
    print("\n" + "=" * 65)


if __name__ == "__main__":
    try:
        demonstrate_indicators()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback

        traceback.print_exc()
