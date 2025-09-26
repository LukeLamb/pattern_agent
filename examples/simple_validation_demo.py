"""
Simple Pattern Validation Demo

This example demonstrates the PatternValidator functionality with manually created data.
"""

import sys
import os
sys.path.append('src')

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from validation.pattern_validator import PatternValidator, ValidationCriteria
from models.market_data import MarketData, OHLCV, MarketDataType, MarketSession


class MockPattern:
    """Simple mock pattern for demonstration."""
    
    def __init__(self, pattern_type_str="ascending_triangle"):
        from pattern_detection.pattern_engine import PatternType, PatternStrength
        
        self.pattern_type = PatternType.ASCENDING_TRIANGLE
        self.symbol = "DEMO"
        self.timeframe = "1d"
        self.start_time = datetime(2024, 1, 1)
        self.end_time = datetime(2024, 1, 31)  # 30 days
        self.confidence_score = 0.75
        self.strength = PatternStrength.STRONG
        self.key_points = [
            (datetime(2024, 1, 1), 100.0),
            (datetime(2024, 1, 15), 105.0),
            (datetime(2024, 1, 31), 108.0)
        ]
        self.pattern_metrics = {"height": 8.0, "width": 30}
        self.target_price = None
        self.stop_loss = None
        self.volume_confirmation = False


def create_sample_market_data():
    """Create sample market data for testing."""
    print("Creating sample market data...")
    
    # Generate 60 days of synthetic OHLCV data
    start_date = datetime(2023, 12, 1)
    dates = [start_date + timedelta(days=i) for i in range(60)]
    
    # Create realistic price movement with trend
    rng = np.random.default_rng(42)
    base_prices = np.linspace(95, 108, 60) + rng.normal(0, 1, 60)
    
    ohlcv_data = []
    for i, date in enumerate(dates):
        base_price = max(90, base_prices[i])  # Ensure positive prices
        
        high = base_price + abs(rng.normal(0, 2))
        low = base_price - abs(rng.normal(0, 2))
        open_price = base_price + rng.normal(0, 0.5)
        close_price = base_price
        volume = int(abs(rng.normal(100000, 20000)))
        
        ohlcv_data.append(OHLCV(
            timestamp=date,
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume
        ))
    
    return MarketData(
        symbol="DEMO",
        timeframe="1d",
        data_type=MarketDataType.OHLCV,
        data=ohlcv_data,
        start_time=dates[0],
        end_time=dates[-1],
        data_source="demo",
        market_session=MarketSession.REGULAR,
        completeness_score=1.0,
        gaps_detected=0,
        anomalies_detected=0,
        total_volume=sum(d.volume for d in ohlcv_data),
        vwap=102.5,
        price_range=15.0,
        volatility=0.025
    )


def main():
    """Run the validation demonstration."""
    print("=" * 80)
    print("PATTERN VALIDATOR DEMO")
    print("=" * 80)
    
    # 1. Create sample data
    market_data = create_sample_market_data()
    print(f"Created {len(market_data.data)} data points")
    print(f"Date range: {market_data.start_time.date()} to {market_data.end_time.date()}")
    
    # 2. Create mock pattern
    pattern = MockPattern()
    print(f"\\nMock Pattern: {pattern.pattern_type.value}")
    print(f"Symbol: {pattern.symbol}")
    print(f"Formation period: {pattern.start_time.date()} to {pattern.end_time.date()}")
    print(f"Original confidence: {pattern.confidence_score:.3f}")
    
    # 3. Initialize validator
    validator = PatternValidator(
        min_formation_days=5,
        max_formation_days=60,
        min_volume_confirmation_ratio=1.2,
        historical_lookback_days=365
    )
    print(f"\\nInitialized PatternValidator")
    
    # 4. Run comprehensive validation
    print(f"\\n{'='*40}")
    print("VALIDATION RESULTS")
    print(f"{'='*40}")
    
    validation_results = validator.validate_pattern(pattern, market_data)
    overall_score = validator.calculate_overall_validation_score(validation_results)
    quality_metrics = validator.calculate_pattern_quality_metrics(pattern, market_data)
    
    print(f"\\n📊 OVERALL SCORES:")
    print(f"• Validation Score: {overall_score:.3f}")
    print(f"• Quality Score: {quality_metrics.overall_quality_score:.3f}")
    print(f"• Confidence Adjustment: {quality_metrics.confidence_adjustment:.3f}")
    
    print(f"\\n🔍 DETAILED VALIDATION:")
    for result in validation_results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"• {result.criteria.value.replace('_', ' ').title()}: {result.score:.3f} {status}")
        
        # Show some key details
        if result.criteria == ValidationCriteria.VOLUME_CONFIRMATION:
            volume_ratio = result.details.get('volume_ratio', 0)
            threshold = result.details.get('threshold', 0)
            print(f"  Volume Ratio: {volume_ratio:.2f} (threshold: {threshold})")
            
        elif result.criteria == ValidationCriteria.TIMEFRAME_CONSISTENCY:
            formation_days = result.details.get('formation_days', 0)
            print(f"  Formation Days: {formation_days}")
            
        elif result.criteria == ValidationCriteria.HISTORICAL_SUCCESS_RATE:
            success_rate = result.details.get('success_rate', 0)
            count = result.details.get('historical_count', 0)
            print(f"  Success Rate: {success_rate:.2f} ({count} historical patterns)")
    
    print(f"\\n⚙️ QUALITY METRICS BREAKDOWN:")
    print(f"• Formation Time: {quality_metrics.formation_time_score:.3f}")
    print(f"• Symmetry: {quality_metrics.symmetry_score:.3f}")
    print(f"• Volume Pattern: {quality_metrics.volume_pattern_score:.3f}")
    print(f"• Technical Strength: {quality_metrics.technical_strength_score:.3f}")
    print(f"• Market Context: {quality_metrics.market_context_score:.3f}")
    
    # Calculate adjusted confidence
    original_confidence = pattern.confidence_score
    adjusted_confidence = original_confidence * quality_metrics.confidence_adjustment
    
    print(f"\\n🎯 CONFIDENCE ADJUSTMENT:")
    print(f"• Original: {original_confidence:.3f}")
    print(f"• Adjusted: {adjusted_confidence:.3f}")
    print(f"• Factor: {quality_metrics.confidence_adjustment:.3f}")
    
    # Final recommendation
    if overall_score >= 0.7 and quality_metrics.overall_quality_score >= 0.6:
        recommendation = "🟢 STRONG"
        advice = "High confidence pattern - consider for trading"
    elif overall_score >= 0.5 and quality_metrics.overall_quality_score >= 0.4:
        recommendation = "🟡 MODERATE"
        advice = "Moderate confidence - requires additional confirmation"
    else:
        recommendation = "🔴 WEAK"
        advice = "Low confidence - avoid trading on this pattern"
    
    print(f"\\n📈 FINAL RECOMMENDATION: {recommendation}")
    print(f"• {advice}")
    
    print(f"\\n{'='*80}")
    print("✅ Pattern Validation Demo Complete!")
    print(f"{'='*80}")
    
    return validation_results, quality_metrics


if __name__ == "__main__":
    main()