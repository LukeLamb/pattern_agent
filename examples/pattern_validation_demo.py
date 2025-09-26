"""
Pattern Validation Integration Example

This example demonstrates how to use the PatternValidator with the Pattern Detection Engine
for comprehensive pattern analysis and validation.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from src.pattern_detection.pattern_engine import PatternDetectionEngine, PatternType, PatternStrength
    from src.validation.pattern_validator import PatternValidator, ValidationCriteria
    from src.models.market_data import MarketData, OHLCV, MarketDataType, MarketSession
    from src.data_processing.synthetic_data_generator import SyntheticDataGenerator
except ImportError:
    # For standalone execution
    import sys
    import os
    sys.path.append('src')
    
    from pattern_detection.pattern_engine import PatternDetectionEngine, PatternType, PatternStrength
    from validation.pattern_validator import PatternValidator, ValidationCriteria
    from models.market_data import MarketData, OHLCV, MarketDataType, MarketSession
    from data_processing.synthetic_data_generator import SyntheticDataGenerator


def demonstrate_pattern_validation():
    """Demonstrate complete pattern detection and validation workflow."""
    print("=" * 80)
    print("PATTERN VALIDATION INTEGRATION DEMO")
    print("=" * 80)
    
    # 1. Generate synthetic market data
    print("\n1. Generating synthetic market data...")
    generator = SyntheticDataGenerator()
    
    # Generate data with triangle pattern
    market_data = generator.generate_triangle_pattern(
        symbol="DEMO",
        pattern_type="ascending",
        duration_days=30,
        start_price=100.0,
        base_volume=100000
    )
    
    print(f"   Generated {len(market_data.data)} data points for {market_data.symbol}")
    print(f"   Time range: {market_data.start_time.date()} to {market_data.end_time.date()}")
    
    # 2. Initialize pattern detection engine
    print("\n2. Initializing pattern detection engine...")
    detection_engine = PatternDetectionEngine(
        min_pattern_length=10,
        min_touches=2
    )
    
    # 3. Detect patterns
    print("\n3. Detecting patterns...")
    detected_patterns = detection_engine.detect_patterns(market_data)
    print(f"   Detected {len(detected_patterns)} patterns:")
    
    for i, pattern in enumerate(detected_patterns):
        print(f"   [{i+1}] {pattern.pattern_type.value} - Confidence: {pattern.confidence_score:.2f}")
        print(f"       Period: {pattern.start_time.date()} to {pattern.end_time.date()}")
        print(f"       Strength: {pattern.strength.value}")
    
    # 4. Initialize pattern validator
    print("\n4. Initializing pattern validator...")
    validator = PatternValidator(
        min_formation_days=5,
        max_formation_days=60,
        min_volume_confirmation_ratio=1.2,
        historical_lookback_days=365
    )
    
    # 5. Validate each detected pattern
    print("\n5. Validating detected patterns...")
    
    for i, pattern in enumerate(detected_patterns):
        print(f"\n   --- PATTERN {i+1}: {pattern.pattern_type.value.upper()} ---")
        
        # Perform comprehensive validation
        validation_results = validator.validate_pattern(pattern, market_data)
        
        # Calculate overall validation score
        overall_score = validator.calculate_overall_validation_score(validation_results)
        
        # Calculate quality metrics
        quality_metrics = validator.calculate_pattern_quality_metrics(pattern, market_data)
        
        print(f"   Overall Validation Score: {overall_score:.3f}")
        print(f"   Quality Score: {quality_metrics.overall_quality_score:.3f}")
        print(f"   Confidence Adjustment: {quality_metrics.confidence_adjustment:.3f}")
        
        print("\n   Detailed Validation Results:")
        for result in validation_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"   • {result.criteria.value}: {result.score:.3f} {status}")
            
            # Show key details for some criteria
            if result.criteria == ValidationCriteria.VOLUME_CONFIRMATION:
                volume_ratio = result.details.get('volume_ratio', 0)
                print(f"     Volume Ratio: {volume_ratio:.2f} (threshold: {result.details.get('threshold', 'N/A')})")
            
            elif result.criteria == ValidationCriteria.TIMEFRAME_CONSISTENCY:
                formation_days = result.details.get('formation_days', 0)
                print(f"     Formation Time: {formation_days} days")
            
            elif result.criteria == ValidationCriteria.HISTORICAL_SUCCESS_RATE:
                success_rate = result.details.get('success_rate', 0)
                historical_count = result.details.get('historical_count', 0)
                print(f"     Historical Success Rate: {success_rate:.2f} ({historical_count} samples)")
        
        print("\n   Quality Metrics Breakdown:")
        print(f"   • Formation Time Score: {quality_metrics.formation_time_score:.3f}")
        print(f"   • Symmetry Score: {quality_metrics.symmetry_score:.3f}")
        print(f"   • Volume Pattern Score: {quality_metrics.volume_pattern_score:.3f}")
        print(f"   • Technical Strength Score: {quality_metrics.technical_strength_score:.3f}")
        print(f"   • Market Context Score: {quality_metrics.market_context_score:.3f}")
        
        # Adjusted confidence score
        original_confidence = pattern.confidence_score
        adjusted_confidence = original_confidence * quality_metrics.confidence_adjustment
        
        print(f"\n   Confidence Score Adjustment:")
        print(f"   • Original Confidence: {original_confidence:.3f}")
        print(f"   • Adjusted Confidence: {adjusted_confidence:.3f}")
        print(f"   • Adjustment Factor: {quality_metrics.confidence_adjustment:.3f}")
        
        # Overall recommendation
        if overall_score >= 0.7 and quality_metrics.overall_quality_score >= 0.6:
            recommendation = "🟢 STRONG - High confidence pattern"
        elif overall_score >= 0.5 and quality_metrics.overall_quality_score >= 0.4:
            recommendation = "🟡 MODERATE - Consider additional analysis"
        else:
            recommendation = "🔴 WEAK - Low confidence pattern"
        
        print(f"\n   📊 RECOMMENDATION: {recommendation}")
    
    # 6. Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if detected_patterns:
        high_quality_patterns = sum(1 for pattern in detected_patterns 
                                  if validator.calculate_overall_validation_score(
                                      validator.validate_pattern(pattern, market_data)) >= 0.7)
        
        print(f"Total Patterns Detected: {len(detected_patterns)}")
        print(f"High Quality Patterns: {high_quality_patterns}")
        print(f"Quality Rate: {high_quality_patterns/len(detected_patterns)*100:.1f}%")
        
        # Validation criteria performance
        print(f"\nValidation Criteria Weights:")
        for criteria, weight in validator.validation_weights.items():
            print(f"• {criteria.value}: {weight*100:.1f}%")
    else:
        print("No patterns detected in the provided data.")
    
    print(f"\nValidation Configuration:")
    print(f"• Min Formation Days: {validator.min_formation_days}")
    print(f"• Max Formation Days: {validator.max_formation_days}")
    print(f"• Min Volume Confirmation Ratio: {validator.min_volume_confirmation_ratio}")
    print(f"• Historical Lookback Days: {validator.historical_lookback_days}")
    
    print("\n✅ Pattern validation integration demo completed!")
    return detected_patterns, validation_results if detected_patterns else []


def demonstrate_validation_criteria():
    """Demonstrate individual validation criteria in detail."""
    print("\n" + "=" * 80)
    print("VALIDATION CRITERIA DEEP DIVE")
    print("=" * 80)
    
    print("\n📋 VALIDATION CRITERIA EXPLAINED:")
    
    criteria_explanations = {
        ValidationCriteria.VOLUME_CONFIRMATION: {
            "description": "Validates volume behavior during pattern formation",
            "checks": [
                "Volume increase during pattern formation vs. pre-pattern period",
                "Minimum volume confirmation ratio threshold",
                "Volume trend consistency with pattern expectations"
            ],
            "weight": "25%",
            "importance": "Critical for confirming genuine breakouts"
        },
        
        ValidationCriteria.TIMEFRAME_CONSISTENCY: {
            "description": "Validates pattern formation timeframe appropriateness",
            "checks": [
                "Formation time within acceptable range",
                "Not too fast (< min days) or too slow (> max days)",
                "Optimal formation time scoring"
            ],
            "weight": "20%", 
            "importance": "Ensures patterns have proper development time"
        },
        
        ValidationCriteria.HISTORICAL_SUCCESS_RATE: {
            "description": "Evaluates pattern based on historical performance",
            "checks": [
                "Success rate of similar patterns in the past",
                "Sufficient historical data for reliable statistics",
                "Pattern type and strength historical correlation"
            ],
            "weight": "30%",
            "importance": "Highest weight - real performance matters most"
        },
        
        ValidationCriteria.PATTERN_QUALITY: {
            "description": "Assesses technical quality of pattern formation", 
            "checks": [
                "Formation time appropriateness",
                "Pattern symmetry and proportions",
                "Volume behavior during formation",
                "Technical indicator strength",
                "Market context suitability"
            ],
            "weight": "15%",
            "importance": "Comprehensive quality assessment"
        },
        
        ValidationCriteria.MARKET_CONTEXT: {
            "description": "Validates suitability for current market conditions",
            "checks": [
                "Market volatility regime appropriateness",
                "Trend strength and direction suitability", 
                "Pattern type performance in current conditions"
            ],
            "weight": "10%",
            "importance": "Context-dependent pattern effectiveness"
        }
    }
    
    for criteria, info in criteria_explanations.items():
        print(f"\n🔍 {criteria.value.upper().replace('_', ' ')}")
        print(f"   Weight: {info['weight']} | {info['importance']}")
        print(f"   {info['description']}")
        print("   Validation Checks:")
        for check in info['checks']:
            print(f"   • {check}")
    
    print(f"\n⚖️ SCORING METHODOLOGY:")
    print("• Each criterion scores 0.0 to 1.0 (0% to 100%)")
    print("• Weighted combination produces overall validation score")
    print("• Quality metrics provide additional confidence adjustment")
    print("• Boolean pass/fail threshold varies by criterion")
    print("• Historical success rate has highest weight (30%)")


if __name__ == "__main__":
    # Run the integration demonstration
    patterns, results = demonstrate_pattern_validation()
    
    # Run the criteria explanation
    demonstrate_validation_criteria()
    
    print(f"\n{'='*80}")
    print("🎯 PHASE 1.5 PATTERN VALIDATION - IMPLEMENTATION COMPLETE!")
    print(f"{'='*80}")