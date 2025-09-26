"""
Phase 1.7 Visualization and Reporting Demo

This script demonstrates the basic visualization and reporting capabilities
implemented for Phase 1.7 of the Pattern Recognition Agent.

Features demonstrated:
- Pattern chart visualization with detected patterns
- Technical indicators overlay
- Basic reporting with pattern summaries
- Export functionality for charts and reports
"""

import sys
import os
from datetime import datetime, timedelta
import warnings

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress matplotlib backend warnings for headless environments
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

try:
    # Core components
    from models.market_data import MarketData, OHLCV
    from technical_indicators.indicator_engine import TechnicalIndicatorEngine
    from pattern_detection.pattern_engine import PatternDetectionEngine
    from validation.pattern_validator import PatternValidator
    
    # Phase 1.7 components
    from visualization.pattern_plotter import PatternPlotter
    from reporting.basic_reporter import BasicReporter
    
    print("✅ Successfully imported all components for Phase 1.7 demo")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("  pip install matplotlib seaborn")
    sys.exit(1)


def create_demo_market_data() -> MarketData:
    """Create realistic market data for demonstration."""
    print("📊 Creating demo market data...")
    
    # Create 100 data points with realistic OHLCV data
    data_points = []
    base_price = 150.0
    base_time = datetime.now() - timedelta(days=100)
    
    # Simulate price movement with trend and volatility
    import random
    random.seed(42)  # For reproducible demo
    
    for i in range(100):
        # Add some trend and volatility
        price_change = random.gauss(0, 2)  # Random walk with volatility
        trend_component = 0.1 * i  # Slight upward trend
        
        close_price = base_price + trend_component + price_change
        open_price = close_price + random.gauss(0, 0.5)
        
        # Create realistic high/low
        high_price = max(open_price, close_price) + abs(random.gauss(0, 1))
        low_price = min(open_price, close_price) - abs(random.gauss(0, 1))
        
        volume = random.randint(50000, 200000)
        
        ohlcv = OHLCV(
            timestamp=base_time + timedelta(days=i),
            open=max(1.0, open_price),
            high=max(1.0, high_price),
            low=max(1.0, low_price),
            close=max(1.0, close_price),
            volume=volume
        )
        data_points.append(ohlcv)
    
    market_data = MarketData(
        symbol="DEMO",
        timeframe="daily",
        data=data_points,
        start_time=data_points[0].timestamp,
        end_time=data_points[-1].timestamp,
        data_source="demo_generator"
    )
    
    print(f"✅ Created market data: {len(data_points)} data points from {market_data.start_time.date()} to {market_data.end_time.date()}")
    return market_data


def demo_pattern_detection_pipeline(market_data: MarketData):
    """Demonstrate the complete pattern detection pipeline."""
    print("\n🔍 Running pattern detection pipeline...")
    
    # Initialize engines
    indicator_engine = TechnicalIndicatorEngine()
    pattern_engine = PatternDetectionEngine()
    validator = PatternValidator()
    
    # Step 1: Calculate technical indicators
    print("  Step 1: Calculating technical indicators...")
    indicators = indicator_engine.calculate_indicators(market_data)
    
    if 'error' in indicators:
        print(f"  ❌ Error calculating indicators: {indicators['error']}")
        return None, None, None
    
    print(f"  ✅ Calculated indicators: {len([k for k in indicators.keys() if isinstance(indicators[k], dict)])} categories")
    
    # Step 2: Detect patterns
    print("  Step 2: Detecting patterns...")
    patterns = pattern_engine.detect_patterns(market_data)
    print(f"  ✅ Detected {len(patterns)} patterns")
    
    # Step 3: Validate patterns
    print("  Step 3: Validating patterns...")
    validation_results = []
    
    for pattern in patterns:
        try:
            result = validator.validate_pattern(pattern, market_data)
            validation_results.append(result)
        except Exception as e:
            print(f"  ⚠️ Validation error for pattern {pattern.pattern_type}: {e}")
            validation_results.append(None)
    
    valid_results = [r for r in validation_results if r is not None]
    print(f"  ✅ Validated {len(valid_results)}/{len(patterns)} patterns")
    
    return indicators, patterns, validation_results


def demo_visualization(market_data: MarketData, patterns, indicators):
    """Demonstrate pattern visualization capabilities."""
    print("\n📈 Creating pattern visualization...")
    
    # Initialize plotter
    plotter = PatternPlotter(figsize=(14, 8))
    
    try:
        # Create pattern chart
        print("  Creating comprehensive pattern chart...")
        fig = plotter.plot_pattern_chart(
            market_data=market_data,
            patterns=patterns,
            indicators=indicators,
            title=f"{market_data.symbol} - Pattern Detection Analysis",
            show=False,  # Don't show in headless environment
            save_path="demo_pattern_chart.png"
        )
        
        print("  ✅ Pattern chart created and saved as 'demo_pattern_chart.png'")
        
        # Create pattern summary chart
        if patterns:
            print("  Creating pattern summary chart...")
            summary_fig = plotter.create_pattern_summary_chart(
                patterns=patterns,
                title="Pattern Detection Summary",
                show=False,
                save_path="demo_summary_chart.png"
            )
            print("  ✅ Summary chart created and saved as 'demo_summary_chart.png'")
        
        # Close figures to free memory
        plotter.close_all_figures()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Visualization error: {e}")
        print("  Note: Visualization may not work in headless environments")
        return False


def demo_reporting(market_data: MarketData, patterns, validation_results):
    """Demonstrate reporting capabilities."""
    print("\n📄 Creating pattern reports...")
    
    # Initialize reporter
    reporter = BasicReporter(output_dir="reports")
    
    try:
        # Generate comprehensive report
        print("  Generating comprehensive pattern report...")
        comprehensive_report = reporter.generate_pattern_summary_report(
            patterns=patterns,
            market_data=market_data,
            validation_results=validation_results,
            export_format='json'
        )
        
        print(f"  ✅ Comprehensive report generated with {len(comprehensive_report.get('patterns', []))} patterns analyzed")
        
        # Generate confidence score report
        print("  Generating confidence score analysis...")
        confidence_report = reporter.create_confidence_score_report(
            patterns=patterns,
            market_data=market_data
        )
        
        print(f"  ✅ Confidence analysis completed - Average confidence: {confidence_report.get('confidence_statistics', {}).get('mean', 0):.2f}")
        
        # Generate pattern statistics
        print("  Generating pattern statistics...")
        stats_report = reporter.generate_pattern_statistics(
            patterns=patterns,
            market_data=market_data,
            time_period_days=100
        )
        
        overall_stats = stats_report.get('overall_statistics', {})
        print(f"  ✅ Statistics generated - {overall_stats.get('total_patterns_detected', 0)} patterns, {overall_stats.get('unique_pattern_types', 0)} types")
        
        # Export to CSV
        print("  Exporting data to CSV...")
        csv_path = reporter.export_to_csv(
            patterns=patterns,
            market_data=market_data,
            validation_results=validation_results
        )
        print(f"  ✅ CSV export completed: {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Reporting error: {e}")
        return False


def print_demo_summary(patterns, validation_results):
    """Print a summary of the demo results."""
    print("\n" + "="*60)
    print("🎉 PHASE 1.7 DEMO COMPLETED")
    print("="*60)
    
    print(f"\n📊 Results Summary:")
    print(f"  • Patterns Detected: {len(patterns)}")
    
    if patterns:
        pattern_types = {}
        confidence_scores = []
        
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
            confidence_scores.append(pattern.confidence)
        
        print(f"  • Pattern Types: {list(pattern_types.keys())}")
        print(f"  • Average Confidence: {sum(confidence_scores)/len(confidence_scores):.2f}")
        print(f"  • Highest Confidence: {max(confidence_scores):.2f}")
    
    valid_validations = [r for r in validation_results if r is not None]
    print(f"  • Validation Success: {len(valid_validations)}/{len(patterns)} patterns")
    
    print(f"\n📈 Phase 1.7 Features Demonstrated:")
    print(f"  ✅ Pattern Visualization (PatternPlotter)")
    print(f"  ✅ Technical Indicators Overlay")  
    print(f"  ✅ Pattern Annotations and Charts")
    print(f"  ✅ Basic Reporting (BasicReporter)")
    print(f"  ✅ Confidence Score Analysis")
    print(f"  ✅ Pattern Statistics Generation")
    print(f"  ✅ CSV Export Functionality")
    print(f"  ✅ Multi-format Report Export")
    
    print(f"\n📁 Output Files Generated:")
    print(f"  • demo_pattern_chart.png - Comprehensive pattern chart")
    print(f"  • demo_summary_chart.png - Pattern summary visualization")
    print(f"  • reports/ - Directory containing JSON and CSV reports")
    
    print(f"\n🚀 Ready for Phase 2: Multi-Timeframe Analysis")


def main():
    """Main demo function."""
    print("🌟 PATTERN RECOGNITION AGENT - PHASE 1.7 DEMO")
    print("Basic Visualization & Output Capabilities")
    print("="*60)
    
    try:
        # Create demo data
        market_data = create_demo_market_data()
        
        # Run pattern detection pipeline
        indicators, patterns, validation_results = demo_pattern_detection_pipeline(market_data)
        
        if patterns is None:
            print("❌ Pattern detection failed, cannot proceed with demo")
            return
        
        # Demonstrate visualization
        viz_success = demo_visualization(market_data, patterns, indicators)
        
        # Demonstrate reporting
        report_success = demo_reporting(market_data, patterns, validation_results)
        
        # Print summary
        print_demo_summary(patterns, validation_results)
        
        if viz_success and report_success:
            print("\n✅ All Phase 1.7 features successfully demonstrated!")
        else:
            print("\n⚠️ Some features may have limitations in current environment")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check dependencies and environment setup")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()