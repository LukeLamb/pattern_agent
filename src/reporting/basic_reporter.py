"""
Basic Reporting - Generate pattern detection summaries and statistics.

This module provides basic reporting capabilities including:
- Pattern detection summary reports
- Confidence score analysis
- Pattern statistics and metrics
- Export functionality for reports
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import os
from pathlib import Path

try:
    from ..models.market_data import MarketData
    from ..models.pattern import DetectedPattern
    from ..validation.pattern_validator import PatternValidator, ValidationResult
except ImportError:
    # For testing and development
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.market_data import MarketData
    from models.pattern import DetectedPattern
    from validation.pattern_validator import PatternValidator, ValidationResult


@dataclass
class PatternSummary:
    """Summary information for a detected pattern."""
    pattern_type: str
    symbol: str
    timeframe: str
    confidence: float
    start_time: str
    end_time: str
    validation_score: Optional[float] = None
    validation_recommendation: Optional[str] = None
    key_metrics: Optional[Dict[str, float]] = None


@dataclass
class ReportMetrics:
    """Overall metrics for a pattern detection report."""
    total_patterns: int
    patterns_by_type: Dict[str, int]
    average_confidence: float
    high_confidence_patterns: int  # Confidence >= 0.7
    validated_patterns: int
    strong_recommendations: int
    analysis_timeframe: str
    report_generation_time: str


class BasicReporter:
    """
    Basic pattern detection reporter.
    
    Provides functionality to generate comprehensive reports including:
    - Pattern detection summaries
    - Confidence score analysis
    - Validation results compilation  
    - Export to various formats (JSON, CSV, HTML)
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the BasicReporter.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Report configuration
        self.confidence_thresholds = {
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
        
        self.validation_score_thresholds = {
            'strong': 0.7,
            'moderate': 0.5,
            'weak': 0.3
        }

    def generate_pattern_summary_report(
        self,
        patterns: List[DetectedPattern],
        market_data: MarketData,
        validation_results: Optional[List[ValidationResult]] = None,
        export_format: str = 'json'
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive pattern summary report.
        
        Args:
            patterns: List of detected patterns
            market_data: Market data used for pattern detection
            validation_results: Optional validation results for patterns
            export_format: Export format ('json', 'csv', 'html')
            
        Returns:
            Dictionary containing the complete report
        """
        # Create pattern summaries
        pattern_summaries = []
        validation_dict = {}
        
        # Map validation results to patterns
        if validation_results:
            for i, result in enumerate(validation_results):
                if i < len(patterns):
                    validation_dict[i] = result
        
        for i, pattern in enumerate(patterns):
            validation_result = validation_dict.get(i)
            
            # Create pattern summary
            summary = PatternSummary(
                pattern_type=pattern.pattern_type,
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                confidence=pattern.confidence,
                start_time=self._format_datetime(pattern.start_time),
                end_time=self._format_datetime(pattern.end_time),
                validation_score=validation_result.overall_score if validation_result else None,
                validation_recommendation=self._get_recommendation(validation_result) if validation_result else None,
                key_metrics=self._extract_pattern_metrics(pattern)
            )
            
            pattern_summaries.append(summary)
        
        # Calculate overall metrics
        metrics = self._calculate_report_metrics(patterns, validation_results, market_data)
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'symbol': market_data.symbol,
                'timeframe': market_data.timeframe,
                'data_points': len(market_data),
                'analysis_period': {
                    'start': self._format_datetime(market_data.start_time),
                    'end': self._format_datetime(market_data.end_time)
                },
                'generation_time': datetime.now().isoformat(),
                'report_version': '1.0'
            },
            'summary_metrics': asdict(metrics),
            'patterns': [asdict(summary) for summary in pattern_summaries],
            'confidence_distribution': self._calculate_confidence_distribution(patterns),
            'pattern_type_analysis': self._analyze_patterns_by_type(patterns, validation_results),
            'recommendations': self._generate_recommendations(patterns, validation_results)
        }
        
        # Export report
        filename = self._generate_filename(market_data, export_format)
        self._export_report(report, filename, export_format)
        
        return report

    def create_confidence_score_report(
        self,
        patterns: List[DetectedPattern],
        market_data: MarketData
    ) -> Dict[str, Any]:
        """
        Create a detailed confidence score analysis report.
        
        Args:
            patterns: List of detected patterns
            market_data: Market data used for analysis
            
        Returns:
            Dictionary containing confidence analysis
        """
        if not patterns:
            return {'error': 'No patterns provided for confidence analysis'}
        
        confidences = [p.confidence for p in patterns]
        
        # Statistical analysis
        confidence_stats = {
            'count': len(confidences),
            'mean': sum(confidences) / len(confidences),
            'median': sorted(confidences)[len(confidences)//2],
            'min': min(confidences),
            'max': max(confidences),
            'std_dev': self._calculate_std_dev(confidences)
        }
        
        # Distribution analysis
        distribution = {
            'high_confidence': len([c for c in confidences if c >= self.confidence_thresholds['high']]),
            'medium_confidence': len([c for c in confidences if self.confidence_thresholds['medium'] <= c < self.confidence_thresholds['high']]),
            'low_confidence': len([c for c in confidences if c < self.confidence_thresholds['medium']])
        }
        
        # Pattern type confidence analysis
        type_confidence = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in type_confidence:
                type_confidence[pattern_type] = []
            type_confidence[pattern_type].append(pattern.confidence)
        
        # Calculate average confidence per pattern type
        type_avg_confidence = {
            pt: sum(confidences) / len(confidences)
            for pt, confidences in type_confidence.items()
        }
        
        report = {
            'symbol': market_data.symbol,
            'timeframe': market_data.timeframe,
            'analysis_time': datetime.now().isoformat(),
            'confidence_statistics': confidence_stats,
            'confidence_distribution': distribution,
            'pattern_type_confidence': type_avg_confidence,
            'detailed_patterns': [
                {
                    'pattern_type': p.pattern_type,
                    'confidence': p.confidence,
                    'confidence_category': self._categorize_confidence(p.confidence),
                    'start_time': self._format_datetime(p.start_time),
                    'end_time': self._format_datetime(p.end_time)
                }
                for p in patterns
            ]
        }
        
        return report

    def generate_pattern_statistics(
        self,
        patterns: List[DetectedPattern],
        market_data: MarketData,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate comprehensive pattern statistics.
        
        Args:
            patterns: List of detected patterns
            market_data: Market data for context
            time_period_days: Analysis time period in days
            
        Returns:
            Dictionary containing pattern statistics
        """
        if not patterns:
            return {'error': 'No patterns provided for statistics'}
        
        # Pattern type frequency
        type_counts = {}
        type_confidences = {}
        formation_times = {}
        
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            
            # Count patterns by type
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
            
            # Track confidences by type
            if pattern_type not in type_confidences:
                type_confidences[pattern_type] = []
            type_confidences[pattern_type].append(pattern.confidence)
            
            # Calculate formation time
            if hasattr(pattern, 'start_time') and hasattr(pattern, 'end_time'):
                try:
                    start_time = pd.Timestamp(pattern.start_time)
                    end_time = pd.Timestamp(pattern.end_time)
                    formation_days = (end_time - start_time).days
                    
                    if pattern_type not in formation_times:
                        formation_times[pattern_type] = []
                    formation_times[pattern_type].append(formation_days)
                except Exception:
                    pass  # Skip if datetime parsing fails
        
        # Calculate statistics
        pattern_stats = {}
        for pattern_type in type_counts:
            confidences = type_confidences[pattern_type]
            times = formation_times.get(pattern_type, [])
            
            pattern_stats[pattern_type] = {
                'count': type_counts[pattern_type],
                'frequency_percent': (type_counts[pattern_type] / len(patterns)) * 100,
                'average_confidence': sum(confidences) / len(confidences),
                'max_confidence': max(confidences),
                'min_confidence': min(confidences),
                'average_formation_days': sum(times) / len(times) if times else 0,
                'confidence_std_dev': self._calculate_std_dev(confidences)
            }
        
        # Overall statistics
        overall_stats = {
            'total_patterns_detected': len(patterns),
            'unique_pattern_types': len(type_counts),
            'analysis_period_days': time_period_days,
            'patterns_per_day': len(patterns) / max(time_period_days, 1),
            'average_overall_confidence': sum(p.confidence for p in patterns) / len(patterns),
            'most_common_pattern': max(type_counts, key=type_counts.get),
            'highest_confidence_pattern': max(patterns, key=lambda p: p.confidence).pattern_type
        }
        
        return {
            'symbol': market_data.symbol,
            'timeframe': market_data.timeframe,
            'generation_time': datetime.now().isoformat(),
            'overall_statistics': overall_stats,
            'pattern_type_statistics': pattern_stats,
            'summary': self._generate_statistics_summary(overall_stats, pattern_stats)
        }

    def export_to_csv(
        self,
        patterns: List[DetectedPattern],
        market_data: MarketData,
        validation_results: Optional[List[ValidationResult]] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Export patterns to CSV format.
        
        Args:
            patterns: List of detected patterns
            market_data: Market data context
            validation_results: Optional validation results
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the exported CSV file
        """
        # Create DataFrame from patterns
        data = []
        
        for i, pattern in enumerate(patterns):
            validation_result = validation_results[i] if validation_results and i < len(validation_results) else None
            
            row = {
                'symbol': market_data.symbol,
                'timeframe': market_data.timeframe,
                'pattern_type': pattern.pattern_type,
                'confidence': pattern.confidence,
                'start_time': self._format_datetime(pattern.start_time),
                'end_time': self._format_datetime(pattern.end_time),
                'validation_score': validation_result.overall_score if validation_result else None,
                'recommendation': self._get_recommendation(validation_result) if validation_result else None
            }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pattern_report_{market_data.symbol}_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        
        return str(filepath)

    def _calculate_report_metrics(
        self,
        patterns: List[DetectedPattern],
        validation_results: Optional[List[ValidationResult]],
        market_data: MarketData
    ) -> ReportMetrics:
        """Calculate overall metrics for the report."""
        patterns_by_type = {}
        high_confidence_count = 0
        validated_count = 0
        strong_recommendations = 0
        
        for pattern in patterns:
            # Count by type
            patterns_by_type[pattern.pattern_type] = patterns_by_type.get(pattern.pattern_type, 0) + 1
            
            # Count high confidence patterns
            if pattern.confidence >= self.confidence_thresholds['high']:
                high_confidence_count += 1
        
        # Count validation metrics
        if validation_results:
            for result in validation_results:
                if result and hasattr(result, 'overall_score'):
                    validated_count += 1
                    if result.overall_score >= self.validation_score_thresholds['strong']:
                        strong_recommendations += 1
        
        # Calculate average confidence
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns) if patterns else 0
        
        return ReportMetrics(
            total_patterns=len(patterns),
            patterns_by_type=patterns_by_type,
            average_confidence=avg_confidence,
            high_confidence_patterns=high_confidence_count,
            validated_patterns=validated_count,
            strong_recommendations=strong_recommendations,
            analysis_timeframe=f"{market_data.timeframe} from {self._format_datetime(market_data.start_time)} to {self._format_datetime(market_data.end_time)}",
            report_generation_time=datetime.now().isoformat()
        )

    def _calculate_confidence_distribution(self, patterns: List[DetectedPattern]) -> Dict[str, Any]:
        """Calculate confidence score distribution."""
        if not patterns:
            return {}
        
        confidences = [p.confidence for p in patterns]
        
        return {
            'high': len([c for c in confidences if c >= self.confidence_thresholds['high']]),
            'medium': len([c for c in confidences if self.confidence_thresholds['medium'] <= c < self.confidence_thresholds['high']]),
            'low': len([c for c in confidences if c < self.confidence_thresholds['medium']]),
            'average': sum(confidences) / len(confidences),
            'range': {'min': min(confidences), 'max': max(confidences)}
        }

    def _analyze_patterns_by_type(
        self,
        patterns: List[DetectedPattern],
        validation_results: Optional[List[ValidationResult]]
    ) -> Dict[str, Any]:
        """Analyze patterns grouped by type."""
        type_analysis = {}
        
        for i, pattern in enumerate(patterns):
            pattern_type = pattern.pattern_type
            
            if pattern_type not in type_analysis:
                type_analysis[pattern_type] = {
                    'count': 0,
                    'confidences': [],
                    'validation_scores': [],
                    'average_confidence': 0,
                    'average_validation_score': 0
                }
            
            analysis = type_analysis[pattern_type]
            analysis['count'] += 1
            analysis['confidences'].append(pattern.confidence)
            
            if validation_results and i < len(validation_results) and validation_results[i]:
                analysis['validation_scores'].append(validation_results[i].overall_score)
        
        # Calculate averages
        for pattern_type, analysis in type_analysis.items():
            confidences = analysis['confidences']
            validation_scores = analysis['validation_scores']
            
            analysis['average_confidence'] = sum(confidences) / len(confidences)
            analysis['average_validation_score'] = (
                sum(validation_scores) / len(validation_scores) if validation_scores else 0
            )
            
            # Clean up raw lists for export
            del analysis['confidences']
            del analysis['validation_scores']
        
        return type_analysis

    def _generate_recommendations(
        self,
        patterns: List[DetectedPattern],
        validation_results: Optional[List[ValidationResult]]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if not patterns:
            recommendations.append("No patterns detected. Consider adjusting detection parameters or using different timeframes.")
            return recommendations
        
        # High confidence patterns
        high_conf_patterns = [p for p in patterns if p.confidence >= self.confidence_thresholds['high']]
        if high_conf_patterns:
            recommendations.append(f"Found {len(high_conf_patterns)} high-confidence patterns worth closer examination.")
        
        # Validation results analysis
        if validation_results:
            strong_validations = [r for r in validation_results if r and hasattr(r, 'overall_score') and r.overall_score >= self.validation_score_thresholds['strong']]
            if strong_validations:
                recommendations.append(f"Found {len(strong_validations)} patterns with strong validation scores - these are prime candidates for trading signals.")
            
            weak_validations = [r for r in validation_results if r and hasattr(r, 'overall_score') and r.overall_score < self.validation_score_thresholds['weak']]
            if len(weak_validations) > len(patterns) * 0.5:
                recommendations.append("Many patterns have weak validation scores. Consider reviewing detection parameters or market conditions.")
        
        # Pattern diversity
        unique_types = len(set(p.pattern_type for p in patterns))
        if unique_types == 1:
            recommendations.append(f"Only detected {list(set(p.pattern_type for p in patterns))[0]} patterns. Consider expanding pattern detection algorithms.")
        
        return recommendations

    def _extract_pattern_metrics(self, pattern: DetectedPattern) -> Dict[str, float]:
        """Extract key metrics from a pattern."""
        metrics = {
            'confidence': pattern.confidence
        }
        
        # Add pattern-specific metrics if available
        if hasattr(pattern, 'strength'):
            metrics['strength'] = pattern.strength
        
        if hasattr(pattern, 'key_points') and pattern.key_points:
            metrics['key_points_count'] = len(pattern.key_points)
        
        return metrics

    def _format_datetime(self, dt: Any) -> str:
        """Format datetime for report output."""
        if pd.isna(dt):
            return "N/A"
        
        try:
            if isinstance(dt, str):
                return dt
            return pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(dt)

    def _get_recommendation(self, validation_result: ValidationResult) -> str:
        """Get recommendation based on validation result."""
        if not validation_result or not hasattr(validation_result, 'overall_score'):
            return "UNKNOWN"
        
        score = validation_result.overall_score
        if score >= self.validation_score_thresholds['strong']:
            return "STRONG"
        elif score >= self.validation_score_thresholds['moderate']:
            return "MODERATE"
        else:
            return "WEAK"

    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence level."""
        if confidence >= self.confidence_thresholds['high']:
            return "HIGH"
        elif confidence >= self.confidence_thresholds['medium']:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _generate_filename(self, market_data: MarketData, format: str) -> str:
        """Generate filename for report export."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"pattern_report_{market_data.symbol}_{market_data.timeframe}_{timestamp}.{format}"

    def _export_report(self, report: Dict[str, Any], filename: str, format: str) -> None:
        """Export report in specified format."""
        filepath = self.output_dir / filename
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            # Export patterns as CSV
            if 'patterns' in report:
                df = pd.DataFrame(report['patterns'])
                csv_path = filepath.with_suffix('.csv')
                df.to_csv(csv_path, index=False)
        
        elif format.lower() == 'html':
            self._export_html_report(report, filepath)

    def _export_html_report(self, report: Dict[str, Any], filepath: Path) -> None:
        """Export report as HTML."""
        html_content = self._generate_html_report(report)
        
        with open(filepath, 'w') as f:
            f.write(html_content)

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML content for report."""
        # Simple HTML template - could be enhanced with proper templating
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pattern Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4fd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Pattern Detection Report</h1>
                <p>Symbol: {report.get('report_metadata', {}).get('symbol', 'Unknown')}</p>
                <p>Timeframe: {report.get('report_metadata', {}).get('timeframe', 'Unknown')}</p>
                <p>Generated: {report.get('report_metadata', {}).get('generation_time', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h2>Summary Metrics</h2>
                <div class="metric">Total Patterns: {report.get('summary_metrics', {}).get('total_patterns', 0)}</div>
                <div class="metric">Average Confidence: {report.get('summary_metrics', {}).get('average_confidence', 0):.2f}</div>
                <div class="metric">High Confidence: {report.get('summary_metrics', {}).get('high_confidence_patterns', 0)}</div>
            </div>
            
            <div class="section">
                <h2>Detected Patterns</h2>
                <table>
                    <tr>
                        <th>Pattern Type</th>
                        <th>Confidence</th>
                        <th>Start Time</th>
                        <th>End Time</th>
                        <th>Validation</th>
                    </tr>
                    {"".join(self._generate_pattern_row(pattern) for pattern in report.get('patterns', []))}
                </table>
            </div>
        </body>
        </html>
        """
        return html

    def _generate_pattern_row(self, pattern: Dict[str, Any]) -> str:
        """Generate HTML table row for a pattern."""
        return f"""
        <tr>
            <td>{pattern.get('pattern_type', 'Unknown')}</td>
            <td>{pattern.get('confidence', 0):.2f}</td>
            <td>{pattern.get('start_time', 'Unknown')}</td>
            <td>{pattern.get('end_time', 'Unknown')}</td>
            <td>{pattern.get('validation_recommendation', 'N/A')}</td>
        </tr>
        """

    def _generate_statistics_summary(self, overall_stats: Dict, pattern_stats: Dict) -> List[str]:
        """Generate a summary of key statistics."""
        summary = []
        
        summary.append(f"Detected {overall_stats['total_patterns_detected']} patterns across {overall_stats['unique_pattern_types']} different types")
        summary.append(f"Average confidence score: {overall_stats['average_overall_confidence']:.2f}")
        summary.append(f"Most common pattern: {overall_stats['most_common_pattern']}")
        summary.append(f"Highest confidence pattern type: {overall_stats['highest_confidence_pattern']}")
        
        if overall_stats['patterns_per_day'] > 0:
            summary.append(f"Pattern detection rate: {overall_stats['patterns_per_day']:.1f} patterns per day")
        
        return summary