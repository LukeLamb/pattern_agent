"""
Multi-Timeframe Analysis System

This module provides comprehensive multi-timeframe pattern analysis including:
- Timeframe hierarchy and weighting
- Cross-timeframe pattern validation
- Trend alignment analysis
- Signal strength aggregation
- Confluence scoring
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import pandas as pd
import numpy as np

try:
    from ..models.market_data import MarketData
    from ..models.pattern import PatternType, PatternDirection
    from ..pattern_detection.pattern_engine import DetectedPattern
    from ..technical_indicators.indicator_engine import TechnicalIndicatorEngine
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.market_data import MarketData
    from models.pattern import PatternType, PatternDirection
    from pattern_detection.pattern_engine import DetectedPattern
    from technical_indicators.indicator_engine import TechnicalIndicatorEngine


class Timeframe(str, Enum):
    """Supported timeframe types."""
    ONE_MIN = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    THIRTY_MIN = "30min"
    ONE_HOUR = "1hr"
    FOUR_HOUR = "4hr"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class TimeframeHierarchy:
    """
    Defines the relationship and weights between timeframes.

    Higher timeframes have more weight in decision-making.
    """

    # Timeframe importance weights (0.0 to 1.0)
    weights: Dict[Timeframe, float] = field(default_factory=lambda: {
        Timeframe.MONTHLY: 1.0,
        Timeframe.WEEKLY: 0.95,
        Timeframe.DAILY: 0.9,
        Timeframe.FOUR_HOUR: 0.75,
        Timeframe.ONE_HOUR: 0.6,
        Timeframe.THIRTY_MIN: 0.5,
        Timeframe.FIFTEEN_MIN: 0.4,
        Timeframe.FIVE_MIN: 0.3,
        Timeframe.ONE_MIN: 0.2,
    })

    # Conversion factors (in minutes)
    timeframe_minutes: Dict[Timeframe, int] = field(default_factory=lambda: {
        Timeframe.ONE_MIN: 1,
        Timeframe.FIVE_MIN: 5,
        Timeframe.FIFTEEN_MIN: 15,
        Timeframe.THIRTY_MIN: 30,
        Timeframe.ONE_HOUR: 60,
        Timeframe.FOUR_HOUR: 240,
        Timeframe.DAILY: 1440,
        Timeframe.WEEKLY: 10080,
        Timeframe.MONTHLY: 43200,
    })

    def get_weight(self, timeframe: str) -> float:
        """Get the weight for a given timeframe."""
        try:
            tf = Timeframe(timeframe.lower())
            return self.weights.get(tf, 0.5)
        except ValueError:
            return 0.5  # Default weight for unknown timeframes

    def get_higher_timeframes(self, timeframe: str) -> List[Timeframe]:
        """Get all timeframes higher than the given timeframe."""
        try:
            tf = Timeframe(timeframe.lower())
            current_minutes = self.timeframe_minutes[tf]
            return [
                higher_tf
                for higher_tf, minutes in self.timeframe_minutes.items()
                if minutes > current_minutes
            ]
        except (ValueError, KeyError):
            return []

    def get_lower_timeframes(self, timeframe: str) -> List[Timeframe]:
        """Get all timeframes lower than the given timeframe."""
        try:
            tf = Timeframe(timeframe.lower())
            current_minutes = self.timeframe_minutes[tf]
            return [
                lower_tf
                for lower_tf, minutes in self.timeframe_minutes.items()
                if minutes < current_minutes
            ]
        except (ValueError, KeyError):
            return []

    def is_higher_timeframe(self, tf1: str, tf2: str) -> bool:
        """Check if tf1 is a higher timeframe than tf2."""
        try:
            minutes1 = self.timeframe_minutes[Timeframe(tf1.lower())]
            minutes2 = self.timeframe_minutes[Timeframe(tf2.lower())]
            return minutes1 > minutes2
        except (ValueError, KeyError):
            return False


@dataclass
class TimeframeAlignment:
    """
    Represents trend/pattern alignment across multiple timeframes.
    """

    aligned_timeframes: List[Timeframe] = field(default_factory=list)
    conflicting_timeframes: List[Timeframe] = field(default_factory=list)
    alignment_score: float = 0.0  # 0.0 to 1.0
    dominant_direction: Optional[PatternDirection] = None

    def is_aligned(self, threshold: float = 0.7) -> bool:
        """Check if timeframes are sufficiently aligned."""
        return self.alignment_score >= threshold


@dataclass
class ConfluenceScore:
    """
    Represents confluence (agreement) across multiple timeframes.

    Higher scores indicate stronger multi-timeframe agreement.
    """

    overall_score: float = 0.0  # 0.0 to 1.0
    pattern_confluence: float = 0.0  # Pattern type agreement
    direction_confluence: float = 0.0  # Trend direction agreement
    strength_confluence: float = 0.0  # Signal strength agreement
    timeframe_count: int = 0
    contributing_timeframes: List[Timeframe] = field(default_factory=list)

    details: Dict[str, any] = field(default_factory=dict)


@dataclass
class MultiTimeframePattern:
    """
    Represents a pattern detected across multiple timeframes.
    """

    primary_timeframe: str
    primary_pattern: DetectedPattern
    supporting_patterns: Dict[str, DetectedPattern] = field(default_factory=dict)

    alignment: TimeframeAlignment = field(default_factory=TimeframeAlignment)
    confluence: ConfluenceScore = field(default_factory=ConfluenceScore)

    aggregated_confidence: float = 0.0
    recommendation_strength: str = "WEAK"  # WEAK, MODERATE, STRONG, VERY_STRONG


class MultiTimeframeAnalyzer:
    """
    Multi-Timeframe Pattern Analysis System

    Analyzes patterns across multiple timeframes to generate high-confidence
    trading signals through confluence and alignment validation.
    """

    def __init__(
        self,
        hierarchy: Optional[TimeframeHierarchy] = None,
        min_confluence_threshold: float = 0.6,
        min_alignment_threshold: float = 0.7,
    ):
        """
        Initialize the multi-timeframe analyzer.

        Args:
            hierarchy: Timeframe hierarchy configuration
            min_confluence_threshold: Minimum confluence score for valid signals
            min_alignment_threshold: Minimum alignment score for trend agreement
        """
        self.hierarchy = hierarchy or TimeframeHierarchy()
        self.min_confluence_threshold = min_confluence_threshold
        self.min_alignment_threshold = min_alignment_threshold
        self.indicator_engine = TechnicalIndicatorEngine()

    def analyze_pattern_confluence(
        self,
        primary_timeframe: str,
        primary_pattern: DetectedPattern,
        market_data_by_timeframe: Dict[str, MarketData],
        detected_patterns_by_timeframe: Optional[Dict[str, List[DetectedPattern]]] = None,
    ) -> MultiTimeframePattern:
        """
        Analyze pattern confluence across multiple timeframes.

        Args:
            primary_timeframe: The main timeframe being analyzed
            primary_pattern: The pattern detected on primary timeframe
            market_data_by_timeframe: Market data for each timeframe
            detected_patterns_by_timeframe: Pre-detected patterns for each timeframe

        Returns:
            MultiTimeframePattern with confluence analysis
        """
        # Initialize multi-timeframe pattern
        mtf_pattern = MultiTimeframePattern(
            primary_timeframe=primary_timeframe,
            primary_pattern=primary_pattern,
        )

        # Find supporting patterns on other timeframes
        supporting_patterns = self._find_supporting_patterns(
            primary_pattern,
            primary_timeframe,
            detected_patterns_by_timeframe or {}
        )
        mtf_pattern.supporting_patterns = supporting_patterns

        # Analyze trend alignment
        alignment = self._analyze_trend_alignment(
            primary_timeframe,
            primary_pattern,
            market_data_by_timeframe
        )
        mtf_pattern.alignment = alignment

        # Calculate confluence score
        confluence = self._calculate_confluence_score(
            primary_timeframe,
            primary_pattern,
            supporting_patterns,
            alignment
        )
        mtf_pattern.confluence = confluence

        # Aggregate confidence across timeframes
        mtf_pattern.aggregated_confidence = self._aggregate_confidence(
            primary_pattern,
            supporting_patterns,
            confluence
        )

        # Determine recommendation strength
        mtf_pattern.recommendation_strength = self._determine_recommendation_strength(
            mtf_pattern.aggregated_confidence,
            confluence.overall_score,
            alignment.alignment_score
        )

        return mtf_pattern

    def _find_supporting_patterns(
        self,
        primary_pattern: DetectedPattern,
        primary_timeframe: str,
        patterns_by_timeframe: Dict[str, List[DetectedPattern]]
    ) -> Dict[str, DetectedPattern]:
        """
        Find patterns on other timeframes that support the primary pattern.

        Looks for same pattern type or compatible patterns within the
        primary pattern's timeframe.
        """
        supporting = {}

        for timeframe, patterns in patterns_by_timeframe.items():
            if timeframe == primary_timeframe:
                continue

            # Find best matching pattern
            best_match = None
            best_score = 0.0

            for pattern in patterns:
                # Check if patterns overlap in time
                if not self._patterns_overlap(primary_pattern, pattern):
                    continue

                # Calculate similarity score
                similarity = self._calculate_pattern_similarity(
                    primary_pattern, pattern
                )

                if similarity > best_score:
                    best_score = similarity
                    best_match = pattern

            # Add if similarity is high enough
            if best_match and best_score > 0.5:
                supporting[timeframe] = best_match

        return supporting

    def _patterns_overlap(
        self, pattern1: DetectedPattern, pattern2: DetectedPattern
    ) -> bool:
        """Check if two patterns overlap in time."""
        # Simple overlap check - patterns should occur around same time
        time_diff = abs((pattern1.start_time - pattern2.start_time).total_seconds())

        # Allow patterns to be within timeframe of each other
        # Adjust this threshold based on timeframe differences
        max_diff = timedelta(days=30).total_seconds()

        return time_diff <= max_diff

    def _calculate_pattern_similarity(
        self, pattern1: DetectedPattern, pattern2: DetectedPattern
    ) -> float:
        """
        Calculate similarity between two patterns.

        Returns score from 0.0 to 1.0.
        """
        score = 0.0

        # Same pattern type gets highest score
        if pattern1.pattern_type == pattern2.pattern_type:
            score += 0.5
        # Compatible pattern types (e.g., both bullish) get partial score
        elif self._are_compatible_patterns(pattern1, pattern2):
            score += 0.3

        # Similar confidence scores
        conf_diff = abs(pattern1.confidence_score - pattern2.confidence_score)
        score += (1.0 - conf_diff) * 0.3

        # Similar directions
        if pattern1.direction == pattern2.direction:
            score += 0.2

        return min(score, 1.0)

    def _are_compatible_patterns(
        self, pattern1: DetectedPattern, pattern2: DetectedPattern
    ) -> bool:
        """Check if patterns are compatible (same directional bias)."""
        # Patterns with same direction are compatible
        if pattern1.direction == pattern2.direction:
            return True

        # Specific pattern compatibility logic can be added here
        # e.g., ascending triangle + bullish flag are compatible

        return False

    def _analyze_trend_alignment(
        self,
        primary_timeframe: str,
        primary_pattern: DetectedPattern,
        market_data_by_timeframe: Dict[str, MarketData]
    ) -> TimeframeAlignment:
        """
        Analyze trend alignment across multiple timeframes.

        Uses moving averages and trend direction to determine alignment.
        """
        alignment = TimeframeAlignment()

        direction_counts = {
            PatternDirection.BULLISH: 0,
            PatternDirection.BEARISH: 0,
            PatternDirection.NEUTRAL: 0,
        }

        total_weight = 0.0
        aligned_weight = 0.0

        for timeframe, market_data in market_data_by_timeframe.items():
            try:
                # Get trend direction for this timeframe
                trend_direction = self._determine_trend_direction(market_data)

                # Get timeframe weight
                tf_weight = self.hierarchy.get_weight(timeframe)
                total_weight += tf_weight

                # Count direction
                direction_counts[trend_direction] += 1

                # Check if aligned with primary pattern
                if trend_direction == primary_pattern.direction:
                    alignment.aligned_timeframes.append(Timeframe(timeframe))
                    aligned_weight += tf_weight
                else:
                    alignment.conflicting_timeframes.append(Timeframe(timeframe))

            except Exception:
                # Skip timeframes that cause errors
                continue

        # Calculate alignment score (weighted)
        if total_weight > 0:
            alignment.alignment_score = aligned_weight / total_weight

        # Determine dominant direction
        dominant = max(direction_counts.items(), key=lambda x: x[1])
        alignment.dominant_direction = dominant[0]

        return alignment

    def _determine_trend_direction(self, market_data: MarketData) -> PatternDirection:
        """
        Determine trend direction using moving averages.

        Uses SMA 20/50/200 alignment.
        """
        try:
            df = market_data.to_dataframe()

            if len(df) < 200:
                return PatternDirection.NEUTRAL

            # Calculate moving averages
            indicators = self.indicator_engine.calculate_all_indicators(market_data)

            # Get latest values
            sma_20 = indicators.get('sma_20', pd.Series()).iloc[-1] if 'sma_20' in indicators else None
            sma_50 = indicators.get('sma_50', pd.Series()).iloc[-1] if 'sma_50' in indicators else None
            current_price = df['close'].iloc[-1]

            if sma_20 is None or sma_50 is None:
                return PatternDirection.NEUTRAL

            # Bullish: Price > SMA20 > SMA50
            if current_price > sma_20 and sma_20 > sma_50:
                return PatternDirection.BULLISH
            # Bearish: Price < SMA20 < SMA50
            elif current_price < sma_20 and sma_20 < sma_50:
                return PatternDirection.BEARISH
            else:
                return PatternDirection.NEUTRAL

        except Exception:
            return PatternDirection.NEUTRAL

    def _calculate_confluence_score(
        self,
        primary_timeframe: str,
        primary_pattern: DetectedPattern,
        supporting_patterns: Dict[str, DetectedPattern],
        alignment: TimeframeAlignment
    ) -> ConfluenceScore:
        """
        Calculate overall confluence score across timeframes.

        Considers pattern agreement, direction agreement, and strength.
        """
        confluence = ConfluenceScore()

        # Count total timeframes involved
        confluence.timeframe_count = 1 + len(supporting_patterns)
        confluence.contributing_timeframes = [Timeframe(primary_timeframe)]
        confluence.contributing_timeframes.extend([
            Timeframe(tf) for tf in supporting_patterns.keys()
        ])

        # Pattern confluence: How many timeframes show similar patterns
        if confluence.timeframe_count > 1:
            confluence.pattern_confluence = len(supporting_patterns) / (confluence.timeframe_count - 1)
        else:
            confluence.pattern_confluence = 0.0

        # Direction confluence: From alignment analysis
        confluence.direction_confluence = alignment.alignment_score

        # Strength confluence: Average confidence across timeframes
        total_confidence = primary_pattern.confidence_score
        total_weight = self.hierarchy.get_weight(primary_timeframe)

        for tf, pattern in supporting_patterns.items():
            weight = self.hierarchy.get_weight(tf)
            total_confidence += pattern.confidence_score * weight
            total_weight += weight

        if total_weight > 0:
            confluence.strength_confluence = total_confidence / total_weight

        # Overall confluence score (weighted combination)
        confluence.overall_score = (
            confluence.pattern_confluence * 0.35 +
            confluence.direction_confluence * 0.40 +
            confluence.strength_confluence * 0.25
        )

        # Store details
        confluence.details = {
            'pattern_types': {primary_timeframe: primary_pattern.pattern_type.value},
            'directions': {primary_timeframe: primary_pattern.direction if isinstance(primary_pattern.direction, str) else primary_pattern.direction.value},
            'confidences': {primary_timeframe: primary_pattern.confidence_score},
        }

        for tf, pattern in supporting_patterns.items():
            confluence.details['pattern_types'][tf] = pattern.pattern_type.value
            confluence.details['directions'][tf] = pattern.direction if isinstance(pattern.direction, str) else pattern.direction.value
            confluence.details['confidences'][tf] = pattern.confidence_score

        return confluence

    def _aggregate_confidence(
        self,
        primary_pattern: DetectedPattern,
        supporting_patterns: Dict[str, DetectedPattern],
        confluence: ConfluenceScore
    ) -> float:
        """
        Aggregate confidence scores across timeframes.

        Uses weighted average with confluence boost.
        """
        # Start with primary pattern confidence
        total_confidence = primary_pattern.confidence_score
        total_weight = 1.0

        # Add supporting patterns with their timeframe weights
        for tf, pattern in supporting_patterns.items():
            weight = self.hierarchy.get_weight(tf)
            total_confidence += pattern.confidence_score * weight
            total_weight += weight

        # Base aggregated confidence
        base_confidence = total_confidence / total_weight if total_weight > 0 else 0.0

        # Apply confluence boost (up to 20% increase for high confluence)
        confluence_boost = confluence.overall_score * 0.2
        aggregated = base_confidence * (1.0 + confluence_boost)

        # Cap at 1.0
        return min(aggregated, 1.0)

    def _determine_recommendation_strength(
        self,
        aggregated_confidence: float,
        confluence_score: float,
        alignment_score: float
    ) -> str:
        """
        Determine recommendation strength based on multiple factors.

        Returns: WEAK, MODERATE, STRONG, or VERY_STRONG
        """
        # VERY_STRONG: High confidence + high confluence + high alignment
        if (aggregated_confidence >= 0.8 and
            confluence_score >= 0.75 and
            alignment_score >= 0.8):
            return "VERY_STRONG"

        # STRONG: Good confidence + good confluence/alignment
        if (aggregated_confidence >= 0.7 and
            confluence_score >= 0.6 and
            alignment_score >= 0.6):
            return "STRONG"

        # MODERATE: Moderate confidence or partial agreement
        if (aggregated_confidence >= 0.5 and
            confluence_score >= 0.4 and
            alignment_score >= 0.5):
            return "MODERATE"

        # WEAK: Everything else
        return "WEAK"

    def get_optimal_entry_timeframe(
        self,
        mtf_pattern: MultiTimeframePattern,
        market_data_by_timeframe: Dict[str, MarketData]
    ) -> Optional[str]:
        """
        Determine the optimal timeframe for entry based on multi-timeframe analysis.

        Generally, use higher timeframe for direction and lower for entry timing.
        """
        # If we have a strong higher timeframe pattern, use lower timeframe for entry
        primary_tf = mtf_pattern.primary_timeframe

        # Get lower timeframes
        lower_tfs = self.hierarchy.get_lower_timeframes(primary_tf)

        # Find the lowest timeframe with data that still shows alignment
        for tf in reversed(lower_tfs):  # Start from highest of the lower timeframes
            if tf.value in market_data_by_timeframe:
                # Check if this timeframe is aligned
                if tf in mtf_pattern.alignment.aligned_timeframes:
                    return tf.value

        # Fallback to primary timeframe
        return primary_tf
