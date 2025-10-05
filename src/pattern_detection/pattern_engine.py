"""
Pattern Detection Engine - Core pattern recognition algorithms.

This module provides comprehensive pattern detection capabilities including:
- Support and resistance level identification
- Pivot point detection system
- Basic trendline detection algorithms
- Triangle pattern recognition (ascending, descending, symmetrical)
- Head and shoulders pattern detection
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    from ..models.market_data import MarketData
    from ..technical_indicators.indicator_engine import TechnicalIndicatorEngine
except ImportError:
    # For testing and development
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.market_data import MarketData
    from technical_indicators.indicator_engine import TechnicalIndicatorEngine


class PatternType(Enum):
    """Enumeration of detectable pattern types."""

    # Triangle Patterns
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"

    # Head and Shoulders Patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"

    # Flag and Pennant Patterns (Phase 2.2)
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    PENNANT = "pennant"

    # Double/Triple Patterns (Phase 2.2)
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"

    # Channel Patterns (Phase 2.2)
    RECTANGLE = "rectangle"
    ASCENDING_CHANNEL = "ascending_channel"
    DESCENDING_CHANNEL = "descending_channel"

    # Support and Resistance
    SUPPORT_LEVEL = "support_level"
    RESISTANCE_LEVEL = "resistance_level"

    # Trendlines
    UPTREND_LINE = "uptrend_line"
    DOWNTREND_LINE = "downtrend_line"


class PatternStrength(Enum):
    """Pattern strength classification."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class PivotPoint:
    """Represents a pivot point in price data."""

    index: int
    timestamp: datetime
    price: float
    pivot_type: str  # 'high' or 'low'
    strength: float  # Relative strength of the pivot


@dataclass
class SupportResistanceLevel:
    """Represents a support or resistance level."""

    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float
    touch_count: int
    first_touch: datetime
    last_touch: datetime
    touches: List[Tuple[datetime, float]]  # List of (timestamp, price) touches


@dataclass
class TrendLine:
    """Represents a detected trendline."""

    start_point: Tuple[datetime, float]
    end_point: Tuple[datetime, float]
    slope: float
    r_squared: float  # Goodness of fit
    touches: List[Tuple[datetime, float]]
    line_type: str  # 'support' or 'resistance'


@dataclass
class DetectedPattern:
    """Represents a detected chart pattern."""

    pattern_type: PatternType
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    confidence_score: float
    strength: PatternStrength
    key_points: List[Tuple[datetime, float]]
    pattern_metrics: Dict[str, Any]
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    volume_confirmation: bool = False
    direction: str = "neutral"  # bullish, bearish, neutral


class PatternDetectionEngine:
    """
    Comprehensive pattern detection engine for technical analysis.

    Provides algorithms to detect various chart patterns including triangles,
    head and shoulders, support/resistance levels, and trendlines.
    """

    def __init__(self, min_pattern_length: int = 10, min_touches: int = 2):
        """Initialize the Pattern Detection Engine."""
        self.min_pattern_length = min_pattern_length
        self.min_touches = min_touches
        self.pivot_points: List[PivotPoint] = []
        self.support_resistance_levels: List[SupportResistanceLevel] = []
        self.trendlines: List[TrendLine] = []
        self.detected_patterns: List[DetectedPattern] = []

    def detect_patterns(self, market_data: MarketData) -> List[DetectedPattern]:
        """
        Detect all available patterns in the given market data.

        Args:
            market_data: MarketData object containing OHLCV data

        Returns:
            List of DetectedPattern objects
        """
        df = market_data.to_dataframe(set_timestamp_index=False)

        if len(df) < self.min_pattern_length:
            return []

        # Clear previous results
        self.pivot_points.clear()
        self.support_resistance_levels.clear()
        self.trendlines.clear()
        self.detected_patterns.clear()

        # Step 1: Identify pivot points
        self._identify_pivot_points(df)

        # Step 2: Detect support and resistance levels
        self._detect_support_resistance_levels()

        # Step 3: Detect trendlines
        self._detect_trendlines()

        # Step 4: Detect triangle patterns
        triangle_patterns = self._detect_triangle_patterns(df)
        self.detected_patterns.extend(triangle_patterns)

        # Step 5: Detect head and shoulders patterns
        hs_patterns = self._detect_head_shoulders_patterns(df)
        self.detected_patterns.extend(hs_patterns)

        return self.detected_patterns

    def _identify_pivot_points(self, df: pd.DataFrame, window: int = 5) -> None:
        """Identify pivot highs and lows in the price data."""
        highs = np.array(df["high"])
        lows = np.array(df["low"])
        timestamps = pd.to_datetime(df["timestamp"]).values

        for i in range(window, len(df) - window):
            # Check for pivot high
            is_pivot_high = all(
                highs[i] >= highs[i - j] and highs[i] >= highs[i + j]
                for j in range(1, window + 1)
            )

            if is_pivot_high:
                strength = self._calculate_pivot_strength(highs, i, window, "high")
                pivot = PivotPoint(
                    index=i,
                    timestamp=timestamps[i],
                    price=highs[i],
                    pivot_type="high",
                    strength=strength,
                )
                self.pivot_points.append(pivot)

            # Check for pivot low
            is_pivot_low = all(
                lows[i] <= lows[i - j] and lows[i] <= lows[i + j]
                for j in range(1, window + 1)
            )

            if is_pivot_low:
                strength = self._calculate_pivot_strength(lows, i, window, "low")
                pivot = PivotPoint(
                    index=i,
                    timestamp=timestamps[i],
                    price=lows[i],
                    pivot_type="low",
                    strength=strength,
                )
                self.pivot_points.append(pivot)

    def _calculate_pivot_strength(
        self,
        prices: Union[np.ndarray, pd.Series],
        index: int,
        window: int,
        pivot_type: str,
    ) -> float:
        """Calculate the relative strength of a pivot point."""
        if pivot_type == "high":
            # Strength based on how much higher than surrounding points
            surrounding = prices[index - window : index + window + 1]
            strength = (prices[index] - np.mean(surrounding)) / np.std(surrounding)
        else:
            # For lows, strength is how much lower than surrounding points
            surrounding = prices[index - window : index + window + 1]
            strength = (np.mean(surrounding) - prices[index]) / np.std(surrounding)

        return max(0.1, min(3.0, strength))  # Clamp between 0.1 and 3.0

    def _detect_support_resistance_levels(self, tolerance: float = 0.02) -> None:
        """Detect support and resistance levels using pivot points."""
        if len(self.pivot_points) < 2:
            return

        # Group pivot points by price level (with tolerance)
        price_clusters = []

        for pivot in self.pivot_points:
            # Find existing cluster or create new one
            found_cluster = False
            for cluster in price_clusters:
                cluster_price = np.mean([p.price for p in cluster])
                if abs(pivot.price - cluster_price) / cluster_price <= tolerance:
                    cluster.append(pivot)
                    found_cluster = True
                    break

            if not found_cluster:
                price_clusters.append([pivot])

        # Convert clusters to support/resistance levels
        for cluster in price_clusters:
            if len(cluster) >= self.min_touches:
                cluster_price = np.mean([p.price for p in cluster])
                cluster_strength = np.mean([p.strength for p in cluster])

                # Determine if it's support or resistance
                high_count = sum(1 for p in cluster if p.pivot_type == "high")
                low_count = sum(1 for p in cluster if p.pivot_type == "low")

                level_type = "resistance" if high_count >= low_count else "support"

                touches = [(p.timestamp, p.price) for p in cluster]
                touches.sort(key=lambda x: x[0])

                level = SupportResistanceLevel(
                    price=float(cluster_price),
                    level_type=level_type,
                    strength=float(cluster_strength),
                    touch_count=len(cluster),
                    first_touch=touches[0][0],
                    last_touch=touches[-1][0],
                    touches=touches,
                )

                self.support_resistance_levels.append(level)

    def _detect_trendlines(self, min_r_squared: float = 0.7) -> None:
        """Detect trendlines connecting pivot points."""
        if len(self.pivot_points) < 2:
            return

        # Separate highs and lows
        pivot_highs = [p for p in self.pivot_points if p.pivot_type == "high"]
        pivot_lows = [p for p in self.pivot_points if p.pivot_type == "low"]

        # Detect resistance trendlines (connecting pivot highs)
        self._find_trendlines(pivot_highs, "resistance", min_r_squared)

        # Detect support trendlines (connecting pivot lows)
        self._find_trendlines(pivot_lows, "support", min_r_squared)

    def _find_trendlines(
        self, pivots: List[PivotPoint], line_type: str, min_r_squared: float
    ) -> None:
        """Find trendlines connecting pivot points of the same type."""
        if len(pivots) < 2:
            return

        # Sort pivots by time
        pivots.sort(key=lambda p: p.timestamp)

        # Try combinations of pivot points for trendlines
        for i in range(len(pivots)):
            for j in range(i + 2, min(i + 6, len(pivots))):  # Look ahead up to 5 points

                # Get pivots in range for trendline calculation
                intermediate_pivots = pivots[i : j + 1]

                if len(intermediate_pivots) < 3:
                    continue

                # Calculate linear regression
                x_values = np.array([p.index for p in intermediate_pivots])
                y_values = np.array([p.price for p in intermediate_pivots])

                # Linear regression
                slope, intercept = np.polyfit(x_values, y_values, 1)
                y_pred = slope * x_values + intercept

                # Calculate R-squared
                ss_res = np.sum((y_values - y_pred) ** 2)
                ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                if (
                    r_squared >= min_r_squared
                    and len(intermediate_pivots) >= self.min_touches
                ):
                    start_pivot = intermediate_pivots[0]
                    end_pivot = intermediate_pivots[-1]

                    trendline = TrendLine(
                        start_point=(start_pivot.timestamp, start_pivot.price),
                        end_point=(end_pivot.timestamp, end_pivot.price),
                        slope=slope,
                        r_squared=r_squared,
                        touches=[(p.timestamp, p.price) for p in intermediate_pivots],
                        line_type=line_type,
                    )

                    self.trendlines.append(trendline)

    def _detect_triangle_patterns(self, df: pd.DataFrame) -> List[DetectedPattern]:
        """Detect triangle patterns (ascending, descending, symmetrical)."""
        patterns = []

        if len(self.trendlines) < 2:
            return patterns

        # Look for convergent trendlines
        support_lines = [tl for tl in self.trendlines if tl.line_type == "support"]
        resistance_lines = [
            tl for tl in self.trendlines if tl.line_type == "resistance"
        ]

        for support_line in support_lines:
            for resistance_line in resistance_lines:
                # Check if trendlines converge and overlap in time
                pattern = self._analyze_triangle_convergence(
                    support_line, resistance_line, df
                )

                if pattern:
                    patterns.append(pattern)

        return patterns

    def _analyze_triangle_convergence(
        self, support_line: TrendLine, resistance_line: TrendLine, df: pd.DataFrame
    ) -> Optional[DetectedPattern]:
        """Analyze if two trendlines form a valid triangle pattern."""

        # Check time overlap
        support_start = min(touch[0] for touch in support_line.touches)
        support_end = max(touch[0] for touch in support_line.touches)
        resistance_start = min(touch[0] for touch in resistance_line.touches)
        resistance_end = max(touch[0] for touch in resistance_line.touches)

        # Calculate overlap
        overlap_start = max(support_start, resistance_start)
        overlap_end = min(support_end, resistance_end)

        if overlap_start >= overlap_end:
            return None  # No time overlap

        # Determine triangle type based on slopes
        support_slope = support_line.slope
        resistance_slope = resistance_line.slope

        triangle_type = None
        confidence_base = 0.6

        # Normalize slopes for comparison (price change per time unit)
        time_delta = overlap_end - overlap_start
        if isinstance(time_delta, pd.Timedelta):
            time_range = time_delta.total_seconds() / 86400  # days
        else:
            # Handle numpy.timedelta64 objects
            time_range = pd.Timedelta(time_delta).total_seconds() / 86400  # days
        if time_range > 0:
            support_slope_norm = support_slope * time_range
            resistance_slope_norm = resistance_slope * time_range
        else:
            return None

        # Ascending triangle: flat resistance, rising support
        if abs(resistance_slope_norm) < 0.02 and support_slope_norm > 0.01:
            triangle_type = PatternType.ASCENDING_TRIANGLE
            confidence_base = 0.7

        # Descending triangle: flat support, falling resistance
        elif abs(support_slope_norm) < 0.02 and resistance_slope_norm < -0.01:
            triangle_type = PatternType.DESCENDING_TRIANGLE
            confidence_base = 0.7

        # Symmetrical triangle: converging lines
        elif (
            support_slope_norm > 0
            and resistance_slope_norm < 0
            and abs(support_slope_norm + resistance_slope_norm) < 0.03
        ):
            triangle_type = PatternType.SYMMETRICAL_TRIANGLE
            confidence_base = 0.6

        if not triangle_type:
            return None

        # Calculate confidence based on trendline quality
        avg_r_squared = (support_line.r_squared + resistance_line.r_squared) / 2
        touch_count = len(support_line.touches) + len(resistance_line.touches)

        confidence_score = confidence_base * avg_r_squared * min(1.0, touch_count / 6)

        # Collect key points
        key_points = []
        key_points.extend(support_line.touches)
        key_points.extend(resistance_line.touches)
        key_points.sort(key=lambda x: x[0])

        # Calculate pattern metrics
        pattern_metrics = {
            "support_slope": support_slope,
            "resistance_slope": resistance_slope,
            "support_r_squared": support_line.r_squared,
            "resistance_r_squared": resistance_line.r_squared,
            "touch_count": touch_count,
            "convergence_angle": abs(support_slope_norm - resistance_slope_norm),
            "pattern_duration_days": time_range,
        }

        # Determine strength
        strength = PatternStrength.WEAK
        if confidence_score > 0.8:
            strength = PatternStrength.VERY_STRONG
        elif confidence_score > 0.7:
            strength = PatternStrength.STRONG
        elif confidence_score > 0.6:
            strength = PatternStrength.MODERATE

        return DetectedPattern(
            pattern_type=triangle_type,
            symbol=(
                df.get("symbol", ["UNKNOWN"])[0]
                if "symbol" in df.columns
                else "UNKNOWN"
            ),
            timeframe="unknown",
            start_time=overlap_start,
            end_time=overlap_end,
            confidence_score=confidence_score,
            strength=strength,
            key_points=key_points,
            pattern_metrics=pattern_metrics,
        )

    def _detect_head_shoulders_patterns(
        self, df: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect head and shoulders patterns."""
        patterns = []

        if len(self.pivot_points) < 5:
            return patterns

        # Separate highs and lows
        pivot_highs = [p for p in self.pivot_points if p.pivot_type == "high"]
        pivot_lows = [p for p in self.pivot_points if p.pivot_type == "low"]

        # Sort by time
        pivot_highs.sort(key=lambda p: p.timestamp)
        pivot_lows.sort(key=lambda p: p.timestamp)

        # Look for head and shoulders in highs (regular pattern)
        hs_patterns = self._find_head_shoulders_in_pivots(pivot_highs, "regular", df)
        patterns.extend(hs_patterns)

        # Look for inverse head and shoulders in lows
        ihs_patterns = self._find_head_shoulders_in_pivots(pivot_lows, "inverse", df)
        patterns.extend(ihs_patterns)

        return patterns

    def _find_head_shoulders_in_pivots(
        self, pivots: List[PivotPoint], pattern_variant: str, df: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Find head and shoulders patterns in pivot points."""
        patterns = []

        if len(pivots) < 3:
            return patterns

        # Look for sequences of 3 pivots that could form head and shoulders
        for i in range(len(pivots) - 2):
            left_shoulder = pivots[i]
            head = pivots[i + 1]
            right_shoulder = pivots[i + 2]

            # For regular H&S: head should be highest, shoulders similar height
            # For inverse H&S: head should be lowest, shoulders similar depth

            if pattern_variant == "regular":
                # Check height relationships
                if (
                    head.price > left_shoulder.price
                    and head.price > right_shoulder.price
                    and abs(left_shoulder.price - right_shoulder.price)
                    / max(left_shoulder.price, right_shoulder.price)
                    < 0.1
                ):

                    pattern = self._create_head_shoulders_pattern(
                        [left_shoulder, head, right_shoulder],
                        PatternType.HEAD_AND_SHOULDERS,
                        df,
                    )
                    if pattern:
                        patterns.append(pattern)

            else:  # inverse
                # Check depth relationships
                if (
                    head.price < left_shoulder.price
                    and head.price < right_shoulder.price
                    and abs(left_shoulder.price - right_shoulder.price)
                    / max(left_shoulder.price, right_shoulder.price)
                    < 0.1
                ):

                    pattern = self._create_head_shoulders_pattern(
                        [left_shoulder, head, right_shoulder],
                        PatternType.INVERSE_HEAD_AND_SHOULDERS,
                        df,
                    )
                    if pattern:
                        patterns.append(pattern)

        return patterns

    def _create_head_shoulders_pattern(
        self, key_pivots: List[PivotPoint], pattern_type: PatternType, df: pd.DataFrame
    ) -> Optional[DetectedPattern]:
        """Create a head and shoulders pattern from key pivot points."""

        if len(key_pivots) < 3:
            return None

        left_shoulder, head, right_shoulder = key_pivots

        # Calculate neckline (simplified - connect shoulders)
        neckline_price = (left_shoulder.price + right_shoulder.price) / 2

        # Calculate confidence based on pattern quality
        shoulder_symmetry = 1 - abs(left_shoulder.price - right_shoulder.price) / max(
            left_shoulder.price, right_shoulder.price
        )
        head_prominence = abs(head.price - neckline_price) / neckline_price

        confidence_score = min(0.9, shoulder_symmetry * head_prominence * 3)

        # Minimum confidence threshold
        if confidence_score < 0.3:
            return None

        # Determine strength
        strength = PatternStrength.WEAK
        if confidence_score > 0.8:
            strength = PatternStrength.VERY_STRONG
        elif confidence_score > 0.7:
            strength = PatternStrength.STRONG
        elif confidence_score > 0.5:
            strength = PatternStrength.MODERATE

        # Pattern metrics
        pattern_metrics = {
            "neckline_price": neckline_price,
            "head_height": abs(head.price - neckline_price),
            "shoulder_symmetry": shoulder_symmetry,
            "head_prominence": head_prominence,
            "left_shoulder_height": abs(left_shoulder.price - neckline_price),
            "right_shoulder_height": abs(right_shoulder.price - neckline_price),
        }

        # Calculate target price (pattern height projected from neckline)
        pattern_height = abs(head.price - neckline_price)
        if pattern_type == PatternType.HEAD_AND_SHOULDERS:
            target_price = neckline_price - pattern_height  # Bearish target
        else:
            target_price = neckline_price + pattern_height  # Bullish target

        key_points = [(p.timestamp, p.price) for p in key_pivots]

        return DetectedPattern(
            pattern_type=pattern_type,
            symbol=(
                df.get("symbol", ["UNKNOWN"])[0]
                if "symbol" in df.columns
                else "UNKNOWN"
            ),
            timeframe="unknown",
            start_time=left_shoulder.timestamp,
            end_time=right_shoulder.timestamp,
            confidence_score=confidence_score,
            strength=strength,
            key_points=key_points,
            pattern_metrics=pattern_metrics,
            target_price=target_price,
        )

    def get_support_resistance_levels(self) -> List[SupportResistanceLevel]:
        """Get detected support and resistance levels."""
        return self.support_resistance_levels.copy()

    def get_trendlines(self) -> List[TrendLine]:
        """Get detected trendlines."""
        return self.trendlines.copy()

    def get_pivot_points(self) -> List[PivotPoint]:
        """Get identified pivot points."""
        return self.pivot_points.copy()

    def get_pattern_summary(self) -> Dict[str, int]:
        """Get summary of detected patterns by type."""
        summary = {}
        for pattern in self.detected_patterns:
            pattern_name = pattern.pattern_type.value
            summary[pattern_name] = summary.get(pattern_name, 0) + 1

        return summary

    # Legacy methods for backward compatibility
    def calculate_pattern_strength(self, pattern: Dict, context: Dict) -> float:
        """Calculate pattern strength (legacy method)."""
        base_score = 0.6
        context_adjustment = 0.1 if context.get("trending", False) else 0.0
        return min(1.0, base_score + context_adjustment)

    def validate_pattern(self, pattern: Dict, multiple_timeframes: List[str]) -> Dict:
        """Cross-timeframe pattern validation (legacy method)."""
        return {
            "is_valid": True,
            "confidence_adjustment": 1.0,
            "timeframe_confluence": len(multiple_timeframes) * 0.1,
        }
