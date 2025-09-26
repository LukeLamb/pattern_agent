"""
Triangle Pattern Detection - Specialized algorithms for triangle patterns.

This module provides detailed triangle pattern detection including:
- Ascending triangles (bullish continuation)
- Descending triangles (bearish continuation)
- Symmetrical triangles (neutral breakout patterns)
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .pattern_engine import (
    DetectedPattern,
    PatternType,
    PatternStrength,
    TrendLine,
    PivotPoint,
)


class TrianglePatternDetector:
    """
    Specialized detector for triangle chart patterns.

    Provides advanced triangle pattern recognition with detailed
    validation criteria and volume confirmation analysis.
    """

    def __init__(self, min_touches: int = 4, min_duration_days: int = 10):
        """Initialize Triangle Pattern Detector."""
        self.min_touches = min_touches
        self.min_duration_days = min_duration_days

    def detect_ascending_triangles(
        self,
        df: pd.DataFrame,
        support_lines: List[TrendLine],
        resistance_lines: List[TrendLine],
    ) -> List[DetectedPattern]:
        """
        Detect ascending triangle patterns.

        Ascending triangles feature:
        - Horizontal resistance level (flat top)
        - Rising support trendline
        - Bullish bias with upward breakout expected
        """
        patterns = []

        for resistance_line in resistance_lines:
            # Check if resistance is relatively flat
            if abs(resistance_line.slope) > 0.05:  # Too steep to be flat
                continue

            for support_line in support_lines:
                # Check if support is rising
                if support_line.slope <= 0.01:  # Not rising enough
                    continue

                pattern = self._validate_ascending_triangle(
                    support_line, resistance_line, df
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    def detect_descending_triangles(
        self,
        df: pd.DataFrame,
        support_lines: List[TrendLine],
        resistance_lines: List[TrendLine],
    ) -> List[DetectedPattern]:
        """
        Detect descending triangle patterns.

        Descending triangles feature:
        - Horizontal support level (flat bottom)
        - Falling resistance trendline
        - Bearish bias with downward breakout expected
        """
        patterns = []

        for support_line in support_lines:
            # Check if support is relatively flat
            if abs(support_line.slope) > 0.05:  # Too steep to be flat
                continue

            for resistance_line in resistance_lines:
                # Check if resistance is falling
                if resistance_line.slope >= -0.01:  # Not falling enough
                    continue

                pattern = self._validate_descending_triangle(
                    support_line, resistance_line, df
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    def detect_symmetrical_triangles(
        self,
        df: pd.DataFrame,
        support_lines: List[TrendLine],
        resistance_lines: List[TrendLine],
    ) -> List[DetectedPattern]:
        """
        Detect symmetrical triangle patterns.

        Symmetrical triangles feature:
        - Rising support trendline
        - Falling resistance trendline
        - Converging lines forming triangle
        - Neutral bias - can break either direction
        """
        patterns = []

        for support_line in support_lines:
            if support_line.slope <= 0:  # Must be rising
                continue

            for resistance_line in resistance_lines:
                if resistance_line.slope >= 0:  # Must be falling
                    continue

                # Check convergence
                if abs(support_line.slope + resistance_line.slope) > 0.1:
                    continue  # Lines not converging properly

                pattern = self._validate_symmetrical_triangle(
                    support_line, resistance_line, df
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _validate_ascending_triangle(
        self, support_line: TrendLine, resistance_line: TrendLine, df: pd.DataFrame
    ) -> Optional[DetectedPattern]:
        """Validate and create ascending triangle pattern."""

        # Check time overlap
        overlap_start, overlap_end, time_range = self._calculate_time_overlap(
            support_line, resistance_line
        )

        if not overlap_start or time_range < self.min_duration_days:
            return None

        # Validate touch count
        total_touches = len(support_line.touches) + len(resistance_line.touches)
        if total_touches < self.min_touches:
            return None

        # Check volume confirmation
        volume_confirmed = (
            self._check_volume_confirmation(df, overlap_start, overlap_end)
            if overlap_start and overlap_end
            else False
        )

        # Calculate confidence
        resistance_flatness = max(0, 1 - abs(resistance_line.slope) * 10)
        support_rising = min(1, support_line.slope * 20)
        quality_score = (support_line.r_squared + resistance_line.r_squared) / 2

        confidence_score = (
            resistance_flatness * 0.3 + support_rising * 0.3 + quality_score * 0.4
        )

        # Apply volume bonus
        if volume_confirmed:
            confidence_score *= 1.1

        # Determine strength
        strength = self._determine_pattern_strength(confidence_score)

        # Calculate target price (height of triangle projected upward)
        triangle_height = self._calculate_triangle_height(support_line, resistance_line)
        resistance_level = np.mean([touch[1] for touch in resistance_line.touches])
        target_price = resistance_level + triangle_height

        # Calculate metrics
        pattern_metrics = {
            "resistance_flatness": resistance_flatness,
            "support_slope": support_line.slope,
            "resistance_slope": resistance_line.slope,
            "triangle_height": triangle_height,
            "touch_count": total_touches,
            "volume_confirmed": volume_confirmed,
            "duration_days": time_range,
        }

        # Collect key points
        key_points = []
        key_points.extend(support_line.touches)
        key_points.extend(resistance_line.touches)
        key_points.sort(key=lambda x: x[0])

        return DetectedPattern(
            pattern_type=PatternType.ASCENDING_TRIANGLE,
            symbol=(
                df.get("symbol", ["UNKNOWN"])[0]
                if "symbol" in df.columns
                else "UNKNOWN"
            ),
            timeframe="unknown",
            start_time=overlap_start or datetime.now(),
            end_time=overlap_end or datetime.now(),
            confidence_score=min(1.0, confidence_score),
            strength=strength,
            key_points=key_points,
            pattern_metrics=pattern_metrics,
            target_price=float(target_price),
            volume_confirmation=volume_confirmed,
        )

    def _validate_descending_triangle(
        self, support_line: TrendLine, resistance_line: TrendLine, df: pd.DataFrame
    ) -> Optional[DetectedPattern]:
        """Validate and create descending triangle pattern."""

        # Check time overlap
        overlap_start, overlap_end, time_range = self._calculate_time_overlap(
            support_line, resistance_line
        )

        if not overlap_start or time_range < self.min_duration_days:
            return None

        # Validate touch count
        total_touches = len(support_line.touches) + len(resistance_line.touches)
        if total_touches < self.min_touches:
            return None

        # Check volume confirmation
        volume_confirmed = (
            self._check_volume_confirmation(df, overlap_start, overlap_end)
            if overlap_start and overlap_end
            else False
        )

        # Calculate confidence
        support_flatness = max(0, 1 - abs(support_line.slope) * 10)
        resistance_falling = min(1, abs(resistance_line.slope) * 20)
        quality_score = (support_line.r_squared + resistance_line.r_squared) / 2

        confidence_score = (
            support_flatness * 0.3 + resistance_falling * 0.3 + quality_score * 0.4
        )

        # Apply volume bonus
        if volume_confirmed:
            confidence_score *= 1.1

        # Determine strength
        strength = self._determine_pattern_strength(confidence_score)

        # Calculate target price (height of triangle projected downward)
        triangle_height = self._calculate_triangle_height(support_line, resistance_line)
        support_level = np.mean([touch[1] for touch in support_line.touches])
        target_price = support_level - triangle_height

        # Calculate metrics
        pattern_metrics = {
            "support_flatness": support_flatness,
            "support_slope": support_line.slope,
            "resistance_slope": resistance_line.slope,
            "triangle_height": triangle_height,
            "touch_count": total_touches,
            "volume_confirmed": volume_confirmed,
            "duration_days": time_range,
        }

        # Collect key points
        key_points = []
        key_points.extend(support_line.touches)
        key_points.extend(resistance_line.touches)
        key_points.sort(key=lambda x: x[0])

        return DetectedPattern(
            pattern_type=PatternType.DESCENDING_TRIANGLE,
            symbol=(
                df.get("symbol", ["UNKNOWN"])[0]
                if "symbol" in df.columns
                else "UNKNOWN"
            ),
            timeframe="unknown",
            start_time=overlap_start or datetime.now(),
            end_time=overlap_end or datetime.now(),
            confidence_score=min(1.0, confidence_score),
            strength=strength,
            key_points=key_points,
            pattern_metrics=pattern_metrics,
            target_price=float(target_price),
            volume_confirmation=volume_confirmed,
        )

    def _validate_symmetrical_triangle(
        self, support_line: TrendLine, resistance_line: TrendLine, df: pd.DataFrame
    ) -> Optional[DetectedPattern]:
        """Validate and create symmetrical triangle pattern."""

        # Check time overlap
        overlap_start, overlap_end, time_range = self._calculate_time_overlap(
            support_line, resistance_line
        )

        if not overlap_start or time_range < self.min_duration_days:
            return None

        # Validate touch count
        total_touches = len(support_line.touches) + len(resistance_line.touches)
        if total_touches < self.min_touches:
            return None

        # Check volume confirmation
        volume_confirmed = (
            self._check_volume_confirmation(df, overlap_start, overlap_end)
            if overlap_start and overlap_end
            else False
        )

        # Calculate convergence quality
        convergence_quality = max(
            0, 1 - abs(support_line.slope + resistance_line.slope) * 5
        )
        slope_balance = 1 - abs(abs(support_line.slope) - abs(resistance_line.slope))
        quality_score = (support_line.r_squared + resistance_line.r_squared) / 2

        confidence_score = (
            convergence_quality * 0.4 + slope_balance * 0.2 + quality_score * 0.4
        )

        # Apply volume bonus
        if volume_confirmed:
            confidence_score *= 1.1

        # Determine strength
        strength = self._determine_pattern_strength(confidence_score)

        # Calculate triangle height for target calculation
        triangle_height = self._calculate_triangle_height(support_line, resistance_line)

        # For symmetrical triangles, we can't predict direction, so provide both targets
        mid_price = (
            np.mean([touch[1] for touch in support_line.touches])
            + np.mean([touch[1] for touch in resistance_line.touches])
        ) / 2

        # Calculate metrics
        pattern_metrics = {
            "convergence_quality": convergence_quality,
            "slope_balance": slope_balance,
            "support_slope": support_line.slope,
            "resistance_slope": resistance_line.slope,
            "triangle_height": triangle_height,
            "touch_count": total_touches,
            "volume_confirmed": volume_confirmed,
            "duration_days": time_range,
            "upside_target": mid_price + triangle_height,
            "downside_target": mid_price - triangle_height,
        }

        # Collect key points
        key_points = []
        key_points.extend(support_line.touches)
        key_points.extend(resistance_line.touches)
        key_points.sort(key=lambda x: x[0])

        return DetectedPattern(
            pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
            symbol=(
                df.get("symbol", ["UNKNOWN"])[0]
                if "symbol" in df.columns
                else "UNKNOWN"
            ),
            timeframe="unknown",
            start_time=overlap_start or datetime.now(),
            end_time=overlap_end or datetime.now(),
            confidence_score=min(1.0, confidence_score),
            strength=strength,
            key_points=key_points,
            pattern_metrics=pattern_metrics,
            target_price=float(mid_price),  # Neutral midpoint
            volume_confirmation=volume_confirmed,
        )

    def _calculate_time_overlap(
        self, support_line: TrendLine, resistance_line: TrendLine
    ) -> Tuple[Optional[datetime], Optional[datetime], float]:
        """Calculate time overlap between two trendlines."""
        support_start = min(touch[0] for touch in support_line.touches)
        support_end = max(touch[0] for touch in support_line.touches)
        resistance_start = min(touch[0] for touch in resistance_line.touches)
        resistance_end = max(touch[0] for touch in resistance_line.touches)

        overlap_start = max(support_start, resistance_start)
        overlap_end = min(support_end, resistance_end)

        if overlap_start >= overlap_end:
            return None, None, 0.0

        time_range = (overlap_end - overlap_start).total_seconds() / 86400  # days
        return overlap_start, overlap_end, time_range

    def _calculate_triangle_height(
        self, support_line: TrendLine, resistance_line: TrendLine
    ) -> float:
        """Calculate the height of the triangle pattern."""
        support_prices = [touch[1] for touch in support_line.touches]
        resistance_prices = [touch[1] for touch in resistance_line.touches]

        avg_support = np.mean(support_prices)
        avg_resistance = np.mean(resistance_prices)

        return float(abs(avg_resistance - avg_support))

    def _check_volume_confirmation(
        self, df: pd.DataFrame, start_time: datetime, end_time: datetime
    ) -> bool:
        """Check if volume pattern confirms the triangle formation."""
        if "volume" not in df.columns:
            return False

        # Filter data to triangle period
        df_filtered = df[
            (pd.to_datetime(df["timestamp"]) >= start_time)
            & (pd.to_datetime(df["timestamp"]) <= end_time)
        ].copy()

        if len(df_filtered) < 10:  # Not enough data
            return False

        # Volume should generally decrease during triangle formation
        volumes = np.array(df_filtered["volume"].values)

        # Check for declining volume trend
        first_half_avg = np.mean(volumes[: len(volumes) // 2])
        second_half_avg = np.mean(volumes[len(volumes) // 2 :])

        # Volume decline indicates consolidation (typical in triangles)
        volume_decline = (first_half_avg - second_half_avg) / first_half_avg

        return bool(volume_decline > 0.1)  # At least 10% volume decline

    def _determine_pattern_strength(self, confidence_score: float) -> PatternStrength:
        """Determine pattern strength based on confidence score."""
        if confidence_score >= 0.85:
            return PatternStrength.VERY_STRONG
        elif confidence_score >= 0.75:
            return PatternStrength.STRONG
        elif confidence_score >= 0.65:
            return PatternStrength.MODERATE
        else:
            return PatternStrength.WEAK
