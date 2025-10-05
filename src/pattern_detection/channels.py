"""
Rectangle and Channel Pattern Detection

This module provides detection for consolidation and trending patterns:
- Horizontal Rectangles (range-bound consolidation)
- Ascending Channels (uptrend with parallel lines)
- Descending Channels (downtrend with parallel lines)

These patterns show price moving between parallel support and resistance.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from .pattern_engine import (
        DetectedPattern,
        PatternType,
        PatternStrength,
        TrendLine,
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from pattern_detection.pattern_engine import (
        DetectedPattern,
        PatternType,
        PatternStrength,
        TrendLine,
    )


class ChannelDetector:
    """
    Detector for rectangle and channel patterns.

    Identifies parallel support and resistance levels where price
    oscillates within defined boundaries.
    """

    def __init__(
        self,
        min_touches: int = 4,  # Minimum touches of support/resistance
        min_duration_days: int = 15,
        max_duration_days: int = 120,
        parallel_tolerance: float = 0.02,  # 2% tolerance for parallel lines
        min_channel_width: float = 0.03,  # 3% minimum width
    ):
        """Initialize Channel Detector."""
        self.min_touches = min_touches
        self.min_duration_days = min_duration_days
        self.max_duration_days = max_duration_days
        self.parallel_tolerance = parallel_tolerance
        self.min_channel_width = min_channel_width

    def detect_rectangles(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "daily"
    ) -> List[DetectedPattern]:
        """
        Detect horizontal rectangle patterns.

        Rectangle characteristics:
        - Horizontal support and resistance (parallel)
        - Multiple touches on both levels
        - Price oscillates between levels
        - Breakout signals trend continuation/reversal
        """
        patterns = []

        if len(df) < self.min_duration_days:
            return patterns

        # Find horizontal support and resistance levels
        support_levels = self._find_horizontal_levels(df, level_type='support')
        resistance_levels = self._find_horizontal_levels(df, level_type='resistance')

        # Match parallel support/resistance pairs
        for support in support_levels:
            for resistance in resistance_levels:
                # Check if they form a valid rectangle
                pattern = self._validate_rectangle(
                    df, support, resistance, symbol, timeframe
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    def detect_ascending_channels(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "daily"
    ) -> List[DetectedPattern]:
        """
        Detect ascending channel patterns.

        Ascending channel characteristics:
        - Rising support trendline
        - Rising resistance trendline (parallel to support)
        - Higher highs and higher lows
        - Bullish continuation pattern
        """
        patterns = []

        if len(df) < self.min_duration_days:
            return patterns

        # Find rising trendlines
        support_lines = self._find_trendlines(df, direction='rising', line_type='support')
        resistance_lines = self._find_trendlines(df, direction='rising', line_type='resistance')

        # Match parallel rising lines
        for support in support_lines:
            for resistance in resistance_lines:
                # Check if parallel and valid
                pattern = self._validate_ascending_channel(
                    df, support, resistance, symbol, timeframe
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    def detect_descending_channels(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "daily"
    ) -> List[DetectedPattern]:
        """
        Detect descending channel patterns.

        Descending channel characteristics:
        - Falling support trendline
        - Falling resistance trendline (parallel to support)
        - Lower highs and lower lows
        - Bearish continuation pattern
        """
        patterns = []

        if len(df) < self.min_duration_days:
            return patterns

        # Find falling trendlines
        support_lines = self._find_trendlines(df, direction='falling', line_type='support')
        resistance_lines = self._find_trendlines(df, direction='falling', line_type='resistance')

        # Match parallel falling lines
        for support in support_lines:
            for resistance in resistance_lines:
                # Check if parallel and valid
                pattern = self._validate_descending_channel(
                    df, support, resistance, symbol, timeframe
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _find_horizontal_levels(
        self,
        df: pd.DataFrame,
        level_type: str = 'support'
    ) -> List[dict]:
        """Find horizontal support or resistance levels."""
        levels = []

        # Use highs for resistance, lows for support
        prices = df['high'].values if level_type == 'resistance' else df['low'].values

        # Find clusters of similar prices
        from scipy.cluster.hierarchy import fclusterdata

        if len(prices) < 10:
            return levels

        # Cluster prices
        try:
            price_array = prices.reshape(-1, 1)
            clusters = fclusterdata(price_array, t=0.02, criterion='distance', metric='euclidean')

            # For each cluster, find the level
            for cluster_id in np.unique(clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]

                if len(cluster_indices) < self.min_touches:
                    continue

                # Calculate level price (mean of cluster)
                level_price = prices[cluster_indices].mean()

                # Find touches
                touches = [(self._get_timestamp(df, idx), prices[idx]) for idx in cluster_indices]

                levels.append({
                    'price': level_price,
                    'touches': touches,
                    'touch_count': len(cluster_indices),
                    'type': level_type,
                    'start_idx': cluster_indices[0],
                    'end_idx': cluster_indices[-1]
                })
        except:
            # Fallback: simple approach
            # Find local extrema
            from scipy.signal import argrelextrema
            extrema_indices = argrelextrema(prices, np.less if level_type == 'support' else np.greater, order=5)[0]

            # Group similar prices
            if len(extrema_indices) >= self.min_touches:
                level_price = prices[extrema_indices].mean()
                touches = [(self._get_timestamp(df, idx), prices[idx]) for idx in extrema_indices]

                levels.append({
                    'price': level_price,
                    'touches': touches,
                    'touch_count': len(extrema_indices),
                    'type': level_type,
                    'start_idx': extrema_indices[0],
                    'end_idx': extrema_indices[-1]
                })

        return levels

    def _find_trendlines(
        self,
        df: pd.DataFrame,
        direction: str = 'rising',
        line_type: str = 'support'
    ) -> List[TrendLine]:
        """Find rising or falling trendlines."""
        trendlines = []

        # Get appropriate price points
        if line_type == 'support':
            from scipy.signal import argrelextrema
            prices = df['low'].values
            indices = argrelextrema(prices, np.less, order=5)[0]
        else:
            from scipy.signal import argrelextrema
            prices = df['high'].values
            indices = argrelextrema(prices, np.greater, order=5)[0]

        if len(indices) < 2:
            return trendlines

        # Try to fit trendlines through points
        for i in range(len(indices) - 1):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]

                # Calculate slope
                x1, y1 = idx1, prices[idx1]
                x2, y2 = idx2, prices[idx2]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0

                # Check direction
                if direction == 'rising' and slope <= 0:
                    continue
                if direction == 'falling' and slope >= 0:
                    continue

                # Count touches
                touches = self._count_trendline_touches(df, idx1, idx2, slope, y1, line_type)

                if len(touches) >= self.min_touches:
                    trendline = TrendLine(
                        start_point=(self._get_timestamp(df, idx1), y1),
                        end_point=(self._get_timestamp(df, idx2), y2),
                        slope=slope,
                        r_squared=0.85,  # Simplified
                        touches=touches,
                        line_type=line_type
                    )
                    trendlines.append(trendline)

        return trendlines

    def _count_trendline_touches(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        slope: float,
        intercept: float,
        line_type: str
    ) -> List[Tuple[datetime, float]]:
        """Count how many times price touches a trendline."""
        touches = []
        tolerance = 0.015  # 1.5% tolerance

        prices = df['low'].values if line_type == 'support' else df['high'].values

        for idx in range(start_idx, end_idx + 1):
            expected = intercept + slope * (idx - start_idx)
            actual = prices[idx]

            # Check if price touches the line
            if abs(actual - expected) / expected <= tolerance:
                touches.append((self._get_timestamp(df, idx), actual))

        return touches

    def _validate_rectangle(
        self,
        df: pd.DataFrame,
        support: dict,
        resistance: dict,
        symbol: str,
        timeframe: str
    ) -> Optional[DetectedPattern]:
        """Validate and create rectangle pattern."""

        # Check if support and resistance are parallel (horizontal)
        # Both should have very low slope (horizontal)

        # Check time overlap
        start_idx = max(support['start_idx'], resistance['start_idx'])
        end_idx = min(support['end_idx'], resistance['end_idx'])

        if end_idx - start_idx < self.min_duration_days:
            return None

        # Check channel width
        channel_width = (resistance['price'] - support['price']) / support['price']
        if channel_width < self.min_channel_width:
            return None

        # Calculate confidence
        confidence = self._calculate_rectangle_confidence(
            support, resistance, df, start_idx, end_idx
        )

        # Determine breakout direction based on current price
        current_price = df['close'].iloc[-1]
        midpoint = (support['price'] + resistance['price']) / 2

        if current_price > midpoint:
            direction = "bullish"
            target_price = resistance['price'] + (resistance['price'] - support['price'])
            stop_loss = support['price'] * 0.98
        else:
            direction = "bearish"
            target_price = support['price'] - (resistance['price'] - support['price'])
            stop_loss = resistance['price'] * 1.02

        # Create pattern
        pattern = DetectedPattern(
            pattern_type=PatternType.RECTANGLE,
            symbol=symbol,
            timeframe=timeframe,
            start_time=self._get_timestamp(df, start_idx),
            end_time=self._get_timestamp(df, end_idx),
            confidence_score=confidence,
            strength=self._determine_strength(confidence),
            key_points=[
                (self._get_timestamp(df, start_idx), support['price']),
                (self._get_timestamp(df, start_idx), resistance['price']),
                (self._get_timestamp(df, end_idx), resistance['price']),
                (self._get_timestamp(df, end_idx), support['price'])
            ],
            pattern_metrics={
                'support_level': support['price'],
                'resistance_level': resistance['price'],
                'channel_width_pct': channel_width,
                'duration_days': end_idx - start_idx,
                'touch_count': support['touch_count'] + resistance['touch_count']
            },
            direction=direction,
            target_price=target_price,
            stop_loss=stop_loss,
            volume_confirmation=False
        )

        return pattern

    def _validate_ascending_channel(
        self,
        df: pd.DataFrame,
        support: TrendLine,
        resistance: TrendLine,
        symbol: str,
        timeframe: str
    ) -> Optional[DetectedPattern]:
        """Validate and create ascending channel pattern."""

        # Check if lines are parallel
        if not self._are_parallel(support.slope, resistance.slope):
            return None

        # Check channel width
        # Estimate average distance between lines
        start_support_y = support.start_point[1]
        start_resistance_y = resistance.start_point[1]

        channel_width = (start_resistance_y - start_support_y) / start_support_y
        if channel_width < self.min_channel_width:
            return None

        # Calculate confidence
        confidence = self._calculate_channel_confidence(
            support, resistance, 'ascending'
        )

        # Calculate target (channel continuation)
        # Use channel width to project target
        current_resistance = resistance.end_point[1]
        target_price = current_resistance * 1.10  # 10% above current resistance
        stop_loss = support.end_point[1] * 0.98

        # Create pattern
        pattern = DetectedPattern(
            pattern_type=PatternType.ASCENDING_CHANNEL,
            symbol=symbol,
            timeframe=timeframe,
            start_time=support.start_point[0],
            end_time=support.end_point[0],
            confidence_score=confidence,
            strength=self._determine_strength(confidence),
            key_points=[
                support.start_point,
                resistance.start_point,
                resistance.end_point,
                support.end_point
            ],
            pattern_metrics={
                'support_slope': support.slope,
                'resistance_slope': resistance.slope,
                'channel_width_pct': channel_width,
                'support_touches': len(support.touches),
                'resistance_touches': len(resistance.touches)
            },
            direction="bullish",
            target_price=target_price,
            stop_loss=stop_loss,
            volume_confirmation=False
        )

        return pattern

    def _validate_descending_channel(
        self,
        df: pd.DataFrame,
        support: TrendLine,
        resistance: TrendLine,
        symbol: str,
        timeframe: str
    ) -> Optional[DetectedPattern]:
        """Validate and create descending channel pattern."""

        # Check if lines are parallel
        if not self._are_parallel(support.slope, resistance.slope):
            return None

        # Check channel width
        start_support_y = support.start_point[1]
        start_resistance_y = resistance.start_point[1]

        channel_width = (start_resistance_y - start_support_y) / start_support_y
        if channel_width < self.min_channel_width:
            return None

        # Calculate confidence
        confidence = self._calculate_channel_confidence(
            support, resistance, 'descending'
        )

        # Calculate target (channel continuation downward)
        current_support = support.end_point[1]
        target_price = current_support * 0.90  # 10% below current support
        stop_loss = resistance.end_point[1] * 1.02

        # Create pattern
        pattern = DetectedPattern(
            pattern_type=PatternType.DESCENDING_CHANNEL,
            symbol=symbol,
            timeframe=timeframe,
            start_time=support.start_point[0],
            end_time=support.end_point[0],
            confidence_score=confidence,
            strength=self._determine_strength(confidence),
            key_points=[
                support.start_point,
                resistance.start_point,
                resistance.end_point,
                support.end_point
            ],
            pattern_metrics={
                'support_slope': support.slope,
                'resistance_slope': resistance.slope,
                'channel_width_pct': channel_width,
                'support_touches': len(support.touches),
                'resistance_touches': len(resistance.touches)
            },
            direction="bearish",
            target_price=target_price,
            stop_loss=stop_loss,
            volume_confirmation=False
        )

        return pattern

    def _are_parallel(self, slope1: float, slope2: float) -> bool:
        """Check if two slopes are parallel within tolerance."""
        if slope1 == 0 and slope2 == 0:
            return True

        if slope1 == 0 or slope2 == 0:
            return False

        # Calculate difference
        slope_diff = abs(slope1 - slope2) / max(abs(slope1), abs(slope2))

        return slope_diff <= self.parallel_tolerance

    def _calculate_rectangle_confidence(
        self,
        support: dict,
        resistance: dict,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int
    ) -> float:
        """Calculate confidence for rectangle pattern."""
        confidence = 0.5

        # More touches = higher confidence
        total_touches = support['touch_count'] + resistance['touch_count']
        if total_touches >= 8:
            confidence += 0.20
        elif total_touches >= 6:
            confidence += 0.15
        elif total_touches >= self.min_touches:
            confidence += 0.10

        # Good channel width (not too narrow)
        width = (resistance['price'] - support['price']) / support['price']
        if 0.05 <= width <= 0.15:  # 5-15% is ideal
            confidence += 0.15
        elif width >= 0.03:
            confidence += 0.10

        # Longer duration = more significant
        duration = end_idx - start_idx
        if duration >= 30:
            confidence += 0.15
        elif duration >= 20:
            confidence += 0.10

        return min(confidence, 1.0)

    def _calculate_channel_confidence(
        self,
        support: TrendLine,
        resistance: TrendLine,
        channel_type: str
    ) -> float:
        """Calculate confidence for channel pattern."""
        confidence = 0.5

        # Good parallelism
        if self._are_parallel(support.slope, resistance.slope):
            confidence += 0.15

        # Multiple touches
        total_touches = len(support.touches) + len(resistance.touches)
        if total_touches >= 8:
            confidence += 0.20
        elif total_touches >= 6:
            confidence += 0.15

        # High R-squared (good fit)
        avg_r_squared = (support.r_squared + resistance.r_squared) / 2
        if avg_r_squared >= 0.90:
            confidence += 0.15
        elif avg_r_squared >= 0.80:
            confidence += 0.10

        return min(confidence, 1.0)

    def _determine_strength(self, confidence: float) -> PatternStrength:
        """Determine pattern strength from confidence."""
        if confidence >= 0.80:
            return PatternStrength.VERY_STRONG
        elif confidence >= 0.70:
            return PatternStrength.STRONG
        elif confidence >= 0.60:
            return PatternStrength.MODERATE
        else:
            return PatternStrength.WEAK

    def _get_timestamp(self, df: pd.DataFrame, idx: int) -> datetime:
        """Get timestamp from dataframe."""
        if 'timestamp' in df.columns:
            return df['timestamp'].iloc[idx]
        elif df.index.name == 'timestamp':
            return df.index[idx]
        else:
            return datetime.now() - timedelta(days=len(df) - idx)
