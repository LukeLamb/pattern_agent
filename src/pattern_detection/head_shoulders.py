"""
Head & Shoulders Pattern Detection - Specialized algorithms for H&S patterns.

This module provides detailed head and shoulders pattern detection including:
- Head & Shoulders (bearish reversal)
- Inverse Head & Shoulders (bullish reversal)
- Volume confirmation and neckline analysis
"""

from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .pattern_engine import (
    DetectedPattern,
    PatternType,
    PatternStrength,
    PivotPoint,
    TrendLine,
)


class HeadShouldersDetector:
    """
    Specialized detector for Head & Shoulders chart patterns.

    Provides advanced H&S pattern recognition with volume confirmation,
    neckline validation, and target price calculation.
    """

    def __init__(
        self, min_prominence: float = 0.02, volume_confirmation_threshold: float = 1.2
    ):
        """Initialize Head & Shoulders Detector."""
        self.min_prominence = min_prominence  # Minimum height difference for pattern
        self.volume_confirmation_threshold = (
            volume_confirmation_threshold  # Volume ratio for confirmation
        )

    def detect_head_and_shoulders(
        self, df: pd.DataFrame, pivot_points: List[PivotPoint]
    ) -> List[DetectedPattern]:
        """
        Detect classic Head & Shoulders patterns (bearish reversal).

        Pattern structure:
        - Left shoulder (peak)
        - Head (higher peak)
        - Right shoulder (peak, similar height to left)
        - Neckline connecting the valleys between shoulders and head
        """
        patterns = []

        # Filter for peaks only
        peaks = [p for p in pivot_points if p.pivot_type == "peak"]

        if len(peaks) < 3:
            return patterns

        # Look for potential H&S patterns
        for i in range(len(peaks) - 2):
            for j in range(i + 1, len(peaks) - 1):
                for k in range(j + 1, len(peaks)):
                    left_shoulder = peaks[i]
                    head = peaks[j]
                    right_shoulder = peaks[k]

                    pattern = self._validate_head_and_shoulders(
                        left_shoulder, head, right_shoulder, df, pivot_points
                    )
                    if pattern:
                        patterns.append(pattern)

        return patterns

    def detect_inverse_head_and_shoulders(
        self, df: pd.DataFrame, pivot_points: List[PivotPoint]
    ) -> List[DetectedPattern]:
        """
        Detect Inverse Head & Shoulders patterns (bullish reversal).

        Pattern structure:
        - Left shoulder (valley)
        - Head (lower valley)
        - Right shoulder (valley, similar depth to left)
        - Neckline connecting the peaks between shoulders and head
        """
        patterns = []

        # Filter for valleys only
        valleys = [p for p in pivot_points if p.pivot_type == "valley"]

        if len(valleys) < 3:
            return patterns

        # Look for potential inverse H&S patterns
        for i in range(len(valleys) - 2):
            for j in range(i + 1, len(valleys) - 1):
                for k in range(j + 1, len(valleys)):
                    left_shoulder = valleys[i]
                    head = valleys[j]
                    right_shoulder = valleys[k]

                    pattern = self._validate_inverse_head_and_shoulders(
                        left_shoulder, head, right_shoulder, df, pivot_points
                    )
                    if pattern:
                        patterns.append(pattern)

        return patterns

    def _validate_head_and_shoulders(
        self,
        left_shoulder: PivotPoint,
        head: PivotPoint,
        right_shoulder: PivotPoint,
        df: pd.DataFrame,
        pivot_points: List[PivotPoint],
    ) -> Optional[DetectedPattern]:
        """Validate and create head and shoulders pattern."""

        # 1. Head must be higher than both shoulders
        if not (head.price > left_shoulder.price and head.price > right_shoulder.price):
            return None

        # 2. Check shoulder symmetry (should be roughly equal heights)
        shoulder_height_diff = abs(left_shoulder.price - right_shoulder.price) / max(
            left_shoulder.price, right_shoulder.price
        )
        if shoulder_height_diff > 0.1:  # More than 10% difference
            return None

        # 3. Check head prominence (head should be significantly higher)
        head_prominence_left = (head.price - left_shoulder.price) / left_shoulder.price
        head_prominence_right = (
            head.price - right_shoulder.price
        ) / right_shoulder.price

        if (
            head_prominence_left < self.min_prominence
            or head_prominence_right < self.min_prominence
        ):
            return None

        # 4. Find valleys between shoulders and head for neckline
        valleys_between = self._find_valleys_between_peaks(
            left_shoulder, head, right_shoulder, pivot_points
        )

        if len(valleys_between) < 2:
            return None

        left_valley = valleys_between[0]
        right_valley = valleys_between[1]

        # 5. Calculate neckline
        _, _ = self._calculate_neckline(left_valley, right_valley)

        # 6. Check time sequence (left -> head -> right chronologically)
        if not (left_shoulder.timestamp < head.timestamp < right_shoulder.timestamp):
            return None

        # 7. Volume analysis
        volume_confirmation = self._check_hs_volume_pattern(
            df, left_shoulder, head, right_shoulder
        )

        # 8. Calculate pattern metrics
        pattern_metrics = self._calculate_hs_metrics(
            left_shoulder,
            head,
            right_shoulder,
            left_valley,
            right_valley,
            shoulder_height_diff,
            head_prominence_left,
            head_prominence_right,
            volume_confirmation,
        )

        # 9. Calculate target price (pattern height projected down from neckline)
        pattern_height = head.price - min(left_valley.price, right_valley.price)
        neckline_price = min(
            left_valley.price, right_valley.price
        )  # Conservative neckline
        target_price = neckline_price - pattern_height

        # 10. Determine confidence and strength
        confidence_score = self._calculate_hs_confidence(
            pattern_metrics, volume_confirmation
        )
        strength = self._determine_pattern_strength(confidence_score)

        # 11. Collect key points
        key_points = [
            (left_shoulder.timestamp, left_shoulder.price),
            (left_valley.timestamp, left_valley.price),
            (head.timestamp, head.price),
            (right_valley.timestamp, right_valley.price),
            (right_shoulder.timestamp, right_shoulder.price),
        ]

        return DetectedPattern(
            pattern_type=PatternType.HEAD_AND_SHOULDERS,
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
            target_price=float(target_price),
            volume_confirmation=volume_confirmation,
        )

    def _validate_inverse_head_and_shoulders(
        self,
        left_shoulder: PivotPoint,
        head: PivotPoint,
        right_shoulder: PivotPoint,
        df: pd.DataFrame,
        pivot_points: List[PivotPoint],
    ) -> Optional[DetectedPattern]:
        """Validate and create inverse head and shoulders pattern."""

        # 1. Head must be lower than both shoulders
        if not (head.price < left_shoulder.price and head.price < right_shoulder.price):
            return None

        # 2. Check shoulder symmetry
        shoulder_depth_diff = abs(left_shoulder.price - right_shoulder.price) / min(
            left_shoulder.price, right_shoulder.price
        )
        if shoulder_depth_diff > 0.1:
            return None

        # 3. Check head prominence (head should be significantly lower)
        head_prominence_left = (left_shoulder.price - head.price) / head.price
        head_prominence_right = (right_shoulder.price - head.price) / head.price

        if (
            head_prominence_left < self.min_prominence
            or head_prominence_right < self.min_prominence
        ):
            return None

        # 4. Find peaks between shoulders and head for neckline
        peaks_between = self._find_peaks_between_valleys(
            left_shoulder, head, right_shoulder, pivot_points
        )

        if len(peaks_between) < 2:
            return None

        left_peak = peaks_between[0]
        right_peak = peaks_between[1]

        # 5. Calculate neckline
        _, _ = self._calculate_neckline(left_peak, right_peak)

        # 6. Check time sequence
        if not (left_shoulder.timestamp < head.timestamp < right_shoulder.timestamp):
            return None

        # 7. Volume analysis
        volume_confirmation = self._check_ihs_volume_pattern(
            df, left_shoulder, head, right_shoulder
        )

        # 8. Calculate pattern metrics
        pattern_metrics = self._calculate_ihs_metrics(
            left_shoulder,
            head,
            right_shoulder,
            left_peak,
            right_peak,
            shoulder_depth_diff,
            head_prominence_left,
            head_prominence_right,
            volume_confirmation,
        )

        # 9. Calculate target price (pattern height projected up from neckline)
        pattern_height = max(left_peak.price, right_peak.price) - head.price
        neckline_price = max(left_peak.price, right_peak.price)  # Conservative neckline
        target_price = neckline_price + pattern_height

        # 10. Determine confidence and strength
        confidence_score = self._calculate_ihs_confidence(
            pattern_metrics, volume_confirmation
        )
        strength = self._determine_pattern_strength(confidence_score)

        # 11. Collect key points
        key_points = [
            (left_shoulder.timestamp, left_shoulder.price),
            (left_peak.timestamp, left_peak.price),
            (head.timestamp, head.price),
            (right_peak.timestamp, right_peak.price),
            (right_shoulder.timestamp, right_shoulder.price),
        ]

        return DetectedPattern(
            pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
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
            target_price=float(target_price),
            volume_confirmation=volume_confirmation,
        )

    def _find_valleys_between_peaks(
        self,
        left_peak: PivotPoint,
        center_peak: PivotPoint,
        right_peak: PivotPoint,
        pivot_points: List[PivotPoint],
    ) -> List[PivotPoint]:
        """Find valleys between the three peaks for neckline calculation."""
        valleys = []

        # Find valley between left peak and center peak
        left_valley = None
        for point in pivot_points:
            if (
                point.pivot_type == "valley"
                and left_peak.timestamp < point.timestamp < center_peak.timestamp
            ):
                if left_valley is None or point.price < left_valley.price:
                    left_valley = point

        # Find valley between center peak and right peak
        right_valley = None
        for point in pivot_points:
            if (
                point.pivot_type == "valley"
                and center_peak.timestamp < point.timestamp < right_peak.timestamp
            ):
                if right_valley is None or point.price < right_valley.price:
                    right_valley = point

        if left_valley:
            valleys.append(left_valley)
        if right_valley:
            valleys.append(right_valley)

        return valleys

    def _find_peaks_between_valleys(
        self,
        left_valley: PivotPoint,
        center_valley: PivotPoint,
        right_valley: PivotPoint,
        pivot_points: List[PivotPoint],
    ) -> List[PivotPoint]:
        """Find peaks between the three valleys for neckline calculation."""
        peaks = []

        # Find peak between left valley and center valley
        left_peak = None
        for point in pivot_points:
            if (
                point.pivot_type == "peak"
                and left_valley.timestamp < point.timestamp < center_valley.timestamp
            ):
                if left_peak is None or point.price > left_peak.price:
                    left_peak = point

        # Find peak between center valley and right valley
        right_peak = None
        for point in pivot_points:
            if (
                point.pivot_type == "peak"
                and center_valley.timestamp < point.timestamp < right_valley.timestamp
            ):
                if right_peak is None or point.price > right_peak.price:
                    right_peak = point

        if left_peak:
            peaks.append(left_peak)
        if right_peak:
            peaks.append(right_peak)

        return peaks

    def _calculate_neckline(
        self, point1: PivotPoint, point2: PivotPoint
    ) -> Tuple[float, float]:
        """Calculate neckline slope and intercept."""
        time_diff = (
            point2.timestamp - point1.timestamp
        ).total_seconds() / 86400  # days
        if time_diff == 0:
            return 0.0, point1.price

        slope = (point2.price - point1.price) / time_diff
        intercept = point1.price - slope * 0  # At point1's time

        return slope, intercept

    def _check_hs_volume_pattern(
        self,
        df: pd.DataFrame,
        left_shoulder: PivotPoint,
        head: PivotPoint,
        right_shoulder: PivotPoint,
    ) -> bool:
        """Check volume pattern for Head & Shoulders (volume should decline on right shoulder)."""
        if "volume" not in df.columns:
            return False

        # Get volume around each peak
        left_vol = self._get_volume_around_point(df, left_shoulder.timestamp)
        head_vol = self._get_volume_around_point(df, head.timestamp)
        right_vol = self._get_volume_around_point(df, right_shoulder.timestamp)

        if not all([left_vol, head_vol, right_vol]):
            return False

        # Classic H&S: highest volume on head, declining on right shoulder
        return (
            head_vol is not None
            and left_vol is not None
            and right_vol is not None
            and head_vol > left_vol
            and right_vol < head_vol
            and right_vol < left_vol * self.volume_confirmation_threshold
        )

    def _check_ihs_volume_pattern(
        self,
        df: pd.DataFrame,
        left_shoulder: PivotPoint,
        head: PivotPoint,
        right_shoulder: PivotPoint,
    ) -> bool:
        """Check volume pattern for Inverse H&S (volume should increase on right shoulder)."""
        if "volume" not in df.columns:
            return False

        # Get volume around each valley
        left_vol = self._get_volume_around_point(df, left_shoulder.timestamp)
        head_vol = self._get_volume_around_point(df, head.timestamp)
        right_vol = self._get_volume_around_point(df, right_shoulder.timestamp)

        if not all([left_vol, head_vol, right_vol]):
            return False

        # Inverse H&S: volume should increase on right shoulder
        return (
            left_vol is not None
            and head_vol is not None
            and right_vol is not None
            and right_vol > left_vol * self.volume_confirmation_threshold
            and right_vol >= head_vol
        )

    def _get_volume_around_point(
        self, df: pd.DataFrame, timestamp: datetime, window_hours: int = 24
    ) -> Optional[float]:
        """Get average volume around a specific timestamp."""
        start_time = timestamp - timedelta(hours=window_hours // 2)
        end_time = timestamp + timedelta(hours=window_hours // 2)

        filtered_df = df[
            (pd.to_datetime(df["timestamp"]) >= start_time)
            & (pd.to_datetime(df["timestamp"]) <= end_time)
        ]

        if len(filtered_df) == 0:
            return None

        return float(np.mean(np.array(filtered_df["volume"].values)))

    def _calculate_hs_metrics(
        self,
        left_shoulder: PivotPoint,
        head: PivotPoint,
        right_shoulder: PivotPoint,
        left_valley: PivotPoint,
        right_valley: PivotPoint,
        shoulder_height_diff: float,
        head_prominence_left: float,
        head_prominence_right: float,
        volume_confirmation: bool,
    ) -> Dict:
        """Calculate metrics for Head & Shoulders pattern."""
        pattern_duration = (
            right_shoulder.timestamp - left_shoulder.timestamp
        ).total_seconds() / 86400
        pattern_height = head.price - min(left_valley.price, right_valley.price)

        return {
            "pattern_height": float(pattern_height),
            "pattern_duration_days": pattern_duration,
            "shoulder_symmetry": 1.0 - shoulder_height_diff,
            "head_prominence_left": head_prominence_left,
            "head_prominence_right": head_prominence_right,
            "left_shoulder_price": float(left_shoulder.price),
            "head_price": float(head.price),
            "right_shoulder_price": float(right_shoulder.price),
            "left_valley_price": float(left_valley.price),
            "right_valley_price": float(right_valley.price),
            "volume_confirmed": volume_confirmation,
            "neckline_slope": self._calculate_neckline(left_valley, right_valley)[0],
        }

    def _calculate_ihs_metrics(
        self,
        left_shoulder: PivotPoint,
        head: PivotPoint,
        right_shoulder: PivotPoint,
        left_peak: PivotPoint,
        right_peak: PivotPoint,
        shoulder_depth_diff: float,
        head_prominence_left: float,
        head_prominence_right: float,
        volume_confirmation: bool,
    ) -> Dict:
        """Calculate metrics for Inverse Head & Shoulders pattern."""
        pattern_duration = (
            right_shoulder.timestamp - left_shoulder.timestamp
        ).total_seconds() / 86400
        pattern_height = max(left_peak.price, right_peak.price) - head.price

        return {
            "pattern_height": float(pattern_height),
            "pattern_duration_days": pattern_duration,
            "shoulder_symmetry": 1.0 - shoulder_depth_diff,
            "head_prominence_left": head_prominence_left,
            "head_prominence_right": head_prominence_right,
            "left_shoulder_price": float(left_shoulder.price),
            "head_price": float(head.price),
            "right_shoulder_price": float(right_shoulder.price),
            "left_peak_price": float(left_peak.price),
            "right_peak_price": float(right_peak.price),
            "volume_confirmed": volume_confirmation,
            "neckline_slope": self._calculate_neckline(left_peak, right_peak)[0],
        }

    def _calculate_hs_confidence(
        self, metrics: Dict, volume_confirmation: bool
    ) -> float:
        """Calculate confidence score for Head & Shoulders pattern."""
        base_score = (
            metrics["shoulder_symmetry"] * 0.3
            + min(metrics["head_prominence_left"], metrics["head_prominence_right"])
            * 0.4
            + min(1.0, metrics["pattern_duration_days"] / 30) * 0.3
        )

        # Volume confirmation bonus
        if volume_confirmation:
            base_score *= 1.15

        return min(1.0, base_score)

    def _calculate_ihs_confidence(
        self, metrics: Dict, volume_confirmation: bool
    ) -> float:
        """Calculate confidence score for Inverse Head & Shoulders pattern."""
        base_score = (
            metrics["shoulder_symmetry"] * 0.3
            + min(metrics["head_prominence_left"], metrics["head_prominence_right"])
            * 0.4
            + min(1.0, metrics["pattern_duration_days"] / 30) * 0.3
        )

        # Volume confirmation bonus
        if volume_confirmation:
            base_score *= 1.15

        return min(1.0, base_score)

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
