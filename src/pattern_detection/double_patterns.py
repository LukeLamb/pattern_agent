"""
Double and Triple Top/Bottom Pattern Detection

This module provides detection for reversal patterns:
- Double Tops (bearish reversal)
- Double Bottoms (bullish reversal)
- Triple Tops (strong bearish reversal)
- Triple Bottoms (strong bullish reversal)

These patterns indicate potential trend reversals with multiple tests of a level.
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
        PivotPoint,
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from pattern_detection.pattern_engine import (
        DetectedPattern,
        PatternType,
        PatternStrength,
        PivotPoint,
    )


class DoublePatternDetector:
    """
    Detector for double and triple top/bottom patterns.

    These are reversal patterns formed by 2-3 similar peaks or troughs
    with a neckline that, when broken, signals the reversal.
    """

    def __init__(
        self,
        peak_similarity_tolerance: float = 0.03,  # 3% tolerance for peak similarity
        min_retracement: float = 0.05,  # 5% minimum retracement between peaks
        min_pattern_duration_days: int = 10,
        max_pattern_duration_days: int = 90,
    ):
        """Initialize Double Pattern Detector."""
        self.peak_similarity_tolerance = peak_similarity_tolerance
        self.min_retracement = min_retracement
        self.min_pattern_duration_days = min_pattern_duration_days
        self.max_pattern_duration_days = max_pattern_duration_days

    def detect_double_tops(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "daily"
    ) -> List[DetectedPattern]:
        """
        Detect double top patterns.

        Double top: Two similar peaks with a valley, followed by neckline break.
        """
        patterns = []
        peaks = self._find_peaks(df)

        # Need at least 2 peaks
        if len(peaks) < 2:
            return patterns

        # Check consecutive peak pairs
        for i in range(len(peaks) - 1):
            peak1 = peaks[i]
            peak2 = peaks[i + 1]

            # Check if peaks are similar in price
            if not self._are_similar_levels(peak1['price'], peak2['price']):
                continue

            # Find valley between peaks
            valley = self._find_valley_between(df, peak1['idx'], peak2['idx'])
            if valley is None:
                continue

            # Check retracement depth
            avg_peak_price = (peak1['price'] + peak2['price']) / 2
            retracement = (avg_peak_price - valley['price']) / avg_peak_price
            if retracement < self.min_retracement:
                continue

            # Check duration
            duration = (peak2['time'] - peak1['time']).days
            if not (self.min_pattern_duration_days <= duration <= self.max_pattern_duration_days):
                continue

            # Calculate confidence
            confidence = self._calculate_double_top_confidence(
                peak1, peak2, valley, df
            )

            # Create pattern
            pattern = self._create_double_top_pattern(
                peak1, peak2, valley, confidence, symbol, timeframe, df
            )
            patterns.append(pattern)

        return patterns

    def detect_double_bottoms(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "daily"
    ) -> List[DetectedPattern]:
        """Detect double bottom patterns (bullish reversal)."""
        patterns = []
        troughs = self._find_troughs(df)

        if len(troughs) < 2:
            return patterns

        for i in range(len(troughs) - 1):
            trough1 = troughs[i]
            trough2 = troughs[i + 1]

            if not self._are_similar_levels(trough1['price'], trough2['price']):
                continue

            peak = self._find_peak_between(df, trough1['idx'], trough2['idx'])
            if peak is None:
                continue

            avg_trough_price = (trough1['price'] + trough2['price']) / 2
            bounce = (peak['price'] - avg_trough_price) / avg_trough_price
            if bounce < self.min_retracement:
                continue

            duration = (trough2['time'] - trough1['time']).days
            if not (self.min_pattern_duration_days <= duration <= self.max_pattern_duration_days):
                continue

            confidence = self._calculate_double_bottom_confidence(
                trough1, trough2, peak, df
            )

            pattern = self._create_double_bottom_pattern(
                trough1, trough2, peak, confidence, symbol, timeframe, df
            )
            patterns.append(pattern)

        return patterns

    def detect_triple_tops(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "daily"
    ) -> List[DetectedPattern]:
        """Detect triple top patterns (strong bearish reversal)."""
        patterns = []
        peaks = self._find_peaks(df)

        if len(peaks) < 3:
            return patterns

        for i in range(len(peaks) - 2):
            peak1 = peaks[i]
            peak2 = peaks[i + 1]
            peak3 = peaks[i + 2]

            # All three peaks should be similar
            if not (self._are_similar_levels(peak1['price'], peak2['price']) and
                    self._are_similar_levels(peak2['price'], peak3['price'])):
                continue

            # Find valleys
            valley1 = self._find_valley_between(df, peak1['idx'], peak2['idx'])
            valley2 = self._find_valley_between(df, peak2['idx'], peak3['idx'])

            if valley1 is None or valley2 is None:
                continue

            # Valleys should also be similar (support level)
            if not self._are_similar_levels(valley1['price'], valley2['price'], tolerance=0.05):
                continue

            duration = (peak3['time'] - peak1['time']).days
            if not (self.min_pattern_duration_days <= duration <= self.max_pattern_duration_days):
                continue

            confidence = self._calculate_triple_top_confidence(
                peak1, peak2, peak3, valley1, valley2, df
            )

            pattern = self._create_triple_top_pattern(
                peak1, peak2, peak3, valley1, valley2, confidence, symbol, timeframe, df
            )
            patterns.append(pattern)

        return patterns

    def detect_triple_bottoms(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "daily"
    ) -> List[DetectedPattern]:
        """Detect triple bottom patterns (strong bullish reversal)."""
        patterns = []
        troughs = self._find_troughs(df)

        if len(troughs) < 3:
            return patterns

        for i in range(len(troughs) - 2):
            trough1 = troughs[i]
            trough2 = troughs[i + 1]
            trough3 = troughs[i + 2]

            if not (self._are_similar_levels(trough1['price'], trough2['price']) and
                    self._are_similar_levels(trough2['price'], trough3['price'])):
                continue

            peak1 = self._find_peak_between(df, trough1['idx'], trough2['idx'])
            peak2 = self._find_peak_between(df, trough2['idx'], trough3['idx'])

            if peak1 is None or peak2 is None:
                continue

            if not self._are_similar_levels(peak1['price'], peak2['price'], tolerance=0.05):
                continue

            duration = (trough3['time'] - trough1['time']).days
            if not (self.min_pattern_duration_days <= duration <= self.max_pattern_duration_days):
                continue

            confidence = self._calculate_triple_bottom_confidence(
                trough1, trough2, trough3, peak1, peak2, df
            )

            pattern = self._create_triple_bottom_pattern(
                trough1, trough2, trough3, peak1, peak2, confidence, symbol, timeframe, df
            )
            patterns.append(pattern)

        return patterns

    def _find_peaks(self, df: pd.DataFrame, order: int = 5) -> List[dict]:
        """Find local maxima (peaks) in price data."""
        from scipy.signal import argrelextrema

        highs = df['high'].values
        peak_indices = argrelextrema(highs, np.greater, order=order)[0]

        peaks = []
        for idx in peak_indices:
            peaks.append({
                'idx': idx,
                'price': highs[idx],
                'time': self._get_timestamp(df, idx)
            })

        return peaks

    def _find_troughs(self, df: pd.DataFrame, order: int = 5) -> List[dict]:
        """Find local minima (troughs) in price data."""
        from scipy.signal import argrelextrema

        lows = df['low'].values
        trough_indices = argrelextrema(lows, np.less, order=order)[0]

        troughs = []
        for idx in trough_indices:
            troughs.append({
                'idx': idx,
                'price': lows[idx],
                'time': self._get_timestamp(df, idx)
            })

        return troughs

    def _find_valley_between(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[dict]:
        """Find the lowest point between two peaks."""
        if start_idx >= end_idx:
            return None

        segment = df.iloc[start_idx:end_idx + 1]
        min_idx = segment['low'].idxmin()

        return {
            'idx': min_idx,
            'price': df.loc[min_idx, 'low'],
            'time': self._get_timestamp(df, min_idx)
        }

    def _find_peak_between(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[dict]:
        """Find the highest point between two troughs."""
        if start_idx >= end_idx:
            return None

        segment = df.iloc[start_idx:end_idx + 1]
        max_idx = segment['high'].idxmax()

        return {
            'idx': max_idx,
            'price': df.loc[max_idx, 'high'],
            'time': self._get_timestamp(df, max_idx)
        }

    def _are_similar_levels(self, price1: float, price2: float, tolerance: float = None) -> bool:
        """Check if two price levels are similar within tolerance."""
        if tolerance is None:
            tolerance = self.peak_similarity_tolerance

        avg_price = (price1 + price2) / 2
        diff_pct = abs(price1 - price2) / avg_price

        return diff_pct <= tolerance

    def _calculate_double_top_confidence(
        self, peak1: dict, peak2: dict, valley: dict, df: pd.DataFrame
    ) -> float:
        """Calculate confidence for double top pattern."""
        confidence = 0.5

        # Peak similarity
        similarity = 1.0 - abs(peak1['price'] - peak2['price']) / ((peak1['price'] + peak2['price']) / 2)
        confidence += similarity * 0.20

        # Sufficient retracement
        avg_peak = (peak1['price'] + peak2['price']) / 2
        retracement = (avg_peak - valley['price']) / avg_peak
        if retracement > 0.10:
            confidence += 0.15
        elif retracement > 0.07:
            confidence += 0.10

        # Volume divergence (second peak on lower volume)
        vol1 = df.loc[peak1['idx'], 'volume']
        vol2 = df.loc[peak2['idx'], 'volume']
        if vol2 < vol1 * 0.8:
            confidence += 0.15

        return min(confidence, 1.0)

    def _calculate_double_bottom_confidence(
        self, trough1: dict, trough2: dict, peak: dict, df: pd.DataFrame
    ) -> float:
        """Calculate confidence for double bottom pattern."""
        confidence = 0.5

        similarity = 1.0 - abs(trough1['price'] - trough2['price']) / ((trough1['price'] + trough2['price']) / 2)
        confidence += similarity * 0.20

        avg_trough = (trough1['price'] + trough2['price']) / 2
        bounce = (peak['price'] - avg_trough) / avg_trough
        if bounce > 0.10:
            confidence += 0.15
        elif bounce > 0.07:
            confidence += 0.10

        vol1 = df.loc[trough1['idx'], 'volume']
        vol2 = df.loc[trough2['idx'], 'volume']
        if vol2 > vol1 * 1.2:  # Second trough on higher volume (buying interest)
            confidence += 0.15

        return min(confidence, 1.0)

    def _calculate_triple_top_confidence(
        self, peak1: dict, peak2: dict, peak3: dict, valley1: dict, valley2: dict, df: pd.DataFrame
    ) -> float:
        """Calculate confidence for triple top (stronger signal than double)."""
        base_confidence = 0.6  # Higher base for triple patterns

        # All peaks similar
        avg_peak = (peak1['price'] + peak2['price'] + peak3['price']) / 3
        max_dev = max(abs(p['price'] - avg_peak) for p in [peak1, peak2, peak3])
        similarity = 1.0 - (max_dev / avg_peak)
        base_confidence += similarity * 0.20

        # Volume progression (ideally declining)
        vols = [df.loc[p['idx'], 'volume'] for p in [peak1, peak2, peak3]]
        if vols[2] < vols[1] < vols[0]:
            base_confidence += 0.15

        return min(base_confidence, 1.0)

    def _calculate_triple_bottom_confidence(
        self, trough1: dict, trough2: dict, trough3: dict, peak1: dict, peak2: dict, df: pd.DataFrame
    ) -> float:
        """Calculate confidence for triple bottom."""
        base_confidence = 0.6

        avg_trough = (trough1['price'] + trough2['price'] + trough3['price']) / 3
        max_dev = max(abs(t['price'] - avg_trough) for t in [trough1, trough2, trough3])
        similarity = 1.0 - (max_dev / avg_trough)
        base_confidence += similarity * 0.20

        vols = [df.loc[t['idx'], 'volume'] for t in [trough1, trough2, trough3]]
        if vols[2] > vols[1] > vols[0]:  # Increasing volume on bottoms
            base_confidence += 0.15

        return min(base_confidence, 1.0)

    def _create_double_top_pattern(
        self, peak1: dict, peak2: dict, valley: dict, confidence: float,
        symbol: str, timeframe: str, df: pd.DataFrame
    ) -> DetectedPattern:
        """Create DetectedPattern for double top."""
        avg_peak = (peak1['price'] + peak2['price']) / 2
        neckline = valley['price']
        target_price = neckline - (avg_peak - neckline)

        return DetectedPattern(
            pattern_type=PatternType.DOUBLE_TOP,
            symbol=symbol,
            timeframe=timeframe,
            start_time=peak1['time'],
            end_time=peak2['time'],
            confidence_score=confidence,
            strength=self._determine_strength(confidence),
            key_points=[(peak1['time'], peak1['price']), (valley['time'], valley['price']), (peak2['time'], peak2['price'])],
            pattern_metrics={'neckline': neckline, 'avg_peak': avg_peak},
            direction="bearish",
            target_price=target_price,
            stop_loss=avg_peak * 1.02,
            volume_confirmation=False
        )

    def _create_double_bottom_pattern(
        self, trough1: dict, trough2: dict, peak: dict, confidence: float,
        symbol: str, timeframe: str, df: pd.DataFrame
    ) -> DetectedPattern:
        """Create DetectedPattern for double bottom."""
        avg_trough = (trough1['price'] + trough2['price']) / 2
        neckline = peak['price']
        target_price = neckline + (neckline - avg_trough)

        return DetectedPattern(
            pattern_type=PatternType.DOUBLE_BOTTOM,
            symbol=symbol,
            timeframe=timeframe,
            start_time=trough1['time'],
            end_time=trough2['time'],
            confidence_score=confidence,
            strength=self._determine_strength(confidence),
            key_points=[(trough1['time'], trough1['price']), (peak['time'], peak['price']), (trough2['time'], trough2['price'])],
            pattern_metrics={'neckline': neckline, 'avg_trough': avg_trough},
            direction="bullish",
            target_price=target_price,
            stop_loss=avg_trough * 0.98,
            volume_confirmation=False
        )

    def _create_triple_top_pattern(
        self, peak1: dict, peak2: dict, peak3: dict, valley1: dict, valley2: dict,
        confidence: float, symbol: str, timeframe: str, df: pd.DataFrame
    ) -> DetectedPattern:
        """Create DetectedPattern for triple top."""
        avg_peak = (peak1['price'] + peak2['price'] + peak3['price']) / 3
        neckline = min(valley1['price'], valley2['price'])
        target_price = neckline - (avg_peak - neckline)

        return DetectedPattern(
            pattern_type=PatternType.TRIPLE_TOP,
            symbol=symbol,
            timeframe=timeframe,
            start_time=peak1['time'],
            end_time=peak3['time'],
            confidence_score=confidence,
            strength=self._determine_strength(confidence),
            key_points=[(peak1['time'], peak1['price']), (peak2['time'], peak2['price']), (peak3['time'], peak3['price'])],
            pattern_metrics={'neckline': neckline, 'avg_peak': avg_peak},
            direction="bearish",
            target_price=target_price,
            stop_loss=avg_peak * 1.02,
            volume_confirmation=False
        )

    def _create_triple_bottom_pattern(
        self, trough1: dict, trough2: dict, trough3: dict, peak1: dict, peak2: dict,
        confidence: float, symbol: str, timeframe: str, df: pd.DataFrame
    ) -> DetectedPattern:
        """Create DetectedPattern for triple bottom."""
        avg_trough = (trough1['price'] + trough2['price'] + trough3['price']) / 3
        neckline = max(peak1['price'], peak2['price'])
        target_price = neckline + (neckline - avg_trough)

        return DetectedPattern(
            pattern_type=PatternType.TRIPLE_BOTTOM,
            symbol=symbol,
            timeframe=timeframe,
            start_time=trough1['time'],
            end_time=trough3['time'],
            confidence_score=confidence,
            strength=self._determine_strength(confidence),
            key_points=[(trough1['time'], trough1['price']), (trough2['time'], trough2['price']), (trough3['time'], trough3['price'])],
            pattern_metrics={'neckline': neckline, 'avg_trough': avg_trough},
            direction="bullish",
            target_price=target_price,
            stop_loss=avg_trough * 0.98,
            volume_confirmation=False
        )

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
