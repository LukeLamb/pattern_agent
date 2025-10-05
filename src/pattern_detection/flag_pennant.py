"""
Flag and Pennant Pattern Detection

This module provides detection algorithms for continuation patterns:
- Bull Flags (uptrend consolidation with downward drift)
- Bear Flags (downtrend consolidation with upward drift)
- Pennants (symmetrical triangle after sharp move)

Flags and pennants are short-term continuation patterns that form after a sharp move.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    from .pattern_engine import (
        DetectedPattern,
        PatternType,
        PatternStrength,
        TrendLine,
        PivotPoint,
    )
    from ..models.market_data import MarketData
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from pattern_detection.pattern_engine import (
        DetectedPattern,
        PatternType,
        PatternStrength,
        TrendLine,
        PivotPoint,
    )
    from models.market_data import MarketData


@dataclass
class FlagPole:
    """Represents the sharp move before flag/pennant formation."""
    start_price: float
    end_price: float
    start_time: datetime
    end_time: datetime
    direction: str  # 'up' or 'down'
    magnitude: float  # Percentage move
    duration_days: float


class FlagPennantDetector:
    """
    Specialized detector for flag and pennant patterns.

    These are continuation patterns that form after a sharp price move (flagpole).
    Flags show parallel consolidation, pennants show converging consolidation.
    """

    def __init__(
        self,
        min_flagpole_move: float = 0.08,  # 8% minimum move for flagpole
        max_flag_duration_days: int = 21,  # Flags are typically 1-3 weeks
        min_flag_duration_days: int = 5,
        volume_decline_threshold: float = 0.7,  # Volume should decline 30%+ during flag
    ):
        """
        Initialize Flag & Pennant Detector.

        Args:
            min_flagpole_move: Minimum % move to qualify as flagpole
            max_flag_duration_days: Maximum days for flag formation
            min_flag_duration_days: Minimum days for flag formation
            volume_decline_threshold: Volume during flag vs flagpole (0.7 = 30% decline)
        """
        self.min_flagpole_move = min_flagpole_move
        self.max_flag_duration_days = max_flag_duration_days
        self.min_flag_duration_days = min_flag_duration_days
        self.volume_decline_threshold = volume_decline_threshold

    def detect_bull_flags(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "daily"
    ) -> List[DetectedPattern]:
        """
        Detect bull flag patterns.

        Bull flag characteristics:
        - Sharp upward move (flagpole)
        - Downward sloping consolidation (flag)
        - Volume declines during flag
        - Breakout continuation upward

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            List of detected bull flag patterns
        """
        patterns = []

        # Ensure we have timestamp column
        if 'timestamp' not in df.columns and df.index.name != 'timestamp':
            return patterns

        # Look for sharp upward moves (potential flagpoles)
        flagpoles = self._find_flagpoles(df, direction='up')

        for flagpole in flagpoles:
            # Analyze consolidation after flagpole
            flag_pattern = self._validate_bull_flag(df, flagpole, symbol, timeframe)
            if flag_pattern:
                patterns.append(flag_pattern)

        return patterns

    def detect_bear_flags(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "daily"
    ) -> List[DetectedPattern]:
        """
        Detect bear flag patterns.

        Bear flag characteristics:
        - Sharp downward move (flagpole)
        - Upward sloping consolidation (flag)
        - Volume declines during flag
        - Breakout continuation downward

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            List of detected bear flag patterns
        """
        patterns = []

        # Ensure we have timestamp column
        if 'timestamp' not in df.columns and df.index.name != 'timestamp':
            return patterns

        # Look for sharp downward moves (potential flagpoles)
        flagpoles = self._find_flagpoles(df, direction='down')

        for flagpole in flagpoles:
            # Analyze consolidation after flagpole
            flag_pattern = self._validate_bear_flag(df, flagpole, symbol, timeframe)
            if flag_pattern:
                patterns.append(flag_pattern)

        return patterns

    def detect_pennants(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "daily"
    ) -> List[DetectedPattern]:
        """
        Detect pennant patterns.

        Pennant characteristics:
        - Sharp move in either direction (flagpole)
        - Symmetrical triangle consolidation (pennant)
        - Much briefer than flags (1-3 weeks max)
        - Volume declines during pennant
        - Breakout in direction of flagpole

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            List of detected pennant patterns
        """
        patterns = []

        # Ensure we have timestamp column
        if 'timestamp' not in df.columns and df.index.name != 'timestamp':
            return patterns

        # Look for sharp moves in both directions
        flagpoles = self._find_flagpoles(df, direction='both')

        for flagpole in flagpoles:
            # Analyze symmetrical consolidation after flagpole
            pennant_pattern = self._validate_pennant(df, flagpole, symbol, timeframe)
            if pennant_pattern:
                patterns.append(pennant_pattern)

        return patterns

    def _find_flagpoles(
        self,
        df: pd.DataFrame,
        direction: str = 'both'
    ) -> List[FlagPole]:
        """
        Identify sharp price moves that could be flagpoles.

        Args:
            df: DataFrame with price data
            direction: 'up', 'down', or 'both'

        Returns:
            List of potential flagpole moves
        """
        flagpoles = []

        # Use rolling window to find sharp moves
        window = min(10, len(df) // 4)  # Adaptive window size

        for i in range(window, len(df) - self.min_flag_duration_days):
            # Calculate move over window
            start_price = df['close'].iloc[i - window]
            end_price = df['close'].iloc[i]
            move_pct = (end_price - start_price) / start_price

            # Check if move is significant enough
            if abs(move_pct) < self.min_flagpole_move:
                continue

            # Check direction
            if direction == 'up' and move_pct <= 0:
                continue
            if direction == 'down' and move_pct >= 0:
                continue

            # Create flagpole
            flagpole = FlagPole(
                start_price=start_price,
                end_price=end_price,
                start_time=self._get_timestamp(df, i - window),
                end_time=self._get_timestamp(df, i),
                direction='up' if move_pct > 0 else 'down',
                magnitude=abs(move_pct),
                duration_days=(self._get_timestamp(df, i) - self._get_timestamp(df, i - window)).days
            )

            flagpoles.append(flagpole)

        return flagpoles

    def _validate_bull_flag(
        self,
        df: pd.DataFrame,
        flagpole: FlagPole,
        symbol: str,
        timeframe: str
    ) -> Optional[DetectedPattern]:
        """Validate and create bull flag pattern."""

        # Find flag period (after flagpole)
        flagpole_end_idx = self._find_index_by_timestamp(df, flagpole.end_time)
        if flagpole_end_idx is None or flagpole_end_idx >= len(df) - self.min_flag_duration_days:
            return None

        # Analyze consolidation period
        max_flag_idx = min(flagpole_end_idx + self.max_flag_duration_days, len(df))
        flag_df = df.iloc[flagpole_end_idx:max_flag_idx].copy()

        if len(flag_df) < self.min_flag_duration_days:
            return None

        # Bull flag should drift downward or sideways during consolidation
        flag_slope = self._calculate_price_slope(flag_df)
        if flag_slope > 0.02:  # Too much upward drift
            return None

        # Check volume decline
        flagpole_volume = df.iloc[flagpole_end_idx - 5:flagpole_end_idx]['volume'].mean()
        flag_volume = flag_df['volume'].mean()

        if flag_volume / flagpole_volume > self.volume_decline_threshold:
            return None  # Volume didn't decline enough

        # Calculate pattern metrics
        flag_high = flag_df['high'].max()
        flag_low = flag_df['low'].min()
        flag_range = (flag_high - flag_low) / flagpole.end_price

        # Flag should be relatively narrow (consolidation)
        if flag_range > 0.15:  # More than 15% range is too wide
            return None

        # Calculate confidence score
        confidence = self._calculate_flag_confidence(
            flagpole_magnitude=flagpole.magnitude,
            flag_slope=abs(flag_slope),
            volume_ratio=flag_volume / flagpole_volume,
            flag_duration=len(flag_df)
        )

        # Determine pattern strength
        strength = self._determine_pattern_strength(confidence)

        # Calculate target price (flagpole height projected from flag top)
        target_price = flag_high + (flagpole.end_price - flagpole.start_price)
        stop_loss = flag_low * 0.98  # 2% below flag low

        # Create detected pattern
        pattern = DetectedPattern(
            pattern_type=PatternType.BULL_FLAG,
            symbol=symbol,
            timeframe=timeframe,
            start_time=flagpole.start_time,
            end_time=self._get_timestamp(df, max_flag_idx - 1),
            confidence_score=confidence,
            strength=strength,
            key_points=[
                (flagpole.start_time, flagpole.start_price),
                (flagpole.end_time, flagpole.end_price),
                (self._get_timestamp(df, max_flag_idx - 1), flag_df['close'].iloc[-1])
            ],
            pattern_metrics={
                'flagpole_magnitude': flagpole.magnitude,
                'flagpole_duration_days': flagpole.duration_days,
                'flag_duration_days': len(flag_df),
                'flag_slope': flag_slope,
                'volume_decline': 1.0 - (flag_volume / flagpole_volume),
                'flag_range_pct': flag_range
            },
            direction="bullish",
            target_price=target_price,
            stop_loss=stop_loss,
            volume_confirmation=True
        )

        return pattern

    def _validate_bear_flag(
        self,
        df: pd.DataFrame,
        flagpole: FlagPole,
        symbol: str,
        timeframe: str
    ) -> Optional[DetectedPattern]:
        """Validate and create bear flag pattern."""

        # Find flag period (after flagpole)
        flagpole_end_idx = self._find_index_by_timestamp(df, flagpole.end_time)
        if flagpole_end_idx is None or flagpole_end_idx >= len(df) - self.min_flag_duration_days:
            return None

        # Analyze consolidation period
        max_flag_idx = min(flagpole_end_idx + self.max_flag_duration_days, len(df))
        flag_df = df.iloc[flagpole_end_idx:max_flag_idx].copy()

        if len(flag_df) < self.min_flag_duration_days:
            return None

        # Bear flag should drift upward or sideways during consolidation
        flag_slope = self._calculate_price_slope(flag_df)
        if flag_slope < -0.02:  # Too much downward drift
            return None

        # Check volume decline
        flagpole_volume = df.iloc[flagpole_end_idx - 5:flagpole_end_idx]['volume'].mean()
        flag_volume = flag_df['volume'].mean()

        if flag_volume / flagpole_volume > self.volume_decline_threshold:
            return None  # Volume didn't decline enough

        # Calculate pattern metrics
        flag_high = flag_df['high'].max()
        flag_low = flag_df['low'].min()
        flag_range = (flag_high - flag_low) / flagpole.end_price

        # Flag should be relatively narrow
        if flag_range > 0.15:
            return None

        # Calculate confidence score
        confidence = self._calculate_flag_confidence(
            flagpole_magnitude=flagpole.magnitude,
            flag_slope=abs(flag_slope),
            volume_ratio=flag_volume / flagpole_volume,
            flag_duration=len(flag_df)
        )

        # Determine pattern strength
        strength = self._determine_pattern_strength(confidence)

        # Calculate target price (flagpole height projected from flag bottom)
        target_price = flag_low - (flagpole.start_price - flagpole.end_price)
        stop_loss = flag_high * 1.02  # 2% above flag high

        # Create detected pattern
        pattern = DetectedPattern(
            pattern_type=PatternType.BEAR_FLAG,
            symbol=symbol,
            timeframe=timeframe,
            start_time=flagpole.start_time,
            end_time=self._get_timestamp(df, max_flag_idx - 1),
            confidence_score=confidence,
            strength=strength,
            key_points=[
                (flagpole.start_time, flagpole.start_price),
                (flagpole.end_time, flagpole.end_price),
                (self._get_timestamp(df, max_flag_idx - 1), flag_df['close'].iloc[-1])
            ],
            pattern_metrics={
                'flagpole_magnitude': flagpole.magnitude,
                'flagpole_duration_days': flagpole.duration_days,
                'flag_duration_days': len(flag_df),
                'flag_slope': flag_slope,
                'volume_decline': 1.0 - (flag_volume / flagpole_volume),
                'flag_range_pct': flag_range
            },
            direction="bearish",
            target_price=target_price,
            stop_loss=stop_loss,
            volume_confirmation=True
        )

        return pattern

    def _validate_pennant(
        self,
        df: pd.DataFrame,
        flagpole: FlagPole,
        symbol: str,
        timeframe: str
    ) -> Optional[DetectedPattern]:
        """Validate and create pennant pattern."""

        # Find pennant period (after flagpole)
        flagpole_end_idx = self._find_index_by_timestamp(df, flagpole.end_time)
        if flagpole_end_idx is None or flagpole_end_idx >= len(df) - self.min_flag_duration_days:
            return None

        # Pennants are briefer than flags
        max_pennant_days = min(self.max_flag_duration_days, 15)  # Max 15 days
        max_pennant_idx = min(flagpole_end_idx + max_pennant_days, len(df))
        pennant_df = df.iloc[flagpole_end_idx:max_pennant_idx].copy()

        if len(pennant_df) < self.min_flag_duration_days:
            return None

        # Pennant should show converging highs and lows (symmetrical triangle)
        convergence = self._check_convergence(pennant_df)
        if not convergence:
            return None

        # Check volume decline
        flagpole_volume = df.iloc[flagpole_end_idx - 5:flagpole_end_idx]['volume'].mean()
        pennant_volume = pennant_df['volume'].mean()

        if pennant_volume / flagpole_volume > self.volume_decline_threshold:
            return None

        # Calculate confidence
        confidence = self._calculate_pennant_confidence(
            flagpole_magnitude=flagpole.magnitude,
            convergence_quality=0.75,  # From convergence check
            volume_ratio=pennant_volume / flagpole_volume,
            pennant_duration=len(pennant_df)
        )

        # Determine pattern strength
        strength = self._determine_pattern_strength(confidence)

        # Calculate target price
        direction = flagpole.direction
        pennant_midpoint = (pennant_df['high'].max() + pennant_df['low'].min()) / 2

        if direction == 'up':
            target_price = pennant_midpoint + (flagpole.end_price - flagpole.start_price)
            stop_loss = pennant_df['low'].min() * 0.98
            pattern_direction = "bullish"
        else:
            target_price = pennant_midpoint - (flagpole.start_price - flagpole.end_price)
            stop_loss = pennant_df['high'].max() * 1.02
            pattern_direction = "bearish"

        # Create detected pattern
        pattern = DetectedPattern(
            pattern_type=PatternType.PENNANT,
            symbol=symbol,
            timeframe=timeframe,
            start_time=flagpole.start_time,
            end_time=self._get_timestamp(df, max_pennant_idx - 1),
            confidence_score=confidence,
            strength=strength,
            key_points=[
                (flagpole.start_time, flagpole.start_price),
                (flagpole.end_time, flagpole.end_price),
                (self._get_timestamp(df, max_pennant_idx - 1), pennant_df['close'].iloc[-1])
            ],
            pattern_metrics={
                'flagpole_magnitude': flagpole.magnitude,
                'flagpole_duration_days': flagpole.duration_days,
                'pennant_duration_days': len(pennant_df),
                'volume_decline': 1.0 - (pennant_volume / flagpole_volume),
                'flagpole_direction': direction
            },
            direction=pattern_direction,
            target_price=target_price,
            stop_loss=stop_loss,
            volume_confirmation=True
        )

        return pattern

    def _calculate_price_slope(self, df: pd.DataFrame) -> float:
        """Calculate the slope of prices over the dataframe."""
        if len(df) < 2:
            return 0.0

        prices = df['close'].values
        x = np.arange(len(prices))

        # Linear regression
        coeffs = np.polyfit(x, prices, 1)
        slope = coeffs[0] / prices[0]  # Normalize by starting price

        return slope

    def _check_convergence(self, df: pd.DataFrame) -> bool:
        """Check if price range is converging (for pennants)."""
        if len(df) < 5:
            return False

        # Split into first and second half
        mid = len(df) // 2
        first_half = df.iloc[:mid]
        second_half = df.iloc[mid:]

        # Calculate ranges
        first_range = first_half['high'].max() - first_half['low'].min()
        second_range = second_half['high'].max() - second_half['low'].min()

        # Second half should be narrower (converging)
        if second_range >= first_range * 0.8:  # Allow some tolerance
            return False

        return True

    def _calculate_flag_confidence(
        self,
        flagpole_magnitude: float,
        flag_slope: float,
        volume_ratio: float,
        flag_duration: int
    ) -> float:
        """Calculate confidence score for flag pattern."""
        confidence = 0.5  # Base confidence

        # Stronger flagpole increases confidence
        if flagpole_magnitude > 0.15:  # >15% move
            confidence += 0.15
        elif flagpole_magnitude > 0.10:
            confidence += 0.10

        # Proper consolidation slope
        if flag_slope < 0.01:  # Very flat
            confidence += 0.10
        elif flag_slope < 0.03:
            confidence += 0.05

        # Good volume decline
        if volume_ratio < 0.5:  # >50% volume decline
            confidence += 0.15
        elif volume_ratio < 0.7:
            confidence += 0.10

        # Appropriate duration
        if 5 <= flag_duration <= 15:  # Ideal duration
            confidence += 0.10
        elif flag_duration <= 21:
            confidence += 0.05

        return min(confidence, 1.0)

    def _calculate_pennant_confidence(
        self,
        flagpole_magnitude: float,
        convergence_quality: float,
        volume_ratio: float,
        pennant_duration: int
    ) -> float:
        """Calculate confidence score for pennant pattern."""
        confidence = 0.5  # Base confidence

        # Strong flagpole
        if flagpole_magnitude > 0.15:
            confidence += 0.15
        elif flagpole_magnitude > 0.10:
            confidence += 0.10

        # Good convergence
        confidence += convergence_quality * 0.15

        # Volume decline
        if volume_ratio < 0.5:
            confidence += 0.15
        elif volume_ratio < 0.7:
            confidence += 0.10

        # Brief formation (pennants are quick)
        if pennant_duration <= 10:
            confidence += 0.10
        elif pennant_duration <= 15:
            confidence += 0.05

        return min(confidence, 1.0)

    def _determine_pattern_strength(self, confidence: float) -> PatternStrength:
        """Determine pattern strength from confidence score."""
        if confidence >= 0.80:
            return PatternStrength.VERY_STRONG
        elif confidence >= 0.70:
            return PatternStrength.STRONG
        elif confidence >= 0.60:
            return PatternStrength.MODERATE
        else:
            return PatternStrength.WEAK

    def _get_timestamp(self, df: pd.DataFrame, idx: int) -> datetime:
        """Get timestamp from dataframe at index."""
        if 'timestamp' in df.columns:
            return df['timestamp'].iloc[idx]
        elif df.index.name == 'timestamp':
            return df.index[idx]
        else:
            # Fallback to creating timestamps
            return datetime.now() - timedelta(days=len(df) - idx)

    def _find_index_by_timestamp(self, df: pd.DataFrame, timestamp: datetime) -> Optional[int]:
        """Find dataframe index closest to given timestamp."""
        if 'timestamp' in df.columns:
            # Find closest timestamp
            time_diffs = (df['timestamp'] - timestamp).abs()
            return time_diffs.idxmin()
        elif df.index.name == 'timestamp':
            time_diffs = (df.index - timestamp).abs()
            return time_diffs.argmin()
        else:
            return None
