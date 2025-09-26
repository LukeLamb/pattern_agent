"""
Pattern Validation Engine - Comprehensive pattern quality assessment.

This module provides validation and scoring systems for detected patterns including:
- Historical success rate calculation framework
- Volume confirmation validation
- Timeframe consistency checks
- Pattern strength scoring system
- Market context validation
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    from ..pattern_detection.pattern_engine import (
        DetectedPattern,
        PatternType,
        PatternStrength,
        PivotPoint,
        SupportResistanceLevel,
        TrendLine,
    )
    from ..models.market_data import MarketData
except ImportError:
    # For testing and development
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from pattern_detection.pattern_engine import (
        DetectedPattern,
        PatternType,
        PatternStrength,
        PivotPoint,
        SupportResistanceLevel,
        TrendLine,
    )
    from models.market_data import MarketData


class ValidationCriteria(Enum):
    """Validation criteria for pattern assessment."""

    VOLUME_CONFIRMATION = "volume_confirmation"
    TIMEFRAME_CONSISTENCY = "timeframe_consistency"
    HISTORICAL_SUCCESS_RATE = "historical_success_rate"
    PATTERN_QUALITY = "pattern_quality"
    MARKET_CONTEXT = "market_context"


@dataclass
class ValidationResult:
    """Result of pattern validation assessment."""

    pattern_id: str
    criteria: ValidationCriteria
    score: float  # 0.0 to 1.0
    passed: bool
    details: Dict[str, Union[str, float, bool]]
    validation_timestamp: datetime


@dataclass
class PatternQualityMetrics:
    """Comprehensive pattern quality metrics."""

    formation_time_score: float  # Time taken to form pattern
    symmetry_score: float  # Pattern symmetry and proportion
    volume_pattern_score: float  # Volume behavior during formation
    technical_strength_score: float  # Based on technical indicators
    market_context_score: float  # Market environment suitability
    overall_quality_score: float  # Weighted combination
    confidence_adjustment: float  # Adjustment factor for original confidence


class PatternValidator:
    """
    Comprehensive pattern validation and quality assessment engine.

    Provides systematic validation of detected patterns using multiple criteria
    including historical performance, volume analysis, technical context, and
    pattern formation quality metrics.
    """

    def __init__(
        self,
        min_formation_days: int = 5,
        max_formation_days: int = 60,
        min_volume_confirmation_ratio: float = 1.2,
        historical_lookback_days: int = 365,
    ):
        """Initialize Pattern Validator."""
        self.min_formation_days = min_formation_days
        self.max_formation_days = max_formation_days
        self.min_volume_confirmation_ratio = min_volume_confirmation_ratio
        self.historical_lookback_days = historical_lookback_days

        # Historical pattern performance database (would be loaded from storage)
        self._pattern_history: Dict[str, List[Dict]] = {}

        # Validation weights for overall scoring
        self.validation_weights = {
            ValidationCriteria.VOLUME_CONFIRMATION: 0.25,
            ValidationCriteria.TIMEFRAME_CONSISTENCY: 0.20,
            ValidationCriteria.HISTORICAL_SUCCESS_RATE: 0.30,
            ValidationCriteria.PATTERN_QUALITY: 0.15,
            ValidationCriteria.MARKET_CONTEXT: 0.10,
        }

    def validate_pattern(
        self, pattern: DetectedPattern, market_data: MarketData
    ) -> List[ValidationResult]:
        """
        Comprehensive pattern validation using all criteria.

        Args:
            pattern: DetectedPattern to validate
            market_data: MarketData for context analysis

        Returns:
            List of ValidationResult objects for each criteria
        """
        validation_results = []

        # Volume confirmation validation
        volume_result = self._validate_volume_confirmation(pattern, market_data)
        validation_results.append(volume_result)

        # Timeframe consistency validation
        timeframe_result = self._validate_timeframe_consistency(pattern, market_data)
        validation_results.append(timeframe_result)

        # Historical success rate validation
        historical_result = self._validate_historical_success_rate(pattern)
        validation_results.append(historical_result)

        # Pattern quality validation
        quality_result = self._validate_pattern_quality(pattern, market_data)
        validation_results.append(quality_result)

        # Market context validation
        context_result = self._validate_market_context(pattern, market_data)
        validation_results.append(context_result)

        return validation_results

    def calculate_overall_validation_score(
        self, validation_results: List[ValidationResult]
    ) -> float:
        """
        Calculate weighted overall validation score.

        Args:
            validation_results: List of individual validation results

        Returns:
            Overall validation score (0.0 to 1.0)
        """
        weighted_score = 0.0
        total_weight = 0.0

        for result in validation_results:
            if result.criteria in self.validation_weights:
                weight = self.validation_weights[result.criteria]
                weighted_score += result.score * weight
                total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def calculate_pattern_quality_metrics(
        self, pattern: DetectedPattern, market_data: MarketData
    ) -> PatternQualityMetrics:
        """
        Calculate comprehensive pattern quality metrics.

        Args:
            pattern: DetectedPattern to analyze
            market_data: MarketData for analysis context

        Returns:
            PatternQualityMetrics with detailed quality assessment
        """
        df = market_data.to_dataframe()

        # Formation time score
        formation_time_score = self._calculate_formation_time_score(pattern)

        # Symmetry score (pattern-specific)
        symmetry_score = self._calculate_symmetry_score(pattern)

        # Volume pattern score
        volume_pattern_score = self._calculate_volume_pattern_score(pattern, df)

        # Technical strength score
        technical_strength_score = self._calculate_technical_strength_score(df)

        # Market context score
        market_context_score = self._calculate_market_context_score(pattern, df)

        # Overall quality score (weighted combination)
        overall_quality_score = (
            formation_time_score * 0.20
            + symmetry_score * 0.25
            + volume_pattern_score * 0.20
            + technical_strength_score * 0.20
            + market_context_score * 0.15
        )

        # Confidence adjustment factor
        confidence_adjustment = min(1.5, max(0.5, overall_quality_score * 1.2))

        return PatternQualityMetrics(
            formation_time_score=formation_time_score,
            symmetry_score=symmetry_score,
            volume_pattern_score=volume_pattern_score,
            technical_strength_score=technical_strength_score,
            market_context_score=market_context_score,
            overall_quality_score=overall_quality_score,
            confidence_adjustment=confidence_adjustment,
        )

    # Private validation methods

    def _validate_volume_confirmation(
        self, pattern: DetectedPattern, market_data: MarketData
    ) -> ValidationResult:
        """Validate volume confirmation for pattern."""
        df = market_data.to_dataframe()

        # Calculate average volume before and during pattern formation
        # Find the closest timestamps and get positions
        pattern_data = df[
            (df.index >= pattern.start_time) & (df.index <= pattern.end_time)
        ]

        if len(pattern_data) > 0:
            # Volume during pattern formation
            pattern_volume = pattern_data["volume"].mean()

            # Get the position range for lookback
            pattern_length = len(pattern_data)
            pattern_start_pos = df.index.get_loc(pattern_data.index[0])
            if isinstance(pattern_start_pos, slice):
                pattern_start_pos = pattern_start_pos.start or 0
            elif isinstance(pattern_start_pos, np.ndarray):
                pattern_start_pos = np.nonzero(pattern_start_pos)[0][0]

            # Average volume before pattern (same period length)
            if pattern_start_pos >= pattern_length:
                pre_pattern_data = df.iloc[
                    pattern_start_pos - pattern_length : pattern_start_pos
                ]
                pre_pattern_volume = pre_pattern_data["volume"].mean()
            else:
                pre_pattern_data = df.iloc[:pattern_start_pos]
                pre_pattern_volume = (
                    pre_pattern_data["volume"].mean()
                    if len(pre_pattern_data) > 0
                    else pattern_volume
                )
        else:
            # Fallback if no pattern data found
            pattern_volume = df["volume"].mean()
            pre_pattern_volume = pattern_volume

        # Calculate volume confirmation ratio
        volume_ratio = (
            pattern_volume / pre_pattern_volume if pre_pattern_volume > 0 else 1.0
        )

        # Score based on volume confirmation
        if volume_ratio >= self.min_volume_confirmation_ratio * 1.5:
            score = 1.0
        elif volume_ratio >= self.min_volume_confirmation_ratio:
            score = 0.8
        elif volume_ratio >= 1.0:
            score = 0.6
        else:
            score = 0.3

        passed = bool(volume_ratio >= self.min_volume_confirmation_ratio)

        details = {
            "volume_ratio": volume_ratio,
            "pattern_volume": pattern_volume,
            "pre_pattern_volume": pre_pattern_volume,
            "threshold": self.min_volume_confirmation_ratio,
        }

        return ValidationResult(
            pattern_id=pattern.symbol,
            criteria=ValidationCriteria.VOLUME_CONFIRMATION,
            score=score,
            passed=passed,
            details=details,
            validation_timestamp=datetime.now(),
        )

    def _validate_timeframe_consistency(
        self, pattern: DetectedPattern, market_data: MarketData
    ) -> ValidationResult:
        """Validate timeframe consistency for pattern formation."""
        # Calculate pattern formation duration
        time_delta = pattern.end_time - pattern.start_time
        if isinstance(time_delta, pd.Timedelta):
            formation_duration = time_delta.days
        else:
            # Handle numpy.timedelta64 objects
            formation_duration = pd.Timedelta(time_delta).days

        # Score based on formation time appropriateness
        if self.min_formation_days <= formation_duration <= self.max_formation_days:
            if formation_duration <= self.max_formation_days * 0.5:
                score = 1.0  # Optimal formation time
            else:
                score = 0.8  # Good formation time
        elif formation_duration < self.min_formation_days:
            score = 0.4  # Too fast formation
        else:
            score = 0.3  # Too slow formation

        passed = bool(
            self.min_formation_days <= formation_duration <= self.max_formation_days
        )

        details = {
            "formation_days": formation_duration,
            "min_threshold": self.min_formation_days,
            "max_threshold": self.max_formation_days,
            "is_optimal": formation_duration <= self.max_formation_days * 0.5,
        }

        return ValidationResult(
            pattern_id=pattern.symbol,
            criteria=ValidationCriteria.TIMEFRAME_CONSISTENCY,
            score=score,
            passed=passed,
            details=details,
            validation_timestamp=datetime.now(),
        )

    def _validate_historical_success_rate(
        self, pattern: DetectedPattern
    ) -> ValidationResult:
        """Validate based on historical success rate of similar patterns."""
        pattern_type_key = f"{pattern.pattern_type.value}_{pattern.strength.value}"

        # Get historical data for this pattern type (placeholder - would load from database)
        historical_patterns = self._pattern_history.get(pattern_type_key, [])

        if len(historical_patterns) < 5:
            # Insufficient historical data - use moderate score
            score = 0.6
            success_rate = 0.6
            passed = True
        else:
            # Calculate success rate from historical patterns
            successful_patterns = sum(
                1 for p in historical_patterns if p.get("success", False)
            )
            success_rate = successful_patterns / len(historical_patterns)

            # Score based on historical success rate
            if success_rate >= 0.8:
                score = 1.0
            elif success_rate >= 0.6:
                score = 0.8
            elif success_rate >= 0.4:
                score = 0.6
            else:
                score = 0.3

            passed = bool(success_rate >= 0.4)  # Minimum acceptable success rate

        details = {
            "historical_count": len(historical_patterns),
            "success_rate": success_rate,
            "pattern_type": pattern_type_key,
            "sufficient_history": len(historical_patterns) >= 5,
        }

        return ValidationResult(
            pattern_id=pattern.symbol,
            criteria=ValidationCriteria.HISTORICAL_SUCCESS_RATE,
            score=score,
            passed=passed,
            details=details,
            validation_timestamp=datetime.now(),
        )

    def _validate_pattern_quality(
        self, pattern: DetectedPattern, market_data: MarketData
    ) -> ValidationResult:
        """Validate overall pattern quality and formation."""
        quality_metrics = self.calculate_pattern_quality_metrics(pattern, market_data)

        score = quality_metrics.overall_quality_score
        passed = bool(score >= 0.6)  # Quality threshold

        details = {
            "formation_time_score": quality_metrics.formation_time_score,
            "symmetry_score": quality_metrics.symmetry_score,
            "volume_pattern_score": quality_metrics.volume_pattern_score,
            "technical_strength_score": quality_metrics.technical_strength_score,
            "market_context_score": quality_metrics.market_context_score,
            "confidence_adjustment": quality_metrics.confidence_adjustment,
        }

        return ValidationResult(
            pattern_id=pattern.symbol,
            criteria=ValidationCriteria.PATTERN_QUALITY,
            score=score,
            passed=passed,
            details=details,
            validation_timestamp=datetime.now(),
        )

    def _validate_market_context(
        self, pattern: DetectedPattern, market_data: MarketData
    ) -> ValidationResult:
        """Validate pattern within current market context."""
        df = market_data.to_dataframe()

        # Calculate market volatility
        returns = df["close"].pct_change().dropna()
        volatility = (
            returns.rolling(window=20).std().iloc[-1]
            if len(returns) >= 20
            else returns.std()
        )

        # Calculate trend strength
        sma_20 = df["close"].rolling(window=20).mean()
        sma_50 = df["close"].rolling(window=50).mean()
        trend_strength = (
            abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            if len(df) >= 50
            else 0.02
        )

        # Score based on market conditions suitability for pattern type
        volatility_score = self._score_volatility_for_pattern(pattern, volatility)
        trend_score = self._score_trend_for_pattern(pattern, trend_strength, df)

        score = (volatility_score + trend_score) / 2
        passed = bool(score >= 0.5)

        details = {
            "volatility": volatility,
            "trend_strength": trend_strength,
            "volatility_score": volatility_score,
            "trend_score": trend_score,
            "market_regime": "trending" if trend_strength > 0.05 else "ranging",
        }

        return ValidationResult(
            pattern_id=pattern.symbol,
            criteria=ValidationCriteria.MARKET_CONTEXT,
            score=score,
            passed=passed,
            details=details,
            validation_timestamp=datetime.now(),
        )

    # Private helper methods for quality metrics calculation

    def _calculate_formation_time_score(self, pattern: DetectedPattern) -> float:
        """Calculate formation time appropriateness score."""
        time_delta = pattern.end_time - pattern.start_time
        if isinstance(time_delta, pd.Timedelta):
            formation_days = time_delta.days
        else:
            # Handle numpy.timedelta64 objects
            formation_days = pd.Timedelta(time_delta).days

        if formation_days < self.min_formation_days:
            return 0.4  # Too fast
        elif formation_days > self.max_formation_days:
            return 0.2  # Too slow
        elif formation_days <= self.max_formation_days * 0.6:
            return 1.0  # Optimal formation time
        else:
            return 0.8  # Acceptable formation time

    def _calculate_symmetry_score(self, pattern: DetectedPattern) -> float:
        """Calculate pattern symmetry and proportion score."""
        # This would be pattern-specific - placeholder implementation
        if pattern.pattern_type == PatternType.HEAD_AND_SHOULDERS:
            # For H&S, check shoulder symmetry
            return self._calculate_hs_symmetry(pattern)
        elif pattern.pattern_type in [
            PatternType.ASCENDING_TRIANGLE,
            PatternType.DESCENDING_TRIANGLE,
            PatternType.SYMMETRICAL_TRIANGLE,
        ]:
            return self._calculate_triangle_symmetry(pattern)
        else:
            return 0.7  # Default moderate score

    def _calculate_volume_pattern_score(
        self, pattern: DetectedPattern, df: pd.DataFrame
    ) -> float:
        """Calculate volume behavior appropriateness score."""
        # Check if volume behaves as expected for pattern type
        pattern_data = df[
            (df.index >= pattern.start_time) & (df.index <= pattern.end_time)
        ]

        if len(pattern_data) > 0:
            pattern_volume = pattern_data["volume"]
        else:
            # Fallback to small sample if no exact match
            pattern_volume = df["volume"].iloc[-10:]

        # Volume should generally decrease during consolidation patterns
        volume_trend = np.corrcoef(range(len(pattern_volume)), pattern_volume)[0, 1]

        if pattern.pattern_type in [
            PatternType.ASCENDING_TRIANGLE,
            PatternType.DESCENDING_TRIANGLE,
            PatternType.SYMMETRICAL_TRIANGLE,
        ]:
            # Triangles typically show decreasing volume
            if volume_trend < -0.3:
                return 1.0
            elif volume_trend < 0:
                return 0.7
            else:
                return 0.4
        elif pattern.pattern_type == PatternType.HEAD_AND_SHOULDERS:
            # H&S patterns often show volume patterns
            return self._calculate_hs_volume_score(pattern)
        else:
            return 0.6  # Default moderate score

    def _calculate_technical_strength_score(self, df: pd.DataFrame) -> float:
        """Calculate technical strength based on indicators."""
        # RSI divergence, support/resistance strength, etc.
        rsi = self._calculate_rsi(df["close"])

        # Check for divergence (simplified)
        price_trend = df["close"].iloc[-10:].diff().mean()
        rsi_trend = rsi.iloc[-10:].diff().mean()

        divergence_score = (
            1.0
            if (price_trend > 0 and rsi_trend < 0)
            or (price_trend < 0 and rsi_trend > 0)
            else 0.6
        )

        return min(1.0, max(0.2, divergence_score))

    def _calculate_market_context_score(
        self, pattern: DetectedPattern, df: pd.DataFrame
    ) -> float:
        """Calculate market context appropriateness score."""
        # Market regime suitability
        volatility = df["close"].pct_change().rolling(window=20).std().iloc[-1]

        # Some patterns work better in different market conditions
        if pattern.pattern_type in [
            PatternType.ASCENDING_TRIANGLE,
            PatternType.DESCENDING_TRIANGLE,
        ]:
            # Triangles work well in moderate volatility
            return 1.0 if 0.01 < volatility < 0.03 else 0.7
        else:
            return 0.7  # Default moderate score

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        # Convert to float to ensure proper operations
        delta = delta.astype(float)
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)  # Add small value to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _score_volatility_for_pattern(
        self, pattern: DetectedPattern, volatility: float
    ) -> float:
        """Score volatility suitability for pattern type."""
        if pattern.pattern_type in [
            PatternType.ASCENDING_TRIANGLE,
            PatternType.DESCENDING_TRIANGLE,
            PatternType.SYMMETRICAL_TRIANGLE,
        ]:
            # Triangles prefer moderate volatility
            return 1.0 if 0.01 < volatility < 0.04 else 0.6
        elif pattern.pattern_type == PatternType.HEAD_AND_SHOULDERS:
            # H&S can work in various volatility regimes
            return 0.8
        else:
            return 0.7

    def _score_trend_for_pattern(
        self, pattern: DetectedPattern, trend_strength: float, df: pd.DataFrame
    ) -> float:
        """Score trend conditions for pattern type."""
        if pattern.pattern_type == PatternType.ASCENDING_TRIANGLE:
            # Ascending triangles prefer uptrend context
            return (
                1.0
                if trend_strength > 0.02
                and df["close"].iloc[-1] > df["close"].iloc[-20]
                else 0.6
            )
        elif pattern.pattern_type == PatternType.DESCENDING_TRIANGLE:
            # Descending triangles prefer downtrend context
            return (
                1.0
                if trend_strength > 0.02
                and df["close"].iloc[-1] < df["close"].iloc[-20]
                else 0.6
            )
        else:
            return 0.7

    def _calculate_hs_symmetry(self, pattern: DetectedPattern) -> float:
        """Calculate Head and Shoulders pattern symmetry."""
        # Placeholder - would analyze shoulder heights and positions
        return 0.8

    def _calculate_triangle_symmetry(self, pattern: DetectedPattern) -> float:
        """Calculate triangle pattern symmetry."""
        # Placeholder - would analyze trendline convergence
        return 0.8

    def _calculate_hs_volume_score(self, pattern: DetectedPattern) -> float:
        """Calculate H&S volume pattern score."""
        # Placeholder - would analyze volume at head vs shoulders
        return 0.7
