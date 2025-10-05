"""
Enhanced Pattern Validator - Phase 2.4

Context-aware pattern validation with market regime integration.
Extends the base PatternValidator with adaptive confidence scoring.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

try:
    from ..pattern_detection.pattern_engine import DetectedPattern, PatternType
    from ..models.market_data import MarketData
    from ..market_context import (
        MarketContext,
        MarketRegime,
        VolatilityRegime,
        TrendDirection,
    )
    from .pattern_validator import PatternValidator, ValidationResult
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from pattern_detection.pattern_engine import DetectedPattern, PatternType
    from models.market_data import MarketData
    from market_context import (
        MarketContext,
        MarketRegime,
        VolatilityRegime,
        TrendDirection,
    )
    from pattern_validator import PatternValidator, ValidationResult


@dataclass
class EnhancedValidationResult:
    """
    Enhanced validation result with market context integration.

    Extends base validation with context-aware confidence adjustment.
    """
    # Base validation fields
    pattern_id: str
    symbol: str
    pattern_type: PatternType
    base_confidence: float  # Original confidence (0-1)
    is_valid: bool

    # Context-aware fields (Phase 2.4)
    market_context: Optional[MarketContext] = None
    adjusted_confidence: float = 0.0  # After context adjustment
    regime_affinity: float = 0.0      # Pattern-regime match score (0-1)
    context_boost: float = 0.0        # Confidence adjustment (+/-)

    # Scoring breakdown
    volume_score: float = 0.0
    timeframe_score: float = 0.0
    historical_score: float = 0.0
    quality_score: float = 0.0
    context_score: float = 0.0

    # Enhanced recommendations
    recommendation_strength: str = "MODERATE"  # WEAK/MODERATE/STRONG/VERY_STRONG
    supporting_reasons: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)

    # Metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)


# Pattern-Regime Affinity Matrix
# Maps how well each pattern performs in different market regimes
PATTERN_REGIME_AFFINITY: Dict[PatternType, Dict[MarketRegime, float]] = {
    # Continuation patterns favor trending markets
    PatternType.BULL_FLAG: {
        MarketRegime.TRENDING_BULL: 1.0,   # Perfect match
        MarketRegime.BREAKOUT: 0.8,         # Good in breakouts
        MarketRegime.RANGE_BOUND: 0.3,      # Poor in range
        MarketRegime.TRENDING_BEAR: 0.0,    # Incompatible
        MarketRegime.VOLATILE: 0.4,         # Risky in volatility
    },

    PatternType.BEAR_FLAG: {
        MarketRegime.TRENDING_BEAR: 1.0,
        MarketRegime.BREAKOUT: 0.8,
        MarketRegime.RANGE_BOUND: 0.3,
        MarketRegime.TRENDING_BULL: 0.0,
        MarketRegime.VOLATILE: 0.4,
    },

    PatternType.PENNANT: {
        MarketRegime.TRENDING_BULL: 0.8,
        MarketRegime.TRENDING_BEAR: 0.8,
        MarketRegime.BREAKOUT: 0.9,         # Great for breakouts
        MarketRegime.RANGE_BOUND: 0.4,
        MarketRegime.VOLATILE: 0.5,
    },

    # Reversal patterns favor range-bound or end-of-trend
    PatternType.DOUBLE_TOP: {
        MarketRegime.RANGE_BOUND: 1.0,      # Perfect for tops
        MarketRegime.TRENDING_BULL: 0.8,    # Top at end of uptrend
        MarketRegime.BREAKOUT: 0.5,
        MarketRegime.TRENDING_BEAR: 0.2,    # Already trending down
        MarketRegime.VOLATILE: 0.6,
    },

    PatternType.DOUBLE_BOTTOM: {
        MarketRegime.RANGE_BOUND: 1.0,
        MarketRegime.TRENDING_BEAR: 0.8,    # Bottom at end of downtrend
        MarketRegime.BREAKOUT: 0.5,
        MarketRegime.TRENDING_BULL: 0.2,
        MarketRegime.VOLATILE: 0.6,
    },

    PatternType.TRIPLE_TOP: {
        MarketRegime.RANGE_BOUND: 1.0,
        MarketRegime.TRENDING_BULL: 0.8,
        MarketRegime.BREAKOUT: 0.4,
        MarketRegime.TRENDING_BEAR: 0.2,
        MarketRegime.VOLATILE: 0.5,
    },

    PatternType.TRIPLE_BOTTOM: {
        MarketRegime.RANGE_BOUND: 1.0,
        MarketRegime.TRENDING_BEAR: 0.8,
        MarketRegime.BREAKOUT: 0.4,
        MarketRegime.TRENDING_BULL: 0.2,
        MarketRegime.VOLATILE: 0.5,
    },

    # Triangle patterns adapt to multiple regimes
    PatternType.ASCENDING_TRIANGLE: {
        MarketRegime.TRENDING_BULL: 0.9,
        MarketRegime.RANGE_BOUND: 0.8,
        MarketRegime.BREAKOUT: 0.8,
        MarketRegime.TRENDING_BEAR: 0.4,
        MarketRegime.VOLATILE: 0.5,
    },

    PatternType.DESCENDING_TRIANGLE: {
        MarketRegime.TRENDING_BEAR: 0.9,
        MarketRegime.RANGE_BOUND: 0.8,
        MarketRegime.BREAKOUT: 0.8,
        MarketRegime.TRENDING_BULL: 0.4,
        MarketRegime.VOLATILE: 0.5,
    },

    PatternType.SYMMETRICAL_TRIANGLE: {
        MarketRegime.RANGE_BOUND: 0.9,
        MarketRegime.BREAKOUT: 0.8,
        MarketRegime.TRENDING_BULL: 0.6,
        MarketRegime.TRENDING_BEAR: 0.6,
        MarketRegime.VOLATILE: 0.6,
    },

    # Head & Shoulders - major reversal
    PatternType.HEAD_AND_SHOULDERS: {
        MarketRegime.TRENDING_BULL: 0.9,    # Reversal at top
        MarketRegime.RANGE_BOUND: 0.7,
        MarketRegime.BREAKOUT: 0.5,
        MarketRegime.TRENDING_BEAR: 0.3,
        MarketRegime.VOLATILE: 0.5,
    },

    PatternType.INVERSE_HEAD_AND_SHOULDERS: {
        MarketRegime.TRENDING_BEAR: 0.9,    # Reversal at bottom
        MarketRegime.RANGE_BOUND: 0.7,
        MarketRegime.BREAKOUT: 0.5,
        MarketRegime.TRENDING_BULL: 0.3,
        MarketRegime.VOLATILE: 0.5,
    },

    # Channel patterns
    PatternType.RECTANGLE: {
        MarketRegime.RANGE_BOUND: 1.0,      # Perfect match
        MarketRegime.TRENDING_BULL: 0.6,
        MarketRegime.TRENDING_BEAR: 0.6,
        MarketRegime.BREAKOUT: 0.7,
        MarketRegime.VOLATILE: 0.4,
    },

    PatternType.ASCENDING_CHANNEL: {
        MarketRegime.TRENDING_BULL: 0.9,
        MarketRegime.RANGE_BOUND: 0.6,
        MarketRegime.BREAKOUT: 0.7,
        MarketRegime.TRENDING_BEAR: 0.3,
        MarketRegime.VOLATILE: 0.5,
    },

    PatternType.DESCENDING_CHANNEL: {
        MarketRegime.TRENDING_BEAR: 0.9,
        MarketRegime.RANGE_BOUND: 0.6,
        MarketRegime.BREAKOUT: 0.7,
        MarketRegime.TRENDING_BULL: 0.3,
        MarketRegime.VOLATILE: 0.5,
    },
}


class EnhancedPatternValidator(PatternValidator):
    """
    Enhanced Pattern Validator with Market Context Integration.

    Extends base PatternValidator with:
    - Context-aware confidence adjustment
    - Pattern-regime affinity scoring
    - Adaptive recommendation system
    - Historical success tracking by regime
    """

    def __init__(self, *args, **kwargs):
        """Initialize enhanced validator"""
        super().__init__(*args, **kwargs)

        # Pattern-regime success tracking (would be persisted in production)
        self.regime_performance: Dict[str, Dict] = {}

    def validate_pattern_with_context(
        self,
        pattern: DetectedPattern,
        market_data: MarketData,
        context: Optional[MarketContext] = None
    ) -> EnhancedValidationResult:
        """
        Validate pattern with optional market context for adaptive scoring.

        Args:
            pattern: DetectedPattern to validate
            market_data: MarketData for validation
            context: Optional MarketContext for adaptive scoring

        Returns:
            EnhancedValidationResult with context-aware confidence
        """
        # Get base validation results
        base_results = self.validate_pattern(pattern, market_data)
        base_confidence = self.calculate_overall_validation_score(base_results)

        # Extract individual scores
        volume_score = next((r.score for r in base_results
                           if r.criteria.value == 'volume_confirmation'), 0.0)
        timeframe_score = next((r.score for r in base_results
                              if r.criteria.value == 'timeframe_consistency'), 0.0)
        historical_score = next((r.score for r in base_results
                               if r.criteria.value == 'historical_success_rate'), 0.0)
        quality_score = next((r.score for r in base_results
                            if r.criteria.value == 'pattern_quality'), 0.0)

        # Initialize result
        result = EnhancedValidationResult(
            pattern_id=f"{pattern.symbol}_{pattern.pattern_type.value}_{pattern.start_time}",
            symbol=pattern.symbol,
            pattern_type=pattern.pattern_type,
            base_confidence=base_confidence,
            is_valid=all(r.passed for r in base_results),
            volume_score=volume_score,
            timeframe_score=timeframe_score,
            historical_score=historical_score,
            quality_score=quality_score,
        )

        # If no context provided, return base validation
        if context is None:
            result.adjusted_confidence = base_confidence
            result.context_boost = 0.0
            result.recommendation_strength = self._determine_recommendation_strength(
                base_confidence
            )
            return result

        # Calculate context-aware enhancements
        result.market_context = context

        # 1. Calculate context score
        result.context_score = self._calculate_enhanced_context_score(
            pattern, context
        )

        # 2. Calculate pattern-regime affinity
        result.regime_affinity = self._get_pattern_regime_affinity(
            pattern.pattern_type, context.market_regime
        )

        # 3. Apply context adjustment to confidence
        result.adjusted_confidence, result.context_boost = self._apply_context_adjustment(
            base_confidence, context, result.regime_affinity
        )

        # 4. Generate enhanced recommendations
        result.recommendation_strength, result.supporting_reasons, result.risk_warnings = (
            self._generate_enhanced_recommendation(
                result.adjusted_confidence, context, pattern, result.regime_affinity
            )
        )

        return result

    def _calculate_enhanced_context_score(
        self,
        pattern: DetectedPattern,
        context: MarketContext
    ) -> float:
        """
        Calculate enhanced market context score (0-1).

        Considers:
        - Volatility regime suitability
        - Trend alignment
        - Market breadth support
        - Pattern-regime affinity
        """
        score = 0.0

        # 1. Volatility suitability (25%)
        vol_score = self._score_volatility_suitability(
            pattern, context.volatility_regime
        )
        score += vol_score * 0.25

        # 2. Trend alignment (35%)
        trend_score = self._score_trend_alignment(
            pattern, context.trend_direction, context.trend_strength
        )
        score += trend_score * 0.35

        # 3. Breadth support (20%)
        breadth_score = self._score_breadth_support(context.breadth.breadth_score)
        score += breadth_score * 0.20

        # 4. Pattern-regime affinity (20%)
        affinity_score = self._get_pattern_regime_affinity(
            pattern.pattern_type, context.market_regime
        )
        score += affinity_score * 0.20

        return score

    def _score_volatility_suitability(
        self,
        pattern: DetectedPattern,
        vol_regime: VolatilityRegime
    ) -> float:
        """Score how suitable volatility is for this pattern"""
        # Most patterns prefer low-medium volatility
        continuation_patterns = [
            PatternType.BULL_FLAG, PatternType.BEAR_FLAG, PatternType.PENNANT
        ]

        if pattern.pattern_type in continuation_patterns:
            if vol_regime == VolatilityRegime.LOW:
                return 1.0  # Perfect for flags/pennants
            elif vol_regime == VolatilityRegime.MEDIUM:
                return 0.8
            elif vol_regime == VolatilityRegime.HIGH:
                return 0.5
            else:  # EXTREME
                return 0.3
        else:
            # Reversal patterns can work in various volatilities
            if vol_regime == VolatilityRegime.LOW:
                return 0.9
            elif vol_regime == VolatilityRegime.MEDIUM:
                return 1.0
            elif vol_regime == VolatilityRegime.HIGH:
                return 0.7
            else:  # EXTREME
                return 0.4

    def _score_trend_alignment(
        self,
        pattern: DetectedPattern,
        trend_direction: TrendDirection,
        trend_strength: float
    ) -> float:
        """Score trend alignment with pattern"""
        # Continuation patterns need matching trend
        if pattern.pattern_type == PatternType.BULL_FLAG:
            if trend_direction == TrendDirection.BULLISH:
                return min(1.0, 0.7 + trend_strength)
            else:
                return 0.3

        elif pattern.pattern_type == PatternType.BEAR_FLAG:
            if trend_direction == TrendDirection.BEARISH:
                return min(1.0, 0.7 + trend_strength)
            else:
                return 0.3

        # Reversal patterns benefit from clear trend to reverse
        elif pattern.pattern_type in [PatternType.DOUBLE_TOP, PatternType.TRIPLE_TOP]:
            if trend_direction == TrendDirection.BULLISH and trend_strength > 0.4:
                return 1.0  # Strong uptrend to reverse
            elif trend_direction == TrendDirection.SIDEWAYS:
                return 0.8  # Range-bound tops
            else:
                return 0.5

        elif pattern.pattern_type in [PatternType.DOUBLE_BOTTOM, PatternType.TRIPLE_BOTTOM]:
            if trend_direction == TrendDirection.BEARISH and trend_strength > 0.4:
                return 1.0  # Strong downtrend to reverse
            elif trend_direction == TrendDirection.SIDEWAYS:
                return 0.8  # Range-bound bottoms
            else:
                return 0.5

        # Default: moderate score
        return 0.7

    def _score_breadth_support(self, breadth_score: float) -> float:
        """Score market breadth support"""
        # Simply use the breadth score directly (already 0-1)
        return breadth_score

    def _get_pattern_regime_affinity(
        self,
        pattern_type: PatternType,
        regime: MarketRegime
    ) -> float:
        """Get pattern-regime affinity score"""
        affinity_map = PATTERN_REGIME_AFFINITY.get(pattern_type, {})
        return affinity_map.get(regime, 0.5)  # Default 0.5 if unknown

    def _apply_context_adjustment(
        self,
        base_confidence: float,
        context: MarketContext,
        regime_affinity: float
    ) -> Tuple[float, float]:
        """
        Apply market context adjustment to base confidence.

        Combines:
        - Context adaptive multiplier
        - Pattern-regime affinity

        Returns:
            Tuple of (adjusted_confidence, boost)
        """
        # Get base multiplier from context
        multiplier = context.adaptation.confidence_multiplier

        # Adjust multiplier based on pattern-regime affinity
        # High affinity (>0.8) → slight boost
        # Low affinity (<0.4) → additional penalty
        if regime_affinity > 0.8:
            multiplier *= 1.1  # 10% bonus for perfect match
        elif regime_affinity < 0.4:
            multiplier *= 0.9  # 10% penalty for poor match

        # Apply multiplier
        adjusted = base_confidence * multiplier

        # Clamp to valid range (0.0-1.0)
        adjusted = max(0.0, min(1.0, adjusted))

        # Calculate boost
        boost = adjusted - base_confidence

        return adjusted, boost

    def _determine_recommendation_strength(self, confidence: float) -> str:
        """Determine recommendation strength from confidence"""
        if confidence >= 0.85:
            return "VERY_STRONG"
        elif confidence >= 0.70:
            return "STRONG"
        elif confidence >= 0.55:
            return "MODERATE"
        else:
            return "WEAK"

    def _generate_enhanced_recommendation(
        self,
        adjusted_confidence: float,
        context: MarketContext,
        pattern: DetectedPattern,
        regime_affinity: float
    ) -> Tuple[str, List[str], List[str]]:
        """
        Generate enhanced recommendation with reasoning.

        Returns:
            Tuple of (strength, supporting_reasons, risk_warnings)
        """
        # Determine strength
        strength = self._determine_recommendation_strength(adjusted_confidence)

        # Compile supporting reasons
        supporting = []

        # Context factors
        if context.volatility_regime == VolatilityRegime.LOW:
            supporting.append("Low volatility environment favors clear pattern formation")

        if context.trend_strength > 0.5:
            supporting.append(
                f"Strong {context.trend_direction.value} trend supports pattern direction"
            )

        if context.breadth.breadth_score > 0.6:
            supporting.append("Positive market breadth confirms pattern strength")

        if regime_affinity > 0.8:
            supporting.append(
                f"{pattern.pattern_type.value} pattern highly suitable for {context.market_regime.value} regime"
            )

        # Add context supporting factors
        supporting.extend(context.supporting_factors)

        # Compile risk warnings
        warnings = []

        if context.volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            warnings.append(
                f"{context.volatility_regime.value.capitalize()} volatility increases pattern failure risk"
            )

        if context.trend_strength < 0.3:
            warnings.append("Weak trend reduces pattern reliability")

        if regime_affinity < 0.4:
            warnings.append(
                f"{pattern.pattern_type.value} pattern not well-suited for {context.market_regime.value} regime"
            )

        # Add context risk factors
        warnings.extend(context.risk_factors)

        return strength, supporting, warnings
