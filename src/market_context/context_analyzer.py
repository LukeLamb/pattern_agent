"""
Market Context Analyzer - Phase 2.3

Provides comprehensive market regime detection and context-aware pattern analysis.
Includes volatility regime detection, trend analysis, market breadth, and adaptive parameters.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


class VolatilityRegime(str, Enum):
    """Volatility regime classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class TrendDirection(str, Enum):
    """Trend direction classification"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    CHOPPY = "choppy"


class MarketRegime(str, Enum):
    """Overall market regime"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE_BOUND = "range_bound"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"


@dataclass
class MarketBreadth:
    """Market breadth metrics"""
    advance_decline_ratio: float  # Ratio of advancing to declining issues
    new_highs_lows_ratio: float   # Ratio of new highs to new lows
    volume_breadth: float          # Up volume vs down volume ratio
    breadth_score: float          # Overall breadth score (0-1)


@dataclass
class RegimeAdaptation:
    """Adaptive parameters based on market regime"""
    confidence_multiplier: float   # Adjust pattern confidence (0.5-2.0)
    lookback_adjustment: float     # Adjust pattern lookback period (0.5-2.0)
    volume_threshold: float        # Adjust volume requirements (0.5-2.0)
    breakout_threshold: float      # Adjust breakout validation (0.5-2.0)
    risk_adjustment: float         # Risk multiplier for position sizing (0.5-2.0)


@dataclass
class MarketContext:
    """Complete market context analysis"""
    timestamp: datetime
    volatility_regime: VolatilityRegime
    volatility_percentile: float
    trend_direction: TrendDirection
    trend_strength: float
    market_regime: MarketRegime
    breadth: MarketBreadth
    adaptation: RegimeAdaptation
    supporting_factors: List[str]
    risk_factors: List[str]


class MarketContextAnalyzer:
    """
    Comprehensive market context analysis engine.

    Provides:
    - Volatility regime detection (VIX-based or ATR-based)
    - Trend direction analysis (multi-method)
    - Market breadth analysis
    - Adaptive parameter adjustment based on regime
    """

    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
        breadth_window: int = 10
    ):
        """
        Initialize the Market Context Analyzer.

        Args:
            volatility_window: Window for volatility calculations
            trend_window: Window for trend calculations
            breadth_window: Window for breadth calculations
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.breadth_window = breadth_window

        # Volatility thresholds (percentiles)
        self.volatility_thresholds = {
            VolatilityRegime.LOW: 0.25,      # Below 25th percentile
            VolatilityRegime.MEDIUM: 0.75,   # 25th to 75th percentile
            VolatilityRegime.HIGH: 0.90,     # 75th to 90th percentile
            VolatilityRegime.EXTREME: 1.0    # Above 90th percentile
        }

    def analyze_context(
        self,
        market_data: pd.DataFrame,
        indicators: Optional[Dict] = None,
        vix_data: Optional[pd.DataFrame] = None
    ) -> MarketContext:
        """
        Perform comprehensive market context analysis.

        Args:
            market_data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            indicators: Optional pre-calculated technical indicators
            vix_data: Optional VIX data for volatility regime (if available)

        Returns:
            MarketContext object with complete analysis
        """
        # Detect volatility regime
        volatility_regime, vol_percentile = self._detect_volatility_regime(
            market_data, vix_data
        )

        # Analyze trend direction and strength
        trend_direction, trend_strength = self._analyze_trend(
            market_data, indicators
        )

        # Calculate market breadth (if multi-symbol data available)
        breadth = self._calculate_market_breadth(market_data)

        # Determine overall market regime
        market_regime = self._determine_market_regime(
            volatility_regime, trend_direction, trend_strength
        )

        # Generate adaptive parameters
        adaptation = self._generate_adaptation(
            volatility_regime, market_regime, trend_strength
        )

        # Identify supporting and risk factors
        supporting_factors = self._identify_supporting_factors(
            volatility_regime, trend_direction, trend_strength, breadth
        )
        risk_factors = self._identify_risk_factors(
            volatility_regime, trend_direction, trend_strength, breadth
        )

        return MarketContext(
            timestamp=datetime.now(),
            volatility_regime=volatility_regime,
            volatility_percentile=vol_percentile,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            market_regime=market_regime,
            breadth=breadth,
            adaptation=adaptation,
            supporting_factors=supporting_factors,
            risk_factors=risk_factors
        )

    def _detect_volatility_regime(
        self,
        df: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None
    ) -> Tuple[VolatilityRegime, float]:
        """
        Detect current volatility regime.

        Uses VIX if available, otherwise uses ATR-based calculation.

        Args:
            df: OHLCV data
            vix_data: Optional VIX data

        Returns:
            Tuple of (VolatilityRegime, percentile)
        """
        if vix_data is not None and not vix_data.empty:
            # Use VIX-based regime detection
            current_vix = vix_data['close'].iloc[-1]
            vix_percentile = self._calculate_percentile(
                vix_data['close'], current_vix, window=252
            )
        else:
            # Use ATR-based regime detection
            atr = self._calculate_atr(df, window=self.volatility_window)
            current_atr = atr.iloc[-1]
            atr_percentile = self._calculate_percentile(
                atr, current_atr, window=252
            )
            vix_percentile = atr_percentile

        # Classify regime based on percentile
        if vix_percentile <= self.volatility_thresholds[VolatilityRegime.LOW]:
            regime = VolatilityRegime.LOW
        elif vix_percentile <= self.volatility_thresholds[VolatilityRegime.MEDIUM]:
            regime = VolatilityRegime.MEDIUM
        elif vix_percentile <= self.volatility_thresholds[VolatilityRegime.HIGH]:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME

        return regime, vix_percentile

    def _analyze_trend(
        self,
        df: pd.DataFrame,
        indicators: Optional[Dict] = None
    ) -> Tuple[TrendDirection, float]:
        """
        Analyze trend direction and strength using multiple methods.

        Methods:
        1. Moving Average alignment (SMA 20, 50, 200)
        2. ADX (Average Directional Index)
        3. Higher Highs / Higher Lows analysis
        4. Price momentum

        Args:
            df: OHLCV data
            indicators: Optional pre-calculated indicators

        Returns:
            Tuple of (TrendDirection, strength 0-1)
        """
        methods_count = 0
        bullish_votes = 0
        bearish_votes = 0
        strength_sum = 0.0

        # Method 1: Moving Average Alignment
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()

        if len(df) >= 50:
            current_price = df['close'].iloc[-1]
            ma_20 = sma_20.iloc[-1]
            ma_50 = sma_50.iloc[-1]

            if current_price > ma_20 > ma_50:
                bullish_votes += 1
                strength_sum += 0.3
            elif current_price < ma_20 < ma_50:
                bearish_votes += 1
                strength_sum += 0.3

            methods_count += 1

        # Method 2: ADX (if available in indicators)
        if indicators and 'adx' in indicators:
            adx = indicators['adx']
            if adx > 25:  # Strong trend
                # Check +DI vs -DI
                if indicators.get('plus_di', 0) > indicators.get('minus_di', 0):
                    bullish_votes += 1
                else:
                    bearish_votes += 1
                strength_sum += min(adx / 100, 0.4)
            methods_count += 1

        # Method 3: Higher Highs / Higher Lows
        if len(df) >= 20:
            recent_highs = df['high'].iloc[-20:].values
            recent_lows = df['low'].iloc[-20:].values

            # Check if making higher highs
            if recent_highs[-1] > recent_highs[-10]:
                if recent_lows[-1] > recent_lows[-10]:
                    bullish_votes += 1
                    strength_sum += 0.2
            # Check if making lower lows
            elif recent_lows[-1] < recent_lows[-10]:
                if recent_highs[-1] < recent_highs[-10]:
                    bearish_votes += 1
                    strength_sum += 0.2

            methods_count += 1

        # Method 4: Price Momentum
        if len(df) >= 20:
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            if abs(price_change) > 0.05:  # 5% threshold
                if price_change > 0:
                    bullish_votes += 1
                else:
                    bearish_votes += 1
                strength_sum += min(abs(price_change), 0.3)
            methods_count += 1

        # Determine direction based on votes
        if methods_count == 0:
            return TrendDirection.SIDEWAYS, 0.0

        vote_ratio = (bullish_votes - bearish_votes) / methods_count
        avg_strength = strength_sum / methods_count

        if vote_ratio > 0.5:
            direction = TrendDirection.BULLISH
        elif vote_ratio < -0.5:
            direction = TrendDirection.BEARISH
        elif abs(vote_ratio) < 0.3:
            direction = TrendDirection.SIDEWAYS
        else:
            direction = TrendDirection.CHOPPY

        return direction, avg_strength

    def _calculate_market_breadth(
        self,
        df: pd.DataFrame
    ) -> MarketBreadth:
        """
        Calculate market breadth metrics.

        Note: For single-symbol data, provides simplified breadth analysis.
        For multi-symbol data, can calculate true advance/decline ratios.

        Args:
            df: OHLCV data

        Returns:
            MarketBreadth object
        """
        # For single symbol, use simplified breadth metrics
        # In production, this would aggregate across multiple symbols

        # Calculate up/down days ratio
        if len(df) >= self.breadth_window:
            recent_closes = df['close'].iloc[-self.breadth_window:]
            up_days = (recent_closes.diff() > 0).sum()
            down_days = (recent_closes.diff() < 0).sum()

            ad_ratio = up_days / max(down_days, 1) if down_days > 0 else 2.0
        else:
            ad_ratio = 1.0

        # Calculate new highs/lows ratio (20-day high/low)
        if len(df) >= 20:
            rolling_high = df['high'].rolling(window=20).max()
            rolling_low = df['low'].rolling(window=20).min()

            at_high = 1 if df['close'].iloc[-1] >= rolling_high.iloc[-2] else 0
            at_low = 1 if df['close'].iloc[-1] <= rolling_low.iloc[-2] else 0

            hl_ratio = 2.0 if at_high else (0.5 if at_low else 1.0)
        else:
            hl_ratio = 1.0

        # Calculate volume breadth
        if len(df) >= self.breadth_window:
            recent_data = df.iloc[-self.breadth_window:]
            up_volume = recent_data.loc[recent_data['close'] > recent_data['open'], 'volume'].sum()
            down_volume = recent_data.loc[recent_data['close'] < recent_data['open'], 'volume'].sum()

            vol_ratio = up_volume / max(down_volume, 1) if down_volume > 0 else 2.0
        else:
            vol_ratio = 1.0

        # Calculate overall breadth score (0-1 scale)
        # Normalize ratios to 0-1 range
        ad_score = min(ad_ratio / 2.0, 1.0)
        hl_score = min(hl_ratio / 2.0, 1.0)
        vol_score = min(vol_ratio / 2.0, 1.0)

        breadth_score = (ad_score + hl_score + vol_score) / 3.0

        return MarketBreadth(
            advance_decline_ratio=ad_ratio,
            new_highs_lows_ratio=hl_ratio,
            volume_breadth=vol_ratio,
            breadth_score=breadth_score
        )

    def _determine_market_regime(
        self,
        volatility_regime: VolatilityRegime,
        trend_direction: TrendDirection,
        trend_strength: float
    ) -> MarketRegime:
        """
        Determine overall market regime based on volatility and trend.

        Args:
            volatility_regime: Current volatility regime
            trend_direction: Current trend direction
            trend_strength: Trend strength (0-1)

        Returns:
            MarketRegime classification
        """
        # High volatility overrides trend analysis
        if volatility_regime == VolatilityRegime.EXTREME:
            return MarketRegime.VOLATILE

        # Strong trends with decent strength
        if trend_strength > 0.4:
            if trend_direction == TrendDirection.BULLISH:
                return MarketRegime.TRENDING_BULL
            elif trend_direction == TrendDirection.BEARISH:
                return MarketRegime.TRENDING_BEAR

        # Breakout conditions (high volatility + strong trend)
        if volatility_regime == VolatilityRegime.HIGH and trend_strength > 0.5:
            return MarketRegime.BREAKOUT

        # Default to range-bound for sideways/weak trends
        return MarketRegime.RANGE_BOUND

    def _generate_adaptation(
        self,
        volatility_regime: VolatilityRegime,
        market_regime: MarketRegime,
        trend_strength: float
    ) -> RegimeAdaptation:
        """
        Generate adaptive parameters based on market regime.

        Args:
            volatility_regime: Current volatility regime
            market_regime: Current market regime
            trend_strength: Trend strength (0-1)

        Returns:
            RegimeAdaptation with adjusted parameters
        """
        # Base multipliers (neutral regime)
        confidence_mult = 1.0
        lookback_adj = 1.0
        volume_thresh = 1.0
        breakout_thresh = 1.0
        risk_adj = 1.0

        # Adjust based on volatility regime
        if volatility_regime == VolatilityRegime.LOW:
            # Low volatility: tighter patterns, less confirmation needed
            confidence_mult *= 1.2
            breakout_thresh *= 0.8
            risk_adj *= 1.2
        elif volatility_regime == VolatilityRegime.HIGH:
            # High volatility: require stronger confirmation
            confidence_mult *= 0.8
            volume_thresh *= 1.3
            breakout_thresh *= 1.3
            risk_adj *= 0.7
        elif volatility_regime == VolatilityRegime.EXTREME:
            # Extreme volatility: very conservative
            confidence_mult *= 0.6
            volume_thresh *= 1.5
            breakout_thresh *= 1.5
            risk_adj *= 0.5

        # Adjust based on market regime
        if market_regime == MarketRegime.TRENDING_BULL or market_regime == MarketRegime.TRENDING_BEAR:
            # Trending: favor trend-continuation patterns
            confidence_mult *= 1.3
            lookback_adj *= 1.2
        elif market_regime == MarketRegime.RANGE_BOUND:
            # Range-bound: favor reversal patterns
            confidence_mult *= 1.1
            lookback_adj *= 0.9
        elif market_regime == MarketRegime.BREAKOUT:
            # Breakout: require strong volume confirmation
            volume_thresh *= 1.4
            breakout_thresh *= 0.9

        # Adjust based on trend strength
        if trend_strength > 0.6:
            confidence_mult *= 1.2
        elif trend_strength < 0.3:
            confidence_mult *= 0.9

        # Clamp values to reasonable ranges
        return RegimeAdaptation(
            confidence_multiplier=max(0.5, min(2.0, confidence_mult)),
            lookback_adjustment=max(0.5, min(2.0, lookback_adj)),
            volume_threshold=max(0.5, min(2.0, volume_thresh)),
            breakout_threshold=max(0.5, min(2.0, breakout_thresh)),
            risk_adjustment=max(0.5, min(2.0, risk_adj))
        )

    def _identify_supporting_factors(
        self,
        volatility_regime: VolatilityRegime,
        trend_direction: TrendDirection,
        trend_strength: float,
        breadth: MarketBreadth
    ) -> List[str]:
        """Identify factors supporting pattern validity"""
        factors = []

        if volatility_regime == VolatilityRegime.LOW:
            factors.append("Low volatility environment favors pattern clarity")

        if trend_strength > 0.5:
            factors.append(f"Strong {trend_direction.value} trend supports directional patterns")

        if breadth.breadth_score > 0.6:
            factors.append("Positive market breadth supports pattern strength")

        if breadth.volume_breadth > 1.3:
            factors.append("Strong volume breadth confirms price action")

        return factors

    def _identify_risk_factors(
        self,
        volatility_regime: VolatilityRegime,
        trend_direction: TrendDirection,
        trend_strength: float,
        breadth: MarketBreadth
    ) -> List[str]:
        """Identify risk factors that may invalidate patterns"""
        factors = []

        if volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            factors.append(f"{volatility_regime.value.capitalize()} volatility may cause false breakouts")

        if trend_direction == TrendDirection.CHOPPY:
            factors.append("Choppy price action reduces pattern reliability")

        if trend_strength < 0.3:
            factors.append("Weak trend may lead to pattern failure")

        if breadth.breadth_score < 0.4:
            factors.append("Negative breadth suggests weak market support")

        if breadth.volume_breadth < 0.7:
            factors.append("Poor volume breadth indicates lack of conviction")

        return factors

    # Helper methods

    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        return true_range.rolling(window=window).mean()

    def _calculate_percentile(
        self,
        series: pd.Series,
        value: float,
        window: int = 252
    ) -> float:
        """Calculate percentile of value within rolling window"""
        if len(series) < window:
            window = len(series)

        recent_data = series.iloc[-window:]
        percentile = (recent_data < value).sum() / len(recent_data)

        return percentile
