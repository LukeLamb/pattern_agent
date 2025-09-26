"""
Technical Indicator Engine - Calculate and monitor technical indicators.

This module provides comprehensive technical analysis capabilities including:
- Trend indicators (SMA, EMA, MACD, ADX)
- Momentum indicators (RSI, Stochastic, Williams %R, ROC)
- Volume indicators (OBV, VPT, A/D Line, CMF)
- Volatility indicators (Bollinger Bands, ATR, Standard Deviation)
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

try:
    from ..models.market_data import MarketData
except ImportError:
    # For testing and development
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.market_data import MarketData


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""

    # Trend indicators
    sma_periods: Optional[List[int]] = None
    ema_periods: Optional[List[int]] = None
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14

    # Momentum indicators
    rsi_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    williams_r_period: int = 14
    roc_period: int = 10

    # Volume indicators
    obv_enabled: bool = True
    vpt_enabled: bool = True
    ad_line_enabled: bool = True
    cmf_period: int = 21

    # Volatility indicators
    bb_period: int = 20
    bb_std_dev: float = 2.0
    atr_period: int = 14
    std_dev_period: int = 20

    def __post_init__(self):
        """Set default values if None."""
        if self.sma_periods is None:
            self.sma_periods = [10, 20, 50, 200]
        if self.ema_periods is None:
            self.ema_periods = [12, 26, 50]


class TechnicalIndicatorEngine:
    """
    Comprehensive technical indicator calculation engine.

    Supports all major categories of technical indicators used in
    pattern recognition and trading signal generation.
    """

    def __init__(self, config: Optional[IndicatorConfig] = None):
        """Initialize the Technical Indicator Engine."""
        self.config = config or IndicatorConfig()
        self.indicators = {}
        self.last_calculation_time = None

    def calculate_indicators(self, market_data: MarketData) -> Dict:
        """
        Calculate comprehensive technical indicators for given market data.

        Args:
            market_data: MarketData object with OHLCV data

        Returns:
            Dictionary of calculated indicators organized by category
        """
        df = market_data.to_dataframe()

        if df.empty or not self._validate_data(df):
            return {"error": "Invalid or insufficient market data"}

        indicators = {
            "symbol": market_data.symbol,
            "timeframe": market_data.timeframe,
            "calculation_time": datetime.now(),
            "data_points": len(df),
            "trend": {},
            "momentum": {},
            "volume": {},
            "volatility": {},
            "composite": {},
        }

        try:
            # Calculate trend indicators
            indicators["trend"] = self._calculate_trend_indicators(df)

            # Calculate momentum indicators
            indicators["momentum"] = self._calculate_momentum_indicators(df)

            # Calculate volume indicators
            if "volume" in df.columns:
                indicators["volume"] = self._calculate_volume_indicators(df)

            # Calculate volatility indicators
            indicators["volatility"] = self._calculate_volatility_indicators(df)

            # Calculate composite indicators
            indicators["composite"] = self._calculate_composite_indicators(
                df, indicators
            )

            self.last_calculation_time = datetime.now()
            self.indicators[market_data.symbol] = indicators

        except Exception as e:
            indicators["error"] = f"Calculation failed: {str(e)}"

        return indicators

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that DataFrame has required columns and sufficient data."""
        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            return False

        # Need at least enough data for longest period indicator
        sma_max = max(self.config.sma_periods) if self.config.sma_periods else 50
        ema_max = max(self.config.ema_periods) if self.config.ema_periods else 50
        min_periods_needed = max(
            sma_max,
            ema_max,
            self.config.macd_slow + self.config.macd_signal,
            self.config.bb_period + 10,  # Buffer for volatility calculations
        )

        return len(df) >= min_periods_needed

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all trend-based technical indicators."""
        trend_indicators = {}

        # Simple Moving Averages
        if self.config.sma_periods:
            for period in self.config.sma_periods:
                trend_indicators[f"sma_{period}"] = self._calculate_sma(
                    df["close"], period
                )

        # Exponential Moving Averages
        if self.config.ema_periods:
            for period in self.config.ema_periods:
                trend_indicators[f"ema_{period}"] = self._calculate_ema(
                    df["close"], period
                )

        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(
            df["close"],
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal,
        )
        trend_indicators["macd"] = {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        }

        # Average Directional Index (ADX)
        adx, plus_di, minus_di = self._calculate_adx(
            df["high"], df["low"], df["close"], self.config.adx_period
        )
        trend_indicators["adx"] = {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}

        return trend_indicators

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all momentum-based technical indicators."""
        momentum_indicators = {}

        # RSI
        momentum_indicators["rsi"] = self._calculate_rsi(
            df["close"], self.config.rsi_period
        )

        # Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic(
            df["high"],
            df["low"],
            df["close"],
            self.config.stoch_k_period,
            self.config.stoch_d_period,
        )
        momentum_indicators["stochastic"] = {"k": stoch_k, "d": stoch_d}

        # Williams %R
        momentum_indicators["williams_r"] = self._calculate_williams_r(
            df["high"], df["low"], df["close"], self.config.williams_r_period
        )

        # Rate of Change
        momentum_indicators["roc"] = self._calculate_roc(
            df["close"], self.config.roc_period
        )

        return momentum_indicators

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all volume-based technical indicators."""
        volume_indicators = {}

        if self.config.obv_enabled:
            volume_indicators["obv"] = self._calculate_obv(df["close"], df["volume"])

        if self.config.vpt_enabled:
            volume_indicators["vpt"] = self._calculate_vpt(df["close"], df["volume"])

        if self.config.ad_line_enabled:
            volume_indicators["ad_line"] = self._calculate_ad_line(
                df["high"], df["low"], df["close"], df["volume"]
            )

        volume_indicators["cmf"] = self._calculate_cmf(
            df["high"], df["low"], df["close"], df["volume"], self.config.cmf_period
        )

        return volume_indicators

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all volatility-based technical indicators."""
        volatility_indicators = {}

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            df["close"], self.config.bb_period, self.config.bb_std_dev
        )
        volatility_indicators["bollinger_bands"] = {
            "upper": bb_upper,
            "middle": bb_middle,
            "lower": bb_lower,
            "width": (bb_upper - bb_lower) / bb_middle * 100,
        }

        # Average True Range
        volatility_indicators["atr"] = self._calculate_atr(
            df["high"], df["low"], df["close"], self.config.atr_period
        )

        # Standard Deviation
        volatility_indicators["std_dev"] = self._calculate_std_dev(
            df["close"], self.config.std_dev_period
        )

        return volatility_indicators

    def _calculate_composite_indicators(
        self, df: pd.DataFrame, indicators: Dict
    ) -> Dict:
        """Calculate composite indicators that combine multiple base indicators."""
        composite = {}
        _ = df  # Acknowledge unused parameter

        # Moving Average Convergence/Divergence signals
        if "macd" in indicators.get("trend", {}):
            macd_data = indicators["trend"]["macd"]
            composite["macd_signals"] = self._generate_macd_signals(macd_data)

        # RSI overbought/oversold levels
        if "rsi" in indicators.get("momentum", {}):
            rsi_data = indicators["momentum"]["rsi"]
            composite["rsi_signals"] = self._generate_rsi_signals(rsi_data)

        # Bollinger Band squeeze detection
        if "bollinger_bands" in indicators.get("volatility", {}):
            bb_data = indicators["volatility"]["bollinger_bands"]
            composite["bb_squeeze"] = self._detect_bb_squeeze(bb_data)

        return composite

    # ===========================================
    # TREND INDICATOR CALCULATIONS
    # ===========================================

    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_macd(
        self, prices: pd.Series, fast: int, slow: int, signal: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD line, signal line, and histogram."""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Average Directional Index and directional indicators."""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr_df = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3})
        tr = tr_df.max(axis=1)

        # Directional Movement using numpy for comparison operations
        high_values = np.array(high, dtype=float)
        low_values = np.array(low, dtype=float)

        plus_dm_values = np.diff(high_values, prepend=high_values[0])
        minus_dm_values = -np.diff(low_values, prepend=low_values[0])

        # Apply conditions using numpy
        plus_dm = pd.Series(
            np.where(
                (plus_dm_values > minus_dm_values) & (plus_dm_values > 0),
                plus_dm_values,
                0.0,
            )
        )
        minus_dm = pd.Series(
            np.where(
                (minus_dm_values > plus_dm_values) & (minus_dm_values > 0),
                minus_dm_values,
                0.0,
            )
        )

        # Smoothed values
        atr = pd.Series(tr).rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # ADX calculation
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = pd.Series(dx).fillna(0)  # Handle division by zero
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    # ===========================================
    # MOMENTUM INDICATOR CALCULATIONS
    # ===========================================

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        # Convert to numpy for calculations, then back to Series
        prices_values = np.array(prices, dtype=float)
        delta = np.diff(prices_values, prepend=prices_values[0])

        # Calculate gains and losses using numpy
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        # Create rolling averages
        avg_gains = pd.Series(gains).rolling(window=period, min_periods=1).mean()
        avg_losses = pd.Series(losses).rolling(window=period, min_periods=1).mean()

        # Calculate RSI
        rs = avg_gains / avg_losses.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int,
        d_period: int,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator %K and %D."""
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()

        return k_percent.fillna(50), d_percent.fillna(50)

    def _calculate_williams_r(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()

        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r.fillna(-50)

    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change."""
        roc = ((prices / prices.shift(period)) - 1) * 100
        return roc.fillna(0)

    # ===========================================
    # VOLUME INDICATOR CALCULATIONS
    # ===========================================

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = np.where(
            close > close.shift(1), volume, np.where(close < close.shift(1), -volume, 0)
        )
        return pd.Series(obv, index=close.index).cumsum()

    def _calculate_vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Price Trend."""
        vpt = volume * (close.pct_change())
        return vpt.cumsum().fillna(0)

    def _calculate_ad_line(
        self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero when high == low
        ad_line = (clv * volume).cumsum()
        return ad_line

    def _calculate_cmf(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int,
    ) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        money_flow_volume = clv * volume
        cmf = (
            money_flow_volume.rolling(window=period, min_periods=1).sum()
            / volume.rolling(window=period, min_periods=1).sum()
        )
        return cmf.fillna(0)

    # ===========================================
    # VOLATILITY INDICATOR CALCULATIONS
    # ===========================================

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int, std_dev: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = self._calculate_sma(prices, period)
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return upper_band, sma, lower_band

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))

        tr_df = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3})
        true_range = tr_df.max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()

        return atr

    def _calculate_std_dev(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Standard Deviation."""
        return prices.rolling(window=period, min_periods=1).std().fillna(0)

    # ===========================================
    # SIGNAL GENERATION METHODS
    # ===========================================

    def _generate_macd_signals(self, macd_data: Dict) -> Dict:
        """Generate MACD-based trading signals."""
        macd_line = macd_data["macd_line"]
        signal_line = macd_data["signal_line"]
        histogram = macd_data["histogram"]

        # Bullish: MACD crosses above signal line
        bullish_crossover = (macd_line > signal_line) & (
            macd_line.shift(1) <= signal_line.shift(1)
        )

        # Bearish: MACD crosses below signal line
        bearish_crossover = (macd_line < signal_line) & (
            macd_line.shift(1) >= signal_line.shift(1)
        )

        # Histogram divergence
        histogram_increasing = histogram > histogram.shift(1)
        histogram_decreasing = histogram < histogram.shift(1)

        return {
            "bullish_crossover": bullish_crossover,
            "bearish_crossover": bearish_crossover,
            "histogram_increasing": histogram_increasing,
            "histogram_decreasing": histogram_decreasing,
            "above_zero": macd_line > 0,
            "below_zero": macd_line < 0,
        }

    def _generate_rsi_signals(self, rsi: pd.Series) -> Dict:
        """Generate RSI-based trading signals."""
        return {
            "oversold": rsi < 30,
            "overbought": rsi > 70,
            "oversold_extreme": rsi < 20,
            "overbought_extreme": rsi > 80,
            "neutral": (rsi >= 45) & (rsi <= 55),
            "bullish_divergence": self._detect_rsi_divergence(rsi, "bullish"),
            "bearish_divergence": self._detect_rsi_divergence(rsi, "bearish"),
        }

    def _detect_bb_squeeze(self, bb_data: Dict) -> pd.Series:
        """Detect Bollinger Band squeeze (low volatility periods)."""
        bb_width = bb_data["width"]
        bb_width_sma = bb_width.rolling(window=20, min_periods=1).mean()

        # Squeeze when current width is below average
        squeeze = bb_width < bb_width_sma * 0.8
        return squeeze

    def _detect_rsi_divergence(self, rsi: pd.Series, direction: str) -> pd.Series:
        """Detect RSI divergence patterns (simplified implementation)."""
        # This is a simplified divergence detection
        # In production, this would be more sophisticated
        rsi_slope = rsi.rolling(window=5, min_periods=3).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0, raw=True
        )

        if direction == "bullish":
            return (rsi < 35) & (rsi_slope > 0)
        else:  # bearish
            return (rsi > 65) & (rsi_slope < 0)

    def get_latest_signals(self, symbol: str) -> Optional[Dict]:
        """Get the latest calculated indicators and signals for a symbol."""
        if symbol in self.indicators:
            return self.indicators[symbol]
        return None

    def get_indicator_summary(self, symbol: str) -> Optional[Dict]:
        """Get a summary of key indicators for quick analysis."""
        indicators = self.get_latest_signals(symbol)
        if not indicators:
            return None

        # Get latest values (last non-NaN value for each indicator)
        def get_latest_value(series):
            if isinstance(series, pd.Series):
                return series.dropna().iloc[-1] if not series.dropna().empty else None
            return series

        summary = {
            "symbol": symbol,
            "timestamp": indicators["calculation_time"],
            "trend_strength": "neutral",
            "momentum_status": "neutral",
            "volume_trend": "neutral",
            "volatility_level": "normal",
            "key_levels": {},
            "signals": [],
        }

        # Analyze trend
        trend = indicators.get("trend", {})
        if "adx" in trend:
            adx_value = get_latest_value(trend["adx"]["adx"])
            if adx_value:
                if adx_value > 25:
                    summary["trend_strength"] = "strong"
                elif adx_value > 20:
                    summary["trend_strength"] = "moderate"

        # Analyze momentum
        momentum = indicators.get("momentum", {})
        if "rsi" in momentum:
            rsi_value = get_latest_value(momentum["rsi"])
            if rsi_value:
                if rsi_value > 70:
                    summary["momentum_status"] = "overbought"
                elif rsi_value < 30:
                    summary["momentum_status"] = "oversold"

        return summary
