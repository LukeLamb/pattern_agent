"""
Market Data Model - Core market data structures
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd


class MarketDataType(str, Enum):
    """Type of market data."""
    OHLCV = "ohlcv"
    TICK = "tick"
    QUOTE = "quote"
    TRADE = "trade"
    LEVEL2 = "level2"


class MarketSession(str, Enum):
    """Market session types."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    POST_MARKET = "post_market"
    EXTENDED = "extended"


class OHLCV(BaseModel):
    """Open, High, Low, Close, Volume data point."""
    timestamp: datetime = Field(..., description="Data point timestamp")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Volume")
    
    @field_validator('high')
    @classmethod
    def validate_high(cls, v):
        """Validate high is valid price."""
        if v <= 0:
            raise ValueError("High must be > 0")
        return v
    
    @field_validator('low')
    @classmethod
    def validate_low(cls, v):
        """Validate low is valid price."""
        if v <= 0:
            raise ValueError("Low must be > 0")
        return v


class MarketData(BaseModel):
    """
    Comprehensive market data container.
    
    Stores OHLCV data with metadata and validation for technical analysis.
    """
    
    # Core Identification
    symbol: str = Field(..., min_length=1, description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe (1min, 5min, 1hr, daily, etc.)")
    data_type: MarketDataType = Field(MarketDataType.OHLCV, description="Type of market data")
    
    # Data
    data: List[OHLCV] = Field(..., description="OHLCV data points", min_length=1)
    
    # Metadata
    start_time: datetime = Field(..., description="Start time of data range")
    end_time: datetime = Field(..., description="End time of data range")
    data_source: str = Field("unknown", description="Data source provider")
    market_session: MarketSession = Field(MarketSession.REGULAR, description="Market session")
    
    # Quality metrics
    completeness_score: float = Field(1.0, ge=0, le=1, description="Data completeness (0-1)")
    gaps_detected: int = Field(0, ge=0, description="Number of data gaps detected")
    anomalies_detected: int = Field(0, ge=0, description="Number of anomalies detected")
    
    # Derived metrics
    total_volume: Optional[int] = Field(None, description="Total volume across all periods")
    vwap: Optional[float] = Field(None, description="Volume weighted average price")
    price_range: Optional[float] = Field(None, description="Price range (high - low)")
    volatility: Optional[float] = Field(None, description="Price volatility measure")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @field_validator('data')
    @classmethod
    def validate_data_chronological(cls, v):
        """Validate data is in chronological order."""
        if len(v) > 1:
            for i in range(1, len(v)):
                if v[i].timestamp <= v[i-1].timestamp:
                    raise ValueError("Data must be in chronological order")
        return v
    
    @model_validator(mode='after')
    def validate_time_range(self):
        """Validate end time is after start time."""
        if self.end_time <= self.start_time:
            raise ValueError("End time must be after start time")
        return self
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert market data to pandas DataFrame."""
        data_dicts = []
        for ohlcv in self.data:
            data_dicts.append({
                'timestamp': ohlcv.timestamp,
                'open': ohlcv.open,
                'high': ohlcv.high,
                'low': ohlcv.low,
                'close': ohlcv.close,
                'volume': ohlcv.volume
            })
        
        df = pd.DataFrame(data_dicts)
        df.set_index('timestamp', inplace=True)
        return df
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, symbol: str, timeframe: str, **kwargs) -> 'MarketData':
        """Create MarketData from pandas DataFrame."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Convert DataFrame to OHLCV objects
        ohlcv_data = []
        for timestamp, row in df.iterrows():
            if pd.isna(row['open']) or pd.isna(row['high']) or pd.isna(row['low']) or pd.isna(row['close']):
                continue  # Skip rows with NaN values
                
            # Handle timestamp conversion safely
            if isinstance(timestamp, datetime):
                ts = timestamp
            else:
                ts = pd.to_datetime(str(timestamp))
            
            ohlcv = OHLCV(
                timestamp=ts,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume']) if not pd.isna(row['volume']) else 0
            )
            ohlcv_data.append(ohlcv)
        
        if not ohlcv_data:
            raise ValueError("No valid OHLCV data found in DataFrame")
        
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            data=ohlcv_data,
            start_time=ohlcv_data[0].timestamp,
            end_time=ohlcv_data[-1].timestamp,
            **kwargs
        )
    
    def get_price_series(self, price_type: str = 'close') -> List[float]:
        """Get price series of specified type."""
        if price_type not in ['open', 'high', 'low', 'close']:
            raise ValueError("price_type must be one of: open, high, low, close")
        
        return [getattr(ohlcv, price_type) for ohlcv in self.data]
    
    def get_volume_series(self) -> List[int]:
        """Get volume series."""
        return [ohlcv.volume for ohlcv in self.data]
    
    def get_typical_price_series(self) -> List[float]:
        """Get typical price series (HLC/3)."""
        return [(ohlcv.high + ohlcv.low + ohlcv.close) / 3 for ohlcv in self.data]
    
    def get_latest_price(self, price_type: str = 'close') -> float:
        """Get the latest price of specified type."""
        if not self.data:
            raise ValueError("No data available")
        return getattr(self.data[-1], price_type)
    
    def get_price_range(self) -> float:
        """Get price range (highest high - lowest low)."""
        if not self.data:
            return 0.0
        
        highest = max(ohlcv.high for ohlcv in self.data)
        lowest = min(ohlcv.low for ohlcv in self.data)
        return highest - lowest
    
    def calculate_returns(self, price_type: str = 'close') -> List[float]:
        """Calculate price returns."""
        prices = self.get_price_series(price_type)
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1] * 100
            returns.append(ret)
        
        return returns
    
    def calculate_vwap(self) -> float:
        """Calculate Volume Weighted Average Price."""
        total_pv = sum(ohlcv.volume * ((ohlcv.high + ohlcv.low + ohlcv.close) / 3) 
                      for ohlcv in self.data)
        total_volume = sum(ohlcv.volume for ohlcv in self.data)
        
        if total_volume == 0:
            return 0.0
        
        return total_pv / total_volume
    
    def detect_gaps(self, gap_threshold_percent: float = 2.0) -> List[Dict[str, Any]]:
        """Detect price gaps between periods."""
        gaps = []
        
        for i in range(1, len(self.data)):
            prev_close = self.data[i-1].close
            current_open = self.data[i].open
            
            gap_percent = abs(current_open - prev_close) / prev_close * 100
            
            if gap_percent >= gap_threshold_percent:
                gap_type = "gap_up" if current_open > prev_close else "gap_down"
                gaps.append({
                    'index': i,
                    'timestamp': self.data[i].timestamp,
                    'gap_type': gap_type,
                    'gap_percent': gap_percent,
                    'prev_close': prev_close,
                    'current_open': current_open
                })
        
        return gaps
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and return metrics."""
        total_periods = len(self.data)
        gaps = self.detect_gaps()
        
        # Check for missing periods (basic check)
        expected_periods = self._calculate_expected_periods()
        missing_periods = max(0, expected_periods - total_periods)
        
        # Check for anomalies (prices outside reasonable range)
        anomalies = self._detect_price_anomalies()
        
        completeness = 1.0 - (missing_periods / expected_periods) if expected_periods > 0 else 1.0
        
        return {
            'total_periods': total_periods,
            'expected_periods': expected_periods,
            'missing_periods': missing_periods,
            'gaps_detected': len(gaps),
            'anomalies_detected': len(anomalies),
            'completeness_score': completeness,
            'quality_score': max(0.0, completeness - (len(gaps) * 0.01) - (len(anomalies) * 0.02))
        }
    
    def _calculate_expected_periods(self) -> int:
        """Calculate expected number of periods based on timeframe."""
        # Simplified calculation - would need more sophisticated logic for real implementation
        time_diff = (self.end_time - self.start_time).total_seconds()
        
        timeframe_seconds = {
            '1min': 60, '5min': 300, '15min': 900, '30min': 1800, 
            '1hr': 3600, '4hr': 14400, 'daily': 86400, 'weekly': 604800
        }
        
        period_seconds = timeframe_seconds.get(self.timeframe, 3600)  # Default to 1 hour
        return int(time_diff / period_seconds)
    
    def _detect_price_anomalies(self) -> List[int]:
        """Detect price anomalies using simple statistical method."""
        if len(self.data) < 10:
            return []
        
        closes = self.get_price_series('close')
        mean_price = sum(closes) / len(closes)
        
        # Calculate standard deviation
        variance = sum((price - mean_price) ** 2 for price in closes) / len(closes)
        std_dev = variance ** 0.5
        
        # Flag prices more than 3 standard deviations from mean
        anomalies = []
        threshold = 3 * std_dev
        
        for i, ohlcv in enumerate(self.data):
            if (abs(ohlcv.close - mean_price) > threshold or
                abs(ohlcv.high - mean_price) > threshold or
                abs(ohlcv.low - mean_price) > threshold):
                anomalies.append(i)
        
        return anomalies
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the market data."""
        if not self.data:
            return {}
        
        closes = self.get_price_series('close')
        volumes = self.get_volume_series()
        returns = self.calculate_returns()
        
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'periods': len(self.data),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'first_price': self.data[0].close,
            'last_price': self.data[-1].close,
            'price_change': self.data[-1].close - self.data[0].close,
            'price_change_percent': ((self.data[-1].close - self.data[0].close) / self.data[0].close) * 100,
            'highest_price': max(ohlcv.high for ohlcv in self.data),
            'lowest_price': min(ohlcv.low for ohlcv in self.data),
            'average_price': sum(closes) / len(closes),
            'total_volume': sum(volumes),
            'average_volume': sum(volumes) / len(volumes) if volumes else 0,
            'vwap': self.calculate_vwap(),
            'volatility': (sum((r / 100) ** 2 for r in returns) / len(returns)) ** 0.5 * 100 if returns else 0
        }
    
    def __len__(self) -> int:
        """Return number of data points."""
        return len(self.data)
    
    def __getitem__(self, index: int) -> OHLCV:
        """Get data point by index."""
        return self.data[index]
    
    def __str__(self) -> str:
        """String representation."""
        return f"MarketData({self.symbol}, {self.timeframe}, {len(self.data)} periods)"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"MarketData(symbol={self.symbol}, timeframe={self.timeframe}, periods={len(self.data)}, range={self.start_time} to {self.end_time})"