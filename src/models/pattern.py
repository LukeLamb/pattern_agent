"""
Pattern Data Model - Core pattern recognition data structures
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd


class PatternType(str, Enum):
    """Enumeration of supported pattern types."""

    # Continuation Patterns
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    PENNANT = "pennant"
    RECTANGLE = "rectangle"
    ASCENDING_CHANNEL = "ascending_channel"
    DESCENDING_CHANNEL = "descending_channel"
    CUP_AND_HANDLE = "cup_and_handle"

    # Reversal Patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    ROUNDING_TOP = "rounding_top"
    ROUNDING_BOTTOM = "rounding_bottom"

    # Breakout Patterns
    SUPPORT_BREAK = "support_break"
    RESISTANCE_BREAK = "resistance_break"
    TRENDLINE_BREAK = "trendline_break"
    VOLUME_BREAKOUT = "volume_breakout"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"


class PatternStatus(str, Enum):
    """Pattern lifecycle status."""

    FORMING = "forming"
    COMPLETE = "complete"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    EXPIRED = "expired"


class PatternDirection(str, Enum):
    """Pattern direction bias."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SupportResistanceLevel(BaseModel):
    """Support or resistance level data structure."""

    price: float = Field(..., gt=0, description="Price level")
    strength: int = Field(..., ge=1, le=10, description="Level strength (1-10)")
    touches: int = Field(
        ..., ge=1, description="Number of times price touched this level"
    )
    last_touch: datetime = Field(..., description="Last time price touched this level")
    volume_confirmation: float = Field(
        0.0, ge=0, description="Average volume at this level"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PatternMetrics(BaseModel):
    """Pattern formation and validation metrics."""

    formation_duration_days: float = Field(
        ..., gt=0, description="Days taken to form pattern"
    )
    price_range_percent: float = Field(
        ..., gt=0, description="Price range as percentage of entry price"
    )
    volume_trend: float = Field(
        ..., description="Volume trend during formation (-1 to 1)"
    )
    volatility_normalized: float = Field(
        ..., ge=0, description="Volatility normalized to historical average"
    )
    symmetry_score: float = Field(
        0.0, ge=0, le=1, description="Pattern symmetry score (0-1)"
    )
    breakout_strength: Optional[float] = Field(
        None, ge=0, description="Breakout strength if applicable"
    )


class ValidationCriteria(BaseModel):
    """Pattern validation criteria and results."""

    min_touches: int = Field(
        2, ge=1, description="Minimum touches required for pattern"
    )
    min_duration_days: float = Field(
        1.0, gt=0, description="Minimum formation duration"
    )
    max_duration_days: float = Field(
        365.0, gt=0, description="Maximum formation duration"
    )
    min_price_range_percent: float = Field(
        1.0, gt=0, description="Minimum price range percentage"
    )
    volume_confirmation_required: bool = Field(
        True, description="Whether volume confirmation is required"
    )
    volume_threshold_multiplier: float = Field(
        1.5, gt=0, description="Volume multiplier for confirmation"
    )

    # Validation results
    touches_validated: bool = Field(
        False, description="Whether touch count is validated"
    )
    duration_validated: bool = Field(False, description="Whether duration is validated")
    volume_validated: bool = Field(False, description="Whether volume is validated")
    price_action_validated: bool = Field(
        False, description="Whether price action is validated"
    )
    overall_validated: bool = Field(False, description="Overall validation status")
    validation_score: float = Field(
        0.0, ge=0, le=1, description="Composite validation score"
    )
    validation_reasons: List[str] = Field(
        default_factory=list, description="Validation failure reasons"
    )


class Pattern(BaseModel):
    """
    Core Pattern data model representing a detected technical analysis pattern.

    This model captures all essential information about a pattern including:
    - Pattern identification and classification
    - Key price levels and metrics
    - Formation timeline and validation
    - Confidence scoring and market context
    """

    # Core Identification
    pattern_id: str = Field(..., description="Unique pattern identifier")
    symbol: str = Field(
        ..., min_length=1, description="Trading symbol (e.g., AAPL, BTCUSD)"
    )
    pattern_type: PatternType = Field(..., description="Type of pattern detected")
    status: PatternStatus = Field(
        PatternStatus.FORMING, description="Current pattern status"
    )
    direction: PatternDirection = Field(..., description="Pattern directional bias")

    # Timeline
    formation_start: datetime = Field(..., description="When pattern formation began")
    formation_date: datetime = Field(..., description="When pattern was first detected")
    completion_date: Optional[datetime] = Field(
        None, description="When pattern completed formation"
    )
    breakout_date: Optional[datetime] = Field(
        None, description="When breakout occurred"
    )
    expiry_date: Optional[datetime] = Field(
        None, description="When pattern expires if not confirmed"
    )

    # Timeframe
    timeframe: str = Field(..., description="Primary timeframe (e.g., '1hr', 'daily')")
    additional_timeframes: List[str] = Field(
        default_factory=list, description="Supporting timeframes"
    )

    # Price Levels
    support_levels: List[SupportResistanceLevel] = Field(
        default_factory=list, description="Support levels"
    )
    resistance_levels: List[SupportResistanceLevel] = Field(
        default_factory=list, description="Resistance levels"
    )
    entry_price: Optional[float] = Field(
        None, gt=0, description="Recommended entry price"
    )
    target_prices: List[float] = Field(
        default_factory=list, description="Target price levels"
    )
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")

    # Pattern Metrics
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Pattern confidence (0-1)"
    )
    strength_score: float = Field(..., ge=0, le=1, description="Pattern strength (0-1)")
    reliability_score: float = Field(
        ..., ge=0, le=1, description="Historical reliability (0-1)"
    )
    pattern_metrics: PatternMetrics = Field(..., description="Detailed pattern metrics")

    # Validation
    validation_criteria: ValidationCriteria = Field(
        ..., description="Validation criteria and results"
    )

    # Market Context
    market_context: Dict[str, Any] = Field(
        default_factory=dict, description="Market conditions during formation"
    )
    sector_context: Dict[str, Any] = Field(
        default_factory=dict, description="Sector-specific context"
    )

    # Risk Assessment
    risk_reward_ratio: Optional[float] = Field(
        None, gt=0, description="Calculated risk-reward ratio"
    )
    max_risk_percent: Optional[float] = Field(
        None, gt=0, le=100, description="Maximum risk as percentage"
    )
    probability_success: Optional[float] = Field(
        None, ge=0, le=1, description="Success probability based on history"
    )

    # Technical Data
    formation_data: Dict[str, Any] = Field(
        default_factory=dict, description="Raw formation data and calculations"
    )
    indicator_values: Dict[str, float] = Field(
        default_factory=dict, description="Technical indicator values at detection"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.now, description="When pattern record was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="When pattern record was last updated"
    )
    version: str = Field("1.0", description="Pattern model version")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            PatternType: lambda v: v.value,
            PatternStatus: lambda v: v.value,
            PatternDirection: lambda v: v.value,
        }

    @field_validator("expiry_date")
    @classmethod
    def set_default_expiry(cls, v, info):
        """Set default expiry date if not provided."""
        if v is None:
            # Default expiry: 30 days from now
            return datetime.now() + timedelta(days=30)
        return v

    @field_validator("target_prices")
    @classmethod
    def validate_targets(cls, v):
        """Validate target prices are reasonable."""
        if not v:
            return v
        return sorted(v)

    @model_validator(mode="after")
    def validate_timeline(self):
        """Validate timeline consistency."""
        if self.formation_start and self.formation_date:
            if self.formation_date < self.formation_start:
                raise ValueError("Formation date cannot be before formation start")

        if self.formation_date and self.completion_date:
            if self.completion_date < self.formation_date:
                raise ValueError("Completion date cannot be before formation date")

        if self.completion_date and self.breakout_date:
            if self.breakout_date < self.completion_date:
                raise ValueError("Breakout date cannot be before completion date")

        return self

    @model_validator(mode="after")
    def validate_risk_reward(self):
        """Validate risk-reward calculation."""
        entry_price = self.entry_price
        target_prices = self.target_prices
        stop_loss = self.stop_loss

        if entry_price and target_prices and stop_loss:
            # Calculate risk-reward ratio
            potential_profit = (
                abs(target_prices[0] - entry_price) if target_prices else 0
            )
            potential_loss = abs(entry_price - stop_loss)

            if potential_loss > 0:
                calculated_rr = potential_profit / potential_loss
                self.risk_reward_ratio = round(calculated_rr, 2)

        return self

    def update_status(
        self, new_status: PatternStatus, timestamp: Optional[datetime] = None
    ):
        """Update pattern status with timestamp."""
        self.status = new_status
        self.updated_at = timestamp or datetime.now()

        if new_status == PatternStatus.COMPLETE and not self.completion_date:
            self.completion_date = self.updated_at
        elif new_status == PatternStatus.CONFIRMED and not self.breakout_date:
            self.breakout_date = self.updated_at

    def is_expired(self) -> bool:
        """Check if pattern has expired."""
        if self.expiry_date:
            return datetime.now() > self.expiry_date
        return False

    def is_active(self) -> bool:
        """Check if pattern is currently active (not failed or expired)."""
        return (
            self.status not in [PatternStatus.FAILED, PatternStatus.EXPIRED]
            and not self.is_expired()
        )

    def get_age_days(self) -> float:
        """Get pattern age in days since formation."""
        return (datetime.now() - self.formation_date).total_seconds() / 86400

    def get_formation_duration_days(self) -> float:
        """Get pattern formation duration in days."""
        return (self.formation_date - self.formation_start).total_seconds() / 86400

    def add_support_level(
        self, price: float, strength: int, touches: int, last_touch: datetime
    ):
        """Add a support level to the pattern."""
        level = SupportResistanceLevel(
            price=price,
            strength=strength,
            touches=touches,
            last_touch=last_touch,
            volume_confirmation=True,
        )
        self.support_levels.append(level)
        self.updated_at = datetime.now()

    def add_resistance_level(
        self, price: float, strength: int, touches: int, last_touch: datetime
    ):
        """Add a resistance level to the pattern."""
        level = SupportResistanceLevel(
            price=price,
            strength=strength,
            touches=touches,
            last_touch=last_touch,
            volume_confirmation=True,
        )
        self.resistance_levels.append(level)
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return self.dict()

    def to_json(self) -> str:
        """Convert pattern to JSON string."""
        return self.json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """Create pattern from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Pattern":
        """Create pattern from JSON string."""
        return cls.parse_raw(json_str)

    def __str__(self) -> str:
        """String representation of pattern."""
        return f"{self.pattern_type.value} on {self.symbol} ({self.timeframe}) - {self.status.value} - Confidence: {self.confidence_score:.2f}"

    def __repr__(self) -> str:
        """Detailed representation of pattern."""
        return f"Pattern(id={self.pattern_id}, type={self.pattern_type.value}, symbol={self.symbol}, status={self.status.value}, confidence={self.confidence_score:.2f})"
