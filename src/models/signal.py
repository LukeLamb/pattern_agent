"""
Trading Signal Data Model - Core trading signal data structures
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from .pattern import PatternType


class SignalType(str, Enum):
    """Type of trading signal."""

    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"
    HOLD = "hold"
    CLOSE = "close"


class SignalStatus(str, Enum):
    """Signal lifecycle status."""

    PENDING = "pending"
    ACTIVE = "active"
    EXECUTED = "executed"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    STOPPED_OUT = "stopped_out"
    TARGET_HIT = "target_hit"


class SignalPriority(str, Enum):
    """Signal priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PositionSizing(BaseModel):
    """Position sizing parameters."""

    shares: Optional[int] = Field(None, gt=0, description="Number of shares/units")
    dollar_amount: Optional[float] = Field(
        None, gt=0, description="Dollar amount to invest"
    )
    percentage_of_portfolio: Optional[float] = Field(
        None, gt=0, le=100, description="Percentage of portfolio"
    )
    risk_amount: Optional[float] = Field(
        None, gt=0, description="Maximum risk amount in dollars"
    )
    risk_percentage: Optional[float] = Field(
        None, gt=0, le=100, description="Maximum risk as percentage"
    )
    kelly_fraction: Optional[float] = Field(
        None, gt=0, le=1, description="Kelly criterion fraction"
    )
    volatility_adjusted: bool = Field(
        False, description="Whether sizing is volatility adjusted"
    )

    @root_validator
    def validate_sizing_method(cls, values):
        """Ensure at least one sizing method is specified."""
        sizing_fields = [
            "shares",
            "dollar_amount",
            "percentage_of_portfolio",
            "risk_amount",
        ]
        if not any(values.get(field) for field in sizing_fields):
            raise ValueError("At least one position sizing method must be specified")
        return values


class RiskManagement(BaseModel):
    """Risk management parameters."""

    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    stop_loss_percentage: Optional[float] = Field(
        None, gt=0, le=100, description="Stop loss as percentage from entry"
    )
    trailing_stop: Optional[float] = Field(
        None, gt=0, description="Trailing stop distance"
    )
    trailing_stop_percentage: Optional[float] = Field(
        None, gt=0, le=100, description="Trailing stop percentage"
    )

    # Advanced risk management
    time_stop: Optional[datetime] = Field(None, description="Time-based stop")
    volatility_stop: Optional[float] = Field(
        None, gt=0, description="Volatility-based stop multiplier"
    )
    drawdown_stop_percentage: Optional[float] = Field(
        None, gt=0, le=100, description="Maximum drawdown stop"
    )

    # Risk metrics
    max_risk_per_trade: Optional[float] = Field(
        None, gt=0, le=100, description="Maximum risk per trade percentage"
    )
    risk_reward_ratio: Optional[float] = Field(
        None, gt=0, description="Target risk-reward ratio"
    )
    win_rate_required: Optional[float] = Field(
        None, gt=0, le=1, description="Required win rate for profitability"
    )


class TargetPrices(BaseModel):
    """Target price levels and management."""

    primary_target: Optional[float] = Field(
        None, gt=0, description="Primary target price"
    )
    secondary_targets: List[float] = Field(
        default_factory=list, description="Additional target levels"
    )

    # Target management
    scale_out_levels: List[float] = Field(
        default_factory=list, description="Scale out price levels"
    )
    scale_out_percentages: List[float] = Field(
        default_factory=list, description="Scale out percentages at each level"
    )

    # Target calculations
    measured_move_target: Optional[float] = Field(
        None, gt=0, description="Measured move target price"
    )
    fibonacci_targets: List[float] = Field(
        default_factory=list, description="Fibonacci extension targets"
    )
    support_resistance_targets: List[float] = Field(
        default_factory=list, description="Support/resistance targets"
    )

    @validator("scale_out_percentages")
    def validate_scale_out_percentages(cls, v):
        """Validate scale out percentages sum to 100 or less."""
        if v and sum(v) > 100:
            raise ValueError("Scale out percentages cannot exceed 100%")
        return v

    @root_validator
    def validate_scale_out_consistency(cls, values):
        """Ensure scale out levels and percentages are consistent."""
        levels = values.get("scale_out_levels", [])
        percentages = values.get("scale_out_percentages", [])

        if len(levels) != len(percentages) and levels and percentages:
            raise ValueError("Scale out levels and percentages must have same length")

        return values


class ExecutionParameters(BaseModel):
    """Trade execution parameters."""

    order_type: str = Field(
        "market", description="Order type (market, limit, stop, etc.)"
    )
    time_in_force: str = Field("day", description="Time in force (day, gtc, ioc, fok)")
    limit_price: Optional[float] = Field(
        None, gt=0, description="Limit price for limit orders"
    )

    # Advanced execution
    iceberg_size: Optional[int] = Field(
        None, gt=0, description="Iceberg order visible size"
    )
    minimum_fill_size: Optional[int] = Field(
        None, gt=0, description="Minimum fill size"
    )
    all_or_none: bool = Field(False, description="All or none execution")

    # Timing
    execution_window_start: Optional[datetime] = Field(
        None, description="Earliest execution time"
    )
    execution_window_end: Optional[datetime] = Field(
        None, description="Latest execution time"
    )
    market_session: str = Field(
        "regular", description="Market session (regular, pre, post, extended)"
    )


class SignalMetrics(BaseModel):
    """Signal performance and quality metrics."""

    confidence_score: float = Field(
        ..., ge=0, le=1, description="Signal confidence (0-1)"
    )
    strength_score: float = Field(..., ge=0, le=1, description="Signal strength (0-1)")
    quality_score: float = Field(
        ..., ge=0, le=1, description="Overall signal quality (0-1)"
    )

    # Historical performance
    historical_success_rate: Optional[float] = Field(
        None, ge=0, le=1, description="Historical success rate"
    )
    average_return: Optional[float] = Field(
        None, description="Average return from similar signals"
    )
    average_holding_period_days: Optional[float] = Field(
        None, gt=0, description="Average holding period"
    )

    # Pattern-specific metrics
    pattern_reliability: Optional[float] = Field(
        None, ge=0, le=1, description="Pattern reliability score"
    )
    market_condition_score: Optional[float] = Field(
        None, ge=0, le=1, description="Market condition favorability"
    )
    sector_momentum_score: Optional[float] = Field(
        None, ge=0, le=1, description="Sector momentum score"
    )

    # Risk metrics
    sharpe_ratio: Optional[float] = Field(None, description="Expected Sharpe ratio")
    maximum_adverse_excursion: Optional[float] = Field(
        None, description="Maximum adverse excursion"
    )
    maximum_favorable_excursion: Optional[float] = Field(
        None, description="Maximum favorable excursion"
    )


class TradingSignal(BaseModel):
    """
    Comprehensive Trading Signal data model.

    Represents a complete trading signal generated from pattern analysis,
    including entry/exit criteria, risk management, and execution parameters.
    """

    # Core Identification
    signal_id: str = Field(..., description="Unique signal identifier")
    symbol: str = Field(..., min_length=1, description="Trading symbol")
    signal_type: SignalType = Field(..., description="Type of signal")
    status: SignalStatus = Field(
        SignalStatus.PENDING, description="Current signal status"
    )
    priority: SignalPriority = Field(
        SignalPriority.MEDIUM, description="Signal priority"
    )

    # Source Information
    source_pattern_id: Optional[str] = Field(
        None, description="Source pattern ID if applicable"
    )
    pattern_type: Optional[PatternType] = Field(None, description="Source pattern type")
    timeframe: str = Field(..., description="Primary analysis timeframe")
    supporting_timeframes: List[str] = Field(
        default_factory=list, description="Supporting timeframes"
    )

    # Price Information
    entry_price: float = Field(..., gt=0, description="Recommended entry price")
    current_price: Optional[float] = Field(
        None, gt=0, description="Current market price"
    )

    # Targets and Risk Management
    targets: TargetPrices = Field(..., description="Target price levels")
    risk_management: RiskManagement = Field(
        ..., description="Risk management parameters"
    )
    position_sizing: PositionSizing = Field(
        ..., description="Position sizing parameters"
    )

    # Execution
    execution_params: ExecutionParameters = Field(
        ..., description="Execution parameters"
    )

    # Signal Quality and Metrics
    metrics: SignalMetrics = Field(..., description="Signal quality metrics")

    # Market Context
    market_context: Dict[str, Any] = Field(
        default_factory=dict, description="Market conditions at signal generation"
    )
    technical_indicators: Dict[str, float] = Field(
        default_factory=dict, description="Technical indicator values"
    )

    # Timeline
    generated_at: datetime = Field(
        default_factory=datetime.now, description="When signal was generated"
    )
    valid_until: Optional[datetime] = Field(None, description="Signal expiry time")
    executed_at: Optional[datetime] = Field(
        None, description="When signal was executed"
    )
    closed_at: Optional[datetime] = Field(None, description="When position was closed")

    # Performance Tracking
    fill_price: Optional[float] = Field(None, gt=0, description="Actual fill price")
    exit_price: Optional[float] = Field(None, gt=0, description="Exit price")
    realized_return: Optional[float] = Field(
        None, description="Realized return percentage"
    )
    unrealized_return: Optional[float] = Field(
        None, description="Current unrealized return percentage"
    )
    max_favorable_excursion: Optional[float] = Field(
        None, description="Maximum favorable move"
    )
    max_adverse_excursion: Optional[float] = Field(
        None, description="Maximum adverse move"
    )

    # Metadata
    created_by: str = Field("pattern_recognition_agent", description="Signal creator")
    version: str = Field("1.0", description="Signal model version")
    notes: List[str] = Field(default_factory=list, description="Additional notes")
    tags: List[str] = Field(default_factory=list, description="Signal tags")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            SignalType: lambda v: v.value,
            SignalStatus: lambda v: v.value,
            SignalPriority: lambda v: v.value,
            PatternType: lambda v: v.value if v else None,
        }

    @validator("valid_until", always=True)
    def set_default_expiry(cls, v, values):
        """Set default expiry if not provided."""
        if v is None and "generated_at" in values:
            # Default: valid for 24 hours
            return values["generated_at"] + timedelta(hours=24)
        return v

    @root_validator
    def validate_timeline(cls, values):
        """Validate timeline consistency."""
        generated_at = values.get("generated_at")
        valid_until = values.get("valid_until")
        executed_at = values.get("executed_at")
        closed_at = values.get("closed_at")

        if generated_at and valid_until:
            if valid_until <= generated_at:
                raise ValueError("Valid until must be after generated at")

        if executed_at and generated_at:
            if executed_at < generated_at:
                raise ValueError("Executed at cannot be before generated at")

        if closed_at and executed_at:
            if closed_at < executed_at:
                raise ValueError("Closed at cannot be before executed at")

        return values

    @root_validator
    def validate_prices(cls, values):
        """Validate price relationships."""
        entry_price = values.get("entry_price")
        targets = values.get("targets")
        risk_mgmt = values.get("risk_management")
        signal_type = values.get("signal_type")

        if not all([entry_price, targets, risk_mgmt, signal_type]):
            return values

        # Validate target prices make sense for signal direction
        if signal_type in [SignalType.BUY, SignalType.BUY_TO_COVER]:
            if targets.primary_target and targets.primary_target <= entry_price:
                raise ValueError("Buy signal target must be above entry price")
        elif signal_type in [SignalType.SELL, SignalType.SELL_SHORT]:
            if targets.primary_target and targets.primary_target >= entry_price:
                raise ValueError("Sell signal target must be below entry price")

        # Validate stop loss
        if risk_mgmt.stop_loss:
            if signal_type in [SignalType.BUY, SignalType.BUY_TO_COVER]:
                if risk_mgmt.stop_loss >= entry_price:
                    raise ValueError("Buy signal stop loss must be below entry price")
            elif signal_type in [SignalType.SELL, SignalType.SELL_SHORT]:
                if risk_mgmt.stop_loss <= entry_price:
                    raise ValueError("Sell signal stop loss must be above entry price")

        return values

    def update_status(
        self, new_status: SignalStatus, timestamp: Optional[datetime] = None
    ):
        """Update signal status with timestamp."""
        self.status = new_status
        timestamp = timestamp or datetime.now()

        if new_status == SignalStatus.EXECUTED and not self.executed_at:
            self.executed_at = timestamp
        elif (
            new_status in [SignalStatus.TARGET_HIT, SignalStatus.STOPPED_OUT]
            and not self.closed_at
        ):
            self.closed_at = timestamp

    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.valid_until:
            return datetime.now() > self.valid_until
        return False

    def is_active(self) -> bool:
        """Check if signal is currently active."""
        return (
            self.status
            in [
                SignalStatus.PENDING,
                SignalStatus.ACTIVE,
                SignalStatus.EXECUTED,
                SignalStatus.FILLED,
                SignalStatus.PARTIAL_FILL,
            ]
            and not self.is_expired()
        )

    def get_age_hours(self) -> float:
        """Get signal age in hours."""
        return (datetime.now() - self.generated_at).total_seconds() / 3600

    def calculate_current_return(self, current_price: float) -> float:
        """Calculate current return percentage."""
        if not self.fill_price:
            entry = self.entry_price
        else:
            entry = self.fill_price

        if self.signal_type in [SignalType.BUY, SignalType.BUY_TO_COVER]:
            return ((current_price - entry) / entry) * 100
        else:
            return ((entry - current_price) / entry) * 100

    def calculate_risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk-reward ratio."""
        if not all(
            [
                self.entry_price,
                self.targets.primary_target,
                self.risk_management.stop_loss,
            ]
        ):
            return None

        if self.signal_type in [SignalType.BUY, SignalType.BUY_TO_COVER]:
            potential_reward = self.targets.primary_target - self.entry_price
            potential_risk = self.entry_price - self.risk_management.stop_loss
        else:
            potential_reward = self.entry_price - self.targets.primary_target
            potential_risk = self.risk_management.stop_loss - self.entry_price

        if potential_risk > 0:
            return potential_reward / potential_risk
        return None

    def get_priority_score(self) -> int:
        """Get numeric priority score for sorting."""
        priority_scores = {
            SignalPriority.LOW: 1,
            SignalPriority.MEDIUM: 2,
            SignalPriority.HIGH: 3,
            SignalPriority.CRITICAL: 4,
        }
        return priority_scores.get(self.priority, 2)

    def add_note(self, note: str):
        """Add a note to the signal."""
        self.notes.append(f"{datetime.now().isoformat()}: {note}")

    def add_tag(self, tag: str):
        """Add a tag to the signal."""
        if tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return self.dict()

    def to_json(self) -> str:
        """Convert signal to JSON string."""
        return self.json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSignal":
        """Create signal from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "TradingSignal":
        """Create signal from JSON string."""
        return cls.parse_raw(json_str)

    def __str__(self) -> str:
        """String representation of signal."""
        return f"{self.signal_type.value.upper()} {self.symbol} @ {self.entry_price} - Priority: {self.priority.value} - Confidence: {self.metrics.confidence_score:.2f}"

    def __repr__(self) -> str:
        """Detailed representation of signal."""
        return f"TradingSignal(id={self.signal_id}, type={self.signal_type.value}, symbol={self.symbol}, status={self.status.value}, confidence={self.metrics.confidence_score:.2f})"
