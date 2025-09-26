# Pattern Recognition Agent - Development Prompt

## Project Overview

You are tasked with building a sophisticated Pattern Recognition Agent that analyzes market data to identify high-probability trading patterns and setups. This agent operates as a Level 3-4 Agentic AI system following the SPAR framework (Sense, Plan, Act, Reflect) and serves as the analytical brain for trading decision support.

## Core Objectives

### Primary Goals

- Identify classic technical analysis patterns with high accuracy
- Detect emerging patterns and anomalies in real-time
- Provide probability-based pattern assessments using historical data
- Learn and adapt pattern detection based on market outcomes
- Generate actionable trading signals with confidence scores

### Success Metrics

- 80%+ accuracy on classic pattern identification
- 70%+ success rate on pattern-based trade recommendations
- Sub-500ms pattern detection latency for real-time analysis
- 90%+ uptime during market hours
- Continuous learning and improvement of pattern detection algorithms

## Agentic AI Framework Implementation

### SPAR Framework Application

#### Sense (Data Ingestion & Processing)

- **Multi-source market data**: OHLCV, volume, volatility indicators
- **Multi-timeframe analysis**: 1min, 5min, 15min, 1hr, daily, weekly
- **Cross-asset correlation**: Monitor related symbols and sectors
- **Market context**: Volatility regimes, trend direction, market breadth

#### Plan (Pattern Analysis Strategy)

- **Pattern classification**: Categorize detected patterns by type and strength
- **Confirmation requirements**: Multi-timeframe pattern validation
- **Risk assessment**: Evaluate pattern failure probability
- **Execution timing**: Optimal entry/exit point identification

#### Act (Signal Generation & Communication)

- **Alert generation**: Create prioritized trading alerts
- **Pattern visualization**: Generate charts and pattern annotations
- **Signal distribution**: Notify other agents and human operators
- **Pattern tracking**: Monitor active patterns through completion

#### Reflect (Learning & Adaptation)

- **Outcome tracking**: Monitor pattern success/failure rates
- **Algorithm refinement**: Adjust detection parameters based on performance
- **Market regime awareness**: Adapt to changing market conditions
- **Memory integration**: Store learnings for future pattern recognition

## Technical Architecture

### Core Components

#### 1. Pattern Detection Engine

**Responsibility**: Core pattern recognition and classification
**Supported Patterns**:

**Continuation Patterns**:

- Triangles (ascending, descending, symmetrical)
- Flags and pennants
- Rectangles and channels
- Cup and handle formations

**Reversal Patterns**:

- Head and shoulders (regular and inverse)
- Double/triple tops and bottoms
- Wedges (rising and falling)
- Rounding tops and bottoms

**Breakout Patterns**:

- Support/resistance breaks
- Trendline breaks
- Volume breakouts
- Gap patterns

**Implementation Structure**:

```python
class PatternDetectionEngine:
    def __init__(self):
        self.pattern_detectors = {}
        self.confidence_thresholds = {}
        self.historical_success_rates = {}
        
    def detect_patterns(self, market_data, timeframe):
        # Multi-pattern detection pipeline
        
    def calculate_pattern_strength(self, pattern, context):
        # Historical success rate + current market conditions
        
    def validate_pattern(self, pattern, multiple_timeframes):
        # Cross-timeframe pattern confirmation
```

#### 2. Technical Indicator Engine

**Responsibility**: Calculate and monitor technical indicators
**Supported Indicators**:

**Trend Indicators**:

- Moving averages (SMA, EMA, WMA)
- MACD and signal lines
- Parabolic SAR
- Average Directional Index (ADX)

**Momentum Indicators**:

- RSI (Relative Strength Index)
- Stochastic oscillator
- Williams %R
- Rate of Change (ROC)

**Volume Indicators**:

- On-Balance Volume (OBV)
- Volume Price Trend (VPT)
- Accumulation/Distribution Line
- Chaikin Money Flow

**Volatility Indicators**:

- Bollinger Bands
- Average True Range (ATR)
- Volatility Index
- Standard deviation bands

#### 3. Multi-Timeframe Analyzer

**Responsibility**: Coordinate pattern analysis across different timeframes
**Features**:

- Timeframe hierarchy validation
- Pattern consistency checks
- Trend alignment analysis
- Signal strength aggregation

**Timeframe Strategy**:

```python
class MultiTimeframeAnalyzer:
    def __init__(self):
        self.timeframes = ['1min', '5min', '15min', '1hr', 'daily', 'weekly']
        self.pattern_weights = {
            'daily': 1.0,    # Highest weight
            '1hr': 0.8,
            '15min': 0.6,
            '5min': 0.4,
            '1min': 0.2      # Lowest weight
        }
    
    def analyze_pattern_confluence(self, symbol, pattern_type):
        # Check pattern across multiple timeframes
        
    def calculate_composite_signal(self, timeframe_signals):
        # Weighted signal strength calculation
```

#### 4. Market Context Analyzer

**Responsibility**: Provide market regime and environmental context
**Context Factors**:

- Market volatility regime (low, medium, high)
- Trend direction (bull, bear, sideways)
- Sector rotation patterns
- Overall market breadth
- Economic calendar events

#### 5. Pattern Validation System

**Responsibility**: Validate patterns using historical data and market context
**Validation Criteria**:

- **Historical Success Rate**: Pattern performance in similar market conditions
- **Volume Confirmation**: Adequate volume supporting the pattern
- **Timeframe Alignment**: Pattern consistency across timeframes
- **Market Context**: Pattern appropriateness for current market regime

#### 6. Signal Generation Engine

**Responsibility**: Convert patterns into actionable trading signals
**Signal Components**:

```python
class TradingSignal:
    def __init__(self):
        self.symbol = ""
        self.pattern_type = ""
        self.signal_strength = 0.0  # 0-100
        self.confidence_score = 0.0  # 0-100
        self.entry_price = 0.0
        self.target_prices = []     # Multiple targets
        self.stop_loss = 0.0
        self.timeframe = ""
        self.risk_reward_ratio = 0.0
        self.historical_success_rate = 0.0
        self.market_context = {}
        self.expiry_time = None
```

## MCP Server Integration

### Memory Server Usage

Store and track comprehensive pattern intelligence:

**Entities**:

```bash
- pattern_type (name: "head_and_shoulders", base_success_rate: 0.72)
- symbol_pattern (symbol: "AAPL", pattern: "ascending_triangle", date: "2025-01-15")
- market_regime (type: "high_volatility", start_date: "2025-01-10", vix_level: 28.5)
- pattern_outcome (pattern_id, success: true, actual_move: 0.08, expected_move: 0.06)
- trading_signal (signal_id, symbol, entry_price, outcome, profit_loss: 0.045)
```

**Relationships**:

```bash
- pattern_type -> performs_better_in -> market_regime
- symbol -> exhibits -> pattern_type -> during -> time_period
- trading_signal -> based_on -> pattern_outcome -> in -> market_context
- market_regime -> affects -> pattern_success_rate
```

### Sequential Thinking Integration

Use for complex pattern analysis workflows:

**Pattern Detection Workflow**:

1. Receive market data from Market Data Agent
2. Scan for pattern formations across all timeframes
3. Calculate pattern strength and confidence scores
4. Validate patterns using historical success rates
5. Check market context and regime appropriateness
6. Generate trading signals with risk parameters
7. Update memory with pattern tracking information

**Multi-Timeframe Validation Workflow**:

1. Detect pattern on primary timeframe
2. Check for pattern consistency on higher timeframes
3. Validate with lower timeframe entries
4. Calculate weighted confidence score
5. Determine optimal entry and exit points

### Time Server Integration

- Pattern formation timing analysis
- Market session awareness (pre-market, regular, after-hours)
- Pattern expiry calculations
- Historical pattern performance by time of day/week

### Context7 Server Integration

- Stay current with technical analysis methodologies
- Access latest financial library documentation
- Validate indicator calculations against standard implementations

## Pattern Library Specifications

### 1. Triangle Patterns

```python
class TrianglePattern:
    def detect_ascending_triangle(self, price_data):
        # Flat resistance, rising support
        # Minimum 4 touches (2 resistance, 2 support)
        # Volume typically decreases during formation
        # Breakout volume should be 50%+ above average
        
    def detect_descending_triangle(self, price_data):
        # Flat support, declining resistance
        
    def detect_symmetrical_triangle(self, price_data):
        # Converging support and resistance lines
        # Usually 4-6 weeks duration
        # Volume decreases as pattern develops
```

### 2. Head and Shoulders

```python
class HeadShouldersPattern:
    def detect_head_shoulders(self, price_data):
        # Three peaks: left shoulder, head (highest), right shoulder
        # Neckline connects lows between shoulders and head
        # Right shoulder typically on lower volume
        # Target: neckline break distance from head to neckline
        
    def detect_inverse_head_shoulders(self, price_data):
        # Three troughs: left shoulder, head (lowest), right shoulder
        # Bullish reversal pattern
```

### 3. Flag and Pennant Patterns

```python
class FlagPennantPattern:
    def detect_flag(self, price_data):
        # Sharp move followed by consolidation
        # Flag slopes against trend direction
        # Duration: 1-3 weeks typically
        # Volume decreases during flag formation
        
    def detect_pennant(self, price_data):
        # Sharp move followed by small symmetrical triangle
        # Brief consolidation period
```

## Implementation Phases

### Phase 1: Core Pattern Detection (Week 1-2)

- [ ] Implement basic pattern detection algorithms
- [ ] Create 5-10 classic pattern detectors
- [ ] Build simple technical indicator calculations
- [ ] Test with historical data from Market Data Agent
- [ ] Create basic pattern visualization

### Phase 2: Multi-Timeframe Analysis (Week 3-4)

- [ ] Implement multi-timeframe pattern validation
- [ ] Add weighted signal calculation
- [ ] Create pattern strength scoring system
- [ ] Integrate with memory server for pattern tracking
- [ ] Add basic market context awareness

### Phase 3: Advanced Analytics (Week 5-6)

- [ ] Implement comprehensive technical indicators
- [ ] Add pattern outcome tracking and learning
- [ ] Create advanced signal generation
- [ ] Add risk-reward ratio calculations
- [ ] Implement pattern expiry and timing

### Phase 4: Production Ready (Week 7-8)

- [ ] Add comprehensive error handling
- [ ] Implement real-time pattern monitoring
- [ ] Create pattern performance analytics
- [ ] Add alert system and notifications
- [ ] Complete integration testing with Market Data Agent

## Pattern Detection Algorithms

### Support and Resistance Detection

```python
def detect_support_resistance(price_data, lookback=20, min_touches=2):
    """
    Detect support and resistance levels using pivot points
    """
    pivots = find_pivot_points(price_data, lookback)
    levels = cluster_pivot_points(pivots, tolerance=0.02)
    
    support_levels = []
    resistance_levels = []
    
    for level in levels:
        touches = count_level_touches(price_data, level, tolerance=0.01)
        if touches >= min_touches:
            if level < current_price:
                support_levels.append({
                    'price': level,
                    'strength': touches,
                    'last_touch': get_last_touch_date(price_data, level)
                })
            else:
                resistance_levels.append({
                    'price': level,
                    'strength': touches,
                    'last_touch': get_last_touch_date(price_data, level)
                })
    
    return support_levels, resistance_levels
```

### Trend Analysis

```python
def analyze_trend(price_data, timeframe='daily'):
    """
    Multi-method trend analysis
    """
    # Method 1: Higher highs, higher lows
    hh_hl = detect_higher_highs_lows(price_data)
    
    # Method 2: Moving average alignment
    ma_trend = analyze_ma_alignment(price_data)
    
    # Method 3: Trendline analysis
    trendlines = detect_trendlines(price_data)
    
    # Composite trend score
    trend_score = calculate_composite_trend(hh_hl, ma_trend, trendlines)
    
    return {
        'direction': get_trend_direction(trend_score),
        'strength': abs(trend_score),
        'components': {
            'hh_hl': hh_hl,
            'moving_averages': ma_trend,
            'trendlines': trendlines
        }
    }
```

## Signal Generation Logic

### Signal Prioritization

```python
class SignalPrioritizer:
    def prioritize_signals(self, signals):
        priorities = []
        
        for signal in signals:
            priority_score = (
                signal.confidence_score * 0.4 +
                signal.historical_success_rate * 0.3 +
                signal.risk_reward_ratio * 0.2 +
                signal.volume_confirmation * 0.1
            )
            
            # Adjust for market context
            if signal.market_context.get('regime') == 'trending':
                if signal.pattern_type in ['breakout', 'continuation']:
                    priority_score *= 1.2
            
            priorities.append((signal, priority_score))
        
        return sorted(priorities, key=lambda x: x[1], reverse=True)
```

### Risk-Reward Calculation

```python
def calculate_risk_reward(entry_price, target_price, stop_loss):
    """
    Calculate risk-reward ratio for a trade setup
    """
    potential_profit = abs(target_price - entry_price)
    potential_loss = abs(entry_price - stop_loss)
    
    if potential_loss == 0:
        return float('inf')
    
    risk_reward_ratio = potential_profit / potential_loss
    
    # Minimum acceptable ratio
    if risk_reward_ratio < 1.5:
        return None  # Reject poor risk-reward setups
    
    return risk_reward_ratio
```

## Pattern Validation Framework

### Historical Success Rate Calculation

```python
def calculate_pattern_success_rate(pattern_type, market_conditions, lookback_days=365):
    """
    Calculate historical success rate for a pattern type
    """
    historical_patterns = memory_server.query(
        entity_type="pattern_outcome",
        filters={
            "pattern_type": pattern_type,
            "market_conditions": market_conditions,
            "date_range": (datetime.now() - timedelta(days=lookback_days), datetime.now())
        }
    )
    
    if not historical_patterns:
        return 0.5  # Default 50% if no historical data
    
    successful_patterns = [p for p in historical_patterns if p.success]
    success_rate = len(successful_patterns) / len(historical_patterns)
    
    # Weight recent patterns more heavily
    weighted_success_rate = calculate_time_weighted_success(historical_patterns)
    
    return weighted_success_rate
```

### Market Regime Adaptation

```python
def adapt_to_market_regime(pattern_detection_params, current_regime):
    """
    Adjust pattern detection parameters based on market regime
    """
    if current_regime == "high_volatility":
        # Require stronger confirmation in volatile markets
        pattern_detection_params['min_confirmation_score'] *= 1.2
        pattern_detection_params['volume_threshold'] *= 1.5
        
    elif current_regime == "trending":
        # Lower thresholds for continuation patterns
        pattern_detection_params['continuation_pattern_threshold'] *= 0.8
        
    elif current_regime == "range_bound":
        # Favor reversal patterns
        pattern_detection_params['reversal_pattern_weight'] *= 1.3
        
    return pattern_detection_params
```

## API Design for Integration

### REST Endpoints

```python
# Get current patterns for symbol
GET /api/v1/patterns/{symbol}?timeframe=daily

# Get all active patterns
GET /api/v1/patterns/active

# Get pattern analysis
GET /api/v1/analysis/{symbol}?timeframes=5min,15min,1hr

# Get trading signals
GET /api/v1/signals?min_confidence=70

# Pattern performance statistics
GET /api/v1/patterns/{pattern_type}/performance

# Market context analysis
GET /api/v1/market/context
```

### WebSocket Streams

```python
# Real-time pattern detection
WS /api/v1/stream/patterns
Subscribe: {"symbols": ["AAPL", "GOOGL"], "min_confidence": 75}

# Trading signals stream
WS /api/v1/stream/signals
Subscribe: {"signal_types": ["breakout", "reversal"]}
```

## Performance Monitoring

### Key Metrics to Track

- **Pattern Detection Accuracy**: % of correctly identified patterns
- **Signal Success Rate**: % of profitable signals generated
- **False Positive Rate**: % of incorrect pattern identifications
- **Response Time**: Time from pattern formation to signal generation
- **Pattern Coverage**: % of significant market moves captured

### Performance Dashboard

```python
class PerformanceDashboard:
    def generate_performance_report(self, period_days=30):
        return {
            'pattern_accuracy': self.calculate_pattern_accuracy(period_days),
            'signal_success_rate': self.calculate_signal_success(period_days),
            'best_performing_patterns': self.get_top_patterns(period_days),
            'worst_performing_patterns': self.get_bottom_patterns(period_days),
            'market_regime_performance': self.analyze_regime_performance(period_days),
            'timeframe_effectiveness': self.analyze_timeframe_performance(period_days)
        }
```

## Testing Strategy

### Backtesting Framework

```python
class PatternBacktester:
    def backtest_pattern(self, pattern_type, start_date, end_date):
        """
        Comprehensive backtesting of pattern performance
        """
        historical_data = self.get_historical_data(start_date, end_date)
        detected_patterns = self.detect_patterns(historical_data, pattern_type)
        
        results = []
        for pattern in detected_patterns:
            # Simulate trade execution
            entry_price = pattern.breakout_price
            target_price = pattern.target_price
            stop_loss = pattern.stop_loss
            
            # Calculate outcome
            outcome = self.simulate_trade_outcome(
                entry_price, target_price, stop_loss, 
                historical_data, pattern.formation_date
            )
            
            results.append({
                'pattern': pattern,
                'outcome': outcome,
                'profit_loss': outcome.profit_loss,
                'success': outcome.success
            })
        
        return self.analyze_backtest_results(results)
```

### Unit Testing Framework

```python
def test_triangle_detection():
    # Test ascending triangle detection with known data
    test_data = create_ascending_triangle_data()
    pattern = TrianglePattern().detect_ascending_triangle(test_data)
    
    assert pattern is not None
    assert pattern.type == "ascending_triangle"
    assert pattern.confidence_score > 0.7

def test_support_resistance():
    # Test support/resistance level detection
    test_data = create_support_resistance_data()
    support, resistance = detect_support_resistance(test_data)
    
    assert len(support) > 0
    assert len(resistance) > 0
    assert support[0]['strength'] >= 2
```

## Configuration Management

### Pattern Detection Configuration

```yaml
# pattern_config.yaml
pattern_detection:
  triangle_patterns:
    min_touches: 4
    max_duration_days: 90
    min_price_range: 0.05
    volume_confirmation_threshold: 1.5
    
  head_shoulders:
    shoulder_symmetry_tolerance: 0.1
    neckline_slope_tolerance: 0.02
    volume_decline_requirement: true
    
  breakout_patterns:
    volume_multiplier: 2.0
    price_confirmation_percentage: 0.02
    max_false_breakout_tolerance: 3
    
market_regimes:
  high_volatility:
    vix_threshold: 25
    pattern_confirmation_multiplier: 1.2
    
  trending:
    adx_threshold: 25
    continuation_pattern_weight: 1.3
```

### Alert Configuration

```yaml
# alert_config.yaml
alerts:
  signal_generation:
    min_confidence_score: 70
    max_alerts_per_hour: 10
    duplicate_signal_timeout: 3600  # seconds
    
  pattern_formation:
    notify_on_pattern_completion: true
    notify_on_pattern_failure: true
    min_pattern_strength: 0.6
```

## Learning and Adaptation System

### Pattern Outcome Tracking

```python
class PatternLearningSystem:
    def update_pattern_performance(self, pattern_id, outcome):
        """
        Update pattern performance based on actual outcomes
        """
        pattern_outcome = {
            'pattern_id': pattern_id,
            'success': outcome.profitable,
            'actual_return': outcome.return_percentage,
            'duration_hours': outcome.duration,
            'market_conditions': outcome.market_context,
            'timestamp': datetime.now()
        }
        
        # Store in memory server
        memory_server.create_entity(
            entity_type="pattern_outcome",
            properties=pattern_outcome
        )
        
        # Update pattern success rates
        self.recalculate_pattern_success_rates(pattern_id.pattern_type)
        
        # Adapt detection parameters if needed
        self.adapt_detection_parameters(pattern_outcome)
```

### Adaptive Parameter Tuning

```python
def adapt_detection_parameters(self, recent_outcomes):
    """
    Automatically adjust pattern detection parameters based on performance
    """
    if self.calculate_recent_success_rate() < 0.6:
        # Increase confirmation requirements
        self.increase_pattern_thresholds()
        
    if self.calculate_false_positive_rate() > 0.3:
        # Require stronger pattern confirmation
        self.strengthen_validation_criteria()
        
    # Market regime specific adaptations
    current_regime = self.determine_current_market_regime()
    self.apply_regime_specific_parameters(current_regime)
```

## Success Criteria

### Technical Metrics

- [ ] 80%+ accuracy on pattern identification (validated through backtesting)
- [ ] 70%+ success rate on trading signals (profitable outcomes)
- [ ] <500ms pattern detection latency for real-time analysis
- [ ] 99%+ uptime during market hours
- [ ] <5% false positive rate on high-confidence signals

### Business Metrics

- [ ] Support 100+ symbols with real-time pattern monitoring
- [ ] Generate 5-15 high-quality signals per trading day
- [ ] Achieve 2:1 or better average risk-reward ratio on signals
- [ ] Demonstrate continuous learning and improvement over time
- [ ] Successfully integrate with Market Data Agent for seamless operation

## Integration with Market Data Agent

### Data Pipeline

```python
class MarketDataIntegration:
    def __init__(self, market_data_agent_endpoint):
        self.market_data_client = MarketDataClient(market_data_agent_endpoint)
        self.data_cache = {}
        
    def subscribe_to_real_time_data(self, symbols):
        """
        Subscribe to real-time market data for pattern detection
        """
        for symbol in symbols:
            self.market_data_client.subscribe_websocket(
                symbol=symbol,
                callback=self.process_real_time_data,
                data_types=['ohlcv', 'volume', 'trades']
            )
    
    def process_real_time_data(self, data):
        """
        Process incoming real-time data for pattern detection
        """
        # Update local cache
        self.update_data_cache(data)
        
        # Trigger pattern detection
        self.trigger_pattern_analysis(data.symbol)
        
        # Check for pattern completions
        self.check_pattern_completions(data.symbol)
```

## Next Steps After Completion

Once the Pattern Recognition Agent is complete and integrated:

1. **Risk Management Agent**: Add position sizing and risk controls
2. **Portfolio Coordination Agent**: Multi-symbol portfolio management
3. **News Sentiment Agent**: Add fundamental context to technical patterns
4. **Execution Agent**: Automate trade execution based on signals
5. **Performance Analytics Agent**: Comprehensive strategy performance tracking

## Resources and References

### Technical Analysis Resources

- [Technical Analysis of the Financial Markets - John J. Murphy](https://www.amazon.com/Technical-Analysis-Financial-Markets-Comprehensive/dp/0735200661)
- [Encyclopedia of Chart Patterns - Thomas Bulkowski](https://www.amazon.com/Encyclopedia-Chart-Patterns-Thomas-Bulkowski/dp/1119739683)
- [TA-Lib Technical Analysis Library](https://ta-lib.org/)

### Pattern Recognition Papers

- [Automated Technical Analysis Pattern Recognition](https://arxiv.org/abs/1509.08936)
- [Machine Learning for Financial Market Pattern Recognition](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3420722)

### Implementation Libraries

- [pandas-ta](https://github.com/twopirllc/pandas-ta) - Technical Analysis Indicators
- [ta](https://github.com/bukosabino/ta) - Technical Analysis Library
- [mplfinance](https://github.com/matplotlib/mplfinance) - Financial plotting
- [scikit-learn](https://scikit-learn.org/) - Machine learning algorithms

---

**Remember**: Focus on accuracy over quantity. A few high-quality, well-validated patterns will be more valuable than many mediocre signals. The pattern recognition agent should complement human intuition, not replace it entirely.
