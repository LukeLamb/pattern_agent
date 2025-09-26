# Pattern Recognition Agent

A sophisticated AI-powered trading pattern recognition system that analyzes market data to identify high-probability trading patterns and generate actionable trading signals.

## Overview

This Pattern Recognition Agent operates as a Level 3-4 Agentic AI system following the SPAR framework (Sense, Plan, Act, Reflect) and serves as the analytical brain for trading decision support.

## Key Features

- **Multi-Pattern Detection**: Identifies 15+ classic technical analysis patterns
- **Multi-Timeframe Analysis**: Analyzes patterns across multiple timeframes
- **Real-time Processing**: Sub-500ms pattern detection latency
- **Machine Learning**: Adaptive pattern detection based on historical performance
- **Risk Management**: Built-in risk-reward calculations and position sizing
- **Memory Integration**: Persistent learning using MCP memory server

## Supported Patterns

### Continuation Patterns

- Triangles (ascending, descending, symmetrical)
- Flags and pennants
- Rectangles and channels
- Cup and handle formations

### Reversal Patterns

- Head and shoulders (regular and inverse)
- Double/triple tops and bottoms
- Wedges (rising and falling)
- Rounding tops and bottoms

### Breakout Patterns

- Support/resistance breaks
- Trendline breaks
- Volume breakouts
- Gap patterns

## Architecture

```bash
src/
├── pattern_detection/     # Core pattern recognition engines
├── technical_indicators/  # Technical analysis calculations
├── signal_generation/     # Trading signal generation
├── market_context/        # Market regime and context analysis
├── validation/           # Pattern validation and backtesting
└── api/                 # REST and WebSocket APIs
```

## Installation

1. Clone the repository:

```bash
git clone git@github.com:LukeLamb/pattern_agent.git
cd pattern_agent
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure MCP servers (see `.mcp.json`)

## Configuration

### MCP Server Setup

The project uses several MCP servers for enhanced functionality:

- **Memory Server**: Persistent pattern learning and tracking
- **Time Server**: Market session and timing analysis
- **Context7 Server**: Current technical analysis documentation
- **Sequential Thinking**: Complex pattern analysis workflows

### Settings

Configure the agent through `settings.local.json`:

```json
{
  "enableAllProjectMcpServers": true,
  "dangerouslySkipPermissions": true,
  "pattern_detection": {
    "min_confidence_threshold": 0.7,
    "max_patterns_per_symbol": 5
  }
}
```

## Usage

### Basic Pattern Detection

```python
from src.pattern_detection import PatternDetectionEngine

engine = PatternDetectionEngine()
patterns = engine.detect_patterns(market_data, timeframe='daily')
```

### Real-time Signal Generation

```python
from src.signal_generation import SignalGenerator

generator = SignalGenerator()
signals = generator.generate_signals(patterns, market_context)
```

### API Endpoints

- `GET /api/v1/patterns/{symbol}` - Get patterns for symbol
- `GET /api/v1/signals` - Get active trading signals
- `WS /api/v1/stream/patterns` - Real-time pattern updates

## Performance Targets

- **Accuracy**: 80%+ on pattern identification
- **Success Rate**: 70%+ on trading signals
- **Latency**: <500ms pattern detection
- **Uptime**: 99%+ during market hours
- **False Positives**: <5% on high-confidence signals

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## Development Phases

- [x] **Phase 1**: Core pattern detection algorithms
- [ ] **Phase 2**: Multi-timeframe analysis
- [ ] **Phase 3**: Advanced analytics and learning
- [ ] **Phase 4**: Production deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue on GitHub.
