# Pattern Recognition Agent

A sophisticated AI-powered trading pattern recognition system that analyzes market data to identify high-probability trading patterns and generate actionable trading signals.

## 📊 Project Status

**Current Phase:** Phase 2.1, 2.2 & 2.3 Complete ✅ → Phase 2.4 Next
**Total Codebase:** 10,250+ lines of production Python
**Test Coverage:** 78/78 tests passing (100% pass rate)
**Performance:** <500ms pattern detection, multi-timeframe confluence, context-aware adaptation

## Overview

This Pattern Recognition Agent operates as a Level 3-4 Agentic AI system following the SPAR framework (Sense, Plan, Act, Reflect) and serves as the analytical brain for trading decision support.

## 🎯 Current Capabilities (Phase 1 & Phase 2.1-2.3 Complete)

### Core Features
- **23 Technical Indicators** across 4 categories (Trend, Momentum, Volume, Volatility)
- **15 Pattern Types** with mathematical validation (5 basic + 10 advanced)
- **Multi-Timeframe Analysis** with 9 timeframe support and confluence scoring
- **Market Context Analysis** with 4 volatility regimes, 4 trend directions, 5 market regimes
- **Adaptive Parameters** that adjust based on market conditions
- **5-Criteria Validation** system with weighted scoring (0.0-1.0 scale)
- **Multi-Format Output** (PNG charts, JSON reports, CSV exports)
- **Production-Ready Performance** (<500ms target achieved)

## 📋 Implemented Patterns

### ✅ Basic Patterns (Phase 1 - 5 Patterns)
- **Triangles** - Ascending, Descending, Symmetrical (with volume confirmation)
- **Head & Shoulders** - Classic bearish and Inverse bullish reversals

### ✅ Advanced Patterns (Phase 2.2 - 10 Patterns)
- **Flags & Pennants** - Bull Flag, Bear Flag, Pennant (continuation patterns)
- **Double/Triple Tops/Bottoms** - Multiple peak/trough reversals with volume divergence
- **Rectangles & Channels** - Horizontal rectangles, Ascending/Descending channels

### 🚧 Coming Soon (Phase 3+)
- **Cup & Handle** - Bullish continuation pattern
- **Wedges** - Rising/Falling wedge patterns
- **Gaps** - Breakaway, Runaway, Exhaustion gaps
- **Complex Reversals** - Island reversals, Key reversals

## 🏗️ Architecture

```bash
src/
├── models/                # Core data structures (Pattern, MarketData, Signal)
├── pattern_detection/     # 15 pattern types with mathematical validation
├── technical_indicators/  # 23 indicators across 4 categories
├── validation/            # 5-criteria pattern validation system
├── visualization/         # Chart generation with matplotlib/seaborn
├── reporting/             # JSON/CSV/HTML report generation
├── analysis/              # Multi-timeframe analysis (Phase 2.1) ✅
├── market_context/        # Market regime analysis (Phase 2.3) ✅
├── signal_generation/     # Trading signal generation (Phase 2.5+)
└── memory/                # Pattern learning system (Phase 2.5+)

tests/                     # Comprehensive test suite (78/78 passing)
├── test_integration.py              # 7 integration tests
├── test_multi_timeframe.py          # 18 multi-timeframe tests
├── test_advanced_patterns.py        # 25 advanced pattern tests
├── test_market_context.py           # 28 market context tests
└── ...

docs/
├── TODO.md                # Complete roadmap and task tracking
├── architecture/          # Design documents and phase details
├── reports/               # Phase completion reports
└── guides/                # Implementation guides (coming soon)
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

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests (78/78 tests passing)
pytest tests/ -v --cov=src

# Individual test categories
pytest tests/test_integration.py -v           # 7 integration tests
pytest tests/test_multi_timeframe.py -v       # 18 multi-timeframe tests
pytest tests/test_advanced_patterns.py -v     # 25 advanced pattern tests
pytest tests/test_market_context.py -v        # 28 market context tests
```

**Current Test Status:** ✅ 78/78 tests passing (100% pass rate)

## 🗺️ Development Roadmap

### ✅ Phase 1: Core Pattern Detection (Complete)
- [x] 1.1: Project setup & environment
- [x] 1.2: Core data structures (Pattern, MarketData, Signal)
- [x] 1.3: Technical indicator engine (23 indicators)
- [x] 1.4: Pattern detection algorithms (5 patterns)
- [x] 1.5: Pattern validation framework
- [x] 1.6: Testing framework setup
- [x] 1.7: Visualization & output systems

### ✅ Phase 2: Multi-Timeframe Analysis (In Progress - 3/7 Complete)
- [x] 2.1: Multi-timeframe analysis system (9 timeframes, confluence scoring)
- [x] 2.2: Advanced pattern detection (10 new patterns: Flags, Pennants, Doubles, Channels)
- [x] 2.3: Market context analysis (4 volatility regimes, 4 trends, 5 market regimes, adaptive parameters)
- [ ] 2.4: Enhanced pattern strength scoring (integrate context with validation)
- [ ] 2.5: Memory server integration for learning
- [ ] 2.6: Enhanced signal generation
- [ ] 2.7: Backtesting framework

### 📋 Phase 3: Advanced Analytics & Learning
- [ ] Machine learning pattern enhancement
- [ ] Advanced technical indicators (Ichimoku, Fibonacci, etc.)
- [ ] Pattern learning and adaptation system
- [ ] Advanced risk management

### 📋 Phase 4: Production Deployment
- [ ] Real-time processing engine
- [ ] Market Data Agent integration
- [ ] Alert and notification system
- [ ] Production infrastructure & monitoring

**📋 Full Roadmap:** See [docs/TODO.md](docs/TODO.md) for complete task breakdown

## 📚 Documentation

- **[TODO.md](docs/TODO.md)** - Complete roadmap and task tracking
- **[Implementation Phases](docs/architecture/IMPLEMENTATION_PHASES.md)** - Detailed phase breakdown
- **[Pattern Agent Prompt](docs/architecture/pattern-agent-prompt.md)** - Original design document
- **[Phase Reports](docs/reports/)** - Completion reports for each phase

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-pattern`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions and support, please open an issue on GitHub.

---

**Built with:** Python 3.13.7 | pandas | numpy | matplotlib | seaborn | pydantic
**Framework:** SPAR (Sense, Plan, Act, Reflect) Level 3-4 Agentic AI
