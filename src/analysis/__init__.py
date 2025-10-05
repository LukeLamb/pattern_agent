"""
Multi-Timeframe Analysis Module

Provides cross-timeframe pattern validation and signal synthesis.
"""

from .multi_timeframe import (
    MultiTimeframeAnalyzer,
    TimeframeHierarchy,
    TimeframeAlignment,
    ConfluenceScore,
    MultiTimeframePattern,
)

__all__ = [
    "MultiTimeframeAnalyzer",
    "TimeframeHierarchy",
    "TimeframeAlignment",
    "ConfluenceScore",
    "MultiTimeframePattern",
]
