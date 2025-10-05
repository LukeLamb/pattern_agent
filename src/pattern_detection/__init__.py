"""
Pattern Detection Module

Core pattern recognition algorithms and engines.
"""

from .pattern_engine import PatternDetectionEngine, DetectedPattern, PatternType, PatternStrength
from .triangle_patterns import TrianglePatternDetector
from .head_shoulders import HeadShouldersDetector
from .flag_pennant import FlagPennantDetector
from .double_patterns import DoublePatternDetector
from .channels import ChannelDetector

__all__ = [
    'PatternDetectionEngine',
    'DetectedPattern',
    'PatternType',
    'PatternStrength',
    'TrianglePatternDetector',
    'HeadShouldersDetector',
    'FlagPennantDetector',
    'DoublePatternDetector',
    'ChannelDetector',
]
