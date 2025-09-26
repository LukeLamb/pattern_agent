"""
Basic Pattern Visualization - Create charts with detected patterns.

This module provides basic pattern visualization capabilities including:
- Price chart plotting with OHLCV data
- Pattern overlay and annotation
- Support/resistance level highlighting
- Chart export functionality
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
from datetime import datetime
import warnings

try:
    from ..models.market_data import MarketData
    from ..models.pattern import DetectedPattern, SupportResistanceLevel
    from ..pattern_detection.pattern_engine import PatternDetectionEngine
    from ..technical_indicators.indicator_engine import TechnicalIndicatorEngine
except ImportError:
    # For testing and development
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.market_data import MarketData
    from models.pattern import DetectedPattern, SupportResistanceLevel
    from pattern_detection.pattern_engine import PatternDetectionEngine
    from technical_indicators.indicator_engine import TechnicalIndicatorEngine


class PatternPlotter:
    """
    Basic pattern visualization and chart plotting.
    
    Provides functionality to create charts showing:
    - OHLCV candlestick data
    - Detected patterns with annotations
    - Support and resistance levels
    - Technical indicators overlay
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 8), style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the PatternPlotter.
        
        Args:
            figsize: Figure size in inches (width, height)
            style: Matplotlib style to use
        """
        self.figsize = figsize
        
        # Handle matplotlib style availability
        available_styles = plt.style.available
        if style in available_styles:
            plt.style.use(style)
        elif 'seaborn-v0_8' in str(available_styles):
            # Try alternative seaborn style
            seaborn_styles = [s for s in available_styles if 'seaborn' in s and 'darkgrid' in s]
            if seaborn_styles:
                plt.style.use(seaborn_styles[0])
        else:
            # Fallback to default
            warnings.warn(f"Style '{style}' not available, using default")
        
        # Color scheme for patterns
        self.pattern_colors = {
            'ascending_triangle': '#2E8B57',    # Sea Green
            'descending_triangle': '#DC143C',   # Crimson
            'symmetrical_triangle': '#4169E1',  # Royal Blue  
            'head_shoulders': '#FF6347',        # Tomato
            'inverse_head_shoulders': '#32CD32', # Lime Green
            'support': '#228B22',               # Forest Green
            'resistance': '#B22222'             # Fire Brick
        }
        
        # Default chart settings
        self.chart_config = {
            'candlestick_up_color': '#26A69A',   # Teal
            'candlestick_down_color': '#EF5350', # Red
            'volume_color': '#78909C',           # Blue Grey
            'grid_alpha': 0.3,
            'pattern_alpha': 0.7,
            'line_width': 1.5,
            'annotation_fontsize': 10
        }

    def plot_pattern_chart(
        self, 
        market_data: MarketData, 
        patterns: Optional[List[DetectedPattern]] = None,
        indicators: Optional[Dict] = None,
        support_resistance: Optional[List[SupportResistanceLevel]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create a comprehensive pattern chart.
        
        Args:
            market_data: MarketData object with OHLCV data
            patterns: List of detected patterns to overlay
            indicators: Technical indicators to display
            support_resistance: Support/resistance levels to show
            title: Chart title
            save_path: Path to save the chart image
            show: Whether to display the chart
            
        Returns:
            matplotlib Figure object
        """
        df = market_data.to_dataframe(set_timestamp_index=True)
        
        # Create figure and subplots
        fig = plt.figure(figsize=self.figsize)
        
        # Main price chart (70% of height)
        ax_price = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        
        # Volume chart (30% of height)  
        ax_volume = plt.subplot2grid((4, 1), (3, 0), sharex=ax_price)
        
        # Plot candlestick chart
        self._plot_candlesticks(ax_price, df)
        
        # Plot volume
        self._plot_volume(ax_volume, df)
        
        # Plot technical indicators
        if indicators:
            self._plot_indicators(ax_price, df, indicators)
        
        # Plot support/resistance levels
        if support_resistance:
            self._plot_support_resistance(ax_price, df, support_resistance)
        
        # Plot detected patterns
        if patterns:
            self._plot_patterns(ax_price, df, patterns)
        
        # Configure chart appearance
        self._configure_chart(ax_price, ax_volume, market_data, title)
        
        # Save chart if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show chart if requested
        if show:
            plt.show()
        
        return fig

    def _plot_candlesticks(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot candlestick chart."""
        dates = df.index
        
        for i, (date, row) in enumerate(df.iterrows()):
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Determine color
            color = (self.chart_config['candlestick_up_color'] 
                    if close_price >= open_price 
                    else self.chart_config['candlestick_down_color'])
            
            # Draw high-low line
            ax.plot([date, date], [low_price, high_price], 
                   color='black', linewidth=0.8, alpha=0.8)
            
            # Draw open-close rectangle
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            
            rect = Rectangle((mdates.date2num(date) - 0.3, bottom), 
                           0.6, height, 
                           facecolor=color, edgecolor='black', 
                           linewidth=0.5, alpha=0.8)
            ax.add_patch(rect)

    def _plot_volume(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot volume bars."""
        dates = df.index
        volumes = df['volume']
        
        # Create volume bars
        bars = ax.bar(dates, volumes, width=0.8, 
                     color=self.chart_config['volume_color'], alpha=0.6)
        
        ax.set_ylabel('Volume', fontsize=10)
        ax.grid(True, alpha=self.chart_config['grid_alpha'])

    def _plot_indicators(self, ax: plt.Axes, df: pd.DataFrame, indicators: Dict) -> None:
        """Plot technical indicators overlay."""
        dates = df.index
        
        # Plot trend indicators
        if 'trend' in indicators:
            trend = indicators['trend']
            
            # Simple Moving Averages
            if 'sma' in trend:
                for period, values in trend['sma'].items():
                    if isinstance(values, pd.Series):
                        ax.plot(dates, values, label=f'SMA {period}', 
                               linewidth=self.chart_config['line_width'], alpha=0.8)
            
            # Exponential Moving Averages
            if 'ema' in trend:
                for period, values in trend['ema'].items():
                    if isinstance(values, pd.Series):
                        ax.plot(dates, values, label=f'EMA {period}', 
                               linewidth=self.chart_config['line_width'], alpha=0.8, linestyle='--')
            
            # Bollinger Bands
            if 'bollinger_bands' in trend:
                bb = trend['bollinger_bands']
                if isinstance(bb, dict) and all(k in bb for k in ['upper', 'middle', 'lower']):
                    ax.plot(dates, bb['upper'], label='BB Upper', color='gray', alpha=0.6)
                    ax.plot(dates, bb['middle'], label='BB Middle', color='blue', alpha=0.6)
                    ax.plot(dates, bb['lower'], label='BB Lower', color='gray', alpha=0.6)
                    
                    # Fill between bands
                    ax.fill_between(dates, bb['upper'], bb['lower'], alpha=0.1, color='blue')

        # Plot volatility indicators
        if 'volatility' in indicators:
            volatility = indicators['volatility']
            
            # Bollinger Bands (if not already plotted)
            if 'bollinger_bands' in volatility and 'trend' not in indicators:
                bb = volatility['bollinger_bands']
                if isinstance(bb, dict) and all(k in bb for k in ['upper', 'middle', 'lower']):
                    ax.plot(dates, bb['upper'], label='BB Upper', color='purple', alpha=0.6)
                    ax.plot(dates, bb['middle'], label='BB Middle', color='purple', alpha=0.8)
                    ax.plot(dates, bb['lower'], label='BB Lower', color='purple', alpha=0.6)

    def _plot_support_resistance(self, ax: plt.Axes, df: pd.DataFrame, 
                                levels: List[SupportResistanceLevel]) -> None:
        """Plot support and resistance levels."""
        for level in levels:
            color = (self.pattern_colors['support'] if level.level_type == 'support' 
                    else self.pattern_colors['resistance'])
            
            # Draw horizontal line across the visible data range
            ax.axhline(y=level.price, color=color, linestyle='-', 
                      linewidth=self.chart_config['line_width'], 
                      alpha=self.chart_config['pattern_alpha'],
                      label=f'{level.level_type.title()} {level.price:.2f}')

    def _plot_patterns(self, ax: plt.Axes, df: pd.DataFrame, patterns: List[DetectedPattern]) -> None:
        """Plot detected patterns with annotations."""
        for pattern in patterns:
            color = self.pattern_colors.get(pattern.pattern_type, '#808080')
            
            # Plot pattern-specific elements
            if 'triangle' in pattern.pattern_type:
                self._plot_triangle_pattern(ax, df, pattern, color)
            elif 'head_shoulders' in pattern.pattern_type:
                self._plot_head_shoulders_pattern(ax, df, pattern, color)
            
            # Add pattern annotation
            self._add_pattern_annotation(ax, pattern, color)

    def _plot_triangle_pattern(self, ax: plt.Axes, df: pd.DataFrame, 
                             pattern: DetectedPattern, color: str) -> None:
        """Plot triangle pattern lines."""
        # Extract pattern points if available
        if hasattr(pattern, 'key_points') and pattern.key_points:
            points = pattern.key_points
            
            # Plot trendlines if we have enough points
            if len(points) >= 4:  # At least 2 points for each line
                # Sort points by time
                sorted_points = sorted(points, key=lambda p: p['timestamp'])
                
                # Separate support and resistance points (simplified approach)
                mid_price = df['close'].median()
                support_points = [p for p in sorted_points if p['price'] <= mid_price]
                resistance_points = [p for p in sorted_points if p['price'] > mid_price]
                
                # Plot support line (lower line)
                if len(support_points) >= 2:
                    support_x = [p['timestamp'] for p in support_points]
                    support_y = [p['price'] for p in support_points]
                    ax.plot(support_x, support_y, color=color, linewidth=2, alpha=0.8)
                    ax.scatter(support_x, support_y, color=color, s=50, alpha=0.8, zorder=5)
                
                # Plot resistance line (upper line)
                if len(resistance_points) >= 2:
                    resistance_x = [p['timestamp'] for p in resistance_points]
                    resistance_y = [p['price'] for p in resistance_points]
                    ax.plot(resistance_x, resistance_y, color=color, linewidth=2, alpha=0.8)
                    ax.scatter(resistance_x, resistance_y, color=color, s=50, alpha=0.8, zorder=5)

    def _plot_head_shoulders_pattern(self, ax: plt.Axes, df: pd.DataFrame, 
                                   pattern: DetectedPattern, color: str) -> None:
        """Plot head and shoulders pattern elements."""
        # Extract pattern points if available
        if hasattr(pattern, 'key_points') and pattern.key_points:
            points = pattern.key_points
            
            if len(points) >= 5:  # Head and shoulders needs at least 5 points
                # Sort points by time
                sorted_points = sorted(points, key=lambda p: p['timestamp'])
                
                # Plot the pattern line connecting peaks and valleys
                x_coords = [p['timestamp'] for p in sorted_points]
                y_coords = [p['price'] for p in sorted_points]
                
                ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)
                ax.scatter(x_coords, y_coords, color=color, s=60, alpha=0.8, zorder=5)
                
                # Highlight the head (middle peak/valley)
                if len(sorted_points) >= 3:
                    head_point = sorted_points[len(sorted_points)//2]
                    ax.scatter([head_point['timestamp']], [head_point['price']], 
                             color='red', s=100, alpha=0.9, zorder=6, 
                             marker='*', label='Head')

    def _add_pattern_annotation(self, ax: plt.Axes, pattern: DetectedPattern, color: str) -> None:
        """Add pattern annotation with details."""
        # Determine annotation position
        if hasattr(pattern, 'key_points') and pattern.key_points:
            # Use the center of pattern for annotation
            timestamps = [p['timestamp'] for p in pattern.key_points]
            prices = [p['price'] for p in pattern.key_points]
            
            mid_time = timestamps[len(timestamps)//2]
            max_price = max(prices)
            
            # Format pattern info
            pattern_name = pattern.pattern_type.replace('_', ' ').title()
            confidence_text = f"Confidence: {pattern.confidence:.2f}"
            
            # Add annotation
            ax.annotate(f'{pattern_name}\n{confidence_text}', 
                       xy=(mid_time, max_price), 
                       xytext=(10, 10), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                       fontsize=self.chart_config['annotation_fontsize'],
                       color='white', weight='bold')

    def _configure_chart(self, ax_price: plt.Axes, ax_volume: plt.Axes, 
                        market_data: MarketData, title: Optional[str]) -> None:
        """Configure chart appearance and labels."""
        # Set title
        if title is None:
            title = f"{market_data.symbol} - {market_data.timeframe} Pattern Analysis"
        
        ax_price.set_title(title, fontsize=14, weight='bold', pad=20)
        
        # Configure price axis
        ax_price.set_ylabel('Price', fontsize=12)
        ax_price.grid(True, alpha=self.chart_config['grid_alpha'])
        ax_price.legend(loc='upper left', framealpha=0.9)
        
        # Configure volume axis
        ax_volume.set_xlabel('Date', fontsize=12)
        ax_volume.set_ylabel('Volume', fontsize=10)
        
        # Format x-axis dates
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_price.xaxis.set_major_locator(mdates.WeekdayLocator())
        
        # Rotate date labels
        plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Remove x-axis labels from price chart (shared with volume chart)
        plt.setp(ax_price.get_xticklabels(), visible=False)
        
        # Tight layout
        plt.tight_layout()

    def create_pattern_summary_chart(
        self, 
        patterns: List[DetectedPattern], 
        title: str = "Pattern Detection Summary",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create a summary chart showing pattern statistics.
        
        Args:
            patterns: List of detected patterns
            title: Chart title
            save_path: Path to save the chart
            show: Whether to display the chart
            
        Returns:
            matplotlib Figure object
        """
        if not patterns:
            print("No patterns provided for summary chart")
            return None
        
        # Analyze patterns
        pattern_counts = {}
        confidence_by_type = {}
        
        for pattern in patterns:
            pattern_type = pattern.pattern_type.replace('_', ' ').title()
            
            # Count patterns by type
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            # Track confidence scores
            if pattern_type not in confidence_by_type:
                confidence_by_type[pattern_type] = []
            confidence_by_type[pattern_type].append(pattern.confidence)
        
        # Create summary figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pattern count bar chart
        pattern_types = list(pattern_counts.keys())
        counts = list(pattern_counts.values())
        colors = [self.pattern_colors.get(pt.lower().replace(' ', '_'), '#808080') 
                 for pt in pattern_types]
        
        bars = ax1.bar(pattern_types, counts, color=colors, alpha=0.8)
        ax1.set_title('Pattern Count by Type', fontsize=12, weight='bold')
        ax1.set_ylabel('Number of Patterns')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{count}', ha='center', va='bottom')
        
        # Average confidence by pattern type
        avg_confidences = [np.mean(confidence_by_type[pt]) for pt in pattern_types]
        
        bars2 = ax2.bar(pattern_types, avg_confidences, color=colors, alpha=0.8)
        ax2.set_title('Average Confidence by Pattern Type', fontsize=12, weight='bold')
        ax2.set_ylabel('Average Confidence Score')
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add confidence labels on bars
        for bar, conf in zip(bars2, avg_confidences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{conf:.2f}', ha='center', va='bottom')
        
        # Overall title
        fig.suptitle(title, fontsize=14, weight='bold')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save chart if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show chart if requested
        if show:
            plt.show()
        
        return fig

    def export_chart(self, fig: plt.Figure, filepath: str, dpi: int = 300, 
                    format: str = 'png') -> bool:
        """
        Export chart to file.
        
        Args:
            fig: matplotlib Figure to export
            filepath: Output file path
            dpi: Resolution in dots per inch
            format: Image format ('png', 'jpg', 'pdf', 'svg')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            fig.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight')
            return True
        except Exception as e:
            print(f"Error saving chart: {e}")
            return False

    def close_all_figures(self) -> None:
        """Close all matplotlib figures to free memory."""
        plt.close('all')