import matplotlib
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QGroupBox, QListWidget, QListWidgetItem,
                             QPushButton, QMessageBox, QFrame, QToolTip)
from PyQt6.QtCore import Qt, QEvent, QPoint
from PyQt6.QtGui import QCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from backend.trading_logic_new import get_portfolio_history, get_orders
from frontend.logging_config import get_logger
from frontend.gui.styles import ModernStyles

# Configure logging
logger = get_logger(__name__)

class CustomFigureCanvas(FigureCanvas):
    """Custom canvas to handle hover tooltips for plots."""
    def __init__(self, figure, parent=None, get_tooltip_text=None):
        super().__init__(figure)
        self.setParent(parent)
        self.get_tooltip_text = get_tooltip_text  # Callback to get tooltip text
        self.setMouseTracking(True)  # Enable mouse tracking for hover events

    def enterEvent(self, event):
        """Show tooltip at the current mouse cursor position."""
        if self.get_tooltip_text:
            tooltip_text = self.get_tooltip_text()
            if tooltip_text:
                pos = QCursor.pos() + QPoint(10, 10)  # Offset from cursor
                QToolTip.showText(pos, tooltip_text, self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide tooltip when mouse leaves the canvas."""
        QToolTip.hideText()
        super().leaveEvent(event)

class CustomComboBox(QComboBox):
    """Custom QComboBox to show tooltip with plot description on hover."""
    def __init__(self, parent=None, get_tooltip_text=None):
        super().__init__(parent)
        self.get_tooltip_text = get_tooltip_text  # Callback to get tooltip text
        self.setMouseTracking(True)  # Enable mouse tracking for hover events

    def enterEvent(self, event):
        """Show tooltip at the current mouse cursor position."""
        if self.get_tooltip_text:
            tooltip_text = self.get_tooltip_text()
            if tooltip_text:
                pos = QCursor.pos() + QPoint(10, 10)  # Offset from cursor
                QToolTip.showText(pos, tooltip_text, self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide tooltip when mouse leaves the combo box."""
        QToolTip.hideText()
        super().leaveEvent(event)

class AnalysisDashboard(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.is_dark_mode = True
        self.ticker_colors = {}  # Cache ticker colors for consistency
        # Define plot descriptions for tooltips
        self.plot_descriptions = {
            "Portfolio Performance": "Shows the portfolio value over time compared to the S&P 500 benchmark.",
            "Sharpe Ratio Box Plot": "Displays the distribution of predicted and actual Sharpe ratios for selected tickers.",
            "Sharpe Prediction Error": "Plots the difference between actual and predicted Sharpe ratios over time.",
            "Portfolio Drawdown": "Illustrates the portfolio's percentage decline from its peak value, with -10% and -20% thresholds.",
            "Cumulative Returns by Ticker": "Shows normalized cumulative returns for selected tickers compared to the S&P 500.",
            "Profit/Loss by Ticker": "Bar plot of profit or loss for each ticker based on trade history."
        }
        # Select available font
        available_fonts = set(f.lower() for f in matplotlib.font_manager.findSystemFonts())
        font_priority = ['Arial', 'DejaVu Sans', 'sans-serif']
        font_family = [f for f in font_priority if f.lower() in available_fonts or f == 'sans-serif']
        if not font_family:
            font_family = ['sans-serif']
        
        # Set initial matplotlib style
        self._configure_matplotlib_style()
        
        logger.debug(f"Using font family: {font_family}")
        self.setup_ui()
        logger.info("AnalysisDashboard initialized")

    def _configure_matplotlib_style(self):
        """Configure matplotlib styling based on current theme."""
        if self.is_dark_mode:
            plt.style.use('dark_background')
            plt.rcParams.update({
                'font.family': ['Arial', 'DejaVu Sans', 'sans-serif'],
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 8,
                'axes.facecolor': '#2b2b2b',
                'figure.facecolor': '#212121',
                'axes.labelcolor': '#ffffff',
                'xtick.color': '#ffffff',
                'ytick.color': '#ffffff',
                'text.color': '#ffffff',
                'axes.edgecolor': '#ffffff',
                'grid.color': '#444444'
            })
        else:
            plt.style.use('default')
            plt.rcParams.update({
                'font.family': ['Arial', 'DejaVu Sans', 'sans-serif'],
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 8,
                'axes.facecolor': '#ffffff',
                'figure.facecolor': '#f5f5f5',
                'axes.labelcolor': '#333333',
                'xtick.color': '#333333',
                'ytick.color': '#333333',
                'text.color': '#333333',
                'axes.edgecolor': '#333333',
                'grid.color': '#cccccc'
            })

    def setup_ui(self):
        """Set up the UI components with a sidebar for tickers and a top row for graph type."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(24)

        # Left sidebar for ticker selection
        ticker_container = QFrame()
        ticker_layout = QVBoxLayout(ticker_container)
        ticker_layout.setContentsMargins(0, 0, 0, 0)
        ticker_layout.setSpacing(8)
        
        ticker_label = QLabel("Tickers")
        ticker_label.setProperty("class", "dropdown-label")
        ticker_layout.addWidget(ticker_label)
        
        self.ticker_list = QListWidget()
        self.ticker_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.ticker_list.setMaximumWidth(200)  # Fixed width for sidebar
        self.ticker_list.setProperty("class", "dropdown-input")
        if self.data_manager.data is not None and not self.data_manager.data.empty:
            tickers = sorted(self.data_manager.data['Ticker'].unique())
            for ticker in tickers:
                item = QListWidgetItem(ticker)
                self.ticker_list.addItem(item)
        else:
            logger.warning("No tickers available due to missing market data")
        self.ticker_list.itemSelectionChanged.connect(self.update_selected_tickers)
        ticker_layout.addWidget(self.ticker_list)

        self.selected_tickers_layout = QHBoxLayout()
        self.selected_tickers_layout.setSpacing(5)
        self.selected_tickers_label = QLabel("Selected:")
        self.selected_tickers_layout.addWidget(self.selected_tickers_label)
        self.selected_tickers_buttons = {}
        self.selected_tickers_layout.addStretch()

        self.clear_tickers_button = QPushButton("Clear All")
        self.clear_tickers_button.clicked.connect(self.clear_all_tickers)
        self.clear_tickers_button.setProperty("class", "secondary")
        self.clear_tickers_button.setMaximumHeight(30)
        self.selected_tickers_layout.addWidget(self.clear_tickers_button)

        ticker_layout.addLayout(self.selected_tickers_layout)
        ticker_layout.addStretch()  # Push content to top of sidebar
        main_layout.addWidget(ticker_container, stretch=0)

        # Right side: graph type and plot canvas
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)

        # Graph type row
        graph_container = QFrame()
        graph_layout = QHBoxLayout(graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(12)
        
        graph_label = QLabel("Graph Type:")
        graph_label.setProperty("class", "dropdown-label")
        graph_layout.addWidget(graph_label)
        
        self.graph_combo = CustomComboBox(get_tooltip_text=self.get_current_plot_description)
        self.graph_combo.addItems([
            "Portfolio Performance",
            "Sharpe Ratio Box Plot",
            "Sharpe Prediction Error",
            "Portfolio Drawdown",
            "Cumulative Returns by Ticker",
            "Profit/Loss by Ticker"
        ])
        self.graph_combo.currentIndexChanged.connect(self.change_graph_type)
        self.graph_combo.setProperty("class", "dropdown-input")
        graph_layout.addWidget(self.graph_combo)
        graph_layout.addStretch()  # Align to left
        right_layout.addWidget(graph_container)

        # Chart area
        self.chart_fig = Figure(figsize=(10, 6))
        self.chart_canvas = CustomFigureCanvas(self.chart_fig, parent=self, get_tooltip_text=self.get_current_plot_description)
        colors = ModernStyles.COLORS['dark' if self.is_dark_mode else 'light']
        self.chart_canvas.setStyleSheet(f"""
            background-color: {colors['surface']};
            border: none;
        """)
        right_layout.addWidget(self.chart_canvas, stretch=2)

        main_layout.addLayout(right_layout, stretch=1)
        self.set_theme(self.is_dark_mode)
        self.update_visualizations()

    def get_current_plot_description(self):
        """Return the description of the currently selected plot."""
        return self.plot_descriptions.get(self.graph_combo.currentText(), "No description available.")

    def set_initial_capital(self, initial_capital):
        """Set the initial capital for the portfolio."""
        self.initial_capital = initial_capital
        self.update_visualizations()
        logger.debug(f"Set initial capital: ${initial_capital:,.2f}")

    def change_graph_type(self, index):
        """Handle graph type change from combo box."""
        self.update_visualizations()
        logger.debug(f"Changed graph type to: {self.graph_combo.currentText()}")

    def set_theme(self, is_dark_mode):
        """Apply the specified theme to the dashboard."""
        self.is_dark_mode = is_dark_mode
        
        # Configure matplotlib styling
        self._configure_matplotlib_style()
        
        # Update figure colors
        self._update_figure_colors()

        # Apply modern styling
        style = ModernStyles.get_complete_style(self.is_dark_mode)
        colors = ModernStyles.COLORS['dark' if self.is_dark_mode else 'light']
        
        additional_styles = f"""
            QLabel[class="dropdown-label"] {{
                color: {colors['text_primary']};
                font-size: 14px;
                font-weight: 600;
                font-family: 'Segoe UI';
                background-color: transparent;
            }}
            QComboBox[class="dropdown-input"] {{
                border: 2px solid {colors['border_light']};
                border-radius: 6px;
                background-color: {'#3A3A54' if self.is_dark_mode else '#F8FAFC'};
                padding: 8px 12px;
                font-size: 14px;
                color: {colors['text_primary']};
                margin: 2px;
            }}
            QComboBox[class="dropdown-input"]:focus {{
                border: 2px solid {colors['accent']};
                background-color: {'#404060' if self.is_dark_mode else '#FFFFFF'};
            }}
            QComboBox[class="dropdown-input"]:hover {{
                border: 2px solid {colors['accent_hover']};
            }}
            QComboBox[class="dropdown-input"]::drop-down {{
                border: none;
                border-left: 1px solid {colors['border']};
                border-radius: 0px 4px 4px 0px;
                background-color: {colors['secondary']};
                width: 20px;
            }}
            QComboBox[class="dropdown-input"]::drop-down:hover {{
                background-color: {colors['accent']};
            }}
            QComboBox[class="dropdown-input"]::down-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {colors['text_primary']};
                width: 0px;
                height: 0px;
            }}
            QComboBox[class="dropdown-input"] QAbstractItemView {{
                background-color: {colors['surface']};
                color: {colors['text_primary']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                selection-background-color: {colors['accent']};
                outline: none;
            }}
            QListWidget[class="dropdown-input"] {{
                background-color: {'#3A3A54' if self.is_dark_mode else '#F8FAFC'};
                color: {colors['text_primary']};
                border: 2px solid {colors['border_light']};
                border-radius: 6px;
                padding: 4px;
                font-size: 13px;
                margin: 2px;
            }}
            QListWidget::item {{
                border-radius: 4px;
                padding: 6px 10px;
                margin: 1px;
            }}
            QListWidget::item:selected {{
                background-color: {colors['accent']};
                color: white;
            }}
            QListWidget::item:hover {{
                background-color: {colors['hover']};
            }}
            QPushButton[class="secondary"] {{
                background-color: {colors['secondary']};
                color: {colors['text_primary']};
                border: 2px solid {colors['border']};
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton[class="secondary"]:hover {{
                background-color: {colors['hover']};
                border: 2px solid {colors['accent']};
            }}
        """
        complete_style = style + additional_styles
        self.setStyleSheet(complete_style)

        # Update canvas stylesheet
        self.chart_canvas.setStyleSheet(f"""
            background-color: {colors['surface']};
            border: none;
        """)

        # Update ticker button colors
        self._update_ticker_button_colors()
        
        self.selected_tickers_label.setStyleSheet(f"color: {colors['text_primary']}; font-size: 12px; font-weight: 500;")
        
        self.update_visualizations()
        logger.debug(f"Applied theme: {'dark' if is_dark_mode else 'light'}")

    def _update_ticker_button_colors(self):
        """Update ticker button colors to match plot colors."""
        selected_tickers = list(self.selected_tickers_buttons.keys())
        if not selected_tickers:
            self.ticker_colors = {}
            return
            
        self.ticker_colors = self._get_ticker_colors(selected_tickers)
        
        for ticker, button in self.selected_tickers_buttons.items():
            if ticker in self.ticker_colors:
                color_info = self.ticker_colors[ticker]
                button_style = f"""
                    QPushButton {{
                        background-color: {color_info['hex']}; 
                        color: white; 
                        border: none; 
                        padding: 4px 8px; 
                        border-radius: 4px; 
                        font-size: 11px;
                        font-weight: bold;
                    }} 
                    QPushButton:hover {{
                        background-color: {color_info['hover']};
                    }}
                """
                button.setStyleSheet(button_style)

    def _update_figure_colors(self):
        """Update figure and canvas colors to match the theme."""
        colors = ModernStyles.COLORS['dark' if self.is_dark_mode else 'light']
        self.chart_fig.patch.set_facecolor(colors['surface'])
        self.chart_canvas.setStyleSheet(f"""
            background-color: {colors['surface']};
            border: none;
        """)

    def _get_theme_colors(self):
        """Get appropriate colors for current theme."""
        colors = ModernStyles.COLORS['dark' if self.is_dark_mode else 'light']
        return {
            'text': colors['text_primary'],
            'bg': colors['surface'],
            'surface': colors['surface'],
            'grid': '#444444' if self.is_dark_mode else '#cccccc',
            'legend_bg': '#2b2b2b' if self.is_dark_mode else '#ffffff'
        }

    def _get_ticker_colors(self, tickers):
        """Get consistent colors for tickers that match the plot colors."""
        if not tickers:
            return {}
        
        colors_map = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
        ticker_colors = {}
        
        for idx, ticker in enumerate(tickers):
            plot_color = colors_map[idx]
            hex_color = plt.cm.colors.rgb2hex(plot_color[:3])  # Use normalized RGB for hex
            hover_color = self.adjust_color_brightness(hex_color, 0.8)  # Adjusted hover color
            ticker_colors[ticker] = {
                'hex': hex_color,
                'rgb': plot_color[:3],  # Normalized RGB values (0-1)
                'hover': hover_color
            }
        
        return ticker_colors

    def update_selected_tickers(self):
        """Update the display of selected tickers with consistent colors."""
        selected_items = self.ticker_list.selectedItems()
        selected_tickers = [item.text() for item in selected_items]

        if len(selected_tickers) > 5:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Selection Limit")
            msg.setText("Please select up to 5 tickers only.")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec()
            for item in selected_items[5:]:
                item.setSelected(False)
            selected_tickers = selected_tickers[:5]

        for ticker in list(self.selected_tickers_buttons.keys()):
            if ticker not in selected_tickers:
                button = self.selected_tickers_buttons.pop(ticker)
                button.deleteLater()

        # Update cached ticker colors
        self.ticker_colors = self._get_ticker_colors(selected_tickers)

        for ticker in selected_tickers:
            if ticker not in self.selected_tickers_buttons:
                # Use cached color from self.ticker_colors
                button_color = self.ticker_colors.get(ticker, {}).get('hex', '#3366CC')
                hover_color = self.ticker_colors.get(ticker, {}).get('hover', self.adjust_color_brightness('#3366CC', 0.8))
                
                button = QPushButton(ticker)
                button.setFixedHeight(25)
                button.clicked.connect(lambda _, t=ticker: self.remove_ticker(t))
                
                button_style = f"""
                    QPushButton {{
                        background-color: {button_color}; 
                        color: white; 
                        border: none; 
                        padding: 4px 8px; 
                        border-radius: 4px; 
                        font-size: 11px;
                        font-weight: bold;
                    }} 
                    QPushButton:hover {{
                        background-color: {hover_color};
                    }}
                """
                button.setStyleSheet(button_style)
                self.selected_tickers_layout.insertWidget(self.selected_tickers_layout.count()-2, button)
                self.selected_tickers_buttons[ticker] = button

        self.update_visualizations()
        logger.debug(f"Selected tickers: {selected_tickers}")

    def adjust_color_brightness(self, hex_color, factor):
        """Adjust the brightness of a hex color for hover effect."""
        from matplotlib.colors import hex2color, rgb2hex
        rgb = hex2color(hex_color)
        adjusted_rgb = tuple(min(max(c * factor, 0), 1) for c in rgb[:3])
        return rgb2hex(adjusted_rgb)

    def remove_ticker(self, ticker):
        """Remove a ticker from the selection."""
        for item in self.ticker_list.findItems(ticker, Qt.MatchFlag.MatchExactly):
            item.setSelected(False)
        self.update_selected_tickers()

    def clear_all_tickers(self):
        """Clear all selected tickers."""
        self.ticker_list.clearSelection()
        self.update_selected_tickers()
        logger.debug("Cleared all tickers")

    def get_message_box_style(self):
        """Get stylesheet for message boxes."""
        colors = ModernStyles.COLORS['dark' if self.is_dark_mode else 'light']
        return (
            f"QMessageBox {{ background-color: {colors['surface']}; color: {colors['text_primary']}; }}"
            f"QMessageBox QLabel {{ color: {colors['text_primary']}; }}"
            f"QPushButton {{ background-color: {colors['accent']}; color: white; border: none; padding: 5px; border-radius: 3px; }}"
            f"QPushButton:hover {{ background-color: {colors['accent_hover']}; }}"
        )

    def update_visualizations(self):
        """Update the visualization based on the selected graph type."""
        graph_type = self.graph_combo.currentText()
        selected_tickers = [item.text() for item in self.ticker_list.selectedItems()]

        self.chart_fig.clear()
        self._update_figure_colors()

        try:
            start_date = pd.to_datetime(self.data_manager.start_date, utc=True) if self.data_manager.start_date else None
            end_date = pd.to_datetime(self.data_manager.end_date, utc=True) if self.data_manager.end_date else None
            if (start_date is None or end_date is None) and self.data_manager.data is not None and not self.data_manager.data.empty:
                dates = pd.to_datetime(self.data_manager.data['date'], utc=True)
                start_date = start_date or dates.min()
                end_date = end_date or dates.max()
            if start_date is None:
                start_date = pd.Timestamp('2000-01-01', tz='UTC')
            if end_date is None:
                end_date = pd.Timestamp.now(tz='UTC')
            self.start_date = start_date
            self.end_date = end_date
            logger.debug(f"Using date range: {self.start_date} to {self.end_date}")
        except Exception as e:
            logger.error(f"Error setting dates: {e}")
            theme_colors = self._get_theme_colors()
            self.chart_fig.add_subplot(111).text(0.5, 0.5, 'Error: Invalid date range',
                                                 horizontalalignment='center', verticalalignment='center',
                                                 color=theme_colors['text'])
            self.chart_canvas.draw()
            return

        if graph_type == "Portfolio Performance":
            self.plot_portfolio_performance()
        elif graph_type == "Sharpe Ratio Box Plot":
            self.plot_sharpe_box_plot(selected_tickers)
        elif graph_type == "Sharpe Prediction Error":
            self.plot_sharpe_prediction_error(selected_tickers)
        elif graph_type == "Portfolio Drawdown":
            self.plot_portfolio_drawdown()
        elif graph_type == "Cumulative Returns by Ticker":
            self.plot_cumulative_returns(selected_tickers)
        elif graph_type == "Profit/Loss by Ticker":
            self.plot_profit_loss_by_ticker()

        self.chart_canvas.draw()

    def update_dashboard(self):
        """Refresh the dashboard visualizations."""
        self.update_visualizations()
        logger.info("Dashboard updated")

    def plot_portfolio_performance(self):
        """Plot portfolio value over time with S&P 500 benchmark."""
        ax = self.chart_fig.add_subplot(111)
        theme_colors = self._get_theme_colors()
        
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        if df.empty:
            ax.text(0.5, 0.5, 'No portfolio history in date range', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        ax.plot(df['date'], df['value'], label='Portfolio Value', color='#2196F3', linewidth=2)
        # Add S&P 500 benchmark if available
        if self.data_manager.data is not None and 'Ticker' in self.data_manager.data and 'SPY' in self.data_manager.data['Ticker'].values:
            sp500_data = self.data_manager.data[self.data_manager.data['Ticker'] == 'SPY'].copy()
            sp500_data['date'] = pd.to_datetime(sp500_data['date'], utc=True)
            sp500_data = sp500_data[(sp500_data['date'] >= self.start_date) & (sp500_data['date'] <= self.end_date)]
            if not sp500_data.empty:
                sp500_data['value'] = sp500_data['Close'] / sp500_data['Close'].iloc[0] * df['value'].iloc[0]
                ax.plot(sp500_data['date'], sp500_data['value'], label='S&P 500', color='#F59E0B', linestyle='--', linewidth=1.5)

        ax.set_title('Portfolio Performance', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Value ($)', color=theme_colors['text'])
        ax.legend(loc='best', frameon=True, facecolor=theme_colors['legend_bg'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_sharpe_box_plot(self, selected_tickers):
        """Plot box plot of predicted and actual Sharpe ratios."""
        ax = self.chart_fig.add_subplot(111)
        theme_colors = self._get_theme_colors()
        
        data = self.data_manager.data.copy() if self.data_manager.data is not None else None
        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No market data available', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Select at least one ticker', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for selected tickers', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        # Use cached ticker colors
        colors = [self.ticker_colors[ticker]['rgb'] for ticker in selected_tickers if ticker in self.ticker_colors]
        if not colors:
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        # Prepare data for box plot
        plot_data = []
        labels = []
        color_list = []
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker]
            predicted = ticker_data['Best_Prediction'].dropna()
            actual = ticker_data[ticker_data['Actual_Sharpe'] != -1.0]['Actual_Sharpe'].dropna()
            if not predicted.empty:
                plot_data.append(predicted)
                labels.append(f'{ticker} Pred.')
                color_list.append(colors[idx % len(colors)])
            if not actual.empty:
                plot_data.append(actual)
                labels.append(f'{ticker} Actual')
                color_list.append(colors[idx % len(colors)])

        if not plot_data:
            ax.text(0.5, 0.5, 'No valid Sharpe data available', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        # Create box plot
        box = ax.boxplot(plot_data, patch_artist=True, vert=True, labels=labels)
        for patch, color in zip(box['boxes'], color_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in box['medians']:
            median.set_color(theme_colors['text'])
            median.set_linewidth(2)

        ax.set_title('Sharpe Ratio Box Plot', pad=10, color=theme_colors['text'])
        ax.set_ylabel('Sharpe Ratio', color=theme_colors['text'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_sharpe_prediction_error(self, selected_tickers):
        """Plot prediction error (Actual - Predicted Sharpe) over time."""
        ax = self.chart_fig.add_subplot(111)
        theme_colors = self._get_theme_colors()
        
        data = self.data_manager.data.copy() if self.data_manager.data is not None else None
        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No market data available', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Select at least one ticker', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for selected tickers', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        # Use cached ticker colors
        colors = [self.ticker_colors[ticker]['rgb'] for ticker in selected_tickers if ticker in self.ticker_colors]
        if not colors:
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data = ticker_data[ticker_data['Actual_Sharpe'] != -1.0]
            if not ticker_data.empty:
                ticker_data['error'] = ticker_data['Actual_Sharpe'] - ticker_data['Best_Prediction']
                ax.plot(ticker_data['date'], ticker_data['error'],
                        label=f'{ticker} Error', color=colors[idx], linewidth=1.5)

        ax.axhline(0, color='#F44336', linestyle='--', label='Zero Error')
        ax.set_title('Sharpe Ratio Prediction Error', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Actual - Predicted Sharpe', color=theme_colors['text'])
        ax.legend(loc='best', frameon=True, facecolor=theme_colors['legend_bg'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_portfolio_drawdown(self):
        """Plot portfolio drawdown with threshold lines."""
        ax = self.chart_fig.add_subplot(111)
        theme_colors = self._get_theme_colors()
        
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        if df.empty:
            ax.text(0.5, 0.5, 'No portfolio history in date range', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        df['peak'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['peak']) / df['peak'] * 100

        ax.plot(df['date'], df['drawdown'], label='Drawdown (%)', color='#F44336', linewidth=2)
        ax.fill_between(df['date'], df['drawdown'], 0, color='#F44336', alpha=0.2)
        ax.axhline(-10, color='#F59E0B', linestyle='--', label='-10% Threshold', alpha=0.5)
        ax.axhline(-20, color='#EF4444', linestyle='--', label='-20% Threshold', alpha=0.5)
        ax.set_title('Portfolio Drawdown', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Drawdown (%)', color=theme_colors['text'])
        ax.legend(loc='best', frameon=True, facecolor=theme_colors['legend_bg'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_cumulative_returns(self, selected_tickers):
        """Plot normalized cumulative returns with S&P 500 benchmark."""
        ax = self.chart_fig.add_subplot(111)
        theme_colors = self._get_theme_colors()
        
        data = self.data_manager.data.copy() if self.data_manager.data is not None else None
        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No market data available', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Select at least one ticker', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for selected tickers', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        # Use cached ticker colors
        colors = [self.ticker_colors[ticker]['rgb'] for ticker in selected_tickers if ticker in self.ticker_colors]
        if not colors:
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker].sort_values('date')
            if not ticker_data.empty:
                returns = ticker_data['Close'].pct_change().fillna(0)
                cumulative = (1 + returns).cumprod() * 100
                ax.plot(ticker_data['date'], cumulative, label=f'{ticker} Returns',
                        color=colors[idx], linewidth=1.5)

        # Add S&P 500 benchmark
        if 'SPY' in self.data_manager.data['Ticker'].values:
            sp500_data = self.data_manager.data[self.data_manager.data['Ticker'] == 'SPY'].copy()
            sp500_data['date'] = pd.to_datetime(sp500_data['date'], utc=True)
            sp500_data = sp500_data[(sp500_data['date'] >= self.start_date) & (sp500_data['date'] <= self.end_date)]
            if not sp500_data.empty:
                returns = sp500_data['Close'].pct_change().fillna(0)
                cumulative = (1 + returns).cumprod() * 100
                ax.plot(sp500_data['date'], cumulative, label='S&P 500 Returns', color='#F59E0B', linestyle='--', linewidth=1.5)

        ax.set_title('Cumulative Returns by Ticker (Normalized)', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Cumulative Return (%)', color=theme_colors['text'])
        ax.legend(loc='best', frameon=True, facecolor=theme_colors['legend_bg'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_profit_loss_by_ticker(self):
        """Plot profit/loss by ticker as a bar plot."""
        ax = self.chart_fig.add_subplot(111)
        theme_colors = self._get_theme_colors()
        
        orders = get_orders()
        if not orders:
            ax.text(0.5, 0.5, 'No trade history available', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        orders_df = pd.DataFrame(orders)
        orders_df['date'] = pd.to_datetime(orders_df['date'], utc=True)
        orders_df = orders_df[(orders_df['date'] >= self.start_date) & (orders_df['date'] <= self.end_date)]

        if orders_df.empty:
            ax.text(0.5, 0.5, 'No trades in date range', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        # Calculate profit/loss by ticker
        profits = orders_df.groupby('ticker').apply(
            lambda x: (
                -x[x['action'] == 'buy']['investment_amount'].sum() +
                x[x['action'] == 'sell']['investment_amount'].sum() -
                x['transaction_cost'].sum()
            )
        ).reset_index(name='profit_loss')

        # Use cached ticker colors
        selected_tickers = profits['ticker'].tolist()
        colors = [self.ticker_colors[ticker]['rgb'] for ticker in selected_tickers if ticker in self.ticker_colors]
        if not colors:
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        # Convert colors to list to avoid UserWarning
        sns.barplot(data=profits, x='ticker', y='profit_loss', hue='ticker', legend=False, ax=ax, palette=colors.tolist())
        ax.axhline(0, color='#F44336', linestyle='--', label='Break-even')
        ax.set_title('Profit/Loss by Ticker', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Ticker', color=theme_colors['text'])
        ax.set_ylabel('Profit/Loss ($)', color=theme_colors['text'])
        ax.legend(loc='best', frameon=True, facecolor=theme_colors['legend_bg'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()