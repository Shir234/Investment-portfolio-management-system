import matplotlib
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QGroupBox, QListWidget, QListWidgetItem,
                             QPushButton, QMessageBox, QFrame, QToolTip, QLayout, QGridLayout, QSizePolicy)
from PyQt6.QtCore import Qt, QEvent, QPoint, QSize, QRect
from PyQt6.QtGui import QCursor, QFontMetrics, QFont
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

class FlowLayout(QLayout):
    """A layout that arranges widgets in a flowing grid, wrapping to new rows as needed."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(2, 2, 2, 2)
        self._spacing = 5

    def addItem(self, item):
        """Add an item to the layout."""
        self._items.append(item)

    def count(self):
        """Return the number of items in the layout."""
        return len(self._items)

    def itemAt(self, index):
        """Return the item at the given index."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        """Remove and return the item at the given index."""
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        """Return the directions in which the layout can expand."""
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        """Return whether the layout has height for width."""
        return True

    def heightForWidth(self, width):
        """Return the preferred height for the given width."""
        return self._do_layout(QRect(0, 0, width, 0), False)

    def setGeometry(self, rect):
        """Set the geometry of the layout."""
        super().setGeometry(rect)
        self._do_layout(rect, True)

    def sizeHint(self):
        """Return the preferred size of the layout."""
        return self.minimumSize()

    def minimumSize(self):
        """Return the minimum size of the layout."""
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        size += QSize(2 * self.contentsMargins().left(), 2 * self.contentsMargins().top())
        return size

    def _do_layout(self, rect, apply_geometry):
        """Perform the layout calculations."""
        x = rect.x()
        y = rect.y()
        line_height = 0
        max_height = rect.height()
        for item in self._items:
            wid = item.widget()
            if wid is None:
                continue
            item_size = item.sizeHint()
            item_width = item_size.width()
            item_height = item_size.height()
            space_x = self._spacing
            space_y = self._spacing
            next_x = x + item_width + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item_width + space_x
                line_height = 0
            if y + item_height <= rect.y() + max_height:
                if apply_geometry:
                    item.setGeometry(QRect(QPoint(x, y), QSize(item_width, item_height)))
                x = next_x
                line_height = max(line_height, item_height)
        return y + line_height - rect.y()

class CustomFigureCanvas(FigureCanvas):
    """Custom canvas to handle hover tooltips for plots."""
    def __init__(self, figure, parent=None, get_tooltip_text=None):
        super().__init__(figure)
        self.setParent(parent)
        self.get_tooltip_text = get_tooltip_text
        self.setMouseTracking(True)

    def enterEvent(self, event):
        """Show tooltip at the current mouse cursor position."""
        if self.get_tooltip_text:
            tooltip_text = self.get_tooltip_text()
            if tooltip_text:
                pos = QCursor.pos() + QPoint(10, 10)
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
        self.get_tooltip_text = get_tooltip_text
        self.setMouseTracking(True)

    def enterEvent(self, event):
        """Show tooltip at the current mouse cursor position."""
        if self.get_tooltip_text:
            tooltip_text = self.get_tooltip_text()
            if tooltip_text:
                pos = QCursor.pos() + QPoint(10, 10)
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
        self.ticker_colors = {}
        self.selected_tickers_order = []
        
        # Define plot descriptions for tooltips
        self.plot_descriptions = {
            "Predicted vs Actual Sharpe": "Plots predicted Sharpe ratios as dots over and actual Sharpe ratios as solid lines time.",
            "Profit over Std Dev": "Shows profit divided by average annual standard deviation of portfolio value over rolling one-year periods.",
            "Portfolio Value Over Time": "Displays the portfolio value over time compared to the S&P 500 benchmark.",
            "Buy/Sell Distribution": "Pie chart showing the distribution of buy and sell transactions in the current portfolio.",
            "Win Rate Over Time": "Shows the percentage of profitable trades over time using a rolling window analysis."
        }
        
        # Define a distinguishable color palette
        # self.color_palette = [
        #     '#1f77b4', '#87ceeb', '#ff7f0e', '#2ca02c', '#d62728',
        #     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'
        # ]
        # Define a highly distinguishable color palette with good contrast
        self.color_palette = [
            '#E53E3E',  # Red - high contrast
            '#38A169',  # Green - distinct from red
            '#3182CE',  # Blue - but darker than before
            '#D69E2E',  # Orange/Gold - warm color
            '#805AD5'   # Purple - cool contrast
        ]
        
        # Select available font
        available_fonts = set(f.lower() for f in matplotlib.font_manager.findSystemFonts())
        font_priority = ['Arial', 'DejaVu Sans', 'sans-serif']
        font_family = [f for f in font_priority if f.lower() in available_fonts or f == 'sans-serif']
        if not font_family:
            font_family = ['sans-serif']
        
        self._configure_matplotlib_style()
        
        logger.debug(f"Using font family: {font_family}")
        self.setAutoFillBackground(True)  # Ensure widget background is rendered
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

        # Left sidebar for ticker selection and legend
        self.ticker_container = QFrame()
        self.ticker_container.setFixedWidth(200)  # Lock sidebar width
        self.ticker_container.setStyleSheet(f"background-color: {'#212121' if self.is_dark_mode else '#f5f5f5'};")
        ticker_layout = QVBoxLayout(self.ticker_container)
        ticker_layout.setContentsMargins(0, 0, 0, 0)
        ticker_layout.setSpacing(8)
        
        ticker_label = QLabel("Tickers")
        ticker_label.setProperty("class", "dropdown-label")
        ticker_layout.addWidget(ticker_label)
        
        self.ticker_list = QListWidget()
        self.ticker_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.ticker_list.setMaximumWidth(200)
        self.ticker_list.setMinimumHeight(300)  # Ensure ticker list doesn't shrink
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

        # Selected tickers header (label and clear button)
        self.selected_tickers_header = QHBoxLayout()
        self.selected_tickers_header.setSpacing(5)
        self.selected_tickers_label = QLabel("Selected:")
        self.selected_tickers_header.addWidget(self.selected_tickers_label)
        self.selected_tickers_header.addStretch()
        self.clear_tickers_button = QPushButton("Clear All")
        self.clear_tickers_button.clicked.connect(self.clear_all_tickers)
        self.clear_tickers_button.setProperty("class", "secondary")
        self.clear_tickers_button.setMaximumHeight(30)
        self.selected_tickers_header.addWidget(self.clear_tickers_button)
        ticker_layout.addLayout(self.selected_tickers_header)

        # Selected tickers area with QVBoxLayout
        self.selected_tickers_widget = QWidget()
        self.selected_tickers_widget.setProperty("class", "selected-tickers")
        self.selected_tickers_layout = QVBoxLayout(self.selected_tickers_widget)
        self.selected_tickers_layout.setContentsMargins(2, 2, 2, 2)
        self.selected_tickers_layout.setSpacing(2)
        self.selected_tickers_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Stack from top
        self.selected_tickers_buttons = {}
        ticker_layout.addWidget(self.selected_tickers_widget)

        # Legend area with QGridLayout
        self.legend_widget = QWidget()
        self.legend_widget.setFixedHeight(100)  # Lock legend area height
        self.legend_layout = QGridLayout(self.legend_widget)
        self.legend_layout.setContentsMargins(2, 2, 2, 2)
        self.legend_layout.setSpacing(2)
        self.legend_layout.setRowStretch(0, 1)
        self.legend_layout.setRowStretch(1, 1)
        self.legend_layout.setRowStretch(2, 1)
        self.legend_layout.setRowStretch(3, 1)
        self.legend_layout.setRowStretch(4, 1)
        ticker_layout.addWidget(self.legend_widget)
        
        ticker_layout.addStretch()
        main_layout.addWidget(self.ticker_container, stretch=0)

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
        graph_label.setMouseTracking(True)
        graph_label.enterEvent = lambda event: self._show_label_tooltip(event, graph_label)
        graph_label.leaveEvent = lambda event: QToolTip.hideText()
        graph_layout.addWidget(graph_label)

        self.graph_combo = CustomComboBox(get_tooltip_text=self.get_current_plot_description)
        self.graph_combo.addItems([
            "Select Graph Type",
            "Predicted vs Actual Sharpe",
            "Profit over Std Dev",
            "Portfolio Value Over Time",
            "Buy/Sell Distribution",
            "Win Rate Over Time"  # NEW: Added win rate plot
        ])
        self.graph_combo.currentIndexChanged.connect(self.change_graph_type)
        self.graph_combo.setProperty("class", "dropdown-input")
        graph_layout.addWidget(self.graph_combo)
        graph_layout.addStretch()
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
        logger.debug("UI setup completed for AnalysisDashboard")

    def _show_label_tooltip(self, event, label):
        """Show tooltip for the graph type label."""
        tooltip_text = self.get_current_plot_description()
        if tooltip_text:
            pos = QCursor.pos() + QPoint(10, 10)
            QToolTip.showText(pos, tooltip_text, label)

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
        self._update_legend() 
        self.update_visualizations()
        logger.debug(f"Changed graph type to: {self.graph_combo.currentText()}")

    def set_theme(self, is_dark_mode):
        """Apply the specified theme to the dashboard, ensuring consistent background colors."""
        self.is_dark_mode = is_dark_mode
        
        self._configure_matplotlib_style()
        self._update_figure_colors()

        style = ModernStyles.get_complete_style(self.is_dark_mode)
        colors = ModernStyles.COLORS['dark' if self.is_dark_mode else 'light']
        
        # Set main widget background to match ticker_container
        self.setStyleSheet(f"background-color: {colors['surface']};")
        
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
            QLabel[class="legend-label"] {{
                color: {colors['text_primary']};
                font-size: 9px;
                font-family: 'Segoe UI';
                margin: 0px;
                padding: 0px;
            }}
            QWidget[class="legend-item"] {{
                margin: 0px;
                padding: 0px;
                max-height: 18px;
            }}
            QWidget[class="selected-tickers"] {{
                background-color: transparent;
                border: none;
            }}
            QPushButton[class="ticker-button"] {{
                color: white;
                border: none;
                padding: 1px 4px;
                border-radius: 4px;
                font-size: 9px;
                font-weight: bold;
                min-height: 14px;
                max-height: 14px;
                margin: 0px;
            }}
            QPushButton[class="ticker-button"]:hover {{
                opacity: 0.8;
            }}
        """
        complete_style = style + additional_styles
        self.setStyleSheet(complete_style)
        self.selected_tickers_widget.setStyleSheet(f"""
            QWidget[class="selected-tickers"] {{
                background-color: transparent;
                border: none;
            }}
        """)
        self.ticker_container.setStyleSheet(f"""
            QFrame {{
                background-color: {colors['surface']};
                border: none;
            }}
        """)
        self.chart_canvas.setStyleSheet(f"""
            background-color: {colors['surface']};
            border: none;
        """)
        self.selected_tickers_label.setStyleSheet(f"color: {colors['text_primary']}; font-size: 12px; font-weight: 500;")
        
        self._update_ticker_button_colors()
        self.update_visualizations()
        logger.debug(f"Applied theme: {'dark' if is_dark_mode else 'light'}")

    def _update_ticker_button_colors(self):
        """Update ticker button colors to match plot colors."""
        if not self.selected_tickers_order:
            self.ticker_colors = {}
            return
            
        self.ticker_colors = self._get_ticker_colors(self.selected_tickers_order)
        
        for ticker, button in self.selected_tickers_buttons.items():
            if ticker in self.ticker_colors and button:
                color_info = self.ticker_colors[ticker]
                button_style = f"""
                    QPushButton[class="ticker-button"] {{
                        background-color: {color_info['hex']}; 
                        color: white; 
                        border: none; 
                        padding: 1px 4px; 
                        border-radius: 4px; 
                        font-size: 9px;
                        font-weight: bold;
                        min-height: 14px;
                        max-height: 14px;
                        margin: 0px;
                    }} 
                    QPushButton[class="ticker-button"]:hover {{
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
        """Get consistent colors for tickers using the color palette."""
        if not tickers:
            return {}
        
        ticker_colors = {}
        for idx, ticker in enumerate(tickers):
            color_hex = self.color_palette[idx % len(self.color_palette)]
            rgb = tuple(int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5))
            hover_color = self.adjust_color_brightness(color_hex, 0.8)
            
            ticker_colors[ticker] = {
                'hex': color_hex,
                'rgb': rgb,
                'hover': hover_color
            }
        
        return ticker_colors

    def _update_legend(self):
        """Clear and update the legend in a 5x2 grid below selected tickers."""
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub_item = item.layout().takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().deleteLater()
                item.layout().deleteLater()

        graph_type = self.graph_combo.currentText()
        theme_colors = self._get_theme_colors()

        if graph_type == "Select Graph Type":
            return

        if graph_type == "Predicted vs Actual Sharpe":
            for idx, ticker in enumerate(self.selected_tickers_order[:5]):
                if ticker in self.ticker_colors:
                    color_hex = self.ticker_colors[ticker]['hex']
                    row = idx
                    legend_widget = QWidget()
                    legend_widget.setProperty("class", "legend-item")
                    legend_widget.setFixedHeight(18)
                    legend_item = QHBoxLayout(legend_widget)
                    legend_item.setContentsMargins(1, 0, 1, 0)
                    legend_item.setSpacing(1)
                    color_square = QLabel()
                    color_square.setFixedSize(8, 8)
                    color_square.setStyleSheet(f"background-color: {color_hex}; border: none;")
                    legend_item.addWidget(color_square)
                    label = QLabel(f"{ticker} Predicted")
                    label.setProperty("class", "legend-label")
                    legend_item.addWidget(label)
                    legend_item.addStretch()
                    self.legend_layout.addWidget(legend_widget, row, 0)
                    
                    legend_widget_actual = QWidget()
                    legend_widget_actual.setProperty("class", "legend-item")
                    legend_widget_actual.setFixedHeight(18)
                    legend_item_actual = QHBoxLayout(legend_widget_actual)
                    legend_item_actual.setContentsMargins(1, 0, 1, 0)
                    legend_item_actual.setSpacing(1)
                    color_square_actual = QLabel()
                    color_square_actual.setFixedSize(8, 8)
                    color_square_actual.setStyleSheet(f"background-color: {color_hex}; border: none;")
                    legend_item_actual.addWidget(color_square_actual)
                    label_actual = QLabel(f"{ticker} Actual")
                    label_actual.setProperty("class", "legend-label")
                    legend_item_actual.addWidget(label_actual)
                    legend_item_actual.addStretch()
                    self.legend_layout.addWidget(legend_widget_actual, row, 1)

        elif graph_type == "Profit over Std Dev":
            legend_widget = QWidget()
            legend_widget.setProperty("class", "legend-item")
            legend_widget.setFixedHeight(18)
            legend_item = QHBoxLayout(legend_widget)
            legend_item.setContentsMargins(1, 0, 1, 0)
            legend_item.setSpacing(1)
            color_square = QLabel()
            color_square.setFixedSize(8, 8)
            color_square.setStyleSheet("background-color: #2196F3; border: none;")
            legend_item.addWidget(color_square)
            label = QLabel("Profit / Std Dev")
            label.setProperty("class", "legend-label")
            legend_item.addWidget(label)
            legend_item.addStretch()
            self.legend_layout.addWidget(legend_widget, 0, 0)

        elif graph_type == "Portfolio Value Over Time":
            legend_widget = QWidget()
            legend_widget.setProperty("class", "legend-item")
            legend_widget.setFixedHeight(18)
            legend_item = QHBoxLayout(legend_widget)
            legend_item.setContentsMargins(1, 0, 1, 0)
            legend_item.setSpacing(1)
            color_square = QLabel()
            color_square.setFixedSize(8, 8)
            color_square.setStyleSheet("background-color: #2196F3; border: none;")
            legend_item.addWidget(color_square)
            label = QLabel("Portfolio Value")
            label.setProperty("class", "legend-label")
            legend_item.addWidget(label)
            legend_item.addStretch()
            self.legend_layout.addWidget(legend_widget, 0, 0)
            
            if self.data_manager.data is not None and 'Ticker' in self.data_manager.data and 'SPY' in self.data_manager.data['Ticker'].values:
                legend_widget_spy = QWidget()
                legend_widget_spy.setProperty("class", "legend-item")
                legend_widget_spy.setFixedHeight(18)
                legend_item_spy = QHBoxLayout(legend_widget_spy)
                legend_item_spy.setContentsMargins(1, 0, 1, 0)
                legend_item_spy.setSpacing(1)
                color_square_spy = QLabel()
                color_square_spy.setFixedSize(8, 8)
                color_square_spy.setStyleSheet("background-color: #F59E0B; border: none;")
                legend_item_spy.addWidget(color_square_spy)
                label_spy = QLabel("S&P 500")
                label_spy.setProperty("class", "legend-label")
                legend_item_spy.addWidget(label_spy)
                legend_item_spy.addStretch()
                self.legend_layout.addWidget(legend_widget_spy, 0, 1)

        elif graph_type == "Buy/Sell Distribution":
            for idx, (label_text, color) in enumerate(zip(['Buy', 'Sell'], ['#2196F3', '#F59E0B'])):
                legend_widget = QWidget()
                legend_widget.setProperty("class", "legend-item")
                legend_widget.setFixedHeight(18)
                legend_item = QHBoxLayout(legend_widget)
                legend_item.setContentsMargins(1, 0, 1, 0)
                legend_item.setSpacing(1)
                color_square = QLabel()
                color_square.setFixedSize(8, 8)
                color_square.setStyleSheet(f"background-color: {color}; border: none;")
                legend_item.addWidget(color_square)
                label = QLabel(label_text)
                label.setProperty("class", "legend-label")
                legend_item.addWidget(label)
                legend_item.addStretch()
                self.legend_layout.addWidget(legend_widget, 0, idx)

        elif graph_type == "Win Rate Over Time":
            # Legend for win rate plot
            legend_items = [
                ("Overall Win Rate", "#2196F3"),
                ("Rolling 30-Day", "#F59E0B"),
                ("Target (70%)", "#d62728")
            ]
            
            for idx, (label_text, color) in enumerate(legend_items):
                if idx < 3:  # Limit to 3 items to fit in the legend area
                    legend_widget = QWidget()
                    legend_widget.setProperty("class", "legend-item")
                    legend_widget.setFixedHeight(18)
                    legend_item = QHBoxLayout(legend_widget)
                    legend_item.setContentsMargins(1, 0, 1, 0)
                    legend_item.setSpacing(1)
                    color_square = QLabel()
                    color_square.setFixedSize(8, 8)
                    color_square.setStyleSheet(f"background-color: {color}; border: none;")
                    legend_item.addWidget(color_square)
                    label = QLabel(label_text)
                    label.setProperty("class", "legend-label")
                    legend_item.addWidget(label)
                    legend_item.addStretch()
                    row = idx // 2
                    col = idx % 2
                    self.legend_layout.addWidget(legend_widget, row, col)

    def update_selected_tickers(self):
        """Update the display of selected tickers in a vertical layout with dynamic sizing."""
        if hasattr(self, '_updating_tickers') and self._updating_tickers:
            return
        self._updating_tickers = True

        try:
            # Clear existing buttons in the layout and dictionary
            while self.selected_tickers_layout.count():
                item = self.selected_tickers_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.selected_tickers_buttons.clear()  # Clear dictionary to avoid stale references
            logger.debug("Cleared selected tickers layout and buttons dictionary")

            # Get selected tickers
            selected_items = self.ticker_list.selectedItems()
            selected_tickers = [item.text() for item in selected_items]

            # Enforce 5-ticker limit
            if len(selected_tickers) > 5:
                msg = QMessageBox(self)  # Set parent
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle("Selection Limit")
                msg.setText("Please select up to 5 tickers only.")
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec()
                
                # Deselect excess items and prevent double processing
                for item in selected_items[5:]:
                    item.setSelected(False)
                selected_tickers = selected_tickers[:5]
                
                # Update the selected_items list to match the corrected selection
                selected_items = [item for item in selected_items if item.text() in selected_tickers]

            # Update ticker order
            new_order = []
            for ticker in self.selected_tickers_order:
                if ticker in selected_tickers:
                    new_order.append(ticker)
            for ticker in selected_tickers:
                if ticker not in new_order:
                    new_order.append(ticker)
            
            self.selected_tickers_order = new_order[:5]  # Ensure max 5 tickers
            logger.debug(f"Updated selected tickers order: {self.selected_tickers_order}")

            # Update ticker colors
            self.ticker_colors = self._get_ticker_colors(self.selected_tickers_order)

            # Add buttons for selected tickers
            font = QFont("Segoe UI", 9)
            font_metrics = QFontMetrics(font)
            for ticker in self.selected_tickers_order:
                color_info = self.ticker_colors.get(ticker, {'hex': '#1f77b4', 'hover': '#184f8d'})
                
                button = QPushButton(ticker)
                button.setProperty("class", "ticker-button")
                button.setFixedHeight(14)
                button.setMinimumWidth(0)  # Allow shrinking
                button.setMaximumWidth(190)  # Cap at container width
                button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                button.setVisible(True)
                button.clicked.connect(lambda _, t=ticker: self.remove_ticker(t))
                
                button_style = f"""
                    QPushButton[class="ticker-button"] {{
                        background-color: {color_info['hex']}; 
                        color: white; 
                        border: none; 
                        padding: 1px 4px; 
                        border-radius: 4px; 
                        font-size: 9px;
                        font-weight: bold;
                        min-height: 14px;
                        max-height: 14px;
                        margin: 0px;
                    }} 
                    QPushButton[class="ticker-button"]:hover {{
                        background-color: {color_info['hover']};
                    }}
                """
                button.setStyleSheet(button_style)
                self.selected_tickers_layout.addWidget(button)
                self.selected_tickers_buttons[ticker] = button
                text_width = font_metrics.horizontalAdvance(ticker) + 10
                logger.debug(f"Added ticker button: {ticker}, color {color_info['hex']}, width {text_width}")

            self._update_legend()
            self.update_visualizations()
            logger.debug(f"Selected tickers displayed: {self.selected_tickers_order}")
        
        finally:
            # ADD THIS TO RESET THE FLAG:
            self._updating_tickers = False


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
        logger.debug(f"Removed ticker: {ticker}")

    def clear_all_tickers(self):
        """Clear all selected tickers."""
        self.ticker_list.clearSelection()
        self.selected_tickers_order = []
        self.update_selected_tickers()
        logger.debug("Cleared all tickers")

    def get_message_box_style(self):
        """Get stylesheet for message boxes."""
        colors = ModernStyles.COLORS['dark' if self.is_dark_mode else 'light']
        return f"""
            QMessageBox {{
                background-color: {colors['primary']};
                color: {colors['text_primary']};
                font-size: 14px;
                border: 1px solid {colors['border']};
                border-radius: 8px;
                padding: 16px;
            }}
            QMessageBox QLabel {{
                color: {colors['text_primary']};
                padding: 12px;
                font-size: 13px;
            }}
            QMessageBox QPushButton {{
                background-color: {colors['accent']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                min-width: 80px;
                margin: 4px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: {colors['accent_hover']};
            }}
        """

    def calculate_win_rate_over_time(self):
        """Calculate win rate over time from both completed trades and open positions."""
        try:
            # Get both orders and portfolio history for complete analysis
            orders = get_orders()
            portfolio_history = get_portfolio_history()
            
            if not orders and not portfolio_history:
                logger.warning("No orders or portfolio history found for win rate calculation")
                return pd.DataFrame()

            all_trades = []
            
            # PART 1: Completed trades (sell orders with profit/loss info)
            if orders:
                orders_df = pd.DataFrame(orders)
                orders_df['date'] = pd.to_datetime(orders_df['date'], utc=True)
                
                # Filter by date range
                orders_df = orders_df[(orders_df['date'] >= self.start_date) & (orders_df['date'] <= self.end_date)]
                
                # Get completed trades (sell orders)
                sell_orders = orders_df[orders_df['action'] == 'sell'].copy()
                
                for _, sell_order in sell_orders.iterrows():
                    trade_data = {
                        'date': sell_order['date'],
                        'ticker': sell_order['ticker'],
                        'trade_type': 'COMPLETED',
                        'profit_loss': sell_order.get('profit_loss', 0),
                        'is_profitable': sell_order.get('profit_loss', 0) > 0
                    }
                    all_trades.append(trade_data)
            
            # PART 2: Open positions (current holdings with unrealized P&L)
            if portfolio_history:
                # Get current holdings from latest portfolio state
                current_portfolio = portfolio_history[-1] if portfolio_history else {}
                current_holdings = current_portfolio.get('holdings', {})
                
                # Get latest prices for current valuation
                latest_prices = {}
                if self.data_manager.data is not None and not self.data_manager.data.empty:
                    latest_date = self.data_manager.data['date'].max()
                    latest_data = self.data_manager.data[self.data_manager.data['date'] == latest_date]
                    for _, row in latest_data.iterrows():
                        latest_prices[row['Ticker']] = row['Close']
                
                # Analyze each open position
                for ticker, holding in current_holdings.items():
                    shares = holding.get('shares', 0)
                    purchase_price = holding.get('purchase_price', 0)
                    purchase_date = holding.get('purchase_date')
                    
                    if shares <= 0 or purchase_price <= 0:
                        continue
                    
                    # Get current price
                    current_price = latest_prices.get(ticker, purchase_price)
                    
                    # Calculate unrealized profit/loss
                    current_value = shares * current_price
                    invested_value = shares * purchase_price
                    unrealized_pnl = current_value - invested_value
                    
                    # Convert purchase_date to datetime if needed
                    if isinstance(purchase_date, str):
                        purchase_date = pd.to_datetime(purchase_date, utc=True)
                    elif isinstance(purchase_date, pd.Timestamp) and purchase_date.tz is None:
                        purchase_date = purchase_date.tz_localize('UTC')
                    
                    # Only include if purchase date is within our analysis range
                    if purchase_date and purchase_date >= self.start_date and purchase_date <= self.end_date:
                        trade_data = {
                            'date': purchase_date,  # Use purchase date for chronological ordering
                            'ticker': ticker,
                            'trade_type': 'OPEN',
                            'profit_loss': unrealized_pnl,
                            'is_profitable': unrealized_pnl > 0
                        }
                        all_trades.append(trade_data)
            
            if not all_trades:
                logger.warning("No trades (completed or open) found for win rate analysis")
                return pd.DataFrame()
            
            # Convert to DataFrame and sort by date
            trades_df = pd.DataFrame(all_trades)
            trades_df = trades_df.sort_values('date')
            
            # Calculate cumulative win rate over time (including both completed and open trades)
            trades_df['cumulative_wins'] = trades_df['is_profitable'].cumsum()
            trades_df['trade_number'] = range(1, len(trades_df) + 1)
            trades_df['cumulative_win_rate'] = (trades_df['cumulative_wins'] / trades_df['trade_number']) * 100
            
            # Calculate rolling win rate using recent trades
            if len(trades_df) > 10:
                window_size = min(10, len(trades_df))  # Use up to 10 recent trades
                trades_df['rolling_win_rate'] = trades_df['is_profitable'].rolling(
                    window=window_size, min_periods=1
                ).mean() * 100
            else:
                trades_df['rolling_win_rate'] = trades_df['cumulative_win_rate']
            
            return trades_df[['date', 'cumulative_win_rate', 'rolling_win_rate', 'is_profitable', 'trade_number', 'trade_type', 'ticker']]
            
        except Exception as e:
            logger.error(f"Error calculating win rate over time: {e}", exc_info=True)
            return pd.DataFrame()

    def plot_win_rate_over_time(self):
        """Plot win rate over time with rolling window and target line, including both completed and open trades."""
        ax = self.chart_fig.add_subplot(111)
        theme_colors = self._get_theme_colors()
        
        win_rate_data = self.calculate_win_rate_over_time()
        
        if win_rate_data.empty:
            ax.text(0.5, 0.5, 'No trade data available for win rate analysis', 
                   horizontalalignment='center', verticalalignment='center', 
                   color=theme_colors['text'])
            return

        # Plot cumulative win rate
        ax.plot(win_rate_data['date'], win_rate_data['cumulative_win_rate'], 
               label='Overall Win Rate', color='#2196F3', linewidth=2.5, alpha=0.8)
        
        # Plot rolling win rate
        ax.plot(win_rate_data['date'], win_rate_data['rolling_win_rate'], 
               label='Rolling Win Rate', color='#F59E0B', linewidth=2.0, alpha=0.9)
        
        # Add 70% target line
        ax.axhline(y=70, color='#d62728', linestyle='--', linewidth=1.5, 
                  label='Target (70%)', alpha=0.8)
        
        # Add markers for individual trades (differentiate completed vs open)
        profitable_completed = win_rate_data[(win_rate_data['is_profitable']) & (win_rate_data['trade_type'] == 'COMPLETED')]
        losing_completed = win_rate_data[(~win_rate_data['is_profitable']) & (win_rate_data['trade_type'] == 'COMPLETED')]
        profitable_open = win_rate_data[(win_rate_data['is_profitable']) & (win_rate_data['trade_type'] == 'OPEN')]
        losing_open = win_rate_data[(~win_rate_data['is_profitable']) & (win_rate_data['trade_type'] == 'OPEN')]
        
        # Completed trades (solid markers)
        if not profitable_completed.empty:
            ax.scatter(profitable_completed['date'], profitable_completed['cumulative_win_rate'], 
                      color='#2ca02c', s=40, alpha=0.8, marker='^', 
                      label='Profitable (Completed)', zorder=5)
        
        if not losing_completed.empty:
            ax.scatter(losing_completed['date'], losing_completed['cumulative_win_rate'], 
                      color='#d62728', s=40, alpha=0.8, marker='v', 
                      label='Loss (Completed)', zorder=5)
        
        # Open positions (hollow markers)
        if not profitable_open.empty:
            ax.scatter(profitable_open['date'], profitable_open['cumulative_win_rate'], 
                      color='#2ca02c', s=35, alpha=0.6, marker='^', facecolors='none', 
                      edgecolors='#2ca02c', linewidth=2, label='Profitable (Open)', zorder=6)
        
        if not losing_open.empty:
            ax.scatter(losing_open['date'], losing_open['cumulative_win_rate'], 
                      color='#d62728', s=35, alpha=0.6, marker='v', facecolors='none', 
                      edgecolors='#d62728', linewidth=2, label='Loss (Open)', zorder=6)
        
        # Styling
        ax.set_title('Win Rate Over Time (All Trades: Completed + Open)', pad=15, color=theme_colors['text'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Win Rate (%)', color=theme_colors['text'])
        ax.grid(True, linestyle='--', alpha=0.3, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        # Set y-axis limits to show percentage properly
        ax.set_ylim(0, 100)
        
        # Calculate detailed statistics
        final_win_rate = win_rate_data['cumulative_win_rate'].iloc[-1]
        total_trades = len(win_rate_data)
        completed_trades = len(win_rate_data[win_rate_data['trade_type'] == 'COMPLETED'])
        open_trades = len(win_rate_data[win_rate_data['trade_type'] == 'OPEN'])
        
        # Add comprehensive text box with current statistics
        textstr = f'Overall Win Rate: {final_win_rate:.1f}%\nTotal Trades: {total_trades}\nCompleted: {completed_trades} | Open: {open_trades}\nTarget: 70%'
        props = dict(boxstyle='round', facecolor=theme_colors['surface'], alpha=0.9, 
                    edgecolor=theme_colors['grid'])
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props, color=theme_colors['text'])
        
        # Color-code the current win rate text based on performance
        if final_win_rate >= 70:
            status_color = '#2ca02c'  # Green for good performance
            status_text = 'TARGET MET ✓'
        elif final_win_rate >= 60:
            status_color = '#F59E0B'  # Orange for moderate performance
            status_text = 'CLOSE TO TARGET'
        else:
            status_color = '#d62728'  # Red for poor performance
            status_text = 'BELOW TARGET'
        
        ax.text(0.98, 0.98, status_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='top', horizontalalignment='right', color=status_color)
        
        # Add a breakdown line showing the mix of completed vs open trades
        breakdown_text = f'Mix: {(completed_trades/total_trades)*100:.0f}% Completed, {(open_trades/total_trades)*100:.0f}% Open'
        ax.text(0.98, 0.02, breakdown_text, transform=ax.transAxes, fontsize=9, style='italic',
               verticalalignment='bottom', horizontalalignment='right', color=theme_colors['text'], alpha=0.7)
        
        self.chart_fig.tight_layout()

    def update_visualizations(self):
        """Update the visualization based on the selected graph type."""
        graph_type = self.graph_combo.currentText()

        self.chart_fig.clear()
        self._update_figure_colors()

        if graph_type == "Select Graph Type":
            ax = self.chart_fig.add_subplot(111)
            theme_colors = self._get_theme_colors()
            ax.text(0.5, 0.5, 'Please select a graph type to display visualization', 
                horizontalalignment='center', verticalalignment='center', 
                color=theme_colors['text'], fontsize=14)
            ax.set_facecolor(theme_colors['surface'])
            self.chart_canvas.draw()
            return

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

        if graph_type == "Predicted vs Actual Sharpe":
            self.plot_predicted_vs_actual_sharpe(self.selected_tickers_order)
        elif graph_type == "Profit over Std Dev":
            self.plot_profit_over_std_dev()
        elif graph_type == "Portfolio Value Over Time":
            self.plot_portfolio_value()
        elif graph_type == "Buy/Sell Distribution":
            self.plot_buy_sell_distribution()
        elif graph_type == "Win Rate Over Time":  # NEW: Handle win rate plot
            self.plot_win_rate_over_time()

        self.chart_canvas.draw()

    def update_dashboard(self):
        """Refresh the dashboard visualizations."""
        self.update_visualizations()
        logger.info("Dashboard updated")

    def plot_predicted_vs_actual_sharpe(self, selected_tickers):
        """Plot predicted Sharpe as dots and actual Sharpe as solid lines over time."""
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

        colors = [self.ticker_colors[ticker]['rgb'] for ticker in selected_tickers if ticker in self.ticker_colors]

        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker].sort_values('date')
            if not ticker_data.empty:
                # Predicted values as dots/scatter
                ax.scatter(ticker_data['date'], ticker_data['Best_Prediction'],
                        label=f'{ticker} Predicted', color=colors[idx], s=50)
                
                # Actual values as solid lines (only where available)
                actual_data = ticker_data[ticker_data['Actual_Sharpe'] != -1.0]
                if not actual_data.empty:
                    ax.plot(actual_data['date'], actual_data['Actual_Sharpe'],
                        label=f'{ticker} Actual', color=colors[idx], linewidth=2.0)

        ax.set_title('Predicted vs Actual Sharpe Ratios', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Sharpe Ratio', color=theme_colors['text'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_profit_over_std_dev(self):
        """Plot profit divided by annual standard deviation over rolling one-year periods."""
        ax = self.chart_fig.add_subplot(111)
        theme_colors = self._get_theme_colors()
        
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        df = pd.DataFrame(portfolio_history)

        try:
            df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True)
        except Exception as e:
            # Fallback: try without utc=True first, then convert
            try:
                df['date'] = pd.to_datetime(df['date'], format='mixed')
                if df['date'].dt.tz is None:
                    df['date'] = df['date'].dt.tz_localize('UTC')
                else:
                    df['date'] = df['date'].dt.tz_convert('UTC')
            except Exception as fallback_error:
                logger.error(f"Failed to parse dates: {e}, fallback error: {fallback_error}")
                ax.text(0.5, 0.5, 'Error parsing portfolio history dates', 
                    horizontalalignment='center', verticalalignment='center', 
                    color=theme_colors['text'])
                return
        
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        if df.empty:
            ax.text(0.5, 0.5, 'No portfolio history in date range', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        ratios = []
        dates = []
        window_days = 365
        df = df.sort_values('date')
        
        for i in range(len(df) - window_days):
            period = df.iloc[i:i + window_days]
            if len(period) == window_days:
                profit = period['value'].iloc[-1] - period['value'].iloc[0]
                daily_returns = period['value'].pct_change().dropna()
                annual_std = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else np.nan
                ratio = profit / annual_std if annual_std != 0 and not np.isnan(annual_std) else np.nan
                ratios.append(ratio)
                dates.append(period['date'].iloc[-1])

        if not ratios or np.isnan(ratios).all():
            ax.text(0.5, 0.5, 'Insufficient data for one-year periods', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        ax.plot(dates, ratios, label='Profit / Std Dev', color='#2196F3', linewidth=2.0)
        ax.set_title('Profit over Annual Std Dev (Rolling One-Year)', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Profit / Std Dev', color=theme_colors['text'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_portfolio_value(self):
        """Plot portfolio value over time with S&P 500 benchmark."""
        ax = self.chart_fig.add_subplot(111)
        theme_colors = self._get_theme_colors()
        
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        df = pd.DataFrame(portfolio_history)

        try:
            df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True)
        except:
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            if df['date'].dt.tz is None:
                df['date'] = df['date'].dt.tz_localize('UTC')
            else:
                df['date'] = df['date'].dt.tz_convert('UTC')

        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        if df.empty:
            ax.text(0.5, 0.5, 'No portfolio history in date range', horizontalalignment='center',
                    verticalalignment='center', color=theme_colors['text'])
            return

        ax.plot(df['date'], df['value'], label='Portfolio Value', color='#2196F3', linewidth=2.0)
        if self.data_manager.data is not None and 'Ticker' in self.data_manager.data and 'SPY' in self.data_manager.data['Ticker'].values:
            sp500_data = self.data_manager.data[self.data_manager.data['Ticker'] == 'SPY'].copy()
            sp500_data['date'] = pd.to_datetime(sp500_data['date'], utc=True)
            sp500_data = sp500_data[(sp500_data['date'] >= self.start_date) & (sp500_data['date'] <= self.end_date)]
            if not sp500_data.empty:
                sp500_data['value'] = sp500_data['Close'] / sp500_data['Close'].iloc[0] * df['value'].iloc[0]
                ax.plot(sp500_data['date'], sp500_data['value'], label='S&P 500', color='#F59E0B', linestyle='--', linewidth=1.5)

        ax.set_title('Portfolio Value Over Time', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Value ($)', color=theme_colors['text'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_buy_sell_distribution(self):
        """Plot pie chart of buy vs sell transactions in the current portfolio."""
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

        action_counts = orders_df['action'].value_counts()
        labels = action_counts.index
        sizes = action_counts.values
        colors = ['#2196F3', '#F59E0B']

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Buy/Sell Distribution', pad=10, color=theme_colors['text'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()