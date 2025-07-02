# analysis_dashboard.py
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from PyQt6.QtCore import Qt, QEvent, QPoint, QSize, QRect
from PyQt6.QtGui import QCursor, QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from backend.trading_logic_new import get_orders, get_portfolio_history
from frontend.gui.styles import ModernStyles
from frontend.logging_config import get_logger

logger = get_logger(__name__)

# Constants
MAX_TICKERS = 5
WINDOW_DAYS = 365
COLOR_PALETTE = [
    "#E53E3E",  # Red
    "#38A169",  # Green
    "#3182CE",  # Blue
    "#D69E2E",  # Orange
    "#805AD5",  # Purple
]


class FlowLayout(QLayout):
    """A layout that arranges widgets in a flowing grid, wrapping to new rows as needed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(2, 2, 2, 2)
        self._spacing = 5

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(2 * margins.left(), 2 * margins.top())
        return size

    def _do_layout(self, rect, apply_geometry):
        x, y = rect.x(), rect.y()
        line_height = 0
        max_height = rect.height()
        for item in self._items:
            widget = item.widget()
            if widget is None:
                continue
            item_size = item.sizeHint()
            item_width, item_height = item_size.width(), item_size.height()
            space_x = space_y = self._spacing
            next_x = x + item_width + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y += line_height + space_y
                next_x = x + item_width + space_x
                line_height = 0
            if y + item_height <= rect.y() + max_height:
                if apply_geometry:
                    item.setGeometry(QRect(QPoint(x, y), QSize(item_width, item_height)))
                x = next_x
                line_height = max(line_height, item_height)
        return y + line_height - rect.y()


class CustomFigureCanvas(FigureCanvas):
    """Custom canvas with hover tooltips for plots."""

    def __init__(self, figure, parent=None, get_tooltip_text=None):
        super().__init__(figure)
        self.setParent(parent)
        self.get_tooltip_text = get_tooltip_text
        self.setMouseTracking(True)

    def enterEvent(self, event):
        if self.get_tooltip_text:
            tooltip_text = self.get_tooltip_text()
            if tooltip_text:
                pos = QCursor.pos() + QPoint(10, 10)
                QToolTip.showText(pos, tooltip_text, self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)


class CustomComboBox(QComboBox):
    """Custom QComboBox with hover tooltip for plot descriptions."""

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
        QToolTip.hideText()
        super().leaveEvent(event)


class AnalysisDashboard(QWidget):
    """A PyQt6-based dashboard for visualizing trading analysis data."""

    PLOT_DESCRIPTIONS = {
        "Predicted vs Actual Sharpe": "Plots predicted Sharpe ratios as dots over actual Sharpe ratios as lines over time.",
        "Profit over Std Dev": "Shows profit divided by annual standard deviation over rolling one-year periods.",
        "Portfolio Value Over Time": "Displays portfolio value over time compared to the S&P 500 benchmark.",
        "Buy/Sell Distribution": "Pie chart of buy vs sell transactions in the portfolio.",
        "Win Rate Over Time": "Percentage of profitable trades over time using rolling window analysis.",
    }

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.is_dark_mode = True
        self.ticker_colors = {}
        self.selected_tickers_order = []
        self._updating_tickers = False
        self._configure_matplotlib()
        self.setAutoFillBackground(True)
        self._setup_ui()
        self._connect_signals()
        logger.info("AnalysisDashboard initialized")

    def _configure_matplotlib(self):
        """Configure matplotlib style based on theme."""
        plt.style.use("dark_background" if self.is_dark_mode else "default")
        params = {
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.facecolor": "#2b2b2b" if self.is_dark_mode else "#ffffff",
            "figure.facecolor": "#212121" if self.is_dark_mode else "#f5f5f5",
            "axes.labelcolor": "#ffffff" if self.is_dark_mode else "#333333",
            "xtick.color": "#ffffff" if self.is_dark_mode else "#333333",
            "ytick.color": "#ffffff" if self.is_dark_mode else "#333333",
            "text.color": "#ffffff" if self.is_dark_mode else "#333333",
            "axes.edgecolor": "#ffffff" if self.is_dark_mode else "#333333",
            "grid.color": "#444444" if self.is_dark_mode else "#cccccc",
        }
        plt.rcParams.update(params)

    def _setup_ui(self):
        """Set up the UI with a sidebar for tickers and a graph display area."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(24)

        self._setup_ticker_sidebar(main_layout)
        self._setup_graph_area(main_layout)
        self.set_theme(self.is_dark_mode)
        self.update_visualizations()

    def _setup_ticker_sidebar(self, main_layout):
        """Configure the ticker selection sidebar."""
        self.ticker_container = QFrame()  # Assign to self.ticker_container
        self.ticker_container.setFixedWidth(200)
        ticker_layout = QVBoxLayout(self.ticker_container)
        ticker_layout.setContentsMargins(0, 0, 0, 0)
        ticker_layout.setSpacing(8)

        ticker_label = QLabel("Tickers")
        ticker_label.setProperty("class", "dropdown-label")
        ticker_layout.addWidget(ticker_label)

        self.ticker_list = QListWidget()
        self.ticker_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.ticker_list.setMaximumWidth(200)
        self.ticker_list.setMinimumHeight(300)
        self.ticker_list.setProperty("class", "dropdown-input")
        self._populate_ticker_list()
        ticker_layout.addWidget(self.ticker_list)

        self._setup_selected_tickers_header(ticker_layout)
        self._setup_selected_tickers_area(ticker_layout)
        self._setup_legend_area(ticker_layout)
        ticker_layout.addStretch()
        main_layout.addWidget(self.ticker_container, stretch=0)

    def _populate_ticker_list(self):
        """Populate the ticker list with available tickers."""
        if self.data_manager.data is not None and not self.data_manager.data.empty:
            tickers = sorted(self.data_manager.data["Ticker"].unique())
            for ticker in tickers:
                self.ticker_list.addItem(QListWidgetItem(ticker))
        else:
            logger.warning("No tickers available due to missing market data")

    def _setup_selected_tickers_header(self, ticker_layout):
        """Set up the header for selected tickers with a clear button."""
        self.selected_tickers_header = QHBoxLayout()
        self.selected_tickers_header.setSpacing(5)
        self.selected_tickers_label = QLabel("Selected:")
        self.selected_tickers_header.addWidget(self.selected_tickers_label)
        self.selected_tickers_header.addStretch()
        self.clear_tickers_button = QPushButton("Clear All")
        self.clear_tickers_button.setProperty("class", "secondary")
        self.clear_tickers_button.setMaximumHeight(30)
        self.selected_tickers_header.addWidget(self.clear_tickers_button)
        ticker_layout.addLayout(self.selected_tickers_header)

    def _setup_selected_tickers_area(self, ticker_layout):
        """Set up the area for displaying selected ticker buttons."""
        self.selected_tickers_widget = QWidget()
        self.selected_tickers_widget.setProperty("class", "selected-tickers")
        self.selected_tickers_layout = QVBoxLayout(self.selected_tickers_widget)
        self.selected_tickers_layout.setContentsMargins(2, 2, 2, 2)
        self.selected_tickers_layout.setSpacing(2)
        self.selected_tickers_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.selected_tickers_buttons = {}
        ticker_layout.addWidget(self.selected_tickers_widget)

    def _setup_legend_area(self, ticker_layout):
        """Set up the legend area with a grid layout."""
        self.legend_widget = QWidget()
        self.legend_widget.setFixedHeight(100)
        self.legend_layout = QGridLayout(self.legend_widget)
        self.legend_layout.setContentsMargins(2, 2, 2, 2)
        self.legend_layout.setSpacing(2)
        for row in range(5):
            self.legend_layout.setRowStretch(row, 1)
        ticker_layout.addWidget(self.legend_widget)

    def _setup_graph_area(self, main_layout):
        """Configure the graph type selection and chart display area."""
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)

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
            "Win Rate Over Time",
        ])
        self.graph_combo.setProperty("class", "dropdown-input")
        graph_layout.addWidget(self.graph_combo)
        graph_layout.addStretch()
        right_layout.addWidget(graph_container)

        self.chart_fig = Figure(figsize=(10, 6))
        self.chart_canvas = CustomFigureCanvas(
            self.chart_fig, parent=self, get_tooltip_text=self.get_current_plot_description
        )
        right_layout.addWidget(self.chart_canvas, stretch=2)
        main_layout.addLayout(right_layout, stretch=1)

    def _connect_signals(self):
        """Connect UI signals to their respective slots."""
        self.ticker_list.itemSelectionChanged.connect(self.update_selected_tickers)
        self.clear_tickers_button.clicked.connect(self.clear_all_tickers)
        self.graph_combo.currentIndexChanged.connect(self.change_graph_type)

    def _show_label_tooltip(self, event, label):
        """Show tooltip for the graph type label."""
        tooltip_text = self.get_current_plot_description()
        if tooltip_text:
            pos = QCursor.pos() + QPoint(10, 10)
            QToolTip.showText(pos, tooltip_text, label)

    def get_current_plot_description(self):
        """Return the description of the currently selected plot."""
        return self.PLOT_DESCRIPTIONS.get(self.graph_combo.currentText(), "No description available.")

    def set_initial_capital(self, initial_capital):
        """Set the initial capital for the portfolio."""
        self.initial_capital = initial_capital
        self.update_visualizations()
        logger.debug(f"Set initial capital: ${initial_capital:,.2f}")

    def change_graph_type(self):
        """Handle graph type change and update visualizations."""
        self._update_legend()
        self.update_visualizations()
        logger.debug(f"Changed graph type to: {self.graph_combo.currentText()}")

    def set_theme(self, is_dark_mode):
        """Apply the specified theme to the dashboard."""
        self.is_dark_mode = is_dark_mode
        self._configure_matplotlib()
        self._update_figure_colors()

        colors = ModernStyles.COLORS["dark" if self.is_dark_mode else "light"]
        style = ModernStyles.get_complete_style(self.is_dark_mode)
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
        self.setStyleSheet(style + additional_styles)
        self.ticker_container.setStyleSheet(f"background-color: {colors['surface']}; border: none;")
        self.selected_tickers_label.setStyleSheet(
            f"color: {colors['text_primary']}; font-size: 12px; font-weight: 500;"
        )
        self._update_ticker_button_colors()
        self.update_visualizations()
        logger.debug(f"Applied theme: {'dark' if self.is_dark_mode else 'light'}")

    def _update_ticker_button_colors(self):
        """Update ticker button colors to match plot colors."""
        if not self.selected_tickers_order:
            self.ticker_colors = {}
            return
        self.ticker_colors = self._get_ticker_colors(self.selected_tickers_order)
        for ticker, button in self.selected_tickers_buttons.items():
            if ticker in self.ticker_colors and button:
                color_info = self.ticker_colors[ticker]
                button.setStyleSheet(
                    f"""
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
                )

    def _update_figure_colors(self):
        """Update figure and canvas colors to match the theme."""
        colors = ModernStyles.COLORS["dark" if self.is_dark_mode else "light"]
        self.chart_fig.patch.set_facecolor(colors["surface"])
        self.chart_canvas.setStyleSheet(f"background-color: {colors['surface']}; border: none;")

    def _get_theme_colors(self):
        """Return colors for the current theme."""
        colors = ModernStyles.COLORS["dark" if self.is_dark_mode else "light"]
        return {
            "text": colors["text_primary"],
            "bg": colors["surface"],
            "surface": colors["surface"],
            "grid": "#444444" if self.is_dark_mode else "#cccccc",
            "legend_bg": "#2b2b2b" if self.is_dark_mode else "#ffffff",
        }

    def _get_ticker_colors(self, tickers):
        """Assign consistent colors to tickers from the color palette."""
        if not tickers:
            return {}
        ticker_colors = {}
        for idx, ticker in enumerate(tickers):
            color_hex = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
            rgb = tuple(int(color_hex[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
            hover_color = self._adjust_color_brightness(color_hex, 0.8)
            ticker_colors[ticker] = {"hex": color_hex, "rgb": rgb, "hover": hover_color}
        return ticker_colors

    def _update_legend(self):
        """Update the legend based on the selected graph type."""
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        graph_type = self.graph_combo.currentText()
        if graph_type == "Select Graph Type":
            return

        theme_colors = self._get_theme_colors()
        if graph_type == "Predicted vs Actual Sharpe":
            for idx, ticker in enumerate(self.selected_tickers_order[:MAX_TICKERS]):
                if ticker in self.ticker_colors:
                    color_hex = self.ticker_colors[ticker]["hex"]
                    self._add_legend_item(idx, 0, ticker, "Predicted", color_hex)
                    self._add_legend_item(idx, 1, ticker, "Actual", color_hex)
        elif graph_type == "Profit over Std Dev":
            self._add_legend_item(0, 0, "Profit / Std Dev", "", "#2196F3")
        elif graph_type == "Portfolio Value Over Time":
            self._add_legend_item(0, 0, "Portfolio Value", "", "#2196F3")
            if self._has_sp500_data():
                self._add_legend_item(0, 1, "S&P 500", "", "#F59E0B")
        elif graph_type == "Buy/Sell Distribution":
            for idx, (label, color) in enumerate([("Buy", "#2196F3"), ("Sell", "#F59E0B")]):
                self._add_legend_item(0, idx, label, "", color)
        elif graph_type == "Win Rate Over Time":
            legend_items = [
                ("Overall Win Rate", "#2196F3"),
                ("Rolling 30-Day", "#F59E0B"),
                ("Target (70%)", "#d62728"),
            ]
            for idx, (label, color) in enumerate(legend_items[:3]):
                row, col = idx // 2, idx % 2
                self._add_legend_item(row, col, label, "", color)

    def _add_legend_item(self, row, col, label_prefix, label_suffix, color):
        """Add a single legend item to the grid."""
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

        label_text = f"{label_prefix} {label_suffix}".strip()
        label = QLabel(label_text)
        label.setProperty("class", "legend-label")
        legend_item.addWidget(label)
        legend_item.addStretch()
        self.legend_layout.addWidget(legend_widget, row, col)

    def _has_sp500_data(self):
        """Check if S&P 500 data is available."""
        return (
            self.data_manager.data is not None
            and "Ticker" in self.data_manager.data
            and "SPY" in self.data_manager.data["Ticker"].values
        )

    def update_selected_tickers(self):
        """Update the display of selected tickers."""
        if self._updating_tickers:
            return
        self._updating_tickers = True
        try:
            self._clear_selected_tickers()
            selected_tickers = [item.text() for item in self.ticker_list.selectedItems()]
            if len(selected_tickers) > MAX_TICKERS:
                self._show_ticker_limit_warning()
                selected_tickers = selected_tickers[:MAX_TICKERS]
                for item in self.ticker_list.findItems("", Qt.MatchFlag.MatchContains):
                    if item.text() not in selected_tickers:
                        item.setSelected(False)

            self.selected_tickers_order = [
                t for t in self.selected_tickers_order if t in selected_tickers
            ] + [t for t in selected_tickers if t not in self.selected_tickers_order]
            self.selected_tickers_order = self.selected_tickers_order[:MAX_TICKERS]
            self.ticker_colors = self._get_ticker_colors(self.selected_tickers_order)
            self._display_selected_tickers()
            self._update_legend()
            self.update_visualizations()
            logger.debug(f"Selected tickers: {self.selected_tickers_order}")
        finally:
            self._updating_tickers = False

    def _clear_selected_tickers(self):
        """Remove all existing ticker buttons."""
        while self.selected_tickers_layout.count():
            item = self.selected_tickers_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.selected_tickers_buttons.clear()

    def _show_ticker_limit_warning(self):
        """Display a warning when too many tickers are selected."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Selection Limit")
        msg.setText(f"Please select up to {MAX_TICKERS} tickers only.")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        msg.setStyleSheet(self._get_message_box_style())
        msg.exec()

    def _display_selected_tickers(self):
        """Add buttons for selected tickers."""
        font = QFont("Segoe UI", 9)
        font_metrics = QFontMetrics(font)
        for ticker in self.selected_tickers_order:
            color_info = self.ticker_colors.get(ticker, {"hex": "#1f77b4", "hover": "#184f8d"})
            button = QPushButton(ticker)
            button.setProperty("class", "ticker-button")
            button.setFixedHeight(14)
            button.setMinimumWidth(0)
            button.setMaximumWidth(190)
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _, t=ticker: self.remove_ticker(t))
            button.setStyleSheet(
                f"""
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
            )
            self.selected_tickers_layout.addWidget(button)
            self.selected_tickers_buttons[ticker] = button

    def _adjust_color_brightness(self, hex_color, factor):
        """Adjust the brightness of a hex color."""
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

    def _get_message_box_style(self):
        """Return stylesheet for message boxes."""
        colors = ModernStyles.COLORS["dark" if self.is_dark_mode else "light"]
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
        """Calculate win rate over time including completed and open trades."""
        try:
            orders = get_orders()
            portfolio_history = get_portfolio_history()
            if not orders and not portfolio_history:
                logger.warning("No orders or portfolio history found")
                return pd.DataFrame()

            all_trades = []
            self._process_completed_trades(orders, all_trades)
            self._process_open_positions(portfolio_history, all_trades)
            if not all_trades:
                logger.warning("No trades found for win rate analysis")
                return pd.DataFrame()

            trades_df = pd.DataFrame(all_trades).sort_values("date")
            trades_df["cumulative_wins"] = trades_df["is_profitable"].cumsum()
            trades_df["trade_number"] = range(1, len(trades_df) + 1)
            trades_df["cumulative_win_rate"] = (trades_df["cumulative_wins"] / trades_df["trade_number"]) * 100
            trades_df["rolling_win_rate"] = (
                trades_df["is_profitable"]
                .rolling(window=min(10, len(trades_df)), min_periods=1)
                .mean()
                * 100
            )
            return trades_df[
                [
                    "date",
                    "cumulative_win_rate",
                    "rolling_win_rate",
                    "is_profitable",
                    "trade_number",
                    "trade_type",
                    "ticker",
                ]
            ]
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}", exc_info=True)
            return pd.DataFrame()

    def _process_completed_trades(self, orders, all_trades):
        """Process completed trades from order data."""
        if not orders:
            return
        orders_df = pd.DataFrame(orders)
        orders_df["date"] = pd.to_datetime(orders_df["date"], utc=True)
        orders_df = orders_df[
            (orders_df["date"] >= self.start_date) & (orders_df["date"] <= self.end_date)
        ]
        sell_orders = orders_df[orders_df["action"] == "sell"]
        for _, order in sell_orders.iterrows():
            all_trades.append(
                {
                    "date": order["date"],
                    "ticker": order["ticker"],
                    "trade_type": "COMPLETED",
                    "profit_loss": order.get("profit_loss", 0),
                    "is_profitable": order.get("profit_loss", 0) > 0,
                }
            )

    def _process_open_positions(self, portfolio_history, all_trades):
        """Process open positions from portfolio history."""
        if not portfolio_history:
            return
        current_portfolio = portfolio_history[-1]
        holdings = current_portfolio.get("holdings", {})
        latest_prices = self._get_latest_prices()
        for ticker, holding in holdings.items():
            shares = holding.get("shares", 0)
            purchase_price = holding.get("purchase_price", 0)
            purchase_date = pd.to_datetime(holding.get("purchase_date"), utc=True)
            if (
                shares <= 0
                or purchase_price <= 0
                or not purchase_date
                or not (self.start_date <= purchase_date <= self.end_date)
            ):
                continue
            current_price = latest_prices.get(ticker, purchase_price)
            unrealized_pnl = (current_price - purchase_price) * shares
            all_trades.append(
                {
                    "date": purchase_date,
                    "ticker": ticker,
                    "trade_type": "OPEN",
                    "profit_loss": unrealized_pnl,
                    "is_profitable": unrealized_pnl > 0,
                }
            )

    def _get_latest_prices(self):
        """Return the latest closing prices for tickers."""
        latest_prices = {}
        if self.data_manager.data is not None and not self.data_manager.data.empty:
            latest_date = self.data_manager.data["date"].max()
            latest_data = self.data_manager.data[self.data_manager.data["date"] == latest_date]
            for _, row in latest_data.iterrows():
                latest_prices[row["Ticker"]] = row["Close"]
        return latest_prices

    def plot_win_rate_over_time(self):
        """Plot win rate over time with rolling window and target line."""
        ax = self.chart_fig.add_subplot(111)
        theme_colors = self._get_theme_colors()
        win_rate_data = self.calculate_win_rate_over_time()
        if win_rate_data.empty:
            self._plot_no_data(ax, "No trade data available for win rate analysis")
            return

        ax.plot(
            win_rate_data["date"],
            win_rate_data["cumulative_win_rate"],
            label="Overall Win Rate",
            color="#2196F3",
            linewidth=2.5,
            alpha=0.8,
        )
        ax.plot(
            win_rate_data["date"],
            win_rate_data["rolling_win_rate"],
            label="Rolling Win Rate",
            color="#F59E0B",
            linewidth=2.0,
            alpha=0.9,
        )
        ax.axhline(
            y=70,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="Target (70%)",
            alpha=0.8,
        )
        self._add_trade_markers(ax, win_rate_data)
        self._style_win_rate_plot(ax)
        self._add_win_rate_stats(ax, win_rate_data)
        self.chart_fig.tight_layout()

    def _add_trade_markers(self, ax, win_rate_data):
        """Add markers for individual trades."""
        for condition, style in [
            (
                (win_rate_data["is_profitable"]) & (win_rate_data["trade_type"] == "COMPLETED"),
                {"color": "#2ca02c", "marker": "^", "s": 40, "alpha": 0.8, "label": "Profitable (Completed)"},
            ),
            (
                (~win_rate_data["is_profitable"]) & (win_rate_data["trade_type"] == "COMPLETED"),
                {"color": "#d62728", "marker": "v", "s": 40, "alpha": 0.8, "label": "Loss (Completed)"},
            ),
            (
                (win_rate_data["is_profitable"]) & (win_rate_data["trade_type"] == "OPEN"),
                {
                    "color": "#2ca02c",
                    "marker": "^",
                    "s": 35,
                    "alpha": 0.6,
                    "facecolors": "none",
                    "edgecolors": "#2ca02c",
                    "linewidth": 2,
                    "label": "Profitable (Open)",
                },
            ),
            (
                (~win_rate_data["is_profitable"]) & (win_rate_data["trade_type"] == "OPEN"),
                {
                    "color": "#d62728",
                    "marker": "v",
                    "s": 35,
                    "alpha": 0.6,
                    "facecolors": "none",
                    "edgecolors": "#d62728",
                    "linewidth": 2,
                    "label": "Loss (Open)",
                },
            ),
        ]:
            data = win_rate_data[condition]
            if not data.empty:
                ax.scatter(data["date"], data["cumulative_win_rate"], zorder=5, **style)

    def _style_win_rate_plot(self, ax):
        """Apply styling to the win rate plot."""
        theme_colors = self._get_theme_colors()
        ax.set_title(
            "Win Rate Over Time (All Trades: Completed + Open)",
            pad=15,
            color=theme_colors["text"],
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Date", color=theme_colors["text"])
        ax.set_ylabel("Win Rate (%)", color=theme_colors["text"])
        ax.grid(True, linestyle="--", alpha=0.3, color=theme_colors["grid"])
        ax.set_facecolor(theme_colors["surface"])
        ax.set_ylim(0, 100)

    def _add_win_rate_stats(self, ax, win_rate_data):
        """Add statistics text to the win rate plot."""
        theme_colors = self._get_theme_colors()
        final_win_rate = win_rate_data["cumulative_win_rate"].iloc[-1]
        total_trades = len(win_rate_data)
        completed_trades = len(win_rate_data[win_rate_data["trade_type"] == "COMPLETED"])
        open_trades = len(win_rate_data[win_rate_data["trade_type"] == "OPEN"])

        textstr = (
            f"Overall Win Rate: {final_win_rate:.1f}%\n"
            f"Total Trades: {total_trades}\n"
            f"Completed: {completed_trades} | Open: {open_trades}\n"
            f"Target: 70%"
        )
        props = dict(
            boxstyle="round",
            facecolor=theme_colors["surface"],
            alpha=0.9,
            edgecolor=theme_colors["grid"],
        )
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
            color=theme_colors["text"],
        )

        status_color, status_text = (
            ("#2ca02c", "TARGET MET ✓")
            if final_win_rate >= 70
            else ("#F59E0B", "CLOSE TO TARGET")
            if final_win_rate >= 60
            else ("#d62728", "BELOW TARGET")
        )
        ax.text(
            0.98,
            0.98,
            status_text,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
            color=status_color,
        )

        breakdown_text = (
            f"Mix: {(completed_trades/total_trades)*100:.0f}% Completed, "
            f"{(open_trades/total_trades)*100:.0f}% Open"
        )
        ax.text(
            0.98,
            0.02,
            breakdown_text,
            transform=ax.transAxes,
            fontsize=9,
            style="italic",
            verticalalignment="bottom",
            horizontalalignment="right",
            color=theme_colors["text"],
            alpha=0.7,
        )

    def update_visualizations(self):
        """Update the visualization based on the selected graph type."""
        graph_type = self.graph_combo.currentText()
        self.chart_fig.clear()
        self._update_figure_colors()

        if graph_type == "Select Graph Type":
            ax = self.chart_fig.add_subplot(111)
            self._plot_no_data(ax, "Please select a graph type to display visualization")
            self.chart_canvas.draw()
            return

        self._set_date_range()
        plot_methods = {
            "Predicted vs Actual Sharpe": self.plot_predicted_vs_actual_sharpe,
            "Profit over Std Dev": self.plot_profit_over_std_dev,
            "Portfolio Value Over Time": self.plot_portfolio_value,
            "Buy/Sell Distribution": self.plot_buy_sell_distribution,
            "Win Rate Over Time": self.plot_win_rate_over_time,
        }
        if graph_type in plot_methods:
            plot_methods[graph_type]()
        self.chart_canvas.draw()

    def _set_date_range(self):
        """Set the date range for data filtering."""
        try:
            start_date = (
                pd.to_datetime(self.data_manager.start_date, utc=True)
                if self.data_manager.start_date
                else None
            )
            end_date = (
                pd.to_datetime(self.data_manager.end_date, utc=True)
                if self.data_manager.end_date
                else None
            )
            if (
                (start_date is None or end_date is None)
                and self.data_manager.data is not None
                and not self.data_manager.data.empty
            ):
                dates = pd.to_datetime(self.data_manager.data["date"], utc=True)
                start_date = start_date or dates.min()
                end_date = end_date or dates.max()
            self.start_date = start_date or pd.Timestamp("2000-01-01", tz="UTC")
            self.end_date = end_date or pd.Timestamp.now(tz="UTC")
            logger.debug(f"Using date range: {self.start_date} to {self.end_date}")
        except Exception as e:
            logger.error(f"Error setting dates: {e}")
            ax = self.chart_fig.add_subplot(111)
            self._plot_no_data(ax, "Error: Invalid date range")

    def update_dashboard(self):
        """Refresh the dashboard visualizations."""
        self.update_visualizations()
        logger.info("Dashboard updated")

    def _plot_no_data(self, ax, message):
        """Display a no-data message on the plot."""
        theme_colors = self._get_theme_colors()
        ax.text(
            0.5,
            0.5,
            message,
            horizontalalignment="center",
            verticalalignment="center",
            color=theme_colors["text"],
            fontsize=14,
        )
        ax.set_facecolor(theme_colors["surface"])

    def plot_predicted_vs_actual_sharpe(self):
        """Plot predicted vs actual Sharpe ratios."""
        ax = self.chart_fig.add_subplot(111)
        data = (
            self.data_manager.data.copy()
            if self.data_manager.data is not None
            else None
        )
        if data is None or data.empty:
            self._plot_no_data(ax, "No market data available")
            return

        data["date"] = pd.to_datetime(data["date"], utc=True)
        data = data[(data["date"] >= self.start_date) & (data["date"] <= self.end_date)]
        if not self.selected_tickers_order:
            self._plot_no_data(ax, "Select at least one ticker")
            return

        data = data[data["Ticker"].isin(self.selected_tickers_order)]
        if data.empty:
            self._plot_no_data(ax, "No data for selected tickers")
            return

        for idx, ticker in enumerate(self.selected_tickers_order):
            ticker_data = data[data["Ticker"] == ticker].sort_values("date")
            if not ticker_data.empty:
                color = self.ticker_colors[ticker]["rgb"]
                ax.scatter(
                    ticker_data["date"],
                    ticker_data["Best_Prediction"],
                    label=f"{ticker} Predicted",
                    color=color,
                    s=50,
                )
                actual_data = ticker_data[ticker_data["Actual_Sharpe"] != -1.0]
                if not actual_data.empty:
                    ax.plot(
                        actual_data["date"],
                        actual_data["Actual_Sharpe"],
                        label=f"{ticker} Actual",
                        color=color,
                        linewidth=2.0,
                    )
        self._style_plot(ax, "Predicted vs Actual Sharpe Ratios", "Date", "Sharpe Ratio")

    def plot_profit_over_std_dev(self):
        """Plot profit over annual standard deviation."""
        ax = self.chart_fig.add_subplot(111)
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            self._plot_no_data(ax, "No portfolio history available")
            return

        df = pd.DataFrame(portfolio_history)
        df["date"] = self._parse_dates(df["date"])
        if df["date"] is None:
            self._plot_no_data(ax, "Error parsing portfolio history dates")
            return

        df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]
        if df.empty:
            self._plot_no_data(ax, "No portfolio history in date range")
            return

        ratios, dates = self._calculate_profit_std_dev(df)
        if not ratios or np.isnan(ratios).all():
            self._plot_no_data(ax, "Insufficient data for one-year periods")
            return

        ax.plot(dates, ratios, label="Profit / Std Dev", color="#2196F3", linewidth=2.0)
        self._style_plot(
            ax, "Profit over Annual Std Dev (Rolling One-Year)", "Date", "Profit / Std Dev"
        )

    def _parse_dates(self, date_series):
        """Parse dates with fallback handling."""
        try:
            dates = pd.to_datetime(date_series, format="mixed", utc=True)
            return dates
        except Exception as e:
            try:
                dates = pd.to_datetime(date_series, format="mixed")
                if dates.dt.tz is None:
                    dates = dates.dt.tz_localize("UTC")
                else:
                    dates = dates.dt.tz_convert("UTC")
                return dates
            except Exception as fallback_error:
                logger.error(f"Failed to parse dates: {e}, fallback error: {fallback_error}")
                return None

    def _calculate_profit_std_dev(self, df):
        """Calculate profit over standard deviation for rolling periods."""
        df = df.sort_values("date")
        ratios = []
        dates = []
        for i in range(len(df) - WINDOW_DAYS):
            period = df.iloc[i : i + WINDOW_DAYS]
            if len(period) == WINDOW_DAYS:
                profit = period["value"].iloc[-1] - period["value"].iloc[0]
                daily_returns = period["value"].pct_change().dropna()
                annual_std = (
                    daily_returns.std() * np.sqrt(252) if not daily_returns.empty else np.nan
                )
                ratio = profit / annual_std if annual_std != 0 and not np.isnan(annual_std) else np.nan
                ratios.append(ratio)
                dates.append(period["date"].iloc[-1])
        return ratios, dates

    def plot_portfolio_value(self):
        """Plot portfolio value over time with S&P 500 benchmark."""
        ax = self.chart_fig.add_subplot(111)
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            self._plot_no_data(ax, "No portfolio history available")
            return

        df = pd.DataFrame(portfolio_history)
        df["date"] = self._parse_dates(df["date"])
        if df["date"] is None:
            self._plot_no_data(ax, "Error parsing portfolio history dates")
            return

        df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]
        if df.empty:
            self._plot_no_data(ax, "No portfolio history in date range")
            return

        ax.plot(df["date"], df["value"], label="Portfolio Value", color="#2196F3", linewidth=2.0)
        if self._has_sp500_data():
            sp500_data = self._get_sp500_data(df)
            if not sp500_data.empty:
                ax.plot(
                    sp500_data["date"],
                    sp500_data["value"],
                    label="S&P 500",
                    color="#F59E0B",
                    linestyle="--",
                    linewidth=1.5,
                )
        self._style_plot(ax, "Portfolio Value Over Time", "Date", "Value ($)")

    def _get_sp500_data(self, portfolio_df):
        """Retrieve and normalize S&P 500 data."""
        sp500_data = self.data_manager.data[self.data_manager.data["Ticker"] == "SPY"].copy()
        sp500_data["date"] = pd.to_datetime(sp500_data["date"], utc=True)
        sp500_data = sp500_data[
            (sp500_data["date"] >= self.start_date) & (sp500_data["date"] <= self.end_date)
        ]
        if not sp500_data.empty:
            sp500_data["value"] = (
                sp500_data["Close"] / sp500_data["Close"].iloc[0] * portfolio_df["value"].iloc[0]
            )
        return sp500_data

    def plot_buy_sell_distribution(self):
        """Plot a pie chart of buy vs sell transactions."""
        ax = self.chart_fig.add_subplot(111)
        orders = get_orders()
        if not orders:
            self._plot_no_data(ax, "No trade history available")
            return

        orders_df = pd.DataFrame(orders)
        orders_df["date"] = pd.to_datetime(orders_df["date"], utc=True)
        orders_df = orders_df[
            (orders_df["date"] >= self.start_date) & (orders_df["date"] <= self.end_date)
        ]
        if orders_df.empty:
            self._plot_no_data(ax, "No trades in date range")
            return

        action_counts = orders_df["action"].value_counts()
        ax.pie(
            action_counts.values,
            labels=action_counts.index,
            colors=["#2196F3", "#F59E0B"],
            autopct="%1.1f%%",
            startangle=90,
        )
        theme_colors = self._get_theme_colors()
        ax.set_title("Buy/Sell Distribution", pad=10, color=theme_colors["text"])
        ax.set_facecolor(theme_colors["surface"])
        self.chart_fig.tight_layout()

    def _style_plot(self, ax, title, xlabel, ylabel):
        """Apply common styling to plots."""
        theme_colors = self._get_theme_colors()
        ax.set_title(title, pad=10, color=theme_colors["text"])
        ax.set_xlabel(xlabel, color=theme_colors["text"])
        ax.set_ylabel(ylabel, color=theme_colors["text"])
        ax.grid(True, linestyle="--", alpha=0.5, color=theme_colors["grid"])
        ax.set_facecolor(theme_colors["surface"])
        self.chart_fig.tight_layout()