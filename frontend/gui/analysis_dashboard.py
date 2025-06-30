import matplotlib
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QGroupBox, QListWidget, QListWidgetItem,
                             QPushButton, QMessageBox, QFrame)
from PyQt6.QtCore import Qt
#from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
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

class AnalysisDashboard(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.is_dark_mode = True
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
        """Configure matplotlib styling based on current theme"""
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
            plt.style.use('default')  # Use matplotlib's default light style
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)

        # Controls layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(20)

        # Graph selection with modern styling (like input panel)
        graph_container = QFrame()
        graph_layout = QVBoxLayout(graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)
        
        # Dark label background frame
        graph_label_frame = QFrame()
        graph_label_frame.setProperty("class", "label-frame")
        graph_label_frame.setFixedHeight(35)
        graph_label_layout = QHBoxLayout(graph_label_frame)
        graph_label_layout.setContentsMargins(12, 8, 12, 8)
        
        graph_label = QLabel("Graph Type:")
        graph_label.setProperty("class", "label-dark")
        graph_label_layout.addWidget(graph_label)
        graph_label_layout.addStretch()
        
        # Input field
        graph_input_frame = QFrame()
        graph_input_frame.setProperty("class", "input-frame")
        graph_input_layout = QHBoxLayout(graph_input_frame)
        graph_input_layout.setContentsMargins(12, 12, 12, 12)
        
        self.graph_combo = QComboBox()
        self.graph_combo.addItems([
            "Portfolio Performance",
            "Sharpe Distribution", 
            "Predicted vs Actual Sharpe",
            "Portfolio Drawdown",
            "Cumulative Returns by Ticker",
            "Trade Profit/Loss Distribution"
        ])
        self.graph_combo.currentIndexChanged.connect(self.change_graph_type)
        self.graph_combo.setProperty("class", "input-field")
        graph_input_layout.addWidget(self.graph_combo)
        
        graph_layout.addWidget(graph_label_frame)
        graph_layout.addWidget(graph_input_frame)
        controls_layout.addWidget(graph_container, stretch=1)

        # Ticker selection with modern styling (like input panel)
        ticker_container = QFrame()
        ticker_layout = QVBoxLayout(ticker_container)
        ticker_layout.setContentsMargins(0, 0, 0, 0)
        ticker_layout.setSpacing(0)
        
        # Dark label background frame
        ticker_label_frame = QFrame()
        ticker_label_frame.setProperty("class", "label-frame")
        ticker_label_frame.setFixedHeight(35)
        ticker_label_layout = QHBoxLayout(ticker_label_frame)
        ticker_label_layout.setContentsMargins(12, 8, 12, 8)
        
        ticker_label = QLabel("Tickers:")
        ticker_label.setProperty("class", "label-dark")
        ticker_label_layout.addWidget(ticker_label)
        ticker_label_layout.addStretch()
        
        # Input field with ticker list and controls
        ticker_input_frame = QFrame()
        ticker_input_frame.setProperty("class", "input-frame")
        ticker_input_layout = QVBoxLayout(ticker_input_frame)
        ticker_input_layout.setContentsMargins(12, 12, 12, 12)
        ticker_input_layout.setSpacing(8)
        
        # Selected tickers display
        self.selected_tickers_layout = QHBoxLayout()
        self.selected_tickers_layout.setSpacing(5)
        self.selected_tickers_label = QLabel("Selected:")
        self.selected_tickers_layout.addWidget(self.selected_tickers_label)
        self.selected_tickers_buttons = {}
        self.selected_tickers_layout.addStretch()

        # Clear all button
        self.clear_tickers_button = QPushButton("Clear All")
        self.clear_tickers_button.clicked.connect(self.clear_all_tickers)
        self.clear_tickers_button.setProperty("class", "secondary")
        self.clear_tickers_button.setMaximumHeight(30)
        self.selected_tickers_layout.addWidget(self.clear_tickers_button)

        ticker_input_layout.addLayout(self.selected_tickers_layout)

        # Ticker list
        self.ticker_list = QListWidget()
        self.ticker_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.ticker_list.setMaximumHeight(150)
        if self.data_manager.data is not None and not self.data_manager.data.empty:
            tickers = sorted(self.data_manager.data['Ticker'].unique())
            for ticker in tickers:
                item = QListWidgetItem(ticker)
                self.ticker_list.addItem(item)
        else:
            logger.warning("No tickers available due to missing market data")
        self.ticker_list.itemSelectionChanged.connect(self.update_selected_tickers)

        ticker_input_layout.addWidget(self.ticker_list)
        
        ticker_layout.addWidget(ticker_label_frame)
        ticker_layout.addWidget(ticker_input_frame)
        controls_layout.addWidget(ticker_container, stretch=1)

        layout.addLayout(controls_layout)

        # Chart area
        self.chart_fig = Figure(figsize=(10, 6))
        self.chart_canvas = FigureCanvas(self.chart_fig)
        layout.addWidget(self.chart_canvas, stretch=2)

        self.set_theme(self.is_dark_mode)
        self.update_visualizations()

    def set_initial_capital(self, initial_capital):
        self.initial_capital = initial_capital
        self.update_visualizations()
        logger.debug(f"Set initial capital: ${initial_capital:,.2f}")

    def change_graph_type(self, index):
        self.update_visualizations()
        logger.debug(f"Changed graph type to: {self.graph_combo.currentText()}")

    def set_theme(self, is_dark_mode):
        self.is_dark_mode = is_dark_mode
        
        # Configure matplotlib styling
        self._configure_matplotlib_style()
        
        # Update figure colors immediately
        self._update_figure_colors()

        # Apply modern styling like the input panel
        style = ModernStyles.get_complete_style(self.is_dark_mode)
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        
        # Additional styles for label frames and input frames (same as input panel)
        additional_styles = f"""
            /* Label Frame Styling - Darker background for labels */
            QFrame[class="label-frame"] {{
                background-color: {'#252538' if self.is_dark_mode else '#9CA3AF'};
                border: 1px solid {colors['border']};
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}
            
            /* Input Frame Styling - Matches the surface color */
            QFrame[class="input-frame"] {{
                background-color: {colors['surface']};
                border: 1px solid {colors['border']};
                border-top: none;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }}
            
            /* Dark label text styling */
            QLabel[class="label-dark"] {{
                color: {'#FFFFFF' if self.is_dark_mode else '#FFFFFF'};
                font-size: 14px;
                font-weight: 600;
                font-family: 'Segoe UI';
                background-color: transparent;
                border: none;
            }}
            
            /* Input field styling with visible borders */
            QComboBox[class="input-field"] {{
                border: 2px solid {colors['border_light']};
                border-radius: 6px;
                background-color: {'#3A3A54' if self.is_dark_mode else '#F8FAFC'};
                padding: 10px 12px;
                font-size: 14px;
                color: {colors['text_primary']};
                margin: 2px;
            }}
            
            QComboBox[class="input-field"]:focus {{
                border: 2px solid {colors['accent']};
                background-color: {'#404060' if self.is_dark_mode else '#FFFFFF'};
            }}
            
            QComboBox[class="input-field"]:hover {{
                border: 2px solid {colors['accent_hover']};
            }}
            
            /* Dropdown styling for combo box */
            QComboBox[class="input-field"]::drop-down {{
                border: none;
                border-left: 1px solid {colors['border']};
                border-radius: 0px 4px 4px 0px;
                background-color: {colors['secondary']};
                width: 20px;
            }}
            
            QComboBox[class="input-field"]::drop-down:hover {{
                background-color: {colors['accent']};
            }}
            
            QComboBox[class="input-field"]::down-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {colors['text_primary']};
                width: 0px;
                height: 0px;
            }}
            
            QComboBox[class="input-field"] QAbstractItemView {{
                background-color: {colors['surface']};
                color: {colors['text_primary']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                selection-background-color: {colors['accent']};
                outline: none;
            }}
            
            /* List widget styling */
            QListWidget {{
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
            
            /* Button styling */
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
        
        # Combine all styles
        complete_style = style + additional_styles
        self.setStyleSheet(complete_style)

        # Update selected ticker buttons with their individual colors
        self._update_ticker_button_colors()
        
        # Style selected tickers label
        self.selected_tickers_label.setStyleSheet(f"color: {colors['text_primary']}; font-size: 12px; font-weight: 500;")
        
        # Regenerate the visualizations with new theme
        self.update_visualizations()
        logger.debug(f"Applied theme: {'dark' if is_dark_mode else 'light'}")

    def _update_ticker_button_colors(self):
        """Update ticker button colors to maintain their individual plot colors"""
        selected_tickers = list(self.selected_tickers_buttons.keys())
        if not selected_tickers:
            return
            
        ticker_colors = self._get_ticker_colors(selected_tickers)
        
        for ticker, button in self.selected_tickers_buttons.items():
            if ticker in ticker_colors:
                color_info = ticker_colors[ticker]
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
        """Update the figure and canvas colors to match the theme"""
        if self.is_dark_mode:
            bg_color = '#212121'
        else:
            bg_color = '#f5f5f5'
        
        # Update figure background
        self.chart_fig.patch.set_facecolor(bg_color)
        
        # Update canvas background
        self.chart_canvas.setStyleSheet(f"background-color: {bg_color};")

    def _get_theme_colors(self):
        """Get appropriate colors for current theme"""
        if self.is_dark_mode:
            return {
                'text': '#ffffff',
                'bg': '#212121',
                'surface': '#2b2b2b',
                'grid': '#444444',
                'legend_bg': '#2b2b2b'
            }
        else:
            return {
                'text': '#333333',
                'bg': '#f5f5f5',
                'surface': '#ffffff',
                'grid': '#cccccc',
                'legend_bg': '#ffffff'
            }

    def update_selected_tickers(self):
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

        # Update selected tickers display
        for ticker in list(self.selected_tickers_buttons.keys()):
            if ticker not in selected_tickers:
                button = self.selected_tickers_buttons.pop(ticker)
                button.deleteLater()

        # Use matplotlib's tab10 colormap to match the plot colors
        import matplotlib.pyplot as plt
        colors_map = plt.cm.tab10(np.linspace(0, 1, max(len(selected_tickers), 1)))
        
        for idx, ticker in enumerate(selected_tickers):
            if ticker not in self.selected_tickers_buttons:
                # Get the color from matplotlib's tab10 colormap (same as used in plots)
                plot_color = colors_map[idx] if len(selected_tickers) > 0 else [0.2, 0.4, 0.8, 1.0]
                # Convert to hex color
                hex_color = f"#{int(plot_color[0]*255):02x}{int(plot_color[1]*255):02x}{int(plot_color[2]*255):02x}"
                
                # Create a darker version for hover
                hover_color = f"#{int(plot_color[0]*200):02x}{int(plot_color[1]*200):02x}{int(plot_color[2]*200):02x}"
                
                button = QPushButton(ticker)
                button.setFixedHeight(25)
                button.clicked.connect(lambda _, t=ticker: self.remove_ticker(t))
                
                # Set color to match the plot line color
                button_style = f"""
                    QPushButton {{
                        background-color: {hex_color}; 
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

    def _get_ticker_colors(self, tickers):
        """Get consistent colors for tickers that match the plot colors"""
        import matplotlib.pyplot as plt
        if not tickers:
            return {}
        
        colors_map = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
        ticker_colors = {}
        
        for idx, ticker in enumerate(tickers):
            plot_color = colors_map[idx]
            hex_color = f"#{int(plot_color[0]*255):02x}{int(plot_color[1]*255):02x}{int(plot_color[2]*255):02x}"
            ticker_colors[ticker] = {
                'hex': hex_color,
                'rgb': plot_color[:3],
                'hover': f"#{int(plot_color[0]*200):02x}{int(plot_color[1]*200):02x}{int(plot_color[2]*200):02x}"
            }
        
        return ticker_colors

    def remove_ticker(self, ticker):
        for item in self.ticker_list.findItems(ticker, Qt.MatchFlag.MatchExactly):
            item.setSelected(False)
        self.update_selected_tickers()

    def clear_all_tickers(self):
        self.ticker_list.clearSelection()
        self.update_selected_tickers()
        logger.debug("Cleared all tickers")

    def get_message_box_style(self):
        return (
            f"QMessageBox {{ background-color: {'#212121' if self.is_dark_mode else '#f5f5f5'}; color: {'#ffffff' if self.is_dark_mode else '#333333'}; }}"
            f"QMessageBox QLabel {{ color: {'#ffffff' if self.is_dark_mode else '#333333'}; }}"
            f"QPushButton {{ background-color: #2196F3; color: #ffffff; border: none; padding: 5px; border-radius: 3px; }}"
            f"QPushButton:hover {{ background-color: #1976D2; }}"
        )

    def update_visualizations(self):
        graph_type = self.graph_combo.currentText()
        selected_tickers = [item.text() for item in self.ticker_list.selectedItems()]

        self.chart_fig.clear()
        
        # Update figure colors for current theme
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
        elif graph_type == "Sharpe Distribution":
            self.plot_sharpe_distribution(selected_tickers)
        elif graph_type == "Predicted vs Actual Sharpe":
            self.plot_time_series_comparison(selected_tickers)
        elif graph_type == "Portfolio Drawdown":
            self.plot_portfolio_drawdown()
        elif graph_type == "Cumulative Returns by Ticker":
            self.plot_cumulative_returns(selected_tickers)
        elif graph_type == "Trade Profit/Loss Distribution":
            self.plot_trade_profit_loss()

        self.chart_canvas.draw()

    def update_dashboard(self):
        self.update_visualizations()
        logger.info("Dashboard updated")

    def plot_portfolio_performance(self):
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
        ax.set_title('Portfolio Performance', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Value ($)', color=theme_colors['text'])
        ax.legend(loc='best', frameon=True, facecolor=theme_colors['legend_bg'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        
        # Set background colors
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_sharpe_distribution(self, selected_tickers):
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

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker]
            sns.histplot(data=ticker_data, x='Best_Prediction', bins=30, ax=ax,
                         color=colors[idx], alpha=0.5, label=f'{ticker} Predicted')

        actual_data = data[data['Actual_Sharpe'] != -1.0]
        if not actual_data.empty:
            for idx, ticker in enumerate(selected_tickers):
                ticker_data = actual_data[actual_data['Ticker'] == ticker]
                sns.histplot(data=ticker_data, x='Actual_Sharpe', bins=15, ax=ax,
                             color=colors[idx], alpha=0.3, label=f'{ticker} Actual', linestyle='--')

        ax.set_title('Sharpe Ratio Distribution', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Sharpe Ratio', color=theme_colors['text'])
        ax.set_ylabel('Count', color=theme_colors['text'])
        ax.legend(loc='best', frameon=True, facecolor=theme_colors['legend_bg'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_time_series_comparison(self, selected_tickers):
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

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker]
            ax.plot(ticker_data['date'], ticker_data['Best_Prediction'],
                    color=colors[idx], alpha=0.7, label=f'{ticker} Predicted', linewidth=1.5)

        actual_data = data[data['Actual_Sharpe'] != -1.0]
        if not actual_data.empty:
            for idx, ticker in enumerate(selected_tickers):
                ticker_data = actual_data[actual_data['Ticker'] == ticker]
                ax.scatter(ticker_data['date'], ticker_data['Actual_Sharpe'],
                           color=colors[idx], s=30, alpha=0.7, label=f'{ticker} Actual')

        ax.set_ylim(-5, 5)
        ax.set_title('Predicted vs Actual Sharpe Ratios', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Sharpe Ratio', color=theme_colors['text'])
        ax.legend(loc='best', frameon=True, facecolor=theme_colors['legend_bg'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_portfolio_drawdown(self):
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

        # Calculate drawdown
        df['peak'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['peak']) / df['peak'] * 100

        ax.plot(df['date'], df['drawdown'], label='Drawdown (%)', color='#F44336', linewidth=2)
        ax.fill_between(df['date'], df['drawdown'], 0, color='#F44336', alpha=0.2)
        ax.set_title('Portfolio Drawdown', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Drawdown (%)', color=theme_colors['text'])
        ax.legend(loc='best', frameon=True, facecolor=theme_colors['legend_bg'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_cumulative_returns(self, selected_tickers):
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

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker].sort_values('date')
            if not ticker_data.empty:
                returns = ticker_data['Close'].pct_change().fillna(0)
                cumulative = (1 + returns).cumprod() * 100 - 100
                ax.plot(ticker_data['date'], cumulative, label=f'{ticker} Returns',
                        color=colors[idx], linewidth=1.5)

        ax.set_title('Cumulative Returns by Ticker', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Date', color=theme_colors['text'])
        ax.set_ylabel('Cumulative Return (%)', color=theme_colors['text'])
        ax.legend(loc='best', frameon=True, facecolor=theme_colors['legend_bg'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()

    def plot_trade_profit_loss(self):
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

        # Calculate profit/loss per trade
        profits = []
        for _, order in orders_df.iterrows():
            if order['action'] == 'buy':
                cost = order['investment_amount'] + order.get('transaction_cost', 0)
                profits.append(-cost)
            elif order['action'] == 'sell':
                proceeds = order['investment_amount'] - order.get('transaction_cost', 0)
                profits.append(proceeds)

        sns.histplot(profits, bins=30, ax=ax, color='#2196F3', alpha=0.7)
        ax.axvline(0, color='#F44336', linestyle='--', label='Break-even')
        ax.set_title('Trade Profit/Loss Distribution', pad=10, color=theme_colors['text'])
        ax.set_xlabel('Profit/Loss ($)', color=theme_colors['text'])
        ax.set_ylabel('Count', color=theme_colors['text'])
        ax.legend(loc='best', frameon=True, facecolor=theme_colors['legend_bg'])
        ax.grid(True, linestyle='--', alpha=0.5, color=theme_colors['grid'])
        ax.set_facecolor(theme_colors['surface'])
        
        self.chart_fig.tight_layout()