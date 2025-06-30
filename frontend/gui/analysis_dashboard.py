import matplotlib
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QGroupBox, QListWidget, QListWidgetItem,
                             QPushButton, QMessageBox, QFrame)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from backend.trading_logic_new import get_portfolio_history, get_orders
from ..logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

class AnalysisDashboard(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.is_dark_mode = True
        # Set initial Matplotlib font and style
        self._configure_matplotlib()
        self.setup_ui()
        logger.info("AnalysisDashboard initialized")

    def _configure_matplotlib(self):
        """Configure Matplotlib with available fonts and theme settings."""
        available_fonts = set(f.lower() for f in matplotlib.font_manager.findSystemFonts())
        font_priority = ['Segoe UI', 'Arial', 'DejaVu Sans', 'sans-serif']
        font_family = next((f for f in font_priority if f.lower() in available_fonts or f == 'sans-serif'), 'sans-serif')
        plt.rcParams.update({
            'font.family': font_family,
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
        })
        self.update_matplotlib_theme()
        logger.debug(f"Matplotlib configured with font: {font_family}")

    def setup_ui(self):
        """Initialize the UI components with modern design."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Controls card
        controls_card = QFrame()
        controls_card.setObjectName("controls_card")
        controls_layout = QHBoxLayout(controls_card)
        controls_layout.setSpacing(15)
        controls_layout.setContentsMargins(15, 15, 15, 15)

        # Graph selection
        self.selection_group = QGroupBox("Graph Type")
        selection_layout = QVBoxLayout()
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
        selection_layout.addWidget(self.graph_combo)
        self.selection_group.setLayout(selection_layout)
        controls_layout.addWidget(self.selection_group, stretch=1)

        # Ticker selection
        self.ticker_group = QGroupBox("Tickers")
        ticker_layout = QVBoxLayout()
        ticker_layout.setSpacing(10)

        # Selected tickers display
        self.selected_tickers_layout = QHBoxLayout()
        self.selected_tickers_label = QLabel("Selected Tickers:")
        self.selected_tickers_layout.addWidget(self.selected_tickers_label)
        self.selected_tickers_buttons = {}
        self.selected_tickers_layout.addStretch()
        self.clear_tickers_button = QPushButton("Clear All")
        self.clear_tickers_button.clicked.connect(self.clear_all_tickers)
        self.selected_tickers_layout.addWidget(self.clear_tickers_button)
        ticker_layout.addLayout(self.selected_tickers_layout)

        # Ticker list
        self.ticker_list = QListWidget()
        self.ticker_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.ticker_list.setMaximumWidth(200)
        if self.data_manager.data is not None and not self.data_manager.data.empty:
            tickers = sorted(self.data_manager.data['Ticker'].unique())
            for ticker in tickers:
                item = QListWidgetItem(ticker)
                self.ticker_list.addItem(item)
        else:
            logger.warning("No tickers available due to missing market data")
        self.ticker_list.itemSelectionChanged.connect(self.update_selected_tickers)
        ticker_layout.addWidget(self.ticker_list)
        self.ticker_group.setLayout(ticker_layout)
        controls_layout.addWidget(self.ticker_group, stretch=1)

        layout.addWidget(controls_card)

        # Chart area
        self.chart_fig = Figure(figsize=(10, 6))
        self.chart_canvas = FigureCanvas(self.chart_fig)
        layout.addWidget(self.chart_canvas, stretch=3)

        # Metrics card
        metrics_card = QFrame()
        metrics_card.setObjectName("metrics_card")
        metrics_layout = QHBoxLayout(metrics_card)
        metrics_layout.setSpacing(15)
        metrics_layout.setContentsMargins(15, 15, 15, 15)
        self.metrics_labels = {}
        for metric in ['Total Value', 'Sharpe Ratio', 'Volatility']:
            label = QLabel(f"{metric}: --")
            label.setObjectName(f"metric_{metric.lower().replace(' ', '_')}")
            self.metrics_labels[metric] = label
            metrics_layout.addWidget(label)
        layout.addWidget(metrics_card)

        self.set_theme(self.is_dark_mode)
        self.update_visualizations()

    def update_matplotlib_theme(self):
        """Update Matplotlib theme based on current mode."""
        plt.style.use('dark_background' if self.is_dark_mode else 'seaborn-v0_8-white')
        plt.rcParams.update({
            'axes.facecolor': '#2D2D2D' if self.is_dark_mode else '#FFFFFF',
            'figure.facecolor': '#2D2D2D' if self.is_dark_mode else '#F5F5F5',
            'axes.labelcolor': '#FFFFFF' if self.is_dark_mode else '#2C2C2C',
            'xtick.color': '#FFFFFF' if self.is_dark_mode else '#2C2C2C',
            'ytick.color': '#FFFFFF' if self.is_dark_mode else '#2C2C2C',
            'grid.color': '#555555' if self.is_dark_mode else '#CCCCCC',
            'text.color': '#FFFFFF' if self.is_dark_mode else '#2C2C2C',
            'axes.edgecolor': '#FFFFFF' if self.is_dark_mode else '#2C2C2C',
            'legend.facecolor': '#2D2D2D' if self.is_dark_mode else '#FFFFFF',
        })

    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the dashboard and plots."""
        self.is_dark_mode = is_dark_mode
        self.update_matplotlib_theme()
        card_style = f"""
            QFrame {{
                background-color: {'#3C3F41' if is_dark_mode else '#FFFFFF'};
                border-radius: 8px;
                border: 1px solid {'#555555' if is_dark_mode else '#CCCCCC'};
            }}
        """
        group_style = f"""
            QGroupBox {{
                color: {'#FFFFFF' if is_dark_mode else '#2C2C2C'};
                font-family: 'Segoe UI';
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {'#555555' if is_dark_mode else '#CCCCCC'};
                border-radius: 6px;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }}
        """
        combo_style = f"""
            QComboBox {{
                background-color: {'#3C3F41' if is_dark_mode else '#FFFFFF'};
                color: {'#FFFFFF' if is_dark_mode else '#2C2C2C'};
                border: 1px solid {'#555555' if is_dark_mode else '#CCCCCC'};
                border-radius: 4px;
                padding: 6px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
        """
        list_style = f"""
            QListWidget {{
                background-color: {'#3C3F41' if is_dark_mode else '#FFFFFF'};
                color: {'#FFFFFF' if is_dark_mode else '#2C2C2C'};
                border: 1px solid {'#555555' if is_dark_mode else '#CCCCCC'};
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background-color: #0078D4;
                color: #FFFFFF;
            }}
        """
        label_style = f"""
            QLabel {{
                color: {'#FFFFFF' if is_dark_mode else '#2C2C2C'};
                font-family: 'Segoe UI';
                font-size: 13px;
            }}
        """
        button_style = f"""
            QPushButton {{
                background-color: #0078D4;
                color: #FFFFFF;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-family: 'Segoe UI';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #005BA1;
            }}
            QPushButton:pressed {{
                background-color: #003E7E;
            }}
        """
        self.setStyleSheet(f"background-color: {'#2D2D2D' if is_dark_mode else '#F5F5F5'};")
        self.findChild(QFrame, "controls_card").setStyleSheet(card_style)
        self.findChild(QFrame, "metrics_card").setStyleSheet(card_style)
        self.selection_group.setStyleSheet(group_style)
        self.ticker_group.setStyleSheet(group_style)
        self.graph_combo.setStyleSheet(combo_style)
        self.ticker_list.setStyleSheet(list_style)
        self.selected_tickers_label.setStyleSheet(label_style)
        self.clear_tickers_button.setStyleSheet(button_style)
        for ticker, button in self.selected_tickers_buttons.items():
            button.setStyleSheet(button_style)
        for label in self.metrics_labels.values():
            label.setStyleSheet(f"font-weight: bold; {label_style}")
        self.update_visualizations()
        logger.debug(f"Applied theme: {'dark' if is_dark_mode else 'light'}")

    def get_message_box_style(self):
        """Return stylesheet for QMessageBox based on the current theme."""
        return f"""
            QMessageBox {{
                background-color: {'#2D2D2D' if self.is_dark_mode else '#F5F5F5'};
                color: {'#FFFFFF' if self.is_dark_mode else '#2C2C2C'};
                font-family: 'Segoe UI';
            }}
            QMessageBox QLabel {{
                color: {'#FFFFFF' if self.is_dark_mode else '#2C2C2C'};
            }}
            QMessageBox QPushButton {{
                background-color: #0078D4;
                color: #FFFFFF;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-family: 'Segoe UI';
                font-weight: bold;
            }}
            QMessageBox QPushButton:hover {{
                background-color: #005BA1;
            }}
            QMessageBox QPushButton:pressed {{
                background-color: #003E7E;
            }}
        """

    def set_initial_capital(self, initial_capital):
        """Set the initial capital and update visualizations."""
        self.initial_capital = initial_capital
        self.update_visualizations()
        logger.debug(f"Set initial capital: ${initial_capital:,.2f}")

    def change_graph_type(self, index):
        """Update visualizations when graph type changes."""
        self.update_visualizations()
        logger.debug(f"Changed graph type to: {self.graph_combo.currentText()}")

    def update_selected_tickers(self):
        """Update selected tickers display and visualizations."""
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

        for ticker in selected_tickers:
            if ticker not in self.selected_tickers_buttons:
                button = QPushButton(ticker)
                button.setFixedHeight(30)
                button.clicked.connect(lambda _, t=ticker: self.remove_ticker(t))
                button.setStyleSheet(self.clear_tickers_button.styleSheet())
                self.selected_tickers_layout.insertWidget(self.selected_tickers_layout.count() - 2, button)
                self.selected_tickers_buttons[ticker] = button

        self.update_visualizations()
        logger.debug(f"Selected tickers: {selected_tickers}")

    def remove_ticker(self, ticker):
        """Remove a ticker from selection."""
        for item in self.ticker_list.findItems(ticker, Qt.MatchFlag.MatchExactly):
            item.setSelected(False)
        self.update_selected_tickers()

    def clear_all_tickers(self):
        """Clear all selected tickers."""
        self.ticker_list.clearSelection()
        self.update_selected_tickers()
        logger.debug("Cleared all tickers")

    def update_visualizations(self):
        """Update the dashboard visualizations based on selected graph type."""
        self.chart_fig.clear()
        self.update_matplotlib_theme()  # Ensure theme is applied before plotting
        graph_type = self.graph_combo.currentText()
        selected_tickers = [item.text() for item in self.ticker_list.selectedItems()]
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
            ax = self.chart_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Error: Invalid date range', horizontalalignment='center', verticalalignment='center',
                    color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
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

        self.update_metrics()
        self.chart_canvas.draw()

    def update_dashboard(self):
        """Refresh the dashboard visualizations and metrics."""
        self.update_visualizations()
        logger.info("Dashboard updated")

    def plot_portfolio_performance(self):
        """Plot portfolio value over time."""
        ax = self.chart_fig.add_subplot(111)
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        if df.empty:
            ax.text(0.5, 0.5, 'No portfolio history in date range', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        ax.plot(df['date'], df['value'], label='Portfolio Value', color='#0078D4', linewidth=2)
        ax.set_title('Portfolio Performance', pad=15)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value ($)')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def plot_sharpe_distribution(self, selected_tickers):
        """Plot distribution of predicted and actual Sharpe ratios."""
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy() if self.data_manager.data is not None else None
        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No market data available', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Select at least one ticker', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for selected tickers', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
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

        ax.set_title('Sharpe Ratio Distribution', pad=15)
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Count')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def plot_time_series_comparison(self, selected_tickers):
        """Plot predicted vs actual Sharpe ratios over time."""
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy() if self.data_manager.data is not None else None
        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No market data available', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Select at least one ticker', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for selected tickers', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
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
        ax.set_title('Predicted vs Actual Sharpe Ratios', pad=15)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def plot_portfolio_drawdown(self):
        """Plot portfolio drawdown over time."""
        ax = self.chart_fig.add_subplot(111)
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        if df.empty:
            ax.text(0.5, 0.5, 'No portfolio history in date range', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        df['peak'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['peak']) / df['peak'] * 100

        ax.plot(df['date'], df['drawdown'], label='Drawdown (%)', color='#D32F2F', linewidth=2)
        ax.fill_between(df['date'], df['drawdown'], 0, color='#D32F2F', alpha=0.2)
        ax.set_title('Portfolio Drawdown', pad=15)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def plot_cumulative_returns(self, selected_tickers):
        """Plot cumulative returns by ticker."""
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy() if self.data_manager.data is not None else None
        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No market data available', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Select at least one ticker', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for selected tickers', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker].sort_values('date')
            if not ticker_data.empty:
                returns = ticker_data['Close'].pct_change().fillna(0)
                cumulative = (1 + returns).cumprod() * 100 - 100
                ax.plot(ticker_data['date'], cumulative, label=f'{ticker} Returns',
                        color=colors[idx], linewidth=1.5)

        ax.set_title('Cumulative Returns by Ticker', pad=15)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def plot_trade_profit_loss(self):
        """Plot distribution of trade profits/losses."""
        ax = self.chart_fig.add_subplot(111)
        orders = get_orders()
        if not orders:
            ax.text(0.5, 0.5, 'No trade history available', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        orders_df = pd.DataFrame(orders)
        orders_df['date'] = pd.to_datetime(orders_df['date'], utc=True)
        orders_df = orders_df[(orders_df['date'] >= self.start_date) & (orders_df['date'] <= self.end_date)]

        if orders_df.empty:
            ax.text(0.5, 0.5, 'No trades in date range', horizontalalignment='center',
                    verticalalignment='center', color='#FFFFFF' if self.is_dark_mode else '#2C2C2C')
            return

        profits = []
        for _, order in orders_df.iterrows():
            if order['action'] == 'buy':
                cost = order['investment_amount'] + order.get('transaction_cost', 0)
                profits.append(-cost)
            elif order['action'] == 'sell':
                proceeds = order['investment_amount'] - order.get('transaction_cost', 0)
                profits.append(proceeds)

        sns.histplot(profits, bins=30, ax=ax, color='#0078D4', alpha=0.7)
        ax.axvline(0, color='#D32F2F', linestyle='--', label='Break-even')
        ax.set_title('Trade Profit/Loss Distribution', pad=15)
        ax.set_xlabel('Profit/Loss ($)')
        ax.set_ylabel('Count')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def update_metrics(self):
        """Update financial metrics display."""
        data = self.data_manager.data
        try:
            portfolio_history = get_portfolio_history()
            total_value = portfolio_history[-1]['value'] if portfolio_history else 10000.0
            sharpe_ratio = data[data['Actual_Sharpe'] != -1.0]['Best_Prediction'].mean() if data is not None and not data.empty else 0.0
            volatility = pd.DataFrame(portfolio_history)['value'].pct_change().std() if portfolio_history else 0.0
            self.metrics_labels['Total Value'].setText(f"Total Value: ${total_value:,.2f}")
            self.metrics_labels['Sharpe Ratio'].setText(f"Avg Sharpe: {sharpe_ratio:.2f}")
            self.metrics_labels['Volatility'].setText(f"Volatility: {volatility:.2%}")
            logger.debug(f"Updated metrics: Total=${total_value:,.2f}, Sharpe={sharpe_ratio:.2f}, Volatility={volatility:.2%}")
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            for label in self.metrics_labels.values():
                label.setText(f"Error: {str(e)}")