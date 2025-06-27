import matplotlib
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QGroupBox, QListWidget, QListWidgetItem,
                             QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from backend.trading_logic_new import get_portfolio_history, get_orders
from frontend.logging_config import get_logger


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
        # Set matplotlib style
        plt.rcParams.update({
            'font.family': font_family,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'axes.facecolor': '#2b2b2b' if self.is_dark_mode else '#ffffff',
            'figure.facecolor': '#212121' if self.is_dark_mode else '#f5f5f5'
        })
        logger.debug(f"Using font family: {font_family}")
        self.setup_ui()
        logger.info("AnalysisDashboard initialized")

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Controls layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

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
        ticker_layout.setSpacing(5)

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
        self.selected_tickers_layout.addWidget(self.clear_tickers_button)

        ticker_layout.addLayout(self.selected_tickers_layout)

        # Ticker list
        self.ticker_list = QListWidget()
        self.ticker_list.setSelectionMode(QListWidget.MultiSelection)
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

        layout.addLayout(controls_layout)

        # Chart area
        self.chart_fig = Figure(figsize=(10, 6))
        self.chart_canvas = FigureCanvas(self.chart_fig)
        layout.addWidget(self.chart_canvas, stretch=2)

        # Metrics display
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(10)
        self.metrics_labels = {}
        for metric in ['Total Value', 'Sharpe Ratio', 'Volatility']:
            label = QLabel(f"{metric}: --")
            label.setObjectName(f"metric_{metric.lower().replace(' ', '_')}")
            self.metrics_labels[metric] = label
            metrics_layout.addWidget(label)

        layout.addLayout(metrics_layout)

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
        plt.style.use('dark_background' if is_dark_mode else 'seaborn-v0_8-white')
        plt.rcParams.update({
            'axes.facecolor': '#2b2b2b' if is_dark_mode else '#ffffff',
            'figure.facecolor': '#212121' if is_dark_mode else '#f5f5f5',
            'axes.labelcolor': '#ffffff' if is_dark_mode else '#333333',
            'xtick.color': '#ffffff' if is_dark_mode else '#333333',
            'ytick.color': '#ffffff' if is_dark_mode else '#333333',
            'grid.color': '#444444' if is_dark_mode else '#cccccc'
        })

        if is_dark_mode:
            group_style = "color: #ffffff; background-color: #212121; border: 1px solid #333333; border-radius: 5px;"
            combo_style = "background-color: #2b2b2b; color: #ffffff; selection-background-color: #2196F3; border: 1px solid #333333; border-radius: 3px;"
            list_style = "background-color: #2b2b2b; color: #ffffff; selection-background-color: #2196F3; border: 1px solid #333333; border-radius: 3px;"
            label_style = "color: #ffffff; background-color: #2b2b2b; padding: 8px; border-radius: 5px;"
            button_style = "QPushButton {background-color: #2196F3; color: #ffffff; border: none; padding: 5px; border-radius: 3px;} QPushButton:hover {background-color: #1976D2;}"
        else:
            group_style = "color: #333333; background-color: #f5f5f5; border: 1px solid #cccccc; border-radius: 5px;"
            combo_style = "background-color: #ffffff; color: #333333; selection-background-color: #2196F3; border: 1px solid #cccccc; border-radius: 3px;"
            list_style = "background-color: #ffffff; color: #333333; selection-background-color: #2196F3; border: 1px solid #cccccc; border-radius: 3px;"
            label_style = "color: #333333; background-color: #f5f5f5; padding: 8px; border-radius: 5px;"
            button_style = "QPushButton {background-color: #2196F3; color: #ffffff; border: none; padding: 5px; border-radius: 3px;} QPushButton:hover {background-color: #1976D2;}"

        self.selection_group.setStyleSheet(group_style)
        self.ticker_group.setStyleSheet(group_style)
        self.graph_combo.setStyleSheet(combo_style)
        self.ticker_list.setStyleSheet(list_style)
        self.clear_tickers_button.setStyleSheet(button_style)
        self.selected_tickers_label.setStyleSheet("color: #ffffff;" if is_dark_mode else "color: #333333;")
        for ticker, button in self.selected_tickers_buttons.items():
            button.setStyleSheet(button_style)
        for label in self.metrics_labels.values():
            label.setStyleSheet(label_style)
        self.update_visualizations()
        logger.debug(f"Applied theme: {'dark' if is_dark_mode else 'light'}")

    def update_selected_tickers(self):
        selected_items = self.ticker_list.selectedItems()
        selected_tickers = [item.text() for item in selected_items]

        if len(selected_tickers) > 5:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Selection Limit")
            msg.setText("Please select up to 5 tickers only.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()
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
                button.setFixedHeight(25)
                button.clicked.connect(lambda _, t=ticker: self.remove_ticker(t))
                button.setStyleSheet("QPushButton {background-color: #2196F3; color: #ffffff; border: none; padding: 5px; border-radius: 3px;} QPushButton:hover {background-color: #1976D2;}")
                self.selected_tickers_layout.insertWidget(self.selected_tickers_layout.count()-2, button)
                self.selected_tickers_buttons[ticker] = button

        self.update_visualizations()
        logger.debug(f"Selected tickers: {selected_tickers}")

    def remove_ticker(self, ticker):
        for item in self.ticker_list.findItems(ticker, Qt.MatchExactly):
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
            self.chart_fig.add_subplot(111).text(0.5, 0.5, 'Error: Invalid date range',
                                                 horizontalalignment='center', verticalalignment='center',
                                                 color='#ffffff' if self.is_dark_mode else '#333333')
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
        self.update_visualizations()
        logger.info("Dashboard updated")

    def plot_portfolio_performance(self):
        ax = self.chart_fig.add_subplot(111)
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        if df.empty:
            ax.text(0.5, 0.5, 'No portfolio history in date range', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        ax.plot(df['date'], df['value'], label='Portfolio Value', color='#2196F3', linewidth=2)
        ax.set_title('Portfolio Performance', pad=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value ($)')
        ax.legend(loc='best', frameon=True, facecolor='#2b2b2b' if self.is_dark_mode else '#ffffff')
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def plot_sharpe_distribution(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy() if self.data_manager.data is not None else None
        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No market data available', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Select at least one ticker', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for selected tickers', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
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

        ax.set_title('Sharpe Ratio Distribution', pad=10)
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Count')
        ax.legend(loc='best', frameon=True, facecolor='#2b2b2b' if self.is_dark_mode else '#ffffff')
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def plot_time_series_comparison(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy() if self.data_manager.data is not None else None
        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No market data available', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Select at least one ticker', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for selected tickers', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
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
        ax.set_title('Predicted vs Actual Sharpe Ratios', pad=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend(loc='best', frameon=True, facecolor='#2b2b2b' if self.is_dark_mode else '#ffffff')
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def plot_portfolio_drawdown(self):
        ax = self.chart_fig.add_subplot(111)
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        if df.empty:
            ax.text(0.5, 0.5, 'No portfolio history in date range', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        # Calculate drawdown
        df['peak'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['peak']) / df['peak'] * 100

        ax.plot(df['date'], df['drawdown'], label='Drawdown (%)', color='#F44336', linewidth=2)
        ax.fill_between(df['date'], df['drawdown'], 0, color='#F44336', alpha=0.2)
        ax.set_title('Portfolio Drawdown', pad=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend(loc='best', frameon=True, facecolor='#2b2b2b' if self.is_dark_mode else '#ffffff')
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def plot_cumulative_returns(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy() if self.data_manager.data is not None else None
        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No market data available', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Select at least one ticker', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for selected tickers', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker].sort_values('date')
            if not ticker_data.empty:
                returns = ticker_data['Close'].pct_change().fillna(0)
                cumulative = (1 + returns).cumprod() * 100 - 100
                ax.plot(ticker_data['date'], cumulative, label=f'{ticker} Returns',
                        color=colors[idx], linewidth=1.5)

        ax.set_title('Cumulative Returns by Ticker', pad=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend(loc='best', frameon=True, facecolor='#2b2b2b' if self.is_dark_mode else '#ffffff')
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def plot_trade_profit_loss(self):
        ax = self.chart_fig.add_subplot(111)
        orders = get_orders()
        if not orders:
            ax.text(0.5, 0.5, 'No trade history available', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
            return

        orders_df = pd.DataFrame(orders)
        orders_df['date'] = pd.to_datetime(orders_df['date'], utc=True)
        orders_df = orders_df[(orders_df['date'] >= self.start_date) & (orders_df['date'] <= self.end_date)]

        if orders_df.empty:
            ax.text(0.5, 0.5, 'No trades in date range', horizontalalignment='center',
                    verticalalignment='center', color='#ffffff' if self.is_dark_mode else '#333333')
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
        ax.set_title('Trade Profit/Loss Distribution', pad=10)
        ax.set_xlabel('Profit/Loss ($)')
        ax.set_ylabel('Count')
        ax.legend(loc='best', frameon=True, facecolor='#2b2b2b' if self.is_dark_mode else '#ffffff')
        ax.grid(True, linestyle='--', alpha=0.5)
        self.chart_fig.tight_layout()

    def update_metrics(self):
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
