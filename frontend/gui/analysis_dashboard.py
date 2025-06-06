from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QGroupBox, QListWidget, QListWidgetItem)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from backend.trading_logic import get_portfolio_history, get_orders
import logging
import numpy as np
import seaborn as sns

# Suppress matplotlib font logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

class AnalysisDashboard(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.is_dark_mode = True  # Default to dark mode
        plt.style.use('dark_background' if self.is_dark_mode else 'default')
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Controls layout (graph selection and ticker selection side-by-side)
        controls_layout = QHBoxLayout()
        
        # Graph selection dropdown
        self.selection_group = QGroupBox("Graph Selection")
        self.selection_group.setObjectName("selection_group")
        self.selection_group.setStyleSheet("color: #ffffff; background-color: #353535;")
        selection_layout = QVBoxLayout()
        
        self.graph_combo = QComboBox()
        self.graph_combo.addItems([
            "Portfolio Performance", 
            "Daily vs Periodic Sharpe", 
            "Sharpe Distribution", 
            "Portfolio Composition",
            "Scatter Plot with Regression Line",
            "Time Series Comparison",
            "Prediction Error Distribution",
            "Performance by Stock",
            "Heatmap of Ensemble Performance",
            "Portfolio vs Max Possible Value",
            "Sharpe Accuracy"
        ])
        self.graph_combo.setStyleSheet("background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;")
        self.graph_combo.currentIndexChanged.connect(self.change_graph_type)
        
        selection_layout.addWidget(self.graph_combo)
        self.selection_group.setLayout(selection_layout)
        controls_layout.addWidget(self.selection_group)
        
        # Ticker selection list
        self.ticker_group = QGroupBox("Ticker Selection (Select up to 5)")
        self.ticker_group.setObjectName("ticker_group")
        self.ticker_group.setStyleSheet("color: #ffffff; background-color: #353535;")
        ticker_layout = QVBoxLayout()
        
        self.ticker_list = QListWidget()
        self.ticker_list.setSelectionMode(QListWidget.MultiSelection)
        self.ticker_list.setMaximumWidth(150)  # Limit width to give more space to graph
        tickers = sorted(self.data_manager.data['Ticker'].unique())
        for ticker in tickers:
            item = QListWidgetItem(ticker)
            self.ticker_list.addItem(item)
        self.ticker_list.setStyleSheet("background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;")
        self.ticker_list.itemSelectionChanged.connect(self.update_visualizations)
        
        ticker_layout.addWidget(self.ticker_list)
        self.ticker_group.setLayout(ticker_layout)
        controls_layout.addWidget(self.ticker_group)
        
        layout.addLayout(controls_layout)
        
        # Chart area
        self.chart_fig = Figure(figsize=(10, 6))
        self.chart_canvas = FigureCanvas(self.chart_fig)
        layout.addWidget(self.chart_canvas, stretch=1)  # Stretch to prioritize graph
        
        # Metrics display
        metrics_layout = QHBoxLayout()
        self.metrics_labels = {}
        for metric in ['Total Value', 'Sharpe Ratio', 'Volatility']:
            label = QLabel(f"{metric}: --")
            label.setObjectName(f"metric_{metric.lower().replace(' ', '_')}")
            self.metrics_labels[metric] = label
            metrics_layout.addWidget(label)
            
        for label in self.metrics_labels.values():
            label.setStyleSheet("color: #ffffff; font-weight: bold; padding: 8px; background-color: #3c3f41; border-radius: 4px;")
            
        layout.addLayout(metrics_layout)
        
    def set_initial_capital(self, initial_capital):
        """Update the initial capital and refresh visualizations."""
        self.initial_capital = initial_capital
        self.update_visualizations()
        
    def change_graph_type(self, index):
        self.update_visualizations()
    
    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the dashboard."""
        self.is_dark_mode = is_dark_mode
        plt.style.use('dark_background' if is_dark_mode else 'default')
        if is_dark_mode:
            group_style = "color: #ffffff; background-color: #353535;"
            combo_style = "background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;"
            list_style = "background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;"
            label_style = "color: #ffffff; font-weight: bold; padding: 8px; background-color: #3c3f41; border-radius: 4px;"
        else:
            group_style = "color: black; background-color: #f0f0f0;"
            combo_style = "background-color: #ffffff; color: black; selection-background-color: #2a82da;"
            list_style = "background-color: #ffffff; color: black; selection-background-color: #2a82da;"
            label_style = "color: black; font-weight: bold; padding: 8px; background-color: #f0f0f0; border-radius: 4px;"
        
        self.selection_group.setStyleSheet(group_style)
        self.ticker_group.setStyleSheet(group_style)
        self.graph_combo.setStyleSheet(combo_style)
        self.ticker_list.setStyleSheet(list_style)
        for label in self.metrics_labels.values():
            label.setStyleSheet(label_style)
        self.update_visualizations()
    
    def update_visualizations(self):
        graph_type = self.graph_combo.currentText()
        selected_items = self.ticker_list.selectedItems()
        selected_tickers = [item.text() for item in selected_items]
        
        self.chart_fig.clear()
        
        try:
            self.start_date = pd.to_datetime(self.data_manager.start_date, utc=True)
            self.end_date = pd.to_datetime(self.data_manager.end_date, utc=True)
            logging.debug(f"Using date range: {self.start_date} to {self.end_date}")
        except Exception as e:
            logging.error(f"Error accessing dates: {str(e)}")
            self.chart_fig.add_subplot(111).text(0.5, 0.5, 'Error: Invalid date range', 
                                                 horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_canvas.draw()
            return
        
        if graph_type == "Portfolio Performance":
            self.plot_portfolio_performance()
        elif graph_type == "Daily vs Periodic Sharpe":
            self.plot_sharpe_comparison(selected_tickers)
        elif graph_type == "Sharpe Distribution":
            self.plot_sharpe_distribution(selected_tickers)
        elif graph_type == "Portfolio Composition":
            self.plot_portfolio_composition()
        elif graph_type == "Scatter Plot with Regression Line":
            self.plot_scatter_with_regression(selected_tickers)
        elif graph_type == "Time Series Comparison":
            self.plot_time_series_comparison(selected_tickers)
        elif graph_type == "Prediction Error Distribution":
            self.plot_error_distribution(selected_tickers)
        elif graph_type == "Performance by Stock":
            self.plot_performance_by_stock(selected_tickers)
        elif graph_type == "Heatmap of Ensemble Performance":
            self.plot_ensemble_heatmap(selected_tickers)
        elif graph_type == "Portfolio vs Max Possible Value":
            self.plot_portfolio_vs_max_possible()
        elif graph_type == "Sharpe Accuracy":
            self.plot_sharpe_accuracy(selected_tickers)
        
        self.update_metrics()
        self.chart_canvas.draw()
        
    def update_dashboard(self):
        """Refresh the dashboard with the latest data."""
        self.update_visualizations()
        
    def plot_portfolio_performance(self):
        ax = self.chart_fig.add_subplot(111)
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return
        
        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        ax.plot(df['date'], df['value'], label='Portfolio Value', color='#2a82da')
        ax.set_title('Portfolio Performance Over Time', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_xlabel('Date', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_ylabel('Value ($)', color='#ffffff' if self.is_dark_mode else 'black')
        ax.legend()
        ax.grid(True, color='#444444' if self.is_dark_mode else '#cccccc')
        ax.tick_params(colors='#ffffff' if self.is_dark_mode else 'black')
        self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
        ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
        
    def plot_portfolio_composition(self):
        ax = self.chart_fig.add_subplot(111)
        orders = get_orders()
        if not orders:
            ax.text(0.5, 0.5, 'No trade history available', horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        orders_df = pd.DataFrame(orders)
        orders_df['date'] = pd.to_datetime(orders_df['date'], utc=True)
        orders_df = orders_df[(orders_df['date'] >= self.start_date) & (orders_df['date'] <= self.end_date)]

        if orders_df.empty:
            ax.text(0.5, 0.5, 'No trades in the selected date range', horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        buy_count = len(orders_df[orders_df['action'] == 'buy'])
        sell_count = len(orders_df[orders_df['action'] == 'sell'])
        total_actions = buy_count + sell_count
        if total_actions == 0:
            ax.text(0.5, 0.5, 'No buy or sell actions recorded', horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        trading_days = (self.end_date - self.start_date).days + 1
        hold_count = trading_days - total_actions
        if hold_count < 0:
            hold_count = 0

        labels = ['Buy', 'Sell', 'Hold']
        sizes = [buy_count, sell_count, hold_count]
        colors = ['#4CAF50', '#F44336', '#2196F3']
        non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        
        if non_zero:
            labels, sizes, colors = zip(*non_zero)
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
            ax.set_title('Trade Action Distribution', color='#ffffff' if self.is_dark_mode else 'black')
        else:
            ax.text(0.5, 0.5, 'No trade actions to display', horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
        self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
        ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
        
    def plot_portfolio_vs_max_possible(self):
        ax = self.chart_fig.add_subplot(111)
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]

        if df.empty:
            ax.text(0.5, 0.5, 'No portfolio history in the selected date range', horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if data.empty:
            ax.text(0.5, 0.5, 'No data available for the date range', horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        initial_investment = df['value'].iloc[0] if not df.empty else 10000.0
        max_portfolio_value = initial_investment
        max_values = []
        holdings = {}
        cash = initial_investment
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')

        for current_date in date_range:
            daily_data = data[data['date'].dt.date == current_date.date()].copy()
            if daily_data.empty:
                max_values.append(max_portfolio_value)
                continue

            daily_data.loc[:, 'Return'] = daily_data.groupby('Ticker')['Close'].pct_change().shift(-1)

            for ticker in list(holdings.keys()):
                holding = holdings[ticker]
                days_held = (current_date - holding['purchase_date']).days
                if days_held >= 5:
                    ticker_data = daily_data[daily_data['Ticker'] == ticker]
                    if not ticker_data.empty:
                        sale_value = holding['shares'] * ticker_data.iloc[0]['Close']
                        cash += sale_value
                        del holdings[ticker]

            if not daily_data.empty:
                daily_data = daily_data.dropna(subset=['Return'])
                if not daily_data.empty:
                    top_stock = daily_data.loc[daily_data['Return'].idxmax()]
                    price = top_stock['Close']
                    shares = int((initial_investment * 0.1) / price)
                    cost = shares * price
                    if shares > 0 and cost <= cash:
                        holdings[top_stock['Ticker']] = {
                            'shares': shares,
                            'purchase_date': current_date
                        }
                        cash -= cost

            current_value = cash
            for ticker, holding in holdings.items():
                ticker_data = daily_data[daily_data['Ticker'] == ticker]
                if not ticker_data.empty:
                    current_value += holding['shares'] * ticker_data.iloc[0]['Close']
            max_portfolio_value = current_value
            max_values.append(max_portfolio_value)

        ax.plot(df['date'], df['value'], label='Actual Portfolio Value', color='#2a82da')
        ax.plot(date_range, max_values, label='Max Possible Value', color='#ff9900', linestyle='--')
        ax.set_title('Portfolio Value vs Maximum Possible Value', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_xlabel('Date', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_ylabel('Value ($)', color='#ffffff' if self.is_dark_mode else 'black')
        ax.legend()
        ax.grid(True, color='#444444' if self.is_dark_mode else '#cccccc')
        ax.tick_params(colors='#ffffff' if self.is_dark_mode else 'black')
        self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
        ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
        
    def update_metrics(self):
        data = self.data_manager.data
        try:
            portfolio_history = get_portfolio_history()
            if portfolio_history:
                total_value = portfolio_history[-1]['value']
            else:
                total_value = data['Close'].iloc[-1]
            sharpe_ratio = data[data['Actual_Sharpe'] != -1.0]['Best_Prediction'].mean()
            if portfolio_history:
                df = pd.DataFrame(portfolio_history)
                volatility = df['value'].pct_change().std()
            else:
                volatility = data['Close'].pct_change().std()
            self.metrics_labels['Total Value'].setText(f"Total Value: ${total_value:,.2f}")
            self.metrics_labels['Sharpe Ratio'].setText(f"Avg Sharpe: {sharpe_ratio:.2f}")
            self.metrics_labels['Volatility'].setText(f"Volatility: {volatility:.2%}")
        except Exception as e:
            for label in self.metrics_labels.values():
                label.setText(f"Error: {str(e)}")
                
    def plot_sharpe_comparison(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker]
            ax.plot(ticker_data['date'], ticker_data['Best_Prediction'], 
                    color=colors[idx], alpha=0.7, label=f'{ticker} Predicted')

        periodic_data = data[data['Actual_Sharpe'] != -1.0]
        if not periodic_data.empty:
            for idx, ticker in enumerate(selected_tickers):
                ticker_data = periodic_data[periodic_data['Ticker'] == ticker]
                ax.scatter(ticker_data['date'], ticker_data['Actual_Sharpe'], 
                           color=colors[idx], s=50, alpha=0.7, marker='o', 
                           label=f'{ticker} Actual')

        ax.set_title('Daily Predicted vs Periodic Actual Sharpe Ratios', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_xlabel('Date', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_ylabel('Sharpe Ratio', color='#ffffff' if self.is_dark_mode else 'black')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444' if self.is_dark_mode else '#cccccc')
        ax.tick_params(colors='#ffffff' if self.is_dark_mode else 'black')
        self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
        ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
        self.chart_fig.tight_layout()
        
    def plot_sharpe_distribution(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]
        
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
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

        ax.set_title('Distribution of Sharpe Ratios', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_xlabel('Sharpe Ratio', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_ylabel('Count', color='#ffffff' if self.is_dark_mode else 'black')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444' if self.is_dark_mode else '#cccccc')
        ax.tick_params(colors='#ffffff' if self.is_dark_mode else 'black')
        self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
        ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
        self.chart_fig.tight_layout()
        
    def plot_scatter_with_regression(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        filtered_data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0].copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'], utc=True)
        filtered_data = filtered_data[(filtered_data['date'] >= self.start_date) & (filtered_data['date'] <= self.end_date)]
        
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        filtered_data = filtered_data[filtered_data['Ticker'].isin(selected_tickers)]
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        filtered_data = filtered_data.dropna(subset=['Best_Prediction', 'Actual_Sharpe'])

        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data with Actual Sharpe != -1.0', horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        overall_data = []
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = filtered_data[filtered_data['Ticker'] == ticker]
            overall_data.append(ticker_data)
            sns.scatterplot(x='Best_Prediction', y='Actual_Sharpe', data=ticker_data, ax=ax, 
                            color=colors[idx], alpha=0.7, label=f'{ticker} Data')

        combined_data = pd.concat(overall_data)
        sns.regplot(x='Best_Prediction', y='Actual_Sharpe', data=combined_data, ax=ax, 
                    scatter=False, color='#ffffff' if self.is_dark_mode else '#2a82da', label='Regression Line')

        correlation = combined_data['Actual_Sharpe'].corr(combined_data['Best_Prediction'])

        min_val = min(filtered_data['Best_Prediction'].min(), filtered_data['Actual_Sharpe'].min())
        max_val = max(filtered_data['Best_Prediction'].max(), filtered_data['Actual_Sharpe'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x Line')

        ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=ax.transAxes, 
                color='#ffffff' if self.is_dark_mode else 'black', fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='#353535' if self.is_dark_mode else '#f0f0f0', alpha=0.8))

        ax.set_title('Actual Sharpe vs Predicted Sharpe', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_xlabel('Predicted Sharpe Ratio', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_ylabel('Actual Sharpe Ratio', color='#ffffff' if self.is_dark_mode else 'black')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444' if self.is_dark_mode else '#cccccc')
        ax.tick_params(colors='#ffffff' if self.is_dark_mode else 'black')
        self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
        ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
        self.chart_fig.tight_layout()
        
    def plot_time_series_comparison(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker]
            ax.plot(ticker_data['date'], ticker_data['Best_Prediction'], 
                    color=colors[idx], alpha=0.7, label=f'{ticker} Predicted')

        actual_data = data[data['Actual_Sharpe'] != -1.0]
        if not actual_data.empty:
            for idx, ticker in enumerate(selected_tickers):
                ticker_data = actual_data[actual_data['Ticker'] == ticker]
                ax.scatter(ticker_data['date'], ticker_data['Actual_Sharpe'], 
                           color=colors[idx], s=50, alpha=0.7, 
                           label=f'{ticker} Actual')

        ax.set_ylim(-5, 5)
        ax.set_title('Time Series Comparison of Sharpe Ratios', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_xlabel('Date', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_ylabel('Sharpe Ratio', color='#ffffff' if self.is_dark_mode else 'black')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444' if self.is_dark_mode else '#cccccc')
        ax.tick_params(colors='#ffffff' if self.is_dark_mode else 'black')
        self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
        ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
        self.chart_fig.tight_layout()
        
    def plot_error_distribution(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        filtered_data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0].copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'], utc=True)
        filtered_data = filtered_data[(filtered_data['date'] >= self.start_date) & (filtered_data['date'] <= self.end_date)]
        
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        filtered_data = filtered_data[filtered_data['Ticker'].isin(selected_tickers)]
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        for idx, ticker in enumerate(selected_tickers):
            ticker_data = filtered_data[filtered_data['Ticker'] == ticker]
            errors = ticker_data['Best_Prediction'] - ticker_data['Actual_Sharpe']
            ax.hist(errors, bins=30, color=colors[idx], alpha=0.5, label=f'{ticker} Errors')

        ax.axvline(0, color='#ff9900', linestyle='--', label='Zero Error')
        ax.set_title('Distribution of Prediction Errors', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_xlabel('Prediction Error', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_ylabel('Frequency', color='#ffffff' if self.is_dark_mode else 'black')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444' if self.is_dark_mode else '#cccccc')
        ax.tick_params(colors='#ffffff' if self.is_dark_mode else 'black')
        self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
        ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
        self.chart_fig.tight_layout()
        
    def plot_performance_by_stock(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        filtered_data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0].copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'], utc=True)
        filtered_data = filtered_data[(filtered_data['date'] >= self.start_date) & (filtered_data['date'] <= self.end_date)]
        
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        filtered_data = filtered_data[filtered_data['Ticker'].isin(selected_tickers)]
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        performance = filtered_data.groupby('Ticker').apply(
            lambda x: np.mean(np.abs(x['Best_Prediction'] - x['Actual_Sharpe']))
        ).sort_values(ascending=False)

        performance.plot(kind='bar', ax=ax, color='#2a82da')
        ax.set_title('Average Prediction Error by Stock', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_xlabel('Stock Ticker', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_ylabel('Mean Absolute Error', color='#ffffff' if self.is_dark_mode else 'black')
        ax.grid(True, color='#444444' if self.is_dark_mode else '#cccccc')
        ax.tick_params(colors='#ffffff' if self.is_dark_mode else 'black')
        self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
        ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
        self.chart_fig.tight_layout()
        
    def plot_ensemble_heatmap(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0].copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]
        
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        pred_cols = [col for col in data.columns if col.startswith('Prediction_') or col == 'Best_Prediction']
        if len(pred_cols) < 2:
            errors = data.groupby('Ticker').apply(
                lambda x: np.mean(np.abs(x['Best_Prediction'] - x['Actual_Sharpe']))
            ).sort_values(ascending=False).to_frame(name='Error')
            if errors.empty:
                ax.text(0.5, 0.5, 'No error data to plot', horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            else:
                sns.heatmap(errors, ax=ax, cmap='viridis', annot=True, fmt='.2f')
                ax.set_title('Prediction Error by Stock', color='#ffffff' if self.is_dark_mode else 'black')
        else:
            performance = {}
            for method in pred_cols:
                errors = np.mean(np.abs(data[method] - data['Actual_Sharpe']))
                performance[method] = [errors]
            performance_matrix = pd.DataFrame(performance, index=['Mean Error'])
            sns.heatmap(performance_matrix, ax=ax, cmap='viridis', annot=True, fmt='.2f')
            ax.set_title('Ensemble Method Performance', color='#ffffff' if self.is_dark_mode else 'black')

        ax.tick_params(colors='#ffffff' if self.is_dark_mode else 'black')
        self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
        ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
        self.chart_fig.tight_layout()
        
    def plot_sharpe_accuracy(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='#ffffff' if self.is_dark_mode else 'black')
            self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
            ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
            return

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        actual_data = data[data['Actual_Sharpe'] != -1.0]
        if not actual_data.empty:
            for idx, ticker in enumerate(selected_tickers):
                ticker_data = actual_data[actual_data['Ticker'] == ticker]
                ax.plot(ticker_data['date'], ticker_data['Actual_Sharpe'], 
                        color=colors[idx], alpha=0.7, label=f'{ticker} Actual')

        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker]
            ax.scatter(ticker_data['date'], ticker_data['Best_Prediction'], 
                       color=colors[idx], s=50, alpha=0.7, label=f'{ticker} Predicted')

        ax.set_title('Predicted vs Actual Sharpe Ratios Over Time', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_xlabel('Date', color='#ffffff' if self.is_dark_mode else 'black')
        ax.set_ylabel('Sharpe Ratio', color='#ffffff' if self.is_dark_mode else 'black')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444' if self.is_dark_mode else '#cccccc')
        ax.tick_params(colors='#ffffff' if self.is_dark_mode else 'black')
        self.chart_fig.patch.set_facecolor('#353535' if self.is_dark_mode else '#ffffff')
        ax.set_facecolor('#2b2b2b' if self.is_dark_mode else '#ffffff')
        self.chart_fig.tight_layout()