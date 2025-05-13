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
        plt.style.use('dark_background')
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Graph selection dropdown
        selection_group = QGroupBox("Graph Selection")
        selection_group.setStyleSheet("color: #ffffff;")
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
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Ticker selection list
        ticker_group = QGroupBox("Ticker Selection (Select up to 5)")
        ticker_group.setStyleSheet("color: #ffffff;")
        ticker_layout = QVBoxLayout()
        
        self.ticker_list = QListWidget()
        self.ticker_list.setSelectionMode(QListWidget.MultiSelection)
        # Populate with unique tickers from data_manager
        tickers = sorted(self.data_manager.data['Ticker'].unique())
        for ticker in tickers:
            item = QListWidgetItem(ticker)
            self.ticker_list.addItem(item)
        self.ticker_list.setStyleSheet("background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;")
        self.ticker_list.itemSelectionChanged.connect(self.update_visualizations)
        
        ticker_layout.addWidget(self.ticker_list)
        ticker_group.setLayout(ticker_layout)
        layout.addWidget(ticker_group)
        
        # Chart area
        self.chart_fig = Figure(figsize=(10, 6))
        self.chart_canvas = FigureCanvas(self.chart_fig)
        layout.addWidget(self.chart_canvas)
        
        # Metrics display
        metrics_layout = QHBoxLayout()
        self.metrics_labels = {}
        for metric in ['Total Value', 'Sharpe Ratio', 'Volatility']:
            label = QLabel(f"{metric}: --")
            self.metrics_labels[metric] = label
            metrics_layout.addWidget(label)
            
        for label in self.metrics_labels.values():
            label.setStyleSheet("color: white; font-weight: bold; padding: 8px; background-color: #3c3f41; border-radius: 4px;")
            
        layout.addLayout(metrics_layout)
        
    def set_initial_capital(self, initial_capital):
        """Update the initial capital and refresh visualizations."""
        self.initial_capital = initial_capital
        self.update_visualizations()
        
    def change_graph_type(self, index):
        self.update_visualizations()
    
    def update_visualizations(self):
        graph_type = self.graph_combo.currentText()
        selected_items = self.ticker_list.selectedItems()
        selected_tickers = [item.text() for item in selected_items]
        
        self.chart_fig.clear()
        
        # Ensure start_date and end_date are properly set
        try:
            self.start_date = pd.to_datetime(self.data_manager.start_date, utc=True)
            self.end_date = pd.to_datetime(self.data_manager.end_date, utc=True)
            logging.debug(f"Using date range: {self.start_date} to {self.end_date}")
        except Exception as e:
            logging.error(f"Error accessing dates: {str(e)}")
            self.chart_fig.add_subplot(111).text(0.5, 0.5, 'Error: Invalid date range', 
                                                 horizontalalignment='center', color='white')
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
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center', color='white')
            return
        
        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        ax.plot(df['date'], df['value'], label='Portfolio Value', color='#2a82da')
        ax.set_title('Portfolio Performance Over Time', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Value ($)', color='white')
        ax.legend()
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        
    def plot_portfolio_composition(self):
        ax = self.chart_fig.add_subplot(111)
        orders = get_orders()
        if not orders:
            ax.text(0.5, 0.5, 'No trade history available', horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            return

        orders_df = pd.DataFrame(orders)
        orders_df['date'] = pd.to_datetime(orders_df['date'], utc=True)
        logging.debug(f"Orders date range: {orders_df['date'].min()} to {orders_df['date'].max()}")
        orders_df = orders_df[(orders_df['date'] >= self.start_date) & (orders_df['date'] <= self.end_date)]

        if orders_df.empty:
            ax.text(0.5, 0.5, 'No trades in the selected date range', horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            return

        buy_count = len(orders_df[orders_df['action'] == 'buy'])
        sell_count = len(orders_df[orders_df['action'] == 'sell'])
        total_actions = buy_count + sell_count
        if total_actions == 0:
            ax.text(0.5, 0.5, 'No buy or sell actions recorded', horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
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
            ax.set_title('Trade Action Distribution', color='white')
        else:
            ax.text(0.5, 0.5, 'No trade actions to display', horizontalalignment='center', color='white')
        self.chart_fig.patch.set_facecolor('#353535')
        
    def plot_portfolio_vs_max_possible(self):
        ax = self.chart_fig.add_subplot(111)
        portfolio_history = get_portfolio_history()
        if not portfolio_history:
            ax.text(0.5, 0.5, 'No portfolio history available', horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            return

        # Actual portfolio value
        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        logging.debug(f"Portfolio history date range: {df['date'].min()} to {df['date'].max()}")

        if df.empty:
            ax.text(0.5, 0.5, 'No portfolio history in the selected date range', horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            return

        # Calculate maximum possible portfolio value
        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        # Filter data to the specified date range
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]
        logging.debug(f"Data date range after filtering: {data['date'].min()} to {data['date'].max()}")

        if data.empty:
            ax.text(0.5, 0.5, 'No data available for the date range', horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            return

        # Initialize simulation
        initial_investment = df['value'].iloc[0] if not df.empty else 10000.0
        max_portfolio_value = initial_investment
        max_values = []
        holdings = {}
        cash = initial_investment
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        logging.debug(f"Simulation date range: {date_range[0]} to {date_range[-1]}")

        for current_date in date_range:
            # Filter data for the current date
            daily_data = data[data['date'].dt.date == current_date.date()].copy()
            logging.debug(f"Date {current_date.date()}: {len(daily_data)} records available")

            # Skip if no data is available for this date
            if daily_data.empty:
                max_values.append(max_portfolio_value)
                continue

            # Calculate daily returns for potential buys
            daily_data.loc[:, 'Return'] = daily_data.groupby('Ticker')['Close'].pct_change().shift(-1)

            # Sell holdings if held for at least 5 days
            for ticker in list(holdings.keys()):
                holding = holdings[ticker]
                days_held = (current_date - holding['purchase_date']).days
                if days_held >= 5:
                    ticker_data = daily_data[daily_data['Ticker'] == ticker]
                    if not ticker_data.empty:
                        sale_value = holding['shares'] * ticker_data.iloc[0]['Close']
                        cash += sale_value
                        logging.debug(f"Sold {ticker} on {current_date.date()}: {holding['shares']} shares for ${sale_value:.2f}")
                        del holdings[ticker]

            # Buy top stock if possible
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
                        logging.debug(f"Bought {top_stock['Ticker']} on {current_date.date()}: {shares} shares at ${price:.2f}")

            # Calculate current portfolio value
            current_value = cash
            for ticker, holding in holdings.items():
                ticker_data = daily_data[daily_data['Ticker'] == ticker]
                if not ticker_data.empty:
                    current_value += holding['shares'] * ticker_data.iloc[0]['Close']
            max_portfolio_value = current_value
            max_values.append(max_portfolio_value)
            logging.debug(f"Date {current_date.date()}: Max portfolio value = ${max_portfolio_value:.2f}, Cash = ${cash:.2f}, Holdings = {holdings}")

        # Plot the results
        ax.plot(df['date'], df['value'], label='Actual Portfolio Value', color='#2a82da')
        ax.plot(date_range, max_values, label='Max Possible Value', color='#ff9900', linestyle='--')
        ax.set_title('Portfolio Value vs Maximum Possible Value', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Value ($)', color='white')
        ax.legend()
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        
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

        # Check if any tickers are selected
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Limit to 5 tickers
        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Filter data for selected tickers
        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Define colors for different tickers
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        # Plot raw daily Best_Prediction for each ticker
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker]
            ax.plot(ticker_data['date'], ticker_data['Best_Prediction'], 
                    color=colors[idx], alpha=0.7, label=f'{ticker} Predicted')

        # Plot raw periodic Actual_Sharpe where available
        periodic_data = data[data['Actual_Sharpe'] != -1.0]
        if not periodic_data.empty:
            for idx, ticker in enumerate(selected_tickers):
                ticker_data = periodic_data[periodic_data['Ticker'] == ticker]
                ax.scatter(ticker_data['date'], ticker_data['Actual_Sharpe'], 
                           color=colors[idx], s=50, alpha=0.7, marker='o', 
                           label=f'{ticker} Actual')

        ax.set_title('Daily Predicted vs Periodic Actual Sharpe Ratios (Raw)', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Sharpe Ratio', color='white')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        self.chart_fig.tight_layout()
        
    def plot_sharpe_distribution(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]
        
        # Check if any tickers are selected
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Limit to 5 tickers
        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Filter data for selected tickers
        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Define colors for different tickers
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        # Plot histogram for Best_Prediction for each ticker
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker]
            sns.histplot(data=ticker_data, x='Best_Prediction', bins=30, ax=ax, 
                         color=colors[idx], alpha=0.5, label=f'{ticker} Predicted')

        # Plot histogram for Actual_Sharpe where available
        actual_data = data[data['Actual_Sharpe'] != -1.0]
        if not actual_data.empty:
            for idx, ticker in enumerate(selected_tickers):
                ticker_data = actual_data[actual_data['Ticker'] == ticker]
                sns.histplot(data=ticker_data, x='Actual_Sharpe', bins=15, ax=ax, 
                             color=colors[idx], alpha=0.3, label=f'{ticker} Actual', linestyle='--')

        ax.set_title('Distribution of Sharpe Ratios', color='white')
        ax.set_xlabel('Sharpe Ratio', color='white')
        ax.set_ylabel('Count', color='white')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        self.chart_fig.tight_layout()
        
    def plot_scatter_with_regression(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        filtered_data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0].copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'], utc=True)
        filtered_data = filtered_data[(filtered_data['date'] >= self.start_date) & (filtered_data['date'] <= self.end_date)]
        
        # Check if any tickers are selected
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Limit to 5 tickers
        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Filter data for selected tickers
        filtered_data = filtered_data[filtered_data['Ticker'].isin(selected_tickers)]
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Remove rows with missing values in Best_Prediction or Actual_Sharpe
        filtered_data = filtered_data.dropna(subset=['Best_Prediction', 'Actual_Sharpe'])

        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data with Actual Sharpe != -1.0', horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Define colors for different tickers
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        # Create scatter plot for each ticker
        overall_data = []
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = filtered_data[filtered_data['Ticker'] == ticker]
            overall_data.append(ticker_data)
            sns.scatterplot(x='Best_Prediction', y='Actual_Sharpe', data=ticker_data, ax=ax, 
                            color=colors[idx], alpha=0.7, label=f'{ticker} Data')

        # Combine data for overall regression line
        combined_data = pd.concat(overall_data)
        sns.regplot(x='Best_Prediction', y='Actual_Sharpe', data=combined_data, ax=ax, 
                    scatter=False, color='white', label='Regression Line')

        # Calculate overall correlation
        correlation = combined_data['Actual_Sharpe'].corr(combined_data['Best_Prediction'])
        logging.debug(f"Correlation coefficient: {correlation:.4f}")

        # Add y=x line for reference
        min_val = min(filtered_data['Best_Prediction'].min(), filtered_data['Actual_Sharpe'].min())
        max_val = max(filtered_data['Best_Prediction'].max(), filtered_data['Actual_Sharpe'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'y--', label='y=x Line')

        # Add correlation coefficient annotation
        ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=ax.transAxes, 
                color='white', fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='#353535', alpha=0.8))

        ax.set_title('Actual Sharpe vs Predicted Sharpe (Raw)', color='white')
        ax.set_xlabel('Predicted Sharpe Ratio (Best_Prediction)', color='white')
        ax.set_ylabel('Actual Sharpe Ratio', color='white')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        self.chart_fig.tight_layout()
        
    def plot_time_series_comparison(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        # Check if any tickers are selected
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Limit to 5 tickers
        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Filter data for selected tickers
        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Define colors for different tickers
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        # Plot raw Best_Prediction for each ticker
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker]
            ax.plot(ticker_data['date'], ticker_data['Best_Prediction'], 
                    color=colors[idx], alpha=0.7, label=f'{ticker} Predicted')

        # Plot raw Actual_Sharpe where available
        actual_data = data[data['Actual_Sharpe'] != -1.0]
        if not actual_data.empty:
            for idx, ticker in enumerate(selected_tickers):
                ticker_data = actual_data[actual_data['Ticker'] == ticker]
                ax.scatter(ticker_data['date'], ticker_data['Actual_Sharpe'], 
                           color=colors[idx], s=50, alpha=0.7, 
                           label=f'{ticker} Actual')

        ax.set_ylim(-5, 5)
        ax.set_title('Time Series Comparison of Sharpe Ratios (Raw)', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Sharpe Ratio', color='white')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        self.chart_fig.tight_layout()
        
    def plot_error_distribution(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        filtered_data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0].copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'], utc=True)
        filtered_data = filtered_data[(filtered_data['date'] >= self.start_date) & (filtered_data['date'] <= self.end_date)]
        
        # Check if any tickers are selected
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Limit to 5 tickers
        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Filter data for selected tickers
        filtered_data = filtered_data[filtered_data['Ticker'].isin(selected_tickers)]
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Define colors for different tickers
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        # Calculate prediction errors and plot histogram for each ticker
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = filtered_data[filtered_data['Ticker'] == ticker]
            errors = ticker_data['Best_Prediction'] - ticker_data['Actual_Sharpe']
            ax.hist(errors, bins=30, color=colors[idx], alpha=0.5, label=f'{ticker} Errors')

        ax.axvline(0, color='yellow', linestyle='--', label='Zero Error')
        ax.set_title('Distribution of Prediction Errors (Raw)', color='white')
        ax.set_xlabel('Prediction Error', color='white')
        ax.set_ylabel('Frequency', color='white')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        self.chart_fig.tight_layout()
        
    def plot_performance_by_stock(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        filtered_data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0].copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'], utc=True)
        filtered_data = filtered_data[(filtered_data['date'] >= self.start_date) & (filtered_data['date'] <= self.end_date)]
        
        # Check if any tickers are selected
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Limit to 5 tickers
        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Filter data for selected tickers
        filtered_data = filtered_data[filtered_data['Ticker'].isin(selected_tickers)]
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Calculate mean absolute error per ticker
        performance = filtered_data.groupby('Ticker').apply(
            lambda x: np.mean(np.abs(x['Best_Prediction'] - x['Actual_Sharpe']))
        ).sort_values(ascending=False)

        performance.plot(kind='bar', ax=ax, color='#2a82da')
        ax.set_title('Average Prediction Error by Stock', color='white')
        ax.set_xlabel('Stock Ticker', color='white')
        ax.set_ylabel('Mean Absolute Error', color='white')
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        self.chart_fig.tight_layout()
        
    def plot_ensemble_heatmap(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0].copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]
        
        # Check if any tickers are selected
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Limit to 5 tickers
        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Filter data for selected tickers
        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Check for ensemble prediction columns
        pred_cols = [col for col in data.columns if col.startswith('Prediction_') or col == 'Best_Prediction']
        if len(pred_cols) < 2:
            # If only Best_Prediction exists, show errors for selected tickers
            errors = data.groupby('Ticker').apply(
                lambda x: np.mean(np.abs(x['Best_Prediction'] - x['Actual_Sharpe']))
            ).sort_values(ascending=False).to_frame(name='Error')
            if errors.empty:
                ax.text(0.5, 0.5, 'No error data to plot', horizontalalignment='center', color='white')
            else:
                sns.heatmap(errors, ax=ax, cmap='viridis', annot=True, fmt='.2f')
                ax.set_title('Prediction Error by Stock (Best_Prediction)', color='white')
        else:
            # Aggregate errors across selected tickers for each method
            performance = {}
            for method in pred_cols:
                errors = np.mean(np.abs(data[method] - data['Actual_Sharpe']))
                performance[method] = [errors]
            performance_matrix = pd.DataFrame(performance, index=['Mean Error'])
            sns.heatmap(performance_matrix, ax=ax, cmap='viridis', annot=True, fmt='.2f')
            ax.set_title('Ensemble Method Performance (Mean Error)', color='white')

        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        self.chart_fig.tight_layout()
        
    def plot_sharpe_accuracy(self, selected_tickers):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        # Check if any tickers are selected
        if not selected_tickers:
            ax.text(0.5, 0.5, 'Please select at least one ticker.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Limit to 5 tickers
        if len(selected_tickers) > 5:
            ax.text(0.5, 0.5, 'Please select 5 or fewer tickers for clarity.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Filter data for selected tickers
        data = data[data['Ticker'].isin(selected_tickers)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data for the selected tickers.', 
                    horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            ax.set_facecolor('#2b2b2b')
            return

        # Define colors for different tickers
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tickers)))

        # Plot raw Best_Prediction for each ticker
        for idx, ticker in enumerate(selected_tickers):
            ticker_data = data[data['Ticker'] == ticker]
            ax.plot(ticker_data['date'], ticker_data['Best_Prediction'], 
                    color=colors[idx], alpha=0.7, label=f'{ticker} Predicted')

        # Plot raw Actual_Sharpe where available
        actual_data = data[data['Actual_Sharpe'] != -1.0]
        if not actual_data.empty:
            for idx, ticker in enumerate(selected_tickers):
                ticker_data = actual_data[actual_data['Ticker'] == ticker]
                ax.scatter(ticker_data['date'], ticker_data['Actual_Sharpe'], 
                           color=colors[idx], s=50, alpha=0.7, 
                           label=f'{ticker} Actual')

        ax.set_title('Predicted vs Actual Sharpe Ratios Over Time (Raw)', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Sharpe Ratio', color='white')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        self.chart_fig.tight_layout()