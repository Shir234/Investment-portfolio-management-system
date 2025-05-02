from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QGroupBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from trading_logic import get_portfolio_history, get_orders
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
            "Aggregated Sharpe Prediction Accuracy"
        ])
        self.graph_combo.setStyleSheet("background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;")
        self.graph_combo.currentIndexChanged.connect(self.change_graph_type)
        
        selection_layout.addWidget(self.graph_combo)
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
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
            self.plot_sharpe_comparison()
        elif graph_type == "Sharpe Distribution":
            self.plot_sharpe_distribution()
        elif graph_type == "Portfolio Composition":
            self.plot_portfolio_composition()
        elif graph_type == "Scatter Plot with Regression Line":
            self.plot_scatter_with_regression()
        elif graph_type == "Time Series Comparison":
            self.plot_time_series_comparison()
        elif graph_type == "Prediction Error Distribution":
            self.plot_error_distribution()
        elif graph_type == "Performance by Stock":
            self.plot_performance_by_stock()
        elif graph_type == "Heatmap of Ensemble Performance":
            self.plot_ensemble_heatmap()
        elif graph_type == "Portfolio vs Max Possible Value":
            self.plot_portfolio_vs_max_possible()
        elif graph_type == "Aggregated Sharpe Prediction Accuracy":
            self.plot_aggregated_sharpe_accuracy()
        
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
            sharpe_ratio = data['Best_Prediction'].mean()
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
                
    def plot_sharpe_comparison(self):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        # Aggregate data: mean Best_Prediction and Actual_Sharpe per day
        daily_agg = data.groupby(data['date'].dt.date).agg({
            'Best_Prediction': 'mean',
            'Actual_Sharpe': 'mean'
        }).reset_index()
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])

        # Plot daily aggregated predicted Sharpe
        ax.plot(daily_agg['date'], daily_agg['Best_Prediction'], label='Daily Predicted Sharpe (Avg)', color='#2a82da', alpha=0.7)

        # Compute monthly actual Sharpe (for dates where Actual_Sharpe != -1)
        periodic_data = data[data['Actual_Sharpe'] != -1].copy()
        if not periodic_data.empty:
            periodic_data.set_index('date', inplace=True)
            monthly_sharpe = periodic_data['Actual_Sharpe'].resample('M').mean().dropna()
            if not monthly_sharpe.empty:
                ax.plot(monthly_sharpe.index, monthly_sharpe, label='Monthly Actual Sharpe (Avg)', 
                        color='#ff9900', marker='o', linestyle='--')

        ax.set_title('Daily Predicted vs Monthly Actual Sharpe Ratio (Aggregated)', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Sharpe Ratio', color='white')
        ax.legend()
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        
    def plot_sharpe_distribution(self):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]
        sns.histplot(data=data, x='Best_Prediction', bins=30, ax=ax, color='#2a82da', alpha=0.7, label='Daily Predicted Sharpe')
        actual_data = data[data['Actual_Sharpe'] != -1]
        if not actual_data.empty:
            sns.histplot(data=actual_data, x='Actual_Sharpe', bins=15, ax=ax, color='#ff9900', alpha=0.7, label='Periodic Actual Sharpe')
        ax.set_title('Distribution of Sharpe Ratios', color='white')
        ax.set_xlabel('Sharpe Ratio', color='white')
        ax.set_ylabel('Count', color='white')
        ax.legend()
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        
    def plot_scatter_with_regression(self):
        ax = self.chart_fig.add_subplot(111)
        filtered_data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0]
        filtered_data = filtered_data[(filtered_data['date'] >= self.start_date) & (filtered_data['date'] <= self.end_date)]
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data with Actual Sharpe != -1.0', horizontalalignment='center', color='white')
            return

        # Aggregate by date to reduce points
        agg_data = filtered_data.groupby(filtered_data['date'].dt.date).agg({
            'Best_Prediction': 'mean',
            'Actual_Sharpe': 'mean'
        }).reset_index()

        colors = ['#4CAF50' if pred >= actual else '#F44336' for pred, actual in zip(agg_data['Best_Prediction'], agg_data['Actual_Sharpe'])]
        ax.scatter(agg_data['Actual_Sharpe'], agg_data['Best_Prediction'], c=colors, alpha=0.7, label='Predictions')
        min_val = min(agg_data['Actual_Sharpe'].min(), agg_data['Best_Prediction'].min())
        max_val = max(agg_data['Actual_Sharpe'].max(), agg_data['Best_Prediction'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'y--', label='Perfect Prediction')
        ax.set_title('Predicted vs Actual Sharpe Ratio (Aggregated)', color='white')
        ax.set_xlabel('Actual Sharpe Ratio', color='white')
        ax.set_ylabel('Predicted Sharpe Ratio', color='white')
        ax.legend()
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        
    def plot_time_series_comparison(self):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        # Aggregate data: mean Best_Prediction and Actual_Sharpe per day
        daily_agg = data.groupby(data['date'].dt.date).agg({
            'Best_Prediction': 'mean',
            'Actual_Sharpe': 'mean'
        }).reset_index()
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])

        # Plot aggregated predicted Sharpe
        ax.plot(daily_agg['date'], daily_agg['Best_Prediction'], label='Predicted Sharpe (Avg)', color='#2a82da', alpha=0.7)

        # Plot smoothed predicted Sharpe
        smoothed_pred = daily_agg['Best_Prediction'].rolling(window=7, min_periods=1).mean()
        ax.plot(daily_agg['date'], smoothed_pred, label='Predicted Sharpe (Smoothed Avg)', color='#2a82da', linewidth=2)

        # Plot aggregated actual Sharpe (where available)
        actual_data = daily_agg[daily_agg['Actual_Sharpe'].notna()]
        if not actual_data.empty:
            ax.scatter(actual_data['date'], actual_data['Actual_Sharpe'], label='Actual Sharpe (Avg)', color='#ff9900', s=50)

        ax.set_ylim(-5, 5)
        ax.set_title('Time Series Comparison of Aggregated Sharpe Ratios', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Sharpe Ratio', color='white')
        ax.legend()
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        
    def plot_error_distribution(self):
        ax = self.chart_fig.add_subplot(111)
        filtered_data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0]
        filtered_data = filtered_data[(filtered_data['date'] >= self.start_date) & (filtered_data['date'] <= self.end_date)]
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data with Actual Sharpe != -1.0', horizontalalignment='center', color='white')
            return

        # Aggregate errors by date
        agg_data = filtered_data.groupby(filtered_data['date'].dt.date).agg({
            'Best_Prediction': 'mean',
            'Actual_Sharpe': 'mean'
        }).reset_index()
        errors = agg_data['Best_Prediction'] - agg_data['Actual_Sharpe']

        ax.hist(errors, bins=30, color='#2a82da', alpha=0.7)
        ax.axvline(0, color='yellow', linestyle='--', label='Zero Error')
        ax.set_title('Distribution of Prediction Errors (Aggregated)', color='white')
        ax.set_xlabel('Prediction Error', color='white')
        ax.set_ylabel('Frequency', color='white')
        ax.legend()
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        
    def plot_performance_by_stock(self):
        ax = self.chart_fig.add_subplot(111)
        filtered_data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0]
        filtered_data = filtered_data[(filtered_data['date'] >= self.start_date) & (filtered_data['date'] <= self.end_date)]
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data with Actual Sharpe != -1.0', horizontalalignment='center', color='white')
            return

        # Calculate mean absolute error per ticker
        performance = filtered_data.groupby('Ticker').apply(
            lambda x: np.mean(np.abs(x['Best_Prediction'] - x['Actual_Sharpe']))
        ).sort_values(ascending=False)

        # Select top 10 tickers for readability
        top_n = 10
        performance_top = performance.head(top_n)

        performance_top.plot(kind='bar', ax=ax, color='#2a82da')
        ax.set_title(f'Average Prediction Error by Stock (Top {top_n})', color='white')
        ax.set_xlabel('Stock Ticker', color='white')
        ax.set_ylabel('Mean Absolute Error', color='white')
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        
    def plot_ensemble_heatmap(self):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0]
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]
        if data.empty:
            ax.text(0.5, 0.5, 'No data available with Actual Sharpe != -1.0', horizontalalignment='center', color='white')
            return

        # Check for ensemble prediction columns
        pred_cols = [col for col in data.columns if col.startswith('Prediction_') or col == 'Best_Prediction']
        if len(pred_cols) < 2:
            # If only Best_Prediction exists, show errors for top N tickers
            top_n = 10
            errors = data.groupby('Ticker').apply(
                lambda x: np.mean(np.abs(x['Best_Prediction'] - x['Actual_Sharpe']))
            ).sort_values(ascending=False).head(top_n).to_frame(name='Error')
            if errors.empty:
                ax.text(0.5, 0.5, 'No error data to plot', horizontalalignment='center', color='white')
            else:
                sns.heatmap(errors, ax=ax, cmap='viridis', annot=True, fmt='.2f')
                ax.set_title(f'Prediction Error by Stock (Best_Prediction, Top {top_n})', color='white')
        else:
            # Aggregate errors across all tickers for each method
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
        
    def plot_aggregated_sharpe_accuracy(self):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        data['date'] = pd.to_datetime(data['date'], utc=True)
        data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

        # Aggregate data: mean Best_Prediction and Actual_Sharpe per day
        daily_agg = data.groupby(data['date'].dt.date).agg({
            'Best_Prediction': 'mean',
            'Actual_Sharpe': 'mean'
        }).reset_index()
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])

        # Plot aggregated predicted vs actual Sharpe
        ax.plot(daily_agg['date'], daily_agg['Best_Prediction'], label='Predicted Sharpe (Avg)', color='#2a82da')
        ax.plot(daily_agg['date'], daily_agg['Actual_Sharpe'], label='Actual Sharpe (Avg)', color='#ff9900', linestyle='--')

        ax.set_title('Aggregated Predicted vs Actual Sharpe Ratio Over Time', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Sharpe Ratio', color='white')
        ax.legend()
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')