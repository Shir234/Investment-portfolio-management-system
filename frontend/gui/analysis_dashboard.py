from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QGroupBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from trading_logic import get_portfolio_history, get_orders  # Added get_orders

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
            "Heatmap of Ensemble Performance"
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
        df['date'] = pd.to_datetime(df['date'])
        ax.plot(df['date'], df['portfolio_value'], label='Portfolio Value', color='#2a82da')
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
        # Use executed trades from orders instead of raw signals
        orders = get_orders()
        if not orders:
            ax.text(0.5, 0.5, 'No trade history available', horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            return

        # Filter trades within the selected date range
        start_date = pd.to_datetime("2023-01-10")
        end_date = pd.to_datetime("2023-11-01")
        orders_df = pd.DataFrame(orders)
        orders_df['date'] = pd.to_datetime(orders_df['date'])
        orders_df = orders_df[(orders_df['date'] >= start_date) & (orders_df['date'] <= end_date)]

        if orders_df.empty:
            ax.text(0.5, 0.5, 'No trades in the selected date range', horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            return

        # Count buy and sell actions
        buy_count = len(orders_df[orders_df['action'] == 'Buy'])
        sell_count = len(orders_df[orders_df['action'] == 'Sell'])
        # Hold count is tricky with executed trades; approximate as days with no trades
        # For simplicity, calculate as a percentage of total actions
        total_actions = buy_count + sell_count
        if total_actions == 0:
            ax.text(0.5, 0.5, 'No buy or sell actions recorded', horizontalalignment='center', color='white')
            self.chart_fig.patch.set_facecolor('#353535')
            return

        # Calculate hold as a derived metric (since hold isn't explicitly recorded)
        # Approximate hold by considering the total number of trading days in the range
        trading_days = (end_date - start_date).days + 1
        hold_count = trading_days - total_actions  # Days with no trades
        if hold_count < 0:
            hold_count = 0  # Ensure non-negative

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
        
    def update_metrics(self):
        data = self.data_manager.data
        try:
            portfolio_history = get_portfolio_history()
            if portfolio_history:
                total_value = portfolio_history[-1]['portfolio_value']
            else:
                total_value = data['Close'].iloc[-1]
            sharpe_ratio = data['Best_Prediction'].mean()
            if portfolio_history:
                df = pd.DataFrame(portfolio_history)
                volatility = df['portfolio_value'].pct_change().std()
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
        ax.plot(data['date'], data['Best_Prediction'], label='Daily Sharpe Prediction', color='#2a82da', alpha=0.7)
        periodic_data = data[data['Actual_Sharpe'] != -1].copy()
        if not periodic_data.empty:
            periodic_data.set_index('date', inplace=True)
            monthly_sharpe = periodic_data['Actual_Sharpe'].resample('M').mean().dropna()
            if not monthly_sharpe.empty:
                ax.plot(monthly_sharpe.index, monthly_sharpe, label='Monthly Actual Sharpe', 
                        color='#ff9900', marker='o', linestyle='--')
        ax.set_title('Daily vs Monthly Sharpe Ratio Comparison', color='white')
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
        import seaborn as sns
        sns.histplot(data=data, x='Best_Prediction', bins=30, ax=ax, color='#2a82da', alpha=0.7, label='Daily Sharpe')
        actual_data = data[data['Actual_Sharpe'] != -1]
        if not actual_data.empty:
            sns.histplot(data=actual_data, x='Actual_Sharpe', bins=15, ax=ax, color='#ff9900', alpha=0.7, label='Periodic Sharpe')
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
        colors = []
        for _, row in filtered_data.iterrows():
            if row['Buy'] != -1.0 and row['Close'] > row['Buy']:
                colors.append('#4CAF50')
            elif row['Sell'] != -1.0 and row['Close'] < row['Sell']:
                colors.append('#4CAF50')
            else:
                colors.append('#F44336')
        ax.scatter(filtered_data['Actual_Sharpe'], filtered_data['Best_Prediction'], c=colors, alpha=0.7, label='Predictions')
        min_val = min(filtered_data['Actual_Sharpe'].min(), filtered_data['Best_Prediction'].min())
        max_val = max(filtered_data['Actual_Sharpe'].max(), filtered_data['Best_Prediction'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'y--', label='Perfect Prediction')
        ax.set_title('Predicted vs Actual Sharpe Ratio', color='white')
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
        ax.plot(data['date'], data['Best_Prediction'], label='Predicted Sharpe (Raw)', color='#2a82da', alpha=0.3)
        smoothed_pred = data['Best_Prediction'].rolling(window=7, min_periods=1).mean()
        ax.plot(data['date'], smoothed_pred, label='Predicted Sharpe (Smoothed)', color='#2a82da', linewidth=2)
        actual_data = data[data['Actual_Sharpe'] != -1.0]
        if not actual_data.empty:
            ax.scatter(actual_data['date'], actual_data['Actual_Sharpe'], label='Actual Sharpe', color='#ff9900', s=50)
        ax.set_ylim(-20, 20)
        ax.set_title('Time Series Comparison with Smoothed Predictions', color='white')
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
        errors = filtered_data['Best_Prediction'] - filtered_data['Actual_Sharpe']
        ax.hist(errors, bins=30, color='#2a82da', alpha=0.7)
        ax.axvline(0, color='yellow', linestyle='--', label='Zero Error')
        ax.set_title('Distribution of Prediction Errors', color='white')
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
        import numpy as np
        performance = filtered_data.groupby('Ticker').apply(
            lambda x: np.mean(np.abs(x['Best_Prediction'] - x['Actual_Sharpe']))
        )
        performance.plot(kind='bar', ax=ax, color='#2a82da')
        ax.set_title('Average Prediction Error by Stock', color='white')
        ax.set_xlabel('Stock Ticker', color='white')
        ax.set_ylabel('Mean Absolute Error', color='white')
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        
    def plot_ensemble_heatmap(self):
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0]
        import seaborn as sns
        import numpy as np
        if data.empty:
            ax.text(0.5, 0.5, 'No data available with Actual Sharpe != -1.0', horizontalalignment='center', color='white')
        else:
            pred_cols = [col for col in data.columns if col.startswith('Prediction_') or col == 'Best_Prediction']
            if len(pred_cols) < 2:
                errors = data.groupby('Ticker').apply(
                    lambda x: np.mean(np.abs(x['Best_Prediction'] - x['Actual_Sharpe']))
                ).to_frame(name='Error')
                if errors.empty:
                    ax.text(0.5, 0.5, 'No error data to plot', horizontalalignment='center', color='white')
                else:
                    sns.heatmap(errors, ax=ax, cmap='viridis', annot=True, fmt='.2f')
                    ax.set_title('Prediction Error by Stock (Best_Prediction)', color='white')
            else:
                performance_matrix = pd.DataFrame()
                for method in pred_cols:
                    errors = data.groupby('Ticker').apply(
                        lambda x: np.mean(np.abs(x[method] - x['Actual_Sharpe']))
                    )
                    performance_matrix[method] = errors
                sns.heatmap(performance_matrix, ax=ax, cmap='viridis', annot=True, fmt='.2f')
                ax.set_title('Ensemble Method Performance by Stock', color='white')
        ax.tick_params(colors='white')
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')