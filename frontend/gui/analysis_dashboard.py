from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QComboBox, QGroupBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class AnalysisDashboard(QWidget):
    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        
        # Set dark mode for matplotlib
        plt.style.use('dark_background')
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Add graph selection dropdown
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
        
        # Create figure for main chart area
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
            
        # Update labels with consistent styling
        for label in self.metrics_labels.values():
            label.setStyleSheet("color: white; font-weight: bold; padding: 8px; background-color: #3c3f41; border-radius: 4px;")
            
        layout.addLayout(metrics_layout)
        
    def change_graph_type(self, index):
        """Handle changing the graph type based on dropdown selection"""
        self.update_visualizations()
    
    def update_visualizations(self):
        """Update visualizations based on selected graph type"""
        graph_type = self.graph_combo.currentText()
        
        # Clear the figure
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
        
        # Update metrics regardless of the graph type
        self.update_metrics()
        
        # Refresh the canvas
        self.chart_canvas.draw()
        
    def plot_portfolio_performance(self):
        """
        Plot the portfolio's performance over time.

        This graph displays the portfolio's value as it changes over time, helping to identify trends in growth or decline.

        Data Used:
        - 'date': The dates corresponding to the portfolio values.
        - 'portfolio_value' or 'Close': The portfolio's value or a simulated value based on closing prices.

        Insights:
        - A rising line indicates portfolio growth.
        - Sharp drops may signal poor trades or market downturns.
        """
        ax = self.chart_fig.add_subplot(111)
        
        # Get portfolio performance data
        data = self.data_manager.data
        
        # Ensure date is in datetime format
        if 'Date' in data.columns:
            data['date'] = pd.to_datetime(data['Date'])
        
        # Plot portfolio value over time
        if 'portfolio_value' in data.columns:
            ax.plot(data['date'], data['portfolio_value'], label='Portfolio Value', color='#2a82da')
        else:
            # Simulate portfolio value if not available
            ax.plot(data['date'], data['Close'].cumsum(), label='Simulated Value', color='#2a82da')
            
        ax.set_title('Portfolio Performance Over Time', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Value ($)', color='white')
        ax.legend()
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        
    def plot_sharpe_comparison(self):
        """
        Compare daily predicted Sharpe ratios with monthly actual Sharpe ratios.

        This graph shows how well daily predictions align with actual performance over longer periods (e.g., monthly).

        Data Used:
        - 'date': The dates for the predictions and actuals.
        - 'Best_Prediction': Daily predicted Sharpe ratios.
        - 'Actual_Sharpe': Actual Sharpe ratios where available.

        Insights:
        - The blue line represents daily predictions.
        - The orange dashed line with dots represents monthly actuals.
        - Close alignment suggests accurate predictions over time.
        """
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        
        # Plot daily predictions
        ax.plot(data['date'], data['Best_Prediction'], label='Daily Sharpe Prediction', color='#2a82da', alpha=0.7)
        
        # Aggregate Actual_Sharpe over a monthly period
        periodic_data = data[data['Actual_Sharpe'] != -1].copy()
        if not periodic_data.empty:
            # Resample to monthly, taking the mean of Actual_Sharpe
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
        """
        Display the distribution of predicted and actual Sharpe ratios.

        This histogram helps assess the consistency and accuracy of predictions compared to actual performance.

        Data Used:
        - 'Best_Prediction': Daily predicted Sharpe ratios.
        - 'Actual_Sharpe': Actual Sharpe ratios where available.

        Insights:
        - Blue bars show daily predictions; orange bars show actuals.
        - Significant overlap indicates accurate predictions.
        """
        ax = self.chart_fig.add_subplot(111)
        
        # Get Sharpe ratio data
        data = self.data_manager.data
        
        # Create two distributions
        sns.histplot(data=data, x='Best_Prediction', bins=30, 
                    ax=ax, color='#2a82da', alpha=0.7, label='Daily Sharpe')
        
        # Filter for Actual_Sharpe != -1 and plot if there's data
        actual_data = data[data['Actual_Sharpe'] != -1]
        if not actual_data.empty:
            sns.histplot(data=actual_data, x='Actual_Sharpe', bins=15, 
                        ax=ax, color='#ff9900', alpha=0.7, label='Periodic Sharpe')
        
        ax.set_title('Distribution of Sharpe Ratios', color='white')
        ax.set_xlabel('Sharpe Ratio', color='white')
        ax.set_ylabel('Count', color='white')
        ax.legend()
        ax.grid(True, color='#444444')
        ax.tick_params(colors='white')
        
        self.chart_fig.patch.set_facecolor('#353535')
        ax.set_facecolor('#2b2b2b')
        
    def plot_portfolio_composition(self):
        """
        Illustrate the portfolio's composition based on trading signals.

        This pie chart shows the proportion of Buy, Sell, and Hold signals, indicating the portfolio's trading strategy balance.

        Data Used:
        - 'Buy', 'Sell': Columns indicating Buy and Sell signals.

        Insights:
        - A large "Buy" slice suggests an aggressive strategy.
        - A large "Hold" slice indicates a cautious approach.
        """
        ax = self.chart_fig.add_subplot(111)
        
        # Get portfolio composition data
        data = self.data_manager.data
        
        try:
            # If sector data is available
            if 'sector' in data.columns and 'value' in data.columns:
                composition = data.groupby('sector')['value'].sum()
                ax.pie(composition, labels=composition.index, autopct='%1.1f%%', 
                      colors=plt.cm.tab10.colors)
                ax.set_title('Portfolio Composition by Sector', color='white')
            else:
                # If sector data isn't available, create a simple buy/sell composition
                buy_count = len(data[data['Buy'] != -1])
                sell_count = len(data[data['Sell'] != -1])
                hold_count = len(data) - buy_count - sell_count
                
                labels = ['Buy', 'Sell', 'Hold']
                sizes = [buy_count, sell_count, hold_count]
                colors = ['#4CAF50', '#F44336', '#2196F3']
                
                # Plot only non-zero values
                non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
                if non_zero:
                    labels, sizes, colors = zip(*non_zero)
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
                    ax.set_title('Trade Signal Distribution', color='white')
                else:
                    ax.text(0.5, 0.5, 'No composition data available', 
                            horizontalalignment='center', color='white')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting composition: {str(e)}', 
                    horizontalalignment='center', color='white')
        
        self.chart_fig.patch.set_facecolor('#353535')
        
    def update_metrics(self):
        """Update the metrics display"""
        data = self.data_manager.data
        
        try:
            # Calculate metrics
            if 'portfolio_value' in data.columns:
                total_value = data['portfolio_value'].iloc[-1]
            else:
                total_value = data['Close'].iloc[-1]
                
            sharpe_ratio = data['Best_Prediction'].mean()
            
            if 'portfolio_value' in data.columns:
                volatility = data['portfolio_value'].pct_change().std()
            else:
                volatility = data['Close'].pct_change().std()
            
            # Update labels
            self.metrics_labels['Total Value'].setText(f"Total Value: ${total_value:,.2f}")
            self.metrics_labels['Sharpe Ratio'].setText(f"Avg Sharpe: {sharpe_ratio:.2f}")
            self.metrics_labels['Volatility'].setText(f"Volatility: {volatility:.2%}")
        except Exception as e:
            for label in self.metrics_labels.values():
                label.setText(f"Error: {str(e)}")
                
    def plot_scatter_with_regression(self):
        """
        Compare predicted vs. actual Sharpe ratios with a regression line.

        This scatter plot assesses prediction accuracy, with points color-coded by trade profitability.

        Data Used:
        - 'Actual_Sharpe': Actual Sharpe ratios.
        - 'Best_Prediction': Predicted Sharpe ratios.
        - 'Buy', 'Sell', 'Close': Used to determine trade profitability.

        Insights:
        - Points near the yellow dashed line indicate accurate predictions.
        - Green points are profitable trades; red points are not.
        """
        ax = self.chart_fig.add_subplot(111)
        filtered_data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0]
        
        # Color-code points based on profitability (if Buy/Sell signals lead to price increase/decrease)
        colors = []
        for _, row in filtered_data.iterrows():
            if row['Buy'] != -1.0 and row['Close'] > row['Buy']:
                colors.append('#4CAF50')  # Green for profitable buy
            elif row['Sell'] != -1.0 and row['Close'] < row['Sell']:
                colors.append('#4CAF50')  # Green for profitable sell
            else:
                colors.append('#F44336')  # Red for unprofitable/no signal
        
        ax.scatter(filtered_data['Actual_Sharpe'], filtered_data['Best_Prediction'], 
                   c=colors, alpha=0.7, label='Predictions')
        
        # Add 45° line
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
        """
        Show predicted and actual Sharpe ratios over time with a smoothed trend.

        This graph highlights trends in predictions and actual performance over time.

        Data Used:
        - 'date': The dates for the predictions and actuals.
        - 'Best_Prediction': Predicted Sharpe ratios.
        - 'Actual_Sharpe': Actual Sharpe ratios where available.

        Insights:
        - The faint blue line is raw predictions; the bold blue line is the smoothed trend.
        - Orange dots represent actual Sharpe ratios.
        - A close following of the trend to actuals suggests good prediction tracking.
        """
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data.copy()
        
        # Plot raw predicted Sharpe
        ax.plot(data['date'], data['Best_Prediction'], label='Predicted Sharpe (Raw)', 
                color='#2a82da', alpha=0.3)
        
        # Add a smoothed trend line for predicted Sharpe (e.g., 7-day rolling mean)
        smoothed_pred = data['Best_Prediction'].rolling(window=7, min_periods=1).mean()
        ax.plot(data['date'], smoothed_pred, label='Predicted Sharpe (Smoothed)', 
                color='#2a82da', linewidth=2)
        
        # Plot actual Sharpe as scatter
        actual_data = data[data['Actual_Sharpe'] != -1.0]
        if not actual_data.empty:
            ax.scatter(actual_data['date'], actual_data['Actual_Sharpe'], 
                    label='Actual Sharpe', color='#ff9900', s=50)
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
        """
        Display the distribution of prediction errors.

        This histogram shows how far off predictions are from actual Sharpe ratios.

        Data Used:
        - 'Best_Prediction' and 'Actual_Sharpe': Used to calculate errors.

        Insights:
        - Blue bars represent the frequency of prediction errors.
        - A peak near zero indicates most predictions are accurate.
        """
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
        """
        Compare prediction accuracy across different stocks.

        This bar chart shows the average prediction error for each stock, highlighting model strengths and weaknesses.

        Data Used:
        - 'Ticker': Stock identifiers.
        - 'Best_Prediction' and 'Actual_Sharpe': Used to calculate errors.

        Insights:
        - Each bar represents a stock.
        - Shorter bars indicate better prediction accuracy.
        """
        ax = self.chart_fig.add_subplot(111)
        filtered_data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0]
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
        """
        Evaluate the performance of multiple prediction methods across stocks.

        This heatmap compares the prediction errors of different ensemble methods for each stock.

        Data Used:
        - 'Ticker': Stock identifiers.
        - Prediction columns (e.g., 'Best_Prediction', 'Prediction_LSTM'): Predicted Sharpe ratios.
        - 'Actual_Sharpe': Actual Sharpe ratios.

        Insights:
        - Each cell shows the error for a method-stock combination.
        - Darker colors indicate lower errors (better performance).
        - Note: Requires multiple prediction columns to display the heatmap.
        """
        ax = self.chart_fig.add_subplot(111)
        data = self.data_manager.data[self.data_manager.data['Actual_Sharpe'] != -1.0]
        
        if data.empty:
            ax.text(0.5, 0.5, 'No data available with Actual Sharpe != -1.0', 
                    horizontalalignment='center', color='white')
        else:
            # Check for multiple prediction columns
            pred_cols = [col for col in data.columns if col.startswith('Prediction_') or col == 'Best_Prediction']
            if len(pred_cols) < 2:
                # Fallback: Plot a heatmap of Best_Prediction error by Ticker
                errors = data.groupby('Ticker').apply(
                    lambda x: np.mean(np.abs(x['Best_Prediction'] - x['Actual_Sharpe']))
                ).to_frame(name='Error')
                if errors.empty:
                    ax.text(0.5, 0.5, 'No error data to plot', 
                            horizontalalignment='center', color='white')
                else:
                    sns.heatmap(errors, ax=ax, cmap='viridis', annot=True, fmt='.2f')
                    ax.set_title('Prediction Error by Stock (Best_Prediction)', color='white')
            else:
                # Original logic for multiple prediction methods
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