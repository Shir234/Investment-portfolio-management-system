import pandas as pd
from datetime import datetime
from trading_logic import map_risk_threshold_to_sharpe
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DataManager:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.load_data()
        
    def load_data(self):
        try:
            self.data = pd.read_csv(self.csv_path)
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True)
                self.data.rename(columns={'Date': 'date'}, inplace=True)
                self.data.sort_values('date', inplace=True)
            logging.info(f"Loaded data with date range: {self.data['date'].min()} to {self.data['date'].max()}")
            logging.info(f"Sample data:\n{self.data.head().to_string()}")
            required_columns = ['date', 'Ticker', 'Close', 'Buy', 'Sell', 'Actual_Sharpe', 'Best_Prediction']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            logging.info(f"Rows with Best_Prediction >= 0.5: {len(self.data[self.data['Best_Prediction'] >= 0.5])}")
            logging.info(f"Best_Prediction distribution:\n{self.data['Best_Prediction'].describe().to_string()}")
            logging.info(f"Buy column distribution:\n{self.data['Buy'].value_counts().to_string()}")
            positive_signals = len(self.data[self.data['Best_Prediction'] >= 0.5])
            if positive_signals < 10:
                logging.warning(f"Only {positive_signals} rows have Best_Prediction >= 0.5. "
                              "The prediction model may need retraining to produce more actionable signals.")
            misaligned = self.data[(self.data['Best_Prediction'] >= 0.5) & (self.data['Buy'] <= 0)]
            if not misaligned.empty:
                logging.warning(f"{len(misaligned)} rows with Best_Prediction >= 0.5 have Buy <= 0. "
                               f"Check model alignment:\n{misaligned[['date', 'Ticker', 'Best_Prediction', 'Buy']].to_string()}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            self.data = pd.DataFrame()
            raise ValueError(f"Failed to load CSV data from {self.csv_path}: {str(e)}")
            
    def get_date_range(self):
        if self.data is None or self.data.empty or 'date' not in self.data.columns:
            return None, None
        min_date = self.data['date'].min().to_pydatetime()
        max_date = self.data['date'].max().to_pydatetime()
        return min_date, max_date
        
    def get_stocks_by_risk_level(self, risk_level):
        if 'Best_Prediction' not in self.data.columns:
            raise ValueError(f"Column 'Best_Prediction' not found in data. Available columns: {self.data.columns.tolist()}")
        min_sharpe = map_risk_threshold_to_sharpe(risk_level * 10, self.data)  # Scale 1-10 to 0-100
        filtered_data = self.data[self.data['Best_Prediction'] >= min_sharpe]
        logging.info(f"Risk level: {risk_level}, Min Sharpe: {min_sharpe}, Filtered rows: {len(filtered_data)}")
        return filtered_data
            
    def get_portfolio_performance(self, start_date, end_date):
        mask = (self.data['date'].dt.date >= start_date) & (self.data['date'].dt.date <= end_date)
        filtered_data = self.data[mask]
        logging.info(f"Portfolio performance for {start_date} to {end_date}: {len(filtered_data)} rows")
        signals_in_range = filtered_data[filtered_data['Best_Prediction'] >= 0.5]
        if not signals_in_range.empty:
            logging.info(f"Signals with Best_Prediction >= 0.5 in range:\n{signals_in_range[['date', 'Ticker', 'Best_Prediction']].to_string()}")
        else:
            logging.warning(f"No signals with Best_Prediction >= 0.5 in {start_date} to {end_date}.")
        return filtered_data
        
    def get_recommendations(self, risk_level, investment_amount):
        return self.get_stocks_by_risk_level(risk_level)