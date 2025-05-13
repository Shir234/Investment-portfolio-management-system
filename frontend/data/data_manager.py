import pandas as pd
from datetime import datetime
from trading_logic import map_risk_threshold_to_sharpe  # Import the function

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
            print(f"Loaded data with date range: {self.data['date'].min()} to {self.data['date'].max()}")
            print("Sample data:")
            print(self.data.head())
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = pd.DataFrame()
            
    def get_date_range(self):
        """Return the earliest and latest dates in the data"""
        if self.data is None or self.data.empty or 'date' not in self.data.columns:
            return None, None
        min_date = self.data['date'].min().to_pydatetime()
        max_date = self.data['date'].max().to_pydatetime()
        return min_date, max_date
        
    def get_stocks_by_risk_level(self, risk_level):
        """Filter stocks based on risk level using 'Best_Prediction'"""
        if 'Best_Prediction' not in self.data.columns:
            raise ValueError("Column 'Best_Prediction' not found in data. Available columns: " + str(self.data.columns.tolist()))
        min_sharpe = map_risk_threshold_to_sharpe(risk_level)
        return self.data[self.data['Best_Prediction'] >= min_sharpe]
            
    def get_portfolio_performance(self, start_date, end_date):
        """Calculate portfolio performance for the given date range"""
        mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
        return self.data[mask]
        
    def get_recommendations(self, risk_level, investment_amount):
        """Generate investment recommendations based on risk level and investment amount"""
        return self.get_stocks_by_risk_level(risk_level)