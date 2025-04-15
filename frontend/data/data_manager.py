import pandas as pd
from datetime import datetime

class DataManager:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the CSV data"""
        self.data = pd.read_csv(self.csv_path)
        if 'Date' in self.data.columns:
            # Parse 'Date' as datetime, converting to UTC
            self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True)
            # Rename to 'date' for consistency
            self.data.rename(columns={'Date': 'date'}, inplace=True)
            # Sort by date as requested
            self.data.sort_values('date', inplace=True)
        # Add any other preprocessing steps if needed
            
    def get_date_range(self):
        if 'Date' in self.data.columns:
            return self.data['Date'].min(), self.data['Date'].max()
        return None, None
        
    def get_stocks_by_risk_level(self, risk_level):
        """Filter stocks based on risk level (Sharpe ratio)"""
        # Implementation will depend on the specific risk level mapping
        # This is a placeholder - adjust according to your risk level criteria
        return self.data[self.data['daily_sharpe'] >= risk_level]
        
    def get_portfolio_performance(self, start_date, end_date):
        """Calculate portfolio performance for the given date range"""
        mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
        return self.data[mask]
        
    def get_recommendations(self, risk_level, investment_amount):
        """Generate investment recommendations based on risk level and investment amount"""
        # Implementation will depend on your recommendation logic
        # This is a placeholder
        return self.get_stocks_by_risk_level(risk_level) 