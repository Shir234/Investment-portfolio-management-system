import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, csv_path):
        self.data = None
        self.start_date = None
        self.end_date = None
        self.load_data(csv_path)
        
    def load_data(self, file_path):
        """Load and preprocess data from CSV."""
        try:
            logger.debug(f"Loading data from {file_path}")
            self.data = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['Date', 'Ticker', 'Close', 'Best_Prediction']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                logger.error(f"Missing columns in dataset: {missing_columns}")
                raise ValueError(f"Missing columns: {missing_columns}")
            
            # Validate and preprocess Close prices
            if self.data['Close'].isnull().any() or (self.data['Close'] <= 0).any():
                logger.error("Invalid Close prices: null or negative values")
                raise ValueError("Invalid Close prices")
            if self.data['Close'].max() > 10000:
                logger.warning(f"Extreme Close price detected: {self.data['Close'].max()}. Verify data scaling.")
            
            # Convert dates to UTC with error handling
            try:
                self.data['date'] = pd.to_datetime(self.data['Date'], errors='coerce', utc=True)
                if self.data['date'].isna().any():
                    logger.error("Invalid dates found in Date column")
                    raise ValueError("Invalid dates in Date column")
            except Exception as e:
                logger.error(f"Date parsing failed: {e}")
                raise ValueError(f"Date parsing failed: {e}")
            
            logger.debug(f"Data loaded: {len(self.data)} rows")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
            
    def set_date_range(self, start_date, end_date):
        """Set the date range for analysis and validate dates."""
        try:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date, utc=True)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date, utc=True)
                
            if start_date >= end_date:
                logger.error("Start date must be before end date")
                return False
                
            self.start_date = start_date
            self.end_date = end_date
            logger.debug(f"Date range set: {start_date} to {end_date}")
            return True
        except Exception as e:
            logger.error(f"Error setting date range: {e}")
            return False
            
    def get_filtered_data(self):
        """Return data filtered by the set date range."""
        try:
            if self.data is None:
                logger.error("No data loaded")
                return None
            if self.start_date is None or self.end_date is None:
                logger.error("Date range not set")
                return None
                
            filtered_data = self.data[
                (self.data['date'] >= self.start_date) & 
                (self.data['date'] <= self.end_date)
            ][['date', 'Ticker', 'Close', 'Best_Prediction', 'Actual_Sharpe']].copy()
            logger.debug(f"Filtered data: {len(filtered_data)} rows")
            return filtered_data
        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            return None
