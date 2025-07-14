# data_manager.py
"""
Data management module for loading, validating, and filtering CSV financial data.
Handles CSV file processing with date range filtering and data validation.
"""
import pandas as pd
import os
from datetime import datetime
from frontend.logging_config import get_logger

logger = get_logger(__name__)


class DataManager:
    """
    Manages financial data loading, validation, and filtering operations.
    
    Handles CSV file processing with validation for required columns,
    date parsing, and provides filtered data access based on date ranges.
    """

    def __init__(self, csv_path):
        """Initialize DataManager with CSV file path and load data."""
        self.data = None
        self.start_date = None
        self.end_date = None
        self.dataset_start_date = None
        self.dataset_end_date = None
        self.csv_path = os.path.abspath(csv_path) if csv_path else None
        self.load_data()

    def load_data(self):
        """
        Load and validate CSV data with error handling, returning (success, error_msg).
        """
        try:
            if not self.csv_path or not os.path.exists(self.csv_path):
                error_msg = f"Data file not found: {self.csv_path}"
                logger.error(error_msg)
                self.data = pd.DataFrame()  # Empty DataFrame as fallback
                return False, error_msg

            logger.debug(f"Loading data from {self.csv_path}")
            self.data = pd.read_csv(self.csv_path)

            # Validate required columns
            required_columns = ['Date', 'Ticker', 'Close', 'Buy', 'Sell', 'Actual_Sharpe', 'Best_Prediction']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                error_msg = f"Invalid CSV file. Missing columns: {', '.join(missing_columns)}"
                logger.error(error_msg)
                self.data = pd.DataFrame()
                return False, error_msg

            # Validate and preprocess Close prices
            if self.data['Close'].isnull().any() or (self.data['Close'] <= 0).any():
                error_msg = "Invalid Close prices: null or negative values detected"
                logger.error(error_msg)
                self.data = pd.DataFrame()
                return False, error_msg
            if self.data['Close'].max() > 10000:
                logger.warning(f"Extreme Close price detected: {self.data['Close'].max()}. Verify data scaling.")

            # Convert dates to UTC with error handling
            try:
                self.data['date'] = pd.to_datetime(self.data['Date'], errors='coerce', utc=True)
                if self.data['date'].isna().any():
                    error_msg = "Invalid dates found in Date column"
                    logger.error(error_msg)
                    self.data = pd.DataFrame()
                    return False, error_msg
            except Exception as e:
                error_msg = f"Date parsing failed: {e}"
                logger.error(error_msg)
                self.data = pd.DataFrame()
                return False, error_msg

            # Determine dataset date range
            self.dataset_start_date = self.data['date'].min()
            self.dataset_end_date = self.data['date'].max()
            logger.debug(f"Dataset date range: {self.dataset_start_date} to {self.dataset_end_date}")

            logger.info(f"Data loaded: {len(self.data)} rows")
            return True, ""
        except Exception as e:
            error_msg = f"Error loading data: {e}"
            logger.error(error_msg)
            self.data = pd.DataFrame()
            return False, error_msg

    def set_date_range(self, start_date, end_date):
        """
        Set analysis date range with automatic bounds checking and adjustment.
        
        Validates date types, ensures logical ordering, and constrains dates
        to the available dataset range with user notification of adjustments.
        
        Args:
        - start_date: Analysis start date (converts to pd.Timestamp if needed)
        - end_date: Analysis end date (converts to pd.Timestamp if needed)
            
        Returns:
        - (success_bool, message) - success status and adjustment details
        """
        try:
            # Ensure consistent timestamp types
            if not isinstance(start_date, pd.Timestamp):
                logger.warning(f"start_date is not a pd.Timestamp: {type(start_date)}. Converting to pd.Timestamp.")
                start_date = pd.to_datetime(start_date, utc=True)
            if not isinstance(end_date, pd.Timestamp):
                logger.warning(f"end_date is not a pd.Timestamp: {type(end_date)}. Converting to pd.Timestamp.")
                end_date = pd.to_datetime(end_date, utc=True)

            # Validate date logic
            if start_date >= end_date:
                logger.error("Start date must be before end date")
                return False, "Start date must be before end date"

            if self.dataset_start_date is None or self.dataset_end_date is None:
                logger.error("Dataset date range not set. Load data first.")
                return False, "Dataset date range not set"

            # Adjust dates to dataset boundaries
            original_start_date = start_date
            original_end_date = end_date
            was_adjusted = False

            if start_date < self.dataset_start_date:
                start_date = self.dataset_start_date
                was_adjusted = True
                logger.debug(f"Start date adjusted from {original_start_date} to {start_date}")

            if end_date > self.dataset_end_date:
                end_date = self.dataset_end_date
                was_adjusted = True
                logger.debug(f"End date adjusted from {original_end_date} to {end_date}")

            self.start_date = start_date
            self.end_date = end_date
            logger.debug(f"Date range set: {start_date} to {end_date}")

            message = ""
            if was_adjusted:
                message = f"Date range adjusted to dataset range: {start_date.date()} to {end_date.date()}"
            return True, message
        except Exception as e:
            logger.error(f"Error setting date range: {e}")
            return False, f"Error setting date range: {e}"

    def get_filtered_data(self):
        """
        Applies date range filtering and returns only essential columns
        needed for analysis and visualization.
        
        Returns:
        - DataFrame with filtered data or None if error/no data available
        """
        try:
            if self.data is None or self.data.empty:
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
