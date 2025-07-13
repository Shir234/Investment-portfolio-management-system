# tests_on_one_ticker.py
"""
Development and debugging utility for testing ML pipeline on single stocks.

Purpose:
- Rapid testing of pipeline changes without full batch processing
- Debugging specific ticker issues
- Feature development and validation
- Resource-efficient pipeline testing

Usage:
1. Update ticker_symbol variable to desired stock
2. Ensure clean_data_date_folder points to existing data
3. Run script to execute full pipeline on single ticker

Output: Same as full pipeline (models, ensembles, predictions) for one stock
"""
import pandas as pd
import os
import datetime

from Full_Pipeline_With_Data import full_pipeline_for_single_stock

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"stock_pipeline_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stock_pipeline")

# ===============================================================================
### Create data directories if they don't exist
# ===============================================================================
os.makedirs('results', exist_ok=True)
# Get current date in YYYYMMDD format
current_date = datetime.datetime.now().strftime("%Y%m%d")
# Create date folder inside results
date_folder = f'results/{current_date}'
os.makedirs(date_folder, exist_ok=True)

base_directory = "results" 
clean_data_date_folder = "20250523"
date_path = os.path.join(base_directory, clean_data_date_folder)

if not os.path.exists(date_path):
    print(f"Folder path '{date_path}' does not exist.")

# read csv -> {ticker}_clean_data: send clean data to full pipeline
ticker_symbol = 'IFF'
ticker_csv_path = os.path.join(date_path, f"{ticker_symbol}_clean_data.csv")
# Check if the file exists
if not os.path.exists(ticker_csv_path):
    logger.warning(f"File not found: {ticker_csv_path}")

# Read the CSV file
# ticker_clean_data = pd.read_csv(ticker_csv_path)

ticker_clean_data = pd.read_csv(ticker_csv_path, parse_dates=['Date'], index_col='Date')
if not isinstance(ticker_clean_data.index, pd.DatetimeIndex):
    logger.warning(f"Index for {ticker_symbol} is not a DatetimeIndex. Converting to datetime.")
    ticker_clean_data.index = pd.to_datetime(ticker_clean_data.index)

#full_pipeline_for_single_stock(ticker_clean_data, logger, date_folder, current_date, 'PTC', "2013-01-01", "2024-01-01")
full_pipeline_for_single_stock(ticker_clean_data, logger, date_folder, current_date, ticker_symbol, "2013-01-01", "2024-01-01")
