# tests_on_one_ticker.py
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import yfinance as yf
import pandas_ta as ta
import os
from sklearn.base import BaseEstimator, TransformerMixin
from pandas_datareader import data as pdr
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
clean_data_date_folder = "20250522"
date_path = os.path.join(base_directory, clean_data_date_folder)

if not os.path.exists(date_path):
    print(f"Folder path '{date_path}' does not exist.")
# read csv -> {ticker}_clean_data: send clean data to full pipelin
ticker_csv_path = os.path.join(date_path, f"IFF_clean_data.csv")
# Check if the file exists
if not os.path.exists(ticker_csv_path):
    logger.warning(f"File not found: {ticker_csv_path}")

# Read the CSV file
# ticker_clean_data = pd.read_csv(ticker_csv_path)

ticker_clean_data = pd.read_csv(ticker_csv_path, parse_dates=['Date'], index_col='Date')
if not isinstance(ticker_clean_data.index, pd.DatetimeIndex):
    logger.warning(f"Index for {ticker} is not a DatetimeIndex. Converting to datetime.")
    ticker_clean_data.index = pd.to_datetime(ticker_clean_data.index)


full_pipeline_for_single_stock(ticker_clean_data, logger, date_folder, current_date, 'IFF', "2013-01-01", "2024-01-01")
