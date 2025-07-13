# Data_Cleaning_Pipelines.py
"""
Data fetching and cleaning pipelines.
Fetch the data, add indicators, calculate buy/sell signals.
Clean the data - use correlation and outliers.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import yfinance as yf
import pandas_ta as ta
from sklearn.base import BaseEstimator, TransformerMixin
from pandas_datareader import data as pdr
import datetime
import time


class AlphaVantageDataFetcher(BaseEstimator, TransformerMixin):
    """
    Fetch data from Alpha Vantage API, using paid API key
    """

    def __init__(self, ticker_symbol, start_date, end_date, api_key, max_retries=3, retry_delay=5):
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        from datetime import datetime
        
        for attempt in range(self.max_retries):
            try:
                # Convert string dates to datetime
                start = datetime.strptime(self.start_date, "%Y-%m-%d")
                end = datetime.strptime(self.end_date, "%Y-%m-%d")
                
                # Fetch data from Alpha Vantage
                history = pdr.DataReader(
                    self.ticker_symbol, 
                    'av-daily', 
                    start=start, 
                    end=end,
                    api_key=self.api_key
                )
                
                if history.empty:
                    raise ValueError(f"No data returned for ticker: {self.ticker_symbol}")
                
                # Rename columns to match expected format from yfinance
                history = history.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Ensure index is DatetimeIndex
                if not isinstance(history.index, pd.DatetimeIndex):
                    history.index = pd.to_datetime(history.index)
                    
                return history
            
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < self.max_retries - 1:
                    print(f"Rate limited for {self.ticker_symbol}. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Error fetching data for ticker {self.ticker_symbol}: {e}")
                    raise ValueError(f"Failed to fetch data for {self.ticker_symbol}: {e}")


class RollingSharpeCalculator(BaseEstimator, TransformerMixin):
    """
    Calculates rolling Sharpe ratio - a risk-adjusted return measure over a moving window.
    
    Formula: (Rolling_Mean_Return - Risk_Free_Rate) / Rolling_Std_Return * √252
    Higher values indicate better risk-adjusted performance.
    
    Parameters:
    - window (int): Rolling window size in days (default: 30)
    - risk_free_rate (float): Annual risk-free rate for comparison (default: 0.02)
    """

    def __init__(self, window=30, risk_free_rate=0.02):
        self.window = window
        self.risk_free_rate = risk_free_rate
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Calculate daily returns
        X['Daily_Return'] = X['Close'].pct_change()
        
        # Calculate rolling mean and standard deviation
        X['Rolling_Mean'] = X['Daily_Return'].rolling(window=self.window).mean()
        X['Rolling_Std'] = X['Daily_Return'].rolling(window=self.window).std()
        
        # Calculate daily risk-free rate
        daily_risk_free = self.risk_free_rate / 252  # Assuming 252 trading days
        
        # Calculate rolling Sharpe ratio (annualized)
        X['Daily_Sharpe_Ratio'] = (X['Rolling_Mean'] - daily_risk_free) / X['Rolling_Std'] * np.sqrt(252)
        
        # Drop intermediate columns
        X.drop(['Daily_Return', 'Rolling_Mean', 'Rolling_Std'], axis=1, inplace=True)
        
        return X


class IndicatorCalculator(BaseEstimator, TransformerMixin):
    """
    Calculates comprehensive technical indicators across multiple categories:
    
    - Trend: SMA, EMA, MACD, ADX (market direction)
    - Oscillators: RSI, Stochastic, Williams %R (overbought/oversold)
    - Volatility: Bollinger Bands, ATR (price volatility)
    - Volume: OBV, VWAP (volume analysis)
    - Momentum: PSAR, MFI (rate of change)
    
    Also includes temporal features and optional prime rate data.
    """

    def __init__(self, include_prime_rate=True, prime_start="2013-01-01", prime_end="2024-01-01"):
        self.indicators = [
            # Original indicators
            'psar', 'mfi', 'mvp',     
            # Moving averages
            'sma', 'ema', 'macd',           
            # Oscillators
            'rsi', 'stoch', 'williams',            
            # Volatility indicators
            'bbands', 'atr', 'hist_vol',         
            # Volume-based indicators
            'obv', 'vwap', 'ad_line',           
            # Trend indicators
            'adx', 'dmi', 'ichimoku',           
            # Temporal features
            'time_features'
        ]
        self.include_prime_rate = include_prime_rate
        self.prime_start = prime_start
        self.prime_end = prime_end
        self.prime_data = None

        # Fetch prime rate during initialization if needed
        if self.include_prime_rate:
            self._fetch_prime_rate()
    
    def _fetch_prime_rate(self):
        "Fetch prime rate data from FRED"
        try:
            start = datetime.datetime.strptime(self.prime_start, "%Y-%m-%d") 
            end = datetime.datetime.strptime(self.prime_end, "%Y-%m-%d")

            self.prime_data = pdr.get_data_fred('PRIME', start, end)
            # Apply forward fill then backward fill
            self.prime_data = self.prime_data.ffill()
            self.prime_data = self.prime_data.bfill()
            print(f"Prime rate data loaded with {len(self.prime_data)} records")
        
        except Exception as e:
            print(f"Error fetching prime rate data in fetch: {e}")
            self.prime_data = None
          
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        
        # Calculate technical indicators
        for indicator in self.indicators:
            try:
                # Original indicators
                if indicator == 'psar':
                    psar = ta.psar(data['High'], data['Low'], data['Close'])
                    data['PSAR'] = psar['PSARl_0.02_0.2']
                
                elif indicator == 'mfi':
                    tmp_volume = data['Volume'].astype(float)
                    mfi_values = ta.mfi(data['High'], data['Low'], data['Close'], tmp_volume)
                    if 'MFI' in data.columns:
                        if data['MFI'].dtype == 'int64':
                            data['MFI'] = data['MFI'].astype(float)
                    data.loc[:, 'MFI'] = mfi_values

                # Calculate Moving Volatility Pattern (MVP) (MVP as a simple moving average of squared returns)
                elif indicator == 'mvp':
                    data['Returns'] = data['Close'].pct_change()                                           # Calculate the daily return
                    data['Volatility'] = data['Returns'].rolling(window=28).std() * np.sqrt(252)           # Calculate 28-day rolling volatility (daily annual volatility)
                    data['MVP'] = data['Volatility'].pct_change()                                          # The pct change of the volatility, Rate of change of volatility

                # Moving Averages
                elif indicator == 'sma':
                    # Multiple timeframes
                    for period in [5, 10, 20, 50, 200]:
                        data[f'SMA_{period}'] = ta.sma(data['Close'], length=period)
                        # Calculate distance from SMA (percentage)
                        data[f'SMA_{period}_Dist'] = ((data['Close'] - data[f'SMA_{period}']) / data[f'SMA_{period}']) * 100
                
                elif indicator == 'ema':
                    # Multiple timeframes
                    for period in [5, 12, 26, 50]:
                        data[f'EMA_{period}'] = ta.ema(data['Close'], length=period)
                    
                elif indicator == 'macd':
                    macd = ta.macd(data['Close'])
                    data['MACD'] = macd['MACD_12_26_9']
                
                # Oscillators
                elif indicator == 'rsi':
                    data['RSI'] = ta.rsi(data['Close'], length=14)
                
                elif indicator == 'stoch':
                    stoch = ta.stoch(data['High'], data['Low'], data['Close'])
                    data['STOCH_K'] = stoch['STOCHk_14_3_3']
                    data['STOCH_D'] = stoch['STOCHd_14_3_3']
                
                elif indicator == 'williams':
                    data['Williams_R'] = ta.willr(data['High'], data['Low'], data['Close'])
                
                # Volatility indicators
                elif indicator == 'bbands':
                    bbands = ta.bbands(data['Close'], length=20, std=2)
                    data['BB_Upper'] = bbands['BBU_20_2.0']
                    data['BB_Middle'] = bbands['BBM_20_2.0']
                    data['BB_Lower'] = bbands['BBL_20_2.0']

                elif indicator == 'atr':
                    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
                    # ATR as percentage of price
                    data['ATR_Percent'] = (data['ATR'] / data['Close']) * 100
                
                elif indicator == 'hist_vol':
                    # Historical volatility with different look-backs
                    for period in [5, 10, 20, 60]:
                        data[f'Hist_Vol_{period}'] = data['Returns'].rolling(window=period).std() * np.sqrt(252)
                
                # Volume-based indicators
                elif indicator == 'obv':
                    data['OBV'] = ta.obv(data['Close'], data['Volume'])
                    # Normalize OBV with 20-day rolling mean
                    data['OBV_Norm'] = data['OBV'] / data['OBV'].rolling(window=20).mean()
                
                elif indicator == 'vwap':
                    # Using close as a proxy for typical price if needed
                    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                
                elif indicator == 'ad_line':
                    data['AD_Line'] = ta.ad(data['High'], data['Low'], data['Close'], data['Volume'])
                
                # Trend indicators
                elif indicator == 'adx':
                    adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
                    data['ADX'] = adx['ADX_14']
                    data['DI_Plus'] = adx['DMP_14']
                    # ADX trend strength
                    data['ADX_Trend'] = np.where(data['ADX'] > 25, 1, 0)

                # Ichimoku Cloud
                elif indicator == 'ichimoku':
                    try:
                        ichimoku = ta.ichimoku(data['High'], data['Low'], data['Close'])
                        
                        # Check if ichimoku is a tuple
                        if isinstance(ichimoku, tuple) and len(ichimoku) > 1 and isinstance(ichimoku[1], pd.DataFrame):
                            ichimoku_df = ichimoku[1]
                          
                            # Map appropriate columns based on what's available
                            if 'ISA_9_26_52' in ichimoku_df.columns:
                                data['Ichimoku_A'] = ichimoku_df['ISA_9_26_52']
                            if 'ISB_9_26_52' in ichimoku_df.columns:
                                data['Ichimoku_B'] = ichimoku_df['ISB_9_26_52'] 
                            if 'ITS_9_26_52' in ichimoku_df.columns:
                                data['Ichimoku_Base'] = ichimoku_df['ITS_9_26_52']
                            if 'ICS_9_26_52' in ichimoku_df.columns:
                                data['Ichimoku_Conversion'] = ichimoku_df['ICS_9_26_52']
                        
                    except Exception as e:
                        print(f"Error processing Ichimoku indicator: {e}")
                        print("Skipping Ichimoku indicator")
                    
                # Temporal features
                elif indicator == 'time_features':
                    # Convert index to datetime if it's not already
                    if not isinstance(data.index, pd.DatetimeIndex):
                        data.index = pd.to_datetime(data.index)
                    
                    # Day of week, month, quarter, etc.
                    data['Day_of_Week'] = data.index.dayofweek
                    data['Month'] = data.index.month
                    data['Quarter'] = data.index.quarter
                    data['Is_Month_End'] = data.index.is_month_end.astype(int)
                    data['Is_Month_Start'] = data.index.is_month_start.astype(int)
                    data['Is_Quarter_End'] = data.index.is_quarter_end.astype(int)
                    
                    # Cyclic encoding of time features
                    data['Sin_Day'] = np.sin(2 * np.pi * data['Day_of_Week'] / 5)
                    data['Cos_Day'] = np.cos(2 * np.pi * data['Day_of_Week'] / 5)
                    data['Sin_Month'] = np.sin(2 * np.pi * data['Month'] / 12)
                    data['Cos_Month'] = np.cos(2 * np.pi * data['Month'] / 12)
                
                    
            except Exception as e:
                print(f"Error adding indicator {indicator}: {e}")
            
            # Add prime rate data
            if self.include_prime_rate and self.prime_data is not None:
                try:
                    # Merge prime rate data with ticker data based on date index, make sure the index is datetime for both df
                    if not isinstance(data.index, pd.DatetimeIndex):
                        print(f"Warning: Stock data index is not DatetimeIndex. Converting...")
                        data.index = pd.to_datetime(data.index)

                    # Add prime rate
                    data['Prime_Rate'] = None

                    for date in data.index:
                        date_str = pd.Timestamp(date)
                        # Find all prime rated before the current date
                        applicable_dates = self.prime_data.index[self.prime_data.index <= date_str.to_datetime64()]
                        # If prime rate was found
                        if len(applicable_dates) > 0:
                            closest_date = applicable_dates[-1] # Get the last (most recent) date
                            data.at[date, 'Prime_Rate'] = self.prime_data.loc[closest_date, 'PRIME']

                except Exception as e:
                    print(f"Error adding prime rate data: {e}")
                  
        return data


class SignalCalculator(BaseEstimator, TransformerMixin):
    """
    Generates buy/sell signals using PSAR (Parabolic SAR, Stop and Reverse) indicator.
    
    Trading Logic:
    - Buy: When price closes ABOVE PSAR dots
    - Sell: When price closes BELOW PSAR dots
    
    PSAR acts as a trailing stop-loss that follows the trend direction.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data['Signal'] = np.where(data['Close'] > data['PSAR'], 1, 0)
        data['Signal_Shift'] = data['Signal'].shift(1)
        data['Buy'] = np.where((data['Signal'] == 1) & (data['Signal_Shift'] == 0), data['Close'], np.nan)
        data['Sell'] = np.where((data['Signal'] == 0) & (data['Signal_Shift'] == 1), data['Close'], np.nan)
        return data


class TransactionMetricsCalculator(BaseEstimator, TransformerMixin):
    """
    For each transaction calculate: volatility, return and sharpe ratio

    Calculates performance metrics for individual trades (buy to sell cycles).
    
    Metrics calculated:
    - Transaction Returns: (Sell_Price - Buy_Price) / Buy_Price
    - Transaction Volatility: Std dev of daily returns during holding period
    - Transaction Sharpe: Risk-adjusted return for the specific trade
    - Transaction Duration: Days held
    """

    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        # Initialize the transaction metrics columns with -1
        data['Transaction_Volatility'] = -1
        data['Transaction_Returns'] = -1
        data['Transaction_Sharpe'] = -1
        data['Transaction_Duration'] = -1
        
        last_buy_index = None
        last_buy_price = None

        for i in range(len(data)):
            # Record buy signal
            if not np.isnan(data['Buy'].iloc[i]):
                last_buy_index = i
                last_buy_price = data['Buy'].iloc[i]  # Use Buy price as the purchase price

            # Calculate metrics only on sell signal
            if not np.isnan(data['Sell'].iloc[i]) and last_buy_index is not None:
                buy_date = data.index[last_buy_index]
                current_date = data.index[i]
                duration = (current_date - buy_date).days

                # Calculate raw return for this transaction
                returns = (data['Sell'].iloc[i] - last_buy_price) / last_buy_price

                # Calculate volatility (standard deviation of daily returns) for this transaction
                # Use Close prices between buy and sell to calculate volatility
                daily_returns = data['Close'].iloc[last_buy_index:i+1].pct_change().dropna()
                volatility = daily_returns.std() if len(daily_returns) > 1 else 0

                # Calculate Sharpe ratio for this transaction
                transaction_risk_free_rate = self.risk_free_rate * (duration / 365)
                sharpe_ratio = (returns - transaction_risk_free_rate) / volatility if volatility != 0 else 0

                # Store the metrics only on the sell date
                data.loc[data.index[i], 'Transaction_Volatility'] = volatility
                data.loc[data.index[i], 'Transaction_Returns'] = returns
                data.loc[data.index[i], 'Transaction_Sharpe'] = sharpe_ratio
                data.loc[data.index[i], 'Transaction_Duration'] = duration

        return data
    

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values in financial data.
    
    For the Daily_Sharpe_Ratio column, use the median of the entire dataset to fill the initial null values.
    For other continuous columns, use forward-fill then backward-fill (standard approach for financial time series).
    For categorical/binary columns, fill with a sentinel value (-1).
    """
    
    def __init__(self, exclude_columns=[
        # Signal and binary columns
        'Signal', 'Signal_Shift', 'Buy', 'Sell',    
        # Categorical time features
        'Day_of_Week', 'Month', 'Quarter',         
        # Binary indicators
        'Is_Month_End', 'Is_Month_Start', 'Is_Quarter_End',
        'ADX_Trend',
    ]):
        self.exclude_columns = exclude_columns
        self.sharpe_median = None
    
    def fit(self, X, y=None):
        # Calculate median Sharpe ratio from non-null values
        if 'Daily_Sharpe_Ratio' in X.columns:
            self.sharpe_median = X['Daily_Sharpe_Ratio'].dropna().median()
            print(f"Median Sharpe ratio (for filling null values): {self.sharpe_median}")
        return self

    def transform(self, X):
        X_ = X.copy()

        # Handle all columns except those in exclude_columns
        for col in X_.columns:
            # Skip Daily_Sharpe_Ratio - already handled
            if col == 'Daily_Sharpe_Ratio':
                X_[col] = X_[col].fillna(self.sharpe_median).infer_objects(copy=False)
            elif any(exclude in col for exclude in self.exclude_columns):
                # For categorical/binary columns, fill with -1 as a sentinel value
                X_[col] = X_[col].fillna(-1).infer_objects(copy=False)
            elif X_[col].dtype in ['float64', 'int64']:
                # Use forward-fill then backward-fill for time series numeric data
                X_[col] = X_[col].ffill().bfill().infer_objects(copy=False)
            else:
                # For any other types, use -1
                X_[col] = X_[col].fillna(-1).infer_objects(copy=False)

        return X_
  

class ComprehensiveOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Two-step outlier handling for financial data:
    
    1. Percentile Capping: Clips values to 5th-95th percentiles to remove extreme outliers
    2. Log Transformation: Reduces skewness in financial data (except Sharpe ratios)
    
    Sharpe ratios are only capped (not log-transformed) to preserve negative values
    which indicate poor risk-adjusted performance.
    """

    def __init__(self, target_columns=['Transaction_Sharpe', 'Daily_Sharpe_Ratio'], lower_percentile=5, 
                 upper_percentile=95, exclude_columns=[
        # Price data - should not be transformed
        'Open', 'High', 'Low', 'Close', 'Volume',
        # Signal and binary columns
        'Signal', 'Signal_Shift', 'Buy', 'Sell',
        # Categorical time features
        'Day_of_Week', 'Month', 'Quarter',
        # Binary indicators
        'Is_Month_End', 'Is_Month_Start', 'Is_Quarter_End',
        'ADX_Trend',
        # Cyclic features (already bounded)
        'Sin_Day', 'Cos_Day', 'Sin_Month', 'Cos_Month',
        # External data 
        'Prime_Rate'
    ]):
        self.target_columns = target_columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.exclude_columns = exclude_columns
        self.percentiles = {}  # Store percentiles for each column
        self.min_vals = {}    # Store minimum values for log transformation
        self.numeric_columns = []

    def fit(self, X, y=None):
        # Identify numeric columns to process (excluding binary/categorical)
        self.numeric_columns = [
            col for col in X.columns
            if X[col].dtype in ['float64', 'int64']
            and not any(exclude in col for exclude in self.exclude_columns)
            ]
        
        # Exclude target_column from log transformation
        self.log_transform_columns = [
            col for col in self.numeric_columns if col not in self.target_columns]

        # Compute percentiles for capping
        for col in self.numeric_columns:
            self.percentiles[col] = np.percentile(X[col].dropna(), [self.lower_percentile, self.upper_percentile])
        return self
    
    def transform(self, X):
        X_ = X.copy()

        # Process each numeric column
        for col in self.numeric_columns:
            p1, p99 = self.percentiles[col]
            # Cap extreme values
            X_[col] = np.clip(X_[col], p1, p99)

            # Apply log transformation only to non-target columns
            if col in self.log_transform_columns:
                col_data = X_[col]
                self.min_vals[col] = col_data.min()
                if self.min_vals[col] < 0:
                    col_shifted = col_data - self.min_vals[col] + 1
                else:
                    col_shifted = col_data + 1
                X_[col] = np.log1p(col_shifted)

        return X_
    
    def inverse_transform(self, X):
        X_ = X.copy()
        # Inverse transform each numeric column
        for col in self.numeric_columns:
            if col in X_.columns:
                p1, p99 = self.percentiles[col]
                col_data = X_[col]
                if col in self.log_transform_columns:
                    # Inverse log transformation
                    col_original = np.expm1(col_data) + self.min_vals.get(col, 0) - 1
                    # Ensure values remain within capped range
                    X_[col] = np.clip(col_original, p1, p99)
                else:
                    # For Transaction_Sharpe, only ensure capping
                    X_[col] = np.clip(col_data, p1, p99)
        return X_


class CorrelationHandler(BaseEstimator, TransformerMixin):
    """
    Handle highly correlated features
    """

    def __init__(self, threshold=0.95, keep_columns=['Close']):
        self.threshold = threshold
        self.columns_to_drop = None
        self.keep_columns = keep_columns

    def fit(self, X, y=None):
        # Handle high correlation of continues columns
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.columns_to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]

        # Check all columns correlation with the target variable, if nan - remove (doesn't help for prediction later)
        if 'Daily_Sharpe_Ratio' in X.columns:
            target_corr = X.corr()['Daily_Sharpe_Ratio']
            nan_corr_columns = target_corr[target_corr.isna()].index.tolist()
            self.columns_to_drop.extend(nan_corr_columns)
    
        for col in self.keep_columns:
            if col in self.columns_to_drop:
                self.columns_to_drop.remove(col)

        if 'Daily_Sharpe_Ratio' in self.columns_to_drop:
            self.columns_to_drop.remove('Daily_Sharpe_Ratio')

        return self
  
    def transform(self, X):
        X_ = X.copy()
        for col in self.columns_to_drop:
            if col in X_.columns:
                print(f"Dropping column {col} due to NaN correlation with target")
        return X_.drop(self.columns_to_drop, axis=1, errors='ignore')


# ===============================================================================
# Create The Pipelines
# ===============================================================================
def create_stock_data_pipeline(ticker_symbol, start_date, end_date, risk_free_rate, api_key):
    """
    Creates a pipeline for fetching and processing stock data.
    
    Parameters:
    - ticker_symbol (str): Stock ticker symbol.
    - start_date (str): Start date for data (YYYY-MM-DD).
    - end_date (str): End date for data (YYYY-MM-DD).
    - risk_free_rate (float): Annual risk-free rate.
    - api_key (str): Alpha Vantage API key.
    
    Returns:
    - Pipeline: Scikit-learn pipeline object.
    """
    return Pipeline([
        ('data_fetcher', AlphaVantageDataFetcher(ticker_symbol, start_date, end_date, api_key)),
        ('rolling_sharpe_calculator', RollingSharpeCalculator(window=30, risk_free_rate=risk_free_rate)),
        ('indicator_calculator', IndicatorCalculator()),
        ('signal_calculator', SignalCalculator()),
        ('transaction_metrics_calculator', TransactionMetricsCalculator(risk_free_rate)),
    ])


def create_data_cleaning_pipeline(correlation_threshold = 0.95):
  """
    Creates a pipeline for cleaning financial data.
    
    The pipeline handles:
    1. Missing values - using median for Sharpe ratio and ffill/bfill for other numeric columns
    2. Outliers - using percentile-based capping
    3. Correlation - removing highly correlated features
    
    Parameters:
    - correlation_threshold (float): Threshold for removing highly correlated features.
    
    Returns:
    - Pipeline: Scikit-learn pipeline object.
    """
  
  return Pipeline([
      ('missing_value_handler', MissingValueHandler()),
      ('outlier_handler', ComprehensiveOutlierHandler()),
      ('correlation_handler', CorrelationHandler(correlation_threshold))
  ])
