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

# ===============================================================================
# Data Pipeline Classes
## Pipeline to fetch stock data and feature engineering
# ===============================================================================
# fetch historical stock data
class DataFetcher(BaseEstimator, TransformerMixin):
    def __init__(self, ticker_symbol, start_date, end_date):
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            ticker = yf.Ticker(self.ticker_symbol)
            history = ticker.history(start=self.start_date, end=self.end_date)
            
            if history.empty:
                raise ValueError(f"No data returned for ticker: {self.ticker_symbol}")

            return history
        
        except Exception as e:
            print(f"Error fetching data for ticker {self.ticker_symbol}: {e}")

            return pd.DataFrame() # return an empty df if an error occurs
      
# TODO -> ADD INDICATORS
# Add market indicators
class IndicatorCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, include_prime_rate=True, prime_start="2019-01-01", prime_end="2024-01-01"):
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
            self.prime_data = self.prime_data.fillna(method='ffill')
            self.prime_data = self.prime_data.fillna(method='bfill')
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
                    data['MFI'] = ta.mfi(data['High'], data['Low'], data['Close'], data['Volume']).astype(float)
                
                # Calculate Moving Volatility Pattern (MVP) (MVP as a simple moving average of squared returns)
                elif indicator == 'mvp':
                    data['Returns'] = data['Close'].pct_change()                                                # calculate the daily return
                    data['Volatility'] = data['Returns'].rolling(window=28).std() * np.sqrt(252)                # calculate 28-day rolling volatility (daily annual volatility)
                    data['MVP'] = data['Volatility'].pct_change()                                               # the pct change of the volatility, Rate of change of volatility

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
                    # TODO -> check what is the meaning if well use these signals in the prediction
                    # data['MACD_Signal'] = macd['MACDs_12_26_9']
                    # data['MACD_Hist'] = macd['MACDh_12_26_9']
                    # data['MACD_CrossOver'] = np.where(data['MACD'] > data['MACD_Signal'], 1, 
                    #                                    np.where(data['MACD'] < data['MACD_Signal'], -1, 0))
                
                # Oscillators
                elif indicator == 'rsi':
                    data['RSI'] = ta.rsi(data['Close'], length=14)
                    # TODO -> check if the thresholds are relevant, and if this data is valuable for prediction more then just RSI
                    # # Overbought/Oversold signals
                    # data['RSI_Overbought'] = np.where(data['RSI'] > 70, 1, 0)
                    # data['RSI_Oversold'] = np.where(data['RSI'] < 30, 1, 0)
                
                elif indicator == 'stoch':
                    stoch = ta.stoch(data['High'], data['Low'], data['Close'])
                    data['STOCH_K'] = stoch['STOCHk_14_3_3']
                    data['STOCH_D'] = stoch['STOCHd_14_3_3']
                    # TODO -> WHAT IS THIS?
                    # # Crossover signals
                    # data['STOCH_Signal'] = np.where(data['STOCH_K'] > data['STOCH_D'], 1, 
                    #                               np.where(data['STOCH_K'] < data['STOCH_D'], -1, 0))
                
                elif indicator == 'williams':
                    data['Williams_R'] = ta.willr(data['High'], data['Low'], data['Close'])
                
                # Volatility indicators
                elif indicator == 'bbands':
                    bbands = ta.bbands(data['Close'], length=20, std=2)
                    data['BB_Upper'] = bbands['BBU_20_2.0']
                    data['BB_Middle'] = bbands['BBM_20_2.0']
                    data['BB_Lower'] = bbands['BBL_20_2.0']
                    # TODO -> WHAT IS THIS?
                    # # Calculate BB width and %B
                    # data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
                    # data['BB_Percent'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
                
                elif indicator == 'atr':
                    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
                    # ATR as percentage of price
                    data['ATR_Percent'] = (data['ATR'] / data['Close']) * 100
                
                elif indicator == 'hist_vol':
                    # Historical volatility with different lookbacks
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
                    data['DI_Minus'] = adx['DMN_14']
                    # ADX trend strength
                    data['ADX_Trend'] = np.where(data['ADX'] > 25, 1, 0)
                    # TODO -> WHAT IS THIS?
                    # # Directional signals
                    # data['ADX_Signal'] = np.where(data['DI_Plus'] > data['DI_Minus'], 1, 
                    #                             np.where(data['DI_Plus'] < data['DI_Minus'], -1, 0))
                
                # Ichimoku Cloud
                elif indicator == 'ichimoku':
                    ichimoku = ta.ichimoku(data['High'], data['Low'], data['Close'])
                    data['Ichimoku_Conversion'] = ichimoku['ICS_9_26_52_26']
                    data['Ichimoku_Base'] = ichimoku['ITS_9_26_52_26']
                    data['Ichimoku_A'] = ichimoku['ISA_9_26_52_26']
                    data['Ichimoku_B'] = ichimoku['ISB_9_26_52_26']
                    # TODO -> WHAT IS THIS?
                    # # Cloud signals
                    # data['Ichimoku_Signal'] = np.where(data['Close'] > data['Ichimoku_A'], 1, 
                    #                                  np.where(data['Close'] < data['Ichimoku_B'], -1, 0))
                
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

# Calculate buy and sell signals based on PSAR indicator
class SignalCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data['Signal'] = np.where(data['Close'] > data['PSAR'], 1, 0)
        data['Signal_Shift'] = data['Signal'].shift(1)
        data['Buy'] = np.where((data['Signal'] == 1) & (data['Signal_Shift'] == 0), data['Close'], np.nan)
        data['Sell'] = np.where((data['Signal'] == 0) & (data['Signal_Shift'] == 1), data['Close'], np.nan)
        return data

# For each transaction calculate: volatility, return and sharpe ratio
class TransactionMetricsCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        # Initialize the transaction metrics columns
        data['Transaction_Volatility'] = np.nan
        data['Transaction_Returns'] = np.nan
        data['Transaction_Sharpe'] = np.nan
        data['Transaction_Duration'] = np.nan
        
        last_buy_index = None
        last_buy_price = None
        in_active_position = False

        # Process the data in a single pass
        for i in range(len(data)):
            # Record buy signal
            if not np.isnan(data['Buy'].iloc[i]):
                last_buy_index = i
                last_buy_price = data['Close'].iloc[i]
                in_active_position = True

            # Calculate metrics based on the last known buy signal
            if last_buy_index is not None:
                buy_date = data.index[last_buy_index]
                current_date = data.index[i]
                duration = (current_date - buy_date).days

                # Calculate raw return for this transaction
                returns = (data['Close'].iloc[i] - last_buy_price) / last_buy_price

                # Calculate volatility (standard deviation of daily returns) for this transaction
                daily_returns = data['Close'].iloc[last_buy_index:i+1].pct_change().dropna()
                volatility = daily_returns.std() if len(daily_returns) > 1 else 0

                # Calculate Sharpe ratio for this transaction
                transaction_risk_free_rate = self.risk_free_rate * (duration / 365)
                sharpe_ratio = (returns - transaction_risk_free_rate) / volatility if volatility != 0 else 0                        # all in the same scale: returns is the return ratio, the vplatility is pct

                # Store the metrics for every day while in a position
                data.loc[data.index[i], 'Transaction_Volatility'] = volatility
                data.loc[data.index[i], 'Transaction_Returns'] = returns
                data.loc[data.index[i], 'Transaction_Sharpe'] = sharpe_ratio
                data.loc[data.index[i], 'Transaction_Duration'] = duration

                # If this is a sell day, mark that we're no longer in an active position
                # but continue calculating using the same buy reference point
                if not np.isnan(data['Sell'].iloc[i]):
                    in_active_position = False

        return data
    
# ===============================================================================
# Data Cleaning Classes
## Pipeline to clean the data, remove outliers, and handle correlation
# ===============================================================================
# Use forward fill then backward fill (normal in financial data) on all continues columns
class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_columns=[
        # Signal and binary columns
        'Signal', 'Signal_Shift', 'Buy', 'Sell', 
        
        # Categorical time features
        'Day_of_Week', 'Month', 'Quarter', 
        
        # Binary indicators
        'Is_Month_End', 'Is_Month_Start', 'Is_Quarter_End',
        # TODO -> ADD THESE IF WE ADD THEM IN THE INDICATOR CALCULATOR
        # 'RSI_Overbought', 'RSI_Oversold',
        # 'STOCH_Signal', 'MACD_CrossOver', 'ADX_Signal',
        # 'Ichimoku_Signal', 'BB_Signal'
        'ADX_Trend'
    ]):
        self.exclude_columns = exclude_columns

    # def __init__(self, fill_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'PSAR', 'MFI', 'MVP']):
    #     self.fill_columns = fill_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        # Handle all columns except those in exclude_columns
        for col in X_.columns:
            if any(exclude in col for exclude in self.exclude_columns):
                # For categorical/binary columns, fill with -1 as a sentinel value
                X_[col] = X_[col].fillna(-1)
            elif X_[col].dtype in ['float64', 'int64']:
                # Use forward-fill then backward-fill for time series numeric data
                X_[col] = X_[col].fillna(method='ffill').fillna(method='bfill')
            else:
                # For any other types, use -1
                X_[col] = X_[col].fillna(-1)

        return X_
  
# Handle outliers: Use Z-score to identify data points that deviate significantly from the mean - rolling window of 28 days
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self,threshold=3, window=28, exclude_columns=[
        # Price data
        'Open', 'High', 'Low', 'Close', 'Volume',
        # Signal columns
        'Buy', 'Sell', 'Signal', 'Signal_Shift',
        # Binary signals or categorical
        'Is_Month_End', 'Is_Month_Start', 'Is_Quarter_End',
        'Day_of_Week', 'Month', 'Quarter', 
        'Sin_Day', 'Cos_Day', 'Sin_Month', 'Cos_Month',
        'RSI_Overbought', 'RSI_Oversold', 'STOCH_Signal', 
        'MACD_CrossOver', 'ADX_Signal', 'ADX_Trend', 
        'Ichimoku_Signal', 'MVP_Signal',
        # Transaction metrics
        'Transaction_Sharpe', 'Transaction_Returns', 
        'Transaction_Volatility', 'Transaction_Duration',
        # External data
        'Prime_Rate'
    ]):
        self.threshold = threshold
        self.window = window
        self.exclude_columns = exclude_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_ = X_.dropna(axis =1, how = 'all')

        for column in X_.columns:
            # Check if column should be excluded
            should_exclude = False
            for exclude_pattern in self.exclude_columns:
                if exclude_pattern in column:
                    should_exclude = True
                    break
            if should_exclude:
                continue
            
            if X_[column].dtype in ['float64', 'int64']:
                rolling_mean = X_[column].rolling(window=self.window).mean()
                rolling_std = X_[column].rolling(window=self.window).std()
                # Avoid division by zero
                valid_std = rolling_std.copy()
                valid_std[valid_std == 0] = 1
                z_scores = np.abs((X_[column] - rolling_mean) / valid_std)
                X_[column] = X_[column].mask(z_scores > self.threshold, rolling_mean)

        return X_
  
# Handle highly correlated features
class CorrelationHandler(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95, keep_columns=['Close']):
        self.threshold = threshold
        self.columns_to_drop = None
        self.keep_columns = keep_columns

    def fit(self, X, y=None):
        # handle high correlation of continues columns
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.columns_to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]

        # check all columns correlation with the target variable, if nan - remove (doesn't help for prediction later)
        if 'Transaction_Sharpe' in X.columns:
            target_corr = X.corr()['Transaction_Sharpe']
            nan_corr_columns = target_corr[target_corr.isna()].index.tolist()
            self.columns_to_drop.extend(nan_corr_columns)
    
        for col in self.keep_columns:
            if col in self.columns_to_drop:
                self.columns_to_drop.remove(col)

        if 'Transaction_Sharpe' in self.columns_to_drop:
            self.columns_to_drop.remove('Transaction_Sharpe')

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
def create_stock_data_pipeline(ticker_symbol, start_date, end_date, risk_free_rate):
    return Pipeline([
        ('data_fetcher', DataFetcher(ticker_symbol, start_date, end_date)),
        ('indicator_calculator', IndicatorCalculator()),
        ('signal_calculator', SignalCalculator()),
        ('transaction_metrics_calculator', TransactionMetricsCalculator(risk_free_rate)),
    ])

def create_data_cleaning_pipeline(correlation_threshold = 0.95):
  return Pipeline([
      ('missing_value_handler', MissingValueHandler()),
      ('outlier_handler', OutlierHandler()),
      ('correlation_handler', CorrelationHandler(correlation_threshold))
  ])
