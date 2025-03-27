"""
Single stock with pipes 
change from collab trying to run for one stock

Run after setting up an environment - setup.py
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
from datetime import datetime

# ===============================================================================
### Create data directories if they don't exist
# ===============================================================================
os.makedirs('results', exist_ok=True)
# Get current date in YYYYMMDD format
current_date = datetime.now().strftime("%Y%m%d")
# Create date folder inside results
date_folder = f'results/{current_date}'
os.makedirs(date_folder, exist_ok=True)

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
      #    return ticker.history(start=self.start_date, end=self.end_date)
      except Exception as e:
        print(f"Error fetching data for ticker {self.ticker_symbol}: {e}")

        return pd.DataFrame() # return an empty df if an error occurs


# add market indicators
class IndicatorCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.indicators = [
            'psar', 'mfi', 'mvp'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        for indicator in self.indicators:
            try:
                if indicator == 'psar':
                    psar = ta.psar(data['High'], data['Low'], data['Close'])
                    data['PSAR'] = psar['PSARl_0.02_0.2']
                elif indicator == 'mfi':
                    data['MFI'] = ta.mfi(data['High'], data['Low'], data['Close'], data['Volume']).astype(float)
                # Calculate Moving Volatility Pattern (MVP)
                # Here we're assuming MVP as a simple moving average of squared returns
                elif indicator == 'mvp':
                  data['Returns'] = data['Close'].pct_change()                                                          # calculate the daily return
                  data['Volatility'] = data['Returns'].rolling(window=28).std() * np.sqrt(252)                          # calculate 28-day rolling volatility (daily annual volatility)
                  data['MVP'] = data['Volatility'].pct_change()                                                         # the pct change of the volatility, Rate of change of volatility
                  threshold = 0.01
                  data['MVP_Signal'] = np.where(abs(data['MVP']) < threshold, 0)                                        # buy when the abs mvp is almost 0
            except Exception as e:
                print(f"Error adding indicator {indicator}: {e}")
        return data

# calculate buy and sell signals based on psar indicator
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

# for each transaction calculate: volatility, return and sharpe ratio
class TransactionMetricsCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        volatility_list = []
        returns_list = []
        sharpe_ratio_list = []
        duration_list = []
        last_buy_index = None
        last_buy_price = None

        for i in range(len(data)):
            if not np.isnan(data['Buy'].iloc[i]):
                last_buy_index = i
                last_buy_price = data['Close'].iloc[i]

            if not np.isnan(data['Sell'].iloc[i]) and last_buy_index is not None:
                sell_date = data.index[i]
                buy_date = data.index[last_buy_index]
                duration = (sell_date - buy_date).days

                # Calculate raw return for this transaction
                returns = (data['Close'].iloc[i] - last_buy_price) / last_buy_price

                ###volatility = data['Close'].iloc[last_buy_index:i+1].std()
                # Calculate volatility (standard deviation of daily returns) for this transaction
                daily_returns = data['Close'].iloc[last_buy_index:i+1].pct_change().dropna()
                volatility = daily_returns.std()

                # Calculate Sharpe ratio for this transaction
                # Assuming risk_free_rate is annual
                transaction_risk_free_rate = self.risk_free_rate * (duration / 365)
                sharpe_ratio = (returns - transaction_risk_free_rate) / volatility if volatility != 0 else 0                        # all in the same scale: returns is the return ratio, the vplatility is pct

                volatility_list.append(volatility)
                returns_list.append(returns)
                sharpe_ratio_list.append(sharpe_ratio)
                duration_list.append(duration)

                last_buy_index = None
                last_buy_price = None

        data['Transaction_Volatility'] = pd.Series(volatility_list, index=data.index[data['Sell'].notna()])
        data['Transaction_Returns'] = pd.Series(returns_list, index=data.index[data['Sell'].notna()])
        data['Transaction_Sharpe'] = pd.Series(sharpe_ratio_list, index=data.index[data['Sell'].notna()])
        # transaction duration in days
        data['Transaction_Duration'] = pd.Series(duration_list, index=data.index[data['Sell'].notna()])

        return data

# ===============================================================================
# Data Cleaning Classes
## Pineline to clean the data, remove outliers, and handle correlation
# ===============================================================================
# use forward fill then backward filll (normal in financial data) on all continues columns
class MissingValueHandler(BaseEstimator, TransformerMixin):
  def __init__(self, fill_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'PSAR', 'MFI', 'MVP']):
    self.fill_columns = fill_columns

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X_ = X.copy()
    for col in self.fill_columns:
      if col in X_.columns:
        X_[col] = X_[col].fillna(method='ffill').fillna(method='bfill')

    # if 'Signal_Shift' in X_.columns:
    #   X_['Signal_Shift']= X_['Signal_Shift'].fillna(-1)
    for col in X_.columns:
      X_[col] = X_[col].fillna(-1)

    return X_
  
# handle outliers: Use Z-score to identify data points that deviate significantly from the mean - rolling eondow of 28 days
class OutlierHandler(BaseEstimator, TransformerMixin):
  def __init__(self,threshold=3, window=28, exclude_columns=['Buy', 'Sell', 'Close']):
    self.threshold = threshold
    self.window = window
    self.exclude_columns = exclude_columns

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X_ = X.copy()
    X_ = X_.dropna(axis =1, how = 'all')
    for column in X_.columns:
      if column in self.exclude_columns:
        continue
      
      if X_[column].dtype in ['float64', 'int64']:
        rolling_mean = X_[column].rolling(window=self.window).mean()
        rolling_std = X_[column].rolling(window=self.window).std()
        z_scores = np.abs((X_[column] - rolling_mean) / rolling_std)
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

    # check all columns correlation with the target variable, if nan - remove (doesnt help for prediction later)
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


# ===============================================================================
# LSTM Support Functions
# ===============================================================================
def train_lstm_model(X_train, y_train, X_val, y_val, params):
    """
    Train an LSTM model with specified parameters

    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - params: Dictionary with LSTM parameters (epochs, batch_size, units, etc.)

    Returns:
    - model: Trained LSTM model
    - mse: Mean squared error on validation data
    - rmse: Root mean squared error on validation data
    - y_pred: Predictions on validation data
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.optimizers import Adam

        # Get parameters with defaults
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        units = params.get('units', 50)
        learning_rate = params.get('learning_rate', 0.01)

        # Reshape data for LSTM [samples, time steps, features]
        features_count = X_train.shape[1]
        X_train_reshaped = X_train.reshape(-1, 1, features_count)
        X_val_reshaped = X_val.reshape(-1, 1, features_count)

        # Create and compile model
        model = Sequential([
            LSTM(units, activation='relu', input_shape=(1, features_count)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        # Train model
        history = model.fit(
            X_train_reshaped, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_reshaped, y_val),
            verbose=0
        )

        # Predict and evaluate
        y_pred = model.predict(X_val_reshaped)
        y_pred = y_pred.flatten()  # Flatten to match y_val shape
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)

        return model, mse, rmse, y_pred
    
    except ImportError:
        print("TensorFlow/Keras not available. LSTM model will be skipped.")
        return None, float('inf'), float('inf'), np.zeros_like(y_val)
       

"""# Train LSTM model using time series cross-validation with manual parameter search.
## Used in train_and_validate_models() function to handle LSTM model seperatly!!
"""
def train_lstm_model_with_cv(X_train_val_scaled, Y_train_val, tscv):
    """
    Train LSTM model using time series cross-validation with manual parameter search.

    Parameters:
    - X_train_val_scaled: Scaled features for training/validation
    - Y_train_val: Target variable for training/validation
    - tscv: Time series cross-validation splitter

    Returns:
    - Dictionary containing training results
    """
    try:
        best_mse_scores = []
        best_rmse_scores = []
        best_params = []
        best_model = None
        best_score = float('inf')
        best_model_prediction = None
        Y_val_best = None

        # Define LSTM parameter grid
        lstm_param_grid = [
            {'epochs': 50, 'batch_size': 32, 'units': 50, 'learning_rate': 0.01},
            {'epochs': 100, 'batch_size': 32, 'units': 50, 'learning_rate': 0.01},
            {'epochs': 50, 'batch_size': 64, 'units': 50, 'learning_rate': 0.01},
            {'epochs': 50, 'batch_size': 32, 'units': 100, 'learning_rate': 0.01}
        ]

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val_scaled)):
            print(f"\nTraining on fold {fold + 1} for LSTM...")

            # Split the data
            X_train, X_val = X_train_val_scaled[train_idx], X_train_val_scaled[val_idx]
            Y_train, Y_val = Y_train_val.iloc[train_idx], Y_train_val.iloc[val_idx]

            # Perform a manual grid search across LSTM parameters
            fold_best_mse = float('inf')
            fold_best_params = None
            fold_best_model = None
            fold_best_preds = None

            print(f"Performing manual grid search for LSTM on fold {fold + 1}...")

            for params in lstm_param_grid:
                try:
                    # Train LSTM model with current parameters
                    model, mse, rmse, y_pred = train_lstm_model(X_train, Y_train, X_val, Y_val, params)

                    print(f"Parameters: {params}, MSE: {mse:.4f}")

                    if mse < fold_best_mse:
                        fold_best_mse = mse
                        fold_best_params = params
                        fold_best_model = model
                        fold_best_preds = y_pred

                except Exception as e:
                    print(f"Error training LSTM with params {params}: {e}")
                    continue

            # If we found a working model
            if fold_best_model is not None:
                # Calculate RMSE
                fold_best_rmse = np.sqrt(fold_best_mse)

                # Save results
                best_mse_scores.append(fold_best_mse)
                best_rmse_scores.append(fold_best_rmse)
                best_params.append(fold_best_params)

                print(f"Best parameters found for fold {fold + 1}: {fold_best_params}")
                print(f"End of Fold {fold + 1} - MSE: {fold_best_mse:.4f}, RMSE: {fold_best_rmse:.4f}")

                # Update best model overall if this fold's model is better
                if fold_best_mse < best_score:
                    best_score = fold_best_mse
                    best_model = fold_best_model
                    best_model_prediction = fold_best_preds
                    Y_val_best = Y_val

        # Return results dictionary
        return {
            'best_mse_scores': best_mse_scores,
            'best_rmse_scores': best_rmse_scores,
            'best_params': best_params,
            'best_model': best_model,
            'best_model_prediction': best_model_prediction,
            'Y_val_best': Y_val_best
        }
    
    except Exception as e:
        print(f"Error in LSTM training: {e}")
        return {
            'best_mse_scores': [],
            'best_rmse_scores': [],
            'best_params': [],
            'best_model': None,
            'best_model_prediction': None,
            'Y_val_best': None
        }
    

# ===============================================================================
# Model Creation and Training
# ===============================================================================
# Six models we took from stage 1 of the article, each model with list af params to do grid search on (optimize)
"""
Support Vector Regression (SVR)
eXtreme Gradient Boosting (XGBoost)
Light Gradient Boosting Machine (LightGBM)
Random Forest regression (RF)
Gradient Boosting Regression (GBR)
Long Short Term Memory model (LSTM)
"""

def create_models():
    try:
        models = {
            'SVR': (SVR(), {
                'kernel': ['rbf', 'linear'],
                'C': [0.1, 1, 10],
                'epsilon': [0.01, 0.1]
            }),
            'XGBoost': (XGBRegressor(random_state=42), {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
            }),
            'LightGBM': (LGBMRegressor(random_state=42, verbose=-1), {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'force_row_wise': [True]
            }),
            'RandomForest': (RandomForestRegressor(random_state=42), {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
            }),
            'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
            }),
            'LSTM': (None, {})
        }
        return models
    except Exception as e:
        print(f"Error creating models: {e}")
        return {}


 # Create all models, split the data - 5 folds, choose best params then train and vlidate each model, save performance results and params
def train_and_validate_models(X_train_val, Y_train_val):
  scaler = StandardScaler()                                                      # Add scaling, the features are scaled to have zero mean and unit variance
  X_train_val_scaled = scaler.fit_transform(X_train_val)

  tscv = TimeSeriesSplit(n_splits=5)                                              # time series cross-validation with 5 folds, to ensure temporal order (sequence of events in time)
  models = create_models()
  results = {}

  for model_name, (model, params_grid) in models.items():                         # iterate each model
    print(f"Training {model_name}")

    # reshape the data for LSTM model
    if model_name == 'LSTM':
      results[model_name] = train_lstm_model_with_cv(X_train_val_scaled, Y_train_val, tscv)
      continue

    # Add timeout or max iterations for SVR
    if model_name == 'SVR':
      model.set_params(max_iter=1000)  # Limit iterations

    best_mse_scores = []                                                               # lists to store scores and parameters for the models
    best_rmse_scores = []
    best_params = []
    best_score = float('inf')
    best_model = None
    best_model_prediction = None
    Y_val_best = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val_scaled)):  # each fold: split the scaled data to train and val (using the tcsv indices)
      print(f"\nTraining on fold {fold + 1} for {model_name}...")
      # Split the data
      X_train, X_val = X_train_val_scaled[train_idx], X_train_val_scaled[val_idx]
      Y_train, Y_val = Y_train_val.iloc[train_idx], Y_train_val.iloc[val_idx]


      # Find the best params for the current model
      print(f"Performing GridSearchCV for {model_name} on fold {fold + 1}...")

      # Reduce the number of CV splits for SVR to speed up
      cv_splits = 2 if model_name == 'SVR' else 3

    #   grid_search = GridSearchCV(estimator=model, param_grid=params_grid, scoring='neg_mean_squared_error', cv=cv_splits, n_jobs=-1, verbose=1) # performs search over set of parameters
      from sklearn.model_selection import GridSearchCV
      grid_search = GridSearchCV(
         estimator=model,
         param_grid=params_grid,
         scoring='neg_mean_squared_error',
         cv=cv_splits,
         n_jobs=-1,
         verbose=1
        )

      try:
        grid_search.fit(X_train, Y_train)                                         # train the model for every combination of parameters on each training set of the fold
        print(f"Best parameters found for fold {fold + 1}: {grid_search.best_params_}")

        # Save the best params
        best_params.append(grid_search.best_params_)

        # Train the best model
        best_model_fold = grid_search.best_estimator_                             # the model that performs the best on the last search

        # Validate using the model
        print("Starting validation: ")
        Y_pred = best_model_fold.predict(X_val)                                   # use the best model after traind and validate

        # Calculate performance for this fold
        mse = mean_squared_error(Y_val, Y_pred)
        rmse = np.sqrt(mse)

        # Save
        best_mse_scores.append(mse)
        best_rmse_scores.append(rmse)

        print(f"End of Fold {fold + 1} - MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        # Update best model if this fold's model is better
        if mse < best_score:
          best_score = mse
          best_model = best_model_fold
          best_model_prediction = Y_pred
          Y_val_best= Y_val


      except Exception as e:
                print(f"Error in {model_name} training: {e}")
                continue

    results[model_name] = {                                                       # save the best model for each model type, the results and the parameters
        'best_mse_scores': best_mse_scores,
        'best_rmse_scores': best_rmse_scores,
        'best_params': best_params,
        'best_model': best_model,
        'best_model_prediction': best_model_prediction,
        'Y_val_best' : Y_val_best
    }

  return results   

# ===============================================================================
# Ensamble
# ===============================================================================
# Linearly Weighted Ensemble
def linearly_weighted_ensemble(models_results, X_test):
    """
    Create a linearly weighted ensemble prediction across different model types.

    Parameters:
    - models_results: Dictionary containing model results
    - X_test: Test data to make predictions on (DataFrame or NumPy array)

    Returns:
    - Final ensemble prediction
    """
    # Convert DataFrame to NumPy array if necessary
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    if hasattr(X_test, 'values'):
        X_test = X_test.values

    mae_values = []
    model_predictions = []

    # Calculate Mean Absolute Error (MAE) for each model
    for model_name, result in models_results.items():
        mae_value = np.mean(np.abs(result['best_model_prediction'] - result['Y_val_best']))
        mae_values.append(mae_value)

        # Prepare predictions based on model type
        if model_name.strip() == 'LSTM':
            # Reshape for LSTM
            X_test_lstm = prepare_lstm_data(X_test, time_steps=1)
            model_pred = result['best_model'].predict(X_test_lstm)
        else:
            # For other models
            model_pred = result['best_model'].predict(X_test)

        # Ensure 1D prediction
        model_predictions.append(model_pred.reshape(-1))

    # Calculate inverse MAE weights
    weights = [mae_value ** -1 for mae_value in mae_values]
    weights = np.array(weights) / np.sum(weights)

    # Compute the final ensemble prediction
    final_prediction = np.zeros(X_test.shape[0], dtype=np.float64)

    # Apply weighted predictions
    for pred, weight in zip(model_predictions, weights):
        final_prediction += weight * pred

    return final_prediction

############### CHECK IF WE NEED THIS FOR SURE AND CANNOT USE SOMETHING ELSE #################
### USED IN THE ENSEMBLE
def prepare_lstm_data(X, time_steps=1, features=None):
    """
    Prepare data for LSTM by reshaping it to fit the expected 3D shape.

    Parameters:
    - X: Input data, can be 1D, 2D, or already 3D.
    - time_steps: Number of time steps in the sequence. Default is 1 for non-time series data.
    - features: Number of features per time step. If None, it will be inferred from the data.

    Returns:
    - Reshaped data with shape (samples, time_steps, features)
    """
    # Convert DataFrame to NumPy array if necessary
    if hasattr(X, 'values'):
        X = X.values

    if len(X.shape) == 1:
        if features is None:
            features = 1  # Assuming single feature if not specified
        X = X.reshape(-1, time_steps, features)
    elif len(X.shape) == 2:
        samples, cols = X.shape
        if features is None:
            features = cols  # All columns are considered as features if not specified
        if cols % time_steps != 0:
            raise ValueError(f"Number of columns ({cols}) must be evenly divisible by time_steps ({time_steps})")
        X = X.reshape(samples, cols // features, features)
    else:
        raise ValueError(f"Input data must be 1D or 2D. Got {len(X.shape)}D data.")

    return X

# Equal Weights Ensemble
def equal_weighted_ensemble(models_results, X_test):
    """
    Calculate an equal weighted ensemble prediction.

    Parameters:
    - models_results: Dictionary containing model results
    - X_test: Test data to make predictions on (DataFrame or NumPy array)

    Returns:
    - Final ensemble prediction
    """
    # Convert DataFrame to NumPy array if necessary
    if isinstance(X_test, pd.DataFrame) or isinstance(X_test, pd.Series):
        X_test = X_test.values

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    model_predictions = []

    # Prepare predictions based on model type
    for model_name, result in models_results.items():
        if 'best_model' in result:
            if model_name.strip() == 'LSTM':
                # Reshape for LSTM
                X_test_lstm = prepare_lstm_data(X_test, time_steps=1)
                model_pred = result['best_model'].predict(X_test_lstm)
            else:
                # For other models
                model_pred = result['best_model'].predict(X_test)

            # Ensure 1D prediction
            model_predictions.append(model_pred.reshape(-1))

    if not model_predictions:
        raise ValueError("No predictions available for ensemble methods")

    # Calculate weight (equal for all models)
    weight = 1.0 / len(model_predictions)

    # Compute the final ensemble prediction
    final_prediction = np.zeros(X_test.shape[0], dtype=np.float64)

    # Apply weighted predictions
    for pred in model_predictions:
        final_prediction += weight * pred

    return final_prediction

# Gradient Boosting Decision Tree Ensemble
def gbdt_ensemble(models_results, X_train, X_test, Y_train_val):
    """
    Use GBDT to predict based on the predictions of base models.

    :param models_results: Dictionary containing model results, with 'best_model' key for each model.
    :param X_train: Training data features to generate meta-features for training GBDT.
    :param X_test: Test data features for final prediction.
    :param Y_train_val: Training labels for fitting the GBDT model.
    :return: GBDT ensemble prediction for test data.
    """
    scaler = StandardScaler()

    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use transform here, not fit_transform

    # Generate meta-features for training GBDT
    train_meta_features = []
    test_meta_features = []

    for model_name, result in models_results.items():
        model = result['best_model']
        if model_name.strip() == 'LSTM':
            X_train_lstm = prepare_lstm_data(X_train_scaled, time_steps=1)
            X_test_lstm = prepare_lstm_data(X_test_scaled, time_steps=1)
            train_pred = model.predict(X_train_lstm)
            test_pred = model.predict(X_test_lstm)
        else:
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)

        train_meta_features.append(train_pred.reshape(-1))
        test_meta_features.append(test_pred.reshape(-1))

    # Stack predictions as meta-features
    X_train_meta = np.column_stack(train_meta_features)
    X_test_meta = np.column_stack(test_meta_features)

    # Ensure consistency in number of samples
    if X_train_meta.shape[0] != Y_train_val.shape[0]:
        raise ValueError(f"Shape mismatch: X_train_meta {X_train_meta.shape[0]} vs Y_train_val {Y_train_val.shape[0]}")

    # Train GBDT on meta-features
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train_meta, Y_train_val)

    # Predict on test meta-features
    final_prediction = gb_model.predict(X_test_meta)

    return final_prediction

# ===============================================================================
# Three Ensambles Pipeline
# ===============================================================================
def ensemble_pipeline(models_results, X_train, X_test, Y_train, Y_test):
    """
    Pipeline to apply and compare different ensemble methods.

    :param models_results: Dictionary containing model results
    :param X_train: Training features
    :param X_test: Testing features
    :param Y_train: Training labels
    :param Y_test: Testing labels
    :return: Dictionary with ensemble results
    """
    ensemble_methods = {
        'linearly_weighted': linearly_weighted_ensemble,
        'equal_weighted': equal_weighted_ensemble,
        'gbdt': lambda results, x_test: gbdt_ensemble(results, X_train, x_test, Y_train)
    }

    results = {}

    for method_name, method in ensemble_methods.items():
        # Note: 'gbdt' method requires X_train and Y_train, others do not
        if method_name == 'gbdt':
            final_prediction = method(models_results, X_test)
        else:
            final_prediction = method(models_results, X_test)

        # Calculate performance metrics
        mse = mean_squared_error(Y_test, final_prediction)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, final_prediction)

        results[method_name] = {
            'prediction': final_prediction,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

    for method_name, metrics in results.items():
        print(f"\n{method_name} Results:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  R2: {metrics['r2']:.6f}")

    # Find the best performing method based on RMSE
    best_method = min(results.items(), key=lambda x: x[1]['rmse'])[0]
    print(f"\nBest Ensemble Method: {best_method}")
    print(f"Performance of {best_method}:")
    print(f"  MSE: {results[best_method]['mse']}")
    print(f"  RMSE: {results[best_method]['rmse']}")
    print(f"  R2: {results[best_method]['r2']}")

    return results

# ===============================================================================
# Full Pipeline For Single Stock
# ===============================================================================
def full_pipeline_for_single_stock(ticker_symbol, start_date, end_date, risk_free_rate = 0.02):
  # run first pipeline, fetch data
  pipeline = create_stock_data_pipeline(ticker_symbol, start_date, end_date, risk_free_rate)
  data = pipeline.fit_transform(pd.DataFrame())

  # run seconed pipeline, clean and proecess
  pipeline_clean = create_data_cleaning_pipeline()
  data_clean = pipeline_clean.fit_transform(data)

#   data_clean.to_csv(f'{ticker_symbol}_clean_data.csv')
  data_clean.to_csv(f'{date_folder}/{ticker_symbol}_clean_data.csv')


  # Split the data to train and test, create train and val the models
  X = data_clean.drop(columns=['Transaction_Sharpe'])
  Y = data_clean['Transaction_Sharpe']
  train_size = 0.8
  split_idx = int(len(data_clean)*train_size)

  X_train_val = X.iloc[:split_idx]
  Y_train_val = Y.iloc[:split_idx]

  X_test = X.iloc[split_idx:]
  Y_test = Y.iloc[split_idx:]

  results = train_and_validate_models(X_train_val, Y_train_val)

  # Ensamble -> Three ensambles pipeline (Linearly Weighted, Equal Weights, GBDT)
  # Run the pipeline
  ensemble_results = ensemble_pipeline(results, X_train_val, X_test, Y_train_val, Y_test)

  # Convert dictionary to DataFrame and save to CSV
  df = pd.DataFrame.from_dict(ensemble_results, orient='index')
  df.index.name = 'Method Name'  # Set the index name
  ###df.to_csv(f'{ticker_symbol}_results.csv')
  df.to_csv(f'{date_folder}/{ticker_symbol}_results.csv')

  #print(f"Results saved to '{ticker_symbol}_results.csv'")

  best_method = min(ensemble_results.items(), key=lambda x: x[1]['rmse'])[0]
  best_prediction = ensemble_results[best_method]['prediction']

  results_df = pd.DataFrame({
    'Close' : X_test.Close,
    'Buy': X_test.Buy,
    'Sell': X_test.Sell,
    'Actual_Sharpe': Y_test,
    'Best_Prediction': best_prediction
    })

  ###results_df.to_csv(f'{ticker_symbol}_ensamble_prediction_results.csv')
  results_df.to_csv(f'{date_folder}/{ticker_symbol}_ensamble_prediction_results.csv')

# ===============================================================================
# Final loop - call the pipeline for each ticker symbol
# ===============================================================================
stakeholder_data = pd.read_csv('final_tickers_score.csv')
top_10 = stakeholder_data.head(15)['Ticker'].tolist()
last_10 = stakeholder_data.tail(10)['Ticker'].tolist()

def is_valid_ticker(ticker):
  try:
    ticker_data = yf.Ticker(ticker)
    if not ticker_data.info or 'symbol' not in ticker_data.info:
      print(f"Ticker {ticker} is not valid.")
      return False
    return True
  except Exception as e:
    print(f"Error validating ticker {ticker}: {e}")
    return False

def get_top_valid_tickers(tickers, top_n=20):
  valid_tickers = []
  for ticker in tickers:
    if is_valid_ticker(ticker):
      valid_tickers.append(ticker)
    if len(valid_tickers) == top_n:
      break

  return valid_tickers

top_10_valid_tickers = get_top_valid_tickers(top_10, top_n=10)

for ticker in top_10_valid_tickers:
  full_pipeline_for_single_stock(ticker, "2020-01-01", "2024-01-01")

# ===============================================================================
# RUN ALL IN ORDER ON ONE STOCK
# ===============================================================================
# Pipeline 1 - 
# ticker_symbol = "OPK"
# start_date = "2020-01-01"
# end_date = "2024-01-01"
# risk_free_rate = 0.02  # 2% annual risk-free rate, you can adjust this value

# pipeline = create_stock_data_pipeline(ticker_symbol, start_date, end_date, risk_free_rate)
# data = pipeline.fit_transform(pd.DataFrame())
# data.to_csv(f'{ticker_symbol}_processed_data.csv')
# print(data.info())

# # Pipeline 2 - 
# pipeline_clean = create_data_cleaning_pipeline()
# data_clean = pipeline_clean.fit_transform(data)
# data_clean.to_csv(f'{ticker_symbol}_clean_data.csv')
# print(data_clean.info())

# # Check data edge values
# max(data_clean['Transaction_Sharpe'])
# min(data_clean['Transaction_Sharpe'])

# # Split the data to train and test
# print(f"Data_clean shape before splitting: {data_clean.shape}")
# # Define the features
# X = data_clean.drop(columns=['Transaction_Sharpe'])
# # Define the target variable
# Y = data_clean['Transaction_Sharpe']
# # Define pct for test / train+validation
# train_size = 0.8
# # Calculate the index where to split the data
# split_idx = int(len(data_clean)*train_size)
# print(f"Split index: {split_idx}")

# # Split the data to test and train/validation sets
# X_train_val = X.iloc[:split_idx]
# Y_train_val = Y.iloc[:split_idx]

# X_test = X.iloc[split_idx:]
# Y_test = Y.iloc[split_idx:]

# results = train_and_validate_models(X_train_val, Y_train_val)

# print(f"X_train_val shape: {X_train_val.shape}")
# print(f"Y_train_val shape: {Y_train_val.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"Y_test shape: {Y_test.shape}")

# # Print train/val results
# for model_name, model_data in results.items():
#   print(model_name)
#   for item in model_data.items():
#     print(item)