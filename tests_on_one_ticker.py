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

from Data_Cleaning_Pipelines import create_stock_data_pipeline, create_data_cleaning_pipeline
from Feature_Selection_and_Optimization import analyze_feature_importance, evaluate_feature_sets, validate_feature_consistency
from Full_Pipeline_With_Data import full_pipeline_for_single_stock
from Models_Creation_and_Training import train_and_validate_models
from Ensembles import ensemble_pipeline
from Logging_and_Validation import log_data_stats, verify_prediction_scale

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

full_pipeline_for_single_stock(logger, date_folder, current_date, 'OPK', "2013-01-01", "2024-01-01")

# # ===============================================================================
# # LSTM Support Functions
# # ===============================================================================
# def train_lstm_model(X_train, y_train, X_val, y_val, params):
#     """
#     Train an LSTM model with specified parameters

#     Parameters:
#     - X_train, y_train: Training data
#     - X_val, y_val: Validation data
#     - params: Dictionary with LSTM parameters (epochs, batch_size, units, etc.)

#     Returns:
#     - model: Trained LSTM model
#     - mse: Mean squared error on validation data
#     - rmse: Root mean squared error on validation data
#     - y_pred: Predictions on validation data
#     """
#     try:
#         from tensorflow.keras.models import Sequential
#         from tensorflow.keras.layers import LSTM, Dense
#         from tensorflow.keras.optimizers import Adam

#         # Get parameters with defaults
#         epochs = params.get('epochs', 100)
#         batch_size = params.get('batch_size', 32)
#         units = params.get('units', 50)
#         learning_rate = params.get('learning_rate', 0.01)

#         # Reshape data for LSTM [samples, time steps, features]
#         features_count = X_train.shape[1]
#         X_train_reshaped = X_train.reshape(-1, 1, features_count)
#         X_val_reshaped = X_val.reshape(-1, 1, features_count)

#         # Create and compile model
#         model = Sequential([
#             LSTM(units, activation='relu', input_shape=(1, features_count)),
#             Dense(1)
#         ])
#         model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

#         # Train model
#         history = model.fit(
#             X_train_reshaped, y_train,
#             epochs=epochs,
#             batch_size=batch_size,
#             validation_data=(X_val_reshaped, y_val),
#             verbose=0
#         )

#         # Predict and evaluate
#         y_pred = model.predict(X_val_reshaped)
#         y_pred = y_pred.flatten()  # Flatten to match y_val shape
#         mse = mean_squared_error(y_val, y_pred)
#         rmse = np.sqrt(mse)

#         return model, mse, rmse, y_pred
    
#     except ImportError:
#         print("TensorFlow/Keras not available. LSTM model will be skipped.")
#         return None, float('inf'), float('inf'), np.zeros_like(y_val)
       

# """# Train LSTM model using time series cross-validation with manual parameter search.
# ## Used in train_and_validate_models() function to handle LSTM model seperatly!!
# """
# def train_lstm_model_with_cv(X_train_val_scaled, Y_train_val, tscv):
#     """
#     Train LSTM model using time series cross-validation with manual parameter search.

#     Parameters:
#     - X_train_val_scaled: Scaled features for training/validation
#     - Y_train_val: Target variable for training/validation
#     - tscv: Time series cross-validation splitter

#     Returns:
#     - Dictionary containing training results
#     """
#     try:
#         best_mse_scores = []
#         best_rmse_scores = []
#         best_params = []
#         best_model = None
#         best_score = float('inf')
#         best_model_prediction = None
#         Y_val_best = None

#         # Define LSTM parameter grid
#         lstm_param_grid = [
#             {'epochs': 50, 'batch_size': 32, 'units': 50, 'learning_rate': 0.01},
#             {'epochs': 100, 'batch_size': 32, 'units': 50, 'learning_rate': 0.01},
#             {'epochs': 50, 'batch_size': 64, 'units': 50, 'learning_rate': 0.01},
#             {'epochs': 50, 'batch_size': 32, 'units': 100, 'learning_rate': 0.01}
#         ]

#         for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val_scaled)):
#             print(f"\nTraining on fold {fold + 1} for LSTM...")

#             # Split the data
#             X_train, X_val = X_train_val_scaled[train_idx], X_train_val_scaled[val_idx]
#             #Y_train, Y_val = Y_train_val.iloc[train_idx], Y_train_val.iloc[val_idx]
#             Y_train, Y_val = Y_train_val[train_idx], Y_train_val[val_idx]

#             # Perform a manual grid search across LSTM parameters
#             fold_best_mse = float('inf')
#             fold_best_params = None
#             fold_best_model = None
#             fold_best_preds = None

#             print(f"Performing manual grid search for LSTM on fold {fold + 1}...")

#             for params in lstm_param_grid:
#                 try:
#                     # Train LSTM model with current parameters
#                     model, mse, rmse, y_pred = train_lstm_model(X_train, Y_train, X_val, Y_val, params)

#                     print(f"Parameters: {params}, MSE: {mse:.4f}")

#                     if mse < fold_best_mse:
#                         fold_best_mse = mse
#                         fold_best_params = params
#                         fold_best_model = model
#                         fold_best_preds = y_pred

#                 except Exception as e:
#                     print(f"Error training LSTM with params {params}: {e}")
#                     continue

#             # If we found a working model
#             if fold_best_model is not None:
#                 # Calculate RMSE
#                 fold_best_rmse = np.sqrt(fold_best_mse)

#                 # Save results
#                 best_mse_scores.append(fold_best_mse)
#                 best_rmse_scores.append(fold_best_rmse)
#                 best_params.append(fold_best_params)

#                 print(f"Best parameters found for fold {fold + 1}: {fold_best_params}")
#                 print(f"End of Fold {fold + 1} - MSE: {fold_best_mse:.4f}, RMSE: {fold_best_rmse:.4f}")

#                 # Update best model overall if this fold's model is better
#                 if fold_best_mse < best_score:
#                     best_score = fold_best_mse
#                     best_model = fold_best_model
#                     best_model_prediction = fold_best_preds
#                     Y_val_best = Y_val

#         # Return results dictionary
#         return {
#             'best_mse_scores': best_mse_scores,
#             'best_rmse_scores': best_rmse_scores,
#             'best_params': best_params,
#             'best_model': best_model,
#             'best_model_prediction': best_model_prediction,
#             'Y_val_best': Y_val_best
#         }
    
#     except Exception as e:
#         print(f"Error in LSTM training: {e}")
#         return {
#             'best_mse_scores': [],
#             'best_rmse_scores': [],
#             'best_params': [],
#             'best_model': None,
#             'best_model_prediction': None,
#             'Y_val_best': None
#         }

# def create_models():
#     try:
#         models = {
#             'SVR': (SVR(), {
#                 'kernel': ['rbf', 'linear'],
#                 'C': [0.1, 1, 10],
#                 'epsilon': [0.01, 0.1]
#             }),
#             'XGBoost': (XGBRegressor(random_state=42), {
#                     'n_estimators': [100, 200],
#                     'max_depth': [3, 5, 7],
#                     'learning_rate': [0.01, 0.1]
#             }),
#             'LightGBM': (LGBMRegressor(random_state=42, verbose=-1), {
#                     'n_estimators': [100, 200],
#                     'max_depth': [3, 5, 7],
#                     'learning_rate': [0.01, 0.1],
#                     'force_row_wise': [True]
#             }),
#             'RandomForest': (RandomForestRegressor(random_state=42), {
#                     'n_estimators': [100, 200],
#                     'max_depth': [3, 5, 7],
#                     'min_samples_split': [2, 5, 10]
#             }),
#             'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
#                     'n_estimators': [100, 200],
#                     'max_depth': [3, 5, 7],
#                     'learning_rate': [0.01, 0.1]
#             }),
#             'LSTM': (None, {})
#         }
#         return models
#     except Exception as e:
#         print(f"Error creating models: {e}")
#         return {}


#  # Create all models, split the data - 5 folds, choose best params then train and vlidate each model, save performance results and params
# # def train_and_validate_models(X_train_val, Y_train_val):
# #   # Scale the features
# #   scaler = StandardScaler()                                                      # Add scaling, the features are scaled to have zero mean and unit variance
# #   X_train_val_scaled = scaler.fit_transform(X_train_val)

# #   # Add target scaling
# #   target_scaler = StandardScaler()
# #   Y_train_val_scaled = target_scaler.fit_transform(Y_train_val.values.reshape(-1, 1)).flatten()
  
# #   # Save the scaler for later use
# #   # This is important for inverse transformation
# #   global shared_target_scaler
# #   shared_target_scaler = target_scaler


# #   tscv = TimeSeriesSplit(n_splits=5)                                              # time series cross-validation with 5 folds, to ensure temporal order (sequence of events in time)
# #   models = create_models()
# #   results = {}

# #   for model_name, (model, params_grid) in models.items():                         # iterate each model
# #     print(f"Training {model_name}")

# #     # reshape the data for LSTM model
# #     if model_name == 'LSTM':
# #       results[model_name] = train_lstm_model_with_cv(X_train_val_scaled, Y_train_val_scaled, tscv)
# #       continue

# #     # Add timeout or max iterations for SVR
# #     if model_name == 'SVR':
# #       model.set_params(max_iter=1000)  # Limit iterations

# #     best_mse_scores = []                                                               # lists to store scores and parameters for the models
# #     best_rmse_scores = []
# #     best_params = []
# #     best_score = float('inf')
# #     best_model = None
# #     best_model_prediction = None
# #     Y_val_best = None

# #     for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val_scaled)):  # each fold: split the scaled data to train and val (using the tcsv indices)
# #       print(f"\nTraining on fold {fold + 1} for {model_name}...")
# #       # Split the data
# #       X_train, X_val = X_train_val_scaled[train_idx], X_train_val_scaled[val_idx]
# #       #Y_train, Y_val = Y_train_val_scaled.iloc[train_idx], Y_train_val_scaled.iloc[val_idx]
# #       Y_train, Y_val = Y_train_val_scaled[train_idx], Y_train_val_scaled[val_idx]


# #       # Find the best params for the current model
# #       print(f"Performing GridSearchCV for {model_name} on fold {fold + 1}...")

# #       # Reduce the number of CV splits for SVR to speed up
# #       cv_splits = 2 if model_name == 'SVR' else 3

# #     #   grid_search = GridSearchCV(estimator=model, param_grid=params_grid, scoring='neg_mean_squared_error', cv=cv_splits, n_jobs=-1, verbose=1) # performs search over set of parameters
# #       from sklearn.model_selection import GridSearchCV
# #       grid_search = GridSearchCV(
# #          estimator=model,
# #          param_grid=params_grid,
# #          scoring='neg_mean_squared_error',
# #          cv=cv_splits,
# #          n_jobs=-1,
# #          verbose=1
# #         )

# #       try:
# #         grid_search.fit(X_train, Y_train)                                         # train the model for every combination of parameters on each training set of the fold
# #         print(f"Best parameters found for fold {fold + 1}: {grid_search.best_params_}")

# #         # Save the best params
# #         best_params.append(grid_search.best_params_)

# #         # Train the best model
# #         best_model_fold = grid_search.best_estimator_                             # the model that performs the best on the last search

# #         # Validate using the model
# #         print("Starting validation: ")
# #         Y_pred = best_model_fold.predict(X_val)                                   # use the best model after traind and validate

# #         # Calculate performance for this fold
# #         mse = mean_squared_error(Y_val, Y_pred)
# #         rmse = np.sqrt(mse)

# #         # Save
# #         best_mse_scores.append(mse)
# #         best_rmse_scores.append(rmse)

# #         print(f"End of Fold {fold + 1} - MSE: {mse:.4f}, RMSE: {rmse:.4f}")

# #         # Update best model if this fold's model is better
# #         if mse < best_score:
# #           best_score = mse
# #           best_model = best_model_fold
# #           best_model_prediction = Y_pred
# #           Y_val_best= Y_val


# #       except Exception as e:
# #                 print(f"Error in {model_name} training: {e}")
# #                 continue

# #     results[model_name] = {                                                       # save the best model for each model type, the results and the parameters
# #         'best_mse_scores': best_mse_scores,
# #         'best_rmse_scores': best_rmse_scores,
# #         'best_params': best_params,
# #         'best_model': best_model,
# #         'best_model_prediction': best_model_prediction,
# #         'Y_val_best' : Y_val_best
# #     }

# #   return results   

# # ===============================================================================
# # Ensamble
# # ===============================================================================
# # Linearly Weighted Ensemble
# # def linearly_weighted_ensemble(models_results, X_test):
# #     """
# #     Create a linearly weighted ensemble prediction across different model types.

# #     Parameters:
# #     - models_results: Dictionary containing model results
# #     - X_test: Test data to make predictions on (DataFrame or NumPy array)

# #     Returns:
# #     - Final ensemble prediction
# #     """
# #     # Convert DataFrame to NumPy array if necessary
# #     scaler = StandardScaler()
# #     X_test = scaler.fit_transform(X_test)
# #     if hasattr(X_test, 'values'):
# #         X_test = X_test.values

# #     mae_values = []
# #     model_predictions = []

# #     # Calculate Mean Absolute Error (MAE) for each model
# #     for model_name, result in models_results.items():
# #         mae_value = np.mean(np.abs(result['best_model_prediction'] - result['Y_val_best']))
# #         mae_values.append(mae_value)

# #         # Prepare predictions based on model type
# #         if model_name.strip() == 'LSTM':
# #             # Reshape for LSTM
# #             X_test_lstm = prepare_lstm_data(X_test, time_steps=1)
# #             model_pred = result['best_model'].predict(X_test_lstm)
# #         else:
# #             # For other models
# #             model_pred = result['best_model'].predict(X_test)

# #         # Ensure 1D prediction
# #         model_predictions.append(model_pred.reshape(-1))

# #     # Calculate inverse MAE weights
# #     weights = [mae_value ** -1 for mae_value in mae_values]
# #     weights = np.array(weights) / np.sum(weights)

# #     # Compute the final ensemble prediction
# #     final_prediction = np.zeros(X_test.shape[0], dtype=np.float64)

# #     # Apply weighted predictions
# #     for pred, weight in zip(model_predictions, weights):
# #         final_prediction += weight * pred
    
# #     # Inverse transform to get back to original scale
# #     final_prediction = shared_target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()

# #     return final_prediction

# # ############### CHECK IF WE NEED THIS FOR SURE AND CANNOT USE SOMETHING ELSE #################
# # ### USED IN THE ENSEMBLE
# # def prepare_lstm_data(X, time_steps=1, features=None):
# #     """
# #     Prepare data for LSTM by reshaping it to fit the expected 3D shape.

# #     Parameters:
# #     - X: Input data, can be 1D, 2D, or already 3D.
# #     - time_steps: Number of time steps in the sequence. Default is 1 for non-time series data.
# #     - features: Number of features per time step. If None, it will be inferred from the data.

# #     Returns:
# #     - Reshaped data with shape (samples, time_steps, features)
# #     """
# #     # Convert DataFrame to NumPy array if necessary
# #     if hasattr(X, 'values'):
# #         X = X.values

# #     if len(X.shape) == 1:
# #         if features is None:
# #             features = 1  # Assuming single feature if not specified
# #         X = X.reshape(-1, time_steps, features)
# #     elif len(X.shape) == 2:
# #         samples, cols = X.shape
# #         if features is None:
# #             features = cols  # All columns are considered as features if not specified
# #         if cols % time_steps != 0:
# #             raise ValueError(f"Number of columns ({cols}) must be evenly divisible by time_steps ({time_steps})")
# #         X = X.reshape(samples, cols // features, features)
# #     else:
# #         raise ValueError(f"Input data must be 1D or 2D. Got {len(X.shape)}D data.")

# #     return X

# # # Equal Weights Ensemble
# # def equal_weighted_ensemble(models_results, X_test):
# #     """
# #     Calculate an equal weighted ensemble prediction.

# #     Parameters:
# #     - models_results: Dictionary containing model results
# #     - X_test: Test data to make predictions on (DataFrame or NumPy array)

# #     Returns:
# #     - Final ensemble prediction
# #     """
# #     # Convert DataFrame to NumPy array if necessary
# #     if isinstance(X_test, pd.DataFrame) or isinstance(X_test, pd.Series):
# #         X_test = X_test.values

# #     scaler = StandardScaler()
# #     X_test = scaler.fit_transform(X_test)

# #     model_predictions = []

# #     # Prepare predictions based on model type
# #     for model_name, result in models_results.items():
# #         if 'best_model' in result:
# #             if model_name.strip() == 'LSTM':
# #                 # Reshape for LSTM
# #                 X_test_lstm = prepare_lstm_data(X_test, time_steps=1)
# #                 model_pred = result['best_model'].predict(X_test_lstm)
# #             else:
# #                 # For other models
# #                 model_pred = result['best_model'].predict(X_test)

# #             # Ensure 1D prediction
# #             model_predictions.append(model_pred.reshape(-1))

# #     if not model_predictions:
# #         raise ValueError("No predictions available for ensemble methods")

# #     # Calculate weight (equal for all models)
# #     weight = 1.0 / len(model_predictions)

# #     # Compute the final ensemble prediction
# #     final_prediction = np.zeros(X_test.shape[0], dtype=np.float64)

# #     # Apply weighted predictions
# #     for pred in model_predictions:
# #         final_prediction += weight * pred
    
# #     # Inverse transform to get back to original scale
# #     final_prediction = shared_target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()

# #     return final_prediction

# # # Gradient Boosting Decision Tree Ensemble
# # def gbdt_ensemble(models_results, X_train, X_test, Y_train_val):
# #     """
# #     Use GBDT to predict based on the predictions of base models.

# #     :param models_results: Dictionary containing model results, with 'best_model' key for each model.
# #     :param X_train: Training data features to generate meta-features for training GBDT.
# #     :param X_test: Test data features for final prediction.
# #     :param Y_train_val: Training labels for fitting the GBDT model.
# #     :return: GBDT ensemble prediction for test data.
# #     """
# #     scaler = StandardScaler()

# #     # Scale the data
# #     X_train_scaled = scaler.fit_transform(X_train)
# #     X_test_scaled = scaler.transform(X_test)  # Use transform here, not fit_transform

# #     # Generate meta-features for training GBDT
# #     train_meta_features = []
# #     test_meta_features = []

# #     for model_name, result in models_results.items():
# #         model = result['best_model']
# #         if model_name.strip() == 'LSTM':
# #             X_train_lstm = prepare_lstm_data(X_train_scaled, time_steps=1)
# #             X_test_lstm = prepare_lstm_data(X_test_scaled, time_steps=1)
# #             train_pred = model.predict(X_train_lstm)
# #             test_pred = model.predict(X_test_lstm)
# #         else:
# #             train_pred = model.predict(X_train_scaled)
# #             test_pred = model.predict(X_test_scaled)

# #         train_meta_features.append(train_pred.reshape(-1))
# #         test_meta_features.append(test_pred.reshape(-1))

# #     # Stack predictions as meta-features
# #     X_train_meta = np.column_stack(train_meta_features)
# #     X_test_meta = np.column_stack(test_meta_features)

# #     # Ensure consistency in number of samples
# #     if X_train_meta.shape[0] != Y_train_val.shape[0]:
# #         raise ValueError(f"Shape mismatch: X_train_meta {X_train_meta.shape[0]} vs Y_train_val {Y_train_val.shape[0]}")

# #     # Train GBDT on meta-features
# #     gb_model = GradientBoostingRegressor(
# #         n_estimators=200,
# #         learning_rate=0.05,
# #         max_depth=5,
# #         min_samples_split=10,
# #         min_samples_leaf=5,
# #         subsample=0.8,
# #         random_state=42
# #     )
# #     gb_model.fit(X_train_meta, Y_train_val)

# #     # Predict on test meta-features
# #     final_prediction = gb_model.predict(X_test_meta)

# #     # Inverse transform to get back to original scale
# #     final_prediction = shared_target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()

# #     return final_prediction

# # # ===============================================================================
# # # Three Ensambles Pipeline
# # # ===============================================================================
# # def ensemble_pipeline(models_results, X_train, X_test, Y_train, Y_test):
# #     """
# #     Pipeline to apply and compare different ensemble methods.

# #     :param models_results: Dictionary containing model results
# #     :param X_train: Training features
# #     :param X_test: Testing features
# #     :param Y_train: Training labels
# #     :param Y_test: Testing labels
# #     :return: Dictionary with ensemble results
# #     """
# #     ensemble_methods = {
# #         'linearly_weighted': linearly_weighted_ensemble,
# #         'equal_weighted': equal_weighted_ensemble,
# #         'gbdt': lambda results, x_test: gbdt_ensemble(results, X_train, x_test, Y_train)
# #     }

# #     results = {}

# #     for method_name, method in ensemble_methods.items():
# #         # Note: 'gbdt' method requires X_train and Y_train, others do not
# #         if method_name == 'gbdt':
# #             final_prediction = method(models_results, X_test)
# #         else:
# #             final_prediction = method(models_results, X_test)

# #         # Calculate performance metrics against unscaled targets
# #         # The predictions are already inverse transformed in the ensemble methods
# #         mse = mean_squared_error(Y_test, final_prediction)
# #         rmse = np.sqrt(mse)
# #         r2 = r2_score(Y_test, final_prediction)

# #         results[method_name] = {
# #             'prediction': final_prediction,
# #             'mse': mse,
# #             'rmse': rmse,
# #             'r2': r2
# #         }

# #     for method_name, metrics in results.items():
# #         print(f"\n{method_name} Results:")
# #         print(f"  RMSE: {metrics['rmse']:.6f}")
# #         print(f"  MSE: {metrics['mse']:.6f}")
# #         print(f"  R2: {metrics['r2']:.6f}")

# #     # Find the best performing method based on RMSE
# #     best_method = min(results.items(), key=lambda x: x[1]['rmse'])[0]
# #     print(f"\nBest Ensemble Method: {best_method}")
# #     print(f"Performance of {best_method}:")
# #     print(f"  MSE: {results[best_method]['mse']}")
# #     print(f"  RMSE: {results[best_method]['rmse']}")
# #     print(f"  R2: {results[best_method]['r2']}")

# #     return results

# # ===============================================================================
# # Full Pipeline For Single Stock
# # ===============================================================================
# def full_pipeline_for_single_stock1(ticker_symbol, start_date, end_date, risk_free_rate = 0.02):
#     print(f"\n{'='*50}")
#     print(f"STARTING PIPELINE FOR TICKER {ticker_symbol}")
#     print(f"{'='*50}")

#     # Run first pipeline, fetch data
#     print(f"\n{'-'*30}\nFetching and processing data for {ticker_symbol}\n{'-'*30}")
#     pipeline = create_stock_data_pipeline(ticker_symbol, start_date, end_date, risk_free_rate)
#     data = pipeline.fit_transform(pd.DataFrame())
#     log_data_stats(data, f"{ticker_symbol} raw data", log_head=True)
#     data.to_csv(f'{date_folder}/{ticker_symbol}_data.csv')

#     # Run second pipeline, clean and process
#     print(f"\n{'-'*30}\nCleaning data for {ticker_symbol}\n{'-'*30}")
#     pipeline_clean = create_data_cleaning_pipeline()
#     data_clean = pipeline_clean.fit_transform(data)
#     log_data_stats(data_clean, f"{ticker_symbol} cleaned data", log_head=True)

#     #   data_clean.to_csv(f'{ticker_symbol}_clean_data.csv')
#     data_clean.to_csv(f'{date_folder}/{ticker_symbol}_clean_data.csv')

#     # Split the data to train and test, create train and val the models
#     print(f"\n{'-'*30}\nSplitting data for {ticker_symbol}\n{'-'*30}")
#     X = data_clean.drop(columns=['Transaction_Sharpe'])
#     Y = data_clean['Transaction_Sharpe']
#     train_size = 0.8
#     split_idx = int(len(data_clean)*train_size)

#     X_train_val = X.iloc[:split_idx]
#     Y_train_val = Y.iloc[:split_idx]
#     X_test = X.iloc[split_idx:]
#     Y_test = Y.iloc[split_idx:]

#     log_data_stats(X_train_val, f"{ticker_symbol} X_train_val", include_stats=False)
#     log_data_stats(Y_train_val, f"{ticker_symbol} Y_train_val", include_stats=True, log_head=False)
#     log_data_stats(X_test, f"{ticker_symbol} X_test", include_stats=False)
#     log_data_stats(Y_test, f"{ticker_symbol} Y_test", include_stats=True, log_head=False)

#     # Feature selection
#     print(f"\n{'-'*30}\nPerforming feature selection for {ticker_symbol}\n{'-'*30}")
#     importance_df = analyze_feature_importance(X_train_val, Y_train_val)
#     print("Feature Importance:")
#     print(importance_df.head(10))

#     # Evaluate different feature sets
#     feature_results = evaluate_feature_sets(X_train_val, Y_train_val, X_test, Y_test)
#     print("\nFeature Selection Method Comparison:")
#     print(feature_results['average_results'][['Feature_Method', 'Num_Features', 'RMSE', 'R2']])

#     # Select best feature set based on results
#     best_method = feature_results['best_method']
#     best_features = feature_results['best_features']
#     print(f"\nBest feature set: {best_method} with {len(best_features)} features")
#     print(f"Selected features: {best_features[:10]}{'...' if len(best_features) > 10 else ''}")

#     # Use these features for model training
#     X_train_val_selected = X_train_val[best_features]
#     X_test_selected = X_test[best_features]

#     # Validate feature consistency
#     print(f"\n{'-'*30}\nValidating feature consistency\n{'-'*30}")
#     validate_feature_consistency(X_train_val, X_train_val_selected, best_features)
#     validate_feature_consistency(X_test, X_test_selected, best_features)

#     log_data_stats(X_train_val_selected, f"{ticker_symbol} X_train_val_selected", include_stats=False)
#     log_data_stats(X_test_selected, f"{ticker_symbol} X_test_selected", include_stats=False)

#     # Model training
#     print(f"\n{'-'*30}\nTraining models for {ticker_symbol}\n{'-'*30}")
#     train_results = train_and_validate_models(X_train_val_selected, Y_train_val)
#     model_results = train_results['model_results']
#     target_scaler = train_results['target_scaler']
#     feature_scaler = train_results['feature_scaler']

#     # Ensemble prediction
#     ensemble_results = ensemble_pipeline(model_results, X_train_val_selected, X_test_selected, Y_train_val, Y_test, target_scaler, feature_scaler)

#     # Convert dictionary to DataFrame and save to CSV
#     df = pd.DataFrame.from_dict(ensemble_results, orient='index')
#     df.index.name = 'Method Name'  # Set the index name
#     ###df.to_csv(f'{ticker_symbol}_results.csv')
#     print(f"\n{'-'*30}\nSaving results for {ticker_symbol}\n{'-'*30}")
#     df.to_csv(f'{date_folder}/{ticker_symbol}_results.csv')

#     #print(f"Results saved to '{ticker_symbol}_results.csv'")

#     best_method = min(ensemble_results.items(), key=lambda x: x[1]['rmse'])[0]
#     best_prediction = ensemble_results[best_method]['prediction']

#     results_df = pd.DataFrame({
#         'Ticker' : ticker_symbol,
#         'Close' : X_test.Close,
#         'Buy': X_test.Buy,
#         'Sell': X_test.Sell,
#         'Actual_Sharpe': Y_test,
#         'Best_Prediction': best_prediction
#         })
#     log_data_stats(results_df, f"{ticker_symbol} final results", log_head=True)

#     ##results_df.to_csv(f'{ticker_symbol}_ensamble_prediction_results.csv')
#     results_df.to_csv(f'{date_folder}/{ticker_symbol}_ensemble_prediction_results.csv')
#     # Verify final prediction scale
#     verify_prediction_scale(Y_test, best_prediction, f"{ticker_symbol} best ensemble method")

#     print(f"\n{'='*50}")
#     print(f"COMPLETED PIPELINE FOR TICKER {ticker_symbol}")
#     print(f"{'='*50}")
        
# # full_pipeline_for_single_stock('OPK', "2013-01-01", "2024-01-01")


