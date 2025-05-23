# Models_Creation_and_Training.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import optuna
from optuna.integration import TFKerasPruningCallback

from scipy.stats import uniform, randint

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam

from joblib import parallel_backend

from Logging_and_Validation import log_data_stats, verify_prediction_scale

import json
import os
from datetime import datetime

# ===============================================================================
# Model Creation and Training
# ===============================================================================
def is_pca_transformed_data(X_data):
    """
    Check if the data appears to be PCA-transformed based on column names.
    
    Parameters:
    -----------
    X_data : DataFrame or ndarray
    
    Returns:
    --------
    bool : True if data appears to be PCA-transformed, False otherwise
    """
    if isinstance(X_data, pd.DataFrame):
        # Check if column names follow PC pattern
        return all(col.startswith('PC') for col in X_data.columns)
    return False

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
                'kernel': ['rbf'],  # Focus on RBF based on fold results
                'C': ('float', 5.0, 20.0),  # Narrowed from 0.1–20
                'epsilon': ('float', 0.05, 0.3),  # Narrowed from 0.01–0.3
                'gamma': ('float', 0.05, 0.5),  # Narrowed from 0.001–0.5
                'max_iter': [15000]
            }),
            'XGBoost': (XGBRegressor(random_state=42), {
                'n_estimators': ('int', 100, 300),  # Narrowed from 100–1000
                'max_depth': ('int', 3, 7),  # Narrowed from 3–15
                'learning_rate': ('float', 0.01, 0.1),  # Narrowed from 0.001–0.2
                'subsample': ('float', 0.7, 1.0),  # Narrowed from 0.5–1.0
                'colsample_bytree': ('float', 0.6, 0.9),  # Narrowed from 0.5–1.0
                'min_child_weight': ('float', 5, 20),  # Narrowed from 1–20
                'gamma': ('float', 0.3, 0.5),
                'reg_alpha': ('float', 0.3, 1.0),
                'reg_lambda': ('float', 0.1, 1.0)
            }),
            'LightGBM': (LGBMRegressor(random_state=42, verbose=-1), {
                'n_estimators': ('int', 100, 500),  # Narrowed from 100–1000
                'max_depth': ('int', 3, 10),  # Narrowed from 3–15
                'learning_rate': ('float', 0.01, 0.05),  # Narrowed from 0.001–0.2
                'subsample': ('float', 0.6, 1.0),  # Narrowed from 0.5–1.0
                'colsample_bytree': ('float', 0.6, 0.9),  # Narrowed from 0.5–1.0
                'num_leaves': ('int', 20, 100),  # Narrowed from 20–150
                'min_child_samples': ('int', 20, 50),  # Narrowed from 5–50
                'reg_alpha': ('float', 0.1, 0.7),
                'reg_lambda': ('float', 0.2, 0.6),
                'force_row_wise': [True]
            }),
            'RandomForest': (RandomForestRegressor(random_state=42), {
                'n_estimators': ('int', 200, 600),  # Narrowed from 100–1000
                'max_depth': ('int', 3, 10),  # Narrowed from 3–15
                'min_samples_split': ('int', 5, 15),  # Narrowed from 2–15
                'min_samples_leaf': ('int', 5, 10),  # Narrowed from 1–10
                'max_features': ['sqrt', 0.3, 0.5, 1.0],  # Removed log2
                'bootstrap': [True],  # Fixed to True
                'warm_start': [False]  # Fixed to False
            }),
            'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
                'n_estimators': ('int', 100, 500),  # Narrowed from 100–1000
                'max_depth': ('int', 3, 7),  # Narrowed from 3–15
                'learning_rate': ('float', 0.01, 0.05),  # Narrowed from 0.001–0.2
                'subsample': ('float', 0.6, 1.0),  # Narrowed from 0.5–1.0
                'min_samples_split': ('int', 2, 10),  # Narrowed from 2–15
                'min_samples_leaf': ('int', 5, 10),  # Narrowed from 1–10
                'max_features': ['sqrt', 0.3, 0.5],  # Removed log2, 0.7, 1.0
                'alpha': ('float', 0.1, 0.7)  # Narrowed from 0.1–0.9
            }),
            'LSTM': (None, {
                'epochs': ('int', 50, 100),  # Reduced from 50–150
                'batch_size': ('int', 16, 32),  # Reduced from 16–64
                'units': ('int', 8, 32),  # Reduced from 8–48
                'learning_rate': ('float', 0.001, 0.01),
                'dropout': ('float', 0.1, 0.3),  # Narrowed from 0.1–0.5
                'time_steps': ('int', 3, 5),  # Narrowed from 3–10
                'layers': ('int', 1, 1),  # Fixed to 1
                'bidirectional': [False],
                'use_batch_norm': [False]
            })
        }
        return models
    except Exception as e:
        print(f"Error creating models: {e}")
        return {}
    
def optimize_model_with_optuna(model, params_grid, X_train, Y_train, X_val, Y_val, model_name, logger, target_scaler, n_trials=30):
    def objective(trial):
        params = {}
        for param, value in params_grid.items():
            if isinstance(value, list):
                params[param] = trial.suggest_categorical(param, value)
            elif isinstance(value, tuple) and value[0] == 'float':
                params[param] = trial.suggest_float(param, value[1], value[2])
            elif isinstance(value, tuple) and value[0] == 'int':
                params[param] = trial.suggest_int(param, value[1], value[2])

        model_instance = model.__class__(**params)
        try:
            model_instance.fit(X_train, Y_train)
            Y_pred_scaled = model_instance.predict(X_val)
            Y_pred = target_scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()
            mse = mean_squared_error(Y_val, Y_pred)
            return mse
        except Exception as e:
            logger.error(f"Error in trial: {e}")
            return float('inf')

    study = optuna.create_study(direction='minimize')
    with parallel_backend('multiprocessing'):
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_params, study.best_value, study.best_trial

# def create_models():
#     """
#     Define models and their Optuna parameter search spaces.
#     """
#     models = {
#         'SVR': (SVR(), {
#             'kernel': ['rbf', 'linear'],
#             'C': lambda trial: trial.suggest_float('C', 0.1, 10, log=True),
#             'epsilon': lambda trial: trial.suggest_float('epsilon', 0.01, 0.2)
#         }),
#         'XGBoost': (XGBRegressor(random_state=42), {
#             'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
#             'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 8),
#             'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
#             'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0)
#         }),
#         'LightGBM': (LGBMRegressor(random_state=42, verbose=-1), {
#             'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
#             'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 8),
#             'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
#             'force_row_wise': [True]
#         }),
#         'RandomForest': (RandomForestRegressor(random_state=42), {
#             'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
#             'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 8),
#             'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 10)
#         }),
#         'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
#             'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
#             'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 8),
#             'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.005, 0.2, log=True)
#         }),
#         'LSTM': (None, {})  # Handled separately
#     }
#     return models


def train_and_validate_models(logger, X_train_val, Y_train_val, current_date, ticker_symbol, date_folder):
    logger.info(f"\n{'-'*30}\nInitializing model training\n{'-'*30}")

    # We assume X_train_val is already PCA-transformed data
    logger.info("Using PCA-transformed input data without additional scaling")

    # Create global scaler for final output
    target_scaler = RobustScaler()
    # The features are already scaled by RobustScaler and transformed by PCA
    X_train_val_scaled = X_train_val  # Use PCA data directly
    # Scale the target variable (same for both PCA and non-PCA)
    target_scaler.fit(Y_train_val.values.reshape(-1, 1))
    Y_train_val_scaled = target_scaler.transform(Y_train_val.values.reshape(-1, 1)).flatten()

    # Time series cross-validation with 5 folds, to ensure temporal order (sequence of events in time)
    tscv = TimeSeriesSplit(n_splits=5)                                   
    models = create_models()
    results = {}

    # Dictionary to store fold metrics for each model
    all_fold_metrics = {}

    # Iterate each model
    for model_name, (model, params_grid) in models.items():    
        logger.info(f"\n{'-'*30}\nTraining {model_name}\n{'-'*30}")                     

        best_mse_scores = []                                                               # lists to store scores and parameters for the models
        best_rmse_scores = []
        best_r2_scores = []
        best_params = []
        best_score = float('inf')
        best_model = None
        best_model_prediction = None
        Y_val_best = None

        # Create a list to store fold metrics for this model
        fold_metrics = []

        # Reshape the data for LSTM model
        if model_name == 'LSTM':
            logger.info(f"Handling LSTM model separately with special reshaping")
            lstm_results = train_lstm_model_with_cv(X_train_val, Y_train_val, tscv, None, target_scaler)
            results[model_name] = lstm_results
            #results[model_name] = train_lstm_model_with_cv(X_train_val, Y_train_val, tscv, None, target_scaler)

            # Extract LSTM fold metrics from the results
            if 'fold_metrics' in lstm_results:
                all_fold_metrics[model_name] = lstm_results['fold_metrics']
            continue

        # Each fold: split the data and to train and val (using the tcsv indices) then scale
        # For each fold, use the pre-scaled data with the global scalers
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)): 
            logger.info(f"\nFold {fold + 1} for {model_name}:")

            # Split the data
            X_train_fold = X_train_val_scaled.iloc[train_idx]
            X_val_fold = X_train_val_scaled.iloc[val_idx]

            Y_train_fold = Y_train_val_scaled[train_idx]
            Y_val_fold = Y_train_val_scaled[val_idx]
            # Original Y values for computing metrics
            Y_val_original = Y_train_val.iloc[val_idx].values

            logger.info(f"  Train fold shapes: X={X_train_fold.shape}, Y={Y_train_fold.shape}")
            logger.info(f"  Validation fold shapes: X={X_val_fold.shape}, Y={Y_val_fold.shape}")

            # Grid search
            logger.info(f"  Running Optuna optimization for {model_name} on fold {fold + 1}...")
            # Increased number of trials by 50% for all models
            n_trials = 30 if model_name != 'SVR' else 20  # increased from 40->60 and from 30->40

            try:
                best_params_fold, best_mse, _ = optimize_model_with_optuna(
                    model, params_grid, X_train_fold, Y_train_fold, X_val_fold, Y_val_original, model_name, logger, target_scaler, n_trials
                )
                best_params.append(best_params_fold)
                logger.info(f"  Best parameters: {best_params_fold}")

                best_model_fold = model.__class__(**best_params_fold)
                best_model_fold.fit(X_train_fold, Y_train_fold)
                Y_pred_scaled = best_model_fold.predict(X_val_fold)                                
            
                # Convert predictions back to original scale
                Y_pred = target_scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()

                # Scale verification
                min_ratio, max_ratio = verify_prediction_scale(logger, Y_val_original, Y_pred, f"{model_name} fold {fold+1}")

                # Calculate performance metrics (on original scale)
                mse = mean_squared_error(Y_val_original, Y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(Y_val_original, Y_pred)
                r2 = r2_score(Y_val_original, Y_pred)
                best_mse_scores.append(mse)
                best_rmse_scores.append(rmse)
                best_r2_scores.append(r2)
                logger.info(f"  Fold {fold + 1} - MSE: {mse:.4f}, RMSE: {rmse:.4f},  MAE: {mae:.4f}")

                # Store fold metrics
                fold_metrics.append({
                    'Fold': fold + 1,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'Min_Ratio': min_ratio,
                    'Max_Ratio': max_ratio,
                    'Parameters': str(best_params_fold)
                })


                # Update best model
                if mse < best_score:
                    best_score = mse
                    best_model = best_model_fold
                    best_model_prediction = Y_pred
                    Y_val_best= Y_val_original
                    logger.info(f"  New best model found with MSE: {mse:.4f}")


            except Exception as e:
                logger.error(f"  Error in {model_name} training on fold {fold + 1}: {e}")
                continue

        # Store fold metrics for this model
        if fold_metrics:
            all_fold_metrics[model_name] = fold_metrics
            
        # Save the best model for each model type, the results and the parameters
        results[model_name] = {                                                       
            'best_mse_scores': best_mse_scores,
            'best_rmse_scores': best_rmse_scores,
            'best_r2_scores': best_r2_scores,
            'best_params': best_params,
            'best_model': best_model,
            'best_model_prediction': best_model_prediction,
            'Y_val_best' : Y_val_best,
            'fold_metrics': fold_metrics
        }

        logger.info(f"\nSummary for {model_name}:")
        if best_mse_scores:
            avg_mse = np.mean(best_mse_scores)
            avg_rmse = np.mean(best_rmse_scores)
            avg_r2 = np.mean(best_r2_scores)
            logger.info(f"  Average MSE: {avg_mse:.4f}, Average RMSE: {avg_rmse:.4f}")
        else:
            logger.info(f"  No successful folds for {model_name}")
        
    # After all folds, retrain best model on all data
    logger.info("\nRetraining all models on full dataset...")

    # For each model (except LSTM which is handled separately)
    for model_name, result in results.items():
        if model_name != 'LSTM' and result.get('best_params') and len(result['best_params']) > 0:
            try:
                logger.info(f"Retraining {model_name} on full dataset...")
                # Get best parameters
                best_fold_idx = np.argmin(result['best_mse_scores'])
                best_params = result['best_params'][best_fold_idx]
                # Create a new model instance with the best parameters
                base_model = create_models()[model_name][0]
                for param, value in best_params.items():
                    setattr(base_model, param, value)
            
                # Fit on all training data
                base_model.fit(X_train_val_scaled, Y_train_val_scaled)

                # Validate the retrained model
                Y_retrain_pred_scaled = base_model.predict(X_train_val_scaled)
                Y_retrain_pred = target_scaler.inverse_transform(Y_retrain_pred_scaled.reshape(-1, 1)).flatten()
                retrain_mse = mean_squared_error(Y_train_val, Y_retrain_pred)
                retrain_rmse = np.sqrt(retrain_mse)
                retrain_r2 = r2_score(Y_train_val, Y_retrain_pred)

                logger.info(f"  Retrained {model_name} - MSE: {retrain_mse:.4f}, RMSE: {retrain_rmse:.4f}")
                min_ratio, max_ratio = verify_prediction_scale(logger, Y_train_val, Y_retrain_pred, f"{model_name} retrained")
            
                # Replace the best model
                results[model_name]['best_model'] = base_model
                 # Add retrained metrics
                results[model_name]['retrained_metrics'] = {
                    'MSE': retrain_mse,
                    'RMSE': retrain_rmse,
                    'R2': retrain_r2,
                    'Min_Ratio': min_ratio,
                    'Max_Ratio': max_ratio
                }
                logger.info(f"  Successfully retrained {model_name}")
                
            except Exception as e:
                logger.error(f"  Error retraining {model_name}: {e}")
                logger.error("  Keeping the best fold model instead")
                # Keep the fold model if retraining fails
                pass
    
    try:
        # Prepare data for metrics dataframe
        metrics_data = []

        for model_name, result in results.items():
            if not result or 'best_mse_scores' not in result or not result['best_mse_scores']:
                    logger.info(f"Skipping {model_name} - no valid results")
                    continue
            
            # Calculate metrics
            avg_mse = np.mean(result['best_mse_scores'])
            avg_rmse = np.mean(result['best_rmse_scores'])
            avg_r2 = np.mean(result['best_r2_scores'])
            
            # Find best fold
            best_fold_idx = np.argmin(result['best_mse_scores'])
            best_fold_mse = result['best_mse_scores'][best_fold_idx]
            best_fold_rmse = result['best_rmse_scores'][best_fold_idx]
            best_fold_r2 = result['best_r2_scores'][best_fold_idx]

            # Get best parameters
            best_fold_params = None
            if result.get('best_params') and len(result['best_params']) > 0:
                best_fold_params = str(result['best_params'][best_fold_idx])
            
            # Add to metrics data
            metrics_data.append({
                'Model': model_name,
                'Average_MSE': avg_mse,
                'Average_RMSE': avg_rmse,
                'Average_R2': avg_r2,
                'Best_Fold_MSE': best_fold_mse,
                'Best_Fold_RMSE': best_fold_rmse,
                'Best_Fold_R2': best_fold_r2,
                'Best_Parameters': best_fold_params
            })

        # Create dataframe
        metrics_df = pd.DataFrame(metrics_data)

        # Add ranking column
        metrics_df = metrics_df.sort_values('Average_MSE')
        metrics_df['Rank'] = range(1, len(metrics_df) + 1)
        
        # Save to CSV
        drive_path = r"G:\.shortcut-targets-by-id\19E5zLX5V27tgCL2D8EysE2nKWTQAEUlg\Investment portfolio management system\code_results\results/"
        # Create date folder inside Google Drive path
        drive_date_folder = os.path.join(drive_path, current_date)
        # Create directory if it doesn't exist
        os.makedirs(drive_date_folder,exist_ok=True)
        try:
            metrics_df.to_csv(f'{date_folder}/{ticker_symbol}_training_validation_results.csv')
            metrics_df.to_csv(os.path.join(drive_date_folder, f"{ticker_symbol}_training_validation_results.csv"))
            
            # Save detailed fold metrics for each model separately
            for model_name, fold_metrics in all_fold_metrics.items():
                if fold_metrics:
                    # Create DataFrame for the fold metrics
                    fold_df = pd.DataFrame(fold_metrics)               
                    # Save to local folder
                    fold_df.to_csv(f'{date_folder}/{ticker_symbol}_{model_name}_fold_metrics.csv', index=False)                 
                    # Save to Google Drive folder
                    fold_df.to_csv(os.path.join(drive_date_folder, f"{ticker_symbol}_{model_name}_fold_metrics.csv"), index=False)

            logger.info(f"Saved clean data for {ticker_symbol} to folders")

        except Exception as e:
            logger.error(f"Error saving to Google Drive: {e}")
            os.makedirs(current_date, exist_ok=True) # Create local date folder if needed
            metrics_df.to_csv(os.path.join(current_date, f"{ticker_symbol}_training_validation_results.csv"))

            # Save fold metrics locally
            for model_name, fold_metrics in all_fold_metrics.items():
                if fold_metrics:
                    fold_df = pd.DataFrame(fold_metrics)
                    fold_df.to_csv(os.path.join(current_date, f"{ticker_symbol}_{model_name}_fold_metrics.csv"), index=False)
                

        # Print ranking to log
        logger.info("\nModel Ranking (by Average MSE):")
        for i, row in metrics_df.iterrows():
            logger.info(f"{row['Rank']}. {row['Model']} - MSE: {row['Average_MSE']:.4f}, RMSE: {row['Average_RMSE']:.4f}")
        
    except Exception as e:
        logger.error(f"Error saving metrics to CSV: {e}")


    # Return the scaler as part of the results
    results_with_scaler = {
        'model_results' : results,
        'target_scaler': target_scaler,
        'feature_scaler': None
    }

    return results_with_scaler


# ===============================================================================
# LSTM Support Functions
# ===============================================================================
# def create_rolling_window_data(X, y, time_steps=5):
#     X_rolled, y_rolled = [], []
#     for i in range(len(X) - time_steps + 1):
#         X_rolled.append(X[i:i + time_steps])
#         y_rolled.append(y[i + time_steps - 1])
#     return np.array(X_rolled), np.array(y_rolled)


def create_rolling_window_data(X, y, time_steps=5):
    """
    Create rolling window data for LSTM training.
    
    Parameters:
    -----------
    X : array-like, Input features
    y : array-like, Target values
    time_steps : int, Number of time steps (window size)
    
    Returns:
    --------
    X_rolled : ndarray, Rolling window input data with shape (samples, time_steps, features)
    y_rolled : ndarray, Target values aligned with the last step of each window
    """
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same number of samples, got X: {len(X)}, y: {len(y)}")
        
    X_rolled, y_rolled = [], []
    for i in range(len(X) - time_steps):  # Note: changed to avoid going past array bounds
        X_rolled.append(X[i:(i + time_steps)])
        y_rolled.append(y[i + time_steps - 1])  # Use the target at the end of the window
    
    # Convert to numpy arrays
    X_rolled = np.array(X_rolled)
    y_rolled = np.array(y_rolled)
    
    # Verify shapes are compatible
    if len(X_rolled) != len(y_rolled):
        raise ValueError(f"Resulting arrays have different lengths: X_rolled: {len(X_rolled)}, y_rolled: {len(y_rolled)}")
        
    return X_rolled, y_rolled



def train_lstm_model(X_train, y_train, X_val, y_val, params, trial=None):
    """
    Train an enhanced LSTM model with various architecture configurations.
    
    Parameters:
    -----------
    X_train : array-like, Input features for training
    y_train : array-like, Target values for training
    X_val : array-like, Input features for validation
    y_val : array-like, Target values for validation
    params : dict, Model hyperparameters
    trial : optuna.Trial, optional, Trial object for hyperparameter optimization
    
    Returns:
    --------
    model : keras.Model, Trained LSTM model
    mse : float, Mean squared error on validation set
    rmse : float, Root mean squared error on validation set
    y_pred : array, Predicted values on validation set
    """

    try:
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        units = params.get('units', 32)
        learning_rate = params.get('learning_rate', 0.005)
        dropout_rate = params.get('dropout', 0.2)
        time_steps = params.get('time_steps', 5)
        num_layers = params.get('layers', 1)
        use_bidirectional = params.get('bidirectional', False)  
        use_batch_norm = params.get('use_batch_norm', False)  

        # Create rolled windows
        X_train_rolled, y_train_rolled = create_rolling_window_data(X_train, y_train, time_steps)
        X_val_rolled, y_val_rolled = create_rolling_window_data(X_val, y_val, time_steps)

        # Create enhanced model architecture with bidirectional layers
        feature_dim = X_train.shape[1]
        inputs = Input(shape=(time_steps, feature_dim))
        x = inputs

        # Simplified LSTM layers implementation
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)  # True for all layers except the last one
            
            if use_bidirectional:
                x = Bidirectional(
                    LSTM(
                        units // 2,  # Decrease units in deeper layers
                        activation='tanh',
                        return_sequences=return_sequences
                    )
                )(x)
            else:
                x = LSTM(
                    units // (i + 1),  # Decrease units in deeper layers
                    activation='tanh',
                    return_sequences=return_sequences
                )(x)
                
            # Add dropout after each LSTM layer
            x = Dropout(dropout_rate)(x)
            
            # Add batch normalization if specified
            if use_batch_norm:
                x = BatchNormalization()(x)
                
        # Add a dense layer before the output
        x = Dense(units // 2, activation='relu')(x)
        x = Dropout(dropout_rate / 2)(x)  # Less dropout before output
        
        # Output layer
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        # Compile with Adam optimizer
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        # Early stopping with longer patience
        callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        if trial:
            callbacks.append(TFKerasPruningCallback(trial, 'val_loss'))

        # Train the model
        history = model.fit(
            X_train_rolled, y_train_rolled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_rolled, y_val_rolled),
            callbacks=callbacks,
            verbose=1
        )

        # Predict on validation set
        y_pred = model.predict(X_val_rolled).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_val_rolled, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_rolled, y_pred)
        mae = mean_absolute_error(y_val_rolled, y_pred)

        print(f"LSTM Fold - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        return model, mse, rmse, y_pred
    
    except Exception as e:
        print(f"Error in LSTM training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, float('inf'), float('inf'), np.zeros_like(y_val[:len(y_val)-time_steps])
       

def train_lstm_model_with_cv(X_train_val, Y_train_val, tscv, feature_scaler, target_scaler):
    """
    Train LSTM model using time series cross-validation with Optuna hyperparameter optimization.
    Handles both regular features and PCA-transformed data.
    
    Parameters:
    - X_train_val: Features for training/validation (DataFrame or ndarray)
    - Y_train_val: Target variable for training/validation (Series or ndarray)
    - tscv: TimeSeriesSplit object for cross-validation
    - feature_scaler: Scaler for features (None for PCA-transformed data)
    - target_scaler: Scaler for target variable (RobustScaler)

    Returns:
    - Dictionary containing training results:
        - best_mse_scores: List of MSE scores for each fold
        - best_rmse_scores: List of RMSE scores for each fold
        - best_r2_scores: List of R² scores for each fold
        - best_params: List of best parameters for each fold
        - best_model: Best trained LSTM model
        - best_model_prediction: Predictions from the best model
        - Y_val_best: Validation targets for the best model
    """

    fold_metrics = []

    def is_pca_transformed_data(X_data):
        if isinstance(X_data, pd.DataFrame):
            return all(col.startswith('PC') for col in X_data.columns)
        return False

    try:
        data_is_pca = is_pca_transformed_data(X_train_val)
        if data_is_pca:
            print("Training LSTM on PCA-transformed data")

        best_mse_scores = []
        best_rmse_scores = []
        best_r2_scores = []  # Added to store R² scores
        best_params = []
        best_model = None
        best_score = float('inf')
        best_model_prediction = None
        Y_val_best = None

        def objective(trial):
            params = {
                'epochs': trial.suggest_int('epochs', 50, 100),
                'batch_size': trial.suggest_int('batch_size', 16, 64),
                'units': trial.suggest_int('units', 8, 32),
                'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.01, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'time_steps': trial.suggest_int('time_steps', 3, 8),
                'layers': trial.suggest_int('layers', 1, 1),
                'bidirectional': trial.suggest_categorical('bidirectional', [False]),
                'use_batch_norm': trial.suggest_categorical('use_batch_norm', [False])
            }
            mse_scores = []

            fold_idx = 0
            for train_idx, val_idx in tscv.split(X_train_val):
                if fold_idx > 0:
                    break
                X_train_fold = X_train_val.iloc[train_idx] if isinstance(X_train_val, pd.DataFrame) else X_train_val[train_idx]
                X_val_fold = X_train_val.iloc[val_idx] if isinstance(X_train_val, pd.DataFrame) else X_train_val[val_idx]
                Y_train_fold = Y_train_val.iloc[train_idx] if isinstance(Y_train_val, pd.Series) else Y_train_val[train_idx]
                Y_val_fold = Y_train_val.iloc[val_idx] if isinstance(Y_train_val, pd.Series) else Y_val_val[val_idx]

                X_train_scaled = X_train_fold.values if isinstance(X_train_fold, pd.DataFrame) else X_train_fold
                X_val_scaled = X_val_fold.values if isinstance(X_val_fold, pd.DataFrame) else X_val_fold
                Y_train_fold_array = Y_train_fold.values if isinstance(Y_train_fold, pd.Series) else Y_train_fold
                Y_val_fold_array = Y_val_fold.values if isinstance(Y_val_fold, pd.Series) else Y_val_fold
                Y_train_scaled = target_scaler.transform(Y_train_fold_array.reshape(-1, 1)).flatten()
                Y_val_scaled = target_scaler.transform(Y_val_fold_array.reshape(-1, 1)).flatten()

                model, mse, _, y_pred_scaled = train_lstm_model(
                    X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, params, trial
                )
                if model is None:
                    return float('inf')

                y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                adjusted_val_fold = Y_val_fold.iloc[:len(y_pred_original)] if isinstance(Y_val_fold, pd.Series) else Y_val_fold[:len(y_pred_original)]
                mse_original = mean_squared_error(adjusted_val_fold, y_pred_original)
                mse_scores.append(mse_original)
                fold_idx += 1

            return np.mean(mse_scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)

        best_params_fold = study.best_params
        best_mse = study.best_value

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
            print(f"\nTraining on fold {fold + 1} for LSTM...")

            X_train_fold = X_train_val.iloc[train_idx] if isinstance(X_train_val, pd.DataFrame) else X_train_val[train_idx]
            X_val_fold = X_train_val.iloc[val_idx] if isinstance(X_train_val, pd.DataFrame) else X_train_val[val_idx]
            Y_train_fold = Y_train_val.iloc[train_idx] if isinstance(Y_train_val, pd.Series) else Y_train_val[train_idx]
            Y_val_fold = Y_train_val.iloc[val_idx] if isinstance(Y_train_val, pd.Series) else Y_train_val[val_idx]

            print(f"Training fold shapes - X: {X_train_fold.shape}, Y: {len(Y_train_fold)}")
            print(f"Validation fold shapes - X: {X_val_fold.shape}, Y: {len(Y_val_fold)}")

            X_train_scaled = X_train_fold.values if isinstance(X_train_fold, pd.DataFrame) else X_train_fold
            X_val_scaled = X_val_fold.values if isinstance(X_val_fold, pd.DataFrame) else X_val_fold
            Y_train_fold_array = Y_train_fold.values if isinstance(Y_train_fold, pd.Series) else Y_train_fold
            Y_val_fold_array = Y_val_fold.values if isinstance(Y_val_fold, pd.Series) else Y_val_fold
            Y_train_scaled = target_scaler.transform(Y_train_fold_array.reshape(-1, 1)).flatten()
            Y_val_scaled = target_scaler.transform(Y_val_fold_array.reshape(-1, 1)).flatten()

            model, mse, rmse, y_pred_scaled = train_lstm_model(
                X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, best_params_fold
            )
            if model is None:
                continue

            y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            adjusted_val_fold = Y_val_fold.iloc[:len(y_pred_original)] if isinstance(Y_val_fold, pd.Series) else Y_val_fold[:len(y_pred_original)]
            mse_original = mean_squared_error(adjusted_val_fold, y_pred_original)
            rmse_original = np.sqrt(mse_original)
            r2_original = r2_score(adjusted_val_fold, y_pred_original)  # Added R² calculation
            mae_original = mean_absolute_error(adjusted_val_fold, y_pred_original)  # Added MAE for consistency
            min_ratio, max_ratio = verify_prediction_scale(None, adjusted_val_fold, y_pred_original, f"LSTM fold {fold+1}")  # Logger not passed, using None

            best_mse_scores.append(mse_original)
            best_rmse_scores.append(rmse_original)
            best_r2_scores.append(r2_original)
            best_params.append(best_params_fold)

            print(f"Best parameters for fold {fold + 1}: {best_params_fold}")
            print(f"End of Fold {fold + 1} - MSE: {mse_original:.4f}, RMSE: {rmse_original:.4f}, R²: {r2_original:.4f}")

            # Store fold metrics
            fold_metrics.append({
                'Fold': fold + 1,
                'MSE': mse_original,
                'RMSE': rmse_original,
                'MAE': mae_original,
                'R2': r2_original,  # Added R²
                'Min_Ratio': min_ratio,
                'Max_Ratio': max_ratio,
                'Parameters': str(best_params_fold)
            })

            if mse_original < best_score:
                best_score = mse_original
                best_model = model
                best_model_prediction = y_pred_original
                Y_val_best = adjusted_val_fold

        if best_params and len(best_params) > 0:
            try:
                print("Retraining LSTM with best parameters on full training set...")
                best_fold_idx = np.argmin(best_mse_scores)
                final_best_params = best_params[best_fold_idx]

                X_train_val_array = X_train_val.values if isinstance(X_train_val, pd.DataFrame) else X_train_val
                Y_train_val_array = Y_train_val.values if isinstance(Y_train_val, pd.Series) else Y_train_val
                X_train_val_scaled = X_train_val_array
                Y_train_val_scaled = target_scaler.transform(Y_train_val_array.reshape(-1, 1)).flatten()

                model, mse, rmse, y_pred_scaled = train_lstm_model(
                    X_train_val_scaled, Y_train_val_scaled, X_train_val_scaled, Y_train_val_scaled, final_best_params
                )
                best_model = model
                print("Successfully retrained LSTM on full dataset")

            except Exception as e:
                print(f"Error retraining LSTM on full dataset: {e}")

        return {
            'best_mse_scores': best_mse_scores,
            'best_rmse_scores': best_rmse_scores,
            'best_r2_scores': best_r2_scores,  # Added R² scores
            'best_params': best_params,
            'best_model': best_model,
            'best_model_prediction': best_model_prediction,
            'Y_val_best': Y_val_best,
            'fold_metrics': fold_metrics
        }

    except Exception as e:
        print(f"Error in LSTM training: {e}")
        return {
            'best_mse_scores': [],
            'best_rmse_scores': [],
            'best_r2_scores': [],  # Added R² scores
            'best_params': [],
            'best_model': None,
            'best_model_prediction': None,
            'Y_val_best': None,
            'fold_metrics': fold_metrics
        }