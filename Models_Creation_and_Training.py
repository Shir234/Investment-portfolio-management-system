# Models_Creation_and_Training.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

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
                'kernel': ['rbf', 'linear'],
                'C': uniform(0.1, 100),  # Range from 0.1 to 100.1
                'epsilon': uniform(0.01, 0.1),  # Range from 0.01 to 0.11
                'gamma': ['scale', 'auto'] + list(uniform(0.01, 0.5).rvs(3))  # Mix of fixed and random values
            }),
            'XGBoost': (XGBRegressor(random_state=42), {
                'n_estimators': randint(100, 300),  # Range from 100 to 300
                'max_depth': randint(3, 10),  # Range from 3 to 9
                'learning_rate': uniform(0.01, 0.1),  # Range from 0.01 to 0.11
                'subsample': uniform(0.7, 0.3),  # Range from 0.7 to 1.0
                'colsample_bytree': uniform(0.7, 0.3)  # Range from 0.7 to 1.0
            }),
            'LightGBM': (LGBMRegressor(random_state=42, verbose=-1), {
                'n_estimators': randint(100, 300),
                'max_depth': randint(3, 8),
                'learning_rate': uniform(0.01, 0.1),
                'subsample': uniform(0.7, 0.3),
                'colsample_bytree': uniform(0.7, 0.3),
                'force_row_wise': [True]
            }),
            'RandomForest': (RandomForestRegressor(random_state=42), {
                'n_estimators': randint(100, 300),
                'max_depth': randint(3, 8),
                'min_samples_split': randint(2, 11),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 0.5]
            }),
            'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
                'n_estimators': randint(100, 300),
                'max_depth': randint(3, 6),
                'learning_rate': uniform(0.01, 0.1),
                'subsample': uniform(0.7, 0.3),
                'min_samples_split': randint(5, 15)
            }),
            'LSTM': (None, {})
            # 'SVR': (SVR(), {
            #     'kernel': ['rbf', 'linear'],
            #     'C': [0.1, 1, 10, 100],
            #     'epsilon': [0.01, 0.05, 0.1],
            #     'gamma': ['scale', 'auto', 0.1]
            # }),
            # 'XGBoost': (XGBRegressor(random_state=42), {
            #         # 'n_estimators': [100, 200],
            #         # 'max_depth': [3, 5, 7],
            #         # 'learning_rate': [0.01, 0.1]
            #         'n_estimators': [100, 200, 300],
            #         'max_depth': [3, 5, 7, 9],
            #         'learning_rate': [0.01, 0.05, 0.1],
            #         'subsample': [0.7, 0.9, 1.0],
            #         'colsample_bytree': [0.7, 0.9, 1.0]
            # }),
            # 'LightGBM': (LGBMRegressor(random_state=42, verbose=-1), {
            #         'n_estimators': [100, 200],
            #         'max_depth': [3, 5, 7],
            #         'learning_rate': [0.01, 0.05, 0.1],
            #         'subsample': [0.7, 0.9, 1.0],
            #         'colsample_bytree': [0.7, 0.9, 1.0],
            #         'force_row_wise': [True]

            # }),
            # 'RandomForest': (RandomForestRegressor(random_state=42), {
            #         'n_estimators': [100, 200],
            #         'max_depth': [3, 5, 7],
            #         'min_samples_split': [2, 5, 10],
            #         'min_samples_leaf' : [1, 2, 4],
            #         'max_features' : ['auto', 'sqrt', 0.5]
            # }),
            # 'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
            #         'n_estimators': [100, 200],
            #         'max_depth': [3, 5, 7],
            #         'learning_rate': [0.01, 0.05, 0.1],
            #         'subsample': [0.7, 0.9, 1.0],
            #         'colsample_bytree': [0.7, 0.9, 1.0]
            # }),
            # 'LSTM': (None, {})
        }
        return models
    except Exception as e:
        print(f"Error creating models: {e}")
        return {}
    

def train_and_validate_models(logger, X_train_val, Y_train_val, current_date, ticker_symbol, date_folder):
    print(f"\n{'-'*30}\nInitializing model training\n{'-'*30}")
    log_data_stats(logger, X_train_val, "X_train_val for models", include_stats=False)
    log_data_stats(logger, Y_train_val, "Y_train_val for models", include_stats=True)

    # Check if data is already PCA-transformed
    data_is_pca = is_pca_transformed_data(X_train_val)
    if data_is_pca:
        logger.info("Detected PCA-transformed input data")

    # Create global scalers for final output
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()

    ##################################
    # Fit the global scalers on all training data for later use with test data
    # For PCA data, we've already scaled before transformation, so we can skip this step
    if data_is_pca:
        # For PCA-transformed data, we might still want to scale, but less aggressively
        # We can use StandardScaler instead of RobustScaler for PCA data
        feature_scaler = StandardScaler()
        X_train_val_scaled = pd.DataFrame(
            feature_scaler.fit_transform(X_train_val),
            columns=X_train_val.columns,
            index=X_train_val.index
        )
    else:
        # Original scaling for normal feature data
        feature_scaler.fit(X_train_val)
        X_train_val_scaled = pd.DataFrame(
            feature_scaler.transform(X_train_val),
            columns=X_train_val.columns,
            index=X_train_val.index
        )
    ##################################

    # Scale the target variable (same for both PCA and non-PCA)
    target_scaler.fit(Y_train_val.values.reshape(-1, 1))
    Y_train_val_scaled = target_scaler.transform(Y_train_val.values.reshape(-1, 1)).flatten()

    # Time series cross-validation with 5 folds, to ensure temporal order (sequence of events in time)
    tscv = TimeSeriesSplit(n_splits=5)                                   
    models = create_models()
    results = {}

    # Iterate each model
    for model_name, (model, params_grid) in models.items():    
        logger.info(f"\n{'-'*30}\nTraining {model_name}\n{'-'*30}")                     

        best_mse_scores = []                                                               # lists to store scores and parameters for the models
        best_rmse_scores = []
        best_params = []
        best_score = float('inf')
        best_model = None
        best_model_prediction = None
        Y_val_best = None

        # Reshape the data for LSTM model
        if model_name == 'LSTM':
            logger.info(f"Handling LSTM model separately with special reshaping")
            results[model_name] = train_lstm_model_with_cv(X_train_val, Y_train_val, tscv, feature_scaler, target_scaler)
            continue

        # Add timeout or max iterations for SVR
        if model_name == 'SVR':
            model.set_params(max_iter=1000)  # Limit iterations

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
            logger.info(f"  Running GridSearchCV for {model_name} on fold {fold + 1}...")
            # Reduce the number of CV splits for SVR to speed up
            cv_splits = 2 if model_name == 'SVR' else 3

            # grid_search = GridSearchCV(
            #     estimator=model,
            #     param_grid=params_grid,
            #     scoring='neg_mean_squared_error',
            #     cv=cv_splits,
            #     n_jobs=-1,
            #     verbose=1
            # )

            n_iter = 30 if model_name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting'] else 15
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params_grid,
                n_iter=n_iter,  # More iterations for complex models
                scoring='neg_mean_squared_error',
                cv=cv_splits,
                n_jobs=-1,
                verbose=1,
                random_state=42  # For reproducibility
            )

            try:
                random_search.fit(X_train_fold, Y_train_fold)
                best_params.append(random_search.best_params_)
                # Train the model for every combination of parameters on each training set of the fold
                # Save the best params
                # grid_search.fit(X_train_fold, Y_train_fold)               
                # best_params.append(grid_search.best_params_)
                logger.info(f"  Best parameters: {random_search.best_params_}")

                # Validate using the best model
                best_model_fold = random_search.best_estimator_
                Y_pred_scaled  = best_model_fold.predict(X_val_fold)                                
            
                # Convert predictions back to original scale
                Y_pred = target_scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()

                # Scale verification
                min_ratio, max_ratio = verify_prediction_scale(logger, Y_val_original, Y_pred, f"{model_name} fold {fold+1}")

                # Calculate performance metrics (on original scale)
                mse = mean_squared_error(Y_val_original, Y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(Y_val_original, Y_pred)
                best_mse_scores.append(mse)
                best_rmse_scores.append(rmse)
                logger.info(f"  Fold {fold + 1} - MSE: {mse:.4f}, RMSE: {rmse:.4f},  MAE: {mae:.4f}")

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
            
        # Save the best model for each model type, the results and the parameters
        results[model_name] = {                                                       
            'best_mse_scores': best_mse_scores,
            'best_rmse_scores': best_rmse_scores,
            'best_params': best_params,
            'best_model': best_model,
            'best_model_prediction': best_model_prediction,
            'Y_val_best' : Y_val_best
        }

        logger.info(f"\nSummary for {model_name}:")
        if best_mse_scores:
            avg_mse = np.mean(best_mse_scores)
            avg_rmse = np.mean(best_rmse_scores)
            logger.info(f"  Average MSE: {avg_mse:.4f}, Average RMSE: {avg_rmse:.4f}")
        else:
            logger.info(f"  No successful folds for {model_name}")
        
    # After all folds, retrain best model on all data using global scalers
    # This ensures the final model can be used with the global scalers
    logger.info("\nRetraining all models on full dataset using global scalers...")
    X_train_val_scaled = pd.DataFrame(
        feature_scaler.transform(X_train_val),
        columns=X_train_val.columns,
        index=X_train_val.index
    )
    Y_train_val_scaled = target_scaler.transform(Y_train_val.values.reshape(-1, 1)).flatten()

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

                logger.info(f"  Retrained {model_name} - MSE: {retrain_mse:.4f}, RMSE: {retrain_rmse:.4f}")
                verify_prediction_scale(logger, Y_train_val, Y_retrain_pred, f"{model_name} retrained")
                
            
                # Replace the best model
                results[model_name]['best_model'] = base_model
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
            
            # Find best fold
            best_fold_idx = np.argmin(result['best_mse_scores'])
            best_fold_mse = result['best_mse_scores'][best_fold_idx]
            best_fold_rmse = result['best_rmse_scores'][best_fold_idx]

            # Get best parameters
            best_fold_params = None
            if result.get('best_params') and len(result['best_params']) > 0:
                best_fold_params = str(result['best_params'][best_fold_idx])
            
            # Add to metrics data
            metrics_data.append({
                'Model': model_name,
                'Average_MSE': avg_mse,
                'Average_RMSE': avg_rmse,
                'Best_Fold_MSE': best_fold_mse,
                'Best_Fold_RMSE': best_fold_rmse,
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
            logger.info(f"Saved clean data for {ticker_symbol} to folders")
        except Exception as e:
            logger.error(f"Error saving to Google Drive: {e}")
            os.makedirs(current_date, exist_ok=True) # Create local date folder if needed
            metrics_df.to_csv(os.path.join(current_date, f"{ticker_symbol}_training_validation_results.csv"))
                
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
        'feature_scaler': feature_scaler
    }

    return results_with_scaler


# ===============================================================================
# LSTM Support Functions
# ===============================================================================
def create_rolling_window_data(X, y, time_steps=5):
    X_rolled, y_rolled = [], []
    for i in range(len(X) - time_steps + 1):
        X_rolled.append(X[i:i + time_steps])
        y_rolled.append(y[i + time_steps - 1])
    return np.array(X_rolled), np.array(y_rolled)


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
        # Get parameters with defaults
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        units = params.get('units', 50)
        learning_rate = params.get('learning_rate', 0.01)
        time_steps = 5

        # # Reshape data for LSTM [samples, time steps, features]
        # features_count = X_train.shape[1]
        # X_train_reshaped = X_train.reshape(-1, 1, features_count)
        # X_val_reshaped = X_val.reshape(-1, 1, features_count)

        # Create rolling window data
        X_train_rolled, y_train_rolled = create_rolling_window_data(X_train, y_train, time_steps)
        X_val_rolled, y_val_rolled = create_rolling_window_data(X_val, y_val, time_steps)

        # Create and compile model
        model = Sequential([
            LSTM(units, activation='relu', input_shape=(time_steps, X_train.shape[1]), return_sequences=False),
            Dropout(0.2),  # Add dropout for regularization
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train model
        history = model.fit(
            X_train_rolled, y_train_rolled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_rolled, y_val_rolled),
            callbacks=[early_stopping],
            verbose=0
        )

        # Predict and evaluate
        y_pred = model.predict(X_val_rolled).flatten()
        mse = mean_squared_error(y_val_rolled, y_pred)
        rmse = np.sqrt(mse)

        return model, mse, rmse, y_pred
    
    except ImportError:
        print("TensorFlow/Keras not available. LSTM model will be skipped.")
        return None, float('inf'), float('inf'), np.zeros_like(y_val)
       

"""
Train LSTM model using time series cross-validation with manual parameter search.
-> Used in train_and_validate_models() function to handle LSTM model separately!!
"""
def train_lstm_model_with_cv(X_train_val, Y_train_val, tscv, feature_scaler, target_scaler):
    """
    Train LSTM model using time series cross-validation with manual parameter search.
    Now handles both regular features and PCA-transformed data.
    
    Parameters:
    - X_train_val: Features for training/validation
    - Y_train_val: Target variable for training/validation
    - tscv: Time series cross-validation splitter
    - feature_scaler: Scaler for features
    - target_scaler: Scaler for target variable

    Returns:
    - Dictionary containing training results
    """

    try:
        # Check if input data is already PCA-transformed
        data_is_pca = is_pca_transformed_data(X_train_val)
        if data_is_pca:
            print("Training LSTM on PCA-transformed data")

        best_mse_scores = []
        best_rmse_scores = []
        best_params = []
        best_model = None
        best_score = float('inf')
        best_model_prediction = None
        Y_val_best = None

        # Define LSTM parameter grid
        # lstm_param_grid = [
        #     {'epochs': 50, 'batch_size': 32, 'units': 50, 'learning_rate': 0.01, 'dropout': 0.0},
        #     {'epochs': 100, 'batch_size': 32, 'units': 50, 'learning_rate': 0.01, 'dropout': 0.2},
        #     {'epochs': 50, 'batch_size': 64, 'units': 100, 'learning_rate': 0.01, 'dropout': 0.0},
        # ]

        # Adjust parameter grid based on data type
        if data_is_pca:
            # For PCA data, we might need different hyperparameters
            # Simpler architecture with fewer units since PCA reduces dimensionality
            lstm_param_grid = [
                {'epochs': 50, 'batch_size': 32, 'units': 32, 'learning_rate': 0.01, 'dropout': 0.1},
                {'epochs': 100, 'batch_size': 32, 'units': 32, 'learning_rate': 0.005, 'dropout': 0.2},
                {'epochs': 75, 'batch_size': 64, 'units': 16, 'learning_rate': 0.01, 'dropout': 0.1},
                {'epochs': 150, 'batch_size': 32, 'units': 24, 'learning_rate': 0.002, 'dropout': 0.2}
            ]
        else:
            # Original parameter grid for regular features
            lstm_param_grid = [
                {'epochs': 50, 'batch_size': 32, 'units': 50, 'learning_rate': 0.01, 'dropout': 0.0},
                {'epochs': 100, 'batch_size': 32, 'units': 50, 'learning_rate': 0.01, 'dropout': 0.2},
                {'epochs': 50, 'batch_size': 64, 'units': 100, 'learning_rate': 0.01, 'dropout': 0.0},
                {'epochs': 75, 'batch_size': 32, 'units': 75, 'learning_rate': 0.005, 'dropout': 0.1}
            ]

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
            print(f"\nTraining on fold {fold + 1} for LSTM...")

            # Split the data
            X_train_fold = X_train_val.iloc[train_idx] if isinstance(X_train_val, pd.DataFrame) else X_train_val[train_idx]
            X_val_fold = X_train_val.iloc[val_idx] if isinstance(X_train_val, pd.DataFrame) else X_train_val[val_idx]
            Y_train_fold = Y_train_val.iloc[train_idx] if isinstance(Y_train_val, pd.Series) else Y_train_val[train_idx]
            Y_val_fold = Y_train_val.iloc[val_idx] if isinstance(Y_train_val, pd.Series) else Y_train_val[val_idx]

            # Scale features and target
            if data_is_pca:
                # For PCA data, use a StandardScaler instead of RobustScaler
                fold_feature_scaler = StandardScaler()
            else:
                fold_feature_scaler = RobustScaler()
            
        
            # Convert to numpy arrays if needed
            X_train_fold_array = X_train_fold.values if isinstance(X_train_fold, pd.DataFrame) else X_train_fold
            X_val_fold_array = X_val_fold.values if isinstance(X_val_fold, pd.DataFrame) else X_val_fold
            
            X_train_scaled = fold_feature_scaler.fit_transform(X_train_fold_array)
            X_val_scaled = fold_feature_scaler.transform(X_val_fold_array)
            
            # Target scaling is the same regardless of PCA
            fold_target_scaler = RobustScaler()
            Y_train_fold_array = Y_train_fold.values if isinstance(Y_train_fold, pd.Series) else Y_train_fold
            Y_val_fold_array = Y_val_fold.values if isinstance(Y_val_fold, pd.Series) else Y_val_fold
            
            Y_train_scaled = fold_target_scaler.fit_transform(Y_train_fold_array.reshape(-1, 1)).flatten()
            Y_val_scaled = fold_target_scaler.transform(Y_val_fold_array.reshape(-1, 1)).flatten()

            # Perform a manual grid search across LSTM parameters
            fold_best_mse = float('inf')
            fold_best_params = None
            fold_best_model = None
            fold_best_preds = None
            fold_best_preds_original = None

            print(f"Performing manual grid search for LSTM on fold {fold + 1}...")

            for params in lstm_param_grid:
                try:
                    # Add dropout parameter to params if needed
                    dropout_rate = params.get('dropout', 0.2)

                    # Train LSTM model with current parameters
                    model, mse, rmse, y_pred_scaled = train_lstm_model(
                        X_train_scaled, Y_train_scaled, 
                        X_val_scaled, Y_val_scaled, 
                        params
                    )

                    # Convert predictions back to original scale for final evaluation
                    y_pred_original = fold_target_scaler.inverse_transform(
                        y_pred_scaled.reshape(-1, 1)
                    ).flatten()

                    # Calculate metrics on original scale
                    mse_original = mean_squared_error(Y_val_fold, y_pred_original)

                    print(f"Parameters: {params}, MSE (scaled): {mse:.4f}, MSE (original): {mse_original:.4f}")

                    if mse < fold_best_mse:
                        fold_best_mse = mse
                        fold_best_params = params
                        fold_best_model = model
                        fold_best_preds = y_pred_scaled
                        fold_best_preds_original = y_pred_original

                except Exception as e:
                    print(f"Error training LSTM with params {params}: {e}")
                    continue

            # If we found a working model
            if fold_best_model is not None:
                # Calculate RMSE on original scale
                fold_best_rmse = np.sqrt(mean_squared_error(Y_val_fold, fold_best_preds_original))

                # Save results
                best_mse_scores.append(mean_squared_error(Y_val_fold, fold_best_preds_original))
                best_rmse_scores.append(fold_best_rmse)
                best_params.append(fold_best_params)

                print(f"Best parameters found for fold {fold + 1}: {fold_best_params}")
                print(f"End of Fold {fold + 1} - MSE: {best_mse_scores[-1]:.4f}, RMSE: {fold_best_rmse:.4f}")


                # Update best model overall if this fold's model is better
                if best_mse_scores[-1] < best_score:
                    best_score = best_mse_scores[-1]
                    best_model = fold_best_model
                    best_model_prediction = fold_best_preds_original
                    Y_val_best = Y_val_fold.values


        # After all folds, retrain the best model on all data using the global scaler
        if best_params and len(best_params) > 0:
            try:
                print("Retraining LSTM with best parameters on full training set...")
            
                # Get best parameters from the best fold
                best_fold_idx = np.argmin(best_mse_scores)
                final_best_params = best_params[best_fold_idx]
                
                # Convert to numpy arrays if needed for scaling
                X_train_val_array = X_train_val.values if isinstance(X_train_val, pd.DataFrame) else X_train_val
                Y_train_val_array = Y_train_val.values if isinstance(Y_train_val, pd.Series) else Y_train_val
                
                # Scale all training data with global scalers
                X_train_val_scaled = feature_scaler.transform(X_train_val_array)
                Y_train_val_scaled = target_scaler.transform(Y_train_val_array.reshape(-1, 1)).flatten()
                
                # Get time steps and dropout rate
                time_steps = 5  # Default value
                dropout_rate = final_best_params.get('dropout', 0.2)
                
                # Create rolling window data for final model training
                X_rolled, Y_rolled = create_rolling_window_data(X_train_val_scaled, Y_train_val_scaled, time_steps)
                
                # Create and compile the final model with dropout                
                features_count = X_train_val_scaled.shape[1]
                final_model = Sequential([
                    LSTM(final_best_params['units'], 
                         activation='relu', 
                         input_shape=(time_steps, features_count), 
                         return_sequences=False),
                    Dropout(dropout_rate),
                    Dense(1)
                ])
                
                final_model.compile(
                    optimizer=Adam(learning_rate=final_best_params['learning_rate']), 
                    loss='mse'
                )
                
                # Add early stopping
                early_stopping = EarlyStopping(
                    monitor='loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                # Train the final model
                final_model.fit(
                    X_rolled, 
                    Y_rolled,
                    epochs=final_best_params['epochs'],
                    batch_size=final_best_params['batch_size'],
                    callbacks=[early_stopping],
                    verbose=1
                )

                # Replace the best model with this retrained version
                best_model = final_model
                print("Successfully retrained LSTM on full dataset")

            except Exception as e:
                print(f"Error retraining LSTM on full dataset: {e}")
                # Keep the fold model if retraining fails
                pass

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