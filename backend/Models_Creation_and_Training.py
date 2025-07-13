# Models_Creation_and_Training.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
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
from Logging_and_Validation import verify_prediction_scale
from Helper_Functions import save_csv_to_drive


def is_pca_transformed_data(X_data):
    """
    Check if the data appears to be PCA-transformed based on column names.
    
    Parameters:
    - X_data : DataFrame or ndarray
    
    Returns:
    - bool : True if data appears to be PCA-transformed, False otherwise
    """

    if isinstance(X_data, pd.DataFrame):
        # Check if column names follow PC pattern
        return all(col.startswith('PC') for col in X_data.columns)
    return False


def create_models():
    """
    Defines the complete model portfolio with hyperparameter search spaces.
    
    Models included:
    - SVR: Kernel-based regression (RBF kernel)
    - XGBoost: Advanced gradient boosting with regularization
    - LightGBM: Microsoft's efficient gradient boosting
    - Random Forest: Ensemble of decision trees
    - Gradient Boosting: Traditional boosting approach
    - LSTM: Deep learning for sequential financial data
    
    Returns: Dictionary with model instances and Optuna-compatible parameter ranges
    """

    try:
        models = {
            'SVR': (SVR(), {
                'kernel': ['rbf'],
                'C': ('float', 5.0, 20.0), 
                'epsilon': ('float', 0.05, 0.3),
                'gamma': ('float', 0.05, 0.5),
                'max_iter': [15000]
            }),
            'XGBoost': (XGBRegressor(random_state=42), {
                'n_estimators': ('int', 100, 500),
                'max_depth': ('int', 3, 7),
                'learning_rate': ('float', 0.01, 0.1), 
                'subsample': ('float', 0.7, 1.0),
                'colsample_bytree': ('float', 0.6, 0.9), 
                'min_child_weight': ('float', 5, 20),  
                'gamma': ('float', 0.3, 0.5),
                'reg_alpha': ('float', 0.3, 1.0),
                'reg_lambda': ('float', 0.1, 1.0)
            }),
            'LightGBM': (LGBMRegressor(random_state=42, verbose=1), {
                'n_estimators': ('int', 10, 50),
                'max_depth': ('int', 2, 10),
                'learning_rate': ('float', 0.01, 0.05),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 0.9),
                'num_leaves': ('int', 5, 15),
                'min_child_samples': ('int', 20, 50),
                'reg_alpha': ('float', 0.0, 0.1),
                'reg_lambda': ('float', 0.0, 0.1),
                'force_row_wise': [True]
            }),
            'RandomForest': (RandomForestRegressor(random_state=42), {
                'n_estimators': ('int', 200, 600),  
                'max_depth': ('int', 3, 10), 
                'min_samples_split': ('int', 5, 15), 
                'min_samples_leaf': ('int', 5, 10), 
                'max_features': ['sqrt', 0.3, 0.5, 1.0],
                'bootstrap': [True],
                'warm_start': [False] 
            }),
            'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
                'n_estimators': ('int', 100, 500),
                'max_depth': ('int', 3, 7), 
                'learning_rate': ('float', 0.01, 0.05), 
                'subsample': ('float', 0.6, 1.0), 
                'min_samples_split': ('int', 2, 10),  
                'min_samples_leaf': ('int', 5, 10), 
                'max_features': ['sqrt', 0.3, 0.5], 
                'alpha': ('float', 0.1, 0.7)
            }),
            'LSTM': (None, {
                'epochs': ('int', 50, 100),
                'batch_size': ('int', 16, 32),
                'units': ('int', 8, 32), 
                'learning_rate': ('float', 0.001, 0.01),
                'dropout': ('float', 0.1, 0.3),
                'time_steps': ('int', 3, 5),  
                'layers': ('int', 1, 1), 
                'bidirectional': [False],
                'use_batch_norm': [False]
            })
        }
        return models
    except Exception as e:
        print(f"Error creating models: {e}")
        return {}
    

def optimize_model_with_optuna(model, params_grid, X_train, Y_train, X_val, Y_val, model_name, logger, target_scaler, n_trials=30):
    """
    Hyperparameter optimization using Optuna's efficient search algorithms.
    
    Process:
    1. Converts parameter definitions to Optuna search space
    2. Trains models with different hyperparameter combinations
    3. Evaluates on validation data (RMSE minimization)
    4. Returns best parameters and performance
    
    Uses parallel processing for faster optimization.
    """

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
            rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
            return rmse
        except Exception as e:
            logger.error(f"Error in trial: {e}")
            return float('inf')

    study = optuna.create_study(direction='minimize')
    with parallel_backend('multiprocessing'):
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_params, study.best_value, study.best_trial


def train_and_validate_models(logger, X_train_val, Y_train_val, current_date, ticker_symbol, date_folder):
    """
    Main training pipeline with time series cross-validation and hyperparameter optimization.
    
    Process:
    1. Data preparation (PCA detection, target scaling with RobustScaler)
    2. Time series CV (3-fold) maintaining temporal order
    3. For each model x fold:
       - Optuna hyperparameter optimization (20-30 trials)
       - Best model training and validation
       - Performance metrics calculation
    4. Model retraining on full dataset
    5. Results compilation and ranking
    
    Handles both traditional ML models and LSTM with proper sequential data processing.
    Returns: Dictionary with trained models, scalers, and comprehensive metrics
    """
    logger.info(f"\n{'-'*30}\nInitializing model training\n{'-'*30}")

    # We know X_train_val is already PCA-transformed data
    logger.info("Using PCA-transformed input data without additional scaling")
    X_train_val_scaled = X_train_val  
    # Create global scaler for final output
    target_scaler = RobustScaler()
    # Scale the target variable
    target_scaler.fit(Y_train_val.values.reshape(-1, 1))
    Y_train_val_scaled = target_scaler.transform(Y_train_val.values.reshape(-1, 1)).flatten()
    # Time series cross-validation with 5 folds, to ensure temporal order (sequence of events in time)
    tscv = TimeSeriesSplit(n_splits=3)                                  
    models = create_models()
    results = {}
    # Dictionary to store fold metrics for each model
    all_fold_metrics = {}

    # Iterate each model
    for model_name, (model, params_grid) in models.items():    
        logger.info(f"\n{'-'*30}\nTraining {model_name}\n{'-'*30}")                     
        # lists to store scores and parameters for the models
        best_mse_scores = []                                                                                                    
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
            lstm_results = train_lstm_model_with_cv(logger,X_train_val_scaled, Y_train_val, tscv, None, target_scaler)
            results[model_name] = lstm_results

            # Extract LSTM fold metrics from the results
            if 'fold_metrics' in lstm_results:
                all_fold_metrics[model_name] = lstm_results['fold_metrics']
            continue
        
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

            if len(X_train_fold) < 300 or len(X_val_fold) < 150:
                logger.warning(f"Skipping fold {fold + 1} for {model_name}: insufficient samples (Train: {len(X_train_fold)}, Val: {len(X_val_fold)})")
                continue

            logger.info(f"  Train fold shapes: X={X_train_fold.shape}, Y={Y_train_fold.shape}")
            logger.info(f"  Validation fold shapes: X={X_val_fold.shape}, Y={Y_val_fold.shape}")
            logger.info(f"  Running Optuna optimization for {model_name} on fold {fold + 1}...")
            n_trials = 30 if model_name != 'SVR' else 20  #

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
                if rmse < best_score:
                    best_score = rmse
                    best_model = best_model_fold
                    best_model_prediction = Y_pred
                    Y_val_best= Y_val_original
                    logger.info(f"  New best model found with RMSE: {rmse:.4f}")

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
                best_fold_idx = np.argmin(result['best_rmse_scores'])
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
                
                # Store retrained model and metrics
                results[model_name]['retrained_model'] = base_model
                results[model_name]['retrained_rmse'] = retrain_rmse
                results[model_name]['retrained_metrics'] = {
                    'MSE': retrain_mse,
                    'RMSE': retrain_rmse,
                    'R2': retrain_r2,
                    'Min_Ratio': min_ratio,
                    'Max_Ratio': max_ratio
                }
                logger.info(f"  Successfully retrained {model_name}, keys: {list(results[model_name].keys())}")
                    
            except Exception as e:
                logger.error(f"  Error retraining {model_name}: {e}")
                logger.error("  Keeping the best fold model instead")
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
        # Save main training results
        save_csv_to_drive(logger, metrics_df, ticker_symbol, 'training_validation_results', date_folder, current_date, index=False)
        # Save detailed fold metrics for each model
        for model_name, fold_metrics in all_fold_metrics.items():
            if fold_metrics:
                fold_df = pd.DataFrame(fold_metrics)
                save_csv_to_drive(logger, fold_df, ticker_symbol, f'{model_name}_fold_metrics', date_folder, current_date, index=False)

        # Print ranking to log
        logger.info("\nModel Ranking (by Average MSE):")
        for i, row in metrics_df.iterrows():
            logger.info(f"{row['Rank']}. {row['Model']} - MSE: {row['Average_MSE']:.4f}, RMSE: {row['Average_RMSE']:.4f}")

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


def create_rolling_window_data(X, y, time_steps=3):
    """
    Converts time series data into overlapping sequences for LSTM training.
    Creates sliding windows where each sample contains 'time_steps' consecutive
    observations, with targets aligned to predict the next time step.
    
    Parameters:
    - X : array-like, Input features (e.g., PCA-transformed with shape (samples, features))
    - y : array-like, Target values
    - time_steps : int, Number of time steps (window size)
    
    Returns:
    - X_rolled : ndarray, Rolling window input data with shape (samples, time_steps, features)
    - y_rolled : ndarray, Target values for the next timestep after each window
    """

    if len(X) != len(y):
        raise ValueError(f"X and y must have same length, got X: {len(X)}, y: {len(y)}")
    
    if len(X) < time_steps + 1:
        raise ValueError(f"Insufficient data for time_steps={time_steps}, need at least {time_steps + 1} samples, got {len(X)}")
    
    X_rolled, y_rolled = [], []
    for i in range(len(X) - time_steps):
        X_rolled.append(X[i:(i + time_steps)])
        y_rolled.append(y[i + time_steps])
    
    X_rolled = np.array(X_rolled)
    y_rolled = np.array(y_rolled)
    
    if len(X_rolled) == 0 or len(y_rolled) == 0:
        raise ValueError("No valid rolling windows created")
    
    return X_rolled, y_rolled


def train_lstm_model(logger, X_train, y_train, X_val, y_val, params, trial=None):
    """
    Train an enhanced LSTM model with various architecture configurations.
    
    Parameters:
    - X_train : array-like, Input features for training
    - y_train : array-like, Target values for training
    - X_val : array-like, Input features for validation
    - y_val : array-like, Target values for validation
    - params : dict, Model hyperparameters
    - trial : optuna.Trial, optional, Trial object for hyperparameter optimization
    
    Returns:
    - model : keras.Model, Trained LSTM model
    - mse : float, Mean squared error on validation set
    - rmse : float, Root mean squared error on validation set
    - y_pred : array, Predicted values on validation set
    """

    try:
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        units = params.get('units', 64)
        learning_rate = params.get('learning_rate', 0.001)
        dropout_rate = params.get('dropout', 0.3)
        time_steps = params.get('time_steps', 3)
        layers = params.get('layers', 1)
        bidirectional = params.get('bidirectional', False)
        use_batch_norm = params.get('use_batch_norm', False)

        if X_train.shape[0] < time_steps + 1:
            raise ValueError(f"Insufficient samples: {X_train.shape[0]} for time_steps={time_steps}")
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            raise ValueError("NaN in input data")

        X_train_rolled, y_train_rolled = create_rolling_window_data(X_train, y_train, time_steps)
        X_val_rolled, y_val_rolled = create_rolling_window_data(X_val, y_val, time_steps)

        inputs = Input(shape=(time_steps, X_train.shape[1]))
        x = inputs
        for i in range(layers):
            if bidirectional:
                x = Bidirectional(LSTM(units, return_sequences=(i < layers-1)))(x)
            else:
                x = LSTM(units, return_sequences=(i < layers-1))(x)
            if use_batch_norm:
                x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        x = Dense(1)(x)
        model = Model(inputs=inputs, outputs=x)

        model.compile(optimizer=Adam(learning_rate), loss='mse')
        callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
        if trial:
            callbacks.append(TFKerasPruningCallback(trial, 'val_loss', interval=5))

        model.fit(
            X_train_rolled, y_train_rolled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_rolled, y_val_rolled),
            callbacks=callbacks,
            verbose=0
        )

        y_pred = model.predict(X_val_rolled, verbose=0).flatten()
        mse = mean_squared_error(y_val_rolled, y_pred)

        return model, mse, np.sqrt(mse), y_pred

    except Exception as e:
        if logger:
            logger.error(f"LSTM error: {e}")
        return None, float('inf'), float('inf'), np.zeros_like(y_val)
       

def train_lstm_model_with_cv(logger, X_train_val, Y_train_val, tscv, feature_scaler, target_scaler):
    """
    LSTM training with time series cross-validation and comprehensive optimization (with Optuna hyperparameter optimization).
    
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
        - fold_metrics: List of fold metrics including model
        - retrained_model: Model retrained on full dataset
        - retrained_rmse: RMSE of retrained model
        - retrained_r2: R² of retrained model


    Two-stage process:
    1. Optuna optimization across multiple folds to find best architecture
    2. Fold-wise training with best parameters for cross-validation metrics
    - Final model retraining on complete dataset
    """

    fold_metrics = []
    data_is_pca = is_pca_transformed_data(X_train_val)
    if data_is_pca and logger:
        logger.info("Training LSTM on PCA-transformed data")

    best_mse_scores, best_rmse_scores, best_r2_scores, best_params = [], [], [], []
    best_model, best_score, best_model_prediction, Y_val_best = None, float('inf'), None, None

    def objective(trial):
        params = {
            'epochs': trial.suggest_int('epochs', 50, 150),
            'batch_size': trial.suggest_int('batch_size', 16, 64),
            'units': trial.suggest_int('units', 32, 128),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
            'dropout': trial.suggest_float('dropout', 0.2, 0.5),
            'time_steps': trial.suggest_int('time_steps', 3, 5),
            'layers': trial.suggest_int('layers', 1, 2),
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False])
        }
        mse_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
            if logger:
                logger.info(f"Optimizing LSTM on fold {fold_idx + 1}")
            X_train_fold = X_train_val.iloc[train_idx] if isinstance(X_train_val, pd.DataFrame) else X_train_val[train_idx]
            X_val_fold = X_train_val.iloc[val_idx] if isinstance(X_train_val, pd.DataFrame) else X_train_val[val_idx]
            Y_train_fold = Y_train_val.iloc[train_idx] if isinstance(Y_train_val, pd.Series) else Y_train_val[train_idx]
            Y_val_fold = Y_train_val.iloc[val_idx] if isinstance(Y_train_val, pd.Series) else Y_train_val[val_idx]

            if len(X_train_fold) < params['time_steps'] + 1 or len(X_val_fold) < params['time_steps'] + 1:
                if logger:
                    logger.warning(f"Skipping fold {fold_idx + 1}: insufficient samples (Train: {len(X_train_fold)}, Val: {len(X_val_fold)})")
                return float('inf')

            X_train_scaled = X_train_fold.values if isinstance(X_train_fold, pd.DataFrame) else X_train_fold
            X_val_scaled = X_val_fold.values if isinstance(X_val_fold, pd.DataFrame) else X_val_fold
            Y_train_scaled = target_scaler.transform(Y_train_fold.values.reshape(-1, 1)).flatten() if isinstance(Y_train_fold, pd.Series) else target_scaler.transform(Y_train_fold.reshape(-1, 1)).flatten()
            Y_val_scaled = target_scaler.transform(Y_val_fold.values.reshape(-1, 1)).flatten() if isinstance(Y_val_fold, pd.Series) else target_scaler.transform(Y_val_fold.reshape(-1, 1)).flatten()

            try:
                model, mse, _, y_pred_scaled = train_lstm_model(logger,
                    X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, params, trial
                )
                if model is None:
                    if logger:
                        logger.error(f"Fold {fold_idx + 1} failed: model is None")
                    return float('inf')

                y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                adjusted_val_fold = Y_val_fold.iloc[:len(y_pred_original)] if isinstance(Y_val_fold, pd.Series) else Y_val_fold[:len(y_pred_original)]
                mse_original = mean_squared_error(adjusted_val_fold, y_pred_original)
                mse_scores.append(mse_original)
                if logger:
                    logger.info(f"Fold {fold_idx + 1} MSE: {mse_original:.4f}")
            except Exception as e:
                if logger:
                    logger.error(f"Fold {fold_idx + 1} error: {str(e)}")
                return float('inf')

        return np.mean(mse_scores) if mse_scores else float('inf')

    try:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=40)  # Increased trials for better optimization

        best_params_fold = study.best_params
        best_mse = study.best_value
        if logger:
            logger.info(f"Best LSTM parameters from Optuna: {best_params_fold}, Best MSE: {best_mse:.4f}")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
            if logger:
                logger.info(f"\nTraining on fold {fold + 1} for LSTM...")
            if len(train_idx) < best_params_fold['time_steps'] + 1 or len(val_idx) < best_params_fold['time_steps'] + 1:
                if logger:
                    logger.warning(f"Skipping fold {fold + 1}: insufficient samples (Train: {len(train_idx)}, Val: {len(val_idx)})")
                continue

            X_train_fold = X_train_val.iloc[train_idx] if isinstance(X_train_val, pd.DataFrame) else X_train_val[train_idx]
            X_val_fold = X_train_val.iloc[val_idx] if isinstance(X_train_val, pd.DataFrame) else X_train_val[val_idx]
            Y_train_fold = Y_train_val.iloc[train_idx] if isinstance(Y_train_val, pd.Series) else Y_train_val[train_idx]
            Y_val_fold = Y_train_val.iloc[val_idx] if isinstance(Y_train_val, pd.Series) else Y_train_val[val_idx]

            if logger:
                logger.info(f"Training fold shapes - X: {X_train_fold.shape}, Y: {len(Y_train_fold)}")
                logger.info(f"Validation fold shapes - X: {X_val_fold.shape}, Y: {len(Y_val_fold)}")

            X_train_scaled = X_train_fold.values if isinstance(X_train_fold, pd.DataFrame) else X_train_fold
            X_val_scaled = X_val_fold.values if isinstance(X_val_fold, pd.DataFrame) else X_val_fold
            Y_train_scaled = target_scaler.transform(Y_train_fold.values.reshape(-1, 1)).flatten() if isinstance(Y_train_fold, pd.Series) else target_scaler.transform(Y_train_fold.reshape(-1, 1)).flatten()
            Y_val_scaled = target_scaler.transform(Y_val_fold.values.reshape(-1, 1)).flatten() if isinstance(Y_val_fold, pd.Series) else target_scaler.transform(Y_val_fold.reshape(-1, 1)).flatten()

            model, mse, rmse, y_pred_scaled = train_lstm_model(logger,
                X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, best_params_fold
            )
            if model is None:
                if logger:
                    logger.error(f"Fold {fold + 1} failed: model is None")
                continue

            y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            adjusted_val_fold = Y_val_fold.iloc[:len(y_pred_original)] if isinstance(Y_val_fold, pd.Series) else Y_val_fold[:len(y_pred_original)]
            mse_original = mean_squared_error(adjusted_val_fold, y_pred_original)
            rmse_original = np.sqrt(mse_original)
            r2_original = r2_score(adjusted_val_fold, y_pred_original)
            mae_original = mean_absolute_error(adjusted_val_fold, y_pred_original)
            min_ratio, max_ratio = verify_prediction_scale(logger, adjusted_val_fold, y_pred_original, f"LSTM fold {fold+1}")

            best_mse_scores.append(mse_original)
            best_rmse_scores.append(rmse_original)
            best_r2_scores.append(r2_original)
            best_params.append(best_params_fold)

            if logger:
                logger.info(f"Best parameters for fold {fold + 1}: {best_params_fold}")
                logger.info(f"End of Fold {fold + 1} - MSE: {mse_original:.4f}, RMSE: {rmse_original:.4f}, R²: {r2_original:.4f}")

            fold_metrics.append({
                'Fold': fold + 1,
                'MSE': mse_original,
                'RMSE': rmse_original,
                'MAE': mae_original,
                'R2': r2_original,
                'Min_Ratio': min_ratio,
                'Max_Ratio': max_ratio,
                'Parameters': best_params_fold,  
                'model': model,
                'y_val': adjusted_val_fold,
                'predictions': y_pred_original
            })

            if rmse_original < best_score:
                best_score = rmse_original
                best_model = model
                best_model_prediction = y_pred_original
                Y_val_best = adjusted_val_fold

        # Retrain on full dataset
        if best_params and best_model:
            try:
                if logger:
                    logger.info("Retraining LSTM with best parameters on full training set...")
                best_fold_idx = np.argmin(best_mse_scores)
                final_best_params = best_params[best_fold_idx]

                X_train_val_array = X_train_val.values if isinstance(X_train_val, pd.DataFrame) else X_train_val
                Y_train_val_array = Y_train_val.values if isinstance(Y_train_val, pd.Series) else Y_train_val
                X_train_val_scaled = X_train_val_array
                Y_train_val_scaled = target_scaler.transform(Y_train_val_array.reshape(-1, 1)).flatten()

                X_full_rolled, y_full_rolled = create_rolling_window_data(X_train_val_scaled, Y_train_val_scaled, final_best_params['time_steps'])
                
                inputs = Input(shape=(final_best_params['time_steps'], X_train_val_scaled.shape[1]))
                x = inputs
                for i in range(final_best_params['layers']):
                    if final_best_params['bidirectional']:
                        x = Bidirectional(LSTM(final_best_params['units'], return_sequences=(i < final_best_params['layers']-1)))(x)
                    else:
                        x = LSTM(final_best_params['units'], return_sequences=(i < final_best_params['layers']-1))(x)
                    if final_best_params['use_batch_norm']:
                        x = BatchNormalization()(x)
                    x = Dropout(final_best_params['dropout'])(x)
                x = Dense(1)(x)
                retrained_model = Model(inputs=inputs, outputs=x)

                retrained_model.compile(optimizer=Adam(learning_rate=final_best_params['learning_rate']), loss='mse')
                retrained_model.fit(
                    X_full_rolled, y_full_rolled,
                    epochs=final_best_params['epochs'],
                    batch_size=final_best_params['batch_size'],
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                    verbose=0
                )

                y_pred_full = retrained_model.predict(X_full_rolled, verbose=0).flatten()
                y_pred_full_original = target_scaler.inverse_transform(y_pred_full.reshape(-1, 1)).flatten()
                y_full_original = target_scaler.inverse_transform(y_full_rolled.reshape(-1, 1)).flatten()
                retrain_mse = mean_squared_error(y_full_original, y_pred_full_original)
                retrain_rmse = np.sqrt(retrain_mse)
                retrain_r2 = r2_score(y_full_original, y_pred_full_original)

                if logger:
                    logger.info(f"Retrained LSTM - MSE: {retrain_mse:.4f}, RMSE: {retrain_rmse:.4f}, R²: {retrain_r2:.4f}")
            except Exception as e:
                if logger:
                    logger.error(f"Error retraining LSTM on full dataset: {e}")
                retrained_model = None
                retrain_rmse = best_score
                retrain_r2 = max(best_r2_scores) if best_r2_scores else 0.0

        result = {
            'best_mse_scores': best_mse_scores,
            'best_rmse_scores': best_rmse_scores,
            'best_r2_scores': best_r2_scores,
            'best_params': best_params,
            'best_model': best_model,
            'best_model_prediction': best_model_prediction,
            'Y_val_best': Y_val_best,
            'fold_metrics': fold_metrics,
            'retrained_model': retrained_model,
            'retrained_rmse': retrain_rmse,
            'retrained_r2': retrain_r2
        }
        
        if logger:
            logger.info(f"LSTM result keys: {list(result.keys())}")
            logger.info(f"LSTM fold_metrics structure: {[list(f.keys()) for f in fold_metrics]}")
            logger.info(f"LSTM retrained_model exists: {retrained_model is not None}")

        return result

    except Exception as e:
        if logger:
            logger.error(f"Error in LSTM training: {e}")
        return {
            'best_mse_scores': [],
            'best_rmse_scores': [],
            'best_r2_scores': [],
            'best_params': [],
            'best_model': None,
            'best_model_prediction': None,
            'Y_val_best': None,
            'fold_metrics': fold_metrics,
            'retrained_model': None,
            'retrained_rmse': float('inf'),
            'retrained_r2': -float('inf')
        }