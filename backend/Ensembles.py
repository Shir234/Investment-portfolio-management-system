# Ensembles.py
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import Models_Creation_and_Training as Models_Creation_and_Training
from Logging_and_Validation import log_data_stats, verify_prediction_scale

# ===============================================================================
# Ensembles
# ===============================================================================
def prepare_lstm_data(X, time_steps=5, features=None):
    """
    Prepare data for LSTM by creating sequences of time_steps.
    For PCA-transformed data, this reshapes it into the proper 3D format.

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
    
    # Check if we have enough samples for the time steps
    if len(X) < time_steps:
        raise ValueError(f"Not enough samples ({len(X)}) for the requested time steps ({time_steps})")
    
    # Create rolling window data
    X_rolled = []
    for i in range(len(X) - time_steps + 1):
        X_rolled.append(X[i:i + time_steps])
    
    return np.array(X_rolled)

# Linearly Weighted Ensemble
def linearly_weighted_ensemble(models_results, X_test, target_scaler, feature_scaler, logger=None):
    """
    Create a linearly weighted ensemble prediction across different model types.
    Parameters:
    - models_results: Dictionary containing model results
    - X_test: Test data to make predictions on (DataFrame or NumPy array)

    Returns:
    - Final ensemble prediction
    """

    # Convert DataFrame to NumPy array if necessary - NO ADDITIONAL SCALING
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    
    # Store model weights and predictions, along with their lengths
    weights = []
    model_predictions = []
    pred_lengths = []

    # Calculate Mean Absolute Error (MAE) for each model
    # First pass: calculate weights based on MAE
    for model_name, result in models_results.items():
        if 'best_model' not in result or result['best_model'] is None:
            continue

        # This handles the shape mismatch by computing MAE safely
        best_pred = result.get('best_model_prediction')
        y_val = result.get('Y_val_best')

        if best_pred is None or y_val is None:
            if logger:
                logger.warning(f"Model {model_name} missing predictions or validation data")
            continue

        # Make sure arrays have the same length for MAE calculation
        min_length = min(len(best_pred), len(y_val))
        if min_length == 0:
            if logger:
                logger.warning(f"Empty predictions for {model_name}")
            continue

        # Compute MAE on the overlap
        # mae_value = np.mean(np.abs(best_pred[:min_length] - y_val[:min_length]))
        # if mae_value <= 0:
        #     if logger:
        #         logger.warning(f"Invalid MAE ({mae_value}) for {model_name}")
        #     continue

        mse_value = mean_squared_error(y_val[:min_length], best_pred[:min_length])
        if mse_value <= 0:
            if logger:
                logger.warning(f"Invalid MSE ({mse_value}) for {model_name}")
            continue

        # Prepare predictions based on model type
        try:
            if model_name.strip() == 'LSTM':
                # Reshape for LSTM
                X_test_lstm = prepare_lstm_data(X_test_array, time_steps=3)
                if len(X_test_lstm) == 0:
                    if logger:
                        logger.warning(f"Not enough data for LSTM predictions")
                    continue
                model_pred = result['best_model'].predict(X_test_lstm).flatten()
            else:
                # For other models
                model_pred = result['best_model'].predict(X_test_array).flatten()
                
            # Store prediction and weight
            model_predictions.append(model_pred)
            #weights.append(1.0 / mae_value)  # Inverse MAE as weight
            weights.append(1.0 / mse_value)  # Inverse MSE as weight
            pred_lengths.append(len(model_pred))
            
            if logger:
                logger.info(f"Model {model_name} - MAE: {mae_value:.4f}, Prediction shape: {model_pred.shape}")
                
        except Exception as e:
            if logger:
                logger.error(f"Error generating predictions for {model_name}: {e}")
            continue

        # Check if we have any valid predictions
    if not model_predictions:
        raise ValueError("No valid model predictions available")

    # Normalize weights to sum to 1
    weights = np.array(weights)
    ###weights = weights / np.sum(weights)
    # Apply minimum weight threshold (e.g., 0.1 * max weight)
    min_weight = 0.1 * np.max(weights)
    weights = np.where(weights < min_weight, 0, weights)
    if np.sum(weights) == 0:
        weights = np.ones_like(weights) / len(weights)  # Fallback to equal weights
    else:
        weights = weights / np.sum(weights)
    
    if logger:
        logger.info(f"Model weights: {weights}")
        logger.info(f"Prediction lengths: {pred_lengths}")
    
    # Find the shortest prediction length to standardize sizes
    min_pred_length = min(pred_lengths)
    # Compute the final ensemble prediction (using the shortest length to avoid shape issues)
    final_prediction = np.zeros(min_pred_length, dtype=np.float64)
    
    # Apply weighted predictions
    for i, (pred, weight) in enumerate(zip(model_predictions, weights)):
        final_prediction += weight * pred[:min_pred_length]
    
    # Inverse transform to get back to original scale
    final_prediction_original = target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()

    return final_prediction_original


# Equal Weights Ensemble
def equal_weighted_ensemble(models_results, X_test, target_scaler, feature_scaler, logger=None):
    """
    Calculate an equal weighted ensemble prediction.
    Parameters:
    - models_results: Dictionary containing model results
    - X_test: Test data to make predictions on (DataFrame or NumPy array)

    Returns:
    - Final ensemble prediction
    """
    
    # Convert DataFrame to NumPy array if necessary - NO ADDITIONAL SCALING
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test

    model_predictions = []
    pred_lengths = []
    
    # Prepare predictions based on model type
    # First pass: generate predictions
    for model_name, result in models_results.items():
        if 'best_model' not in result or result['best_model'] is None:
            continue

        try:
            if model_name.strip() == 'LSTM':
                # Reshape for LSTM, using data directly
                X_test_lstm = prepare_lstm_data(X_test_array, time_steps=3)
                if len(X_test_lstm) == 0:
                    if logger:
                        logger.warning(f"Not enough data for LSTM predictions")
                    continue
                model_pred = result['best_model'].predict(X_test_lstm).flatten()
            else:
                # For other models, use data directly
                model_pred = result['best_model'].predict(X_test_array).flatten()

            # Store prediction
            model_predictions.append(model_pred)
            pred_lengths.append(len(model_pred))
            
            if logger:
                logger.info(f"Model {model_name} - Prediction shape: {model_pred.shape}")
                
        except Exception as e:
            if logger:
                logger.error(f"Error generating predictions for {model_name}: {e}")
            continue

    if not model_predictions:
        raise ValueError("No predictions available for ensemble methods")

    # Find the shortest prediction length
    min_pred_length = min(pred_lengths) if pred_lengths else 0
    if min_pred_length == 0:
        raise ValueError("No valid predictions with length > 0")
        
    if logger:
        logger.info(f"Using minimum prediction length: {min_pred_length}")
        
    # Calculate weight (equal for all models)
    weight = 1.0 / len(model_predictions)

    # Compute the final ensemble prediction
    final_prediction = np.zeros(min_pred_length, dtype=np.float64)

    # Apply weighted predictions
    for pred in model_predictions:
        final_prediction += weight * pred[:min_pred_length]
    
    # Inverse transform to get back to original scale
    final_prediction_original = target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()

    return final_prediction_original

# Gradient Boosting Decision Tree Ensemble
# def gbdt_ensemble(models_results, X_train, X_test, Y_train, target_scaler, feature_scaler, logger=None):
#     """
#     Use GBDT to predict based on the predictions of base models.
#     Implements a validation split approach to prevent data leakage.
#     Parameters:
#     - models_results: Dictionary containing model results, with 'best_model' key for each model.
#     - X_train: Training data features to generate meta-features for training GBDT.
#     - X_test: Test data features for final prediction.
#     - Y_train: Training labels for fitting the GBDT model.
    
#     Returns: 
#     - GBDT ensemble prediction for test data.
#     """

#     # Convert to arrays without scaling - data is already PCA-transformed
#     X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
#     X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
#     Y_train_array = Y_train.values if hasattr(Y_train, 'values') else Y_train

#     # Split training data into training and validation sets to prevent leakage
#     X_meta_train, X_meta_val, Y_meta_train, Y_meta_val = train_test_split(
#         X_train_array, Y_train_array, test_size=0.2, random_state=42, shuffle=False
#     )

#     # Track metadata about models and predictions
#     train_meta_features = {}
#     val_meta_features = {}
#     test_meta_features = {}
#     prediction_lengths = {'train': [], 'val': [], 'test': []}

#     for model_name, result in models_results.items():
#         if 'best_model' not in result or result['best_model'] is None:
#             continue

#         model = result['best_model']

#         try:
#             # Generate predictions for all three sets
#             if model_name.strip() == 'LSTM':
#                 # Reshape for LSTM
#                 X_meta_train_lstm = prepare_lstm_data(X_meta_train, time_steps=5)
#                 X_meta_val_lstm = prepare_lstm_data(X_meta_val, time_steps=5)
#                 X_test_lstm = prepare_lstm_data(X_test_array, time_steps=5)
                
#                 train_pred = model.predict(X_meta_train_lstm).flatten()
#                 val_pred = model.predict(X_meta_val_lstm).flatten()
#                 test_pred = model.predict(X_test_lstm).flatten()
#             else:
#                 train_pred = model.predict(X_meta_train).flatten()
#                 val_pred = model.predict(X_meta_val).flatten()
#                 test_pred = model.predict(X_test_array).flatten()
            
#             # Store predictions for each model separately to handle length differences
#             train_meta_features[model_name] = train_pred
#             val_meta_features[model_name] = val_pred
#             test_meta_features[model_name] = test_pred
            
#             # Record prediction lengths
#             prediction_lengths['train'].append(len(train_pred))
#             prediction_lengths['val'].append(len(val_pred))
#             prediction_lengths['test'].append(len(test_pred))
            
#             if logger:
#                 logger.info(f"Model {model_name} prediction lengths - Train: {len(train_pred)}, "
#                             f"Val: {len(val_pred)}, Test: {len(test_pred)}")
#                 logger.info(f"Model {model_name} prediction stats - Train mean: {np.mean(train_pred):.4f}, "
#                             f"std: {np.std(train_pred):.4f}, Val mean: {np.mean(val_pred):.4f}, "
#                             f"std: {np.std(val_pred):.4f}")
        
#         except Exception as e:
#             if logger:
#                 logger.error(f"Error generating predictions for {model_name}: {e}")
#             continue

#     # If no predictions were successful, exit early
#     if not train_meta_features:
#         raise ValueError("No valid model predictions available for GBDT ensemble")

#     # Calculate minimum lengths for each set to handle size mismatches
#     min_train_length = min(prediction_lengths['train'])
#     min_val_length = min(prediction_lengths['val'])
#     min_test_length = min(prediction_lengths['test'])
    
#     if logger:
#         logger.info(f"Using minimum lengths - Train: {min_train_length}, "
#                     f"Val: {min_val_length}, Test: {min_test_length}")

#     # Prepare arrays with consistent lengths
#     train_features_aligned = np.column_stack([pred[:min_train_length] for pred in train_meta_features.values()])
#     val_features_aligned = np.column_stack([pred[:min_val_length] for pred in val_meta_features.values()])
#     test_features_aligned = np.column_stack([pred[:min_test_length] for pred in test_meta_features.values()])
    
#     # Align target arrays with the meta-features
#     Y_meta_train_aligned = Y_meta_train[:min_train_length]
#     Y_meta_val_aligned = Y_meta_val[:min_val_length]
    
#     if len(Y_meta_train_aligned) == 0 or len(Y_meta_val_aligned) == 0:
#         raise ValueError("Insufficient data for GBDT training after alignment")

#     # Scale meta-features
#     train_features_scaled = train_features_aligned
#     val_features_scaled = val_features_aligned
#     test_features_scaled = test_features_aligned

#     # Train GBDT on scaled meta-features
#     gb_model = GradientBoostingRegressor(
#         n_estimators=50,  # Reduced to prevent overfitting
#         learning_rate=0.01,
#         max_depth=2,  # Shallower trees
#         random_state=42
#     )

#     # Fit on training meta-features
#     gb_model.fit(train_features_scaled, Y_meta_train_aligned)
    
#     # Evaluate on validation set to check for overfitting
#     val_pred = gb_model.predict(val_features_scaled)
#     val_mse = mean_squared_error(Y_meta_val_aligned, val_pred)
#     val_r2 = r2_score(Y_meta_val_aligned, val_pred)
#     if logger:
#         logger.info(f"GBDT Ensemble - Validation MSE: {val_mse:.6f}, RMSE: {np.sqrt(val_mse):.6f}, R²: {val_r2:.6f}")

#     # Predict on test meta-features
#     final_prediction = gb_model.predict(test_features_scaled)

#     # Inverse transform to get back to original scale
#     final_prediction_original = target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()
    
#     return final_prediction_original

# Gradient Boosting Decision Tree Ensemble
def gbdt_ensemble(models_results, X_train, X_test, Y_train, target_scaler, feature_scaler, logger=None):
    """
    Use GBDT to predict based on base model predictions as meta-features.

    Parameters:
    - models_results: Dictionary containing model results.
    - X_train, X_test: Training and test data (DataFrame or array).
    - Y_train: Training labels.
    - target_scaler: Scaler for inverse-transforming predictions.
    - feature_scaler: Not used (kept for compatibility).
    - logger: Logger for debugging.

    Returns:
    - GBDT ensemble prediction in original scale.
    """
    X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    Y_train_array = Y_train.values if hasattr(Y_train, 'values') else Y_train

    # Split training data into training and validation sets
    X_meta_train, X_meta_val, Y_meta_train, Y_meta_val = train_test_split(
        X_train_array, Y_train_array, test_size=0.2, random_state=42, shuffle=False
    )

    train_meta_features = {}
    val_meta_features = {}
    test_meta_features = {}
    prediction_lengths = {'train': [], 'val': [], 'test': []}

    for model_name, result in models_results.items():
        if 'best_model' not in result or result['best_model'] is None:
            continue

        model = result['best_model']

        try:
            if model_name.strip() == 'LSTM':
                X_meta_train_lstm = prepare_lstm_data(X_meta_train, time_steps=3)
                X_meta_val_lstm = prepare_lstm_data(X_meta_val, time_steps=3)
                X_test_lstm = prepare_lstm_data(X_test_array, time_steps=3)

                train_pred = model.predict(X_meta_train_lstm).flatten()
                val_pred = model.predict(X_meta_val_lstm).flatten()
                test_pred = model.predict(X_test_lstm).flatten()
            else:
                train_pred = model.predict(X_meta_train).flatten()
                val_pred = model.predict(X_meta_val).flatten()
                test_pred = model.predict(X_test_array).flatten()

            train_meta_features[model_name] = train_pred
            val_meta_features[model_name] = val_pred
            test_meta_features[model_name] = test_pred

            prediction_lengths['train'].append(len(train_pred))
            prediction_lengths['val'].append(len(val_pred))
            prediction_lengths['test'].append(len(test_pred))

            if logger:
                logger.info(f"Model {model_name} prediction lengths - Train: {len(train_pred)}, "
                            f"Val: {len(val_pred)}, Test: {len(test_pred)}")
                logger.info(f"Model {model_name} prediction stats - Train mean: {np.mean(train_pred):.4f}, "
                            f"std: {np.std(train_pred):.4f}, Val mean: {np.mean(val_pred):.4f}, "
                            f"std: {np.std(val_pred):.4f}")

        except Exception as e:
            if logger:
                logger.error(f"Error generating predictions for {model_name}: {e}")
            continue

    if not train_meta_features:
        raise ValueError("No valid model predictions for GBDT ensemble")

    # Calculate minimum lengths
    min_train_length = min(prediction_lengths['train'])
    min_val_length = min(prediction_lengths['val'])
    min_test_length = min(prediction_lengths['test'])

    if logger:
        logger.info(f"Using minimum lengths - Train: {min_train_length}, Val: {min_val_length}, Test: {min_test_length}")
        logger.info(f"Samples lost - Train: {len(X_meta_train) - min_train_length}, "
                    f"Val: {len(X_meta_val) - min_val_length}")

    # Align meta-features
    train_features_aligned = np.column_stack([pred[:min_train_length] for pred in train_meta_features.values()])
    val_features_aligned = np.column_stack([pred[:min_val_length] for pred in val_meta_features.values()])
    test_features_aligned = np.column_stack([pred[:min_test_length] for pred in test_meta_features.values()])

    Y_meta_train_aligned = Y_meta_train[:min_train_length]
    Y_meta_val_aligned = Y_meta_val[:min_val_length]

    if len(Y_meta_train_aligned) == 0 or len(Y_meta_val_aligned) == 0:
        raise ValueError("Insufficient data for GBDT training after alignment")

    # Scale meta-features
    meta_scaler = StandardScaler()
    train_features_scaled = meta_scaler.fit_transform(train_features_aligned)
    val_features_scaled = meta_scaler.transform(val_features_aligned)
    test_features_scaled = meta_scaler.transform(test_features_aligned)

    # Check meta-feature correlation
    if logger:
        corr_matrix = np.corrcoef(train_features_scaled, rowvar=False)
        logger.info(f"Meta-feature correlation matrix:\n{corr_matrix}")

    # Train GBDT with early stopping
    gb_model = GradientBoostingRegressor(
        n_estimators=30,
        learning_rate=0.01,
        max_depth=1,
        random_state=42,
        validation_fraction=0.2,
        n_iter_no_change=10,
        tol=1e-4
    )

    gb_model.fit(train_features_scaled, Y_meta_train_aligned)

    # Evaluate on validation set
    val_pred = gb_model.predict(val_features_scaled)
    val_mse = mean_squared_error(Y_meta_val_aligned, val_pred)
    val_r2 = r2_score(Y_meta_val_aligned, val_pred)

    if logger:
        logger.info(f"GBDT Ensemble - Validation MSE: {val_mse:.6f}, RMSE: {np.sqrt(val_mse):.6f}, R²: {val_r2:.6f}")

    # Predict on test set
    final_prediction = gb_model.predict(test_features_scaled)
    final_prediction_original = target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()

    # Check prediction variability
    if logger:
        pred_std = np.std(final_prediction_original)
        logger.info(f"GBDT prediction std: {pred_std:.4f}")
        if pred_std < 0.01:
            logger.warning("Low prediction variability, GBDT may be underfitting")

    return final_prediction_original

# ===============================================================================
# Three Ensembles Pipeline
# ===============================================================================
def ensemble_pipeline(logger, models_results, X_train, X_test, Y_train, Y_test, target_scaler, feature_scaler):
    """
    Pipeline to apply and compare different ensemble methods.
    Parameters:
    - models_results: Dictionary containing model results
    - X_train: Training features
    - X_test: Testing features
    - Y_train: Training labels
    - Y_test: Testing labels

    Returns:
    - Dictionary with ensemble results
    """

    logger.info(f"\n{'-'*30}\nInitializing ensemble pipeline\n{'-'*30}")
    # log_data_stats(logger, X_train, "X_train for ensembles", include_stats=False)
    # log_data_stats(logger, X_test, "X_test for ensembles", include_stats=False) 
    # log_data_stats(logger, Y_train, "Y_train for ensembles", include_stats=True)
    # log_data_stats(logger, Y_test, "Y_test for ensembles", include_stats=True)

    # Verify input model results
    logger.info("\nVerifying model results for ensemble:")
    for model_name, result in models_results.items():
        if 'best_model' in result and result['best_model'] is not None:
            logger.info(f"  {model_name}: Model available, Best MSE: {np.min(result['best_mse_scores']):.6f}")
        else:
            logger.info(f"  {model_name}: Model NOT available")

    # Test target scaler to confirm it works properly
    logger.info("\nValidating target scaler:")
    y_sample = Y_test.iloc[:3].values.reshape(-1, 1) if hasattr(Y_test, 'iloc') else Y_test[:3].reshape(-1, 1)
    y_scaled = target_scaler.transform(y_sample)
    y_restored = target_scaler.inverse_transform(y_scaled)
    logger.info(f"  Original Y values: {y_sample.flatten()}")
    logger.info(f"  Scaled Y values: {y_scaled.flatten()}")
    logger.info(f"  Restored Y values: {y_restored.flatten()}")

    def linearly_weighted_wrapper(results, x_test):
        logger.info("Running linearly weighted ensemble...")
        return linearly_weighted_ensemble(results, x_test, target_scaler, None, logger)  # Pass logger

    def equal_weighted_wrapper(results, x_test):
        logger.info("Running equal weighted ensemble...")
        return equal_weighted_ensemble(results, x_test, target_scaler, None, logger)  # Pass logger

    def gbdt_wrapper(results, x_test):
        logger.info("Running GBDT ensemble...")
        return gbdt_ensemble(results, X_train, x_test, Y_train, target_scaler, None, logger)  # Pass logger
        
    
    # Define the ensemble methods with consistent wrapper functions
    ensemble_methods = {
        'linearly_weighted': linearly_weighted_wrapper,
        'equal_weighted': equal_weighted_wrapper,
        'gbdt': gbdt_wrapper
    }

    results = {}

    for method_name, method_func in ensemble_methods.items():
        try:
            logger.info(f"\n{'-'*20} Running {method_name} ensemble {'-'*20}")
            # Time the prediction
            start_time = time.time()
            # Get predictions from the ensemble method
            final_prediction = method_func(models_results, X_test)
            end_time = time.time()

            # Detailed scale checking
            # Verify predictions are in original scale by checking range against Y_test
            y_min, y_max = np.min(Y_test), np.max(Y_test)
            pred_min, pred_max = np.min(final_prediction), np.max(final_prediction)
            y_mean, y_std = np.mean(Y_test), np.std(Y_test)
            pred_mean, pred_std = np.mean(final_prediction), np.std(final_prediction)

            logger.info(f"\nScale Comparison for {method_name}:")
            logger.info(f"  Y_test range: [{y_min:.4f}, {y_max:.4f}], mean: {y_mean:.4f}, std: {y_std:.4f}")
            logger.info(f"  Prediction range: [{pred_min:.4f}, {pred_max:.4f}], mean: {pred_mean:.4f}, std: {pred_std:.4f}")
            logger.info(f"  Time taken: {end_time - start_time:.2f} seconds")
            
            # If the prediction length is shorter than Y_test, we need to truncate Y_test for metrics
            Y_test_aligned = Y_test
            if isinstance(Y_test, pd.Series) or isinstance(Y_test, pd.DataFrame):
                Y_test_aligned = Y_test.iloc[:len(final_prediction)]
            else:
                Y_test_aligned = Y_test[:len(final_prediction)]
                
            logger.info(f"  Length of Y_test: {len(Y_test)}, Length of predictions: {len(final_prediction)}")
            logger.info(f"  Using aligned Y_test with length: {len(Y_test_aligned)}")

            # Calculate performance metrics using aligned data
            mse = mean_squared_error(Y_test_aligned, final_prediction)
            rmse = np.sqrt(mse)
            r2 = r2_score(Y_test_aligned, final_prediction)
            mae = np.mean(np.abs(Y_test_aligned - final_prediction))

            # Store results
            results[method_name] = {
                'prediction': final_prediction,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae
            }

            # Show detailed stats for this method
            logger.info(f"\nDetailed results for {method_name}:")
            logger.info(f"  MSE: {mse:.6f}")
            logger.info(f"  RMSE: {rmse:.6f}")
            logger.info(f"  R²: {r2:.6f}")
            logger.info(f"  MAE: {mae:.6f}")

        except Exception as e:
            logger.error(f"Error in {method_name} ensemble: {e}")
            import traceback
            traceback.print_exc()

            # Provide empty results if method fails
            results[method_name] = {
                'prediction': np.zeros_like(Y_test),
                'mse': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'mae': float('inf')
            }
    # Output results
    valid_results = {k: v for k, v in results.items() if v['mse'] < float('inf')}

    if valid_results:
        logger.info("\nEnsemble Methods Comparison:")
        comparison_df = pd.DataFrame([
            {
                'Method': method_name,
                'RMSE': metrics['rmse'],
                'MSE': metrics['mse'],
                'R²': metrics['r2'],
                'MAE': metrics.get('mae', np.nan)
            }
            for method_name, metrics in valid_results.items()
        ])
        logger.info(f"\n{comparison_df}")
            
        # Find the best performing method based on RMSE
        best_method = min(valid_results.items(), key=lambda x: x[1]['rmse'])[0]
        logger.info(f"\nBest Ensemble Method: {best_method}")
        logger.info(f"Performance of {best_method}:")
        logger.info(f"  MSE: {valid_results[best_method]['mse']:.6f}")
        logger.info(f"  RMSE: {valid_results[best_method]['rmse']:.6f}")
        logger.info(f"  R2: {valid_results[best_method]['r2']:.6f}")

        # Do a final verification of the best method's predictions
        verify_prediction_scale(logger, Y_test, valid_results[best_method]['prediction'], 
                              f"Best ensemble method ({best_method})", tolerance=0.2)
        
    else:
        logger.warning("All ensemble methods failed")

    return results

