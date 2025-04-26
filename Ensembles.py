# Ensembles.py
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import Models_Creation_and_Training
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
def linearly_weighted_ensemble(models_results, X_test, target_scaler, feature_scaler):
    """
    Create a linearly weighted ensemble prediction across different model types.
    Parameters:
    - models_results: Dictionary containing model results
    - X_test: Test data to make predictions on (DataFrame or NumPy array)

    Returns:
    - Final ensemble prediction
    """
    
    # Convert DataFrame to NumPy array if necessary
    # scaler = StandardScaler()
    # X_test = scaler.fit_transform(X_test)

    # Convert DataFrame to NumPy array if necessary - NO ADDITIONAL SCALING
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    
    mae_values = []
    model_predictions = [] 

    # Calculate Mean Absolute Error (MAE) for each model
    for model_name, result in models_results.items():
        if 'best_model' not in result or result['best_model'] is None:
            continue


        mae_value = np.mean(np.abs(result['best_model_prediction'] - result['Y_val_best']))
        mae_values.append(mae_value)

        # Prepare predictions based on model type
        if model_name.strip() == 'LSTM':
            # Reshape for LSTM
            X_test_lstm = prepare_lstm_data(X_test_array, time_steps=5)
            model_pred = result['best_model'].predict(X_test_lstm)
        else:
            # For other models
            model_pred = result['best_model'].predict(X_test_array)

        # Ensure 1D prediction
        model_predictions.append(model_pred.reshape(-1))

    # Calculate inverse MAE weights
    weights = [mae_value ** -1 for mae_value in mae_values]
    weights = np.array(weights) / np.sum(weights)

    # Compute the final ensemble prediction
    final_prediction = np.zeros(len(X_test_array), dtype=np.float64)

    # Apply weighted predictions
    for pred, weight in zip(model_predictions, weights):
        final_prediction += weight * pred
    
    # Inverse transform to get back to original scale
    final_prediction = target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()

    return final_prediction

# Equal Weights Ensemble
def equal_weighted_ensemble(models_results, X_test, target_scaler, feature_scaler):
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
    
    # Prepare predictions based on model type
    for model_name, result in models_results.items():
        if 'best_model' not in result or result['best_model'] is None:
            continue

        if model_name.strip() == 'LSTM':
            # Reshape for LSTM, using data directly
            X_test_lstm = prepare_lstm_data(X_test_array, time_steps=5)  # Use time_steps=5 to match training
            model_pred = result['best_model'].predict(X_test_lstm)
        else:
            # For other models, use data directly
            model_pred = result['best_model'].predict(X_test_array)

        # Ensure 1D prediction
        model_predictions.append(model_pred.reshape(-1))

    if not model_predictions:
        raise ValueError("No predictions available for ensemble methods")

    # Calculate weight (equal for all models)
    weight = 1.0 / len(model_predictions)

    # Compute the final ensemble prediction
    final_prediction = np.zeros(len(X_test_array), dtype=np.float64)

    # Apply weighted predictions
    for pred in model_predictions:
        final_prediction += weight * pred
    
    # Inverse transform to get back to original scale
    final_prediction = target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()

    return final_prediction

# Gradient Boosting Decision Tree Ensemble
def gbdt_ensemble(models_results, X_train, X_test, Y_train, target_scaler, feature_scaler):
    """
    Use GBDT to predict based on the predictions of base models.
    Implements a validation split approach to prevent data leakage.
    Parameters:
    - models_results: Dictionary containing model results, with 'best_model' key for each model.
    - X_train: Training data features to generate meta-features for training GBDT.
    - X_test: Test data features for final prediction.
    - Y_train_val: Training labels for fitting the GBDT model.
    
    Returns: 
    - GBDT ensemble prediction for test data.
    """

    # Convert to arrays without scaling - data is already PCA-transformed
    X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test

    # Split training data into training and validation sets to prevent leakage
    X_meta_train, X_meta_val, Y_meta_train, Y_meta_val = train_test_split(
        X_train_array, Y_train, test_size=0.2, random_state=42, shuffle=False
    )

    # Generate meta-features for both sets
    train_meta_features = []
    val_meta_features = []
    test_meta_features = []

    for model_name, result in models_results.items():
        if 'best_model' not in result or result['best_model'] is None:
            continue

        model = result['best_model']

        # Generate predictions for all three sets
        if model_name.strip() == 'LSTM':
            # Reshape for LSTM
            X_meta_train_lstm = prepare_lstm_data(X_meta_train, time_steps=5)
            X_meta_val_lstm = prepare_lstm_data(X_meta_val, time_steps=5)
            X_test_lstm = prepare_lstm_data(X_test_array, time_steps=5)
            
            train_pred = model.predict(X_meta_train_lstm)
            val_pred = model.predict(X_meta_val_lstm)
            test_pred = model.predict(X_test_lstm)
        else:
            train_pred = model.predict(X_meta_train)
            val_pred = model.predict(X_meta_val)
            test_pred = model.predict(X_test_array)
        
        # Store predictions for each set
        train_meta_features.append(train_pred.reshape(-1))
        val_meta_features.append(val_pred.reshape(-1))
        test_meta_features.append(test_pred.reshape(-1))

    # Stack predictions as meta-features
    X_meta_train_stacked = np.column_stack(train_meta_features)
    X_meta_val_stacked = np.column_stack(val_meta_features)
    X_test_meta = np.column_stack(test_meta_features)

    # Train GBDT on meta-features
    gb_model = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.01, 
        max_depth=3, 
        random_state=42
    )

    # Fit on training meta-features
    gb_model.fit(X_meta_train_stacked, Y_meta_train)
    
    # Evaluate on validation set to check for overfitting
    val_pred = gb_model.predict(X_meta_val_stacked)
    val_mse = mean_squared_error(Y_meta_val, val_pred)
    print(f"GBDT Ensemble - Validation MSE: {val_mse:.6f}, RMSE: {np.sqrt(val_mse):.6f}")

   # Predict on test meta-features
    final_prediction = gb_model.predict(X_test_meta)

    # Inverse transform to get back to original scale
    final_prediction = target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()
    
    return final_prediction

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
    y_sample = Y_test.iloc[:3].values.reshape(-1, 1)
    y_scaled = target_scaler.transform(y_sample)
    y_restored = target_scaler.inverse_transform(y_scaled)
    logger.info(f"  Original Y values: {y_sample.flatten()}")
    logger.info(f"  Scaled Y values: {y_scaled.flatten()}")
    logger.info(f"  Restored Y values: {y_restored.flatten()}")

    def linearly_weighted_wrapper(results, x_test):
        logger.info("Running linearly weighted ensemble...")
        return linearly_weighted_ensemble(results, x_test, target_scaler, None)  # Pass None for feature_scaler

    def equal_weighted_wrapper(results, x_test):
        logger.info("Running equal weighted ensemble...")
        return equal_weighted_ensemble(results, x_test, target_scaler, None)  # Pass None for feature_scaler

    def gbdt_wrapper(results, x_test):
        logger.info("Running GBDT ensemble...")
        return gbdt_ensemble(results, X_train, x_test, Y_train, target_scaler, None)  # Pass None for feature_scaler
        
    
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
            

            # Calculate performance metrics
            mse = mean_squared_error(Y_test, final_prediction)
            rmse = np.sqrt(mse)
            r2 = r2_score(Y_test, final_prediction)
            mae = np.mean(np.abs(Y_test - final_prediction))

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
        print("\nEnsemble Methods Comparison:")
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
        print(comparison_df)
            
        # Find the best performing method based on RMSE
        best_method = min(valid_results.items(), key=lambda x: x[1]['rmse'])[0]
        print(f"\nBest Ensemble Method: {best_method}")
        print(f"Performance of {best_method}:")
        print(f"  MSE: {valid_results[best_method]['mse']:.6f}")
        print(f"  RMSE: {valid_results[best_method]['rmse']:.6f}")
        print(f"  R2: {valid_results[best_method]['r2']:.6f}")

        # Do a final verification of the best method's predictions
        verify_prediction_scale(logger, Y_test, valid_results[best_method]['prediction'], 
                              f"Best ensemble method ({best_method})", tolerance=0.2)
        
    else:
        print("All ensemble methods failed")

    return results


