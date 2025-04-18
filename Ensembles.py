import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import Models_Creation_and_Training

# ===============================================================================
# Ensembles
# ===============================================================================
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
    
    # Inverse transform to get back to original scale
    final_prediction = Models_Creation_and_Training.shared_target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()

    return final_prediction

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
    
    # Inverse transform to get back to original scale
    final_prediction = Models_Creation_and_Training.shared_target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()

    return final_prediction

# Gradient Boosting Decision Tree Ensemble
def gbdt_ensemble(models_results, X_train, X_test, Y_train_val):
    """
    Use GBDT to predict based on the predictions of base models.
    Parameters:
    - models_results: Dictionary containing model results, with 'best_model' key for each model.
    - X_train: Training data features to generate meta-features for training GBDT.
    - X_test: Test data features for final prediction.
    - Y_train_val: Training labels for fitting the GBDT model.
    
    Returns: 
    - GBDT ensemble prediction for test data.
    """
    
    # Scale the data
    scaler = StandardScaler()
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

    # Inverse transform to get back to original scale
    final_prediction = Models_Creation_and_Training.shared_target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()

    return final_prediction

# ===============================================================================
# Three Ensembles Pipeline
# ===============================================================================
def ensemble_pipeline(models_results, X_train, X_test, Y_train, Y_test):
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

        # Calculate performance metrics against unscaled targets
        # The predictions are already inverse transformed in the ensemble methods
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


