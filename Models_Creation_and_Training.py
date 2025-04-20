import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

from Logging_and_Validation import log_data_stats, verify_prediction_scale

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
    

def train_and_validate_models(logger, X_train_val, Y_train_val):
    print(f"\n{'-'*30}\nInitializing model training\n{'-'*30}")
    log_data_stats(logger, X_train_val, "X_train_val for models", include_stats=False)
    log_data_stats(logger, Y_train_val, "Y_train_val for models", include_stats=True)

    # Create global scalers for final output
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit the global scalers on all training data for later use with test data
    feature_scaler.fit(X_train_val)
    target_scaler.fit(Y_train_val.values.reshape(-1, 1))

    # Log scaling info
    X_scaled_sample = feature_scaler.transform(X_train_val.iloc[:5])
    Y_scaled_sample = target_scaler.transform(Y_train_val.iloc[:5].values.reshape(-1, 1))
    print("\nScaling Information:")
    print(f"X scale sample - Before: {X_train_val.iloc[0, :5].values}")
    print(f"X scale sample - After: {X_scaled_sample[0, :5]}")
    print(f"Y scale sample - Before: {Y_train_val.iloc[:5].values}")
    print(f"Y scale sample - After: {Y_scaled_sample.flatten()}")

    # Time series cross-validation with 5 folds, to ensure temporal order (sequence of events in time)
    tscv = TimeSeriesSplit(n_splits=5)                                   
    models = create_models()
    results = {}

    # Iterate each model
    for model_name, (model, params_grid) in models.items():    
        print(f"\n{'-'*30}\nTraining {model_name}\n{'-'*30}")                     

        best_mse_scores = []                                                               # lists to store scores and parameters for the models
        best_rmse_scores = []
        best_params = []
        best_score = float('inf')
        best_model = None
        best_model_prediction = None
        Y_val_best = None

        # Reshape the data for LSTM model
        if model_name == 'LSTM':
            print(f"Handling LSTM model separately with special reshaping")
            results[model_name] = train_lstm_model_with_cv(X_train_val, Y_train_val, tscv, feature_scaler, target_scaler)
            continue

        # Add timeout or max iterations for SVR
        if model_name == 'SVR':
            model.set_params(max_iter=1000)  # Limit iterations

        # Each fold: split the data and to train and val (using the tcsv indices) then scale
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)): 
            print(f"\nFold {fold + 1} for {model_name}:")

            # Split the data
            X_train_fold, X_val_fold = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            Y_train_fold, Y_val_fold = Y_train_val.iloc[train_idx], Y_train_val.iloc[val_idx]

            print(f"  Train fold shapes: X={X_train_fold.shape}, Y={Y_train_fold.shape}")
            print(f"  Validation fold shapes: X={X_val_fold.shape}, Y={Y_val_fold.shape}")

            # Scale features and target using fold-specific scalers
            fold_feature_scaler = StandardScaler()
            X_train_scaled = fold_feature_scaler.fit_transform(X_train_fold)
            X_val_scaled = fold_feature_scaler.transform(X_val_fold)
            
            fold_target_scaler = StandardScaler()
            Y_train_scaled = fold_target_scaler.fit_transform(Y_train_fold.values.reshape(-1, 1)).flatten()
            Y_val_scaled = fold_target_scaler.transform(Y_val_fold.values.reshape(-1, 1)).flatten()

            print(f"  Train scaled shapes: X={X_train_scaled.shape}, Y={Y_train_scaled.shape}")
            print(f"  Validation scaled shapes: X={X_val_scaled.shape}, Y={Y_val_scaled.shape}")

            # Grid search
            print(f"  Running GridSearchCV for {model_name} on fold {fold + 1}...")
            # Reduce the number of CV splits for SVR to speed up
            cv_splits = 2 if model_name == 'SVR' else 3

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params_grid,
                scoring='neg_mean_squared_error',
                cv=cv_splits,
                n_jobs=-1,
                verbose=1
            )

            try:
                # Train the model for every combination of parameters on each training set of the fold
                # Save the best params
                grid_search.fit(X_train_scaled, Y_train_scaled)               
                best_params.append(grid_search.best_params_)
                print(f"  Best parameters: {grid_search.best_params_}")

                # Validate using the best model
                best_model_fold = grid_search.best_estimator_
                Y_pred_scaled  = best_model_fold.predict(X_val_scaled)                                
            
                # Convert predictions back to original scale
                Y_pred = fold_target_scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()
                Y_val_original = Y_val_fold.values       

                # Scale verification
                min_ratio, max_ratio = verify_prediction_scale(logger, Y_val_original, Y_pred, f"{model_name} fold {fold+1}")
                
                # Calculate performance metrics (on original scale)
                mse = mean_squared_error(Y_val_original, Y_pred)
                rmse = np.sqrt(mse)
                best_mse_scores.append(mse)
                best_rmse_scores.append(rmse)
                print(f"  Fold {fold + 1} - MSE: {mse:.4f}, RMSE: {rmse:.4f}")

                # Update best model
                if mse < best_score:
                    best_score = mse
                    best_model = best_model_fold
                    best_model_prediction = Y_pred
                    Y_val_best= Y_val_original
                    print(f"  New best model found with MSE: {mse:.4f}")


            except Exception as e:
                print(f"  Error in {model_name} training on fold {fold + 1}: {e}")
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

        print(f"\nSummary for {model_name}:")
        if best_mse_scores:
            avg_mse = np.mean(best_mse_scores)
            avg_rmse = np.mean(best_rmse_scores)
            print(f"  Average MSE: {avg_mse:.4f}, Average RMSE: {avg_rmse:.4f}")
        else:
            print(f"  No successful folds for {model_name}")
        
    # After all folds, retrain best model on all data using global scalers
    # This ensures the final model can be used with the global scalers
    print("\nRetraining all models on full dataset using global scalers...")
    X_train_val_scaled = feature_scaler.transform(X_train_val)
    Y_train_val_scaled = target_scaler.transform(Y_train_val.values.reshape(-1, 1)).flatten()

    # For each model (except LSTM which is handled separately)
    for model_name, result in results.items():
        if model_name != 'LSTM' and result.get('best_params') and len(result['best_params']) > 0:
            try:
                print(f"Retraining {model_name} on full dataset...")
            
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

                print(f"  Retrained {model_name} - MSE: {retrain_mse:.4f}, RMSE: {retrain_rmse:.4f}")
                verify_prediction_scale(logger, Y_train_val, Y_retrain_pred, f"{model_name} retrained")
                
            
                # Replace the best model
                results[model_name]['best_model'] = base_model
                print(f"  Successfully retrained {model_name}")
                
            except Exception as e:
                print(f"  Error retraining {model_name}: {e}")
                print("  Keeping the best fold model instead")
                # Keep the fold model if retraining fails
                pass

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
       

"""
Train LSTM model using time series cross-validation with manual parameter search.
-> Used in train_and_validate_models() function to handle LSTM model separately!!
"""
def train_lstm_model_with_cv(X_train_val, Y_train_val, tscv, feature_scaler, target_scaler):
    """
    Train LSTM model using time series cross-validation with manual parameter search.
    Handles scaling per fold to prevent data leakage.
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

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
            print(f"\nTraining on fold {fold + 1} for LSTM...")

            # Split the data
            X_train_fold = X_train_val.iloc[train_idx]
            X_val_fold = X_train_val.iloc[val_idx]
            Y_train_fold = Y_train_val.iloc[train_idx]
            Y_val_fold = Y_train_val.iloc[val_idx]

            # Scale features and target using fold-specific scalers
            fold_feature_scaler = StandardScaler()
            X_train_scaled = fold_feature_scaler.fit_transform(X_train_fold)
            X_val_scaled = fold_feature_scaler.transform(X_val_fold)
            
            fold_target_scaler = StandardScaler()
            Y_train_scaled = fold_target_scaler.fit_transform(Y_train_fold.values.reshape(-1, 1)).flatten()
            Y_val_scaled = fold_target_scaler.transform(Y_val_fold.values.reshape(-1, 1)).flatten()

            # Perform a manual grid search across LSTM parameters
            fold_best_mse = float('inf')
            fold_best_params = None
            fold_best_model = None
            fold_best_preds = None
            fold_best_preds_original = None

            print(f"Performing manual grid search for LSTM on fold {fold + 1}...")

            for params in lstm_param_grid:
                try:
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
                
                # Scale all training data with global scalers
                X_train_val_scaled = feature_scaler.transform(X_train_val)
                Y_train_val_scaled = target_scaler.transform(Y_train_val.values.reshape(-1, 1)).flatten()
                
                # Reshape data for LSTM [samples, time steps, features]
                features_count = X_train_val_scaled.shape[1]
                X_train_val_reshaped = X_train_val_scaled.reshape(-1, 1, features_count)

                # Create and compile the final model
                final_model = Sequential([
                    LSTM(final_best_params['units'], activation='relu', input_shape=(1, features_count)),
                    Dense(1)
                ])
                final_model.compile(optimizer=Adam(learning_rate=final_best_params['learning_rate']), loss='mse')
                
                # Train the final model
                final_model.fit(
                    X_train_val_reshaped, 
                    Y_train_val_scaled,
                    epochs=final_best_params['epochs'],
                    batch_size=final_best_params['batch_size'],
                    verbose=0
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