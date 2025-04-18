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
    

def train_and_validate_models(X_train_val, Y_train_val):
    # Scale the features (features are scaled to have zero mean and unit variance)
    scaler = StandardScaler()                                                      
    X_train_val_scaled = scaler.fit_transform(X_train_val)

    # Add target scaling
    target_scaler = StandardScaler()
    Y_train_val_scaled = target_scaler.fit_transform(Y_train_val.values.reshape(-1, 1)).flatten()
    # Save the scaler for later use (important for inverse transformation)
    global shared_target_scaler
    shared_target_scaler = target_scaler

    # Time series cross-validation with 5 folds, to ensure temporal order (sequence of events in time)
    tscv = TimeSeriesSplit(n_splits=5)                                   
    models = create_models()
    results = {}

    # Iterate each model
    for model_name, (model, params_grid) in models.items():                         
        print(f"Training {model_name}")

        # Reshape the data for LSTM model
        if model_name == 'LSTM':
            results[model_name] = train_lstm_model_with_cv(X_train_val_scaled, Y_train_val_scaled, tscv)
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

        # Each fold: split the scaled data to train and val (using the tcsv indices)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val_scaled)): 
            print(f"\nTraining on fold {fold + 1} for {model_name}...")
            # Split the data
            X_train, X_val = X_train_val_scaled[train_idx], X_train_val_scaled[val_idx]
            Y_train, Y_val = Y_train_val_scaled[train_idx], Y_train_val_scaled[val_idx]

            # Find the best params for the current model
            print(f"Performing GridSearchCV for {model_name} on fold {fold + 1}...")

            # Reduce the number of CV splits for SVR to speed up
            cv_splits = 2 if model_name == 'SVR' else 3

            # grid_search = GridSearchCV(estimator=model, param_grid=params_grid, scoring='neg_mean_squared_error', cv=cv_splits, n_jobs=-1, verbose=1) # performs search over set of parameters
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params_grid,
                scoring='neg_mean_squared_error',
                cv=cv_splits,
                n_jobs=-1,
                verbose=1
            )

            try:
                # train the model for every combination of parameters on each training set of the fold
                grid_search.fit(X_train, Y_train)               
                print(f"Best parameters found for fold {fold + 1}: {grid_search.best_params_}")
                # Save the best params
                best_params.append(grid_search.best_params_)
                # Train the best model -> the model that performs the best on the last search
                best_model_fold = grid_search.best_estimator_                             
                # Validate using the model - > use the best model after training and validate
                print("Starting validation: ")
                Y_pred = best_model_fold.predict(X_val)                                   
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
            
            # Save the best model for each model type, the results and the parameters

        results[model_name] = {                                                       
            'best_mse_scores': best_mse_scores,
            'best_rmse_scores': best_rmse_scores,
            'best_params': best_params,
            'best_model': best_model,
            'best_model_prediction': best_model_prediction,
            'Y_val_best' : Y_val_best
        }

    return results  


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
            Y_train, Y_val = Y_train_val[train_idx], Y_train_val[val_idx]

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