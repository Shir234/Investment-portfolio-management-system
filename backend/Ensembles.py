# Ensembles.py
"""
Ensembles.py - Multi-method ensemble learning for financial prediction

Combines predictions from multiple ML models using three approaches:
1. Linearly Weighted: Performance-based weighting (R²/RMSE²)
2. Equal Weighted: Simple averaging of all models
3. GBDT Meta-Learning: Gradient boosting to learn optimal combinations

Key features:
- Dynamic clipping based on validation data ranges
- LSTM-aware prediction handling with time steps
- Comprehensive performance comparison and validation
"""
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from Models_Creation_and_Training import create_rolling_window_data
import optuna
import Models_Creation_and_Training as Models_Creation_and_Training
from Logging_and_Validation import verify_prediction_scale
import traceback


def prepare_lstm_data(X, time_steps=3, features=None):
    """
    Converts regular data into rolling windows for LSTM input.
    
    Creates sequences where each sample contains 'time_steps' consecutive observations.
    Essential for LSTM models that require sequential input data.
    """

    if hasattr(X, 'values'):
        X = X.values
    if features is not None:
        X = X[:, features]
    if len(X) < time_steps + 1:
        raise ValueError(f"Not enough samples ({len(X)}) for time_steps ({time_steps})")
    X_rolled = []
    for i in range(len(X) - time_steps):
        X_rolled.append(X[i:i + time_steps])
    return np.array(X_rolled)


def get_dynamic_clipping_range(models_results, logger=None):
    """
    Calculates reasonable prediction bounds from validation data.
    
    Uses 1st and 99th percentiles of all model validation targets to prevent
    extreme predictions that could destabilize ensemble results.
    """

    y_values = []
    
    for model_name, result in models_results.items():
        # Collect y values from regular models
        if 'Y_val_best' in result and result['Y_val_best'] is not None:
            y_values.extend(result['Y_val_best'])
        
        # Collect y values from LSTM fold metrics
        if model_name.strip() == 'LSTM' and isinstance(result, dict) and 'fold_metrics' in result:
            for fold in result['fold_metrics']:
                if 'y_val' in fold and fold['y_val'] is not None:
                    if hasattr(fold['y_val'], 'values'):
                        y_values.extend(fold['y_val'].values)
                    else:
                        y_values.extend(fold['y_val'])
    
    # Calculate percentiles if we have data
    if y_values:
        y_min = np.percentile(y_values, 1)  # 1st percentile
        y_max = np.percentile(y_values, 99)  # 99th percentile
    else:
        # Fallback to reasonable defaults
        y_min, y_max = -5.0, 5.0
        if logger:
            logger.warning("No validation data found for clipping range, using defaults [-5.0, 5.0]")
    
    if logger:
        logger.info(f"Dynamic clipping range: [{y_min:.4f}, {y_max:.4f}]")
    
    return y_min, y_max


def linearly_weighted_ensemble(models_results, X_test, target_scaler, logger=None):
    """
    Performance-weighted ensemble using inverse RMSE squared weighting.
    
    Weight formula: (R² + 1) / (RMSE²)
    - Rewards models with high R² and low RMSE
    - Filters out poor models (R² <= 0.5)
    """

    if not models_results:
        raise ValueError("Empty models_results dictionary")
    
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test

    if len(X_test_array) < 3 and any('LSTM' in model_name for model_name in models_results):
        raise ValueError("X_test too small for LSTM (needs >= 3 samples)")

    if logger:
        logger.info(f"Models results keys: {list(models_results.keys())}")
        for name, res in models_results.items():
            logger.info(f"{name} keys: {list(res.keys()) if isinstance(res, dict) else 'List of length ' + str(len(res))}")

    weights = []
    model_predictions = []
    target_len = len(X_test)

    # Get dynamic clipping range
    clip_min, clip_max = get_dynamic_clipping_range(models_results, logger)

    for model_name, result in models_results.items():
        try:
            if logger:
                logger.info(f"Processing {model_name} for linearly_weighted_ensemble")
            
            # Handle LSTM results
            if model_name.strip() == 'LSTM' and isinstance(result, dict) and 'fold_metrics' in result:
                fold_metrics = result['fold_metrics']
                if logger:
                    logger.info(f"LSTM fold_metrics keys: {[list(f.keys()) for f in fold_metrics]}")
                if not isinstance(fold_metrics, list):
                    if logger:
                        logger.warning(f"LSTM fold_metrics is not a list, skipping")
                    continue
               
                best_fold_idx = np.argmin([r.get('MSE', float('inf')) for r in fold_metrics])
                fold_result = fold_metrics[best_fold_idx]
                model = fold_result.get('model')
                best_params = fold_result.get('Parameters', {})
                r2_value = fold_result.get('R2', 0.0)
                rmse = fold_result.get('RMSE', fold_result.get('MSE', float('inf')) ** 0.5)

                if model is None:
                    if logger:
                        logger.warning(f"LSTM fold {best_fold_idx} missing model")
                    continue
            else:
                # Handle non-LSTM models
                if 'retrained_model' not in result or result['retrained_model'] is None:
                    if logger:
                        logger.warning(f"No retrained model for {model_name}, using best_model")
                    if 'best_model' not in result or result['best_model'] is None:
                        if logger:
                            logger.warning(f"No best_model for {model_name}, skipping")
                        continue

                    model = result['best_model']
                    best_fold_idx = np.argmin(result['best_rmse_scores'])
                    r2_value = result['best_r2_scores'][best_fold_idx]
                    rmse = result['best_rmse_scores'][best_fold_idx]

                else:
                    model = result['retrained_model']
                    r2_value = result.get('retrained_r2', max(result['best_r2_scores']))
                    rmse = result.get('retrained_rmse', min(result['best_rmse_scores']))
                best_params = result.get('best_params', {})

            if r2_value <= 0.5:
                if logger:
                    logger.warning(f"Skipping {model_name} due to low R²: {r2_value}")
                continue

            # Generate predictions
            if model_name.strip() == 'LSTM':
                time_steps = best_params.get('time_steps', 3)
                X_test_lstm = prepare_lstm_data(X_test_array, time_steps=time_steps)
                if len(X_test_lstm) == 0:
                    if logger:
                        logger.warning(f"Not enough data for LSTM predictions")
                    continue

                model_pred = model.predict(X_test_lstm, verbose=0).flatten()
                # Pad predictions to match target length
                model_pred = np.pad(model_pred, (0, max(0, target_len - len(model_pred))), 
                                    mode='constant', constant_values=np.mean(model_pred) if len(model_pred) > 0 else 0)
            else:
                model_pred = model.predict(X_test_array).flatten()
                model_pred = model_pred[:target_len]

            # Apply inverse transform and clipping
            model_pred = target_scaler.inverse_transform(model_pred.reshape(-1, 1)).flatten()
            model_pred = np.clip(model_pred, clip_min, clip_max)
            model_predictions.append(model_pred)
            
            # Weighting: emphasize low RMSE and high R²
            weight = (r2_value + 1) / (rmse ** 2) if rmse > 0 else 0.0
            weights.append(weight)
            
            if logger:
                logger.info(f"Model {model_name} - R²: {r2_value:.4f}, RMSE: {rmse:.4f}, "
                           f"Weight: {weight:.4f}, Prediction shape: {model_pred.shape}")
        
        except Exception as e:
            if logger:
                logger.error(f"Error generating predictions for {model_name}: {e}")
            continue

    if not model_predictions:
        raise ValueError("No valid model predictions available")

    weights = np.array(weights)
    # Ensure minimum weight to prevent zero weights
    min_weight = 0.05 * np.max(weights)
    weights = np.where(weights < min_weight, min_weight, weights)
    weights = weights / np.sum(weights)
    if logger:
        logger.info(f"Model weights: {dict(zip(models_results.keys(), weights))}")

    final_prediction = np.zeros(target_len, dtype=np.float64)
    for pred, weight in zip(model_predictions, weights):
        final_prediction += weight * pred

    final_prediction = np.clip(final_prediction, clip_min, clip_max)

    return final_prediction


def equal_weighted_ensemble(models_results, X_test, target_scaler, logger=None):
    """
    Simple averaging ensemble - all models weighted equally (1/n each).
    
    Good approach against overfitting to validation metrics.
    Provides a good baseline for ensemble comparison.
    """
    
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    model_predictions = []
    target_len = len(X_test)

    # Get dynamic clipping range
    clip_min, clip_max = get_dynamic_clipping_range(models_results, logger)

    for model_name, result in models_results.items():
        try:
            if logger:
                logger.info(f"Processing {model_name} for equal_weighted_ensemble")
                logger.info(f"{model_name} result keys: {list(result.keys())}")
            
            # Handle LSTM results
            if model_name.strip() == 'LSTM' and isinstance(result, dict) and 'fold_metrics' in result:
                fold_metrics = result['fold_metrics']
                if logger:
                    logger.info(f"LSTM fold_metrics keys: {[list(f.keys()) for f in fold_metrics]}")
                if not isinstance(fold_metrics, list):
                    if logger:
                        logger.warning(f"LSTM fold_metrics is not a list, skipping")
                    continue

                best_fold_idx = np.argmin([r.get('MSE', float('inf')) for r in fold_metrics])
                fold_result = fold_metrics[best_fold_idx]
                model = fold_result.get('model')
                best_params = fold_result.get('Parameters', {})
                
                if model is None:
                    if logger:
                        logger.warning(f"LSTM fold {best_fold_idx} missing model")
                    continue
           
            else:
                # Handle non-LSTM models
                if 'retrained_model' not in result or result['retrained_model'] is None:
                    if logger:
                        logger.warning(f"No retrained model for {model_name}, using best_model")
                    if 'best_model' not in result or result['best_model'] is None:
                        if logger:
                            logger.warning(f"No best_model for {model_name}, skipping")
                        continue
                    model = result['best_model']
                else:
                    model = result['retrained_model']
                best_params = result.get('best_params', {})

            # Generate predictions
            if model_name.strip() == 'LSTM':
                time_steps = best_params.get('time_steps', 3)
                X_test_lstm = prepare_lstm_data(X_test_array, time_steps=time_steps)
                if len(X_test_lstm) == 0:
                    if logger:
                        logger.warning(f"Not enough data for LSTM predictions")
                    continue

                model_pred = model.predict(X_test_lstm, verbose=0).flatten()
                # Pad predictions to match target length
                model_pred = np.pad(model_pred, (0, max(0, target_len - len(model_pred))), 
                                    mode='constant', constant_values=np.mean(model_pred) if len(model_pred) > 0 else 0)
            
            else:
                model_pred = model.predict(X_test_array).flatten()
                model_pred = model_pred[:target_len]
            # Apply inverse transform and clipping
            model_pred = target_scaler.inverse_transform(model_pred.reshape(-1, 1)).flatten()
            model_pred = np.clip(model_pred, clip_min, clip_max)
            model_predictions.append(model_pred)

            if logger:
                logger.info(f"Model {model_name} - Prediction shape: {model_pred.shape}")
        except Exception as e:
            if logger:
                logger.error(f"Error generating predictions for {model_name}: {e}")
            continue
    
    if not model_predictions:
        raise ValueError("No predictions available for ensemble methods")
    
    weight = 1.0 / len(model_predictions)
    final_prediction = np.zeros(target_len, dtype=np.float64)

    for pred in model_predictions:
        final_prediction += weight * pred
    
    final_prediction = np.clip(final_prediction, clip_min, clip_max)

    return final_prediction


def gbdt_ensemble(models_results, X_train, X_test, Y_train, target_scaler, logger):
    """
    Meta-learning ensemble using Gradient Boosting Decision Trees.
    
    Two-stage process:
    1. Generate base model predictions on training data (meta-features)
    2. Train GBDT meta-model to optimally combine these predictions
    3. Apply trained meta-model to test predictions
    
    Can learn complex, non-linear combinations but requires sufficient training data.
    Uses Optuna for hyperparameter optimization of the meta-model.
    """

    try:
        meta_features_train = []
        meta_features_test = []
        min_len = len(X_test)
        logger.info(f"GBDT: Target length for predictions: {min_len}")
        
        valid_models = []
        if not isinstance(models_results, dict):
            logger.error(f"Results is {type(models_results)}, expected a dictionary")
            return np.zeros(min_len)
        
        # Get dynamic clipping range
        clip_min, clip_max = get_dynamic_clipping_range(models_results, logger)
        
        for model_name, result in models_results.items():
            try:
                if logger:
                    logger.info(f"Processing {model_name} for gbdt_ensemble")
                    logger.info(f"{model_name} result keys: {list(result.keys())}")
                
                # Handle LSTM results
                if model_name.strip() == 'LSTM' and isinstance(result, dict) and 'fold_metrics' in result:
                    fold_metrics = result['fold_metrics']
                    if logger:
                        logger.info(f"LSTM fold_metrics keys: {[list(f.keys()) for f in fold_metrics]}")
                    if not isinstance(fold_metrics, list):
                        if logger:
                            logger.warning(f"LSTM fold_metrics is not a list, skipping")
                        continue

                    best_fold_idx = np.argmin([r.get('MSE', float('inf')) for r in fold_metrics])
                    fold_result = fold_metrics[best_fold_idx]
                    model = fold_result.get('model')
                    best_params = fold_result.get('Parameters', {})

                    if model is None:
                        if logger:
                            logger.warning(f"LSTM fold {best_fold_idx} missing model")
                        continue

                else:
                    # Handle non-LSTM models
                    if 'retrained_model' not in result or result['retrained_model'] is None:
                        if logger:
                            logger.warning(f"No retrained model for {model_name}, using best_model")
                        if 'best_model' not in result or result['best_model'] is None:
                            if logger:
                                logger.warning(f"No best_model for {model_name}, skipping")
                            continue
                        model = result['best_model']
                    else:
                        model = result['retrained_model']
                    best_params = result.get('best_params', {})
                
                logger.info(f"Processing {model_name} predictions")

                # Convert inputs to arrays
                X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
                X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
                Y_train_array = Y_train.values if hasattr(Y_train, 'values') else Y_train
                
                if model_name.strip() == 'LSTM':
                    time_steps = best_params.get('time_steps', 3)
                    if X_train.shape[0] < time_steps or X_test.shape[0] < time_steps:
                        logger.warning(f"Insufficient samples for LSTM time_steps={time_steps}")
                        continue

                    X_train_rolled, y_train_rolled = create_rolling_window_data(X_train.values, Y_train.values, time_steps)
                    X_test_rolled, _ = create_rolling_window_data(X_test.values, np.zeros(len(X_test)), time_steps)
                    logger.info(f"LSTM: X_train_rolled shape: {X_train_rolled.shape}, X_test_rolled shape: {X_test_rolled.shape}")
                    
                    train_pred = model.predict(X_train_rolled, verbose=0).flatten()
                    test_pred = model.predict(X_test_rolled, verbose=0).flatten()

                    # Adjust lengths for LSTM predictions
                    train_pred = train_pred[:min_len]
                    test_pred = np.pad(test_pred, (0, max(0, min_len - len(test_pred))), 
                                       mode='constant', constant_values=np.mean(test_pred) if len(test_pred) > 0 else 0)
                
                else:
                    train_pred = model.predict(X_train).flatten()[:min_len]
                    test_pred = model.predict(X_test).flatten()[:min_len]
                
                # Apply inverse transform and clipping
                train_pred = target_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
                test_pred = target_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
                
                # Standardize for meta-model
                scaler = StandardScaler()
                train_pred = scaler.fit_transform(train_pred.reshape(-1, 1)).flatten()
                test_pred = scaler.transform(test_pred.reshape(-1, 1)).flatten()
                
                train_pred = np.clip(train_pred, -3.6815, 4.6223)
                test_pred = np.clip(test_pred, -3.6815, 4.6223)
                
                meta_features_train.append(train_pred)
                meta_features_test.append(test_pred)
                valid_models.append(model_name)
                logger.info(f"{model_name}: Final train_pred length: {len(train_pred)}, test_pred length: {len(test_pred)}")
            
            except Exception as e:
                logger.error(f"Error processing {model_name} for GBDT: {e}")
                continue
        
        if not meta_features_train or not meta_features_test:
            logger.error("No valid meta-features generated")
            return np.zeros(min_len)
        
        meta_features_train = np.array(meta_features_train).T
        meta_features_test = np.array(meta_features_test).T

        # Check correlation and remove highly correlated features
        corr_matrix = np.corrcoef(meta_features_train.T)
        logger.info(f"Meta-feature correlations:\n{corr_matrix}")
        
        to_keep = []
        for i in range(corr_matrix.shape[0]):
            if i == 0 or all(abs(corr_matrix[i, j]) < 0.85 for j in to_keep):
                to_keep.append(i)
        
        meta_features_train = meta_features_train[:, to_keep]
        meta_features_test = meta_features_test[:, to_keep]
        valid_models = [valid_models[i] for i in to_keep]
        logger.info(f"Kept models after correlation filter: {valid_models}")
        
        # Scale meta-features
        scaler = StandardScaler()
        meta_features_train = scaler.fit_transform(meta_features_train)
        meta_features_test = scaler.transform(meta_features_test)

        # Prepare target for meta-model
        Y_train_meta = Y_train_array[:len(meta_features_train)]
        
        meta_train, meta_val, y_train_meta, y_val_meta = train_test_split(
            meta_features_train, Y_train[:min_len], test_size=0.1, random_state=42
        )
        
        # Optimize meta-model
        def optimize_meta_model(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 2, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2)
            }
            meta_model = GradientBoostingRegressor(**params, random_state=42)
            meta_model.fit(meta_train, y_train_meta)
            pred = meta_model.predict(meta_val)
            return np.sqrt(mean_squared_error(y_val_meta, pred))
        
        study = optuna.create_study(direction='minimize')
        study.optimize(optimize_meta_model, n_trials=100)
        best_params = study.best_params
        logger.info(f"Optimized meta-model parameters: {best_params}")
        
        # Train final meta-model
        meta_model = GradientBoostingRegressor(**best_params, random_state=42)
        meta_model.fit(meta_features_train, Y_train[:min_len])
        
        # Make final predictions
        final_prediction = meta_model.predict(meta_features_test)
        final_prediction = np.clip(final_prediction, clip_min, clip_max)

        return final_prediction
    
    except Exception as e:
        logger.error(f"GBDT error: {e}")
        return np.zeros(len(X_test))


def ensemble_pipeline(logger, models_results, X_train, X_test, Y_train, Y_test, target_scaler):
    """
    Main function that runs all ensemble methods and compares results.
    
    Flow:
    1. Validates model availability and target scaler functionality
    2. Executes all three ensemble methods sequentially
    3. Performs scale verification to ensure predictions are reasonable
    4. Calculates comprehensive metrics (MSE, RMSE, R², MAE) for each method
    5. Compares methods and identifies the best performer
    6. Returns complete results with performance rankings
    """
    
    logger.info(f"\n{'-'*30}\nInitializing ensemble pipeline\n{'-'*30}")
    logger.info("\nVerifying model results for ensemble:")
    
    for model_name, result in models_results.items():
        if 'best_model' in result and result['best_model'] is not None:
            logger.info(f"  {model_name}: Model available, Best RMSE: {np.min(result['best_rmse_scores']):.6f}")
        else:
            logger.info(f"  {model_name}: Model NOT available")
    
    logger.info("\nValidating target scaler:")
    
    y_sample = Y_test.iloc[:3].values.reshape(-1, 1) if hasattr(Y_test, 'iloc') else Y_test[:3].reshape(-1, 1)
    y_scaled = target_scaler.transform(y_sample)
    y_restored = target_scaler.inverse_transform(y_scaled)
    
    logger.info(f"  Original Y values: {y_sample.flatten()}")
    logger.info(f"  Scaled Y values: {y_scaled.flatten()}")
    logger.info(f"  Restored Y values: {y_restored.flatten()}")
    
    def linearly_weighted_wrapper(results, x_test):
        logger.info("Running linearly weighted ensemble...")
        return linearly_weighted_ensemble(results, x_test, target_scaler, logger)
    
    def equal_weighted_wrapper(results, x_test):
        logger.info("Running equal weighted ensemble...")
        return equal_weighted_ensemble(results, x_test, target_scaler, logger)
    
    def gbdt_wrapper(results, x_test):
        logger.info("Running GBDT ensemble...")
        return gbdt_ensemble(results, X_train, x_test, Y_train, target_scaler, logger)
    
    ensemble_methods = {
        'linearly_weighted': linearly_weighted_wrapper,
        'equal_weighted': equal_weighted_wrapper,
        'gbdt': gbdt_wrapper
    }
    
    results = {}
    
    for method_name, method_func in ensemble_methods.items():
        try:
            logger.info(f"\n{'-'*20} Running {method_name} ensemble {'-'*20}")
            
            start_time = time.time()
            
            final_prediction = method_func(models_results, X_test)
            end_time = time.time()

            # Scale comparison
            y_min, y_max = np.min(Y_test), np.max(Y_test)
            pred_min, pred_max = np.min(final_prediction), np.max(final_prediction)
            y_mean, y_std = np.mean(Y_test), np.std(Y_test)
            pred_mean, pred_std = np.mean(final_prediction), np.std(final_prediction)
            
            logger.info(f"\nScale Comparison for {method_name}:")
            logger.info(f"  Y_test range: [{y_min:.4f}, {y_max:.4f}], mean: {y_mean:.4f}, std: {y_std:.4f}")
            logger.info(f"  Prediction range: [{pred_min:.4f}, {pred_max:.4f}], mean: {pred_mean:.4f}, std: {pred_std:.4f}")
            logger.info(f"  Time taken: {end_time - start_time:.2f} seconds")
            
            # Align Y_test with predictions
            Y_test_aligned = Y_test
            if isinstance(Y_test, pd.Series) or isinstance(Y_test, pd.DataFrame):
                Y_test_aligned = Y_test.iloc[:len(final_prediction)]
            else:
                Y_test_aligned = Y_test[:len(final_prediction)]
            
            logger.info(f"  Length of Y_test: {len(Y_test)}, Length of predictions: {len(final_prediction)}")
            logger.info(f"  Using aligned Y_test with length: {len(Y_test_aligned)}")
            
            # Calculate metrics
            mse = mean_squared_error(Y_test_aligned, final_prediction)
            rmse = np.sqrt(mse)
            r2 = r2_score(Y_test_aligned, final_prediction)
            mae = np.mean(np.abs(Y_test_aligned - final_prediction))

            results[method_name] = {
                'prediction': final_prediction,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae
            }

            logger.info(f"\nDetailed results for {method_name}:")
            logger.info(f"  MSE: {mse:.6f}")
            logger.info(f"  RMSE: {rmse:.6f}")
            logger.info(f"  R²: {r2:.6f}")
            logger.info(f"  MAE: {mae:.6f}")

        except Exception as e:
            logger.error(f"Error in {method_name} ensemble: {e}")
            traceback.print_exc()
            results[method_name] = {
                'prediction': np.zeros_like(Y_test),
                'mse': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'mae': float('inf')
            }
    
    # Filter valid results
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
       
        comparison_df = comparison_df.sort_values('RMSE')
        logger.info(f"\n{comparison_df}")

        best_method = min(valid_results.items(), key=lambda x: x[1]['rmse'])[0]
        logger.info(f"\nBest Ensemble Method: {best_method}")
        logger.info(f"Performance of {best_method}:")
        logger.info(f"  MSE: {valid_results[best_method]['mse']:.6f}")
        logger.info(f"  RMSE: {valid_results[best_method]['rmse']:.6f}")
        logger.info(f"  R2: {valid_results[best_method]['r2']:.6f}")
        verify_prediction_scale(logger, Y_test, valid_results[best_method]['prediction'], 
                               f"Best ensemble method ({best_method})", tolerance=0.2)
    else:
        logger.warning("All ensemble methods failed")

    return results