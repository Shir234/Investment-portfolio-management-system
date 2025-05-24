# Ensembles.py
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from Models_Creation_and_Training import create_rolling_window_data
import optuna
from scipy.stats import pearsonr
import Models_Creation_and_Training as Models_Creation_and_Training
from Logging_and_Validation import log_data_stats, verify_prediction_scale
from sklearn.metrics import mean_squared_error
import numpy as np

# ===============================================================================
# Ensembles
# ===============================================================================
def prepare_lstm_data(X, time_steps=3, features=None):
    if hasattr(X, 'values'):
        X = X.values
    if len(X) < time_steps + 1:
        raise ValueError(f"Not enough samples ({len(X)}) for time_steps ({time_steps})")
    X_rolled = []
    for i in range(len(X) - time_steps):
        X_rolled.append(X[i:i + time_steps])
    return np.array(X_rolled)

# Linearly Weighted Ensemble


# Linearly Weighted Ensemble
def linearly_weighted_ensemble(models_results, X_test, target_scaler, feature_scaler, logger=None):
    if not models_results:
        raise ValueError("Empty models_results dictionary")
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    if len(X_test_array) < 3 and any('LSTM' in model_name for model_name in models_results):
        raise ValueError("X_test too small for LSTM (needs >= 3 samples)")

    # Debug models_results
    logger.info(f"Models results keys: {list(models_results.keys())}")
    for name, res in models_results.items():
        logger.info(f"{name} keys: {list(res.keys())}")

    weights = []
    model_predictions = []
    pred_lengths = []
    target_len = len(X_test)

    # Select top 3 models by retrained RMSE
    top_models = sorted(models_results.items(), key=lambda x: x[1].get('retrained_rmse', float('inf')))[:3]
    models_results = dict(top_models)
    logger.info(f"Selected top models: {list(models_results.keys())}")

    for model_name, result in models_results.items():
        if 'retrained_model' not in result or result['retrained_model'] is None:
            logger.warning(f"No retrained model for {model_name}, using best_model")
            if 'best_model' not in result or result['best_model'] is None:
                logger.warning(f"No best_model for {model_name}, skipping")
                continue
            result['retrained_model'] = result['best_model']
            result['retrained_rmse'] = result['best_rmse_scores'][best_fold_idx]
        best_pred = result.get('best_model_prediction')
        y_val = result.get('Y_val_best')
        if best_pred is None or y_val is None:
            logger.warning(f"Model {model_name} missing predictions or validation data")
            continue
        min_length = min(len(best_pred), len(y_val))
        if min_length == 0:
            logger.warning(f"Empty predictions for {model_name}")
            continue
        best_fold_idx = np.argmin(result['best_rmse_scores'])
        r2_value = result['best_r2_scores'][best_fold_idx]
        if r2_value <= -0.5:  # Relaxed from -0.1
            logger.warning(f"Skipping {model_name} due to low R²: {r2_value}")
            continue
        try:
            if model_name.strip() == 'LSTM':
                X_test_lstm = prepare_lstm_data(X_test_array, time_steps=3)
                if len(X_test_lstm) == 0:
                    logger.warning(f"Not enough data for LSTM predictions")
                    continue
                model_pred = result['retrained_model'].predict(X_test_lstm, verbose=0).flatten()
                model_pred = np.pad(model_pred, (0, max(0, target_len - len(model_pred))), mode='constant', constant_values=np.mean(model_pred) if len(model_pred) > 0 else 0)
            else:
                model_pred = result['retrained_model'].predict(X_test_array).flatten()
                model_pred = model_pred[:target_len]
            model_pred = target_scaler.inverse_transform(model_pred.reshape(-1, 1)).flatten()
            model_pred = np.clip(model_pred, -3.6815, 4.6223)
            model_predictions.append(model_pred)
            weights.append(np.exp(-result.get('retrained_rmse', result['best_rmse_scores'][best_fold_idx])))
            pred_lengths.append(len(model_pred))
            logger.info(f"Model {model_name} - R²: {r2_value:.4f}, RMSE: {result['best_rmse_scores'][best_fold_idx]:.4f}, Retrained RMSE: {result.get('retrained_rmse', 'N/A')}, Prediction shape: {model_pred.shape}")
        except Exception as e:
            logger.error(f"Error generating predictions for {model_name}: {e}")
            continue
    if not model_predictions:
        raise ValueError("No valid model predictions available")
    weights = np.array(weights)
    min_weight = 0.05 * np.max(weights)
    weights = np.where(weights < min_weight, 0, weights)
    if np.sum(weights) == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / np.sum(weights)
    logger.info(f"Model weights: {dict(zip(models_results.keys(), weights))}")
    logger.info(f"Prediction lengths: {pred_lengths}")
    final_prediction = np.zeros(target_len, dtype=np.float64)
    for pred, weight in zip(model_predictions, weights):
        final_prediction += weight * pred
    final_prediction = np.clip(final_prediction, -3.6815, 4.6223)
    return final_prediction


# Equal Weights Ensemble
def equal_weighted_ensemble(models_results, X_test, target_scaler, feature_scaler, logger=None):
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    model_predictions = []
    pred_lengths = []
    target_len = len(X_test)
    for model_name, result in models_results.items():
        if 'best_model' not in result or result['best_model'] is None:
            continue
        try:
            if model_name.strip() == 'LSTM':
                X_test_lstm = prepare_lstm_data(X_test_array, time_steps=3)
                if len(X_test_lstm) == 0:
                    if logger:
                        logger.warning(f"Not enough data for LSTM predictions")
                    continue
                model_pred = result['best_model'].predict(X_test_lstm, verbose=0).flatten()
                model_pred = np.pad(model_pred, (0, max(0, target_len - len(model_pred))), mode='constant', constant_values=np.mean(model_pred) if len(model_pred) > 0 else 0)
            else:
                model_pred = result['best_model'].predict(X_test_array).flatten()
                model_pred = model_pred[:target_len]
            model_pred = target_scaler.inverse_transform(model_pred.reshape(-1, 1)).flatten()
            model_pred = np.clip(model_pred, -3.6815, 4.6223)
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
    weight = 1.0 / len(model_predictions)
    final_prediction = np.zeros(target_len, dtype=np.float64)
    for pred in model_predictions:
        final_prediction += weight * pred
    final_prediction = np.clip(final_prediction, -3.6815, 4.6223)
    return final_prediction


# Gradient Boosting Decision Tree Ensemble

def gbdt_ensemble(results, X_train, X_test, Y_train, target_scaler, feature_scaler, logger):
    try:
        meta_features_train = []
        meta_features_test = []
        min_len = len(X_test)
        logger.info(f"GBDT: Target length for predictions: {min_len}")
        
        valid_models = []
        # Ensure results is a dictionary
        if not isinstance(results, dict):
            logger.error(f"Results is {type(results)}, expected a dictionary")
            return np.zeros(min_len)
        
        for model_name, result in results.items():
            if 'best_model' not in result or result['best_model'] is None:
                logger.warning(f"No valid model for {model_name}")
                continue
            if 'retrained_model' not in result or result['retrained_model'] is None:
                logger.warning(f"No retrained model for {model_name}, skipping")
                continue
            model = result['retrained_model']
            logger.info(f"Processing {model_name} predictions")
            
            if model_name.strip() == 'LSTM':
                time_steps = result.get('best_params', {}).get('time_steps', 3)
                if X_train.shape[0] < time_steps or X_test.shape[0] < time_steps:
                    logger.warning(f"Insufficient samples for LSTM time_steps={time_steps}")
                    continue
                X_train_rolled, y_train_rolled = create_rolling_window_data(X_train.values, Y_train.values, time_steps)
                X_test_rolled, _ = create_rolling_window_data(X_test.values, np.zeros(len(X_test)), time_steps)
                logger.info(f"LSTM: X_train_rolled shape: {X_train_rolled.shape}, X_test_rolled shape: {X_test_rolled.shape}")
                train_pred = model.predict(X_train_rolled, verbose=0).flatten()
                test_pred = model.predict(X_test_rolled, verbose=0).flatten()
                train_pred = train_pred[:min_len]
                test_pred = np.pad(test_pred, (0, max(0, min_len - len(test_pred))), mode='constant', constant_values=np.mean(test_pred) if len(test_pred) > 0 else 0)
            else:
                train_pred = model.predict(X_train).flatten()[:min_len]
                test_pred = model.predict(X_test).flatten()[:min_len]
            
            train_pred = target_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
            test_pred = target_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
            scaler = StandardScaler()
            train_pred = scaler.fit_transform(train_pred.reshape(-1, 1)).flatten()
            test_pred = scaler.transform(test_pred.reshape(-1, 1)).flatten()
            train_pred = np.clip(train_pred, -3.6815, 4.6223)
            test_pred = np.clip(test_pred, -3.6815, 4.6223)
            
            meta_features_train.append(train_pred)
            meta_features_test.append(test_pred)
            valid_models.append(model_name)
            logger.info(f"{model_name}: Final train_pred length: {len(train_pred)}, test_pred length: {len(test_pred)}")
        
        if not meta_features_train or not meta_features_test:
            logger.error("No valid meta-features generated")
            return np.zeros(min_len)
        
        meta_features_train = np.array(meta_features_train).T
        meta_features_test = np.array(meta_features_test).T
        corr_matrix = np.corrcoef(meta_features_train.T)
        logger.info(f"Meta-feature correlations:\n{corr_matrix}")
        
        to_keep = []
        for i in range(corr_matrix.shape[0]):
            if i == 0 or all(abs(corr_matrix[i, j]) < 0.95 for j in to_keep):
                to_keep.append(i)
        
        meta_features_train = meta_features_train[:, to_keep]
        meta_features_test = meta_features_test[:, to_keep]
        valid_models = [valid_models[i] for i in to_keep]
        logger.info(f"Kept models after correlation filter: {valid_models}")
        
        scaler = StandardScaler()
        meta_features_train = scaler.fit_transform(meta_features_train)
        meta_features_test = scaler.transform(meta_features_test)
        
        meta_train, meta_val, y_train_meta, y_val_meta = train_test_split(
            meta_features_train, Y_train[:min_len], test_size=0.1, random_state=42
        )
        
        def optimize_meta_model(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
            }
            meta_model = GradientBoostingRegressor(**params, random_state=42)
            meta_model.fit(meta_train, y_train_meta)
            pred = meta_model.predict(meta_val)
            return np.sqrt(mean_squared_error(y_val_meta, pred))
        
        study = optuna.create_study(direction='minimize')
        study.optimize(optimize_meta_model, n_trials=30)
        best_params = study.best_params
        logger.info(f"Optimized meta-model parameters: {best_params}")
        
        meta_model = GradientBoostingRegressor(**best_params, random_state=42)
        meta_model.fit(meta_features_train, Y_train[:min_len])
        
        final_prediction = meta_model.predict(meta_features_test)
        final_prediction = target_scaler.inverse_transform(final_prediction.reshape(-1, 1)).flatten()
        final_prediction = np.clip(final_prediction, -3.6815, 4.6223)
        return final_prediction
    except Exception as e:
        logger.error(f"GBDT error: {e}")
        return np.zeros(len(X_test))


# ===============================================================================
# Three Ensembles Pipeline
# ===============================================================================
def ensemble_pipeline(logger, models_results, X_train, X_test, Y_train, Y_test, target_scaler, feature_scaler):
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
        return linearly_weighted_ensemble(results, x_test, target_scaler, None, logger)
    def equal_weighted_wrapper(results, x_test):
        logger.info("Running equal weighted ensemble...")
        return equal_weighted_ensemble(results, x_test, target_scaler, None, logger)
    def gbdt_wrapper(results, x_test):
        logger.info("Running GBDT ensemble...")
        return gbdt_ensemble(results, X_train, x_test, Y_train, target_scaler, None, logger)
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
            y_min, y_max = np.min(Y_test), np.max(Y_test)
            pred_min, pred_max = np.min(final_prediction), np.max(final_prediction)
            y_mean, y_std = np.mean(Y_test), np.std(Y_test)
            pred_mean, pred_std = np.mean(final_prediction), np.std(final_prediction)
            logger.info(f"\nScale Comparison for {method_name}:")
            logger.info(f"  Y_test range: [{y_min:.4f}, {y_max:.4f}], mean: {y_mean:.4f}, std: {y_std:.4f}")
            logger.info(f"  Prediction range: [{pred_min:.4f}, {pred_max:.4f}], mean: {pred_mean:.4f}, std: {pred_std:.4f}")
            logger.info(f"  Time taken: {end_time - start_time:.2f} seconds")
            Y_test_aligned = Y_test
            if isinstance(Y_test, pd.Series) or isinstance(Y_test, pd.DataFrame):
                Y_test_aligned = Y_test.iloc[:len(final_prediction)]
            else:
                Y_test_aligned = Y_test[:len(final_prediction)]
            logger.info(f"  Length of Y_test: {len(Y_test)}, Length of predictions: {len(final_prediction)}")
            logger.info(f"  Using aligned Y_test with length: {len(Y_test_aligned)}")
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
            import traceback
            traceback.print_exc()
            results[method_name] = {
                'prediction': np.zeros_like(Y_test),
                'mse': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'mae': float('inf')
            }
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

