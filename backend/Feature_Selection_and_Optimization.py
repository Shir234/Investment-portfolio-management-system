# Feature_Selection_and_Optimization.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""
Feature selection code!
------------------------
This set of functions helps determine which features are most important for predicting the target variable 
(Sharpe transaction values) and selects optimal feature subsets to improve model performance.
"""

def select_features_with_pca(X_train, n_components=None, variance_threshold=0.90):
    """
    Select features using Principal Component Analysis (PCA).
    
    Parameters:
    -----------
    X_train : DataFrame, Training features
    n_components : int or None, Number of components to keep (if None, use variance_threshold)
    variance_threshold : float, Minimum cumulative explained variance to retain (used if n_components is None)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'pca': Fitted PCA object
        - 'transformed_data': PCA-transformed training data
        - 'components_selected': Number of components selected
        - 'explained_variance_ratio': Explained variance ratio for each component
        - 'cumulative_variance_ratio': Cumulative explained variance ratio
    """
    
    # Scale the features (PCA is sensitive to feature scaling)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # If n_components is None, fit PCA with all components
    if n_components is None:
        # Initially fit with all possible components
        pca_all = PCA()
        pca_all.fit(X_train_scaled)
        
        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)
        
        # Find number of components needed to reach the threshold
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"Selected {n_components} components to explain {variance_threshold*100:.1f}% of variance")
    
    # Fit PCA with the determined number of components
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Create result dictionary
    result = {
        'pca': pca,
        'transformed_data': X_train_pca,
        'components_selected': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
        'scaler': scaler
    }
    
    return result

def plot_pca_explained_variance(pca_result):
    """
    Plot the explained variance ratio and cumulative explained variance for PCA.
    
    Parameters:
    -----------
    pca_result : dict, Output from select_features_with_pca function
    """
    explained_variance = pca_result['explained_variance_ratio']
    cumulative_variance = pca_result['cumulative_variance_ratio']
    
    plt.figure(figsize=(12, 5))
    
    # Plot explained variance ratio
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.xticks(range(1, len(explained_variance) + 1))
    
    # Plot cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.xticks(range(1, len(cumulative_variance) + 1))
    
    plt.tight_layout()
    plt.savefig('pca_explained_variance.png')
    plt.close()

def plot_pca_component_contributions(pca_result, X_train, top_n_features=10):
    """
    Plot the contribution of original features to principal components.
    
    Parameters:
    -----------
    pca_result : dict, Output from select_features_with_pca function
    X_train : DataFrame, Original training features to get feature names
    top_n_features : int, Number of top contributing features to display
    """
    pca = pca_result['pca']
    components = pca.components_
    feature_names = X_train.columns
    
    plt.figure(figsize=(15, 10))
    
    # Plot for top components
    num_components_to_plot = min(3, pca_result['components_selected'])
    
    for i in range(num_components_to_plot):
        plt.subplot(num_components_to_plot, 1, i+1)
        
        # Get absolute component values for ranking
        abs_components = np.abs(components[i])
        
        # Get indices of top contributing features
        indices = np.argsort(abs_components)[-top_n_features:]
        
        # Create horizontal bar chart
        plt.barh(range(top_n_features), abs_components[indices], color='skyblue')
        plt.yticks(range(top_n_features), [feature_names[j] for j in indices])
        plt.xlabel('Absolute Contribution')
        plt.title(f'Top {top_n_features} Features Contributing to Principal Component {i+1}')
        plt.tight_layout()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle('Feature Contributions to Top Principal Components', fontsize=16)
    plt.savefig('pca_feature_contributions.png')
    plt.close()

def analyze_feature_importance(X_train, Y_train, model_type='xgboost'):
    """
    Analyze feature importance to identify which features contribute most to prediction accuracy.
    Parameters:
    -----------
    X_train : DataFrame, Training features
    Y_train : Series, Target variable
    model_type : str, Type of model to use for feature importance ('xgboost', 'randomforest', or 'correlation')
    
    Returns:
    feature_importance_df : DataFrame, DataFrame with features sorted by importance
    """

    # Scale the features
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    feature_importance = None
    
    """
    Calculates the correlation between each feature and the target variable using corrwith()
    Takes the absolute value of correlations with abs()
    Sorts values in descending order
    Creates a DataFrame with feature names and their correlation values
    Higher absolute correlation indicates stronger relationship (linear) with the target
    """
    if model_type == 'correlation':
        # Simple correlation analysis
        correlation = X_train.corrwith(Y_train).abs().sort_values(ascending=False)
        feature_importance = pd.DataFrame({
            'Feature': correlation.index,
            'Importance': correlation.values
        })
        """
    Creates an XGBoost regressor with 100 trees and a learning rate of 0.05
    Fits the model to the scaled training data
    Extracts the built-in feature_importances_ attribute
    XGBoost calculates importance based on how much each feature improves the model's performance (reduction in loss function)
    Creates a DataFrame with feature names and importance scores
    """
    elif model_type == 'xgboost':
        # XGBoost feature importance
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        model.fit(X_train_scaled, Y_train)
        
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        })
        """
    Creates a Random Forest regressor with 100 trees
    Fits the model to the scaled training data
    Extracts the built-in feature_importances_ attribute
    Random Forest calculates importance based on how much each feature reduces impurity (variance) across all trees
    Creates a DataFrame with feature names and importance scores
    """
    elif model_type == 'randomforest':
        # Random Forest feature importance
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, Y_train)
        
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        })
    
    # Sort by importance in descending order
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    return feature_importance

def select_best_features(X_train, Y_train, threshold=0.8, method='importance'):
    """
    Select the most important features based on cumulative importance, using RFECV, lasso, or PCA.
    Parameters:
    -----------
    X_train : DataFrame, Training features
    Y_train : Series, Target variable
    threshold : float -> Cumulative importance threshold for selection (if method='importance')
                     -> Explained variance threshold if method='pca'
    method : str, Feature selection method ('importance', 'rfecv', 'lasso', or 'pca')
    
    Returns:
    selected_features : list, List of selected feature names if not PCA
                        OR dict with PCA results if method='pca'
    """

    if method == 'pca':
        # PCA-based feature selection
        pca_results = select_features_with_pca(X_train, variance_threshold=threshold)
        selected_features = pca_results  # Return the PCA results dictionary
        # Plot PCA explained variance
        # plot_pca_explained_variance(pca_results)
        # # Plot feature contributions to components
        # plot_pca_component_contributions(pca_results, X_train, top_n_features=10)

    return selected_features

def evaluate_feature_sets(X_train, Y_train):
    """
    Compare performance of different feature selection methods using multiple model types.
    With emphasis on PCA for maximizing prediction accuracy.
    
    Parameters:
    -----------
    X_train : DataFrame, Training features
    Y_train : Series, Target variable
        
    Returns:
    --------
    Dictionary containing detailed results and best feature information
    """
    
    # Define feature selection methods
    methods = {
        #'All Features': X_train.columns.tolist(),
        'PCA 85%': select_best_features(X_train, Y_train, threshold=0.85, method='pca'),
        'PCA 90%': select_best_features(X_train, Y_train, threshold=0.90, method='pca'),
        'PCA 95%': select_best_features(X_train, Y_train, threshold=0.95, method='pca'),
        'PCA 97%': select_best_features(X_train, Y_train, threshold=0.97, method='pca')
    }

    # Define different model types to evaluate feature sets
    models = {
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear': LassoCV(cv=3, random_state=42),  # Using LassoCV as a linear model
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)  # Added SVR with default parameters
    }
    
    # Create time series cross-validation for evaluation
    tscv = TimeSeriesSplit(n_splits=5)
    
    results = []

    # Evaluate each feature set with each model type
    for method_name, features in methods.items():
        if method_name == 'All Features':
            print(f"Evaluating feature set: {method_name} with {len(features)} features")
        else:
            print(f"Evaluating feature set: {method_name}")
        

        # Handle PCA methods
        if 'PCA' in method_name:
            print(f"Using PCA with {features['components_selected']} components")
            pca_obj = features['pca']
            scaler = features['scaler']
            
            # Use the number of components for reporting
            features_count = features['components_selected']
            
            # Test with each model type
            for model_name, model in models.items():
                try:
                    # Create and train the model
                    print(f"  Testing with {model_name}...")
                    
                    # Scale per fold to avoid data leakage
                    cv_scores = []
                    
                    for train_idx, val_idx in tscv.split(X_train):
                        # Get fold data
                        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_fold_train, y_fold_val = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
                        
                        # Scale using only training fold
                        scaler = features['scaler']
                        pca_obj = features['pca']
                        X_fold_train_scaled = scaler.transform(X_fold_train)
                        X_fold_val_scaled = scaler.transform(X_fold_val)
                        X_fold_train_pca = pca_obj.transform(X_fold_train_scaled)
                        X_fold_val_pca = pca_obj.transform(X_fold_val_scaled)
                        
                        # Train and evaluate
                        model.fit(X_fold_train_pca, y_fold_train)
                        y_fold_pred = model.predict(X_fold_val_pca)
                        
                        # Score
                        mse = mean_squared_error(y_fold_val, y_fold_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_fold_val, y_fold_pred)
                        
                        cv_scores.append({
                            'MSE': mse,
                            'RMSE': rmse,
                            'R2': r2
                        })
                    
                    # Average cross-validation scores
                    avg_mse = np.mean([score['MSE'] for score in cv_scores])
                    avg_rmse = np.mean([score['RMSE'] for score in cv_scores])
                    avg_r2 = np.mean([score['R2'] for score in cv_scores])
                    
                    # Store results
                    results.append({
                        'Feature_Method': method_name,
                        'Model': model_name,
                        'Num_Features': features_count,
                        'Features': f"PCA Components: {features_count}",
                        'MSE': avg_mse,
                        'RMSE': avg_rmse,
                        'R2': avg_r2
                    })
                    
                except Exception as e:
                    print(f"  Error with {model_name}: {e}")
                    # Add error entry
                    results.append({
                        'Feature_Method': method_name,
                        'Model': model_name,
                        'Num_Features': features_count,
                        'Features': f"PCA Components: {features_count}",
                        'MSE': float('inf'),
                        'RMSE': float('inf'),
                        'R2': float('-inf'),
                        'Error': str(e)
                    })
    
    # Convert to DataFrame
    detailed_results = pd.DataFrame(results)

    # Calculate average performance across models for each feature selection method
    avg_results = detailed_results.groupby('Feature_Method').agg({
        'RMSE': 'mean',
        'MSE': 'mean', 
        'R2': 'mean',
        'Num_Features': 'first'
    }).reset_index()

    # Find best method based on average RMSE
    best_method = avg_results.loc[avg_results['RMSE'].idxmin(), 'Feature_Method']

    # Get best features based on the method
    # Get best features based on the method
    if 'PCA' in best_method:
        best_features = methods[best_method]  # This is the PCA results dictionary
    else:
        best_features = methods[best_method]  # This is a list of feature names

    print(f"\nBest feature selection method across all models: {best_method}")
    if best_method == 'PCA 95%':
        print(f"Number of PCA components: {best_features['components_selected']}")
    else:
        print(f"Number of features: {len(best_features)}")
    print(f"Average RMSE: {avg_results.loc[avg_results['Feature_Method'] == best_method, 'RMSE'].values[0]:.6f}")
    
    return {
        'detailed_results': detailed_results,
        'average_results': avg_results,
        'best_method': best_method,
        'best_features': best_features
    }

# New function to adapt X_test to the feature selection method used
def adapt_test_data_to_feature_method(X_test, best_method, best_features):
    """
    Adapts test data based on the feature selection method used.
    
    Parameters:
    -----------
    X_test : DataFrame, Test features
    best_method : str, The method used for feature selection
    best_features : list or dict, Selected features or PCA results
    
    Returns:
    --------
    X_test_selected : DataFrame or ndarray, Processed test data
    """
    
    if 'PCA' in best_method:
        scaler = best_features['scaler']
        pca = best_features['pca']
        X_test_scaled = scaler.transform(X_test)
        X_test_selected = pca.transform(X_test_scaled)
    else:
        X_test_selected = X_test[best_features]
    
    return X_test_selected

def validate_feature_consistency(X_train_original, X_train_selected, selected_features):
    """
    Validate that PCA feature selection is consistent
    """
    # Get the number of components
    n_components = selected_features['components_selected']
    
    # Check if it's a DataFrame and convert to numpy array if needed
    if isinstance(X_train_selected, pd.DataFrame):
        X_selected_array = X_train_selected.values
    elif isinstance(X_train_selected, np.ndarray):
        X_selected_array = X_train_selected
    else:
        raise TypeError(f"Unexpected type for PCA-transformed data: {type(X_train_selected)}")
    
    # Validate dimensions
    if X_selected_array.shape[1] != n_components:
        raise ValueError(f"PCA-transformed data has {X_selected_array.shape[1]} columns but {n_components} components were selected")
    
    print(f"Feature consistency validation passed: {n_components} PCA components used")


