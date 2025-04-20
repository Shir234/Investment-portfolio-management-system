import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

"""
Feature selection code!
------------------------
This set of functions helps determine which features are most important for predicting the target variable 
(Sharpe transaction values) and selects optimal feature subsets to improve model performance.
"""


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
    scaler = StandardScaler()
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
    Select the most important features based on cumulative importance or using RFECV.
    Parameters:
    -----------
    X_train : DataFrame, Training features
    Y_train : Series, Target variable
    threshold : float -> Cumulative importance threshold for selection (if method='importance')
    method : str, Feature selection method ('importance', 'rfecv', or 'lasso')
    
    Returns:
    selected_features : list, List of selected feature names
    """

    if method == 'importance':
        """
        First calls analyze_feature_importance to get features ranked by importance
        Calculates cumulative importance by:
            - Taking the cumulative sum of importance values
            - Dividing by the total sum to get a percentage
        Selects features whose cumulative importance is below the threshold
        Ensures at least one feature is selected (the most important one)
        """
        # Get feature importance
        importance_df = analyze_feature_importance(X_train, Y_train, model_type='xgboost')
        # Calculate cumulative importance
        importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum() / importance_df['Importance'].sum()
        # Select features based on cumulative importance threshold
        selected_features = importance_df[importance_df['Cumulative_Importance'] <= threshold]['Feature'].tolist()
        
        # Make sure we include at least one feature
        if not selected_features:
            selected_features = [importance_df.iloc[0]['Feature']]
            
    elif method == 'rfecv':
        """
        # Recursive Feature Elimination with Cross-Validation 
        First scales the features using StandardScaler
        Creates a Random Forest model (could be any model) for evaluation
        Creates an RFECV selector with:
            - 5-fold cross-validation
            - Using negative mean squared error as scoring metric
            - Removing one feature at a time (step=1)
        the process:
            - Starts with all features
            - Trains the model and evaluates performance with cross-validation
            - Removes the least important feature
            - Repeats until it reaches min_features_to_select=1
            - Selects the subset of features that gave the best cross-validation score
        Gets the selected features using the support_ attribute, which is a boolean array indicating which features were selected
        """
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        # Create a model for RFECV
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        # Create the RFECV object
        selector = RFECV(estimator=estimator, step=1, cv=5, scoring='neg_mean_squared_error', min_features_to_select=1)
        # Fit the selector
        selector.fit(X_train_scaled, Y_train)
        # Get selected features
        selected_features = X_train.columns[selector.support_].tolist()
        
    elif method == 'lasso':        
        """
        First scales the features using StandardScaler
        Creates a LassoCV model with 5-fold cross-validation
        LassoCV uses L1 regularization which can shrink coefficients to exactly zero
        Fits the model to the scaled training data
        Creates a DataFrame with features and their absolute coefficient values
        Sorts features by absolute coefficient value
        Selects only features with non-zero coefficients
        The key property of Lasso is that it performs feature selection automatically by forcing some coefficients to be exactly zero
        """
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        # Create and fit LassoCV model
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X_train_scaled, Y_train)
        # Get selected features
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': np.abs(lasso.coef_)
        })
        # Sort by absolute coefficient value
        feature_importance = feature_importance.sort_values('Coefficient', ascending=False)
        # Select features with non-zero coefficients
        selected_features = feature_importance[feature_importance['Coefficient'] > 0]['Feature'].tolist()
    
    return selected_features

def evaluate_feature_sets(X_train, Y_train, X_test, Y_test):
    """
    Compare performance of different feature selection methods using multiple model types.
    Parameters:
    -----------
    X_train : DataFrame, Training features
    Y_train : Series, Target variable
    X_test : DataFrame, Test features
    Y_test : Series, Test target
        
    Returns:
    Dictionary containing detailed results and best feature information
    """
    
    # Define feature selection methods
    methods = {
        'All Features': X_train.columns.tolist(),
        'XGBoost Top 80%': select_best_features(X_train, Y_train, threshold=0.8, method='importance'),
        'RFECV': select_best_features(X_train, Y_train, method='rfecv'),
        'Lasso': select_best_features(X_train, Y_train, method='lasso')
    }

    # Define different model types to evaluate feature sets
    models = {
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear': LassoCV(cv=3, random_state=42)  # Using LassoCV as a linear model
    }
    
    results = []

    # Evaluate each feature set with each model type
    for method_name, features in methods.items():
        print(f"Evaluating feature set: {method_name} with {len(features)} features")

        # Subset the data
        X_train_subset = X_train[features]
        X_test_subset = X_test[features]
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_test_scaled = scaler.transform(X_test_subset)

        # Test with each model type
        for model_name, model in models.items():
            try:
                # Create and train the model
                print(f"  Testing with {model_name}...")
                model.fit(X_train_scaled, Y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(Y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(Y_test, y_pred)
                
                # Store results
                results.append({
                    'Feature_Method': method_name,
                    'Model': model_name,
                    'Num_Features': len(features),
                    'Features': features,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2
                })
                
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
                # Add error entry
                results.append({
                    'Feature_Method': method_name,
                    'Model': model_name,
                    'Num_Features': len(features),
                    'Features': features,
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
    best_features = methods[best_method]

    print(f"\nBest feature selection method across all models: {best_method}")
    print(f"Number of features: {len(best_features)}")
    print(f"Average RMSE: {avg_results.loc[avg_results['Feature_Method'] == best_method, 'RMSE'].values[0]:.6f}")
    
    return {
        'detailed_results': detailed_results,
        'average_results': avg_results,
        'best_method': best_method,
        'best_features': best_features
    }


def validate_feature_consistency(X_train_original, X_train_selected, selected_features):
    """
    Validate that feature selection is consistent
    """
    
    # Check all selected features exist in original dataset
    for feature in selected_features:
        if feature not in X_train_original.columns:
            raise ValueError(f"Selected feature {feature} not in original dataset")
    
    # Check dimensions are as expected
    if X_train_selected.shape[1] != len(selected_features):
        raise ValueError(f"Selected data has {X_train_selected.shape[1]} columns but {len(selected_features)} features were selected")
    
    # Check column order matches
    if not all(X_train_selected.columns == selected_features):
        print("Warning: Column order in selected data doesn't match feature list order")
    
    print(f"Feature consistency validation passed: {len(selected_features)} features used")
