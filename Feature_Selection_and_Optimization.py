import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV
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
    Compare performance of different feature selection methods.
    Parameters:
    -----------
    X_train : DataFrame, Training features
    Y_train : Series, Target variable
    X_test : DataFrame, Test features
    Y_test : Series, Test target
        
    Returns:
    results_df : DataFrame, Results comparing different feature sets
    """
    
    # Define feature selection methods
    methods = {
        'All Features': X_train.columns.tolist(),
        'XGBoost Top 80%': select_best_features(X_train, Y_train, threshold=0.8, method='importance'),
        'RFECV': select_best_features(X_train, Y_train, method='rfecv'),
        'Lasso': select_best_features(X_train, Y_train, method='lasso')
    }
    
    results = []
    
    # Evaluate each feature set
    for method_name, features in methods.items():
        # Subset the data
        X_train_subset = X_train[features]
        X_test_subset = X_test[features]
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_test_scaled = scaler.transform(X_test_subset)
        
        # Create and train an XGBoost regressor on the scaled training data
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        model.fit(X_train_scaled, Y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(Y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, y_pred)
        
        # Store results
        results.append({
            'Method': method_name,
            'Num_Features': len(features),
            'Features': features,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    """
    Returns a DataFrame with columns:
        - 'Method': Name of the feature selection method
        - 'Num_Features': Number of features selected
        - 'Features': List of the actual features selected
        - 'MSE': Mean Squared Error on test data
        - 'RMSE': Root Mean Squared Error on test data
        - 'R2': R² score on test data
    """
    return results_df


