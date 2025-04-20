import numpy as np
import pandas as pd


def log_data_stats(logger, data, name, include_stats=True, log_shape=True, log_head=False, n_head=5):
    """
    Log information about a dataframe or numpy array.
    Parameters:
    -----------
    data : DataFrame or ndarray, data to log information about
    name : str, name of the data for logging
    include_stats : bool, whether to log descriptive statistics
    log_shape : bool, whether to log shape
    log_head : bool, whether to log first few rows
    n_head : int, number of rows to log if log_head=True
    """
    if data is None:
        logger.warning(f"{name} is None")
        print(f"{name} is None")
        return
    
    logger.info(f"\n--- {name} Information ---")

    print(f"\n--- {name} Information ---")
    
    if log_shape:
        if hasattr(data, 'shape'):
            logger.info(f"Shape: {data.shape}")
        
    if include_stats and hasattr(data, 'describe'):
        try:
            stats = data.describe()
            logger.info(f"\nSummary Statistics:\n{stats}")
            logger.info(f"\nData Types: {data.dtypes.value_counts()}")
            logger.info(f"Missing Values: {data.isna().sum().sum()}")
        except Exception as e:
            logger.error(f"Could not generate statistics: {e}")
    
    if log_head and hasattr(data, 'head'):
        try:
            logger.info(f"\nFirst {n_head} rows:\n{data.head(n_head)}")
        except Exception as e:
            logger.error(f"Could not show head: {e}")


def verify_lstm_shape(X, expected_time_steps=1):
    """
    Verify that data is properly shaped for LSTM input
    """
    if len(X.shape) != 3:
        raise ValueError(f"LSTM input should be 3D, but got shape {X.shape}")
    if X.shape[1] != expected_time_steps:
        raise ValueError(f"Expected {expected_time_steps} time steps, but got {X.shape[1]}")
    print(f"LSTM shape verification passed: {X.shape}")
    return X


def verify_prediction_scale(logger, original_y, predicted_y, name="", tolerance=0.1):
    """
    Verify that predictions are on the same scale as original data
    """
    orig_min, orig_max = np.min(original_y), np.max(original_y)
    pred_min, pred_max = np.min(predicted_y), np.max(predicted_y)
    
    logger.info(f"\n--- Scale Verification for {name} ---")
    logger.info(f"Original range: [{orig_min:.4f}, {orig_max:.4f}]")
    logger.info(f"Prediction range: [{pred_min:.4f}, {pred_max:.4f}]")
    
    # Check if ranges are roughly similar (allowing for prediction error)
    min_ratio = abs(orig_min - pred_min) / (abs(orig_min) + 1e-10)
    max_ratio = abs(orig_max - pred_max) / (abs(orig_max) + 1e-10)
    
    if min_ratio > tolerance or max_ratio > tolerance:
        logger.warning(f"Prediction scale may be inconsistent. Min ratio: {min_ratio:.4f}, Max ratio: {max_ratio:.4f}")
    else:
        logger.info("Prediction scale verification passed")
    
    return min_ratio, max_ratio


def validate_data_quality(data, name="dataset", check_missing=True, check_outliers=True, 
                         check_dtypes=True, check_duplicates=True, check_scaling=True):
    """
    Comprehensive data validation and quality check
    
    Parameters:
    -----------
    data : DataFrame or Series, data to validate
    name : str, name of the dataset for reporting
    check_missing : bool, whether to check for missing values
    check_outliers : bool, whether to check for outliers
    check_dtypes : bool, whether to check for inconsistent data types
    check_duplicates : bool, whether to check for duplicate rows/columns
    check_scaling : bool, whether to check for scaling issues
    
    Returns:
    --------
    dict : Dictionary with validation results
    """
    print(f"\n{'-'*20} Data Quality Check for {name} {'-'*20}")
    
    if data is None:
        print(f"ERROR: {name} is None")
        return {'status': 'error', 'message': f"{name} is None"}
    
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    
    report = {'status': 'passed', 'warnings': [], 'errors': []}
    
    # Basic shape and type
    print(f"Shape: {data.shape}")
    print(f"Data type: {type(data)}")
    
    # Check for missing values
    if check_missing:
        missing_count = data.isna().sum().sum()
        missing_pct = missing_count / (data.shape[0] * data.shape[1]) * 100
        
        print(f"Missing values: {missing_count} ({missing_pct:.2f}%)")
        
        if missing_count > 0:
            missing_cols = data.columns[data.isna().any()].tolist()
            print(f"Columns with missing values: {missing_cols}")
            report['warnings'].append(f"Missing values found in columns: {missing_cols}")
    
    # Check for duplicates
    if check_duplicates:
        dup_rows = data.duplicated().sum()
        dup_cols = data.T.duplicated().sum()
        
        print(f"Duplicate rows: {dup_rows}")
        print(f"Duplicate columns: {dup_cols}")
        
        if dup_rows > 0:
            report['warnings'].append(f"Found {dup_rows} duplicate rows")
        if dup_cols > 0:
            report['warnings'].append(f"Found {dup_cols} duplicate columns")
    
    # Check data types
    if check_dtypes:
        print(f"Data types:\n{data.dtypes.value_counts()}")
        
        # Check for mixed data types within columns
        mixed_cols = []
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if this column has mixed numeric and non-numeric values
                numeric_count = pd.to_numeric(data[col], errors='coerce').notna().sum()
                if 0 < numeric_count < len(data):
                    mixed_cols.append(col)
        
        if mixed_cols:
            print(f"Columns with mixed data types: {mixed_cols}")
            report['warnings'].append(f"Mixed data types in columns: {mixed_cols}")
    
    # Check for outliers
    if check_outliers and len(data) > 10:
        numeric_cols = data.select_dtypes(include=np.number).columns
        outlier_cols = []
        
        for col in numeric_cols:
            # Use IQR method to detect outliers
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_count = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_pct = outlier_count / len(data) * 100
            
            if outlier_pct > 5:
                outlier_cols.append((col, outlier_count, outlier_pct))
        
        if outlier_cols:
            print("Columns with significant outliers:")
            for col, count, pct in outlier_cols:
                print(f"  {col}: {count} outliers ({pct:.2f}%)")
            
            report['warnings'].append(f"Found {len(outlier_cols)} columns with significant outliers")
    
    # Check for scaling issues
    if check_scaling and len(data) > 10:
        numeric_cols = data.select_dtypes(include=np.number).columns
        scaling_issues = []
        
        for col in numeric_cols:
            col_min = data[col].min()
            col_max = data[col].max()
            col_range = col_max - col_min
            
            # Check for large ranges
            if col_range > 1000:
                scaling_issues.append((col, "large_range", col_range))
            
            # Check for very small ranges
            elif col_range < 0.01 and col_range > 0:
                scaling_issues.append((col, "small_range", col_range))
        
        if scaling_issues:
            print("Columns with potential scaling issues:")
            for col, issue_type, value in scaling_issues:
                print(f"  {col}: {issue_type} ({value})")
            
            report['warnings'].append(f"Found {len(scaling_issues)} columns with potential scaling issues")
    
    # Summary
    if report['errors']:
        report['status'] = 'error'
        print("\nErrors found during data validation!")
    elif report['warnings']:
        report['status'] = 'warning'
        print("\nWarnings found during data validation!")
    else:
        print("\nData validation passed without issues!")
    
    return report