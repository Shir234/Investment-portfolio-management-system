# ticker_combiner.py
import pandas as pd
import os
from datetime import datetime

def merge_ticker_data(base_directory, date_folder, validate=True, sort_by_date=True):
    """
    Merge prediction results from multiple tickers in a specific date folder into a single DataFrame
    with validation to ensure data integrity, and optional sorting by date..
    
    Parameters:
    - base_directory (str): Path to the base results directory
    - date_folder (str): Name of the specific date subfolder (e.g., '20250327')
    - validate (bool): Whether to perform validation checks on the merged data
    - sort_by_date (bool): Whether to sort the merged data by date

    Returns:
    - pd.DataFrame: Merged data from all ticker files in the specified date folder
    """
    all_tickers_data = []
    validation_stats = {}
    total_rows_original = 0
    
    # Build the complete path to the date folder
    date_path = os.path.join(base_directory, date_folder)
    
    # Check if the path exists
    if not os.path.exists(date_path):
        print(f"Date folder path '{date_path}' does not exist.")
        return pd.DataFrame()
    
    try:
        # List all files in the date folder
        files = os.listdir(date_path)
        print(f"Found {len(files)} files in {date_path}")
        
        csv_files = [f for f in files if f.endswith('_ensemble_prediction_results.csv')]
        if not csv_files:
            print(f"No matching CSV files found in {date_path}")
            return pd.DataFrame()
            
        print(f"Found {len(csv_files)} matching CSV files")
        
        # First pass: collect schema information to ensure consistency
        sample_frames = {}
        for file in csv_files:
            ticker = file.split('_')[0]
            file_path = os.path.join(date_path, file)
            
            try:
                # Just read the first few rows to check schema
                sample_df = pd.read_csv(file_path, nrows=5)
                sample_frames[ticker] = {
                    'columns': list(sample_df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in sample_df.dtypes.items()}
                }
            except Exception as e:
                print(f"Error sampling schema from {file_path}: {e}")
        
        # Check for schema consistency
        all_columns = set()
        for ticker, info in sample_frames.items():
            all_columns.update(info['columns'])
        
        print(f"Combined schema has {len(all_columns)} columns: {sorted(all_columns)}")
            
        # Process each CSV file
        for file in csv_files:
            # Extract ticker from filename
            ticker = file.split('_')[0]
            
            # Build complete path to the file
            file_path = os.path.join(date_path, file)
            
            try:
                # Read the CSV file
                print(f"Reading {file_path}")
                ticker_data = pd.read_csv(file_path)
                original_row_count = len(ticker_data)
                total_rows_original += original_row_count
                
                # Record pre-merge stats for validation
                validation_stats[ticker] = {
                    'original_rows': original_row_count,
                    'original_nulls': ticker_data.isnull().sum().sum(),
                    'columns': list(ticker_data.columns)
                }
                
                # Add ticker column if not present
                if 'Ticker' not in ticker_data.columns:
                    ticker_data['Ticker'] = ticker
                
                # Add to the collection
                all_tickers_data.append(ticker_data)
                print(f"Added {len(ticker_data)} rows for ticker {ticker}")
                
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    except Exception as e:
        print(f"Error accessing directory {date_path}: {e}")
        return pd.DataFrame()
    
    # Check if we found any data
    if not all_tickers_data:
        print(f"No valid ticker data collected")
        return pd.DataFrame()
    
    # Merge all the data
    print(f"Merging data from {len(all_tickers_data)} ticker files")
    merged_data = pd.concat(all_tickers_data, ignore_index=True)
    
    # Sort the data by date if requested
    if sort_by_date and 'Date' in merged_data.columns:
        print("Sorting merged data by Date")
        # Convert Date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(merged_data['Date']):
            merged_data['Date'] = pd.to_datetime(merged_data['Date'])
        # Sort by date
        merged_data = merged_data.sort_values('Date')
        print(f"Data sorted - first date: {merged_data['Date'].min()}, last date: {merged_data['Date'].max()}")
    
    # Perform validation checks
    if validate:
        merged_row_count = len(merged_data)
        merged_null_count = merged_data.isnull().sum().sum()
        
        print("\n--- VALIDATION REPORT ---")
        print(f"Total original rows: {total_rows_original}")
        print(f"Merged dataframe rows: {merged_row_count}")
        
        if merged_row_count != total_rows_original:
            print(f"WARNING: Row count mismatch! {merged_row_count} != {total_rows_original}")
            print(f"Difference: {abs(merged_row_count - total_rows_original)} rows")
        else:
            print("Success: Row counts match perfectly")  # Replaced \u2713 with plain text
            
        print(f"Total nulls in merged data: {merged_null_count}")
        
        null_by_column = merged_data.isnull().sum()
        columns_with_nulls = null_by_column[null_by_column > 0]
        if not columns_with_nulls.empty:
            print("\nColumns with null values:")
            for col, count in columns_with_nulls.items():
                print(f"  - {col}: {count} nulls ({count/len(merged_data)*100:.2f}%)")
        else:
            print("Success: No null values found in any column")  # Replaced \u2713 with plain text
            
        duplicate_count = merged_data.duplicated().sum()
        if duplicate_count > 0:
            print(f"\nWARNING: Found {duplicate_count} duplicate rows")
        else:
            print("Success: No duplicate rows found")  # Replaced \u2713 with plain text
            
        print("------------------------\n")
    
    return merged_data

# Example usage:
selected_date = "20250426"
merged_results = merge_ticker_data("results", selected_date, validate=True, sort_by_date=True)
# If validation passes, save the merged data
merged_results.to_csv(f"{selected_date}_all_tickers_results.csv", index=False)