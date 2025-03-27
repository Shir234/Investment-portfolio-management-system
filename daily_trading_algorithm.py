import pandas as pd
import os
from datetime import datetime

def merge_ticker_data(directory_path, date_folder):
    """
    Merge prediction results from multiple tickers into a single DataFrame
    """

    all_tickers_data = []

    # Ensure the base directory exists
    if not os.path.exists(directory_path):
        print(f"Base directory '{directory_path}' does not exist.")
        return pd.DataFrame()   # return empty DataFrame 
    
    # Determine date directory
    if date_folder:
        date_path = os.path.join(directory_path, date_folder)
        if not os.path.exists(date_path):
            print(f"Date folser '{date_folder}' does not exist.")
            return pd.DataFrame()   # return empty DataFrame 
        
        try:
            files = os.listdir(date_path)
        except Exception as e:
            print(f"Error reading ")


    for file in os.listdir(date_folder):
        if file.endswith('__ensamble_prediction_results.csv'):
            ticker = file.split('_')[0]
            file_path = os.path.join(date_folder, file)

            # Read CSV
            ticker_data = pd.read_csv(file_path)

            # Add ticker column 
            if 'ticker' not in ticker_data.columns:
                ticker_data['ticker'] = ticker

            all_tickers_data.append(ticker_data)
    
    # Concatenate all DataFrames 
    merged_data = pd.concat(all_tickers_data, ignore_index=False)

    # Sort by date
    if 'Date' in merged_data.columns:
        merged_data['Date'] = pd.to_datetime(merged_data['Date'])
        merged_data.sort_values('Date', inplace=True)

    return merged_data


data_folder = '20250327'
merged_data = merge_ticker_data('results', data_folder)
merged_data.to_csv('Merged_data.csv')

