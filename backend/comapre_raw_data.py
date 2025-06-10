# compare_raw_data,py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def compare_filling_methods(data_path):
    """
    Compare different methods for filling missing Sharpe ratio values.
    
    Parameters:
    - data_path (str): Path to the raw CSV data file.
    
    Returns:
    - DataFrame: Original data with both filling methods for comparison.
    """
    # Load the data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Make a copy for comparison
    df_original = df.copy()
    
    # Method 1: Forward-Fill & Backward-Fill (Original Method)
    df_ffill = df.copy()
    df_ffill['Daily_Sharpe_Ratio_FFILL'] = df_ffill['Daily_Sharpe_Ratio'].ffill().bfill()
    
    # Method 2: Median Fill (New Method)
    df_median = df.copy()
    median_value = df_median['Daily_Sharpe_Ratio'].dropna().median()
    df_median['Daily_Sharpe_Ratio_MEDIAN'] = df_median['Daily_Sharpe_Ratio'].fillna(median_value)
    
    # Combine results
    df_original['Daily_Sharpe_Ratio_FFILL'] = df_ffill['Daily_Sharpe_Ratio_FFILL']
    df_original['Daily_Sharpe_Ratio_MEDIAN'] = df_median['Daily_Sharpe_Ratio_MEDIAN']
    
    # Print statistics
    print("Statistics for filled values:")
    print(f"Original data null count: {df_original['Daily_Sharpe_Ratio'].isna().sum()}")
    print(f"FFILL method first value: {df_original['Daily_Sharpe_Ratio_FFILL'].iloc[0]}")
    print(f"MEDIAN method value: {median_value}")
    
    # Plot for visual comparison
    plt.figure(figsize=(15, 8))
    
    # Plot the first 60 days to focus on the initial filled values
    plt.subplot(2, 1, 1)
    plt.title('First 60 Days - Comparison of Filling Methods')
    plt.plot(df_original.index[:60], df_original['Daily_Sharpe_Ratio'].iloc[:60], 'o-', label='Original (with nulls)')
    plt.plot(df_original.index[:60], df_original['Daily_Sharpe_Ratio_FFILL'].iloc[:60], 'o-', label='Forward Fill')
    plt.plot(df_original.index[:60], df_original['Daily_Sharpe_Ratio_MEDIAN'].iloc[:60], 'o-', label='Median Fill')
    plt.axhline(y=median_value, color='r', linestyle='--', label=f'Median Value: {median_value:.2f}')
    plt.legend()
    plt.grid(True)
    
    # Plot the full time series
    plt.subplot(2, 1, 2)
    plt.title('Full Time Series - Comparison of Filling Methods')
    plt.plot(df_original.index, df_original['Daily_Sharpe_Ratio'], 'o-', alpha=0.5, label='Original (with nulls)')
    plt.plot(df_original.index, df_original['Daily_Sharpe_Ratio_FFILL'], label='Forward Fill')
    plt.plot(df_original.index, df_original['Daily_Sharpe_Ratio_MEDIAN'], label='Median Fill')
    plt.axhline(y=median_value, color='r', linestyle='--', label=f'Median Value: {median_value:.2f}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sharpe_ratio_filling_comparison.png')
    plt.close()
    
    print("Comparison plot saved as 'sharpe_ratio_filling_comparison.png'")
    
    return df_original



# Example usage:
comparison_df = compare_filling_methods('IFF_raw_data.csv')