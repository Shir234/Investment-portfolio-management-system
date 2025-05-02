# check_negatives.py

import pandas as pd

def check_negative_values(csv_file_path, column_name):
    """
    Check if there are negative values in a specific column of a CSV file.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file
    column_name : str
        Name of the column to check for negative values
    
    Returns:
    --------
    dict : Dictionary containing:
        - has_negatives: Boolean indicating if negative values exist
        - negative_count: Count of negative values
        - negative_values: The actual negative values found
        - negative_rows: DataFrame containing rows with negative values
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if the column exists
        if column_name not in df.columns:
            return {
                'error': f"Column '{column_name}' not found in the CSV file",
                'available_columns': df.columns.tolist()
            }
        
        # Check for negative values
        negative_mask = df[column_name] < 0
        negative_values = df[column_name][negative_mask]
        
        # Get rows with negative values
        negative_rows = df[negative_mask]
        
        result = {
            'has_negatives': negative_mask.any(),
            'negative_count': negative_mask.sum(),
            'negative_values': negative_values.tolist(),
            'negative_rows': negative_rows,
            'total_rows': len(df),
            'percentage_negative': (negative_mask.sum() / len(df)) * 100
        }
        
        return result
        
    except Exception as e:
        return {'error': f"Error reading CSV file: {str(e)}"}

# Example usage
if __name__ == "__main__":
    # Example 1: Basic check
    #result = check_negative_values('OPK_data.csv', 'Transaction_Sharpe')
    result = check_negative_values('OPK_clean_data.csv', 'Transaction_Sharpe')
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        if result['has_negatives']:
            print(f"Found {result['negative_count']} negative values in column 'Transaction_Sharpe'")
            print(f"That's {result['percentage_negative']:.2f}% of all rows")
            print(f"Negative values: {result['negative_values'][:10]}")  # Show first 10 negative values
        else:
            print("No negative values found in column 'Transaction_Sharpe'")
    
    # # Example 2: Check multiple columns
    # columns_to_check = ['Actual_Sharpe', 'Best_Prediction']
    
    # for column in columns_to_check:
    #     result = check_negative_values('20250426_all_tickers_results.csv', column)
    #     if 'error' not in result and result['has_negatives']:
    #         print(f"Column '{column}': {result['negative_count']} negative values")
    
    # # Example 3: Save rows with negative values to another CSV
    # result = check_negative_values('your_file.csv', 'Transaction_Sharpe')
    # if 'error' not in result and result['has_negatives']:
    #     result['negative_rows'].to_csv('negative_values.csv', index=False)
    #     print(f"Saved {result['negative_count']} rows with negative values to 'negative_values.csv'")