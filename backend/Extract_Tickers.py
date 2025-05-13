# Extract_Tickers.py

import os

def extract_ticker_names(base_directory, date_folder, output_file=None):
    """
    Scan a date-specific folder for files named like 'ZTS_clean_data.csv', extract ticker names,
    and save them to a text file in the same folder.

    Parameters:
    base_directory (str): Path to the base results directory (e.g., 'results')
    date_folder (str): Name of the specific date subfolder (e.g., '20250402')
    output_file (str): Name of the output text file (default: '<date_folder>_tickers.txt')

    Returns:
    None
    """
    # Build the complete path to the date folder
    date_path = os.path.join(base_directory, date_folder)
    
    # Set default output file name if not provided
    if output_file is None:
        output_file = f"{date_folder}_tickers.csv"
    
    # Construct full path for the output file
    output_file_path = os.path.join(date_path, output_file)
    
    # Check if the folder exists
    if not os.path.exists(date_path):
        print(f"Folder path '{date_path}' does not exist.")
        return
    
    # Get list of files ending with '_clean_data.csv'
    files = [f for f in os.listdir(date_path) if f.endswith('_clean_data.csv')]
    
    # Extract ticker names by removing '_clean_data.csv'
    tickers = [f.replace('_clean_data.csv', '') for f in files]
    
    if not tickers:
        print(f"No files matching '*_clean_data.csv' found in {date_path}")
        return
    
    # Save ticker names to the output file
    try:
        with open(output_file_path, 'w') as f:
            for ticker in sorted(tickers):  
                f.write(f"{ticker}\n")
        print(f"Saved {len(tickers)} ticker names to {output_file_path}")
    except Exception as e:
        print(f"Error saving ticker names to {output_file_path}: {e}")

# Example usage:
if __name__ == "__main__":
    base_directory = "results" 
    date_folder = "20250426"
    extract_ticker_names(base_directory, date_folder)