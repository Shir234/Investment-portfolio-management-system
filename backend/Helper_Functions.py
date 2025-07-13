# Helper_Functions.py
import os
import pandas as pd

def save_csv_to_drive(logger, df, ticker_symbol, file_name, date_folder, current_date, index=True):
    """
    Save DataFrame to both local folder and Google Drive with error handling.
    
    Parameters:
    - logger : Logger instance for logging operations
    - df : pd.DataFrame DataFrame to save
    - ticker_symbol : str Stock ticker symbol
    - file_name : str Base name for the file (without .csv extension)
    - date_folder : str Local date folder path
    - current_date : str Current date string for subfolder creation
    - index : bool, optional Whether to save DataFrame index (default: True)
    """
     
    drive_path = r"G:\.shortcut-targets-by-id\19E5zLX5V27tgCL2D8EysE2nKWTQAEUlg\Investment portfolio management system\code_results\results\predictions" 
    
    # Create date folder inside Google Drive path
    drive_date_folder = os.path.join(drive_path, current_date)

    # Ensure directories exist
    os.makedirs(date_folder, exist_ok=True)
    os.makedirs(drive_date_folder, exist_ok=True)
    
    # Construct file names
    local_file_path = os.path.join(date_folder, f"{ticker_symbol}_{file_name}.csv")
    drive_file_path = os.path.join(drive_date_folder, f"{ticker_symbol}_{file_name}.csv")

    try:
        # Save to local folder
        df.to_csv(local_file_path, index=index)
        logger.info(f"Saved {ticker_symbol}_{file_name}.csv locally")

        # Save to Google Drive
        df.to_csv(drive_file_path, index=index)
        logger.info(f"Saved {ticker_symbol}_{file_name}.csv to Google Drive")
        
    except Exception as e:
        logger.error(f"Error saving results to Google Drive: {e}")
        
        # Fallback: try saving to current_date folder as backup
        try:
            fallback_folder = current_date
            os.makedirs(fallback_folder, exist_ok=True)
            fallback_path = os.path.join(fallback_folder, f"{ticker_symbol}_{file_name}.csv")
            df.to_csv(fallback_path, index=index)
            logger.info(f"Fallback: Saved {ticker_symbol}_{file_name}.csv to {fallback_folder}")
            
        except Exception as e2:
            logger.error(f"Fallback save also failed for {ticker_symbol}_{file_name}: {e2}")


def load_valid_tickers(logger, file_path="valid_tickers_av.csv"):
    """
    Loads valid tickers from a CSV file
    Parameters:
    - file_path (str): Path to the CSV file containing valid tickers
        
    Returns:
    - list: List of valid ticker symbols
    """

    if not os.path.exists(file_path):
        logger.error(f"Error: File {file_path} not found.")
        return []
    
    try:
        df = pd.read_csv(file_path)
        tickers = df['Ticker'].tolist()
        logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
    except Exception as e:
        logger.error(f"Error loading tickers from {file_path}: {e}")
        return []