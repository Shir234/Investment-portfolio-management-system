# Fetch_and_Clean_Data.py
import pandas as pd
import numpy as np
import os
import time
import logging
import datetime
from Data_Cleaning_Pipelines import create_stock_data_pipeline, create_data_cleaning_pipeline


def load_valid_tickers(logger, file_path="valid_tickers.csv"):
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
    
def full_pipeline_fetch_data_for_single_stock(logger, date_folder, current_date, ticker_symbol, start_date, end_date, risk_free_rate = 0.02):
    logger.info(f"STARTING PIPELINE FOR TICKER {ticker_symbol}")

    try:
        start_time = time.time()
        # TODO -> comment and uncomment if using the machine
        # Shir's Path G:\My Drive\Investment portfolio management system\code_results\results\predictions
        #drive_path = r"G:\My Drive\Investment portfolio management system\code_results\results\predictions/"
        # # machine's Path
        # drive_path = r"G:\.shortcut-targets-by-id\19E5zLX5V27tgCL2D8EysE2nKWTQAEUlg\Investment portfolio management system\code_results\results\predictions/" #(previous path)
        drive_path = r"G:\.shortcut-targets-by-id\19E5zLX5V27tgCL2D8EysE2nKWTQAEUlg\Investment portfolio management system\code_results\results\predictions"
        
        # Create date folder inside Google Drive path
        drive_date_folder = os.path.join(drive_path, current_date)
        # Create directory if it doesn't exist
        os.makedirs(drive_date_folder,exist_ok=True)

        # Run first pipeline, fetch data
        logger.info(f"\n{'-'*30}\nFetching and processing data for {ticker_symbol}\n{'-'*30}")
        
        pipeline = create_stock_data_pipeline(ticker_symbol, start_date, end_date, risk_free_rate)
        data = pipeline.fit_transform(pd.DataFrame())
        data.to_csv(f'{date_folder}/{ticker_symbol}_data.csv')        

        if data.empty:
            logger.error(f"No data returned for ticker {ticker_symbol}")
            return False
        
        # Run second pipeline, clean and process
        logger.info(f"\n{'-'*30}\nCleaning data for {ticker_symbol}\n{'-'*30}")
        pipeline_clean = create_data_cleaning_pipeline()
        data_clean = pipeline_clean.fit_transform(data)

        # Save locally and to Google Drive
        try:
            data_clean.to_csv(f'{date_folder}/{ticker_symbol}_clean_data.csv')
            data_clean.to_csv(os.path.join(drive_date_folder, f"{ticker_symbol}_clean_data.csv"))
            logger.info(f"Saved clean data for {ticker_symbol} to folders")
        except Exception as e:
            logger.error(f"Error saving to Google Drive: {e}")
            os.makedirs(current_date, exist_ok=True) # Create local date folder if needed
            data_clean.to_csv(os.path.join(current_date, f"{ticker_symbol}_clean_data.csv"))
        
        # Check if we have transaction metrics
        if 'Transaction_Sharpe' not in data_clean.columns:
            logger.error(f"No Transaction_Sharpe data for {ticker_symbol}. Skipping.")
            return False
                
        end_time = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"COMPLETED PIPELINE FOR TICKER {ticker_symbol} in {end_time - start_time:.2f} seconds")
        logger.info(f"{'='*50}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in pipeline for {ticker_symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False
    


# ===============================================================================
# Running the -> Full Pipeline For Single Stock
# ===============================================================================
def run_pipeline_fetch_data(logger, date_folder, current_date, tickers_file="valid_tickers.csv", start_date="2013-01-01", end_date="2024-01-01"): 
    # Load tickers
    valid_tickers = load_valid_tickers(logger, tickers_file)
    
    if not valid_tickers:
        logger.error("No valid tickers to process. Exiting.")
        return {'status': 'error', 'message': 'No valid tickers'}
    
    # Track results
    results = {
        'total_tickers': len(valid_tickers),
        'successful': 0,
        'failed': 0,
        'start_time': time.time(),
        'tickers_processed': []
    }

    logger.info(f"Loaded {len(valid_tickers)} valid tickers.") 
    # Process each ticker
    for i, ticker in enumerate(valid_tickers):
        logger.info(f"\nProcessing ticker {i+1}/{len(valid_tickers)}: {ticker}")
        
        try:  
            success = full_pipeline_fetch_data_for_single_stock(logger, date_folder, current_date, ticker, start_date, end_date)
                        
            if success:
                logger.info(f"Successfully processed {ticker}")
                results['successful'] += 1
                results['tickers_processed'].append({'ticker': ticker, 'status': 'success'})
            else:
                logger.warning(f"Failed to process {ticker}")
                results['failed'] += 1
                results['tickers_processed'].append({'ticker': ticker, 'status': 'failed'})
        
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            import traceback
            traceback.print_exc()
            results['failed'] += 1
            results['tickers_processed'].append({'ticker': ticker, 'status': 'error', 'message': str(e)})
    
    # Calculate stats
    end_time = time.time()
    total_time = end_time - results['start_time']
    results['total_time'] = total_time
    results['avg_time_per_ticker'] = total_time / len(valid_tickers) if valid_tickers else 0
    
    # Log summary
    logger.info(f"\n{'='*50}")
    logger.info(f"PIPELINE SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total tickers: {results['total_tickers']}")
    logger.info(f"Successfully processed: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Success rate: {results['successful']/results['total_tickers']*100:.2f}%")
    logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Average time per ticker: {results['avg_time_per_ticker']:.2f} seconds")
    logger.info(f"{'='*50}")
    
    # Save results to file
    summary_df = pd.DataFrame(results['tickers_processed'])
    summary_df.to_csv(f'{date_folder}/pipeline_summary_{current_date}.csv', index=False)
    
    return results




# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"stock_pipeline_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stock_pipeline")

# ===================================================
### Create data directories if they don't exist
# ===================================================
os.makedirs('results', exist_ok=True)
current_date = datetime.datetime.now().strftime("%Y%m%d")
date_folder = f'results/{current_date}'
os.makedirs(date_folder, exist_ok=True)

# Example usage
if __name__ == "__main__":
    import yfinance as yf

    # Call the main pipeline function
    pipeline_results = run_pipeline_fetch_data(logger, date_folder, current_date, tickers_file="valid_tickers.csv", start_date="2013-01-01", end_date="2024-01-01")
   
    # Exit with appropriate code
    if pipeline_results['successful'] > 0:
        print("Pipeline completed successfully.")
        exit(0)
    else:
        print("Pipeline failed to process any tickers successfully.")
        exit(1)
    