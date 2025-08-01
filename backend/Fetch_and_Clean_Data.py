# Fetch_and_Clean_Data.py
import pandas as pd
import os
import time
import logging
import datetime
from Data_Cleaning_Pipelines import create_stock_data_pipeline, create_data_cleaning_pipeline
from Helper_Functions import load_valid_tickers, save_csv_to_drive

    
def full_pipeline_fetch_data_for_single_stock(logger, date_folder, current_date, api_key, 
                                              ticker_symbol, start_date, end_date, risk_free_rate = 0.02, requests_this_minute=0, minute_start_time=None):
    """
    Executes complete data pipeline for a single stock ticker with API rate limiting.
    
    Process:
    1. Manages API rate limiting (75 requests/minute for premium Alpha Vantage)
    2. Fetches historical data and calculates technical indicators
    3. Applies data cleaning (outliers, correlations, missing values)
    4. Saves results to both local and Google Drive locations
    
    Note: API key is expired - was purchased for limited time period only.
    
    Returns:
    - tuple: (success, requests_this_minute, minute_start_time)
    """
     
    logger.info(f"STARTING PIPELINE FOR TICKER {ticker_symbol}")

    if minute_start_time is None:
        minute_start_time = time.time()

    try:
        start_time = time.time()
        # Check if we need to reset our minute counter
        current_time = time.time()
        if current_time - minute_start_time >= 60:
            requests_this_minute = 0
            minute_start_time = current_time
            logger.info("Rate limit counter reset for new minute")
        
        # If we're approaching the rate limit, wait until the next minute starts
        if requests_this_minute >= 72:  # 75 - buffer
            seconds_to_wait = 60 - (current_time - minute_start_time) + 1
            logger.info(f"Approaching rate limit. Waiting {seconds_to_wait:.2f} seconds until next minute...")
            time.sleep(seconds_to_wait)
            requests_this_minute = 0
            minute_start_time = time.time()

        drive_path = r"G:\.shortcut-targets-by-id\19E5zLX5V27tgCL2D8EysE2nKWTQAEUlg\Investment portfolio management system\code_results\results\predictions"
        
        # Create date folder inside Google Drive path
        drive_date_folder = os.path.join(drive_path, current_date)
        # Create directory if it doesn't exist
        os.makedirs(drive_date_folder,exist_ok=True)

        # Run first pipeline, fetch data
        logger.info(f"\n{'-'*30}\nFetching and processing data for {ticker_symbol}\n{'-'*30}")
        
        pipeline = create_stock_data_pipeline(ticker_symbol, start_date, end_date, risk_free_rate, api_key)
        data = pipeline.fit_transform(pd.DataFrame())

        # Increment request counter - assuming pipeline uses about 10 API calls
        requests_this_minute += 10
        logger.info(f"API request count: {requests_this_minute}/75 this minute")

        if data.empty:
            logger.error(f"No data returned for ticker {ticker_symbol}")
            return False, requests_this_minute, minute_start_time
        
        data.index.name = "Date"  # Name the index column
        save_csv_to_drive(logger, data, ticker_symbol, 'raw_data', date_folder, current_date, index=True)
        logger.info(f"Saved raw data for {ticker_symbol} to folders")

        try:
            data.index.name = "Date"  # Name the index column
            data.to_csv(f'{date_folder}/{ticker_symbol}_raw_data.csv')
            data.to_csv(os.path.join(drive_date_folder, f"{ticker_symbol}_raw_data.csv"))
            logger.info(f"Saved raw data for {ticker_symbol} to folders")
        except Exception as e:
            logger.error(f"Error saving to Google Drive: {e}")
            os.makedirs(current_date, exist_ok=True) # Create local date folder if needed
        
        # Run second pipeline, clean and process
        logger.info(f"\n{'-'*30}\nCleaning data for {ticker_symbol}\n{'-'*30}")
        pipeline_clean = create_data_cleaning_pipeline()
        data_clean = pipeline_clean.fit_transform(data)

        data_clean.index.name = "Date"  # Name the index column
        save_csv_to_drive(logger, data_clean, ticker_symbol, 'clean_data', date_folder, current_date, index=True)
        logger.info(f"Saved clean data for {ticker_symbol} to folders")
        
        # Save locally and to Google Drive
        try:
            data_clean.index.name = "Date"  # Name the index column
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
        
        return True, requests_this_minute, minute_start_time
        
    except Exception as e:
        logger.error(f"Error in pipeline for {ticker_symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False, requests_this_minute, minute_start_time
    

def run_pipeline_fetch_data(logger, date_folder, current_date, api_key, tickers_file="valid_tickers_av", start_date="2013-01-01", end_date="2024-01-01"): 
    """
    Main orchestrator for batch processing multiple stock tickers.
    
    Workflow:
    1. Loads ticker list from CSV file
    2. Processes each ticker sequentially with rate limiting
    3. Tracks progress and saves intermediate results every 10 tickers
    4. Generates comprehensive summary with success/failure statistics
    5. Provides timing analysis and error reporting
    
    Handles both single-ticker and multi-ticker processing modes.
    """

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

    # Rate limiting variables
    requests_this_minute = 0
    minute_start_time = time.time()

    logger.info(f"Loaded {len(valid_tickers)} valid tickers.") 
    logger.info(f"Rate limit set to 75 requests per minute (premium API)")

    # Process each ticker
    for i, ticker in enumerate(valid_tickers):
        logger.info(f"\nProcessing ticker {i+1}/{len(valid_tickers)}: {ticker}")
        
        try:  
            success, requests_this_minute, minute_start_time = full_pipeline_fetch_data_for_single_stock(
                logger, date_folder, current_date, api_key, ticker, start_date, end_date,
                requests_this_minute=requests_this_minute, minute_start_time=minute_start_time
            )
                        
            if success:
                logger.info(f"Successfully processed {ticker}")
                results['successful'] += 1
                results['tickers_processed'].append({'ticker': ticker, 'status': 'success'})
            else:
                logger.warning(f"Failed to process {ticker}")
                results['failed'] += 1
                results['tickers_processed'].append({'ticker': ticker, 'status': 'failed'})
            
            # Save intermediate results periodically
            if (i+1) % 10 == 0 or (i+1) == len(valid_tickers):
                summary_df = pd.DataFrame(results['tickers_processed'])
                summary_df.to_csv(f'{date_folder}/pipeline_summary_intermediate_{current_date}.csv', index=False)
                logger.info(f"Saved intermediate results after processing {i+1}/{len(valid_tickers)} tickers")
        
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


# Create data directories if they don't exist
os.makedirs('results', exist_ok=True)
current_date = datetime.datetime.now().strftime("%Y%m%d")
date_folder = f'results/{current_date}'
os.makedirs(date_folder, exist_ok=True)

if __name__ == "__main__":
   # API_KEY - not available anymore!!
    API_KEY = "3T3LNJ2UYGY4R7WO"  # Get from https://www.alphavantage.co/support/#api-key
    # Call the main pipeline function
    pipeline_results = run_pipeline_fetch_data(
        logger, 
        date_folder, 
        current_date, 
        api_key=API_KEY,
        tickers_file="valid_tickers_av.csv", # for more than 1 ticker
        #tickers_file="one_ticker.csv", # for 1 ticker
        start_date="2013-01-01", 
        end_date="2024-01-01"
    )

    # Exit with appropriate code
    if pipeline_results['successful'] > 0:
        print("Pipeline completed successfully.")
        exit(0)
    else:
        print("Pipeline failed to process any tickers successfully.")
        exit(1)
    