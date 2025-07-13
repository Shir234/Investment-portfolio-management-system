import os
import logging
import datetime
from Full_Pipeline_With_Data import run_pipeline

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
    import yfinance as yf
    
    # Call the main pipeline function
    pipeline_results = run_pipeline(logger, date_folder, current_date, tickers_file="valid_tickers_av.csv", start_date="2013-01-01", end_date="2024-01-01")
    
    # Exit with appropriate code
    if pipeline_results['successful'] > 0:
        print("Pipeline completed successfully.")
        exit(0)
    else:
        print("Pipeline failed to process any tickers successfully.")
        exit(1)