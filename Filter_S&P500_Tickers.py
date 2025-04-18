import time
import random
import os
import pandas as pd
import yfinance as yf
import datetime

def validate_and_save_tickers(tickers_list, output_file="valid_tickers.csv", batch_size=5):
    """
    Validates a list of ticker symbols and saves valid ones to a CSV file
    
    Parameters:
        tickers_list (list): List of ticker symbols to validate
        output_file (str): Filename to save valid tickers
        batch_size (int): Number of tickers to process in each batch
    
    Returns:
        list: List of valid ticker symbols
    """
    valid_tickers = []
    total_tickers = len(tickers_list)
    
    print(f"Validating {total_tickers} tickers in batches of {batch_size}...")
    
    # Process tickers in batches
    for i in range(0, total_tickers, batch_size):
        batch = tickers_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_tickers + batch_size - 1)//batch_size}")
        
        for ticker in batch:
            try:
                ticker_data = yf.Ticker(ticker)
                if ticker_data.info and 'symbol' in ticker_data.info:
                    print(f"Ticker {ticker} is valid. Adding to list.")
                    valid_tickers.append(ticker)
                else:
                    print(f"Ticker {ticker} is not valid. Skipping.")
            except Exception as e:
                print(f"Error validating ticker {ticker}: {e}")
                
                # If rate limited, wait before continuing
                if "Rate limited" in str(e):
                    wait_time = random.uniform(3, 6)  # Wait 3-6 seconds
                    print(f"Rate limited. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
        
        # Add delay between batches to avoid rate limiting
        if i + batch_size < total_tickers:
            wait_time = random.uniform(1, 3)  # Random wait between 1-3 seconds
            print(f"Waiting {wait_time:.2f} seconds before next batch...")
            time.sleep(wait_time)
    
    # Save valid tickers to CSV
    df = pd.DataFrame({'Ticker': valid_tickers})
    df.to_csv(output_file, index=False)
    
    print(f"Validation complete. Found {len(valid_tickers)} valid tickers.")
    print(f"Valid tickers saved to {output_file}")
    
    return valid_tickers


"""
1 read the data
2 get all the tickers
3 validate and get only valid tickers
4 run the pipeline on each valid ticker
"""
stakeholder_data = pd.read_csv('final_tickers_score.csv')
# Get all tickers from data
all_tickers = stakeholder_data['Ticker'].tolist()
# Validate tickers and save to CSV
validate_and_save_tickers(all_tickers, "valid_tickers.csv", batch_size=5)

# # Just process the valid tickers
# for ticker in valid_tickers:
#     try:
#         print(f"\nProcessing ticker: {ticker}")
#         full_pipeline_for_single_stock(ticker, "2013-01-01", "2024-01-01")
#         print(f"Successfully processed {ticker}")
#     except Exception as e:
#         print(f"Error processing ticker {ticker}: {e}")