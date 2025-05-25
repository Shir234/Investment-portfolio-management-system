import time
import random
import os
import pandas as pd
import yfinance as yf
import requests
from requests.exceptions import RequestException
from datetime import datetime

def validate_and_save_tickers_alpha_vantage(tickers_list, api_key, output_file="valid_tickers_av.csv", batch_size=15):
    """
    Validates a list of ticker symbols using Alpha Vantage API and saves valid ones to a CSV file
    Optimized for premium API with 75 requests per minute rate limit
    
    Parameters:
        tickers_list (list): List of ticker symbols to validate
        api_key (str): Alpha Vantage API key
        output_file (str): Filename to save valid tickers
        batch_size (int): Number of tickers to process in each batch
    
    Returns:
        list: List of valid ticker symbols
    """
    valid_tickers = []
    invalid_tickers = []
    total_tickers = len(tickers_list)
    requests_this_minute = 0
    minute_start_time = time.time()
    
    print(f"Validating {total_tickers} tickers using Alpha Vantage API (75 requests/min) in batches of {batch_size}...")
    
    # Process tickers in batches
    for i in range(0, total_tickers, batch_size):
        batch = tickers_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_tickers + batch_size - 1)//batch_size}")
        
        for ticker in batch:
            # Check if we need to reset our minute counter
            current_time = time.time()
            if current_time - minute_start_time >= 60:
                requests_this_minute = 0
                minute_start_time = current_time
            
            # If we're approaching the rate limit, wait until the next minute starts
            if requests_this_minute >= 72:  # Leave a small buffer
                seconds_to_wait = 60 - (current_time - minute_start_time) + 1
                print(f"Approaching rate limit. Waiting {seconds_to_wait:.2f} seconds until next minute...")
                time.sleep(seconds_to_wait)
                requests_this_minute = 0
                minute_start_time = time.time()
            
            try:
                # Use Alpha Vantage's Global Quote endpoint to validate ticker
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
                response = requests.get(url)
                requests_this_minute += 1
                data = response.json()
                
                # Check if we got a valid response
                if "Global Quote" in data and data["Global Quote"] and "01. symbol" in data["Global Quote"]:
                    print(f"Ticker {ticker} is valid. Adding to list.")
                    valid_tickers.append(ticker)
                else:
                    print(f"Ticker {ticker} is not valid or no data available. Skipping.")
                    invalid_tickers.append(ticker)
                    
                    # Log any error message if present
                    if "Error Message" in data:
                        print(f"  Error: {data['Error Message']}")
                    elif "Information" in data:
                        print(f"  Info: {data['Information']}")
                    elif "Note" in data:
                        print(f"  Note: {data['Note']}")
                
            except RequestException as e:
                print(f"Error validating ticker {ticker}: {e}")
                invalid_tickers.append(ticker)
            
            # Add a minimal delay between requests
            time.sleep(0.05)
    
    # Save valid tickers to CSV
    df_valid = pd.DataFrame({'Ticker': valid_tickers})
    df_valid.to_csv(output_file, index=False)
    
    # Also save invalid tickers for reference
    df_invalid = pd.DataFrame({'Ticker': invalid_tickers})
    df_invalid.to_csv(f"invalid_{output_file}", index=False)
    
    print(f"Validation complete. Found {len(valid_tickers)} valid tickers out of {total_tickers}.")
    print(f"Valid tickers saved to {output_file}")
    print(f"Invalid tickers saved to invalid_{output_file}")
    
    return valid_tickers

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


if __name__ == "__main__":
    """
    1 read the data
    2 get all the tickers
    3 validate and get only valid tickers
    4 run the pipeline on each valid ticker
    """
    stakeholder_data = pd.read_csv('final_tickers_score.csv')
    # Get all tickers from data
    all_tickers = stakeholder_data['Ticker'].tolist()
    # Your Alpha Vantage API key
    alpha_vantage_api_key = "3T3LNJ2UYGY4R7WO"
    # Validate tickers using Alpha Vantage and save to CSV
    validate_and_save_tickers_alpha_vantage(all_tickers, alpha_vantage_api_key, "valid_tickers_av.csv", batch_size=15)
    
    # Validate tickers and save to CSV
   # validate_and_save_tickers(all_tickers, "valid_tickers.csv", batch_size=5)
