# Imports
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from bs4 import BeautifulSoup
import requests

# Function Definitions
def parse_date(date_str):
    """Format the date"""
    return datetime.strptime(date_str, '%Y-%m-%d')

def calculate_transaction_score(transaction, latest_date):
    """
    Calculate the transaction score based on stakeholder weight, transaction magnitude, and temporal decay
    
    Args:
        transaction: A row from the transactions DataFrame
        latest_date: Reference date for temporal decay calculation (format: 'YYYY-MM-DD')
        
    Returns:
        float: Calculated transaction score
    """
    # 1. Stakeholder Weight
    stakeholder_weights = {
        'Executive Management': 1.0,
        'Board Members': 0.8,
        'Senior Management': 0.6,
        'Other Stakeholders': 0.4
    }

    # Determine stakeholder type
    if transaction['Executive Management'] == 1:
        stakeholder_weight = stakeholder_weights['Executive Management']
    elif transaction['Senior Management'] == 1:
        stakeholder_weight = stakeholder_weights['Senior Management']
    elif transaction['Board Members'] == 1:
        stakeholder_weight = stakeholder_weights['Board Members']
    else:
        stakeholder_weight = stakeholder_weights['Other Stakeholders']

    # 2. Transaction Magnitude and direction (consider quantity, price and type-buy/sell)
    # Uses logarithmic transformation for quantity and price to handle wide ranges
    # Always positive, compresses large values, preserves order of magnitude, prevents extreme scores from outliers
    quantity_score = np.log1p(abs(transaction['Qty']))
    price_score = np.log1p(transaction['Price'])
    
    # Direction score: buy-positive, sell-negative
    direction_score = 1 if transaction['Is_Buy'] == 1 else -1

    # Combine Magnitude and direction
    transaction_magnitude_score = direction_score * quantity_score * price_score

    # 3. Temporal decay
    # Applies exponential decay based on the time difference
    # Older transactions get progressively lower weights
    trade_date = parse_date(transaction['Trade_Date'])
    latest_date = parse_date(latest_date)
    days_diff = (latest_date - trade_date).days

    # Exponential decay with adjustable half-life
    half_life_days = 365 * 2
    temporal_decay = 0.5 ** (days_diff / half_life_days)

    # 4. Final Score Calculation
    # Combines stakeholder weight, transaction magnitude, and temporal decay
    final_score = (
        0.4 * stakeholder_weight *              # Stakeholder importance
        0.3 * transaction_magnitude_score *     # Transaction characteristics
        0.3 * temporal_decay                    # Time decay
    )

    return final_score

def filter_sp500_tickers(df, tickers):
    """
    Filter DataFrame to keep only S&P 500 tickers

    Args:
        df (pandas.DataFrame): Input DataFrame
        tickers (list): List of S&P 500 ticker symbols

    Returns:
        pandas.DataFrame: Filtered DataFrame with only S&P 500 tickers
    """
    # Filter the DataFrame to keep only S&P 500 tickers
    filtered_df = df[df['Ticker'].isin(tickers)]

    # Print some information about the filtering
    print(f"Original DataFrame size: {len(df)} rows")
    print(f"Filtered DataFrame size: {len(filtered_df)} rows")
    print(f"Number of unique S&P 500 tickers in the filtered data: {filtered_df['Ticker'].nunique()}")

    return filtered_df

def process_transactions(df, latest_date):
    """
    Calculate transaction score for all transactions
    
    Args:
        df (pandas.DataFrame): Input DataFrame with transactions
        latest_date (str): Reference date for score calculation (format: 'YYYY-MM-DD')
        
    Returns:
        pandas.DataFrame: DataFrame with added Transaction_Score column
    """
    # Create a copy, avoid modifying the original data
    processed_df = df.copy()

    # Calculate transaction score for each row
    processed_df['Transaction_Score'] = processed_df.apply(
        lambda row: calculate_transaction_score(row, latest_date),
        axis=1
    )

    return processed_df

def Finalaize_scores(df):
    """
    Calculate final score for each ticker by summing all transaction scores
    
    Args:
        df (pandas.DataFrame): DataFrame with Transaction_Score column
        
    Returns:
        pandas.DataFrame: DataFrame with ticker symbols and their final scores
    """
    # Calculate transaction score for each row
    final_scores = df.groupby('Ticker')['Transaction_Score'].sum().reset_index()
    final_scores = final_scores.sort_values(by='Transaction_Score', ascending=False)
    final_scores.to_csv('final_tickers_score.csv', index=False)

    return final_scores

# Main Code - Data Loading and Processing

# Read data from csv (Stakeholders Transactions)
df = pd.read_csv('InsiderTrading_sharp.csv')

# Get a list of S&P500 tickers
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table with S&P 500 companies
table = soup.find('table', {'id': 'constituents'})

# Extract the ticker symbols
tickers = []
for row in table.find_all('tr')[1:]:  # Skip the header row
    cells = row.find_all('td')
    if len(cells) > 0:  # make sure we have data
        ticker = cells[0].text.strip()
        tickers.append(ticker)

# Filter Transactions data to only S&P500 transactions
filtered_df_snp500 = filter_sp500_tickers(df, tickers)
filtered_df_snp500.to_csv('sp500_filtered_data.csv', index=False)

# Calculate the scores (referenced today)
latest_date = '2023-09-13'  # Hardcoded as YYYY-MM-DD
processed_df = process_transactions(df, latest_date)


# Calculate Final Score For Each Ticker Symbol
final_scores = Finalaize_scores(processed_df)
