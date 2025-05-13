# Visualizations.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

# Read the CSV file
df = pd.read_csv('20250426_all_tickers_results.csv')

# Convert Date column to datetime, handling timezone issues
df['Date'] = pd.to_datetime(df['Date'], utc=True)

# Filter for the required tickers
#required_tickers = ['T', 'WBD', 'VICI', 'CTVA', 'FOX']
required_tickers = ['AAPL', 'GOOGL', 'FOX']

df_filtered = df[df['Ticker'].isin(required_tickers)]

# Convert filter dates to datetime with timezone awareness
start_date = pd.to_datetime('2023-09-22', utc=True)
end_date = pd.to_datetime('2023-12-22', utc=True)

# Filter for the last 6 months of 2023 (July-December)
df_filtered = df_filtered[(df_filtered['Date'] >= start_date) & (df_filtered['Date'] <= end_date)]

# Create a time series comparison chart with pastel colors
fig, ax = plt.subplots(figsize=(14, 8))

# Define pastel colors
pastel_colors = ['#AEC6CF', '#FFB7B2', '#B2DFDB', '#FFE4B5', '#E6E6FA']  # Light Blue, Light Coral, Light Teal, Moccasin, Lavender
# Define vibrant colors that are still professional
vibrant_colors = [
    'blue',  # Steel Blue
    'red',  # Cinnabar Red
    'green',  # Persian Green
    '#F4A261',  # Sandy Brown
    '#7209B7'   # Purple
]

for i, ticker in enumerate(required_tickers):
    ticker_data = df_filtered[df_filtered['Ticker'] == ticker].sort_values('Date')
    
    # Convert dates to string format for better display
    dates = ticker_data['Date'].dt.strftime('%Y-%m-%d')
    actual = ticker_data['Actual_Sharpe']
    predicted = ticker_data['Best_Prediction']
    
    # Plot actual as lines
    ax.plot(ticker_data['Date'], actual, color=vibrant_colors[i], linewidth=2.5, 
            label=f'{ticker} Actual', alpha=0.8)
    
    # Plot predicted as dots
    ax.scatter(ticker_data['Date'], predicted, color=vibrant_colors[i], s=60, 
              marker='o', edgecolor='white', linewidth=1.5,
              label=f'{ticker} Predicted', alpha=0.9)

# ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Sharpe Ratio', fontsize=14)
# ax.set_title('Actual vs Predicted Sharpe Ratios - Time Series Comparison', 
#              fontsize=16, fontweight='bold')

# Customize the legend
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
          framealpha=0.9, edgecolor='lightgray')

# Add grid with light style
ax.grid(True, alpha=0.3, linestyle='--', color='gray')

# Set background color to white
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Format x-axis dates
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())

# Make dates horizontal (no rotation)
plt.xticks(rotation=0, ha='center')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('sharpe_time_series_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Time series comparison graph has been saved as 'sharpe_time_series_comparison.png'")
