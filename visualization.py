import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator


# Set the style for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Load the merged data
merged_data = pd.read_csv('20250415_all_tickers_results.csv')

# Ensure date is in datetime format
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# Filter out rows where Actual_Sharpe or Best_Prediction is NaN
filtered_data = merged_data.dropna(subset=['Actual_Sharpe', 'Best_Prediction'])

# 1. SCATTER PLOT: Predicted vs Actual Sharpe
# Create the scatter plot
plt.figure(figsize=(10, 8))

# Create scatter plot with semi-transparent points
scatter = plt.scatter(
    filtered_data['Actual_Sharpe'], 
    filtered_data['Best_Prediction'],
    alpha=0.7,
    s=70,  # Point size
    c=filtered_data['Actual_Sharpe'],  # Color points by actual Sharpe ratio
    cmap='viridis',  # Color map
    edgecolor='white',  # White edge for better visibility
    linewidth=0.5
)

# Add a 45-degree line representing perfect prediction
min_val = min(filtered_data['Actual_Sharpe'].min(), filtered_data['Best_Prediction'].min())
max_val = max(filtered_data['Actual_Sharpe'].max(), filtered_data['Best_Prediction'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1.5, 
         label='Perfect Prediction (Actual = Predicted)')

# Add a regression line to show overall trend
z = np.polyfit(filtered_data['Actual_Sharpe'], filtered_data['Best_Prediction'], 1)
p = np.poly1d(z)
plt.plot(
    np.sort(filtered_data['Actual_Sharpe']),
    p(np.sort(filtered_data['Actual_Sharpe'])),
    color='#FF5722',  # Distinctive orange color for trend line
    linewidth=2.5,
    label=f'Trend Line (y = {z[0]:.2f}x + {z[1]:.2f})'
)

# # Calculate and display R² value
# correlation_matrix = np.corrcoef(filtered_data['Actual_Sharpe'], filtered_data['Best_Prediction'])
# correlation_xy = correlation_matrix[0,1]
# r_squared = correlation_xy**2

# # Add text box with R² value
# props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
# plt.text(
#     0.05, 0.95, 
#     f'R² = {r_squared:.3f}', 
#     transform=plt.gca().transAxes,
#     fontsize=12,
#     verticalalignment='top',
#     bbox=props
# )

# Add colorbar to show what the colors represent
cbar = plt.colorbar(scatter)
cbar.set_label('Actual Sharpe Ratio Value', fontsize=12)

# Add labels and title with clear descriptions
plt.xlabel('Actual Sharpe Ratio', fontsize=14, fontweight='bold')
plt.ylabel('Predicted Sharpe Ratio', fontsize=14, fontweight='bold')
plt.title('Model Accuracy: Predicted vs. Actual Sharpe Ratios', fontsize=16, fontweight='bold')

# Add annotations for clarity
plt.annotate(
    'Points close to this line\nindicate accurate predictions',
    xy=(max_val*0.7, max_val*0.7),
    xytext=(max_val*0.6, max_val*0.5),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
)

# Add legend with clear descriptions
plt.legend(loc='upper left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('sharpe_scatter_plot.png', dpi=300, bbox_inches='tight')


# # 2. TIME SERIES PLOT: Sharpe Ratio Over Time
# # For time series, we need to pick a specific ticker or aggregate across tickers
# # Let's create a version that works with a single ticker first

# # For illustration, let's assume we're analyzing one ticker
# # In your actual code, you might want to loop through tickers or choose a specific one
# if 'Ticker' in filtered_data.columns:
#     unique_tickers = filtered_data['Ticker'].unique()
#     ticker_to_analyze = 'OPK'
#     ticker_data = filtered_data[filtered_data['Ticker'] == ticker_to_analyze]
# else:
#     # If there's no Ticker column, just use all the data
#     ticker_data = filtered_data
#     ticker_to_analyze = "All Tickers"

# # Sort by date
# ticker_data = ticker_data.sort_values('Date')

# # Create the time series plot
# plt.figure(figsize=(12, 6))

# # Plot actual Sharpe ratio
# plt.plot(
#     ticker_data['Date'], 
#     ticker_data['Actual_Sharpe'],
#     color='#2196F3',  # Blue for actual values
#     linewidth=2.5,
#     marker='o',
#     markersize=6,
#     label='Actual Sharpe Ratio'
# )

# # Plot predicted Sharpe ratio
# plt.plot(
#     ticker_data['Date'], 
#     ticker_data['Best_Prediction'],
#     color='#FF5722',  # Orange for predictions
#     linewidth=2.5,
#     marker='^',
#     markersize=6,
#     label='Predicted Sharpe Ratio'
# )

# # Calculate prediction error
# ticker_data['Error'] = abs(ticker_data['Actual_Sharpe'] - ticker_data['Best_Prediction'])
# max_error = ticker_data['Error'].max()
# error_threshold = max_error * 0.6  # Threshold to consider "high error"

# # Highlight regions with high prediction error
# high_error_points = ticker_data[ticker_data['Error'] > error_threshold]
# if not high_error_points.empty:
#     for _, point in high_error_points.iterrows():
#         plt.axvspan(
#             point['Date'] - pd.Timedelta(days=2), 
#             point['Date'] + pd.Timedelta(days=2),
#             alpha=0.2,
#             color='red',
#             zorder=0
#         )
    
#     # Add explanation for highlighted regions
#     plt.text(
#         0.02, 0.02, 
#         'Red areas highlight periods of\nlarger prediction errors',
#         transform=plt.gca().transAxes,
#         fontsize=10,
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
#     )

# # Format x-axis to show dates clearly
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(MaxNLocator(10))  # Limit to ~10 tick marks
# plt.xticks(rotation=45)

# # Add labels and title with clear descriptions
# plt.xlabel('Date', fontsize=14, fontweight='bold')
# plt.ylabel('Sharpe Ratio', fontsize=14, fontweight='bold')
# plt.title(f'Sharpe Ratio Predictions Over Time: {ticker_to_analyze}', fontsize=16, fontweight='bold')

# # Add legend
# plt.legend(loc='upper left', fontsize=12)

# # Add grid for better readability
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('sharpe_time_series.png', dpi=300, bbox_inches='tight')

# # Show overall model performance metrics
# avg_error = ticker_data['Error'].mean()
# max_error = ticker_data['Error'].max()
# print(f"Average Prediction Error: {avg_error:.4f}")
# print(f"Maximum Prediction Error: {max_error:.4f}")