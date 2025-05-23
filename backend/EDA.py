import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

# Define the RollingSharpeCalculator for verification
class RollingSharpeCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, window=30, risk_free_rate=0.02, min_std=1e-6):
        self.window = window
        self.risk_free_rate = risk_free_rate
        self.min_std = min_std
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['Daily_Return'] = X['Close'].pct_change()
        X['Daily_Return'] = X['Daily_Return'].fillna(0)
        X['Rolling_Mean'] = X['Daily_Return'].rolling(window=self.window, min_periods=self.window).mean()
        X['Rolling_Std'] = X['Daily_Return'].rolling(window=self.window, min_periods=self.window).std()
        X['Rolling_Mean'] = X['Rolling_Mean'].fillna(0)
        X['Rolling_Std'] = X['Rolling_Std'].fillna(self.min_std)
        X['Rolling_Std'] = X['Rolling_Std'].clip(lower=self.min_std)
        daily_risk_free = self.risk_free_rate / 252
        X['Daily_Sharpe_Ratio'] = (X['Rolling_Mean'] - daily_risk_free) / X['Rolling_Std']
        X['Daily_Sharpe_Ratio'] = X['Daily_Sharpe_Ratio'].replace([np.inf, -np.inf], 0)
        X['Daily_Sharpe_Ratio'] = X['Daily_Sharpe_Ratio'].fillna(0)
        X.drop(['Daily_Return', 'Rolling_Mean', 'Rolling_Std'], axis=1, inplace=True)
        return X

# Define correlation analysis function
def analyze_correlations(df, ticker, folder_path):
    """
    Analyze correlations between features and Daily_Sharpe_Ratio, and between features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data with Daily_Sharpe_Ratio and other features
    ticker : str
        Ticker symbol for the stock
    folder_path : str
        Path to save the correlation results and heatmap
    
    Returns:
    --------
    None
        Saves correlation matrix to CSV and heatmap to PNG
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    
    # Calculate correlation matrix
    corr_matrix = df_numeric.corr()
    
    # Save correlation matrix to CSV
    corr_file_path = os.path.join(folder_path, f"{ticker}_correlation_matrix.csv")
    corr_matrix.to_csv(corr_file_path)
    print(f"Correlation matrix saved to: {corr_file_path}")
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Correlation Heatmap for {ticker}')
    plt.tight_layout()
    
    # Save heatmap
    heatmap_file_path = os.path.join(folder_path, f"{ticker}_correlation_heatmap.png")
    plt.savefig(heatmap_file_path)
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_file_path}")
    
    # Extract correlations with Daily_Sharpe_Ratio
    if 'Daily_Sharpe_Ratio' in corr_matrix:
        sharpe_corrs = corr_matrix['Daily_Sharpe_Ratio'].sort_values(ascending=False)
        sharpe_corr_file_path = os.path.join(folder_path, f"{ticker}_sharpe_correlations.csv")
        sharpe_corrs.to_csv(sharpe_corr_file_path)
        print(f"Daily_Sharpe_Ratio correlations saved to: {sharpe_corr_file_path}")



# Define folder path
folder_path = r"C:\Users\Shir.Falach\Desktop\SharpSight\Investment-portfolio-management-system\backend\results\20250522"

# Initialize lists to store data
tickers = []
daily_min_values = []
daily_max_values = []
trans_min_values = []
trans_max_values = []
global_daily_min = float('inf')
global_daily_max = float('-inf')
global_trans_min = float('inf')
global_trans_max = float('-inf')
daily_stats = []

# Initialize the calculator
sharpe_calculator = RollingSharpeCalculator(window=30, risk_free_rate=0.02)

# Iterate through files
for file_name in os.listdir(folder_path):
    if file_name.endswith('_clean_data.csv'):
        ticker = file_name.replace('_clean_data.csv', '')
        clean_file_path = os.path.join(folder_path, file_name)
        raw_file_path = os.path.join(folder_path, f"{ticker}_raw_data.csv")
        
        # Read clean data
        df_clean = pd.read_csv(clean_file_path)

        # Perform correlation analysis
        analyze_correlations(df_clean, ticker, folder_path)
        
        # Read raw data if available
        if os.path.exists(raw_file_path):
            df_raw = pd.read_csv(raw_file_path)
            # Recalculate Daily_Sharpe_Ratio on raw data
            df_raw = sharpe_calculator.transform(df_raw)
        else:
            print(f"Raw data for {ticker} not found. Skipping raw data analysis.")
            df_raw = None
        
        # Get min and max for Daily_Sharpe_Ratio (clean data)
        daily_min = df_clean['Daily_Sharpe_Ratio'].min()
        daily_max = df_clean['Daily_Sharpe_Ratio'].max()
        
        # Get min and max for Transaction_Sharpe (clean data), excluding -1
        trans_sharpe = df_clean[df_clean['Transaction_Sharpe'] != -1]['Transaction_Sharpe']
        trans_min = trans_sharpe.min() if not trans_sharpe.empty else float('inf')
        trans_max = trans_sharpe.max() if not trans_sharpe.empty else float('-inf')
        
        # Check for NaN and infinite values in clean data
        daily_nan_count = df_clean['Daily_Sharpe_Ratio'].isna().sum()
        daily_inf_count = np.isinf(df_clean['Daily_Sharpe_Ratio']).sum()
        
        # Stats for raw data if available
        if df_raw is not None:
            raw_daily_min = df_raw['Daily_Sharpe_Ratio'].min()
            raw_daily_max = df_raw['Daily_Sharpe_Ratio'].max()
            raw_daily_mean = df_raw['Daily_Sharpe_Ratio'].mean()
            raw_daily_std = df_raw['Daily_Sharpe_Ratio'].std()
            raw_daily_nan_count = df_raw['Daily_Sharpe_Ratio'].isna().sum()
            raw_daily_inf_count = np.isinf(df_raw['Daily_Sharpe_Ratio']).sum()
        else:
            raw_daily_min = raw_daily_max = raw_daily_mean = raw_daily_std = raw_daily_nan_count = raw_daily_inf_count = np.nan
        
        # Store stats
        daily_stats.append({
            'Ticker': ticker,
            'Clean_Daily_Min': daily_min,
            'Clean_Daily_Max': daily_max,
            'Clean_Daily_Mean': df_clean['Daily_Sharpe_Ratio'].mean(),
            'Clean_Daily_Std': df_clean['Daily_Sharpe_Ratio'].std(),
            'Clean_Daily_NaN': daily_nan_count,
            'Clean_Daily_Inf': daily_inf_count,
            'Raw_Daily_Min': raw_daily_min,
            'Raw_Daily_Max': raw_daily_max,
            'Raw_Daily_Mean': raw_daily_mean,
            'Raw_Daily_Std': raw_daily_std,
            'Raw_Daily_NaN': raw_daily_nan_count,
            'Raw_Daily_Inf': raw_daily_inf_count,
            'Trans_Min': trans_min,
            'Trans_Max': trans_max
        })
        
        # Store values for plotting
        tickers.append(ticker)
        daily_min_values.append(daily_min)
        daily_max_values.append(daily_max)
        trans_min_values.append(trans_min)
        trans_max_values.append(trans_max)
        
        # Update global min and max
        global_daily_min = min(global_daily_min, daily_min)
        global_daily_max = max(global_daily_max, daily_max)
        if trans_min != float('inf'):
            global_trans_min = min(global_trans_min, trans_min)
        if trans_max != float('-inf'):
            global_trans_max = max(global_trans_max, trans_max)

# Create DataFrame for stats
stats_df = pd.DataFrame(daily_stats)

# Save stats to CSV for sharing
stats_df.to_csv(os.path.join(folder_path, 'sharpe_ratio_detailed_stats.csv'), index=False)

# Limit to top 10 tickers by max Daily_Sharpe_Ratio for clarity
if len(tickers) > 10:
    sorted_indices = sorted(range(len(daily_max_values)), key=lambda i: daily_max_values[i], reverse=True)[:10]
    tickers = [tickers[i] for i in sorted_indices]
    daily_min_values = [daily_min_values[i] for i in sorted_indices]
    daily_max_values = [daily_max_values[i] for i in sorted_indices]
    trans_min_values = [trans_min_values[i] for i in sorted_indices]
    trans_max_values = [trans_max_values[i] for i in sorted_indices]

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
x = range(len(tickers))

# Plot bars for Daily_Sharpe_Ratio
ax.bar([i - bar_width*1.5 for i in x], daily_min_values, bar_width, label='Daily Sharpe Min (Clean)', color='#1E90FF')
ax.bar([i - bar_width*0.5 for i in x], daily_max_values, bar_width, label='Daily Sharpe Max (Clean)', color='#87CEFA')

# Plot bars for Transaction_Sharpe
ax.bar([i + bar_width*0.5 for i in x], trans_min_values, bar_width, label='Transaction Sharpe Min', color='#FF4500')
ax.bar([i + bar_width*1.5 for i in x], trans_max_values, bar_width, label='Transaction Sharpe Max', color='#FFA07A')

# Add global min and max lines
ax.axhline(y=global_daily_min, color='#0000FF', linestyle='--', label=f'Daily Global Min: {global_daily_min:.2f}')
ax.axhline(y=global_daily_max, color='#00B7EB', linestyle='--', label=f'Daily Global Max: {global_daily_max:.2f}')
if global_trans_min != float('inf'):
    ax.axhline(y=global_trans_min, color='#FF0000', linestyle='--', label=f'Trans Global Min: {global_trans_min:.2f}')
if global_trans_max != float('-inf'):
    ax.axhline(y=global_trans_max, color='#FF6347', linestyle='--', label=f'Trans Global Max: {global_trans_max:.2f}')

# Customize chart
ax.set_xlabel('Tickers')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Min and Max Sharpe Ratios by Ticker (Clean Data)')
ax.set_xticks(x)
ax.set_xticklabels(tickers, rotation=45, ha='right')
ax.legend()
plt.tight_layout()

# Plot histogram for iff Daily_Sharpe_Ratio (clean and raw)
iff_clean_file = os.path.join(folder_path, 'IFF_clean_data.csv')
iff_raw_file = os.path.join(folder_path, 'IFF_raw_data.csv')
if os.path.exists(iff_clean_file) and os.path.exists(iff_raw_file):
    df_iff_clean = pd.read_csv(iff_clean_file)
    df_iff_raw = pd.read_csv(iff_raw_file)
    df_IFF_raw = sharpe_calculator.transform(df_iff_raw)
    
    plt.figure(figsize=(12, 6))
    
    # Histogram for clean data
    plt.subplot(1, 2, 1)
    plt.hist(df_iff_clean['Daily_Sharpe_Ratio'].dropna(), bins=50, color='#1E90FF', alpha=0.7)
    plt.title('iff Daily_Sharpe_Ratio (Clean)')
    plt.xlabel('Daily Sharpe Ratio')
    plt.ylabel('Frequency')
    
    # Histogram for raw data
    plt.subplot(1, 2, 2)
    plt.hist(df_iff_raw['Daily_Sharpe_Ratio'].dropna(), bins=50, color='#FF4500', alpha=0.7)
    plt.title('iff Daily_Sharpe_Ratio (Raw)')
    plt.xlabel('Daily Sharpe Ratio')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Print global min and max
print(f"Global Daily_Sharpe_Ratio Min: {global_daily_min}")
print(f"Global Daily_Sharpe_Ratio Max: {global_daily_max}")
print(f"Global Transaction_Sharpe Min: {global_trans_min if global_trans_min != float('inf') else 'N/A'}")
print(f"Global Transaction_Sharpe Max: {global_trans_max if global_trans_max != float('-inf') else 'N/A'}")

# Notify user about saved file
print(f"Detailed Sharpe Ratio statistics saved to: {os.path.join(folder_path, 'sharpe_ratio_detailed_stats.csv')}")