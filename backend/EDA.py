import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

# Define the RollingSharpeCalculator
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
def analyze_correlations(df, ticker, folder_path, all_features, feature_corr_dict):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    corr_matrix = df_numeric.corr()
    
    if 'Daily_Sharpe_Ratio' in corr_matrix:
        sharpe_corrs = corr_matrix['Daily_Sharpe_Ratio'].drop('Daily_Sharpe_Ratio', errors='ignore')
        sharpe_corr_file = os.path.join(folder_path, f"{ticker}_sharpe_correlations.csv")
        sharpe_corrs.to_csv(sharpe_corr_file)
        
        for feature in sharpe_corrs.index:
            if feature not in all_features:
                all_features.add(feature)
            feature_corr_dict.setdefault(feature, []).append(sharpe_corrs[feature])

# Define folder path
folder_path = r"C:\Users\Shir.Falach\Desktop\SharpSight\Investment-portfolio-management-system\backend\results\20250522"

# Initialize lists and sets
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
all_features = set()
feature_corr_dict = {}

# Initialize the calculator
sharpe_calculator = RollingSharpeCalculator(window=30, risk_free_rate=0.02)

# Iterate through files
for file_name in os.listdir(folder_path):
    if file_name.endswith('_clean_data.csv'):
        ticker = file_name.replace('_clean_data.csv', '')
        clean_file_path = os.path.join(folder_path, file_name)
        raw_file_path = os.path.join(folder_path, f"{ticker}_raw_data.csv")
        
        df_clean = pd.read_csv(clean_file_path)
        
        analyze_correlations(df_clean, ticker, folder_path, all_features, feature_corr_dict)
        
        if os.path.exists(raw_file_path):
            df_raw = pd.read_csv(raw_file_path)
            df_raw = sharpe_calculator.transform(df_raw)
        else:
            print(f"Raw data for {ticker} not found. Skipping raw data analysis.")
            df_raw = None
        
        daily_min = df_clean['Daily_Sharpe_Ratio'].min()
        daily_max = df_clean['Daily_Sharpe_Ratio'].max()
        
        trans_sharpe = df_clean[df_clean['Transaction_Sharpe'] != -1]['Transaction_Sharpe']
        trans_min = trans_sharpe.min() if not trans_sharpe.empty else float('inf')
        trans_max = trans_sharpe.max() if not trans_sharpe.empty else float('-inf')
        
        daily_nan_count = df_clean['Daily_Sharpe_Ratio'].isna().sum()
        daily_inf_count = np.isinf(df_clean['Daily_Sharpe_Ratio']).sum()
        
        if df_raw is not None:
            raw_daily_min = df_raw['Daily_Sharpe_Ratio'].min()
            raw_daily_max = df_raw['Daily_Sharpe_Ratio'].max()
            raw_daily_mean = df_raw['Daily_Sharpe_Ratio'].mean()
            raw_daily_std = df_raw['Daily_Sharpe_Ratio'].std()
            raw_daily_nan_count = df_raw['Daily_Sharpe_Ratio'].isna().sum()
            raw_daily_inf_count = np.isinf(df_raw['Daily_Sharpe_Ratio']).sum()
        else:
            raw_daily_min = raw_daily_max = raw_daily_mean = raw_daily_std = raw_daily_nan_count = raw_daily_inf_count = np.nan
        
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
        
        tickers.append(ticker)
        daily_min_values.append(daily_min)
        daily_max_values.append(daily_max)
        trans_min_values.append(trans_min)
        trans_max_values.append(trans_max)
        
        global_daily_min = min(global_daily_min, daily_min)
        global_daily_max = max(global_daily_max, daily_max)
        if trans_min != float('inf'):
            global_trans_min = min(global_trans_min, trans_min)
        if trans_max != float('-inf'):
            global_trans_max = max(global_trans_max, trans_max)

# Aggregate correlation statistics
corr_stats = []
for feature in all_features:
    corrs = feature_corr_dict[feature]
    corr_stats.append({
        'Feature': feature,
        'Mean_Corr': np.mean(corrs),
        'Max_Corr': np.max(corrs),
        'Min_Corr': np.min(corrs),
        'Std_Corr': np.std(corrs),
        'Count': len(corrs)
    })
corr_stats_df = pd.DataFrame(corr_stats)

# Save correlation stats
corr_stats_file = os.path.join(folder_path, 'feature_sharpe_correlation_stats.csv')
corr_stats_df.to_csv(corr_stats_file, index=False)
print(f"Feature correlation stats saved to: {corr_stats_file}")

# Create two heatmaps: top 20 positive and top 20 negative correlations
# Positive correlations
corr_stats_df_pos = corr_stats_df[corr_stats_df['Mean_Corr'] > 0].sort_values(by='Mean_Corr', ascending=False)
top_pos_features = corr_stats_df_pos.head(20)['Feature'].tolist()
mean_corrs_pos = corr_stats_df_pos[corr_stats_df_pos['Feature'].isin(top_pos_features)][['Feature', 'Mean_Corr']]
mean_corrs_pos = mean_corrs_pos.set_index('Feature')['Mean_Corr']
print("\nTop 20 Positive Mean Correlations with Daily_Sharpe_Ratio:")
print(mean_corrs_pos.to_string())
plt.figure(figsize=(8, 10))
sns.heatmap(mean_corrs_pos.to_frame(), annot=True, cmap='coolwarm', center=0, fmt='.2f', cbar_kws={'label': 'Mean Correlation with Daily_Sharpe_Ratio'})
plt.title('Top 20 Positive Mean Feature Correlations with Daily_Sharpe_Ratio')
plt.tight_layout()
heatmap_file_pos = os.path.join(folder_path, 'positive_mean_sharpe_correlation_heatmap.png')
plt.savefig(heatmap_file_pos)
plt.close()
print(f"Positive mean correlation heatmap saved to: {heatmap_file_pos}")

# Negative correlations
corr_stats_df_neg = corr_stats_df[corr_stats_df['Mean_Corr'] < 0].sort_values(by='Mean_Corr', ascending=True)
top_neg_features = corr_stats_df_neg.head(20)['Feature'].tolist()
mean_corrs_neg = corr_stats_df_neg[corr_stats_df_neg['Feature'].isin(top_neg_features)][['Feature', 'Mean_Corr']]
mean_corrs_neg = mean_corrs_neg.set_index('Feature')['Mean_Corr']
print("\nTop 20 Negative Mean Correlations with Daily_Sharpe_Ratio:")
if not mean_corrs_neg.empty:
    print(mean_corrs_neg.to_string())
else:
    print("No features with negative mean correlations found.")
plt.figure(figsize=(8, 10))
sns.heatmap(mean_corrs_neg.to_frame(), annot=True, cmap='coolwarm', center=0, fmt='.2f', cbar_kws={'label': 'Mean Correlation with Daily_Sharpe_Ratio'})
plt.title('Top 20 Negative Mean Feature Correlations with Daily_Sharpe_Ratio')
plt.tight_layout()
heatmap_file_neg = os.path.join(folder_path, 'negative_mean_sharpe_correlation_heatmap.png')
plt.savefig(heatmap_file_neg)
plt.close()
print(f"Negative mean correlation heatmap saved to: {heatmap_file_neg}")

# Save detailed stats
stats_df = pd.DataFrame(daily_stats)
stats_df.to_csv(os.path.join(folder_path, 'sharpe_ratio_detailed_stats.csv'), index=False)

# Limit to top 10 tickers by max Daily_Sharpe_Ratio for bar chart
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
plt.show()

# Print global min and max
print(f"Global Daily_Sharpe_Ratio Min: {global_daily_min}")
print(f"Global Daily_Sharpe_Ratio Max: {global_daily_max}")
print(f"Global Transaction_Sharpe Min: {global_trans_min if global_trans_min != float('inf') else 'N/A'}")
print(f"Global Transaction_Sharpe Max: {global_trans_max if global_trans_max != float('-inf') else 'N/A'}")
print(f"Detailed Sharpe Ratio statistics saved to: {os.path.join(folder_path, 'sharpe_ratio_detailed_stats.csv')}")