import pandas as pd
from sklearn.pipeline import Pipeline
import yfinance as yf
import os
import datetime
from Data_Cleaning_Pipelines import create_stock_data_pipeline, create_data_cleaning_pipeline
from Feature_Selection_and_Optimization import analyze_feature_importance, evaluate_feature_sets
from Models_Creation_and_Training import train_and_validate_models
from Ensembles import ensemble_pipeline

# ===============================================================================
### Create data directories if they don't exist
# ===============================================================================
os.makedirs('results', exist_ok=True)
# Get current date in YYYYMMDD format
current_date = datetime.datetime.now().strftime("%Y%m%d")
# Create date folder inside results
date_folder = f'results/{current_date}'
os.makedirs(date_folder, exist_ok=True)

# ===============================================================================
# Full Pipeline For Single Stock
# ===============================================================================
def full_pipeline_for_single_stock(ticker_symbol, start_date, end_date, risk_free_rate = 0.02):
    drive_path = r"G:\.shortcut-targets-by-id\19E5zLX5V27tgCL2D8EysE2nKWTQAEUlg\Investment portfolio management system\code_results\results\predictions/"
    # Create date folder inside Google Drive path
    drive_date_folder = os.path.join(drive_path, current_date)

    # Create directory if it doesn't exist
    os.makedirs(drive_date_folder,exist_ok=True)

    # Run first pipeline, fetch data
    pipeline = create_stock_data_pipeline(ticker_symbol, start_date, end_date, risk_free_rate)
    data = pipeline.fit_transform(pd.DataFrame())

    # Run second pipeline, clean and process
    pipeline_clean = create_data_cleaning_pipeline()
    data_clean = pipeline_clean.fit_transform(data)

    # Save locally
    data_clean.to_csv(f'{date_folder}/{ticker_symbol}_clean_data.csv')

    # Save to Google Drive with date folder
    try:
        data_clean.to_csv(os.path.join(drive_date_folder, f"{ticker_symbol}_clean_data.csv"))
        print(f"Saved clean data for {ticker_symbol} to Google Drive dated folder")
    except Exception as e:
        print(f"Error saving to Google Drive: {e}")
        # Fallback to local save
        os.makedirs(current_date, exist_ok=True)  # Create local date folder if needed
        data_clean.to_csv(os.path.join(current_date, f"{ticker_symbol}_clean_data.csv"))

    # Split the data to train and test, create train and val the models
    X = data_clean.drop(columns=['Transaction_Sharpe'])
    Y = data_clean['Transaction_Sharpe']
    train_size = 0.8
    split_idx = int(len(data_clean)*train_size)

    X_train_val = X.iloc[:split_idx]
    Y_train_val = Y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    Y_test = Y.iloc[split_idx:]

    # TODO -> NEW PART: OPTIMIZATION: FEATURE SELECTION!!
    # Analyze feature importance
    importance_df = analyze_feature_importance(X_train_val, Y_train_val)
    print("Feature Importance:")
    print(importance_df.head(10))
  
    # Evaluate different feature sets
    results = evaluate_feature_sets(X_train_val, Y_train_val, X_test, Y_test)
    print("\nFeature Selection Method Comparison:")
    print(results[['Method', 'Num_Features', 'MSE', 'RMSE', 'R2']])

    # Select best feature set based on results
    best_method = results.loc[results['RMSE'].idxmin(), 'Method']
    best_features = results.loc[results['RMSE'].idxmin(), 'Features']
    print(f"\nBest feature set: {best_method} with {len(best_features)} features")

    # Use these features for model training
    X_train_val_selected = X_train_val[best_features]
    X_test_selected = X_test[best_features]
   
    results = train_and_validate_models(X_train_val_selected, Y_train_val)
    # Ensemble -> Three ensembles pipeline (Linearly Weighted, Equal Weights, GBDT)
    ensemble_results = ensemble_pipeline(results, X_train_val_selected, X_test_selected, Y_train_val, Y_test)

    # Save results to Google Drive with date folder
    try:
        print(f"TRYING TO SAVE RESULTS")
        df = pd.DataFrame.from_dict(ensemble_results, orient='index')
        df.index.name = 'Method Name'
        df.to_csv(f'{date_folder}/{ticker_symbol}_results.csv')
        df.to_csv(os.path.join(drive_date_folder, f"{ticker_symbol}_results.csv"))
    
        best_method = min(ensemble_results.items(), key=lambda x: x[1]['rmse'])[0]
        best_prediction = ensemble_results[best_method]['prediction']
    
        results_df = pd.DataFrame({
            'Ticker': ticker_symbol,
            'Close': X_test.Close,
            'Buy': X_test.Buy,
            'Sell': X_test.Sell,
            'Actual_Sharpe': Y_test,
            'Best_Prediction': best_prediction
        })

        print(f"TRYING TO SAVE ENSEMBLE RESULTS")
        results_df.to_csv(f'{date_folder}/{ticker_symbol}_ensemble_prediction_results.csv')
        results_df.to_csv(os.path.join(drive_date_folder, f"{ticker_symbol}_ensemble_prediction_results.csv"))
        print(f"Saved results for {ticker_symbol} to Google Drive dated folder")
    except Exception as e:
        print(f"Error saving results to Google Drive: {e}")
        # Fallback to local save
        results_df.to_csv(os.path.join(current_date, f"{ticker_symbol}_ensemble_prediction_results.csv"))


def is_valid_ticker(ticker):
  try:
    ticker_data = yf.Ticker(ticker)
    if not ticker_data.info or 'symbol' not in ticker_data.info:
      print(f"Ticker {ticker} is not valid.")
      return False
    return True
  except Exception as e:
    print(f"Error validating ticker {ticker}: {e}")
    return False
  

def get_all_valid_tickers(tickers):
   """
   Return all valid tickers from the given list
   """
   valid_tickers = []
   total_tickers = len(tickers)

   print(f"Validating {total_tickers} tickers...")

   for i, ticker in enumerate(tickers):
      if is_valid_ticker(ticker):
         print(f"Ticker {ticker} is valid. Adding to processing list.")
         valid_tickers.append(ticker)
      else:
         print(f"Ticker {ticker} is invalid. Skipping.")
    
   print(f"Validation complete. Found {len(valid_tickers)} valid tickers out of {total_tickers}.")
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
# Get all valid tickers
valid_tickers = get_all_valid_tickers(all_tickers)

for ticker in valid_tickers:
   try:
      print(f"\nProcessing ticker: {ticker}")
      full_pipeline_for_single_stock(ticker, "2013-01-01", "2024-01-01")
      print(f"Successfully processed {ticker}")
   except Exception as e:
      print(f"Error processing ticker {ticker}: {e}")
    