# Full_Pipeline_With_Data.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import yfinance as yf
import os
import time
from Data_Cleaning_Pipelines import create_stock_data_pipeline, create_data_cleaning_pipeline
from Feature_Selection_and_Optimization import analyze_feature_importance, evaluate_feature_sets, validate_feature_consistency
from Logging_and_Validation import log_data_stats, validate_data_quality, verify_prediction_scale
from Models_Creation_and_Training import train_and_validate_models
from Ensembles import ensemble_pipeline

def load_valid_tickers(logger, file_path="valid_tickers.csv"):
    """
    Loads valid tickers from a CSV file
    Parameters:
    - file_path (str): Path to the CSV file containing valid tickers
        
    Returns:
    - list: List of valid ticker symbols
    """
    if not os.path.exists(file_path):
        logger.error(f"Error: File {file_path} not found.")
        return []
    
    try:
        df = pd.read_csv(file_path)
        tickers = df['Ticker'].tolist()
        logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
    except Exception as e:
        logger.error(f"Error loading tickers from {file_path}: {e}")
        return []
    
# ===============================================================================
# Full Pipeline For Single Stock
# ===============================================================================
def full_pipeline_for_single_stock(data_clean, logger, date_folder, current_date, ticker_symbol, start_date, end_date, risk_free_rate = 0.02, importance_threshold=0.9):
    logger.info(f"STARTING PIPELINE FOR TICKER {ticker_symbol}")

    try:
        start_time = time.time()
        # TODO -> comment and uncomment if using the machine
        # Shir's Path G:\My Drive\Investment portfolio management system\code_results\results\predictions
        #drive_path = r"G:\My Drive\Investment portfolio management system\code_results\results\predictions/"
        # # machine's Path
        # drive_path = r"G:\.shortcut-targets-by-id\19E5zLX5V27tgCL2D8EysE2nKWTQAEUlg\Investment portfolio management system\code_results\results\predictions/" #(previous path)
        drive_path = r"G:\.shortcut-targets-by-id\19E5zLX5V27tgCL2D8EysE2nKWTQAEUlg\Investment portfolio management system\code_results\results\predictions"
        
        # Create date folder inside Google Drive path
        drive_date_folder = os.path.join(drive_path, current_date)
        # Create directory if it doesn't exist
        os.makedirs(drive_date_folder,exist_ok=True)

        # Process the loaded clean data
        logger.info(f"\n{'-'*30}\nProcessing loaded clean data for {ticker_symbol}\n{'-'*30}")
    
        # Check if we have transaction metrics
        if 'Daily_Sharpe_Ratio' not in data_clean.columns:
            logger.error(f"No Daily_Sharpe_Ratio data for {ticker_symbol}. Skipping.")
            return False
       
        # Split the data to train and test, create train and val the models
        logger.info(f"\n{'-'*30}\nSplitting data for {ticker_symbol}\n{'-'*30}")
        # Make sure we're working with numeric data only for X
        numeric_columns = data_clean.select_dtypes(include=np.number).columns.tolist()
        # Ensure 'Daily_Sharpe_Ratio' is included if it's numeric
        numeric_columns = [col for col in numeric_columns if col != 'Daily_Sharpe_Ratio']
        
        # Use only numeric columns for X
        X = data_clean[numeric_columns]
        Y = data_clean['Daily_Sharpe_Ratio']

        logger.info(f"Using {len(numeric_columns)} numeric features for modeling")
        logger.info(f"First 5 features: {numeric_columns[:5]}")

        train_size = 0.8
        split_idx = int(len(data_clean)*train_size)
        X_train_val = X.iloc[:split_idx]
        Y_train_val = Y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        Y_test = Y.iloc[split_idx:]

        # Step 1: Feature Importance Analysis
        logger.info(f"\n{'-'*30}\nAnalyzing feature importance for {ticker_symbol}\n{'-'*30}")
        feature_importance_df = analyze_feature_importance(X_train_val, Y_train_val, model_type='xgboost')
        logger.info(f"Feature Importance Results:\n{feature_importance_df.head(10)}")

        # Step 2: Pre-Filter Features Based on Cumulative Importance
        cumulative_importance = feature_importance_df['Importance'].cumsum()
        selected_features = feature_importance_df[cumulative_importance <= importance_threshold]['Feature'].tolist()

        if not selected_features:
            selected_features = feature_importance_df['Feature'].tolist()[:int(len(feature_importance_df) * 0.5)]
            logger.warning(f"No features met the importance threshold. Using top 50% of features: {len(selected_features)} features selected.")
        else:
            logger.info(f"Selected {len(selected_features)} features based on cumulative importance threshold of {importance_threshold}")

        # Subset the data to selected features
        X_train_val_filtered = X_train_val[selected_features]
        X_test_filtered = X_test[selected_features]


        # Step 3: Feature Selection with PCA on Filtered Features
        logger.info(f"\n{'-'*30}\nPerforming feature selection for {ticker_symbol} with filtered features\n{'-'*30}")
        feature_results = evaluate_feature_sets(X_train_val_filtered, Y_train_val)  # Pass filtered data
        logger.info("\nFeature Selection Method Comparison:")
        logger.info(f"{feature_results['average_results'][['Feature_Method', 'Num_Features', 'RMSE', 'R2']]}")
        
        # Select best feature set
        best_method = feature_results['best_method']
        best_features = feature_results['best_features']
        logger.info(f"\nBest feature set: {best_method} with {best_features['components_selected']} components")
            
        pca = best_features['pca']
        scaler = best_features['scaler']

        # Ensure scaler is fitted on filtered features
        scaler.fit(X_train_val_filtered)  # Re-fit scaler on selected features
        # Transform training data
        X_train_val_scaled = scaler.transform(X_train_val_filtered)
        X_train_val_selected = pca.transform(X_train_val_scaled)
        # Transform test data
        X_test_scaled = scaler.transform(X_test_filtered)
        X_test_selected = pca.transform(X_test_scaled)
        
        # Create dataframes with component names for easier tracking
        component_names = [f'PC{i+1}' for i in range(best_features['components_selected'])]
        X_train_val_selected = pd.DataFrame(X_train_val_selected, index=X_train_val.index, columns=component_names)
        X_test_selected = pd.DataFrame(X_test_selected, index=X_test.index, columns=component_names)

        # Validate feature consistency
        logger.info(f"\n{'-'*30}\nValidating feature consistency\n{'-'*30}")
        validate_feature_consistency(X_train_val_filtered, X_train_val_selected, best_features)
        validate_feature_consistency(X_test_filtered, X_test_selected, best_features)

        # Model training
        logger.info(f"\n{'-'*30}\nTraining models for {ticker_symbol}\n{'-'*30}")
        train_results = train_and_validate_models(logger, X_train_val_selected, Y_train_val, current_date, ticker_symbol, date_folder)
        model_results = train_results['model_results']
        target_scaler = train_results['target_scaler']
        feature_scaler = train_results['feature_scaler']
 
        # Ensemble prediction
        logger.info(f"\n{'-'*30}\nRunning ensemble methods for {ticker_symbol}\n{'-'*30}")
        ensemble_results = ensemble_pipeline(logger, model_results, X_train_val_selected, X_test_selected, Y_train_val, Y_test, target_scaler, feature_scaler)
        
        # Save results
        try:
            logger.info(f"\n{'-'*30}\nSaving results for {ticker_symbol}\n{'-'*30}")
            df = pd.DataFrame.from_dict(ensemble_results, orient='index')
            df.index.name = 'Method Name'
            df.to_csv(f'{date_folder}/{ticker_symbol}_results.csv')
            df.to_csv(os.path.join(drive_date_folder, f"{ticker_symbol}_results.csv"))

            # Find best prediction and get its length
            best_method = min(ensemble_results.items(), key=lambda x: x[1]['rmse'])[0]
            best_prediction = ensemble_results[best_method]['prediction']
            prediction_length = len(best_prediction)  # Get actual prediction length

            # Use the same index as X_test but only up to the prediction length
            results_index = X_test.index[:prediction_length]
            # Create the DataFrame with all aligned data - trimming all data to match prediction length
            # Get the original columns that we need for the results dataframe
            results_df = pd.DataFrame({
                'Ticker': ticker_symbol,
                'Close': data_clean.loc[X_test.index[:prediction_length], 'Close'],
                'Buy': data_clean.loc[X_test.index[:prediction_length], 'Buy'],
                'Sell': data_clean.loc[X_test.index[:prediction_length], 'Sell'],
                'Actual_Sharpe': Y_test.iloc[:prediction_length],
                'Best_Prediction': best_prediction
            }, index=results_index)

            results_df.index = pd.to_datetime(results_index)
            results_df.index.name = "Date"
            # results_df.index.name = "Date"
            results_df.to_csv(f'{date_folder}/{ticker_symbol}_ensemble_prediction_results.csv')
            results_df.to_csv(os.path.join(drive_date_folder, f"{ticker_symbol}_ensemble_prediction_results.csv"))
            logger.info(f"Saved results for {ticker_symbol} to Google Drive dated folder")
            
            # Verify final prediction scale
            verify_prediction_scale(logger, Y_test.iloc[:prediction_length], best_prediction, f"{ticker_symbol} best ensemble method")


        except Exception as e:
            logger.error(f"Error saving results to Google Drive: {e}")
            # Added fallback saving option for emergencies
            try:
                # Try with a simpler approach if the dataframe creation failed
                np.savetxt(os.path.join(current_date, f"{ticker_symbol}_best_prediction.csv"),
                           best_prediction, delimiter=",", header="Prediction")
                
                results_df.to_csv(os.path.join(current_date, f"{ticker_symbol}_ensemble_prediction_results.csv"))
                logger.info(f"Saved simplified prediction to local folder")
            except Exception as e2:
                logger.error(f"Even simple saving failed: {e2}")
                
        end_time = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"COMPLETED PIPELINE FOR TICKER {ticker_symbol} in {end_time - start_time:.2f} seconds")
        logger.info(f"{'='*50}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in pipeline for {ticker_symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
# ===============================================================================
# Running the -> Full Pipeline For Single Stock
# ===============================================================================
def run_pipeline(logger, date_folder, current_date, tickers_file="valid_tickers.csv", start_date="2013-01-01", end_date="2024-01-01"):
    """
    Main function to run the complete pipeline
    
    Parameters:
    - tickers_file: CSV file with ticker symbols
    - start_date: Start date for historical data
    - end_date: End date for historical data
    
    Returns:
    - dict: Summary of processing results
    """
    logger.info(f"Starting stock prediction pipeline with data from {start_date} to {end_date}")
    
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

    logger.info(f"Loaded {len(valid_tickers)} valid tickers.")

    base_directory = "results" 
    clean_data_date_folder = "20250513"
    date_path = os.path.join(base_directory, clean_data_date_folder)

    if not os.path.exists(date_path):
        logger.error(f"Folder path '{date_path}' does not exist.")
        return {'status': 'error', 'message': 'Clean data folder not found'}

    

    # Process each ticker
    for i, ticker in enumerate(valid_tickers):
        logger.info(f"\nProcessing ticker {i+1}/{len(valid_tickers)}: {ticker}")
        
        try:
            # read csv -> {ticker}_clean_data: send clean data to full pipelin
            ticker_csv_path = os.path.join(date_path, f"{ticker}_clean_data.csv")

            # Check if the file exists
            if not os.path.exists(ticker_csv_path):
                logger.warning(f"File not found: {ticker_csv_path}")
                results['failed'] += 1
                results['tickers_processed'].append({'ticker': ticker, 'status': 'file_not_found'})
                continue

            # Read the CSV file - use date parsing to handle the index correctly
            #ticker_clean_data = pd.read_csv(ticker_csv_path, parse_dates=True)
            ticker_clean_data = pd.read_csv(ticker_csv_path, parse_dates=['Date'], index_col='Date')
            if not isinstance(ticker_clean_data.index, pd.DatetimeIndex):
                logger.warning(f"Index for {ticker} is not a DatetimeIndex. Converting to datetime.")
                ticker_clean_data.index = pd.to_datetime(ticker_clean_data.index)

            logger.info(f"Successfully loaded clean data for {ticker}")
            # Debug information about the loaded data
            logger.info(f"Data shape: {ticker_clean_data.shape}")
            logger.info(f"Columns: {ticker_clean_data.columns.tolist()[:5]}...")
            
            success = full_pipeline_for_single_stock(ticker_clean_data, logger, date_folder, current_date, ticker, start_date, end_date)
                        
            if success:
                logger.info(f"Successfully processed {ticker}")
                results['successful'] += 1
                results['tickers_processed'].append({'ticker': ticker, 'status': 'success'})
            else:
                logger.warning(f"Failed to process {ticker}")
                results['failed'] += 1
                results['tickers_processed'].append({'ticker': ticker, 'status': 'failed'})
        
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

