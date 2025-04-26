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
def full_pipeline_for_single_stock(logger, date_folder, current_date, ticker_symbol, start_date, end_date, risk_free_rate = 0.02):
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

        # Run first pipeline, fetch data
        logger.info(f"\n{'-'*30}\nFetching and processing data for {ticker_symbol}\n{'-'*30}")
        pipeline = create_stock_data_pipeline(ticker_symbol, start_date, end_date, risk_free_rate)
        data = pipeline.fit_transform(pd.DataFrame())
        #log_data_stats(logger , data, f"{ticker_symbol} raw data", log_head=True)

        if data.empty:
            logger.error(f"No data returned for ticker {ticker_symbol}")
            return False
        
        # Validate data quality
       # validate_data_quality(data, f"{ticker_symbol} raw data")

        # Run second pipeline, clean and process
        logger.info(f"\n{'-'*30}\nCleaning data for {ticker_symbol}\n{'-'*30}")
        pipeline_clean = create_data_cleaning_pipeline()
        data_clean = pipeline_clean.fit_transform(data)
        #log_data_stats(logger, data_clean, f"{ticker_symbol} cleaned data", log_head=True)
        
        # # Validate cleaned data quality
        # ##validate_data_quality(data_clean, f"{ticker_symbol} cleaned data")

        # Save locally and to Google Drive
        try:
            data_clean.to_csv(f'{date_folder}/{ticker_symbol}_clean_data.csv')
            data_clean.to_csv(os.path.join(drive_date_folder, f"{ticker_symbol}_clean_data.csv"))
            logger.info(f"Saved clean data for {ticker_symbol} to folders")
        except Exception as e:
            logger.error(f"Error saving to Google Drive: {e}")
            os.makedirs(current_date, exist_ok=True) # Create local date folder if needed
            data_clean.to_csv(os.path.join(current_date, f"{ticker_symbol}_clean_data.csv"))
        
        # Check if we have transaction metrics
        if 'Transaction_Sharpe' not in data_clean.columns:
            logger.error(f"No Transaction_Sharpe data for {ticker_symbol}. Skipping.")
            return False
        
        # Split the data to train and test, create train and val the models
        logger.info(f"\n{'-'*30}\nSplitting data for {ticker_symbol}\n{'-'*30}")
        X = data_clean.drop(columns=['Transaction_Sharpe'])
        Y = data_clean['Transaction_Sharpe']
        train_size = 0.8
        split_idx = int(len(data_clean)*train_size)

        X_train_val = X.iloc[:split_idx]
        Y_train_val = Y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        Y_test = Y.iloc[split_idx:]
        
    #     # log_data_stats(logger, X_train_val, f"{ticker_symbol} X_train_val", include_stats=False)
    #     # log_data_stats(logger, Y_train_val, f"{ticker_symbol} Y_train_val", include_stats=True)
    #     # log_data_stats(logger, X_test, f"{ticker_symbol} X_test", include_stats=False)
    #     # log_data_stats(logger, Y_test, f"{ticker_symbol} Y_test", include_stats=True)

        # Feature selection
        # logger.info(f"\n{'-'*30}\nPerforming feature selection for {ticker_symbol}\n{'-'*30}")
        # importance_df = analyze_feature_importance(X_train_val, Y_train_val)
        #logger.info(f"Top 10 features by importance:\n{importance_df.head(10)}")
        # Evaluate different feature sets including PCA
        feature_results = evaluate_feature_sets(X_train_val, Y_train_val)
        logger.info("\nFeature Selection Method Comparison:")
        logger.info(f"{feature_results['average_results'][['Feature_Method', 'Num_Features', 'RMSE', 'R2']]}")

        # Select best feature set based on results
        best_method = feature_results['best_method']
        best_features = feature_results['best_features']
        # Handle PCA differently than other feature selection methods
        #########################################################
        
        logger.info(f"\nBest feature set: {best_method} with {best_features['components_selected']} components")
        # For PCA, we need to transform the data
        pca = best_features['pca']
        scaler = best_features['scaler']
        
        # Transform training data
        X_train_val_scaled = scaler.transform(X_train_val)
        X_train_val_selected = pca.transform(X_train_val_scaled)
        
        # Transform test data
        X_test_scaled = scaler.transform(X_test)
        X_test_selected = pca.transform(X_test_scaled)
        
        # Create dataframes with component names for easier tracking
        component_names = [f'PC{i+1}' for i in range(best_features['components_selected'])]
        X_train_val_selected = pd.DataFrame(X_train_val_selected, index=X_train_val.index, columns=component_names)
        X_test_selected = pd.DataFrame(X_test_selected, index=X_test.index, columns=component_names)

        #########################################################

        # Validate feature consistency
        logger.info(f"\n{'-'*30}\nValidating feature consistency\n{'-'*30}")
        validate_feature_consistency(X_train_val, X_train_val_selected, best_features)
        validate_feature_consistency(X_test, X_test_selected, best_features)

        # log_data_stats(logger, X_train_val_selected, f"{ticker_symbol} X_train_val_selected", include_stats=False)
        # log_data_stats(logger, X_test_selected, f"{ticker_symbol} X_test_selected", include_stats=False)

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
            results_df = pd.DataFrame({
                'Ticker': ticker_symbol,
                'Close': X_test.Close.iloc[:prediction_length],
                'Buy': X_test.Buy.iloc[:prediction_length],
                'Sell': X_test.Sell.iloc[:prediction_length],
                'Actual_Sharpe': Y_test.iloc[:prediction_length],
                'Best_Prediction': best_prediction
            }, index=results_index)

        #     log_data_stats(logger, results_df, f"{ticker_symbol} final results", log_head=True)
            
            results_df.to_csv(f'{date_folder}/{ticker_symbol}_ensemble_prediction_results.csv')
            results_df.to_csv(os.path.join(drive_date_folder, f"{ticker_symbol}_ensemble_prediction_results.csv"))
            logger.info(f"Saved results for {ticker_symbol} to Google Drive dated folder")
            
            # Verify final prediction scale
            verify_prediction_scale(logger, Y_test.iloc[:prediction_length], best_prediction, f"{ticker_symbol} best ensemble method")


        except Exception as e:
            logger.error(f"Error saving results to Google Drive: {e}")
            logger.error(f"Exception details: {str(e)}")

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
    
    # Process each ticker
    for i, ticker in enumerate(valid_tickers):
        logger.info(f"\nProcessing ticker {i+1}/{len(valid_tickers)}: {ticker}")
        
        try:
            success = full_pipeline_for_single_stock(logger, date_folder, current_date, ticker, start_date, end_date)
            
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

