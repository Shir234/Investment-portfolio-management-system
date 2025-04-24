# tests.py
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Import your modules
from Data_Cleaning_Pipelines import (
    create_stock_data_pipeline, create_data_cleaning_pipeline,
    DataFetcher, IndicatorCalculator, SignalCalculator, TransactionMetricsCalculator,
    MissingValueHandler, OutlierHandler, CorrelationHandler
)
from Feature_Selection_and_Optimization import (
    analyze_feature_importance, select_best_features,
    evaluate_feature_sets, validate_feature_consistency
)
from Models_Creation_and_Training import (
    create_models, train_and_validate_models, 
    train_lstm_model, train_lstm_model_with_cv
)
from Ensembles import (
    prepare_lstm_data, linearly_weighted_ensemble,
    equal_weighted_ensemble, gbdt_ensemble, ensemble_pipeline
)

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Create a simple synthetic dataset for testing
        dates = pd.date_range(start='2020-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'Open': np.random.rand(100) * 100 + 50,
            'High': np.random.rand(100) * 100 + 60,
            'Low': np.random.rand(100) * 100 + 40,
            'Close': np.random.rand(100) * 100 + 50,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
    def test_data_fetcher(self):
        """Test DataFetcher with a valid ticker"""
        fetcher = DataFetcher('AAPL', '2023-01-01', '2023-01-10')
        result = fetcher.transform(None)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        
    def test_indicator_calculator(self):
        """Test IndicatorCalculator with synthetic data"""
        calculator = IndicatorCalculator(include_prime_rate=False)
        result = calculator.transform(self.test_data)
        self.assertIn('PSAR', result.columns)
        self.assertIn('MFI', result.columns)
        self.assertIn('MVP', result.columns)
        
    def test_signal_calculator(self):
        """Test SignalCalculator with synthetic data and PSAR"""
        # First add PSAR
        calculator = IndicatorCalculator(include_prime_rate=False)
        data_with_indicators = calculator.transform(self.test_data)
        
        # Then calculate signals
        signal_calc = SignalCalculator()
        result = signal_calc.transform(data_with_indicators)
        
        self.assertIn('Signal', result.columns)
        self.assertIn('Buy', result.columns)
        self.assertIn('Sell', result.columns)
        
    def test_transaction_metrics(self):
        """Test TransactionMetricsCalculator with synthetic data"""
        # Add indicators and signals first
        data = self.test_data.copy()
        data = IndicatorCalculator(include_prime_rate=False).transform(data)
        data = SignalCalculator().transform(data)
        
        # Calculate transaction metrics
        metrics_calc = TransactionMetricsCalculator(risk_free_rate=0.02)
        result = metrics_calc.transform(data)
        
        self.assertIn('Transaction_Volatility', result.columns)
        self.assertIn('Transaction_Returns', result.columns)
        self.assertIn('Transaction_Sharpe', result.columns)
        self.assertIn('Transaction_Duration', result.columns)
        
    def test_missing_value_handler(self):
        """Test MissingValueHandler with synthetic data containing missing values"""
        # Create data with NaN values
        data = self.test_data.copy()
        data.iloc[10:15, 0] = np.nan  # Set some Open values to NaN
        data.iloc[20:25, 2] = np.nan  # Set some Low values to NaN
        
        handler = MissingValueHandler()
        result = handler.transform(data)
        
        # Check if NaNs were filled
        self.assertEqual(result.isna().sum().sum(), 0)
        
    def test_outlier_handler(self):
        """Test OutlierHandler with synthetic data containing outliers"""
        # Create data with outliers
        data = self.test_data.copy()
        data.iloc[10, 0] = 1000  # Add an outlier in Open
        data.iloc[20, 3] = 1500  # Add an outlier in Close
        
        handler = OutlierHandler()
        result = handler.transform(data)
        
        # Check if outliers were handled (values should be modified)
        self.assertNotEqual(result.iloc[10, 0], 1000)
        self.assertNotEqual(result.iloc[20, 3], 1500)
        
    def test_correlation_handler(self):
        """Test CorrelationHandler with synthetic data containing correlated features"""
        # Create highly correlated columns
        data = self.test_data.copy()
        data['Duplicate1'] = data['Close'] + np.random.normal(0, 0.1, len(data))  # Almost identical to Close
        data['Duplicate2'] = data['Open'] + np.random.normal(0, 0.1, len(data))   # Almost identical to Open
        data['Transaction_Sharpe'] = np.random.rand(len(data))  # Add target column
        
        handler = CorrelationHandler(threshold=0.9)
        handler.fit(data)
        result = handler.transform(data)
        
        # Check if correlated columns were removed, but 'Close' was kept
        self.assertIn('Close', result.columns)
        self.assertLess(len(result.columns), len(data.columns))


class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset for feature selection testing
        n_samples = 100
        n_features = 10
        
        # Generate random data
        np.random.seed(42)
        X = np.random.rand(n_samples, n_features)
        
        # Make the first 3 features actually important for the target
        y = 3*X[:, 0] + 2*X[:, 1] - 1.5*X[:, 2] + np.random.normal(0, 0.1, n_samples)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X, columns=feature_names)
        self.y = pd.Series(y, name='target')
        
    def test_analyze_feature_importance(self):
        """Test feature importance analysis with synthetic data"""
        for model_type in ['xgboost', 'randomforest', 'correlation']:
            result = analyze_feature_importance(self.X, self.y, model_type=model_type)
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), len(self.X.columns))
            self.assertIn('Feature', result.columns)
            self.assertIn('Importance', result.columns)
            
            # The top features should include one of the known important ones
            top_features = result.head(3)['Feature'].values
            important_features = ['feature_0', 'feature_1', 'feature_2']
            self.assertTrue(any(f in top_features for f in important_features))
            
    def test_select_best_features(self):
        """Test feature selection with synthetic data"""
        for method in ['importance', 'rfecv', 'lasso']:
            selected = select_best_features(self.X, self.y, method=method)
            
            self.assertIsInstance(selected, list)
            self.assertGreater(len(selected), 0)
            self.assertLessEqual(len(selected), len(self.X.columns))
            
            # Check if selected features are actually in the original dataset
            for feature in selected:
                self.assertIn(feature, self.X.columns)
                
    def test_evaluate_feature_sets(self):
        """Test evaluation of feature sets with synthetic data"""
        # Create a small test dataset to speed up the test
        X_test = self.X.iloc[:20]
        y_test = self.y.iloc[:20]
        
        result = evaluate_feature_sets(self.X, self.y, X_test, y_test)
        
        self.assertIsInstance(result, dict)
        self.assertIn('detailed_results', result)
        self.assertIn('average_results', result)
        self.assertIn('best_method', result)
        self.assertIn('best_features', result)
        
        # Check if the best features include at least one important feature
        important_features = ['feature_0', 'feature_1', 'feature_2']
        self.assertTrue(any(f in result['best_features'] for f in important_features))
        
    def test_validate_feature_consistency(self):
        """Test feature consistency validation"""
        # Create a selected dataset with fewer features
        selected_features = ['feature_0', 'feature_2', 'feature_5']
        X_selected = self.X[selected_features]
        
        # This should pass
        try:
            validate_feature_consistency(self.X, X_selected, selected_features)
            validation_passed = True
        except:
            validation_passed = False
            
        self.assertTrue(validation_passed)
        
        # Test with wrong features (should raise an error)
        wrong_features = ['feature_0', 'feature_2', 'feature_5', 'feature_999']
        with self.assertRaises(ValueError):
            validate_feature_consistency(self.X, X_selected, wrong_features)


class TestEnsembleMethods(unittest.TestCase):
    def setUp(self):
        # Create synthetic data and models for testing ensembles
        n_samples = 100
        n_features = 5
        
        # Generate random data
        np.random.seed(42)
        self.X_train = np.random.rand(n_samples, n_features)
        self.y_train = 3*self.X_train[:, 0] + 2*self.X_train[:, 1] + np.random.normal(0, 0.1, n_samples)
        
        self.X_test = np.random.rand(30, n_features)
        self.y_test = 3*self.X_test[:, 0] + 2*self.X_test[:, 1] + np.random.normal(0, 0.1, 30)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        self.X_train_df = pd.DataFrame(self.X_train, columns=feature_names)
        self.y_train_df = pd.Series(self.y_train, name='target')
        self.X_test_df = pd.DataFrame(self.X_test, columns=feature_names)
        self.y_test_df = pd.Series(self.y_test, name='target')
        
        # Create scalers
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(self.X_train)
        
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(self.y_train.reshape(-1, 1))
        
        # Mock model results (simulating train_and_validate_models output)
        from sklearn.linear_model import LinearRegression
        model1 = LinearRegression()
        model1.fit(self.X_train, self.y_train)
        pred1 = model1.predict(self.X_test)
        
        from sklearn.ensemble import RandomForestRegressor
        model2 = RandomForestRegressor(n_estimators=10, random_state=42)
        model2.fit(self.X_train, self.y_train)
        pred2 = model2.predict(self.X_test)
        
        self.models_results = {
            'Model1': {
                'best_model': model1,
                'best_model_prediction': pred1,
                'Y_val_best': self.y_test,
                'best_mse_scores': [0.1],
                'best_rmse_scores': [0.316]
            },
            'Model2': {
                'best_model': model2,
                'best_model_prediction': pred2,
                'Y_val_best': self.y_test,
                'best_mse_scores': [0.2],
                'best_rmse_scores': [0.447]
            }
        }
        
    def test_prepare_lstm_data(self):
        """Test LSTM data preparation"""
        # Test 2D input
        X_reshaped = prepare_lstm_data(self.X_train, time_steps=1)
        self.assertEqual(len(X_reshaped.shape), 3)
        self.assertEqual(X_reshaped.shape[0], self.X_train.shape[0])
        self.assertEqual(X_reshaped.shape[1], 1)  # One time step
        self.assertEqual(X_reshaped.shape[2], self.X_train.shape[1])
        
        # Test 1D input
        y_reshaped = prepare_lstm_data(self.y_train, time_steps=1)
        self.assertEqual(len(y_reshaped.shape), 3)
        self.assertEqual(y_reshaped.shape[0], self.y_train.shape[0])
        self.assertEqual(y_reshaped.shape[1], 1)  # One time step
        self.assertEqual(y_reshaped.shape[2], 1)  # One feature
        
    def test_linearly_weighted_ensemble(self):
        """Test linearly weighted ensemble with mock models"""
        prediction = linearly_weighted_ensemble(
            self.models_results, self.X_test_df, 
            self.target_scaler, self.feature_scaler
        )
        
        self.assertEqual(prediction.shape, (self.X_test.shape[0],))
        self.assertTrue(np.isfinite(prediction).all())
        
    def test_equal_weighted_ensemble(self):
        """Test equal weighted ensemble with mock models"""
        prediction = equal_weighted_ensemble(
            self.models_results, self.X_test_df,
            self.target_scaler, self.feature_scaler
        )
        
        self.assertEqual(prediction.shape, (self.X_test.shape[0],))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()