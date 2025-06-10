# results_analysis.py

"""
Simple script to run ensemble vs models analysis on our results.
* run on a local folder 
"""




# Ensemble_vs_Models_Analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from Helper_Functions import load_valid_tickers
import warnings
warnings.filterwarnings('ignore')
import logging
import datetime


class EnsembleAnalysis:
    """
    Comprehensive analysis class to compare ensemble methods against individual models
    and extract meaningful insights from stock prediction results.
    """
    
    def __init__(self, results_date_folder, logger=None):
        """
        Initialize the analysis with the date folder containing results.
        
        Parameters:
        -----------
        results_date_folder : str
            Path to the folder containing results for a specific date ('results/20250527')
        logger : Logger, optional
            Logger instance for logging operations
        """
        self.results_date_folder = results_date_folder
        self.logger = logger
        self.ensemble_data = {}
        self.model_data = {}
        self.valid_tickers = []
        
        # Statistics containers
        self.ensemble_stats = {}
        self.model_stats = {}
        self.comparison_results = {}
        
    def log_info(self, message):
        """ Helper method for logging """
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def load_all_results(self, tickers_file="valid_tickers_av.csv"):
        """
        Load all ensemble and model results for valid tickers.
        
        Parameters:
        -----------
        tickers_file : str
            CSV file containing valid ticker symbols
            
        Returns:
        --------
        dict : Summary of loading results
        """
        self.log_info("Loading all results...")
        
        # Load valid tickers
        self.valid_tickers = load_valid_tickers(self.logger, tickers_file)
        if not self.valid_tickers:
            self.log_info("No valid tickers found!")
            return {'status': 'error', 'message': 'No valid tickers'}
        
        loading_summary = {
            'total_tickers': len(self.valid_tickers),
            'ensemble_loaded': 0,
            'model_loaded': 0,
            'missing_files': [],
            'loading_errors': []
        }
        
        for ticker in self.valid_tickers:
            try:
                # Try to load ensemble results
                ensemble_file = os.path.join(self.results_date_folder, f"{ticker}_results.csv")
                if os.path.exists(ensemble_file):
                    ensemble_df = pd.read_csv(ensemble_file, index_col=0)
                    self.ensemble_data[ticker] = ensemble_df
                    loading_summary['ensemble_loaded'] += 1
                else:
                    loading_summary['missing_files'].append(f"{ticker}_results.csv")
                
                # Try to load model training results
                model_file = os.path.join(self.results_date_folder, f"{ticker}_training_validation_results.csv")
                if os.path.exists(model_file):
                    model_df = pd.read_csv(model_file)
                    self.model_data[ticker] = model_df
                    loading_summary['model_loaded'] += 1
                else:
                    loading_summary['missing_files'].append(f"{ticker}_training_validation_results.csv")
                    
            except Exception as e:
                error_msg = f"Error loading {ticker}: {str(e)}"
                loading_summary['loading_errors'].append(error_msg)
                self.log_info(error_msg)
        
        self.log_info(f"Loading Summary:")
        self.log_info(f"- Total tickers: {loading_summary['total_tickers']}")
        self.log_info(f"- Ensemble files loaded: {loading_summary['ensemble_loaded']}")
        self.log_info(f"- Model files loaded: {loading_summary['model_loaded']}")
        self.log_info(f"- Missing files: {len(loading_summary['missing_files'])}")
        self.log_info(f"- Loading errors: {len(loading_summary['loading_errors'])}")
        
        return loading_summary
    
    def calculate_ensemble_statistics(self):
        """
        Calculate statistics for ensemble methods.
        
        Returns:
        --------
        dict : Dictionary containing ensemble statistics
        """
        self.log_info("Calculating ensemble statistics...")
        
        if not self.ensemble_data:
            self.log_info("No ensemble data available!")
            return {}
        
        # Metrics to analyze
        metrics = ['mse', 'rmse', 'r2', 'mae']
        ensemble_methods = ['linearly_weighted', 'equal_weighted', 'gbdt']
        
        ensemble_stats = {}
        
        # Collect all data for each ensemble method
        for method in ensemble_methods:
            ensemble_stats[method] = {metric: [] for metric in metrics}
        
        # Extract data from each ticker
        for ticker, df in self.ensemble_data.items():
            for method in ensemble_methods:
                if method in df.index:
                    for metric in metrics:
                        if metric in df.columns:
                            value = df.loc[method, metric]
                            if pd.notna(value) and np.isfinite(value):
                                ensemble_stats[method][metric].append(value)
        
        # Calculate statistics for each method and metric
        self.ensemble_stats = {}
        for method in ensemble_methods:
            self.ensemble_stats[method] = {}
            for metric in metrics:
                values = ensemble_stats[method][metric]
                if values:
                    self.ensemble_stats[method][metric] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75),
                        'values': values  # Keep raw values for further analysis
                    }
                else:
                    self.ensemble_stats[method][metric] = {
                        'count': 0, 'mean': np.nan, 'median': np.nan, 'std': np.nan,
                        'min': np.nan, 'max': np.nan, 'q25': np.nan, 'q75': np.nan,
                        'values': []
                    }
        
        return self.ensemble_stats
    
    def calculate_model_statistics(self):
        """
        Calculate statistics for individual models.
        
        Returns:
        --------
        dict : Dictionary containing model statistics
        """
        self.log_info("Calculating model statistics...")
        
        if not self.model_data:
            self.log_info("No model data available!")
            return {}
        
        # Metrics to analyze (using Average metrics from cross-validation)
        metrics = ['Average_MSE', 'Average_RMSE', 'Average_R2']
        models = ['SVR', 'XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'LSTM']
        
        model_stats = {}
        
        # Collect all data for each model
        for model in models:
            model_stats[model] = {metric: [] for metric in metrics}
        
        # Extract data from each ticker
        for ticker, df in self.model_data.items():
            for model in models:
                model_rows = df[df['Model'] == model]
                if not model_rows.empty:
                    for metric in metrics:
                        if metric in df.columns:
                            value = model_rows[metric].iloc[0]
                            if pd.notna(value) and np.isfinite(value):
                                model_stats[model][metric].append(value)
        
        # Calculate statistics for each model and metric
        self.model_stats = {}
        for model in models:
            self.model_stats[model] = {}
            for metric in metrics:
                values = model_stats[model][metric]
                if values:
                    self.model_stats[model][metric] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75),
                        'values': values  # Keep raw values for further analysis
                    }
                else:
                    self.model_stats[model][metric] = {
                        'count': 0, 'mean': np.nan, 'median': np.nan, 'std': np.nan,
                        'min': np.nan, 'max': np.nan, 'q25': np.nan, 'q75': np.nan,
                        'values': []
                    }
        
        return self.model_stats
    
    def compare_ensemble_vs_models(self):
        """
        Compare ensemble methods against individual models using statistical tests.
        
        Returns:
        --------
        dict : Comprehensive comparison results
        """
        self.log_info("Comparing ensemble methods vs individual models...")
        
        if not self.ensemble_stats or not self.model_stats:
            self.log_info("Need to calculate statistics first!")
            return {}
        
        comparison_results = {}
        
        # For each ensemble method
        ensemble_methods = ['linearly_weighted', 'equal_weighted', 'gbdt']
        models = ['SVR', 'XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'LSTM']
        
        # Metric mappings (ensemble metric -> model metric)
        metric_mappings = {
            'rmse': 'Average_RMSE',
            'mse': 'Average_MSE',
            'r2': 'Average_R2'
        }
        
        for ensemble_method in ensemble_methods:
            comparison_results[ensemble_method] = {}
            
            for ensemble_metric, model_metric in metric_mappings.items():
                comparison_results[ensemble_method][ensemble_metric] = {}
                
                ensemble_values = self.ensemble_stats[ensemble_method][ensemble_metric]['values']
                
                if not ensemble_values:
                    continue
                
                ensemble_mean = np.mean(ensemble_values)
                
                # Compare against each model
                for model in models:
                    model_values = self.model_stats[model][model_metric]['values']
                    
                    if not model_values:
                        continue
                    
                    model_mean = np.mean(model_values)
                    
                    # Calculate improvement percentage
                    if model_metric == 'Average_R2':  # Higher is better for R2
                        improvement = ((ensemble_mean - model_mean) / model_mean) * 100
                        is_better = ensemble_mean > model_mean
                    else:  # Lower is better for MSE and RMSE
                        improvement = ((model_mean - ensemble_mean) / model_mean) * 100
                        is_better = ensemble_mean < model_mean
                    
                    # Perform statistical test (Wilcoxon signed-rank test for paired samples)
                    # We need to match tickers between ensemble and model data
                    paired_ensemble, paired_model = self._get_paired_values(
                        ensemble_method, ensemble_metric, model, model_metric
                    )
                    
                    p_value = np.nan
                    test_statistic = np.nan
                    
                    if len(paired_ensemble) >= 3 and len(paired_model) >= 3:  # Minimum for statistical test
                        try:
                            test_statistic, p_value = stats.wilcoxon(
                                paired_ensemble, paired_model, alternative='two-sided'
                            )
                        except Exception as e:
                            self.log_info(f"Statistical test failed for {ensemble_method} vs {model}: {e}")
                    
                    comparison_results[ensemble_method][ensemble_metric][model] = {
                        'ensemble_mean': ensemble_mean,
                        'model_mean': model_mean,
                        'improvement_pct': improvement,
                        'is_better': is_better,
                        'p_value': p_value,
                        'test_statistic': test_statistic,
                        'significance': 'significant' if p_value < 0.05 else 'not_significant',
                        'paired_samples': len(paired_ensemble)
                    }
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def _get_paired_values(self, ensemble_method, ensemble_metric, model, model_metric):
        """
        Get paired values for the same tickers between ensemble and model results.
        
        Returns:
        --------
        tuple : (ensemble_values, model_values) for the same tickers
        """
        ensemble_values = []
        model_values = []
        
        # Find common tickers
        common_tickers = set(self.ensemble_data.keys()) & set(self.model_data.keys())
        
        for ticker in common_tickers:
            # Get ensemble value
            if (ticker in self.ensemble_data and 
                ensemble_method in self.ensemble_data[ticker].index and
                ensemble_metric in self.ensemble_data[ticker].columns):
                
                ensemble_val = self.ensemble_data[ticker].loc[ensemble_method, ensemble_metric]
                
                # Get model value
                model_df = self.model_data[ticker]
                model_rows = model_df[model_df['Model'] == model]
                
                if not model_rows.empty and model_metric in model_df.columns:
                    model_val = model_rows[model_metric].iloc[0]
                    
                    # Only include if both values are valid
                    if (pd.notna(ensemble_val) and pd.notna(model_val) and 
                        np.isfinite(ensemble_val) and np.isfinite(model_val)):
                        ensemble_values.append(ensemble_val)
                        model_values.append(model_val)
        
        return np.array(ensemble_values), np.array(model_values)
    
    def find_best_performers(self):
        """
        Identify the best performing methods overall.
        
        Returns:
        --------
        dict : Best performers for each metric
        """
        self.log_info("Finding best performers...")
        
        best_performers = {}
        
        # Metrics to analyze
        metrics = ['rmse', 'mse', 'r2']
        
        for metric in metrics:
            best_performers[metric] = {}
            
            # Collect all methods and their performance
            all_methods = {}
            
            # Add ensemble methods
            for method in ['linearly_weighted', 'equal_weighted', 'gbdt']:
                if (method in self.ensemble_stats and 
                    metric in self.ensemble_stats[method] and
                    self.ensemble_stats[method][metric]['count'] > 0):
                    all_methods[f"Ensemble_{method}"] = self.ensemble_stats[method][metric]['mean']
            
            # Add individual models
            model_metric_map = {'rmse': 'Average_RMSE', 'mse': 'Average_MSE', 'r2': 'Average_R2'}
            model_metric = model_metric_map[metric]
            
            for model in ['SVR', 'XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'LSTM']:
                if (model in self.model_stats and 
                    model_metric in self.model_stats[model] and
                    self.model_stats[model][model_metric]['count'] > 0):
                    all_methods[f"Model_{model}"] = self.model_stats[model][model_metric]['mean']
            
            # Find best performer
            if all_methods:
                if metric == 'r2':  # Higher is better for R2
                    best_method = max(all_methods.items(), key=lambda x: x[1])
                else:  # Lower is better for MSE and RMSE
                    best_method = min(all_methods.items(), key=lambda x: x[1])
                
                best_performers[metric] = {
                    'method': best_method[0],
                    'value': best_method[1],
                    'all_methods': dict(sorted(all_methods.items(), 
                                              key=lambda x: x[1], 
                                              reverse=(metric == 'r2')))
                }
        
        return best_performers
    
    def generate_summary_report(self, save_path=None):
        """
        Generate a comprehensive summary report.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the report as CSV
            
        Returns:
        --------
        dict : Complete analysis summary
        """
        self.log_info("Generating comprehensive summary report...")
        
        summary = {
            'data_overview': {
                'total_tickers_analyzed': len(set(self.ensemble_data.keys()) | set(self.model_data.keys())),
                'tickers_with_ensemble_results': len(self.ensemble_data),
                'tickers_with_model_results': len(self.model_data),
                'common_tickers': len(set(self.ensemble_data.keys()) & set(self.model_data.keys()))
            },
            'ensemble_performance': self.ensemble_stats,
            'model_performance': self.model_stats,
            'comparison_results': self.comparison_results,
            'best_performers': self.find_best_performers()
        }
        
        # Create summary tables for easier interpretation
        summary['summary_tables'] = self._create_summary_tables()
        
        if save_path:
            self._save_summary_to_csv(summary, save_path)
        
        return summary
    
    def _create_summary_tables(self):
        """Create easy-to-read summary tables."""
        
        tables = {}
        
        # Ensemble performance table
        ensemble_table = []
        for method in ['linearly_weighted', 'equal_weighted', 'gbdt']:
            if method in self.ensemble_stats:
                row = {'Method': f"Ensemble_{method}"}
                for metric in ['rmse', 'mse', 'r2']:
                    if metric in self.ensemble_stats[method]:
                        stats = self.ensemble_stats[method][metric]
                        row[f"{metric.upper()}_mean"] = stats['mean']
                        row[f"{metric.upper()}_std"] = stats['std']
                        row[f"{metric.upper()}_count"] = stats['count']
                ensemble_table.append(row)
        
        tables['ensemble_summary'] = pd.DataFrame(ensemble_table)
        
        # Model performance table
        model_table = []
        for model in ['SVR', 'XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'LSTM']:
            if model in self.model_stats:
                row = {'Model': model}
                for metric in ['Average_RMSE', 'Average_MSE', 'Average_R2']:
                    if metric in self.model_stats[model]:
                        stats = self.model_stats[model][metric]
                        row[f"{metric}_mean"] = stats['mean']
                        row[f"{metric}_std"] = stats['std']
                        row[f"{metric}_count"] = stats['count']
                model_table.append(row)
        
        tables['model_summary'] = pd.DataFrame(model_table)
        
        # Improvement table (how much better ensemble is vs each model)
        improvement_table = []
        for ensemble_method in ['linearly_weighted', 'equal_weighted', 'gbdt']:
            if ensemble_method in self.comparison_results:
                for metric in ['rmse', 'mse', 'r2']:
                    if metric in self.comparison_results[ensemble_method]:
                        for model in ['SVR', 'XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'LSTM']:
                            if model in self.comparison_results[ensemble_method][metric]:
                                comp = self.comparison_results[ensemble_method][metric][model]
                                improvement_table.append({
                                    'Ensemble_Method': ensemble_method,
                                    'Model': model,
                                    'Metric': metric.upper(),
                                    'Improvement_Percent': comp['improvement_pct'],
                                    'Is_Better': comp['is_better'],
                                    'P_Value': comp['p_value'],
                                    'Significant': comp['significance'],
                                    'Paired_Samples': comp['paired_samples']
                                })
        
        tables['improvement_summary'] = pd.DataFrame(improvement_table)
        
        return tables
    
    def _save_summary_to_csv(self, summary, save_path):
        """Save summary tables to CSV files."""
        
        base_path = save_path.replace('.csv', '')
        
        # Save ensemble summary
        if 'ensemble_summary' in summary['summary_tables']:
            summary['summary_tables']['ensemble_summary'].to_csv(
                f"{base_path}_ensemble_summary.csv", index=False
            )
        
        # Save model summary
        if 'model_summary' in summary['summary_tables']:
            summary['summary_tables']['model_summary'].to_csv(
                f"{base_path}_model_summary.csv", index=False
            )
        
        # Save improvement summary
        if 'improvement_summary' in summary['summary_tables']:
            summary['summary_tables']['improvement_summary'].to_csv(
                f"{base_path}_improvement_summary.csv", index=False
            )
        
        self.log_info(f"Summary tables saved to {base_path}_*.csv")
    
    def create_visualizations(self, save_dir=None):
        """
        Create comprehensive visualizations of the analysis results.
        
        Parameters:
        -----------
        save_dir : str, optional
            Directory to save the plots
        """
        self.log_info("Creating visualizations...")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Performance comparison boxplots
        self._plot_performance_comparison(save_dir)
        
        # 2. Improvement heatmap
        self._plot_improvement_heatmap(save_dir)
        
        # 3. Statistical significance matrix
        self._plot_significance_matrix(save_dir)
        
        # 4. Best performer ranking
        self._plot_best_performers(save_dir)
    
    def _plot_performance_comparison(self, save_dir):
        """Create boxplots comparing ensemble vs model performance."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['rmse', 'mse', 'r2']
        metric_titles = ['RMSE', 'MSE', 'R²']
        
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            data_for_plot = []
            labels = []
            
            # Add ensemble data
            for method in ['linearly_weighted', 'equal_weighted', 'gbdt']:
                if (method in self.ensemble_stats and 
                    metric in self.ensemble_stats[method] and
                    self.ensemble_stats[method][metric]['values']):
                    data_for_plot.append(self.ensemble_stats[method][metric]['values'])
                    labels.append(f"Ens_{method}")
            
            # Add model data
            model_metric_map = {'rmse': 'Average_RMSE', 'mse': 'Average_MSE', 'r2': 'Average_R2'}
            model_metric = model_metric_map[metric]
            
            for model in ['SVR', 'XGBoost', 'RandomForest', 'LSTM']:  # Show subset for clarity
                if (model in self.model_stats and 
                    model_metric in self.model_stats[model] and
                    self.model_stats[model][model_metric]['values']):
                    data_for_plot.append(self.model_stats[model][model_metric]['values'])
                    labels.append(model)
            
            if data_for_plot:
                axes[i].boxplot(data_for_plot, labels=labels)
                axes[i].set_title(f'{title} Comparison')
                axes[i].set_ylabel(title)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_improvement_heatmap(self, save_dir):
        """Create heatmap showing improvement percentages."""
        
        # Prepare data for heatmap
        improvement_data = []
        ensemble_methods = ['linearly_weighted', 'equal_weighted', 'gbdt']
        models = ['SVR', 'XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'LSTM']
        
        for ensemble_method in ensemble_methods:
            if ensemble_method in self.comparison_results:
                row = []
                for model in models:
                    # Use RMSE as the primary metric for improvement
                    if ('rmse' in self.comparison_results[ensemble_method] and
                        model in self.comparison_results[ensemble_method]['rmse']):
                        improvement = self.comparison_results[ensemble_method]['rmse'][model]['improvement_pct']
                        row.append(improvement)
                    else:
                        row.append(np.nan)
                improvement_data.append(row)
        
        if improvement_data:
            improvement_df = pd.DataFrame(
                improvement_data, 
                index=[f"Ens_{method}" for method in ensemble_methods],
                columns=models
            )
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(improvement_df, annot=True, cmap='RdYlGn', center=0, 
                       fmt='.1f', cbar_kws={'label': 'Improvement %'})
            plt.title('Ensemble vs Model Improvement (RMSE)\nPositive = Ensemble Better')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'improvement_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.show()
    
    def _plot_significance_matrix(self, save_dir):
        """Create matrix showing statistical significance."""
        
        # Prepare significance data
        significance_data = []
        ensemble_methods = ['linearly_weighted', 'equal_weighted', 'gbdt']
        models = ['SVR', 'XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'LSTM']
        
        for ensemble_method in ensemble_methods:
            if ensemble_method in self.comparison_results:
                row = []
                for model in models:
                    if ('rmse' in self.comparison_results[ensemble_method] and
                        model in self.comparison_results[ensemble_method]['rmse']):
                        p_value = self.comparison_results[ensemble_method]['rmse'][model]['p_value']
                        # Convert to significance level
                        if pd.isna(p_value):
                            sig_level = 0  # No test
                        elif p_value < 0.001:
                            sig_level = 3  # Highly significant
                        elif p_value < 0.01:
                            sig_level = 2  # Very significant
                        elif p_value < 0.05:
                            sig_level = 1  # Significant
                        else:
                            sig_level = 0  # Not significant
                        row.append(sig_level)
                    else:
                        row.append(0)
                significance_data.append(row)
        
        if significance_data:
            significance_df = pd.DataFrame(
                significance_data,
                index=[f"Ens_{method}" for method in ensemble_methods],
                columns=models
            )
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(significance_df, annot=True, cmap='Reds', 
                       cbar_kws={'label': 'Significance Level'},
                       fmt='d')
            plt.title('Statistical Significance Matrix\n0=Not Sig, 1=p<0.05, 2=p<0.01, 3=p<0.001')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'significance_matrix.png'), dpi=300, bbox_inches='tight')
            plt.show()
    
    def _plot_best_performers(self, save_dir):
        """Create bar chart of best performers."""
        
        best_performers = self.find_best_performers()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(['rmse', 'mse', 'r2']):
            if metric in best_performers and 'all_methods' in best_performers[metric]:
                methods = list(best_performers[metric]['all_methods'].keys())
                values = list(best_performers[metric]['all_methods'].values())
                
                # Take top 10 for readability
                if len(methods) > 10:
                    methods = methods[:10]
                    values = values[:10]
                
                colors = ['red' if 'Ensemble' in method else 'blue' for method in methods]
                
                axes[i].bar(range(len(methods)), values, color=colors)
                axes[i].set_xticks(range(len(methods)))
                axes[i].set_xticklabels([m.replace('Ensemble_', 'E_').replace('Model_', 'M_') 
                                        for m in methods], rotation=45)
                axes[i].set_title(f'Best {metric.upper()} Performers')
                axes[i].set_ylabel(metric.upper())
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'best_performers.png'), dpi=300, bbox_inches='tight')
        plt.show()

def run_analysis(results_date_folder, logger=None, save_results=True):
    """
    Run the complete ensemble vs models analysis.
    
    Parameters:
    -----------
    results_date_folder : str
        Path to folder containing results ('results/20250527')
    logger : Logger, optional
        Logger instance for logging operations
    save_results : bool
        Whether to save results to files
        
    Returns:
    --------
    dict : Complete analysis results
    """
    
    # Initialize analysis
    analyzer = EnsembleAnalysis(results_date_folder, logger)
    
    # Step 1: Load all results
    loading_summary = analyzer.load_all_results()
    
    if loading_summary['ensemble_loaded'] == 0 and loading_summary['model_loaded'] == 0:
        if logger:
            logger.error("No results files found! Check the results_date_folder path.")
        return {'status': 'error', 'message': 'No results files found'}
    
    # Step 2: Calculate statistics
    ensemble_stats = analyzer.calculate_ensemble_statistics()
    model_stats = analyzer.calculate_model_statistics()
    
    # Step 3: Compare ensembles vs models
    comparison_results = analyzer.compare_ensemble_vs_models()
    
    # Step 4: Generate comprehensive report
    report_path = os.path.join(results_date_folder, 'analysis_summary') if save_results else None
    summary_report = analyzer.generate_summary_report(report_path)
    
    # Step 5: Create visualizations
    viz_dir = os.path.join(results_date_folder, 'visualizations') if save_results else None
    analyzer.create_visualizations(viz_dir)
    
    # Step 6: Print key insights
    print_key_insights(analyzer, logger)
    
    return {
        'loading_summary': loading_summary,
        'ensemble_stats': ensemble_stats,
        'model_stats': model_stats,
        'comparison_results': comparison_results,
        'summary_report': summary_report,
        'analyzer': analyzer  # Return analyzer for further analysis
    }

def print_key_insights(analyzer, logger=None):
    """
    Print key insights from the analysis in a readable format.
    
    Parameters:
    -----------
    analyzer : EnsembleAnalysis
        The analyzer object with completed analysis
    logger : Logger, optional
        Logger instance for logging
    """
    
    def log_print(message):
        if logger:
            logger.info(message)
        else:
            print(message)
    
    log_print("\n" + "="*80)
    log_print("KEY INSIGHTS FROM ENSEMBLE vs MODELS ANALYSIS")
    log_print("="*80)
    
    # Data overview
    total_tickers = len(set(analyzer.ensemble_data.keys()) | set(analyzer.model_data.keys()))
    ensemble_tickers = len(analyzer.ensemble_data)
    model_tickers = len(analyzer.model_data)
    common_tickers = len(set(analyzer.ensemble_data.keys()) & set(analyzer.model_data.keys()))
    
    log_print(f"\n  DATA OVERVIEW:")
    log_print(f"   • Total tickers analyzed: {total_tickers}")
    log_print(f"   • Tickers with ensemble results: {ensemble_tickers}")
    log_print(f"   • Tickers with model results: {model_tickers}")
    log_print(f"   • Common tickers (for direct comparison): {common_tickers}")
    
    # Best performers
    if analyzer.ensemble_stats and analyzer.model_stats:
        best_performers = analyzer.find_best_performers()
        
        log_print(f"\n  BEST PERFORMERS BY METRIC:")
        for metric, data in best_performers.items():
            if 'method' in data:
                method_type = "ENSEMBLE" if "Ensemble" in data['method'] else "MODEL"
                log_print(f"   • Best {metric.upper()}: {data['method'].replace('Ensemble_', '').replace('Model_', '')} "
                         f"({method_type}) = {data['value']:.6f}")
    
    # Ensemble performance summary
    if analyzer.ensemble_stats:
        log_print(f"\n  ENSEMBLE PERFORMANCE SUMMARY:")
        for method in ['linearly_weighted', 'equal_weighted', 'gbdt']:
            if method in analyzer.ensemble_stats:
                log_print(f"     {method.upper().replace('_', ' ')}:")
                for metric in ['rmse', 'mse', 'r2']:
                    if metric in analyzer.ensemble_stats[method]:
                        stats = analyzer.ensemble_stats[method][metric]
                        if stats['count'] > 0:
                            log_print(f"      • {metric.upper()}: μ={stats['mean']:.6f}, "
                                     f"σ={stats['std']:.6f}, n={stats['count']}")
    
    # Model performance summary
    if analyzer.model_stats:
        log_print(f"\n  TOP INDIVIDUAL MODELS:")
        
        # Find top 3 models by RMSE
        model_rmse = []
        for model in analyzer.model_stats:
            if 'Average_RMSE' in analyzer.model_stats[model]:
                stats = analyzer.model_stats[model]['Average_RMSE']
                if stats['count'] > 0:
                    model_rmse.append((model, stats['mean'], stats['count']))
        
        model_rmse.sort(key=lambda x: x[1])  # Sort by RMSE (lower is better)
        
        for i, (model, rmse, count) in enumerate(model_rmse[:3]):
            log_print(f"   {i+1}. {model}: RMSE={rmse:.6f} (n={count})")
    
    # Improvement analysis
    if analyzer.comparison_results:
        log_print(f"\n  ENSEMBLE vs MODELS COMPARISON (RMSE):")
        
        for ensemble_method in ['linearly_weighted', 'equal_weighted', 'gbdt']:
            if (ensemble_method in analyzer.comparison_results and 
                'rmse' in analyzer.comparison_results[ensemble_method]):
                
                log_print(f"     {ensemble_method.upper().replace('_', ' ')} vs Individual Models:")
                
                improvements = []
                significant_wins = 0
                total_comparisons = 0
                
                for model, comp in analyzer.comparison_results[ensemble_method]['rmse'].items():
                    if comp['paired_samples'] >= 3:  # Only consider comparisons with enough data
                        total_comparisons += 1
                        improvement = comp['improvement_pct']
                        is_significant = comp['p_value'] < 0.05 if not pd.isna(comp['p_value']) else False
                        
                        improvements.append(improvement)
                        
                        if comp['is_better'] and is_significant:
                            significant_wins += 1
                        
                        status = "BETTER" if comp['is_better'] else "WORSE"
                        sig_status = " (SIG)" if is_significant else " (NS)"
                        
                        log_print(f"      • vs {model}: {improvement:+.1f}% {status}{sig_status} "
                                 f"(n={comp['paired_samples']})")
                
                if improvements:
                    avg_improvement = np.mean(improvements)
                    positive_improvements = sum(1 for x in improvements if x > 0)
                    
                    log_print(f"        Summary: Avg improvement: {avg_improvement:+.1f}%, "
                             f"Better in {positive_improvements}/{total_comparisons} cases, "
                             f"Significant wins: {significant_wins}/{total_comparisons}")
    
    # Statistical significance summary
    if analyzer.comparison_results:
        log_print(f"\n  STATISTICAL SIGNIFICANCE SUMMARY:")
        
        total_significant = 0
        total_tests = 0
        
        for ensemble_method in analyzer.comparison_results:
            if 'rmse' in analyzer.comparison_results[ensemble_method]:
                for model, comp in analyzer.comparison_results[ensemble_method]['rmse'].items():
                    if not pd.isna(comp['p_value']) and comp['paired_samples'] >= 3:
                        total_tests += 1
                        if comp['p_value'] < 0.05:
                            total_significant += 1
        
        if total_tests > 0:
            sig_percentage = (total_significant / total_tests) * 100
            log_print(f"   • Total statistical tests conducted: {total_tests}")
            log_print(f"   • Statistically significant differences: {total_significant} ({sig_percentage:.1f}%)")
            log_print(f"   • This suggests {'STRONG' if sig_percentage > 50 else 'WEAK'} evidence for ensemble superiority")
    
    # Key recommendations
    log_print(f"\n  KEY RECOMMENDATIONS:")
    
    # Find the best ensemble method
    if analyzer.ensemble_stats:
        best_ensemble_rmse = float('inf')
        best_ensemble_method = None
        
        for method in ['linearly_weighted', 'equal_weighted', 'gbdt']:
            if (method in analyzer.ensemble_stats and 
                'rmse' in analyzer.ensemble_stats[method] and
                analyzer.ensemble_stats[method]['rmse']['count'] > 0):
                rmse = analyzer.ensemble_stats[method]['rmse']['mean']
                if rmse < best_ensemble_rmse:
                    best_ensemble_rmse = rmse
                    best_ensemble_method = method
        
        if best_ensemble_method:
            log_print(f"   1. Best ensemble method: {best_ensemble_method.upper().replace('_', ' ')}")
    
    # Compare best ensemble to best model
    if analyzer.model_stats and best_ensemble_method:
        best_model_rmse = float('inf')
        best_model = None
        
        for model in analyzer.model_stats:
            if 'Average_RMSE' in analyzer.model_stats[model]:
                stats = analyzer.model_stats[model]['Average_RMSE']
                if stats['count'] > 0 and stats['mean'] < best_model_rmse:
                    best_model_rmse = stats['mean']
                    best_model = model
        
        if best_model:
            improvement = ((best_model_rmse - best_ensemble_rmse) / best_model_rmse) * 100
            if improvement > 0:
                log_print(f"   2. Ensemble outperforms best individual model ({best_model}) by {improvement:.1f}%")
            else:
                log_print(f"   2. Best individual model ({best_model}) outperforms ensemble by {-improvement:.1f}%")
    
    log_print(f"   3. Use ensemble methods when you need consistent, robust predictions")
    log_print(f"   4. Consider individual models when interpretability is crucial")
    log_print(f"   5. Monitor performance across different market conditions")
    
    log_print("\n" + "="*80)
    log_print("ANALYSIS COMPLETE")
    log_print("="*80)

# Example usage function
def example_usage():
    """
    Example of how to use the analysis system.
    """

    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"ensemble_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("ensemble_analysis")
    
    # Set your results folder path
    results_date_folder = "results/20250527"  # Update this to your actual date folder
    
    # Run the complete analysis
    results = run_analysis(
        results_date_folder=results_date_folder,
        logger=logger,
        save_results=True
    )
    
    # Access specific results
    if results.get('status') != 'error':
        analyzer = results['analyzer']
        
        # You can do additional custom analysis here
        print("\nCustom Analysis Example:")
        print("="*50)
        
        # Example: Find which ensemble method wins most often
        if analyzer.comparison_results:
            wins_by_method = {}
            for ensemble_method in ['linearly_weighted', 'equal_weighted', 'gbdt']:
                wins = 0
                total = 0
                if ensemble_method in analyzer.comparison_results and 'rmse' in analyzer.comparison_results[ensemble_method]:
                    for model, comp in analyzer.comparison_results[ensemble_method]['rmse'].items():
                        if comp['paired_samples'] >= 3:
                            total += 1
                            if comp['is_better']:
                                wins += 1
                if total > 0:
                    wins_by_method[ensemble_method] = (wins, total, wins/total*100)
            
            print("Win rates by ensemble method:")
            for method, (wins, total, percentage) in wins_by_method.items():
                print(f"  {method}: {wins}/{total} ({percentage:.1f}%)")
    
    return results


def setup_logging():
    """Set up logging for the analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"ensemble_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ensemble_analysis")



def main():
    """Main function to run the analysis."""   

    results_date_folder = "results/20250527"  # <<-- UPDATE THIS PATH AS NEEDED
    tickers_file = "valid_tickers_av.csv"

    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("STARTING ENSEMBLE vs MODELS ANALYSIS")
    logger.info("="*60)
    logger.info(f"Results folder: {results_date_folder}")
    logger.info(f"Tickers file: {tickers_file}")
    
    # Check if results folder exists
    if not os.path.exists(results_date_folder):
        logger.error(f"Results folder '{results_date_folder}' does not exist!")
        logger.error("Please check the path and make sure you have results from a previous run.")
        return
    
    # Check if tickers file exists
    if not os.path.exists(tickers_file):
        logger.error(f"Tickers file '{tickers_file}' does not exist!")
        logger.error("Please make sure the tickers file is in the current directory.")
        return
    
    try:
        # Run the comprehensive analysis
        results = run_analysis(
            results_date_folder=results_date_folder,
            logger=logger,
            save_results=True
        )
        
        if results.get('status') == 'error':
            logger.error(f"Analysis failed: {results.get('message', 'Unknown error')}")
            return
        
        # Print summary of what was created
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETE - FILES CREATED:")
        logger.info("="*60)
        
        # List the files that should have been created
        expected_files = [
            f"{results_date_folder}/analysis_summary_ensemble_summary.csv",
            f"{results_date_folder}/analysis_summary_model_summary.csv", 
            f"{results_date_folder}/analysis_summary_improvement_summary.csv",
            f"{results_date_folder}/visualizations/performance_comparison.png",
            f"{results_date_folder}/visualizations/improvement_heatmap.png",
            f"{results_date_folder}/visualizations/significance_matrix.png",
            f"{results_date_folder}/visualizations/best_performers.png"
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                logger.info(f"Created: {file_path}")
            else:
                logger.warning(f"Missing: {file_path}")
        
        # Print quick summary
        loading_summary = results.get('loading_summary', {})
        logger.info(f"\nQUICK SUMMARY:")
        logger.info(f"• Ensemble files processed: {loading_summary.get('ensemble_loaded', 0)}")
        logger.info(f"• Model files processed: {loading_summary.get('model_loaded', 0)}")
        logger.info(f"• Total tickers analyzed: {loading_summary.get('total_tickers', 0)}")
        
        logger.info(f"\n Check the '{results_date_folder}' folder for:")
        logger.info(f"   • CSV files with detailed statistics")
        logger.info(f"   • 'visualizations' subfolder with plots")
        logger.info(f"   • Analysis log file in current directory")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()