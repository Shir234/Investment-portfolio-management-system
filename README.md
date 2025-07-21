# Investment Portfolio Management System
A sophisticated machine learning system that generates daily Sharpe ratio predictions and uses them to make trading decisions for S&P 500 stocks.

## Features
### Data Preparation & Filtering
- Financial data fetching, preprocessing, and cleaning
- S&P 500 stock filtering with insider trading transaction weighting
- Technical indicator calculation (50+ indicators)

### ML Pipeline Execution
- Advanced feature engineering with PCA optimization
- Multi-model ensemble training (XGBoost, LSTM, Random Forest, etc.)
- Hyperparameter optimization and cross-validation
- Ensemble prediction generation and model validation

### Trading Strategy Execution
- Automated trading with risk management
- Portfolio state tracking and performance analysis
- Dual execution modes (Automatic vs Semi-Automatic)

### Professional Desktop Interface
- Modern PyQt6-based GUI with dark/light themes
- Interactive charts and performance visualization
- Portfolio management with live tracking

## Table of Contents
- Installation & Setup
- [Usage](#Usage)
- Data Requirements
- Trading Logic Overview
- Testing & Validation
- Key Components
- Backend Documentation
- Frontend Documentation


## Installation & Setup
### Quick Start
> Clone the repository and navigate to project directory
- cd investment-portfolio-system
> Run automated setup (creates virtual environment and installs dependencies)
- python setup.py
> Activate virtual environment
> Windows:
- venv\Scripts\activate
> Linux/Mac:
- source venv/bin/activate


### Requirements
- Python 3.8+
- Alpha Vantage API key (premium)
- Insider trading data CSV file

## Usage


Insider Trading Module: Scores stakeholder transactions using rule-based weights to optimize share allocations in trading strategies.
Data Pipeline Module: Fetches historical stock data, computes technical indicators (e.g., PSAR, MFI, MVP), generates trading signals, and ensures data quality through cleaning.
Prediction Module: Trains six machine learning models and integrates them via three ensemble methods for precise Sharpe ratio forecasts.
Portfolio & Interface Module: Executes user-driven or automated trading strategies via a PyQt5-based GUI, providing performance visualizations and trade history.




Prerequisites

Python 3.8+
Libraries: pandas, numpy, scikit-learn, yfinance, matplotlib, seaborn, tensorflow, xgboost, lightgbm, PyQt5
Input: Historical stock data (CSV or fetched via yfinance)
Environment: Local or virtual setup with sufficient memory for ML training

Installation

Clone the repository:git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Create and activate a virtual environment:python -m venv venv


Windows: venv\Scripts\activate
macOS/Linux: source venv/bin/activate


Install dependencies:pip install -r requirements.txt

If requirements.txt is unavailable, install manually:pip install pandas numpy scikit-learn yfinance matplotlib seaborn tensorflow xgboost lightgbm PyQt5



Usage

Backend Execution:

Run the main backend script:python main.py


This processes data, trains models, and generates predictions.


Frontend Execution:

Run the GUI application:python front_main.py


Select a CSV file with stock data (format: Date, Ticker, Close, Best_Prediction, optional Actual_Sharpe).
Configure portfolio settings (investment amount, risk level, date range) in the "Portfolio Setup" tab.
Click "Execute Trading Strategy" to generate recommendations.
View results in the "Recommendations" and "Dashboard" tabs.


Outputs:

Data: Processed stock data with technical indicators and predictions.
Visualizations: Charts for portfolio value, Sharpe ratios, trade actions, and performance metrics.
Recommendations: Trading suggestions (buy, sell, short, cover) based on model outputs.
Logs: Trade history and performance metrics stored for analysis.



Data Format
The system expects stock data in CSV format with:

Date: YYYY-MM-DD
Ticker: Stock symbol
Close: Closing price
Best_Prediction: Predicted Sharpe ratio
Optional: Actual_Sharpe (defaults to -1.0 if unavailable)

Data is fetched from Yahoo Finance via yfinance or provided as a CSV in the data directory.
Methodology

Data Pipeline:

Fetch historical stock data using yfinance.
Compute technical indicators: Parabolic SAR (PSAR), Money Flow Index (MFI), Moving Volatility Pattern (MVP).
Clean data by removing outliers and handling missing values.
Generate trading signals based on indicators and insider transaction scores.


Insider Trading Scoring:

Apply rule-based weights to stakeholder transactions (e.g., volume, timing) to score trading opportunities.
Integrate scores into trading strategy optimization.


Prediction:

Train six models: Support Vector Regression (SVR), XGBoost, LightGBM, Random Forest, Gradient Boosting, LSTM.
Combine predictions using three ensemble methods: linearly weighted, equal weighted, Gradient Boosting Decision Tree (GBDT).
Output: Predicted modified Sharpe ratios for trading decisions.


Portfolio Management:

Execute trades (automated or semi-automated) based on user inputs and model predictions.
Visualize performance via PyQt GUI, including portfolio value, Sharpe ratios, and trade breakdowns.



Key Results

Data Quality:  pipeline ensures clean, reliable stock data with computed indicators.
Prediction Accuracy: Ensemble methods improve Sharpe ratio forecasts
Trading Performance: System optimizes profit-to-risk-to-time ratio, with GUI enabling user-driven strategy adjustments.
User Experience: Intuitive PyQt interface provides clear visualizations and actionable recommendations.

Limitations

Relies on historical data from Yahoo Finance, which may have gaps or inaccuracies.
Model performance depends on data quality and market conditions.
Insider trading scores are rule-based and may not capture all market dynamics.

Future Work

Integrate real-time data feeds for dynamic trading.
Add advanced models (e.g., Transformer-based) for improved predictions.
Enhance GUI 
Incorporate alternative data sources (e.g., news sentiment, social media) for richer insights.
