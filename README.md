# Investment Portfolio Management System
A sophisticated machine learning system that generates daily Sharpe ratio predictions and uses them to make trading decisions for S&P 500 stocks.

## Table of Contents
- [Features](#Features)
- [Installation & Setup](#Installation-&-Setup)
- [Usage](#Usage)
- [Data Requirements](#Data-Requirements)
- [Trading Logic Overview](#Trading-Logic-Overview)
- [Testing & Validation](#Testing-&-Validation)
- [Key Components](#Key-Components)
- [Backend Documentation](#Backend-Documentation)
- [Frontend Documentation](#Frontend-Documentation)


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
### Backend Pipeline (Data → Predictions)
Run complete ML pipeline
1. cd backend
2. python Pre_filtering_and_scoring.py
3. python Filter_S&P500_Tickers.py
4. python Fetch_and_Clean_Data.py
5. python main.py
6. python ticker_combiner.py

### Frontend Application (Trading Interface)
Launch desktop application
1. python frontend/front_main.py
2. Select your all_tickers_results.csv file
3. Configure strategy and execute trades


## Data Requirements
### Input Files
- InsiderTrading_sharp.csv - Insider trading transactions (place in backend/ directory)
- Alpha Vantage API Key - Set in Filter_S&P500_Tickers.py (line 185) and Fetch_and_Clean_Data.py (line 237)

### Output Files
- data/all_tickers_results.csv - Consolidated ML predictions (required for frontend)
- Contains: Date, Ticker, Close, Buy, Sell, Actual_Sharpe, Best_Prediction


## Trading Logic Overview
The system implements a Sharpe ratio-based trading strategy:
1. Signal Generation - ML models predict daily Sharpe ratios for each stock
2. Risk Assessment - Insider trading confidence scores influence position sizing
3. Trade Execution - Buy signals (positive Sharpe) trigger purchases, sell signals (negative Sharpe) trigger sales
4. Portfolio Management - Dynamic risk management with configurable risk levels (0-10 scale)
5. Performance Tracking - P&L analysis with 70% win rate target

### Key Trading Features:
- Automatic Mode - Full strategy execution without intervention
- Semi-Automatic Mode - Windowed trading with user approval (3-7 day windows)
- Risk Management - Position sizing based on available capital and risk tolerance
- Portfolio Persistence - JSON-based state management across sessions


## Testing & Validation
Our system undergoes comprehensive testing to ensure both functional accuracy and non-functional performance requirements.
### Non-Functional Requirements Testing
**Backend Accuracy Tests validate core ML and trading performance:**
- Sharpe Prediction Accuracy - R² ≥ 0.6, RMSE ≤ 2.0 for ML model predictions
- Trading Profitability - ≥70% win rate achieved in at least one tested scenario
- Multiple Scenarios - Tests across 1-month, 3-month, 6-month, and 1-year periods

**Frontend Performance Tests ensure responsive user experience:**
- Trading Execution Speed - 95% of trades complete within 45 seconds
- Input Validation Speed - 95% of validation errors appear within 30 seconds
- UI Responsiveness - Background processing prevents interface blocking

### Running Tests
tests folder [here](tests)
> Prerequisites: Ensure data/all_tickers_results.csv exists
>
**Backend accuracy tests**
- run: pytest tests/test_backend_accuracy.py -v -s

**Frontend performance tests**
- run: pytest tests/test_frontend.py -v -s

**Run specific test categories**
- pytest tests/ -v -k "sharpe"     # ML prediction accuracy
- pytest tests/ -v -k "profit"     # Trading profitability  
- pytest tests/ -v -k "timing"     # Execution speed
- pytest tests/ -v -k "validation" # Input validation speed

Test Results: Detailed logs saved to tests/logs/ with timestamped execution reports and compliance analysis.


## Key Components
### Critical Files
- [setup.py](setup.py) - Automated environment setup and dependency management
- [backend/main.py](backend/main.py) - ML pipeline orchestrator and execution entry point
- [backend/trading_logic_new.py](backend/trading_logic_new.py) - Core trading engine and frontend-backend interface
- [frontend/front_main.py](frontend/front_main.py) - Desktop application entry point and GUI initialization

### Data Pipeline
- [backend/Data_Cleaning_Pipelines.py](backend/Data_Cleaning_Pipelines.py) - Data preprocessing and technical indicators
- [backend/Models_Creation_and_Training.py](backend/Models_Creation_and_Training.py) - Multi-model ML training engine
- [backend/Feature_Selection_and_Optimization.py](backend/Feature_Selection_and_Optimization.py) - PCA-based feature optimization
- [backend/Ensembles.py](backend/Ensembles.py) - Advanced ensemble methods and meta-learning

### Trading System
- [backend/standalone_trading_runner.py](backend/standalone_trading_runner.py) - Complete trading system wrapper
- [frontend/gui/input_panel.py](frontend/gui/input_panel.py) - Strategy configuration and execution control
- [frontend/gui/analysis_dashboard.py](frontend/gui/analysis_dashboard.py) - Performance visualization and analytics

### Testing Framework
- [tests/test_backend_accuracy.py](tests/test_backend_accuracy.py) - ML accuracy and trading profitability validation
- [tests/test_frontend.py](tests/test_frontend.py) - UI performance and responsiveness testing


## Documentation
### Backend Documentation
[Backend README](backend/README.md)- Complete ML pipeline documentation
- Data preparation and filtering workflows
- Model training and ensemble creation
- Trading logic implementation
- File organization and execution order

### Frontend Documentation
[Frontend README](frontend/README.md) - Desktop application documentation
- PyQt6 interface overview
- Trading modes and configuration
- Analytics and visualization features
- User interaction workflows

