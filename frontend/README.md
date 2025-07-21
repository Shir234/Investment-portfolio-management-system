# Investment Portfolio Management System - Frontend
A PyQt6-based desktop interface for portfolio management and trading strategy execution.

## Overview
The frontend provides an intuitive, modern interface for interacting with the trading system through:
- Portfolio Management -Tracking of cash, holdings, and performance
- Strategy Configuration - Risk levels, investment amounts, and date ranges
- Dual Trading Modes - Automatic execution vs Semi-Automatic
- Interactive Analytics - Charts, performance metrics, and trade history


## Dependencies
### Core Requirements
- First-time setup: Run **python setup.py** and activate the virtual environment for all dependencies
- Before use: Always activate the virtual environment

### Data Requirements
- CSV Input File: All tickers results from backend pipeline (typically YYYYMMDD_all_tickers_results.csv)
- Required Columns: Date, Ticker, Close, Buy, Sell, Actual_Sharpe, Best_Prediction


## Quick Start
1. Environment setup:
   - run: python setup.py
   - Activate virtual environment:
     - Windows: venv\Scripts\activate
     - Linux/Mac: source venv/bin/activate
       
2. Launch application:
   - run: python frontend/front_main.py

3. Select data file:
   - Choose your all_tickers_results.csv file from the backend pipeline
   - File will be validated automatically
   
4. Configure strategy:
   - Set investment amount and risk level
   - Choose trading dates and execution mode
   - Execute strategy or run in windowed mode


## Application Interface
### Portfolio Setup Tab
- Investment Configuration - Initial capital, risk level (0-10 scale)
- Date Range Selection - Trading period with constraint validation
- Trading Mode - Automatic vs Semi-Automatic execution
- Portfolio Metrics - Live cash, holdings value, and total portfolio value

### Analytics Tab
- Performance Charts - Predicted vs Actual Sharpe, Portfolio value over time
- Risk Analysis - Profit over standard deviation, win rate tracking
- Ticker Selection - Multi-ticker visualization with color coding

### Trading History Tab
- Order Tracking - Complete trade history with all details
- Filtering Options - View all trades, buys only, or sells only
- Export Functionality - CSV export with current filters
- Performance Indicators - Color-coded success metrics


## Trading Modes
### Automatic Mode
Complete strategy execution without user intervention:
- Configure strategy parameters
- System validates inputs and constraints
- All recommended trades executed automatically
- Portfolio state updated and UI refreshed

### Semi-Automatic Mode
Windowed trading with user approval:
- Trading period divided into configurable windows (3-7 days)
- Each window analyzed for trading opportunities
- User reviews and selects trades to execute
- Only selected trades are executed
- Process continues to next window


## Key Files by Category
### Application Entry
- front_main.py - Main application entry point and file selection
- main_window.py - Application container and theme management
- splash_screen.py - Professional startup screen

### User Interface
- input_panel.py - Strategy configuration and execution control
- analysis_dashboard.py - Performance visualization and analytics
- recommendation_panel.py - Trading history and order tracking
- semi_automated_manager.py - Windowed trading interface

### Data Management
- data_manager.py - CSV validation and data constraints
- trading_connector.py - Frontend-backend bridge for trade execution

### UI Framework
- styles.py - Modern theme system with dark/light modes
- worker.py - Background processing for responsive UI
- wheel_disabled_widgets.py - Enhanced input controls


## Frontend Integration
The frontend connects to the backend through:
- trading_logic_new.py - Core trading engine integration
- Portfolio state management - JSON-based persistence
- Updates - Live portfolio and performance tracking

