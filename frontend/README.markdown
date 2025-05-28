# Investment Portfolio Management System

A PyQt-based desktop application for analyzing stock market data from a CSV file, providing trading recommendations, and visualizing portfolio performance.

## Overview
This application enables users to:
- Load stock market data from a CSV file with columns: `Date`, `Ticker`, `Close`, `Best_Prediction`, `Actual_Sharpe`.
- Configure trading parameters, including investment amount, risk level, date range, and trading mode (Automatic or Semi-automatic).
- Execute trading strategies to generate buy, sell, short, or cover orders.
- Visualize portfolio performance through interactive graphs, such as Portfolio Performance, Sharpe Distribution, and Trade Action Distribution.
- View and filter trading recommendations in a tabular format.
- Reset portfolio state to start fresh.

The application features a user-friendly interface with a splash screen, tabbed layout, and support for dark/light themes to ensure accessibility.

## Features
- **Portfolio Setup**: Input investment parameters and execute trading strategies.
- **Dashboard**: Display interactive graphs and key metrics (Total Value, Sharpe Ratio, Volatility).
- **Recommendations**: Filter and review trading orders by type (All, Buy, Sell, Cover, Short).
- **Data Management**: Efficiently load and filter large CSV datasets.
- **Error Handling**: Robust validation and user feedback for invalid inputs or data issues.
- **Performance Optimization**: Downsampling for large datasets to ensure responsive visualizations.
- **Testing**: Unit tests for non-GUI components to ensure reliability.

## Prerequisites
- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`):
  - `PyQt5`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `numpy`

## Setup
1. **Clone the Repository** (if applicable) or ensure all project files are in your working directory:
   ```
   Project:.
   │   20250415_all_tickers_results.csv
   │   front_main.py
   │   logo.JPG
   │   README.md
   │   requirements.txt
   │
   ├───data
   │       data_manager.py
   │       trading_connector.py
   │       __init__.py
   │
   ├───gui
   │       analysis_dashboard.py
   │       input_panel.py
   │       main_window.py
   │       recommendation_panel.py
   │       splash_screen.py
   │       __init__.py
   │
   ├───tests
   │       test_data_manager.py
   │       test_trading_connector.py
   │       __init__.py
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare CSV File**:
   - Ensure `20250415_all_tickers_results.csv` is in the project root or update the path in `front_main.py`.
   - The CSV must include columns: `Date`, `Ticker`, `Close`, `Best_Prediction`, `Actual_Sharpe`.
   - Example row:
     ```csv
     Date,Ticker,Close,Best_Prediction,Actual_Sharpe
     2023-10-18,AAPL,150.0,1.2,-1.0
     ```

4. **Run the Application**:
   ```bash
   python front_main.py
   ```

## Usage
1. **Launch the Application**:
   - A splash screen with the logo and "Welcome to SharpSight" appears, fading into the main window.

2. **Portfolio Setup Tab**:
   - **Investment Amount**: Enter a dollar amount (e.g., `$10,000`).
   - **Risk Level**: Set a value from 0 to 10 (scaled to 0–100 internally).
   - **Date Range**: Select start and end dates (e.g., 2023-10-18 to 2023-12-22).
   - **Trading Mode**: Choose `Automatic` (executes trades directly) or `Semi-automatic` (prompts for trade confirmation).
   - **Actions**:
     - Click **Execute Trading Strategy** to run the strategy and update the portfolio.
     - Click **Reset Portfolio** to clear `portfolio_state.json` and reset the state.
   - View the current portfolio value below the inputs.

3. **Dashboard Tab**:
   - **Graph Selection**: Choose from 11 graph types, e.g., `Portfolio Performance`, `Sharpe Distribution`, `Portfolio vs Max Possible Value`.
   - **Ticker Selection**: Select up to 5 tickers from the list for detailed analysis.
   - **Metrics**: View Total Value, Average Sharpe Ratio, and Volatility at the bottom.
   - Graphs update automatically when parameters change or tickers are selected.

4. **Recommendations Tab**:
   - **Filter Orders**: Select `All Orders`, `Buy Orders`, `Sell Orders`, `Cover Orders`, or `Short Orders`.
   - **Table**: Displays order details (Date, Ticker, Action, Shares, Price, etc.).
   - **Refresh**: Click **Refresh Recommendations** to update the table.
   - Alerts appear if no orders match the filter or parameters.

## Project Structure
- **`20250415_all_tickers_results.csv`**: Sample stock market data.
- **`front_main.py`**: Application entry point, initializes DataManager and GUI.
- **`logo.JPG`**: Logo for the splash screen.
- **`requirements.txt`**: Lists Python dependencies.
- **`portfolio_state.json`**: Stores trading orders and portfolio history.
- **`data/`**:
  - `data_manager.py`: Loads, validates, and filters CSV data.
  - `trading_connector.py`: Interfaces with trading logic for strategy execution.
  - `__init__.py`: Marks the directory as a Python package.
- **`gui/`**:
  - `input_panel.py`: Handles user inputs and strategy execution.
  - `analysis_dashboard.py`: Displays interactive graphs and metrics.
  - `recommendation_panel.py`: Shows and filters trading recommendations.
  - `main_window.py`: Main window with tabbed interface.
  - `splash_screen.py`: Displays a branded splash screen with fade-out animation.
  - `__init__.py`: Marks the directory as a Python package.
- **`tests/`**:
  - `test_data_manager.py`: Unit tests for `DataManager`.
  - `test_trading_connector.py`: Unit tests for `trading_connector`.
  - `__init__.py`: Marks the directory as a Python package.


## Performance Optimizations
- **Data Filtering**: Uses `pandas.query` for efficient data filtering in `DataManager`.
- **Visualization**: Downsampling in `AnalysisDashboard` (e.g., `plot_portfolio_performance`) to handle large datasets (limits to ~1000 points).
- **Profiling**: Logs rendering times in `AnalysisDashboard.update_visualizations` and `RecommendationPanel.update_recommendations` to monitor performance.

## Notes
- **CSV Path**: If `20250415_all_tickers_results.csv` is not in the root, update the path in `front_main.py`.
- **Themes**: Supports dark and light themes for accessibility; default is dark for better contrast.
- **Semi-automatic Mode**: Prompts for trade confirmation, enhancing user control.
- **Error Handling**: Validates inputs and provides clear error messages (e.g., invalid dates, missing CSV columns).
- **Large Datasets**: Tested for responsiveness with datasets up to 10,000 rows; adjust downsampling in `analysis_dashboard.py` for larger datasets if needed.

## Known Limitations
- Limited to CSV input with specific columns; no support for other formats without modification.
- Semi-automatic mode requires manual confirmation, which may slow workflows for frequent trades.
- Visualization performance may degrade with extremely large datasets (>50,000 rows) without further optimization.

## Future Improvements
- Add support for additional data formats (e.g., JSON, SQL).
- Implement real-time data streaming for live market analysis.
- Enhance accessibility with keyboard navigation and screen reader support.
- Expand unit tests to cover edge cases in trading logic.

## Acknowledgments
- Built with PyQt5 for cross-platform GUI development.
- Uses pandas and matplotlib for data processing and visualization.
- Designed for a final software engineering project, focusing on usability and maintainability.

For issues or contributions, please contact the project maintainer.