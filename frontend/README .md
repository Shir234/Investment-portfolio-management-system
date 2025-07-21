### **Financial ML Trading System - Frontend**

A professional PyQt6-based desktop interface for portfolio management and trading strategy execution.



### **Overview**

The frontend provides an intuitive, modern interface for interacting with the trading system through:

* **Portfolio Management** - Tracking of cash, holdings, and performance
* **Strategy Configuration** - Risk levels, investment amounts, and date ranges
* **Dual Trading Modes** - Automatic execution vs Semi-Automatic
* **Interactive Analytics** - Charts, performance metrics, and trade history
* **Modern** **UI** **Design** - Dark/light themes





#### **Dependencies**

##### **Core Requirements**

* **First-time setup:** Run **python setup.py** and activate the virtual environment for all dependencies
* **Before use:** Always activate the virtual environment



##### **Data Requirements**

* **CSV Input File:** All tickers results from backend pipeline (typically YYYYMMDD\_all\_tickers\_results.csv)
* **Required Columns:** Date, Ticker, Close, Buy, Sell, Actual\_Sharpe, Best\_Prediction

##### 

##### **Quick Start**

1. Environment setup:

run**: python setup.py**

* **Activate virtual environment:**
* 
**&nbsp;	Windows:** venv\\Scripts\\activate

	**Linux/Mac:** source venv/bin/activate



2\. Launch application:

run: **python frontend/front\_main.py**



3\.  Select data file:

* Choose your **all\_tickers\_results.csv** file from the backend pipeline
* File will be validated automatically



4\. Configure strategy:

* Set investment amount and risk level
* Choose trading dates and execution mode
* Execute strategy or run in windowed mode



##### 

##### **Application Interface**

###### **Portfolio Setup Tab**

* **Investment Configuration -** Initial capital, risk level (0-10 scale)
* **Date Range Selection -** Trading period with constraint validation
* **Trading Mode -** Automatic vs Semi-Automatic execution
* **Portfolio Metrics -** Live cash, holdings value, and total portfolio value



###### **Analytics Tab**

* **Performance Charts -** Predicted vs Actual Sharpe, Portfolio value over time
* **Risk Analysis -** Profit over standard deviation, win rate tracking
* **Ticker Selection -** Multi-ticker visualization with color coding



###### **Trading History Tab**

* **Order Tracking -** Complete trade history with all details
* **Filtering Options -** View all trades, buys only, or sells only
* **Export Functionality -** CSV export with current filters
* **Performance Indicators -** Color-coded success metrics





##### **Trading Modes**

###### **Automatic Mode**

Complete strategy execution without user intervention:

1. Configure strategy parameters
2. System validates inputs and constraints
3. All recommended trades executed automatically
4. Portfolio state updated and UI refreshed



###### **Semi-Automatic Mode**

Windowed trading with user approval:

1. Trading period divided into configurable windows (3-7 days)
2. Each window analyzed for trading opportunities
3. User reviews and selects trades to execute
4. Only selected trades are executed
5. Process continues to next window





##### **Key Files by Category**

###### **Application Entry**

* **front\_main.py -** Main application entry point and file selection
* **main\_window.py -** Application container and theme management
* **splash\_screen.py -** Professional startup screen



###### **User Interface**

* **input\_panel.py -** Strategy configuration and execution control
* **analysis\_dashboard.py -** Performance visualization and analytics
* **recommendation\_panel.py** - Trading history and order tracking
* **semi\_automated\_manager.py -** Windowed trading interface



###### **Data Management**

* **data\_manager.py -** CSV validation and data constraints
* **trading\_connector.py -** Frontend-backend bridge for trade execution



###### **UI Framework**

* **styles.py -** Modern theme system with dark/light modes
* **worker.py -** Background processing for responsive UI
* **wheel\_disabled\_widgets.py -** Enhanced input controls





##### **Frontend Integration**

The frontend connects to the backend through:

* **trading\_logic\_new.py** - Core trading engine integration
* **Portfolio state management** - JSON-based persistence
* **Updates** - Portfolio and performance tracking
* **Data validation** - Input constraints and error handling
