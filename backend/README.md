### **Financial ML Trading System - Backend**

A sophisticated machine learning pipeline for financial prediction and automated trading, combining technical analysis, insider trading intelligence, and ensemble modeling to generate actionable trading signals.



#### **Overview**

This system transforms raw financial data into profitable trading strategies through:

* Multi-model ML ensemble (XGBoost, LSTM, Random Forest, etc.)
* Insider trading analysis for position sizing
* Advanced feature engineering with PCA optimization
* Automated trading execution with risk management
* Comprehensive backtesting and performance analysis



##### **Dependencies**

###### **Core Requirements**

* **First-time setup:** Run python setup.py and activate the virtual environment for all dependencies
* **Before use:** always activate the virtual environment



###### **API Requirements**

* **Alpha Vantage API Key** (premium required for rate limits)
* **Insider trading data CSV** (InsiderTrading\_sharp.csv)





##### **Complete Workflow**

###### ***Phase 1:*** *Data Preparation \& Filtering*

1\. Pre-filtering and Scoring

run: **python Pre\_filtering\_and\_scoring.py**

* **Input:** InsiderTrading\_sharp.csv (insider trading transactions)
* **Process:** Filters S\&P 500 companies, calculates insider confidence scores
* **Output:** final\_tickers\_score.csv (ranked tickers by insider activity)



2\. Ticker Validation

run: **python Filter\_S\&P500\_Tickers.py**

* **Input**: final\_tickers\_score.csv
* **Process**: Validates ticker availability via Alpha Vantage API
* **Output**: valid\_tickers\_av.csv (API-accessible tickers only)



3\. Data Fetching \& Cleaning

run: **python Fetch\_and\_Clean\_Data.py**

* **Input**: valid\_tickers\_av.csv + Alpha Vantage API
* **Process**: Fetches OHLCV data, calculates 50+ technical indicators, applies data cleaning
* **Output**: {ticker}\_raw\_data.csv + {ticker}\_clean\_data.csv per ticker



###### ***Phase 2: ML Pipeline Execution***

4\. Main ML Pipeline

run: **python main.py**

* **Input:** Clean data files + valid\_tickers\_av.csv
* **Process:**

 	- Two-stage feature selection (importance filtering + PCA optimization)

 	- Multi-model training (6 algorithms with hyperparameter optimization)

 	- Ensemble methods (weighted, equal, GBDT meta-learning)

 	- Performance evaluation and prediction generation

* **Output:**

 	- {ticker}\_training\_validation\_results.csv (model performance)

 	- {ticker}\_ensemble\_prediction\_results.csv (final predictions)



5\. Results Consolidation

run: **python ticker\_combiner.py**

* **Input**: Individual ticker prediction files
* **Process**: Aggregates all predictions with validation
* **Output**: {date}\_all\_tickers\_results.csv



###### ***Phase 3: Trading \& Analysis***

6\. Trading Execution

run: **python standalone\_trading\_runner.py**

* **Input**: Consolidated predictions CSV
* **Process**: Executes trading strategy with risk management
* **Output**: Trading results, portfolio history, performance analysis





##### **Quick Start**

1. **Environment setup:**

* run: **python setup.py**
* Activate virtual environment:

\*\* 	Windows:\*\* venv\\Scripts\\activate

 	**Linux/Mac:** source venv/bin/activate



2\*\*. Data and API setup:\*\*

* Place InsiderTrading\_sharp.csv in 'backend' directory
* Set your Alpha Vantage API key in scripts (Filter\_S\&P500\_Tickers.py line 185, Fetch\_and\_Clean\_Data.py line 237)



3. **Run complete pipeline:**

python Pre\_filtering\_and\_scoring.py

python Filter\_S\&P500\_Tickers.py

python Fetch\_and\_Clean\_Data.py

python main.py

python ticker\_combiner.py



**4. Execute trading strategy:**

python standalone\_trading\_runner.py





##### **Key Files by Category**

###### **Core Pipeline**

* **main.py** - Main execution entry point
* **Full\_Pipeline\_With\_Data.py** - ML pipeline orchestrator
* **Data\_Cleaning\_Pipelines.py** - Data processing and technical indicators



###### **ML Components**

* **Models\_Creation\_and\_Training.py** - Multi-model training engine
* **Feature\_Selection\_and\_Optimization.py** - PCA-based feature selection
* **Ensembles.py** - Advanced ensemble methods



###### **Trading System**

* **trading\_logic\_new.py** - Core trading logic (frontend-backend interface)
* **standalone\_trading\_runner.py** - Complete trading system wrapper



###### **Utilities**

* **ticker\_combiner.py** - Results aggregation
* **results\_analysis.py** - Performance analysis and visualization
* **tests\_on\_one\_ticker.py** - Development testing utility





###### **Frontend Integration**

The **trading\_logic\_new.py** file serves as the 'frontend-backend' interface, providing:

* Trading strategy execution
* Portfolio state management
* Order generation
* Performance analytics



Ready for integration with PyQt desktop application.

#### 

