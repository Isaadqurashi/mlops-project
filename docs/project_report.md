# AI Stock Prediction & Analysis System - Project Report

## 1. Introduction
The **AI Stock Prediction & Analysis System** is an end-to-end machine learning solution designed to predict stock market prices and analyze market regimes in real-time. By leveraging a combination of ensemble machine learning models and unsupervised learning techniques, the system provides users with actionable insights into stock trends and volatility.

### 1.1 Problem Statement
Stock market prediction is inherently challenging due to the stochastic nature of financial data. Traditional methods often fail to capture complex non-linear patterns or adapt to changing market conditions. This project aims to address these challenges by building a robust, automated pipeline that integrates real-time data ingestion, advanced feature engineering, and ensemble modeling to improve prediction accuracy and market understanding.

### 1.2 Objectives
*   Develop an automated pipeline for fetching and processing daily stock data.
*   Implement ensemble learning models (Linear Regression, Random Forest, SVM) for price prediction.
*   Apply unsupervised learning (Clustering, PCA) to identify market volatility regimes.
*   Deploy a user-friendly interactive dashboard using Streamlit.
*   Ensure system reliability through CI/CD pipelines and automated testing.

---

## 2. System Architecture
The system follows a modular microservices-like architecture, ensuring scalability and maintainability.

### 2.1 Core Components
*   **Frontend (User Interface):** Built with **Streamlit**, providing an interactive dashboard for users to select stocks, view real-time metrics, and visualize predictions.
*   **Backend & Orchestration:**
    *   **Prefect:** Orchestrates the entire ML workflow, from data ingestion to model inference, ensuring reproducible and scheduled runs.
    *   **FastAPI:** (Integrated) Serves as the backend framework for handling API requests and model serving.
*   **Data Layer:**
    *   **Alpha Vantage API:** The primary source for real-time and historical stock market data (Daily Time Series).
    *   **Local Storage/Database:** Stores raw CSVs and processed datasets for training and inference.
*   **Notification Service:** A custom Discord notification module maintains system observability, alerting administrators of pipeline status or errors, featuring a custom DNS bypass for restricted network environments.

### 2.2 Infrastructure & DevOps
*   **Docker:** The entire application is containerized using Docker to ensure consistent environments across development and production.
*   **CI/CD Pipeline:** Hosted on **GitHub Actions**, the pipeline automatically tests the code (pytest, ruff) and deploys changes.
*   **Deployment:** The application is deployed on **Hugging Face Spaces**, providing a publicly accessible interface.

---

## 3. Methodology

### 3.1 Data Ingestion
The system utilizes the `Alpha Vantage` API to fetch daily historical data.
*   **Source:** `src/ingestion/ingest.py`
*   **Process:** The `fetch_daily_data` function retrieves `TIME_SERIES_DAILY` in CSV format, capturing open, high, low, close, and volume data for the last 100 data points (compact mode) or full history.

### 3.2 Feature Engineering
Raw data is transformed into meaningful features to capture market momentum and trends.
*   **Source:** `src/processing/features.py`
*   **Key Indicators:**
    *   **Simple Moving Average (SMA):** Calculated for 20-day and 50-day windows to identify trend direction.
    *   **Relative Strength Index (RSI):** A 14-day momentum oscillator to detect overbought or oversold conditions.
    *   **MACD (Moving Average Convergence Divergence):** Captures changes in the strength, direction, momentum, and duration of a trend.
    *   **Lagged Features:** (Implicit in time-series modeling) used to predict future values.
*   **Target Variables:**
    *   `target_direction`: Binary classification (1 if Price goes Up, 0 if Down).
    *   `target_price`: Regression target (Next day's closing price).

### 3.3 Machine Learning Models
The system employs an **Ensemble Learning** strategy to improve generalization and reduce overfitting.
*   **Regression Models:** Predict the exact future price.
    *   *Linear Regression:* Captures linear relationships.
    *   *Random Forest Regressor:* Handles non-linearities and feature interactions.
    *   *Support Vector Regressor (SVR):* Effective in high-dimensional spaces.
*   **Classification Models:** Predict the directional movement (Up/Down).
*   **Unsupervised Learning:**
    *   **PCA & Clustering:** Used to analyze market regimes, grouping market states based on volatility and price action patterns (e.g., "High Volatility", "Bullish Trend").

### 3.4 Data Validation
To ensure data quality and model reliability, the system integrates **DeepChecks**.
*   **Data Integrity:** automated checks for missing values, duplicates, and conflicting labels.
*   **Drift Detection:** Validates that the training and testing data distributions remain consistent (`train_test_validation`), alerting to potential concept drift.

---

## 4. Implementation & Testing

### 4.1 Development
The project is structured within the `src/` directory, separating concerns into `ingestion`, `processing`, `models`, and `orchestration`.

### 4.2 Quality Assurance
*   **Unit Testing:** Implemented using `pytest` (located in `tests/`) to verify individual components.
*   **Data Validation:** Integrated **DeepChecks** to perform automated integrity checks and detect data drift between training and testing datasets.
*   **Linting:** Code quality is maintained using `ruff` to enforce PEP 8 standards.
*   **Automated Workflows:**
    *   `ci.yml`: Triggers on push/pull request to `main`, running tests and linter.
    *   `deploy_to_hf.yml`: Automatically syncs the repository to Hugging Face Spaces upon successful merge.

---

## 5. Results & Conclusion
The system successfully demonstrates a complete end-to-end ML lifecycle. The Streamlit dashboard provides a seamless user experience, allowing for real-time stock analysis. The integration of Discord notifications ensures that the system is monitored effectively.

### 5.1 Key Achievements
*   Fully automated data pipeline.
*   Robust ensemble model implementation.
*   Resilient deployment on Hugging Face Spaces.
*   High code quality standards enforced via CI/CD.

This project serves as a comprehensive template for scalable financial machine learning applications.
