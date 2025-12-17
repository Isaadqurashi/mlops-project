# Codebase Explanation & Walkthrough

This document provides a detailed technical explanation of the "AI Stock Prediction System" (Stockker). Use this to understand the underlying logic and architecture.

## 1. System Architecture (The Big Picture)
The system operates as a **Microservices-based Pipeline** with four distinct stages:
1.  **Ingestion Layer**: Fetches raw market data (Open, High, Low, Close, Volume) from Alpha Vantage.
2.  **Processing Layer**: Transforms raw data into technical indicators (Features).
3.  **Training Orchestration**: A Prefect pipeline that trains, evaluates, and saves models.
4.  **Inference API**: A FastAPI server that loads these models to provide real-time predictions.

## 2. Key Components Explained

### A. Data Ingestion & Processing
**File:** `src/processing/features.py`
Feature engineering is critical for financial ML. We treat the market data as a time-series problem.
-   **SMA (Simple Moving Average)**: Calculates the trend over 20 and 50 days.
-   **RSI (Relative Strength Index)**: A momentum oscillator (0-100) to identify overbought/oversold conditions.
-   **MACD (Moving Average Convergence Divergence)**: Tracks momentum changes.
-   **Target Variables**:
    -   `target_price`: Next day's closing price (Regression).
    -   `target_direction`: 1 (Up) or 0 (Down) for next day (Classification).

### B. Machine Learning Models (Ensemble Approach)
**File:** `src/models/train.py`
Instead of relying on a single algorithm, we use **Ensemble Learning** for robustness.

#### 1. Regression (Predicting Price)
We use a **Voting Regressor**, which averages predictions from three models:
-   **Linear Regression**: Captures simple linear trends.
-   **Random Forest Regressor**: Captures complex, non-linear patterns (100 Decision Trees).
-   **SVR (Support Vector Regressor)**: Uses an RBF kernel to find the optimal hyperplane in high-dimensional space.
*Why?* Combining these reduces the variance and error of any single model.

#### 2. Classification (Predicting Direction)
We use a **Voting Classifier** (Soft Voting) combining:
-   **Random Forest Classifier**: Robust against overfitting.
-   **SVC (Support Vector Classifier)**: Good at separating classes with clear margins.
*Soft Voting* means we average the *probabilities* of each model class, not just their final votes, leading to more nuanced predictions.

#### 3. Unsupervised Learning (Market Analysis)
-   **K-Means Clustering**: Groups market days into 3 clusters (e.g., Low Volatility, High Volatility) based on volatility and RSI.
-   **PCA**: Reduces our 4 dimensions (SMA20, SMA50, RSI, MACD) into 2 principal components for 2D visualization.

### C. Orchestration (The Pipeline)
**File:** `src/orchestration/flows.py`
We use **Prefect** to manage the workflow.
-   **`main_pipeline`**: Loops through our stock list (`AAPL`, `GOOGL`, `MSFT`, `AMZN`, `TSLA`, `NVDA`).
-   For each stock, it sequentially runs: Fetch -> Process -> Train -> Evaluate -> Notify Discord.
-   **Error Handling**: If one stock fails, the pipeline logs the error (via Discord) and continues to the next, ensuring resilience.

### D. The API (Model Serving)
**File:** `src/api/main.py`
-   **Dynamic Loading**: On startup (`@app.on_event("startup")`), the API scans the `models/` directory. It dynamically loads whatever models it finds (e.g., `models/NVDA/regression_model.pkl`), making the system easily extensible to new stocks without changing code.
-   **Enpoints**: Exposes REST endpoints (`/predict/price`, `/predict/direction`) that the frontend consumes.

## 3. Infrastructure & DevOps

### Docker
**File:** `Dockerfile` & `docker-compose.yml`
-   We containerize the application to ensure it runs identically on your laptop and the cloud.
-   The `docker-compose` setup includes a Postgres service, which is used **exclusively by Prefect** to store flow run history. The main app uses a **file-based system** (CSVs/PKLs) for simplicity and portability.

### CI/CD (GitHub Actions)
**File:** `.github/workflows/deploy_to_hf.yml`
-   On every push to `main`, GitHub Actions automatically:
    1.  Runs `pytest` to verify code correctness.
    2.  Pushes the code to the Hugging Face Space, triggering a new deployment.

## 4. Why This Architecture?
-   **Modularity**: Separation of concerns (Ingestion vs Training vs Serving) makes debugging easy.
-   **Scalability**: Adding a new stock (like NVDA) only required adding a string to the list; the pipeline handled the rest.
-   **Reliability**: Ensembles prevent "putting all eggs in one basket" model-wise.
