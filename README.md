---
title: Nuqta - AI Stock Prediction System
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# � Nuqta | AI Market Insight & Stock Predictor

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nvvy/nuqta-Stock-predictor)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**Nuqta** is an advanced **End-to-End Machine Learning System** designed for real-time stock price prediction and market regime analysis. It leverages ensemble modeling (Linear Regression, Random Forest, SVM) and unsupervised learning to provide actionable financial insights through a premium, "Modern Islamic FinTech" aesthetic dashboard.

---

## 🚀 Live Demo

Check out the deployed application on Hugging Face Spaces:

👉 **[Nuqta Stock Predictor (Live App)](https://huggingface.co/spaces/nvvy/nuqta-Stock-predictor)**

---

## 🌟 Key Features

*   **📈 Multi-Model AI Predictions:**  
    combines **Regression** (price targets), **Classification** (trend direction), and **Clustering** (market volatility regimes) for robust decision support.
*   **⏱️ Real-Time Market Data:**  
    Fetches live stock data for global markets (USA, Pakistan, India, UK, etc.) using the **Alpha Vantage API**.
*   **🎨 Premium UI/UX:**  
    A highly responsive, glassmorphism-inspired interface built with **Streamlit** and custom CSS, featuring interactive **Plotly** charts.
*   **🔔 Smart Notifications:**  
    Integrated **Discord Alerts** to notify users of significant price movements and prediction updates.
*   **🔄 Automated MLOps Pipeline:**  
    Fully automated training and deployment pipelines using **Prefect** for orchestration and **GitHub Actions** for CI/CD.
*   **☁️ Cloud Native:**  
    Containerized with **Docker** and deployed seamlessly on cloud platforms.

---

## 🛠️ Tech Stack

*   **Frontend:** Streamlit, Plotly, HTML/CSS (Custom Styling)
*   **Backend & Logic:** Python, Scikit-Learn, Pandas, NumPy, SciPy
*   **Data Source:** Alpha Vantage, Yahoo Finance (yfinance)
*   **DevOps & MLOps:** Docker, GitHub Actions, Prefect, Hugging Face Hub

---

## � Project Structure

```bash
mlops-project/
├── .github/              # CI/CD workflows (GitHub Actions)
├── data/                 # Raw and processed datasets
├── models/               # Serialized trained models (.pkl)
├── src/                  # Source code modules
│   ├── api/              # API endpoints (if applicable)
│   ├── ingestion/        # Data fetching scripts
│   ├── processing/       # Feature engineering & preprocessing
│   ├── models/           # Model definitions & training logic
│   ├── orchestration/    # Prefect flows & monitoring scripts
├── tests/                # Unit and integration tests
├── app.py                # Main Streamlit application entry point
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Prerequisites

*   Python 3.10 or higher
*   Git
*   [Alpha Vantage API Key](https://www.alphavantage.co/) (Free key available)

### 2. Clone the Repository

```bash
git clone https://github.com/Isaadqurashi/mlops-project.git
cd mlops-project
```

### 3. Set Up Environment

Create a `.env` file in the root directory and add your API keys:

```bash
# .env
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
HF_TOKEN=your_huggingface_token  # Optional: for cloud training
WEBHOOK_URL=your_discord_webhook   # Optional: for notifications
```

### 4. Install Dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🐳 Docker Setup

You can also run the application using Docker to ensure a consistent environment.

```bash
# Build the image
docker build -t nuqta-predictor .

# Run the container
docker run -p 7860:7860 --env-file .env nuqta-predictor
```

Access the app at `http://localhost:7860`.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ by the Nuqta Team
</p>
