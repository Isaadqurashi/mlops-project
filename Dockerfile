# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install Git so we can download pandas_ta directly
RUN apt-get update && apt-get install -y git

# This downloads the code directly without needing git installed
RUN pip install https://github.com/twopirllc/pandas-ta/archive/development.zip
# ------------------------------------------

# 1. Copy the requirements file
COPY requirements.txt .

# 2. Install dependencies from the file
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

# Install dependencies
RUN pip install "numpy<2.0" pandas
RUN pip install --no-cache-dir .

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ src/
COPY tests/ tests/

COPY app.py .

# Copy models (CRITICAL for Standalone Mode)
COPY models/ models/

# Create directories for data and reports
RUN mkdir -p data/processed reports && \
    chmod -R 777 data models reports

# Expose port (Hugging Face Requirement)
EXPOSE 7860

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
