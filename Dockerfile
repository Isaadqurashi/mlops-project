# Build stage
# FIX: Switched to Python 3.12 to support latest pandas_ta
FROM python:3.12-slim as builder

WORKDIR /app

# 1. Install pandas_ta (pre-release version)
# We do this first to cache it.
# --pre allows beta versions, which fixes the "No matching distribution" error.
RUN pip install --no-cache-dir --pre pandas_ta

# 2. Copy requirements
COPY requirements.txt .

# 3. Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

# Install local package
RUN pip install "numpy<2.0" pandas
RUN pip install --no-cache-dir .

# Runtime stage
# FIX: Switched to Python 3.12 to match builder
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
# FIX: CRITICAL CHANGE - Updated path from python3.9 to python3.12
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ src/
COPY tests/ tests/
COPY app.py .

# Copy models
COPY models/ models/

# Create directories and permissions
RUN mkdir -p data/processed reports && \
    chmod -R 777 data models reports

# Expose port
EXPOSE 7860

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]