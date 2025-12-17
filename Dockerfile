# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# --- FIX 1: Install pandas_ta using the "pre-release" flag ---
# This fixes the "No matching distribution" error AND the "404 ZIP" error
RUN pip install --no-cache-dir --pre pandas_ta
# -----------------------------------------------------------

COPY requirements.txt .

# Install dependencies from the file
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

# Install local package
RUN pip install "numpy<2.0" pandas
RUN pip install --no-cache-dir .

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# --- FIX 2: Updated path from python3.9 to python3.10 ---
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# --------------------------------------------------------

# Copy application code
COPY src/ src/
COPY tests/ tests/

COPY app.py .

# Copy models
COPY models/ models/

# Create directories for data and reports
RUN mkdir -p data/processed reports && \
    chmod -R 777 data models reports

# Expose port (Hugging Face Requirement)
EXPOSE 7860

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]