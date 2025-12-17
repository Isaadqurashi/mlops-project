# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

# Install dependencies
RUN pip install "numpy<2.0" pandas
RUN pip install --no-cache-dir .

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
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
