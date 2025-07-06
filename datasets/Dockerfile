# Python gRPC service for ReelRaider
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.12-slim AS production

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy the application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash reelraider && \
    chown -R reelraider:reelraider /app

# Switch to non-root user
USER reelraider

# Set Python path to include the app directory
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Expose the gRPC port
EXPOSE 50051

# Health check - simple Python import test
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import datasets.interop_service; print('Service healthy')" || exit 1

# Run the gRPC service
CMD ["python", "-m", "datasets.interop_service"]
