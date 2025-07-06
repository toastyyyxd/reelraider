# Python gRPC service for ReelRaider
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt* ./
RUN pip install --no-cache-dir \
    grpcio \
    grpcio-tools \
    polars \
    numpy \
    pyarrow \
    httpx \
    tenacity \
    torch \
    sentence-transformers \
    spacy \
    faiss-cpu \
    tqdm \
    scikit-learn \
    matplotlib \
    plotly \
    seaborn \
    coloredlogs

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy the application code
COPY . .

# Expose the gRPC port
EXPOSE 50051

# Run the gRPC service
CMD ["python", "-m", "datasets.interop_service"]
