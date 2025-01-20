# Base image
#FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
FROM python:3.12.8-slim AS base
#FROM python:3.11-slim



# Set the working directory
WORKDIR /


# Install system dependencies
RUN apt update && apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# Copy dependency files first for caching
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
# Copy application code
COPY src/ src/
COPY data/ data/



RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir



# Set the entrypoint to the training script
ENTRYPOINT ["python", "-u", "src/sentiment_analysis/train.py"]


#REPOSITORY                        TAG                       IMAGE ID       CREATED         SIZE
#kasapi/data                       v1                        b4b21eacc4ee   2 hours ago     11GB
#train                             latest                    c4021fce8e89   2 hours ago     11GB
#kasapi/sentiment_analysis_data    latest                    c4021fce8e89   2 hours ago     11GB
#kasapi/sentiment_analysis_train   latest                    b819fa62e75f   3 hours ago     11GB
