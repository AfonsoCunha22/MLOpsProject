# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI application files
COPY src/sentiment_analysis /app/src/sentiment_analysis

# Set the working directory to where main.py is located
WORKDIR /app/src/sentiment_analysis

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]