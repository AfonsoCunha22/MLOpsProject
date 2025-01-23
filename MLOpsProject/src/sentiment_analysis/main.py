from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from datetime import datetime
import pandas as pd
import torch
from prometheus_client import Counter, Histogram, Summary, make_asgi_app
import time
import os

# Prometheus metrics
error_counter = Counter("prediction_error", "Number of prediction errors")
request_counter = Counter("api_requests_total", "Total number of API requests")
request_latency = Histogram("request_latency_seconds", "Request latency in seconds")
review_size_summary = Summary("review_size_bytes", "Size of the reviews classified in bytes")

# Initialize FastAPI app
app = FastAPI()
app.mount("/metrics", make_asgi_app())

# Define input schema using Pydantic for validation
class SentimentRequest(BaseModel):
    text: str

# Load the model and tokenizer globally
model_name = "albert-base-v2"

try:
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {e}")

# Define log file
LOG_FILE = "prediction_logs.csv"

# Create log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "text", "predicted_class", "probabilities"]).to_csv(LOG_FILE, index=False)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.post("/predict/")
def predict_sentiment(request: SentimentRequest, background_tasks: BackgroundTasks):
    """
    Perform sentiment prediction on the input text and log the request.
    """
    # Increment request counter
    request_counter.inc()

    # Measure review size
    review_size_summary.observe(len(request.text.encode("utf-8")))

    # Start timing the request
    start_time = time.time()

    # Tokenize input text
    tokens = tokenizer(request.text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    try:
        # Perform inference
        with torch.no_grad():
            outputs = model(**tokens)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=1).item()

        # Log input and prediction asynchronously
        background_tasks.add_task(log_prediction, request.text, predicted_class, probs.tolist())

    except Exception as e:
        # Increment error counter on failure
        error_counter.inc()
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # Observe latency
    request_latency.observe(time.time() - start_time)

    return {"text": request.text, "predicted_class": predicted_class, "probabilities": probs.tolist()}

def log_prediction(text, predicted_class, probabilities):
    """
    Log the input, prediction class, probabilities, and timestamp into a CSV file.
    """
    timestamp = datetime.now().isoformat()
    new_log = pd.DataFrame([{
        "timestamp": timestamp,
        "text": text,
        "predicted_class": predicted_class,
        "probabilities": probabilities
    }])
    new_log.to_csv(LOG_FILE, mode="a", header=False, index=False)

@app.get("/drift/")
def check_drift():
    """
    Analyze data drift using Evidently.
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    # Load reference (training) data
    try:
        training_data = pd.read_csv("training_data.csv")  # Replace with your actual training data path
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Training data not found.")

    # Load current (logged) data
    try:
        current_data = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Prediction logs not found.")

    # Preprocess data for drift detection
    current_data = current_data.rename(columns={"predicted_class": "target"})
    current_data = current_data[["text", "target"]]
    training_data = training_data[["text", "target"]]  # Ensure the training data has the same format

    # Generate a drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=training_data, current_data=current_data)
    report.save_html("drift_report.html")

    return {"message": "Drift report generated. Check the 'drift_report.html' file."}
