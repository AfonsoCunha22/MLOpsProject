from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch
from prometheus_client import Counter, Histogram, Summary, make_asgi_app
import time

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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.post("/predict/")
def predict_sentiment(request: SentimentRequest):
    """
    Perform sentiment prediction on the input text.
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
    except Exception as e:
        # Increment error counter on failure
        error_counter.inc()
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # Observe latency
    request_latency.observe(time.time() - start_time)

    return {"text": request.text, "predicted_class": predicted_class, "probabilities": probs.tolist()}
