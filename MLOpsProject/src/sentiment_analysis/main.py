from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

# Initialize FastAPI app
app = FastAPI()


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
    # Tokenize input text
    tokens = tokenizer(request.text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    # Perform inference
    try:
        with torch.no_grad():
            outputs = model(**tokens)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=1).item()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return {"text": request.text, "predicted_class": predicted_class, "probabilities": probs.tolist()}
