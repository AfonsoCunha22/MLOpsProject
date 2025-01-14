from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentModel:
    def __init__(self, model_name="tabularisai/multilingual-sentiment-analysis"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return probabilities
    
if __name__ == "__main__":
    model = SentimentModel()
    text = "I love this song!"
    probabilities = model.predict(text)
    labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    for label, probability in zip(labels, probabilities[0]):
        print(f"{label}: {probability.item():.4f}")