# model.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    """
    Simple wrapper around a pre-trained model for sequence classification.
    We assume the model was originally trained for 5 labels:
    [ "Very Negative", "Negative", "Neutral", "Positive", "Very Positive" ]
    
    If your dataset has only 3 labels, be sure to fine-tune your model so that
    it learns to map your dataset's labels [ "negative", "neutral", "positive" ]
    to a subset or appropriate distribution.
    """
    def __init__(self, model_name="tabularisai/multilingual-sentiment-analysis", num_labels=5):
        super().__init__()
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model. If labels are provided, returns loss along with outputs.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs  # outputs has attributes: logits, loss (if labels are passed), etc.

    def predict(self, text):
        """
        Predict sentiment probabilities for a single piece of text.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return probabilities
