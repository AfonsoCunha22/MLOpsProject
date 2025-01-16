import os
import torch
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW
import torch.nn as nn
import typer
from typing import Optional
from data import load_datasets

# Initialize Typer app
app = typer.Typer(help="CLI for training sentiment analysis model.")

@app.command()
def train(
    processed_dir: str = typer.Argument(..., help="Path to directory containing processed data tensors."),
    model_name: str = typer.Option("albert-base-v2", help="Name of the pre-trained model."),
    num_labels: int = typer.Option(3, help="Number of labels for the model (e.g., 3 for Negative, Neutral, Positive)."),
    batch_size: int = typer.Option(8, help="Batch size for training."),
    epochs: int = typer.Option(2, help="Number of training epochs."),
    lr: float = typer.Option(1e-5, help="Learning rate."),
    save_path: Optional[str] = typer.Option(None, help="Path to save the trained model.")
):
    """
    Train the sentiment model.

    Example:
    python train.py train ./data/processed --model-name albert-base-v2 --num-labels 3 --batch-size 8 --epochs 2 --lr 1e-5 --save-path ./models/trained_sentiment_model
    """
    # Load datasets
    train_dataset, test_dataset = load_datasets(processed_dir)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} testing samples.")

    # Initialize the model and tokenizer
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

    # Save the model if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, "sentiment_model.pth")
        torch.save(model.state_dict(), model_save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {model_save_path}")

@app.command()
def evaluate(
    processed_dir: str = typer.Argument(..., help="Path to directory containing processed data tensors."),
    model_name: str = typer.Option("albert-base-v2", help="Name of the pre-trained model."),
    num_labels: int = typer.Option(3, help="Number of labels for the model."),
    batch_size: int = typer.Option(8, help="Batch size for evaluation."),
    model_path: str = typer.Argument(..., help="Path to the saved model."),
):
    """
    Evaluate the sentiment model.

    Command:
    python src/sentiment_analysis/train.py evaluate ./data/processed ./models/trained_sentiment_model --model-name albert-base-v2 --num-labels 3 --batch-size 8
    """
    # Load datasets
    _, test_dataset = load_datasets(processed_dir)
    print(f"Loaded {len(test_dataset)} testing samples.")

    # Create data loader
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and tokenizer
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Load the saved model state dictionary from .pth format
    state_dict_path = os.path.join(model_path, "sentiment_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(state_dict_path, map_location=device))

    print(f"Evaluating on {device}")
    model.to(device)

    # Evaluation loop
    model.eval()
    total, correct = 0, 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            _, preds = torch.max(outputs.logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    app()