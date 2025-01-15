# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import typer
from typing import Optional
from model import SentimentModel
from data import load_datasets
from torch.utils.data import Subset


app = typer.Typer(help="CLI for training the sentiment model.")

@app.command()
def train(
    processed_dir: str = typer.Argument(..., help="Path to directory containing processed data tensors."),
    model_name: str = typer.Option("tabularisai/multilingual-sentiment-analysis", help="Name of the pre-trained model."),
    num_labels: int = typer.Option(5, help="Number of labels for the model."),
    batch_size: int = typer.Option(8, help="Batch size for training."),
    epochs: int = typer.Option(2, help="Number of training epochs."),
    lr: float = typer.Option(1e-5, help="Learning rate.")
):
    """
    Simple training script for the sentiment model.
    
    Assumes the model was originally trained for 5 labels:
    [ "Very Negative", "Negative", "Neutral", "Positive", "Very Positive" ]

    If your dataset has only 3 unique labels, you should consider handling label mapping
    in the data preprocessing pipeline or adjusting the model. For example,
    you could merge "Very Negative"/"Negative" into "negative" 
    and "Very Positive"/"Positive" into "positive," etc.
    """
    
    # Load datasets
    train_dataset, test_dataset = load_datasets(processed_dir)


    # # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SentimentModel(model_name=model_name, num_labels=num_labels)
    device = torch.device("mps")  # 'cpu' or optionally "mps" on Apple Silicon
    model.to(device) 

    # Set up optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        typer.echo(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    typer.echo("Training complete.")

    # Save the trained model
    save_path = f"{processed_dir}/trained_sentiment_model.pt"
    torch.save(model.state_dict(), save_path)
    typer.echo(f"Model saved to {save_path}")

if __name__ == "__main__":
    app()
