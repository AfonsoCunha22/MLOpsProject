from src.sentiment_analysis.model import SentimentModel, export_to_onnx

# Initialize sentiment analysis model
model_instance = SentimentModel()

# Export model to ONNX format, saving to models/model.onnx
export_to_onnx(model_instance, save_path="models")
