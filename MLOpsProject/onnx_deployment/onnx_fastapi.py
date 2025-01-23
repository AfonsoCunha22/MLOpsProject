'''
To run this FastAPI server, use the following command in terminal:
uvicorn onnx_fastapi:app --reload
'''

import numpy as np
import onnxruntime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Initialize FastAPI application
app = FastAPI()

# Load the ONNX model once at startup for better performance
# This avoids loading the model for each request
session = onnxruntime.InferenceSession("models/model.onnx")


# Define the expected input data structure using Pydantic
# This enables automatic request validation and documentation
# Example input format:
# {
#    "input_ids": [[101, 2054, 2003, ...], [101, 2516, 2004, ...]],
#    "attention_mask": [[1, 1, 1, ...], [1, 1, 1, ...]]
# }
class InputData(BaseModel):
    input_ids: List[List[int]]  # Tokenized input text
    attention_mask: List[List[int]]  # Mask indicating real tokens vs padding


# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the ONNX Model API!"}


@app.post("/predict")
def predict(input_data: InputData):
    """
    Endpoint for making predictions using an ONNX-exported Transformer model.
    
    Args:
        input_data: JSON payload containing input_ids and attention_mask
        
    Returns:
        JSON object containing model predictions
    """

    # Convert the input Python lists into NumPy arrays
    # dtype=np.int64 is required as ONNX models expect 64-bit integers
    input_ids_np = np.array(input_data.input_ids, dtype=np.int64)
    attention_mask_np = np.array(input_data.attention_mask, dtype=np.int64)

    # Run inference with the ONNX model
    # None means we want all output nodes from the model
    outputs = session.run(
        None,  # request all outputs
        {"input_ids": input_ids_np, "attention_mask": attention_mask_np},
    )

    # Convert the numpy array output to a Python list for JSON serialization
    # outputs[0] contains the model predictions (e.g., sentiment scores)
    predictions = outputs[0].tolist()
    return {"predictions": predictions}
