import numpy as np
import onnxruntime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load the ONNX model once at startup
session = onnxruntime.InferenceSession("models/model.onnx")


# Pydantic model for input. Each is a list of lists of integers
# so you can handle batch processing, e.g. [[101, 23, 45, ...], [101, 67, 89, ...]]
class InputData(BaseModel):
    input_ids: List[List[int]]
    attention_mask: List[List[int]]


@app.get("/")
def read_root():
    return {"message": "Welcome to the ONNX Model API!"}


@app.post("/predict")
def predict(input_data: InputData):
    """
    Predict using an ONNX-exported Transformer model that expects
    'input_ids' and 'attention_mask'.
    """

    # Convert Python lists into NumPy arrays
    input_ids_np = np.array(input_data.input_ids, dtype=np.int64)
    attention_mask_np = np.array(input_data.attention_mask, dtype=np.int64)

    # Run inference with the correct input names
    outputs = session.run(
        None,  # request all outputs
        {"input_ids": input_ids_np, "attention_mask": attention_mask_np},
    )

    # The ONNX export was defined with output_names=['output']
    # so outputs[0] corresponds to "output" in the graph
    predictions = outputs[0].tolist()
    return {"predictions": predictions}
