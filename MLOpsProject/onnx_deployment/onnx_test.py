"""
To run this script, use the following command:
python -m onnx_deployment.onnx_test

This script tests inference with an ONNX model by running it on dummy input data.
Make sure you have first exported your model to ONNX format using create_onnx.py.
"""

import onnxruntime as ort
import torch


def run_inference_onnx(onnx_model_path, input_ids, attention_mask):
    """
    Run inference using an ONNX model for sentiment analysis.

    Args:
        onnx_model_path (str): Path to the ONNX model file
        input_ids (torch.Tensor): Input token IDs tensor of shape (batch_size, seq_len)
        attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_len)

    Returns:
        list: Model outputs as numpy arrays
    """
    # Load the ONNX model
    session = ort.InferenceSession(onnx_model_path)

    # Convert torch Tensors to numpy for ONNX runtime
    input_ids_np = input_ids.numpy()
    attention_mask_np = attention_mask.numpy()

    # Run inference with the ONNX model
    outputs = session.run(
        None,  # None means return all outputs
        {"input_ids": input_ids_np, "attention_mask": attention_mask_np},
    )

    return outputs


if __name__ == "__main__":
    # Create sample inputs for testing
    dummy_input_ids = torch.randint(0, 30000, (1, 128))
    dummy_attention_mask = torch.ones(1, 128)

    # Test the model with dummy inputs
    outputs = run_inference_onnx("models/model.onnx", dummy_input_ids, dummy_attention_mask)
    print(outputs)
