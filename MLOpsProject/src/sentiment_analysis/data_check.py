import torch

# Load the processed data
train_encodings = torch.load('data/processed/train_encodings.pt')
train_labels = torch.load('data/processed/train_labels.pt')
test_encodings = torch.load('data/processed/test_encodings.pt')
unique_labels = torch.load('data/processed/unique_labels.pt')

# Check the shapes of the tensors
print("Train Encodings Shape:", train_encodings['input_ids'].shape)
print("Train Labels Shape:", train_labels.shape)
print("Test Encodings Shape:", test_encodings['input_ids'].shape)

# Check some sample values
print("Sample Train Encoding:", train_encodings['input_ids'][0])
print("Sample Train Label:", train_labels[0])
print("Unique Labels:", unique_labels)
