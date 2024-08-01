import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import FewShotLearningModel, TransformerEncoder
from data_loader import get_datasets
import numpy as np
import os

# Create a few-shot dataset example
def create_few_shot_dataset(dataset, n_shots=5):
    data, labels, lengths = [], [], []
    unique_labels = torch.unique(dataset.labels)
    for label in unique_labels:
        indices = (dataset.labels == label).nonzero(as_tuple=True)[0]
        chosen_indices = indices[:n_shots]
        data.append(dataset.data[chosen_indices])
        labels.append(dataset.labels[chosen_indices])
        lengths.extend(dataset.lengths[chosen_indices])
    return torch.cat(data), torch.cat(labels), lengths

# Load few-shot datasets
fewshotset_dir = 'fewshotset'
file_paths = [
    os.path.join(fewshotset_dir, 'FaceDetection/FaceDetection_TRAIN.arff'),
    os.path.join(fewshotset_dir, 'HandMovementDirection/HandMovementDirection_TRAIN.arff'),
    os.path.join(fewshotset_dir, 'Handwriting/Handwriting_TRAIN.arff'),
]
dataset = get_datasets(file_paths)
few_shot_data, few_shot_labels, few_shot_lengths = create_few_shot_dataset(dataset)
few_shot_data, few_shot_labels = torch.Tensor(few_shot_data), torch.Tensor(few_shot_labels).long()

# Initialize the Transformer encoder
input_dim = few_shot_data.size(2)  # Number of features per time step
d_model = 64  # Embedding dimension
nhead = 8
num_layers = 3
dim_feedforward = 256
num_classes = len(torch.unique(few_shot_labels))
max_len = 3000  # Use the maximum length from your dataset information

encoder = TransformerEncoder(input_dim, d_model, nhead, num_layers, dim_feedforward, max_len)
few_shot_model = FewShotLearningModel(encoder, num_classes)

# Calculate the mean embedding for each class
def calculate_class_embeddings(encoder, data, labels, lengths, num_classes):
    encoder.eval()
    class_embeddings = []
    with torch.no_grad():
        for i in range(num_classes):
            class_data = data[labels == i]
            class_lengths = lengths[labels == i]
            class_mask = (torch.arange(max_len)[None, :] < class_lengths[:, None]).to(class_data.device)
            class_embedding = encoder(class_data, src_key_padding_mask=class_mask).mean(dim=0)
            class_embeddings.append(class_embedding)
    return torch.stack(class_embeddings)

class_embeddings = calculate_class_embeddings(encoder, few_shot_data, few_shot_labels, few_shot_lengths, num_classes)

# Few-shot classification function
def classify(encoder, class_embeddings, new_data, new_lengths):
    encoder.eval()
    with torch.no_grad():
        new_mask = (torch.arange(max_len)[None, :] < new_lengths[:, None]).to(new_data.device)
        new_embeddings = encoder(new_data, src_key_padding_mask=new_mask)
        distances = torch.cdist(new_embeddings, class_embeddings)
        predicted_labels = torch.argmin(distances, dim=1)
    return predicted_labels

# Example usage with new data
# Here we generate some example new data
num_new_samples = 10
sequence_length = input_dim  # Assuming the same length as input_dim for simplicity
new_time_series_data = [np.random.randn(np.random.randint(50, max_len), sequence_length).astype(np.float32) for _ in range(num_new_samples)]
new_lengths = [len(seq) for seq in new_time_series_data]
new_data = torch.tensor(new_time_series_data, dtype=torch.float32)

predicted_labels = classify(encoder, class_embeddings, new_data, torch.tensor(new_lengths))
print(predicted_labels)
