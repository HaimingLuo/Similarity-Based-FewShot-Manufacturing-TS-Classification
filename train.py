import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import get_datasets, create_pairs, PairsDataset
from models import TransformerEncoder, SiameseNetwork, ContrastiveLoss
import os

def get_file_paths(dataset_dir):
    dataset_names = [
        'ArticularyWordRecognition', 'AtrialFibrillation', 'CharacterTrajectories',
        'Cricket', 'FingerMovements', 'MotorImagery', 'SelfRegulationSCP1', 
        'SelfRegulationSCP2', 'UWaveGestureLibrary'
    ]
    return [os.path.join(dataset_dir, name, f'{name}_TRAIN.arff') for name in dataset_names]

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, ((x1, x2, len1), (x2, len2), y) in enumerate(train_loader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        mask1 = (torch.arange(max_len)[None, :] < len1[:, None]).to(device)
        mask2 = (torch.arange(max_len)[None, :] < len2[:, None]).to(device)
        optimizer.zero_grad()
        output1, output2 = model(x1, x2, mask1, mask2)
        loss = criterion(output1, output2, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, ((x1, x2, len1), (x2, len2), y) in enumerate(val_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            mask1 = (torch.arange(max_len)[None, :] < len1[:, None]).to(device)
            mask2 = (torch.arange(max_len)[None, :] < len2[:, None]).to(device)
            output1, output2 = model(x1, x2, mask1, mask2)
            loss = criterion(output1, output2, y)
            running_loss += loss.item()
    return running_loss / len(val_loader)

if __name__ == '__main__':
    # Load datasets
    dataset_dir = 'dataset'
    file_paths = get_file_paths(dataset_dir)
    dataset = get_datasets(file_paths)

    # Create pairs of data for contrastive learning
    pairs, pair_labels = create_pairs(dataset)
    pair_labels = torch.Tensor(pair_labels).unsqueeze(1)
    pairs_dataset = PairsDataset(pairs, pair_labels)
    train_size = int(0.8 * len(pairs_dataset))
    val_size = len(pairs_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(pairs_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize the Transformer encoder, Siamese network, and loss function
    input_dim = dataset[0][0].size(1)  # Number of features per time step
    d_model = 64  # Embedding dimension
    nhead = 8
    num_layers = 3
    dim_feedforward = 256
    max_len = 3000  # Use the maximum length from your dataset information

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = TransformerEncoder(input_dim, d_model, nhead, num_layers, dim_feedforward, max_len).to(device)
    siamese_network = SiameseNetwork(encoder).to(device)
    contrastive_loss = ContrastiveLoss()

    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.Adam(siamese_network.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss = train(siamese_network, train_loader, optimizer, contrastive_loss, device)
        val_loss = evaluate(siamese_network, val_loader, contrastive_loss, device)
        scheduler.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save the trained model
    torch.save(siamese_network.state_dict(), 'siamese_network.pth')