import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """Dataset for protein embeddings and labels."""
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MultiLabelClassifier(nn.Module):
    """Multi-label protein function prediction model."""
    def __init__(self, input_dim, num_labels, hidden_units=512, dropout=0.2):
        super(MultiLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_units, num_labels)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x


def create_model(input_dim, num_labels, config, device):
    """Create and initialize the model."""
    model = MultiLabelClassifier(
        input_dim=input_dim,
        num_labels=num_labels,
        hidden_units=config["HIDDEN_UNITS"],
        dropout=config["DROPOUT"]
    )
    model = model.to(device)
    return model
