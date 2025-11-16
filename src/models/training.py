import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .neural_network import ProteinDataset


def split_train_validation(X_train_np, Y, config):
    """Split data into training and validation sets."""
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_np, Y,
        test_size=0.15,
        random_state=config["RANDOM_SEED"]
    )
    print(f"[split] Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def create_data_loaders(X_train, y_train, config):
    """Create PyTorch data loaders."""
    train_dataset = ProteinDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True
    )
    return train_loader


def train_model(model, train_loader, config, device):
    """Train the neural network model."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    
    print("[train] Starting training...")
    model.train()
    for epoch in range(config["EPOCHS"]):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"[train] Epoch {epoch+1}/{config['EPOCHS']}, Loss: {avg_loss:.4f}")
    
    print("[train] Training complete")
    return model
