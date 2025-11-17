import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
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




def train_model_single_fold(model, train_loader, val_loader, config, device, save_path, fold_num):
    """
    Train the neural network model for a single fold with early stopping.
    Saves the model with the best validation loss.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration dictionary
        device: torch device
        save_path: Path to save the best model
        fold_num: Current fold number
    
    Returns:
        model: Trained model (loaded with best weights)
        best_val_loss: Best validation loss achieved
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_path, f'best_model_fold_{fold_num}.pt')
    patience_counter = 0
    patience = config.get("PATIENCE", 10)  # Early stopping patience
    
    print(f"[train] Starting training for fold {fold_num}...")
    
    for epoch in range(config["EPOCHS"]):
        # Training phase
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"[train] Fold {fold_num} - Epoch {epoch+1}/{config['EPOCHS']}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[train] Fold {fold_num} - New best model saved! Val Loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"[train] Fold {fold_num} - Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model weights
    model.load_state_dict(torch.load(best_model_path))
    print(f"[train] Fold {fold_num} - Training complete. Best Val Loss: {best_val_loss:.4f}")
    
    return model, best_val_loss



def train_model_kfold(X_train_np, Y, config, device, save_path, model_creator_fn):
    """
    Train models using K-Fold cross-validation.
    
    Args:
        X_train_np: Training features (numpy array)
        Y: Training labels (numpy array)
        config: Configuration dictionary
        device: torch device
        save_path: Path to save models
        model_creator_fn: Function to create a new model instance
    
    Returns:
        fold_results: Dictionary containing results for each fold
    """
    n_splits = config.get("K_FOLDS", 5)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config["RANDOM_SEED"])
    
    fold_results = {
        'models': [],
        'val_losses': [],
        'train_indices': [],
        'val_indices': []
    }
    
    print(f"[kfold] Starting {n_splits}-fold cross-validation")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_np), 1):
        print(f"\n{'='*60}")
        print(f"[kfold] Processing Fold {fold}/{n_splits}")
        print(f"{'='*60}")
        
        # Split data for this fold
        X_train_fold = X_train_np[train_idx]
        X_val_fold = X_train_np[val_idx]
        y_train_fold = Y[train_idx]
        y_val_fold = Y[val_idx]
        
        print(f"[kfold] Fold {fold} - Train: {X_train_fold.shape}, Val: {X_val_fold.shape}")
        
        # Create data loaders
        train_loader = create_data_loaders(X_train_fold, y_train_fold, config)
        val_dataset = ProteinDataset(X_val_fold, y_val_fold)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["BATCH_SIZE"],
            shuffle=False
        )
        
        # Create a new model for this fold
        input_dim = X_train_np.shape[1]
        num_labels = Y.shape[1]
        model = model_creator_fn(input_dim, num_labels, config, device)
        
        # Train model for this fold
        model, best_val_loss = train_model_single_fold(
            model, train_loader, val_loader, config, device, save_path, fold
        )
        
        # Store results
        fold_results['models'].append(model)
        fold_results['val_losses'].append(best_val_loss)
        fold_results['train_indices'].append(train_idx)
        fold_results['val_indices'].append(val_idx)
    
    # Print summary
    print(f"\n{'='*60}")
    print("[kfold] Cross-validation Summary")
    print(f"{'='*60}")
    for fold, val_loss in enumerate(fold_results['val_losses'], 1):
        print(f"Fold {fold}: Val Loss = {val_loss:.4f}")
    avg_val_loss = np.mean(fold_results['val_losses'])
    std_val_loss = np.std(fold_results['val_losses'])
    print(f"\nAverage Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
    
    return fold_results
