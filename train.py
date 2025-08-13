import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import HyperspectralDataset, custom_collate_fn
from model import GeoFusionNet
import numpy as np
def train_model(model, train_loader, optimizer, device, epoch, weights=None):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        
        hsi = batch['hsi'].to(device)
        lidar = batch['lidar'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(hsi, lidar)
        loss = criterion(outputs['logits'], labels) + 0.1 * outputs['mmd_loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def setup_training(run_dir, num_bands, num_classes, device):
    # Initialize model
    model = GeoFusionNet(num_classes, num_bands).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    
    # Create datasets
    train_dataset = HyperspectralDataset(mode='train')
    test_dataset = HyperspectralDataset(mode='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, collate_fn=custom_collate_fn)
    
    # Class weights for imbalanced data
    train_labels = train_dataset.labels - 1  # Convert to 0-indexed
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)  # Avoid division by zero
    weights = 1.0 / class_counts
    weights = weights / weights.sum()  # Normalize to sum to 1
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    return model, optimizer, scheduler, train_loader, test_loader, weights