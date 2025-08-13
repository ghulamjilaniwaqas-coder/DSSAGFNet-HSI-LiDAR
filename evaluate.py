import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader
from dataset import HyperspectralDataset, custom_collate_fn

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            hsi = batch['hsi'].to(device)
            lidar = batch['lidar'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(hsi, lidar)
            _, preds = torch.max(outputs['logits'], 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    aa = np.mean(class_acc)
    
    return acc, aa, kappa, cm, class_acc

def generate_classification_map(model, device, run_dir):
    model.eval()
    # Create full map dataset
    full_map_dataset = HyperspectralDataset(full_map=True)
    
    # Get original shape from test coordinates
    test_coords = np.load('data/test_coords.npy')
    h, w = np.max(test_coords, axis=0) + 1
    pred_map = -np.ones((h, w), dtype=np.int16)
    
    # Create dataloader for full map
    loader = DataLoader(
        full_map_dataset, 
        batch_size=256, 
        shuffle=False, 
        collate_fn=custom_collate_fn
    )
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Map"):
            hsi = batch['hsi'].to(device)
            lidar = batch['lidar'].to(device)
            coords = batch['coords'].cpu().numpy()
            
            outputs = model(hsi, lidar)
            preds = torch.argmax(outputs['logits'], dim=1).cpu().numpy()
            
            # Assign predictions to coordinates
            for i in range(len(preds)):
                y, x = coords[i]
                pred_map[y, x] = preds[i] + 1  # Convert to 1-based indexing
    
    # Save map and visualization
    sio.savemat(f'{run_dir}/classification_map.mat', {'pred_map': pred_map})
    
    plt.figure(figsize=(12, 8))
    plt.imshow(pred_map, cmap='nipy_spectral')
    plt.colorbar(fraction=0.03, pad=0.04)
    plt.axis('off')
    plt.title(f'Classification Map')
    plt.savefig(f'{run_dir}/classification_map.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return pred_map

def save_metrics(all_metrics, class_names, run_dir):
    """Save metrics with standard deviations"""
    # Extract metrics
    oa_vals = [m[0] for m in all_metrics]
    aa_vals = [m[1] for m in all_metrics]
    kappa_vals = [m[2] for m in all_metrics]
    
    # Calculate class-wise metrics
    class_accs = {}
    for class_idx in range(len(class_names)):
        class_accs[class_idx] = [run[4][class_idx] for run in all_metrics]
    
    # Calculate statistics
    oa_mean, oa_std = np.mean(oa_vals), np.std(oa_vals)
    aa_mean, aa_std = np.mean(aa_vals), np.std(aa_vals)
    kappa_mean, kappa_std = np.mean(kappa_vals), np.std(kappa_vals)
    
    # Save to text file
    with open(f'{run_dir}/metrics_summary.txt', 'w') as f:
        f.write(f"Overall Accuracy: {oa_mean:.4f} ± {oa_std:.4f}\n")
        f.write(f"Average Accuracy: {aa_mean:.4f} ± {aa_std:.4f}\n")
        f.write(f"Kappa Coefficient: {kappa_mean:.4f} ± {kappa_std:.4f}\n\n")
        
        f.write("Class-wise Accuracies:\n")
        for class_idx, class_name in class_names.items():
            accs = class_accs[class_idx-1]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            f.write(f"{class_name}: {mean_acc:.4f} ± {std_acc:.4f}\n")
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(oa_vals)), oa_vals, yerr=oa_std, fmt='o', label='OA')
    plt.errorbar(range(len(aa_vals)), aa_vals, yerr=aa_std, fmt='s', label='AA')
    plt.errorbar(range(len(kappa_vals)), kappa_vals, yerr=kappa_std, fmt='^', label='Kappa')
    plt.xlabel('Run')
    plt.ylabel('Value')
    plt.title('Performance Metrics Across Runs')
    plt.legend()
    plt.savefig(f'{run_dir}/performance_metrics.png', bbox_inches='tight')
    plt.close()