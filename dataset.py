import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from config import PATCH_SIZE, HALF_PATCH

class HyperspectralDataset(Dataset):
    def __init__(self, mode='train', full_map=False):
        self.mode = mode
        self.full_map = full_map
        
        if full_map:
            # Load full padded data
            self.hsi = np.load('data/full_hsi.npy')
            self.lidar = np.load('data/full_lidar.npy')
            
            # Get original dimensions from test coordinates
            test_coords = np.load('data/test_coords.npy')
            self.height, self.width = np.max(test_coords, axis=0) + 1
            self.coords = [(i, j) for i in range(self.height) for j in range(self.width)]
        else:
            if mode == 'train':
                self.hsi = np.load('data/train_hsi.npy')
                self.lidar = np.load('data/train_lidar.npy')
                self.labels = np.load('data/train_labels.npy')
                self.coords = np.load('data/train_coords.npy')
            else:  # test
                self.hsi = np.load('data/test_hsi.npy')
                self.lidar = np.load('data/test_lidar.npy')
                self.labels = np.load('data/test_labels.npy')
                self.coords = np.load('data/test_coords.npy')
    
    def __len__(self):
        if self.full_map:
            return len(self.coords)
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.full_map:
            y_orig, x_orig = self.coords[idx]
            # Convert to padded coordinates
            y_pad = y_orig + HALF_PATCH
            x_pad = x_orig + HALF_PATCH
            
            # Extract patch from padded data
            hsi_patch = self.hsi[
                y_pad-HALF_PATCH:y_pad+HALF_PATCH+1, 
                x_pad-HALF_PATCH:x_pad+HALF_PATCH+1, 
                :
            ]
            lidar_patch = self.lidar[
                y_pad-HALF_PATCH:y_pad+HALF_PATCH+1,
                x_pad-HALF_PATCH:x_pad+HALF_PATCH+1
            ]
            
            return {
                'hsi': torch.tensor(hsi_patch).permute(2, 0, 1).unsqueeze(0).float(),
                'lidar': Data(x=torch.tensor(lidar_patch.reshape(-1, 1)).float()),
                'coords': torch.tensor([y_orig, x_orig])
            }
        else:
            return {
                'hsi': torch.tensor(self.hsi[idx]).permute(2, 0, 1).unsqueeze(0).float(),
                'lidar': Data(x=torch.tensor(self.lidar[idx].reshape(-1, 1)).float()),
                'label': torch.tensor(self.labels[idx]-1).long(),
                'coords': torch.tensor(self.coords[idx])
            }

def custom_collate_fn(batch):
    collated = {
        'hsi': torch.stack([item['hsi'] for item in batch]),
        'lidar': Batch.from_data_list([item['lidar'] for item in batch]),
        'coords': torch.stack([item['coords'] for item in batch])
    }
    if 'label' in batch[0]:
        collated['label'] = torch.stack([item['label'] for item in batch])
    return collated