import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, TopKPooling, global_mean_pool
from torch_cluster import knn_graph
PATCH_SIZE=11
import numpy as np

class DynamicSpectralSpatialProjection(nn.Module):
    def __init__(self, in_channels, debug=False):
        super().__init__()
        self.debug = debug
        self.adapt_conv = nn.Conv3d(in_channels, in_channels*2, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Spectral attention
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=1),
            nn.Dropout1d(0.25),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(in_channels, 1, (1, 3, 3), padding=(0, 1, 1)),
            nn.Dropout3d(0.1),
            nn.Sigmoid()
        )
        
        # Depthwise separable convolution
        self.dw_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, (3, 3, 7), 
                      groups=in_channels, padding=(1, 1, 3)),
            nn.Conv3d(in_channels, in_channels * 2, 1),
            nn.ReLU()
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = x
        b, c, d, h, w = x.shape
        
        # Spectral attention
        spec = x.mean(dim=[3, 4])
        spec = self.spectral_conv(spec)
        spec = spec.unsqueeze(-1).unsqueeze(-1)
        
        # Spatial attention
        spatial_weights = self.spatial_attn(x)
        
        # Combine attention mechanisms
        x = self.alpha * (x * spec) + (1 - self.alpha) * (x * spatial_weights)
        x = self.dw_conv(x)
        
        return x + self.adapt_conv(identity)

class DSSP_Lite(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spec_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(in_channels, in_channels//4, 1),
            nn.ReLU(),
            nn.Conv3d(in_channels//4, in_channels, 1),
            nn.Sigmoid()
        ) 

    def forward(self, x):
        return x * self.spec_attn(x)

class HSIEncoder(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, (3, 3, 3), padding=1),
            DSSP_Lite(32),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, (3, 3, 3), padding=1),
            DynamicSpectralSpatialProjection(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        
        # Calculate output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_bands, PATCH_SIZE, PATCH_SIZE)
            self.flatten_dim = self.encoder(dummy).flatten(1).shape[1]
            
        self.proj = nn.Linear(self.flatten_dim, 256)

    def forward(self, x):
        x = self.encoder(x)
        return self.proj(x.flatten(1))

class LiDAREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = EdgeConv(
            nn.Sequential(nn.Linear(2, 64), nn.ReLU()), aggr='mean')
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = EdgeConv(
            nn.Sequential(nn.Linear(128, 128), nn.ReLU()), aggr='mean')
        self.pool2 = TopKPooling(128, ratio=0.5)

    def forward(self, data):
        # Determine device and move to CPU if needed
        device = data.x.device
        if device.type == 'cuda':
            x_cpu = data.x.cpu()
            batch_cpu = data.batch.cpu()
            edge_index = knn_graph(x_cpu, k=8, batch=batch_cpu)
            edge_index = edge_index.to(device)
            x = data.x
        else:
            edge_index = knn_graph(data.x, k=8, batch=data.batch)
            x = data.x
            
        x = F.relu(self.conv1(x, edge_index))
        batch = data.batch
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        
        # Regenerate edges after pooling
        if device.type == 'cuda':
            x_cpu = x.cpu()
            batch_cpu = batch.cpu()
            edge_index = knn_graph(x_cpu, k=8, batch=batch_cpu)
            edge_index = edge_index.to(device)
        else:
            edge_index = knn_graph(x, k=8, batch=batch)
            
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        return global_mean_pool(x, batch)

class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy Loss"""
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
    
    def gaussian_kernel(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.exp(-torch.norm(x - y, dim=-1) ** 2 / (2 * self.sigma ** 2))
    
    def forward(self, hsi, lidar):
        hsi_norm = F.normalize(hsi, p=2, dim=-1)
        lidar_norm = F.normalize(lidar, p=2, dim=-1)
        k_xx = self.gaussian_kernel(hsi_norm, hsi_norm).mean()
        k_yy = self.gaussian_kernel(lidar_norm, lidar_norm).mean()
        k_xy = self.gaussian_kernel(hsi_norm, lidar_norm).mean()
        return torch.abs(k_xx + k_yy - 2 * k_xy)

class CrossModalAttention(nn.Module):
    def __init__(self, hsi_dim=256, lidar_dim=128):
        super().__init__()
        self.hsi_proj = nn.Linear(hsi_dim, 128)
        self.lidar_proj = nn.Linear(lidar_dim, 128)
        self.attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128)
        )

    def forward(self, hsi, lidar):
        h = self.hsi_proj(hsi).unsqueeze(1)
        l = self.lidar_proj(lidar).unsqueeze(1)
        attn_out, _ = self.attn(h, l, l)
        combined = torch.cat([h, attn_out], dim=-1)
        return self.mlp(combined.squeeze(1))

class GeoFusionNet(nn.Module):
    def __init__(self, num_classes, num_bands):
        super().__init__()
        self.hsi_encoder = HSIEncoder(num_bands)
        self.lidar_encoder = LiDAREncoder()
        self.lidar_proj_mmd = nn.Linear(128, 256)
        self.fusion = CrossModalAttention()
        self.mmd_loss = MMDLoss()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, hsi, lidar):
        hsi_feat = self.hsi_encoder(hsi)
        lidar_feat = self.lidar_encoder(lidar)
        
        # MMD alignment loss
        mmd_penalty = self.mmd_loss(hsi_feat, self.lidar_proj_mmd(lidar_feat))
        
        # Cross-modal fusion
        fused = self.fusion(hsi_feat, lidar_feat)
        
        # Classification
        logits = self.classifier(fused)
        
        return {'logits': logits, 'mmd_loss': mmd_penalty}