"""
Advanced Neural Decoder Models
State-of-the-art architectures for neural signal decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class MultiScaleTemporalConvNet(nn.Module):
    """
    Multi-scale temporal convolutional network
    Captures patterns at different time scales
    """
    
    def __init__(self, num_channels: int, num_classes: int, 
                 scales: List[int] = [3, 5, 7, 9]):
        super().__init__()
        
        self.scales = scales
        self.num_channels = num_channels
        
        # Multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(num_channels, 64, kernel_size=k, padding=k//2)
            for k in scales
        ])
        
        # Channel attention
        self.channel_attention = ChannelAttention(num_channels)
        
        # Temporal attention for each scale
        self.temporal_attentions = nn.ModuleList([
            TemporalAttention(64) for _ in scales
        ])
        
        # Feature fusion
        fusion_size = 64 * len(scales)
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_size, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, channels, time]
        """
        # Apply channel attention
        x = self.channel_attention(x) * x
        
        # Multi-scale temporal processing
        multi_scale_features = []
        for conv, att in zip(self.temporal_convs, self.temporal_attentions):
            # Temporal convolution
            feat = conv(x)
            feat = F.relu(feat)
            
            # Temporal attention
            feat = att(feat) * feat
            
            multi_scale_features.append(feat)
        
        # Concatenate multi-scale features
        fused = torch.cat(multi_scale_features, dim=1)
        
        # Feature fusion
        fused = self.fusion(fused)
        
        # Global pooling
        pooled = self.global_pool(fused).squeeze(-1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    
    def __init__(self, num_channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, channels, time]
        returns: [batch, channels, 1]
        """
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        attention = avg_out + max_out
        return attention.unsqueeze(-1)


class TemporalAttention(nn.Module):
    """Temporal attention mechanism"""
    
    def __init__(self, num_features: int):
        super().__init__()
        self.conv = nn.Conv1d(num_features, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, features, time]
        returns: [batch, 1, time]
        """
        attention = torch.sigmoid(self.conv(x))
        return attention


class TransformerDecoder(nn.Module):
    """
    Transformer-based neural decoder
    Captures long-range dependencies in neural signals
    """
    
    def __init__(self, num_channels: int, num_classes: int,
                 d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 1024):
        super().__init__()
        
        self.num_channels = num_channels
        self.d_model = d_model
        
        # Channel embedding
        self.channel_embedding = nn.Linear(num_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, channels, time]
        """
        batch_size, _, seq_len = x.shape
        
        # Transpose to [time, batch, channels]
        x = x.transpose(1, 2).transpose(0, 1)
        
        # Channel embedding
        x = self.channel_embedding(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)
        x = torch.cat([cls_tokens, x], dim=0)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Use CLS token for classification
        cls_output = x[0]  # [batch, d_model]
        
        # Classification
        output = self.classifier(cls_output)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [seq_len, batch, features]
        """
        return x + self.pe[:x.size(0)]


class GraphNeuralDecoder(nn.Module):
    """
    Graph Neural Network for neural decoding
    Models spatial relationships between channels
    """
    
    def __init__(self, num_channels: int, num_classes: int,
                 hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        self.num_channels = num_channels
        
        # Initial node features
        self.node_encoder = nn.Linear(1, hidden_dim)
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Temporal processing
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        
        # Readout
        self.readout = GlobalAttentionPool(hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # Learnable adjacency matrix
        self.adj_matrix = nn.Parameter(torch.randn(num_channels, num_channels) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, channels, time]
        """
        batch_size, num_channels, seq_len = x.shape
        
        # Reshape for graph processing
        x = x.transpose(1, 2)  # [batch, time, channels]
        x = x.reshape(batch_size * seq_len, num_channels, 1)
        
        # Node encoding
        node_features = self.node_encoder(x)  # [batch*time, channels, hidden]
        
        # Get adjacency matrix
        adj = torch.sigmoid(self.adj_matrix)
        
        # Graph convolutions
        for gnn in self.gnn_layers:
            node_features = gnn(node_features, adj)
            node_features = F.relu(node_features)
        
        # Reshape back for temporal processing
        node_features = node_features.reshape(batch_size, seq_len, num_channels, -1)
        node_features = node_features.mean(dim=2)  # Average over nodes
        node_features = node_features.transpose(1, 2)  # [batch, hidden, time]
        
        # Temporal convolution
        temporal_features = self.temporal_conv(node_features)
        
        # Global readout
        graph_embedding = self.readout(temporal_features)
        
        # Classification
        output = self.classifier(graph_embedding)
        
        return output


class GraphConvLayer(nn.Module):
    """Graph convolution layer"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, nodes, features]
        adj: [nodes, nodes]
        """
        # Normalize adjacency matrix
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        degree = adj.sum(dim=1, keepdim=True)
        adj_norm = adj / degree
        
        # Graph convolution
        x = torch.matmul(adj_norm, x)
        x = self.linear(x)
        
        return x


class GlobalAttentionPool(nn.Module):
    """Global attention pooling"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, features, time]
        returns: [batch, features]
        """
        # Compute attention weights
        att_weights = self.attention(x.transpose(1, 2))  # [batch, time, 1]
        att_weights = F.softmax(att_weights, dim=1)
        
        # Weighted sum
        output = torch.sum(x * att_weights.transpose(1, 2), dim=2)
        
        return output


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture
    Combines local feature extraction with global attention
    """
    
    def __init__(self, num_channels: int, num_classes: int,
                 cnn_features: int = 128, d_model: int = 256):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Spatial convolution
            nn.Conv2d(1, 32, kernel_size=(num_channels//4, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Temporal convolution
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            # Further processing
            nn.Conv2d(64, cnn_features, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(cnn_features),
            nn.ReLU()
        )
        
        # Calculate CNN output size
        self.cnn_output_channels = (num_channels - num_channels//4 + 1) // 2
        
        # Project to transformer dimension
        self.projection = nn.Linear(cnn_features * self.cnn_output_channels, d_model)
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1
            ),
            num_layers=4
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, channels, time]
        """
        batch_size = x.size(0)
        
        # Add channel dimension for 2D convolution
        x = x.unsqueeze(1)  # [batch, 1, channels, time]
        
        # CNN feature extraction
        cnn_features = self.cnn(x)  # [batch, features, channels', time']
        
        # Reshape for transformer
        cnn_features = cnn_features.permute(0, 3, 1, 2)  # [batch, time', features, channels']
        seq_len = cnn_features.size(1)
        cnn_features = cnn_features.reshape(batch_size, seq_len, -1)
        
        # Project to transformer dimension
        transformer_input = self.projection(cnn_features)  # [batch, time', d_model]
        
        # Transformer encoding
        transformer_input = transformer_input.transpose(0, 1)  # [time', batch, d_model]
        transformer_output = self.transformer(transformer_input)
        
        # Global average pooling
        output = transformer_output.mean(dim=0)  # [batch, d_model]
        
        # Classification
        output = self.classifier(output)
        
        return output


class EnsembleDecoder(nn.Module):
    """
    Ensemble of multiple decoder architectures
    Combines predictions for improved accuracy
    """
    
    def __init__(self, num_channels: int, num_classes: int, window_size: int):
        super().__init__()
        
        # Different architectures
        self.models = nn.ModuleList([
            # Original CNN-LSTM
            CNNLSTMDecoder(num_channels, window_size),
            
            # Multi-scale temporal
            MultiScaleTemporalConvNet(num_channels, num_classes),
            
            # Transformer
            TransformerDecoder(num_channels, num_classes, d_model=128, num_layers=3),
            
            # Hybrid
            HybridCNNTransformer(num_channels, num_classes)
        ])
        
        # Learnable weights for ensemble
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
        
        # Optional: Meta-learner
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * len(self.models), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self.use_meta_learner = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, channels, time]
        """
        outputs = []
        
        # Get predictions from each model
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Stack outputs
        stacked_outputs = torch.stack(outputs, dim=1)  # [batch, num_models, num_classes]
        
        if self.use_meta_learner:
            # Concatenate all predictions
            concat_outputs = stacked_outputs.reshape(x.size(0), -1)
            output = self.meta_learner(concat_outputs)
        else:
            # Weighted average
            weights = F.softmax(self.ensemble_weights, dim=0)
            output = torch.sum(stacked_outputs * weights.unsqueeze(0).unsqueeze(-1), dim=1)
        
        return output


# Import original model for ensemble
from .models import CNNLSTMDecoder


# Model factory
def build_advanced_model(model_type: str, num_channels: int, 
                        num_classes: int, window_size: int = 64) -> nn.Module:
    """Factory function to build different model types"""
    
    models = {
        'multiscale': lambda: MultiScaleTemporalConvNet(num_channels, num_classes),
        'transformer': lambda: TransformerDecoder(num_channels, num_classes),
        'graph': lambda: GraphNeuralDecoder(num_channels, num_classes),
        'hybrid': lambda: HybridCNNTransformer(num_channels, num_classes),
        'ensemble': lambda: EnsembleDecoder(num_channels, num_classes, window_size)
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type]()
