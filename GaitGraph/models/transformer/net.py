import math
import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    """Basic Transformer Classifier

    Args:
        d_model (int): number of expected features in the input
        nhead (int): number of heads in the multiheadattention models
        num_classes (int): number of classes

    Returns:
        x: result output
        feat: intermediate feature
    """
    def __init__(self, d_model, nhead, num_classes, num_layers=2):
        super(TransformerClassifier, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(d_model*30, num_classes)

    def forward(self, x):
        # Batch size, Sequence length, Num joints, 3=(x,y,c)
        N, S, J, C  = x.shape
        x = x[:,:,:,:2] # only x, y coordinates
        x = x.reshape(N, S, -1).permute(1, 0, 2)
            
        x = self.transformer(x)
        feat = x
        x = x.permute(1, 0, 2).reshape(N, -1)
        x = self.linear(x)

        return x, feat
    

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_joints=18, num_coordinates=2, num_classes=128):
        super(SpatioTemporalTransformer, self).__init__()
        self.spatial_transformer = SpatialTransformer(num_joints, num_coordinates, d_model, nhead, num_layers)
        self.temporal_transformer = TemporalTransformer(d_model, nhead, num_layers, num_classes)

    def forward(self, x):
        x = self.spatial_transformer(x)
        x = self.temporal_transformer(x)
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, num_joints, num_coordinates, d_model, num_heads, num_layers):
        super(SpatialTransformer, self).__init__()

        self.num_joints = num_joints
        self.num_coordinates = num_coordinates
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.positional_encoding = self._get_positional_encoding()

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])

    def _get_positional_encoding(self):
        positional_encoding = torch.zeros(self.num_joints, self.num_coordinates, self.d_model)
        pos = torch.arange(0, self.num_joints).unsqueeze(1).unsqueeze(2).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        positional_encoding[:, :, 0::2] = torch.sin(pos * div_term)
        positional_encoding[:, :, 1::2] = torch.cos(pos * div_term)
        return positional_encoding

    def forward(self, x):
        batch_size, sequence_length, _, _ = x.size()

        # Reshape input tensor
        x = x.view(batch_size * sequence_length, self.num_joints, self.num_coordinates)

        # Add positional encoding
        x = x + self.positional_encoding

        # Transformer layers
        for i in range(self.num_layers):
            x = self.transformer_layers[i](x)

        return x

class TemporalTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, num_classes):
        super(TemporalTransformer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size, sequence_length, num_joints, num_coordinates = x.size()

        # Reshape input tensor
        x = x.view(batch_size * sequence_length, num_joints, num_coordinates)

        # Transformer layers
        for i in range(self.num_layers):
            x = self.transformer_layers[i](x)

        # Final linear layer
        x = self.fc(x)

        return x