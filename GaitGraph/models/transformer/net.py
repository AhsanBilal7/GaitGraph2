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
    
