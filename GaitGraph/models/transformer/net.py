import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, d_model, nhead, num_classes, num_layers=2):
        super(TransformerClassifier, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Batch size, Sequence length, Num joints, 3=(x,y,c)
        N, S, J, C  = x.shape
        x = x.view(N, S, -1).permute(1, 0, 2)
            
        x = self.transformer(x)
        feat = x
        x = self.linear(x[-1, :, :])

        return x, feat