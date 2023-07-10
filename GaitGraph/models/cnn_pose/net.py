import torch
import torch.nn as nn

class CNNPose(nn.Module):
    def __init__(self, num_classes):
        super(CNNPose, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32*7*9, num_classes)
        
    def forward(self, x):
        # Batch size, Sequence length, Num joints, 3=(x,y,c)
        N, S, J, C  = x.shape
        x = x[:,:,:,:2] # only x, y coordinates
        x = x.reshape(N, S, -1).unsqueeze(1)
        # print(x.shape) # (768, 1, 30, 36)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        feat = x
        # print(feat.shape) # (768, 32, 7, 9)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, feat
    

