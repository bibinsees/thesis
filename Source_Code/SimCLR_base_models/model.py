import torch
from torch import nn, optim
import torchvision

from helper import accuracy

class Resnet(nn.Module):
    def __init__(self, hidden_dim):
        super(Resnet, self).__init__()
        self.convnet = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')

        self.convnet.fc = nn.Sequential(
            nn.Linear(self.convnet.fc.in_features, 4 * hidden_dim),  # Linear layer with 4*hidden_dim output
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)  # Output layer with hidden_dim output
        )

    def forward(self, x):
        return self.convnet(x)
    

#print(Resnet(96))