import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, in_size, hidden_size1, hidden_size2, hidden_size3, out_size):
        super().__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(in_size, hidden_size1)
        # hidden layer 2
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        # hidden layer 3
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        # output layer
        self.linear4 = nn.Linear(hidden_size3, out_size)


    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)
        # Get outputs
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out
    
    def predict(self, img):
        xb = img.unsqueeze(0)
        yb = self(xb)
        conf, preds  = torch.max(yb, dim=1)
        return conf, preds[0].item()
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # size: 1 x 28 x 28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # size: 32 x 28 x 28
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # size: 64 x 14 x 14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # size: 128 x 14 x 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # size: 256 x 7 x 7
            nn.Flatten(),
            nn.Linear(256*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)

    def predict(self, img):
        xb = img.unsqueeze(0)
        yb = self(xb)
        conf, preds  = torch.max(yb, dim=1)
        return conf, preds[0].item() 
