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
