import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.lin1 = nn.Linear(input_dim, input_dim)
        self.lin2 = nn.Linear(input_dim, 16)
        self.lin3 = nn.Linear(16, 12)
        self.lin4 = nn.Linear(12, 8)
        self.lin5 = nn.Linear(8, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = self.lin5(x)
        x = self.sigmoid(x)

        return x
    
