import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=2500) # TODO: mess with the hidden layer sizes
        self.fc2 = nn.Linear(in_features=2500, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=out_dim)
        
    
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        
        x = self.fc2(x)
        x = nn.functional.relu(x)
        
        x = self.fc3(x)
        x = nn.functional.relu(x)
        
        x = self.fc4(x)
        
        return x