import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class PyTorchDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=10):
        super().__init__()

        self.name = "dnn"

        # First fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Second fully connected layer (hidden layer)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

    def predict(self, x):

        # 如果输入是 pandas DataFrame，则转换为 torch.FloatTensor
        if isinstance(x, pd.DataFrame):
            x = torch.FloatTensor(x.values)
        else:
            x = torch.FloatTensor(x)
            
        return (self(x).reshape(-1) > 0.5).float().detach().numpy()

    def predict_proba(self, x):
        # 如果输入是 pandas DataFrame，则转换为 torch.FloatTensor
        if isinstance(x, pd.DataFrame):
            x = torch.FloatTensor(x.values)
        else:
            x = torch.FloatTensor(x)

        # Forward pass to get output probabilities for class 1
        probs_class1 = self(x).reshape(-1).detach().numpy()
        # Calculate probabilities for class 0
        probs_class0 = 1 - probs_class1
        # Stack the probabilities for both classes along the last axis
        return np.vstack((probs_class0, probs_class1)).T
