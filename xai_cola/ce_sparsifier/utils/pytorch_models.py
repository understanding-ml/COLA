import torch
import torch.nn as nn


# ============================================
# 1. Linear SVM (Support Vector Machine)
# ============================================
class LinearSVM(nn.Module):
    """Linear SVM Classifier for binary classification"""
    
    def __init__(self, input_dim):
        super(LinearSVM, self).__init__()
        self.name = "svm"
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.fc(x)
        # Check for NaN in the output and replace with zeros
        if torch.isnan(out).any():
            out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
        return out

    def predict(self, x):
        """Predict class labels (0 or 1)"""
        x = torch.FloatTensor(x)
        return (self(x).reshape(-1) > 0.5).float().detach().numpy()


def svm_loss(outputs, labels):
    """Hinge loss for SVM
    
    Args:
        outputs: model predictions
        labels: true labels (+1 or -1)
    
    Returns:
        hinge loss value
    """
    return torch.mean(torch.clamp(1 - outputs.t() * labels, min=0))


# ============================================
# 2. BlackBox Model (Deep Neural Network)
# ============================================
class DNN(nn.Module):
    """Three-layer deep neural network for binary classification"""
    
    def __init__(self, input_dim, hidden_dim=10):
        super(DNN, self).__init__()
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
        """Predict class labels (0 or 1)"""
        x = torch.FloatTensor(x)
        return (self(x).reshape(-1) > 0.5).float().detach().numpy()


# ============================================
# Usage Example
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("Model Testing")
    print("=" * 50)
    
    # Generate sample data
    input_dim = 5
    num_samples = 3
    x_test = torch.randn(num_samples, input_dim)
    
    # 1. Test Linear SVM
    print("\n1. Linear SVM")
    svm_model = LinearSVM(input_dim=input_dim)
    print(f"   Input shape: {x_test.shape}")
    print(f"   Decision values: {svm_model(x_test).squeeze().detach().numpy()}")
    print(f"   Predictions: {svm_model.predict(x_test.numpy())}")
    
    # 2. Test DNN (Deep Neural Network)
    print("\n2. DNN (Deep Neural Network)")
    dnn_model = DNN(input_dim=input_dim, hidden_dim=20)
    print(f"   Input shape: {x_test.shape}")
    print(f"   Output probabilities: {dnn_model(x_test).squeeze().detach().numpy()}")
    print(f"   Predictions: {dnn_model.predict(x_test.numpy())}")
    
    # 3. Test SVM loss
    print("\n3. SVM Loss Calculation")
    labels = torch.tensor([1, -1, 1]).unsqueeze(1).float()
    outputs = svm_model(x_test)
    loss = svm_loss(outputs, labels)
    print(f"   Hinge loss: {loss.item():.4f}")
    
    print("\n" + "=" * 50)
    print("All models tested successfully!")
    print("=" * 50)