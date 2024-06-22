import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

PYTORCH_MODELS = [
    "PyTorchRBFNet",
    "PyTorchDNN",
    "PyTorchLogisticRegression",
    "PyTorchLinearSVM",
]


class ClassifierWrapper:

    def __init__(self, classifier, backend):
        self.classifier = classifier
        self.backend = backend

    def fit(self, X_train, y_train):

        if self.classifier.__class__.__name__ in PYTORCH_MODELS:
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.classifier.parameters(), lr=0.01)

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)

            # Training loop
            num_epochs = 300
            for _ in range(num_epochs):
                # Forward pass
                outputs = self.classifier(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            self.classifier.fit(X_train, y_train)

    def predict_proba(self, X):
        if self.classifier.__class__.__name__ in PYTORCH_MODELS:
            X = np.array(X)
        return self.classifier.predict_proba(X)[:, 1]

    def predict(self, X):
        if self.classifier.__class__.__name__ in PYTORCH_MODELS:
            X = np.array(X)
        return self.classifier.predict(X)

    def to(self, device):
        if self.classifier.__class__.__name__ in PYTORCH_MODELS:
            return self.classifier.to(device)
        else:
            raise NotImplementedError("to function can only be used for PYTORCH_MODELS")

    def __call__(self, x):
        if self.classifier.__class__.__name__ in PYTORCH_MODELS:
            return self.classifier(x)
        else:
            raise NotImplementedError("__call__ can only be used for PYTORCH_MODELS")
