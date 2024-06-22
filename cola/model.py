import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

PYTORCH_MODELS = [
    "PyTorchRBFNet",
    "PyTorchDNN",
    "PyTorchLogisticRegression",
    "PyTorchLinearSVM",
]

class Model:
    def __init__(self, model_path, backend):
        """
        Initialize the PreTrainedModel class
        
        Parameters:
        model_path: Path to the model file
        backend: Backend used for the model
        """
        self.model_path = model_path
        self.backend = backend
        self.model = None

    def load_model(self):
        """
        Load a pre-trained model from a file
        
        Returns:
        Loaded model
        """
        if self.model_path.endswith('.pkl'):
            with open(self.model_path, 'rb') as f:
                self.model = joblib.load(f)
                print(f'----{self.model_path} model has been loaded----')
        elif self.model_path.endswith('.joblib'):
            self.model = joblib.load(self.model_path)
            print(f'----{self.model_path} model has been loaded----')
        else:
            raise ValueError("Unsupported model file format. Please provide a .pkl or .joblib file.")
        return self.model

    def fit(self, X_train, y_train):
        """
        Fit the model
        
        Parameters:
        X_train: Training features
        y_train: Training labels
        """
        if self.model.__class__.__name__ in PYTORCH_MODELS:
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.01)

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)

            # Training loop
            num_epochs = 300
            for _ in range(num_epochs):
                # Forward pass
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            self.model.fit(X_train, y_train)

    def predict(self, x_factual) -> pd.DataFrame:
        """
        Generate predictions using the pre-trained model
        
        Parameters:
        x_factual: DataFrame type data variable
        
        Returns:
        DataFrame type prediction results
        """
        predictions = self.model.predict(x_factual)
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame(name='Prediction')
        elif isinstance(predictions, np.ndarray):
            predictions = pd.DataFrame(predictions, columns=['Prediction'])
        print(f'---- predictions have been made----')
        return predictions

    def predict_proba(self, x_factual):
        """
        Generate probability predictions using the pre-trained model
        
        Parameters:
        x_factual: DataFrame type data variable
        
        Returns:
        Numpy array of prediction probabilities
        """
        if self.model.__class__.__name__ in PYTORCH_MODELS:
            x_factual = np.array(x_factual)
        return self.model.predict_proba(x_factual)[:, 1]

    def to(self, device):
        """
        Move the model to the specified device
        
        Parameters:
        device: Target device (e.g., 'cuda' or 'cpu')
        
        Returns:
        Model on the target device
        """
        if self.model.__class__.__name__ in PYTORCH_MODELS:
            return self.model.to(device)
        else:
            raise NotImplementedError("to function can only be used for PYTORCH_MODELS")

    def __call__(self, x):
        """
        Call the model with input data
        
        Parameters:
        x: Input data
        
        Returns:
        Model output
        """
        if self.model.__class__.__name__ in PYTORCH_MODELS:
            return self.model(x)
        else:
            raise NotImplementedError("__call__ can only be used for PYTORCH_MODELS")