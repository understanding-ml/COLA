import pandas as pd
import numpy as np

from .base_model import BaseModel

PYTORCH_MODELS = [
    "PyTorchRBFNet",
    "PyTorchDNN",
    "PyTorchLogisticRegression",
    "PyTorchLinearSVM",
]

class Model(BaseModel):
    def __init__(self, model, backend):
        super().__init__(model, backend)

    # def predict(self, x_factual) -> pd.DataFrame:
    #     """
    #     Generate predictions using the pre-trained model
        
    #     Returns:
    #     DataFrame type prediction results
    #     """
    #     predictions = self.model.predict(x_factual)
    #     if isinstance(predictions, pd.Series):
    #         predictions = predictions.to_frame(name='Prediction')
    #     elif isinstance(predictions, np.ndarray):
    #         predictions = pd.DataFrame(predictions, columns=['Prediction'])
    #     return predictions

    def predict(self, x_factual):
        """
        Generate predictions using the pre-trained model
        
        Returns:
        Numpy type prediction results
        """
        predictions = self.model.predict(x_factual)
        # if isinstance(predictions, pd.Series):
        #     predictions = predictions.to_frame(name='Prediction')
        # elif isinstance(predictions, np.ndarray):
        #     predictions = pd.DataFrame(predictions, columns=['Prediction'])
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probability function that returns the probability distribution for each class.

        Args:
        X (numpy.ndarray or pandas.DataFrame): Input data for which to predict probabilities.

        Returns:
        numpy.ndarray: Array of shape (n_samples, n_classes) containing the predicted probabilities
                    for each class. Each row corresponds to a sample, and each column corresponds
                    to a class.
        """
        # 如果 X 是 DataFrame，转换为 numpy 数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        # 获取每个样本的类别概率
        probabilities = self.model.predict_proba(X)
        # 检查是否是1维数组（通常发生在二分类情况下，只返回正类的概率）
        
        if probabilities.ndim == 1:
            # 扩展为二维数组，列0为类别0的概率，列1为类别1的概率
            probabilities = np.vstack([1 - probabilities, probabilities]).T
        
        return probabilities

    def to(self, device):
        if self.model.__class__.__name__ in PYTORCH_MODELS:
            return self.model.to(device)
        else:
            raise NotImplementedError("to function can only be used for PYTORCH_MODELS")

    def __call__(self, x):
        if self.model.__class__.__name__ in PYTORCH_MODELS:
            return self.model(x)
        else:
            raise NotImplementedError("__call__ can only be used for PYTORCH_MODELS")
