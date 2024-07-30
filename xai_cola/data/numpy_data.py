import pandas as pd
from .base_data import BaseData

class NumpyData(BaseData):
    def __init__(self, data, target_name, target_index, feature_names=None):
        super().__init__(data, target_name)
        """
        Initialize the NumpyData object
        
        Parameters:
        data (numpy.ndarray): The dataset as a NumPy array.
        target_index (int): The index of the target column in the dataset.
        feature_names (list): List of feature names (optional).
        target_name (str): Name of the target column (optional).
        """
        self.target_index = target_index
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(data.shape[1] - 1)]
        self.df = self._create_dataframe()

    def _create_dataframe(self):
        """
        Create a DataFrame from the numpy array
        """
        columns = self.feature_names.copy()
        columns.insert(self.target_index, self.target_name)
        return pd.DataFrame(self.data, columns=columns)
    
    def get_dataframe(self):
        """
        Return the DataFrame stored in the class
        """
        return self.df
    
    def get_x(self):
        """
        Return the DataFrame excluding the target column
        """
        x_factual = self.df.drop(columns=[self.target_name]).copy()
        return x_factual
    
    def get_y(self):
        """
        Return the data of the target column
        """
        return self.df[self.target_name]
    
    def get_target_name(self):
        """
        Return the name of the target column
        """
        return self.target_name
    
    def get_x_labels(self):
        """
        Return the labels of the feature columns (excluding the target column)
        """
        return self.df.drop(columns=[self.target_name]).columns
