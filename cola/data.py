import pandas as pd
import numpy as np

class Data:
    def __init__(self, dataset, target_name):
        """
        Initialize the Data class, process data based on the input type.
        
        Parameters:
        dataset: Can be a DataFrame, NumPy array, or Excel file path
        target_name: The name of the target column (string)
        """
        self.target_name = target_name
        if isinstance(dataset, pd.DataFrame):
            self.df = dataset
        elif isinstance(dataset, np.ndarray):
            self.df = pd.DataFrame(dataset)
        elif isinstance(dataset, str) and dataset.endswith(('.xls', '.xlsx')):
            self.df = pd.read_excel(dataset)
        else:
            raise ValueError("Unsupported data type. Please provide a DataFrame, NumPy array, or Excel file path.")
        
        if self.target_name not in self.df.columns:
            raise ValueError(f"Target name '{self.target_name}' not found in the dataset columns.")
    
    def get_dataframe(self):
        """
        Return the DataFrame stored in the class
        """
        return self.df
    
    def get_x(self):
        """
        Return the DataFrame excluding the target column
        """
        x_factual  = self.df.drop(columns=[self.target_name]).copy()
        return x_factual
    
    def get_y(self):
        """
        Return the data of the target column
        """
        return self.df[self.target_name]
    
    def get_target_name(self):
        return self.target_name
    
    def get_x_labels(self):
        a = self.df.drop(columns=[self.target_name]).copy()
        return a.columns