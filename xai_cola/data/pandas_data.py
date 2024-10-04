import numpy as np
from .base_data import BaseData

# data contains the x and y, including their labels
class PandasData(BaseData): 
    def __init__(self, data, target_name):
        super().__init__(data, target_name)
    
    def get_x(self):
        """
        Return the DataFrame excluding the target column
        """
        # x_factual  = self.data.drop(columns=[self.target_name]).copy()
        return self.data
    
    def get_y(self):
        """
        Return the data of the target column
        """
        return self.data[self.target_name]
    
    def get_target_name(self):
        return self.target_name
    
    def get_x_labels(self):

        return self.data.columns
    
    def get_numpy(self) -> np.array:
        return self.data.values