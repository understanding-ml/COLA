from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
class BaseData(ABC):
    def __init__(self, data, target_name):
        """
        Initialize the BaseData class
        
        Parameters:
        data : we want to use it as the input data
        """
        self.data = data
        self.target_name = target_name

    def get_x(self):
        """ Get the Dataframe type input data (without target column)        """
        pass


    def get_y(self):
        pass

    
    def get_target_name(self):
        pass

    def get_x_labels(self):
        """ Get the columns of the input data (without target column)       """
        pass

    def get_dataframe(self) -> pd.DataFrame:
        """ Get the Dataframe type input data        """
        pass

    def get_numpy(self) -> np.array:
        """ Get the numpy type input data        """
        pass