import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch

TEST_SIZE = 0.2


class HelocDataset:

    def __init__(self, dataset_ares):
        self.data_path = "data/"
        self.name = "heloc"
        self.data_filename = f"{self.name}.csv"
        self.target_name = "RiskPerformance"

        self.dataset_ares = dataset_ares
        self.df = dataset_ares.data.copy()

        self._preprocessing()

    def get_dataframe(self):
        return self.df.copy()

    def get_Xy(self):
        df_X = self.df.drop(self.target_name, axis=1).copy()
        df_y = self.target

        return df_X, df_y

    def get_standardized_train_test_split(self, random_state=None, return_tensor=False):

        if random_state is not None:
            np.random.seed(random_state)  # for reproducibility

        df_X, df_y = self.get_Xy()

        # Split the dataset into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            df_X, df_y, test_size=TEST_SIZE, random_state=random_state
        )

        std = X_train.std()
        mean = X_train.mean()

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        if return_tensor:
            return (
                torch.FloatTensor(X_train.values),
                torch.FloatTensor(X_test.values),
                torch.FloatTensor(y_train.values).view(-1, 1),
                torch.FloatTensor(y_test.values).view(-1, 1),
            )
        else:
            return (X_train, X_test, y_train, y_test)

    def _load_data(self):
        self.df = pd.read_csv(os.path.join(self.data_path, self.data_filename))

    def _preprocessing(self):
        self.target = self.df[self.target_name]

        std = self.dataset_ares.data.std()
        mean = self.dataset_ares.data.mean()

        self.dataset_ares.data = (self.dataset_ares.data - mean) / std
        self.dataset_ares.data[self.target_name] = self.target
