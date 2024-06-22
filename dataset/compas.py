import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch

TEST_SIZE = 0.2


class CompasDataset:

    def __init__(self, dataset_ares):
        self.data_path = "data/"
        self.name = "compas"
        self.data_filename = f"{self.name}.csv"
        self.target_name = "Status"

        self.dataset_ares = dataset_ares
        self.dataset_ares.data = self.dataset_ares.data.astype("float64")
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

        for col in ["Priors_Count", "Time_Served"]:
            X_train[col] = (X_train[col] - X_train[col].mean()) / X_train[col].std()
            X_test[col] = (X_test[col] - X_test[col].mean()) / X_test[col].std()

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
        self.df = self.df.astype("float64")

    def _preprocessing(self):
        self.target = self.df[self.target_name]
