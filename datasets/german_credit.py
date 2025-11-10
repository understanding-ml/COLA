import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch

TEST_SIZE = 0.3


class GermanCreditDataset:

    def __init__(self):
        self.data_path = "datasets/rawdata/"
        self.name = "german_credit"
        self.data_filename = f"{self.name}.csv"
        self.target_name = "Risk"

        self._load_data()
        self._preprocessing()

    def get_dataframe(self):
        return self.df.copy()

    def get_Xy(self):
        df_y = self.df[self.target_name].copy()
        df_X = self.df.drop(self.target_name, axis=1).copy()
        

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
                torch.FloatTensor(y_train.values).view(-1, 1),
                torch.FloatTensor(X_test.values),
                torch.FloatTensor(y_test.values).view(-1, 1),
            )
        else:
            return (X_train, y_train, X_test, y_test)

    def get_original_train_test_split(self, random_state=None, return_tensor=False):

        if random_state is not None:
            np.random.seed(random_state)  # for reproducibility

        df_X, df_y = self.get_Xy()

        # Split the dataset into training and testing sets without standardization
        X_train, X_test, y_train, y_test = train_test_split(
            df_X, df_y, test_size=TEST_SIZE, random_state=random_state
        )

        if return_tensor:
            return (
                torch.FloatTensor(X_train.values),
                torch.FloatTensor(y_train.values).view(-1, 1),
                torch.FloatTensor(X_test.values),
                torch.FloatTensor(y_test.values).view(-1, 1),
            )
        else:
            return (X_train, y_train, X_test, y_test)

    def _load_data(self):
        self.df = pd.read_csv(os.path.join(self.data_path, self.data_filename))

    def _preprocessing(self):
        # 1) 先把目标映射到数值（可能会产生 NaN：比如既不是 "good" 也不是 "bad" 的脏值）
        self.df[self.target_name] = (
            self.df[self.target_name]
              .replace({"good": 0, "bad": 1})
              .infer_objects(copy=False)
        )

        # 2) 统一删除包含 NaN 的任何行（含特征或目标列）
        before = len(self.df)
        self.df = self.df.dropna(axis=0, how='any').reset_index(drop=True)
        after = len(self.df)
        print(f"[INFO] Dropped rows with NaN: {before - after} (kept {after})")

        # 3) 若你不再需要 label encoding/中位数填充，就不要再调用 _label_encoding()
        # self._label_encoding()  # <-- 保持注释，避免再次引入填充

    def _label_encoding(self):
        # Initialize a label encoder
        self.label_encoder = LabelEncoder()
        self.label_mappings = {}

        # Convert categorical columns to numerical representations using label encoding
        for column in self.df.columns:
            if column is not self.target_name and self.df[column].dtype == "object":
                # Handle missing values by filling with a placeholder and then encoding
                self.df[column] = self.df[column].fillna("Unknown")
                self.df[column] = self.label_encoder.fit_transform(self.df[column])
                self.label_mappings[column] = dict(
                    zip(
                        self.label_encoder.classes_,
                        range(len(self.label_encoder.classes_)),
                    )
                )

        # For columns with NaN values that are numerical, we will impute them with the median of the column
        for column in self.df.columns:
            if self.df[column].isna().any():
                median_val = self.df[column].median()
                self.df[column].fillna(median_val, inplace=True)
