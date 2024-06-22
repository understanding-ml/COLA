import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch

TEST_SIZE = 0.3


class HotelBookingsDataset:

    def __init__(self):
        self.data_path = "data/"
        self.name = "hotel_bookings"
        self.data_filename = f"{self.name}.csv"
        self.target_name = "is_canceled"

        self.features = [
            "hotel",
            "lead_time",
            "arrival_date_year",
            "arrival_date_month",
            "arrival_date_week_number",
            "arrival_date_day_of_month",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "children",
            "babies",
            "meal",
            "country",
            "market_segment",
            "distribution_channel",
            "is_repeated_guest",
            "previous_cancellations",
            "previous_bookings_not_canceled",
            "reserved_room_type",
            "assigned_room_type",
            "booking_changes",
            "deposit_type",
            "agent",
            "company",
            "days_in_waiting_list",
            "customer_type",
            "adr",
            "required_car_parking_spaces",
            "total_of_special_requests",
        ]

        self._load_data()
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
        self.df = self.df[self.features]
        self.df[self.target_name] = self.target

        self._label_encoding()

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
