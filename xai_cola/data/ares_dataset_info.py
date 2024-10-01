import pandas as pd

class AReS_dataInfo:
    def __init__(
        self,
        X,
        categorical_features=None,
        name='Dataset',
        target_column=None,
        n_bins=None,
        one_hot_encoded=False,
        normalize=False,
        ordinal_features=[],
        dropped_features=[]
    ):
        """
        Initialize the AReS_dataInfo class.

        Parameters:
            X (pd.DataFrame): The input dataset containing features (and possibly the target variable).
            categorical_features (list or None): List of names of categorical features. If None, assume all features are numeric.
            name (str): Name of the dataset.
            target_column (str, optional): Name of the target variable column in X. If None, target variable is not separated.
            n_bins (int, optional): Number of bins for discretizing continuous features. If None, no binning is applied.
            one_hot_encoded (bool, optional): Whether the input data is already one-hot encoded.
            normalize (bool, optional): Whether the data has been normalized (standardized). If True, data is considered continuous.
            ordinal_features (list, optional): List of names of ordinal categorical features.
            dropped_features (list, optional): List of feature names to drop from X.
        """
        self.name = name
        self.X = X.copy()  # Make a copy of the input data
        self.categorical_features = categorical_features or []
        self.ordinal_features = ordinal_features
        self.dropped_features = dropped_features
        self.target_column = target_column
        self.n_bins = n_bins
        self.one_hot_encoded = one_hot_encoded  # Indicator for one-hot encoded data
        self.normalize = normalize  # Whether the data has been normalized

        # Drop specified features
        self.X.drop(columns=self.dropped_features, inplace=True, errors='ignore')

        # Separate target variable if specified
        if self.target_column is not None:
            self.y = self.X.pop(self.target_column)
            self.features = self.X.columns.tolist() + [self.target_column]
        else:
            self.y = None
            self.features = self.X.columns.tolist()

        # Initialize features_tree as an empty dictionary
        self.features_tree = {}

        # Determine how to handle features based on whether data is one-hot encoded and/or normalized
        if self.one_hot_encoded:
            self._generate_features_tree_one_hot()
        else:
            self._generate_features_tree()

    def _generate_features_tree(self):
        """
        Generate the features_tree attribute for non-one-hot encoded data.
        Handles categorical and numeric (continuous/discrete) features.
        """
        for col in self.X.columns:
            if col in self.categorical_features:
                # For categorical features, store the unique values
                unique_values = self.X[col].unique()
                unique_values = [str(val) for val in unique_values]  # Convert to strings
                self.features_tree[col] = unique_values
            else:
                if self.normalize:
                    # If the data has been normalized, assume it's continuous (no binning)
                    self.features_tree[col] = []  # Continuous features
                elif self.n_bins is not None:
                    # If binning is applied, prepare the feature for binning
                    self.features_tree[col] = []  # Continuous to be binned
                else:
                    # If no normalization and no binning, it's a discrete feature
                    unique_values = self.X[col].unique()
                    self.features_tree[col] = unique_values

    def _generate_features_tree_one_hot(self):
        """
        Generate the features_tree attribute for one-hot encoded data.
        Each one-hot encoded feature can only take values [0, 1].
        """
        for col in self.X.columns:
            # For one-hot encoded columns, values are always [0, 1]
            self.features_tree[col] = [0, 1]

    def get_data(self):
        """
        Return the processed data.

        Returns:
            tuple: (X, y)
                X (pd.DataFrame): The processed features data.
                y (pd.Series or None): The target variable if specified, else None.
        """
        return self.X, self.y
