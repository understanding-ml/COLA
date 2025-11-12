"""
Preprocessor wrapper for handling complex transformations with proper inverse_transform

This module provides a wrapper class for sklearn ColumnTransformer that ensures
proper inverse transformation for both numerical and categorical features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class ColumnTransformerWrapper:
    """
    Wrapper for sklearn ColumnTransformer with proper inverse_transform support.

    This wrapper is particularly useful when you have both numerical (StandardScaler)
    and categorical (OrdinalEncoder) transformations, as it ensures correct
    inverse transformation for both types of features.

    Parameters:
    -----------
    column_transformer : sklearn.compose.ColumnTransformer
        The fitted ColumnTransformer instance
    numerical_features : List[str]
        List of numerical feature names
    categorical_features : List[str]
        List of categorical feature names

    Example:
    --------
    >>> from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    >>> from sklearn.compose import ColumnTransformer
    >>>
    >>> # Create ColumnTransformer
    >>> preprocessor = ColumnTransformer(
    ...     transformers=[
    ...         ('num', StandardScaler(), numerical_features),
    ...         ('cat', OrdinalEncoder(), categorical_features)
    ...     ],
    ...     remainder='passthrough'
    ... )
    >>> preprocessor.fit(X_train)
    >>>
    >>> # Wrap it
    >>> wrapped_preprocessor = ColumnTransformerWrapper(
    ...     column_transformer=preprocessor,
    ...     numerical_features=numerical_features,
    ...     categorical_features=categorical_features
    ... )
    >>>
    >>> # Use with COLAData
    >>> data = COLAData(
    ...     factual_data=df,
    ...     label_column='Risk',
    ...     transform_method=wrapped_preprocessor
    ... )
    """

    def __init__(
        self,
        column_transformer,
        numerical_features: List[str],
        categorical_features: List[str]
    ):
        self.column_transformer = column_transformer
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        # Validate that the transformer has been fitted
        if not hasattr(self.column_transformer, 'transformers_'):
            raise ValueError(
                "ColumnTransformer must be fitted before wrapping. "
                "Call .fit() on the transformer first."
            )

        # Get individual transformers
        self.num_transformer = self.column_transformer.named_transformers_.get('num')
        self.cat_transformer = self.column_transformer.named_transformers_.get('cat')

        if self.num_transformer is None and self.cat_transformer is None:
            raise ValueError(
                "ColumnTransformer must have at least one of 'num' or 'cat' transformers"
            )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using the ColumnTransformer.

        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input data to transform

        Returns:
        --------
        np.ndarray
            Transformed data
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.column_transformer.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data back to original feature space.

        This method properly handles both numerical and categorical features,
        ensuring that each is correctly inverse transformed.

        Parameters:
        -----------
        X_transformed : np.ndarray
            Transformed data to inverse transform

        Returns:
        --------
        np.ndarray
            Data in original feature space
        """
        n_num = len(self.numerical_features)
        n_cat = len(self.categorical_features)

        # Separate numerical and categorical features
        X_num_scaled = X_transformed[:, :n_num] if n_num > 0 else None
        X_cat_encoded = X_transformed[:, n_num:n_num + n_cat] if n_cat > 0 else None

        results = []

        # Inverse transform numerical features
        if X_num_scaled is not None and self.num_transformer is not None:
            X_num_original = self.num_transformer.inverse_transform(X_num_scaled)
            results.append(X_num_original)

        # Inverse transform categorical features
        if X_cat_encoded is not None and self.cat_transformer is not None:
            # Round to nearest integer for categorical features
            X_cat_encoded_rounded = np.round(X_cat_encoded)
            X_cat_original = self.cat_transformer.inverse_transform(X_cat_encoded_rounded)
            results.append(X_cat_original)

        # Combine results
        if len(results) == 0:
            raise ValueError("No features to inverse transform")
        elif len(results) == 1:
            return results[0]
        else:
            return np.hstack(results)


class CustomTransformer:
    """
    Custom transformer wrapper that allows you to define your own
    transform and inverse_transform functions.

    This is useful when you need complete control over the transformation process.

    Parameters:
    -----------
    transform_func : callable
        Function that transforms the data: transform_func(X) -> X_transformed
    inverse_transform_func : callable
        Function that inverse transforms the data: inverse_transform_func(X_transformed) -> X

    Example:
    --------
    >>> def my_transform(X):
    ...     return (X - X.mean()) / X.std()
    >>>
    >>> def my_inverse_transform(X_transformed):
    ...     return X_transformed * stored_std + stored_mean
    >>>
    >>> custom_transformer = CustomTransformer(
    ...     transform_func=my_transform,
    ...     inverse_transform_func=my_inverse_transform
    ... )
    >>>
    >>> data = COLAData(
    ...     factual_data=df,
    ...     label_column='Risk',
    ...     transform_method=custom_transformer
    ... )
    """

    def __init__(self, transform_func, inverse_transform_func):
        self.transform_func = transform_func
        self.inverse_transform_func = inverse_transform_func

    def transform(self, X):
        """Apply the transform function."""
        return self.transform_func(X)

    def inverse_transform(self, X_transformed):
        """Apply the inverse transform function."""
        return self.inverse_transform_func(X_transformed)


def create_wrapped_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    X_train: pd.DataFrame,
    numerical_transformer=None,
    categorical_transformer=None
) -> ColumnTransformerWrapper:
    """
    Convenience function to create a fitted and wrapped ColumnTransformer.

    Parameters:
    -----------
    numerical_features : List[str]
        List of numerical feature names
    categorical_features : List[str]
        List of categorical feature names
    X_train : pd.DataFrame
        Training data to fit the transformer
    numerical_transformer : sklearn transformer, optional
        Transformer for numerical features (default: StandardScaler)
    categorical_transformer : sklearn transformer, optional
        Transformer for categorical features (default: OrdinalEncoder)

    Returns:
    --------
    ColumnTransformerWrapper
        Fitted and wrapped preprocessor ready to use with COLAData

    Example:
    --------
    >>> wrapped_preprocessor = create_wrapped_preprocessor(
    ...     numerical_features=['Age', 'Credit amount'],
    ...     categorical_features=['Sex', 'Job'],
    ...     X_train=X_train
    ... )
    >>>
    >>> data = COLAData(
    ...     factual_data=df_test,
    ...     label_column='Risk',
    ...     transform_method=wrapped_preprocessor
    ... )
    """
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.compose import ColumnTransformer

    # Use default transformers if not provided
    if numerical_transformer is None:
        numerical_transformer = StandardScaler()

    if categorical_transformer is None:
        categorical_transformer = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )

    # Create ColumnTransformer
    transformers = []
    if numerical_features:
        transformers.append(('num', numerical_transformer, numerical_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )

    # Fit the transformer
    column_transformer.fit(X_train)

    # Wrap it
    wrapped_preprocessor = ColumnTransformerWrapper(
        column_transformer=column_transformer,
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

    return wrapped_preprocessor
