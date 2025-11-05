"""
Pipeline Utilities for COLA

This module provides utility functions for creating sklearn pipelines that are compatible
with both pandas DataFrames and numpy arrays, especially useful for SHAP calculations
where numpy arrays are commonly used.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def create_pipeline_with_column_names(
    numerical_features: List[str],
    categorical_features: List[str],
    classifier,
    reference_dataframe: Optional[pd.DataFrame] = None,
    numerical_transformer=None,
    categorical_transformer=None
) -> Tuple[Pipeline, List[str]]:
    """
    Create a sklearn Pipeline that works with both DataFrames and numpy arrays.

    This function automatically converts column names to indices, making the pipeline
    compatible with numpy arrays (which is required for SHAP calculations).

    Parameters:
    -----------
    numerical_features : list of str
        List of numerical feature column names
    categorical_features : list of str
        List of categorical feature column names
    classifier : sklearn estimator
        Classifier model (e.g., LGBMClassifier, RandomForestClassifier)
    reference_dataframe : pd.DataFrame, optional
        Reference DataFrame to determine column order
        If None, assumes order is numerical_features + categorical_features
    numerical_transformer : sklearn transformer, optional
        Transformer for numerical features (default: StandardScaler())
    categorical_transformer : sklearn transformer, optional
        Transformer for categorical features (default: OneHotEncoder(drop='first', handle_unknown='ignore'))

    Returns:
    --------
    pipe : sklearn.pipeline.Pipeline
        Created pipeline with preprocessing and classifier
    feature_order : list of str
        Feature order used in the pipeline (important for maintaining consistency)

    Raises:
    -------
    ValueError
        If features specified are not found in reference_dataframe

    Example:
    --------
    >>> from lightgbm import LGBMClassifier
    >>>
    >>> pipe, feature_order = create_pipeline_with_column_names(
    ...     numerical_features=['Age', 'Income'],
    ...     categorical_features=['Gender', 'Education'],
    ...     classifier=LGBMClassifier(random_state=42),
    ...     reference_dataframe=X_train
    ... )
    >>>
    >>> # Train with correct column order
    >>> pipe.fit(X_train[feature_order], y_train)
    >>>
    >>> # Create Model wrapper
    >>> from xai_cola.ce_sparsifier.models import Model
    >>> ml_model = Model(model=pipe, backend="sklearn")
    """
    # Set default transformers if not provided
    if numerical_transformer is None:
        numerical_transformer = StandardScaler()

    if categorical_transformer is None:
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    # Determine feature order
    if reference_dataframe is not None:
        # Get feature order from DataFrame
        all_features = [col for col in reference_dataframe.columns
                       if col in numerical_features + categorical_features]
        # Ensure all specified features are in DataFrame
        missing_features = set(numerical_features + categorical_features) - set(all_features)
        if missing_features:
            raise ValueError(f"Features not found in reference_dataframe: {missing_features}")
    else:
        # Use specified order
        all_features = numerical_features + categorical_features

    # Convert column names to indices
    feature_to_index = {name: idx for idx, name in enumerate(all_features)}
    numerical_indices = [feature_to_index[name] for name in numerical_features]
    categorical_indices = [feature_to_index[name] for name in categorical_features]

    # Create preprocessor (using indices instead of column names)
    transformers = []
    if numerical_indices:
        transformers.append(('num', numerical_transformer, numerical_indices))
    if categorical_indices:
        transformers.append(('cat', categorical_transformer, categorical_indices))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any columns not specified
    )

    # Create Pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipe, all_features


def create_simple_pipeline(
    classifier,
    preprocessor: Optional[ColumnTransformer] = None
) -> Pipeline:
    """
    Create a simple pipeline with optional preprocessor.

    This is a convenience function for creating pipelines when you want to
    manually define the preprocessor.

    Parameters:
    -----------
    classifier : sklearn estimator
        Classifier model
    preprocessor : sklearn.compose.ColumnTransformer, optional
        Preprocessing pipeline. If None, only classifier is included.

    Returns:
    --------
    pipe : sklearn.pipeline.Pipeline
        Created pipeline

    Example:
    --------
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
    >>>
    >>> preprocessor = ColumnTransformer([
    ...     ('num', StandardScaler(), [0, 1, 2]),
    ...     ('cat', OneHotEncoder(), [3, 4])
    ... ])
    >>>
    >>> pipe = create_simple_pipeline(
    ...     classifier=LGBMClassifier(),
    ...     preprocessor=preprocessor
    ... )
    """
    if preprocessor is None:
        return Pipeline([
            ('classifier', classifier)
        ])
    else:
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])


def create_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
    classifier,
    numerical_transformer=None,
    categorical_transformer=None
) -> Pipeline:
    """
    Create a sklearn Pipeline using column names (recommended for use with DataFrames).

    This function creates a pipeline that uses feature names directly, making it
    compatible with DataFrame inputs and avoiding sklearn warnings about missing feature names.

    Parameters:
    -----------
    numerical_features : list of str
        List of numerical feature column names
    categorical_features : list of str
        List of categorical feature column names
    classifier : sklearn estimator
        Classifier model (e.g., LGBMClassifier, RandomForestClassifier)
    numerical_transformer : sklearn transformer, optional
        Transformer for numerical features (default: StandardScaler())
    categorical_transformer : sklearn transformer, optional
        Transformer for categorical features (default: OneHotEncoder(drop='first', handle_unknown='ignore'))

    Returns:
    --------
    pipe : sklearn.pipeline.Pipeline
        Created pipeline with preprocessing and classifier

    Example:
    --------
    >>> from lightgbm import LGBMClassifier
    >>>
    >>> pipe = create_pipeline(
    ...     numerical_features=['Age', 'Income'],
    ...     categorical_features=['Gender', 'Education'],
    ...     classifier=LGBMClassifier(random_state=42)
    ... )
    >>>
    >>> # Train with DataFrame
    >>> pipe.fit(X_train, y_train)
    >>>
    >>> # Create Model wrapper
    >>> from xai_cola.ce_sparsifier.models import Model
    >>> ml_model = Model(model=pipe, backend="sklearn")
    """
    # Set default transformers if not provided
    if numerical_transformer is None:
        numerical_transformer = StandardScaler()

    if categorical_transformer is None:
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    # Create preprocessor using column names
    transformers = []
    if numerical_features:
        transformers.append(('num', numerical_transformer, numerical_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any columns not specified
    )

    # Create Pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipe


def ensure_column_order(
    df: pd.DataFrame,
    feature_order: List[str],
    include_target: bool = False,
    target_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Ensure DataFrame columns are in the correct order.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    feature_order : list of str
        Desired feature column order
    include_target : bool, default=False
        Whether to include target column at the end
    target_column : str, optional
        Name of target column (required if include_target=True)

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns in correct order

    Raises:
    -------
    ValueError
        If target_column is not provided when include_target=True

    Example:
    --------
    >>> # Reorder features only
    >>> df_ordered = ensure_column_order(df, feature_order=['Age', 'Income', 'Gender'])
    >>>
    >>> # Reorder features with target at the end
    >>> df_ordered = ensure_column_order(
    ...     df,
    ...     feature_order=['Age', 'Income', 'Gender'],
    ...     include_target=True,
    ...     target_column='Risk'
    ... )
    """
    if include_target:
        if target_column is None:
            raise ValueError("target_column must be provided when include_target=True")
        column_order = feature_order + [target_column]
    else:
        column_order = feature_order

    return df[column_order]
