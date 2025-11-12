"""
COLA Data Module - Unified Data Interface

Supports Pandas DataFrame and NumPy array inputs
Automatically handles target column management
"""

from typing import List, Optional, Union
import pandas as pd
import numpy as np

try:
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class COLAData:
    """
    COLA Unified Data Interface

    Manages both factual and counterfactual data simultaneously
    Automatically validates data consistency

    Parameters:
    -----------
    factual_data : Union[pd.DataFrame, np.ndarray]
        Factual data (original data), must include label column
        If DataFrame: checks if label_column exists
        If numpy: must provide all column names (including label_column)

    label_column : str
        Label column name, should be in the last column by default

    counterfactual_data : Optional[Union[pd.DataFrame, np.ndarray]]
        Counterfactual data (optional)
        If DataFrame: checks column consistency with factual
        If numpy: uses factual column names

    column_names : Optional[List[str]]
        Required only when factual_data is numpy
        Provide all column names (including label_column), order must match

    numerical_features : Optional[List[str]], default=None
        List of numerical features. Used to record which features are continuous numeric.
        If None, all features are assumed to be numerical by default.
        Other features are automatically inferred as categorical.
        Note: This parameter is only used to record feature type information, no data transformation is performed.

    transform_method : Optional[object], default=None
        Data preprocessor (e.g., sklearn's StandardScaler, ColumnTransformer, etc.)
        Must have transform() and inverse_transform() methods
        Used for data transformation before and after generating counterfactuals

    preprocessor : Optional[object], default=None
        **Deprecated.** Alias for transform_method, kept for backward compatibility.
        Use transform_method instead.
    """

    def __init__(
        self,
        factual_data: Union[pd.DataFrame, np.ndarray],
        label_column: str,
        counterfactual_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        column_names: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        transform_method: Optional[object] = None,
        preprocessor: Optional[object] = None
    ):
        # Validate and set label column
        self.label_column = label_column

        # Set data preprocessor (transform_method and preprocessor are aliases, use one)
        if transform_method is not None and preprocessor is not None:
            raise ValueError("Cannot specify both 'transform_method' and 'preprocessor'. Use transform_method only.")

        # preprocessor is a deprecated alias, prefer transform_method
        if preprocessor is not None:
            import warnings
            warnings.warn(
                "'preprocessor' parameter is deprecated and will be removed in a future version. "
                "Use 'transform_method' instead.",
                DeprecationWarning,
                stacklevel=2
            )

        self.transform_method = transform_method if transform_method is not None else preprocessor

        # Validate that transform_method has necessary methods
        if self.transform_method is not None:
            if not hasattr(self.transform_method, 'transform'):
                raise ValueError("transform_method must have a 'transform()' method")

            # For ColumnTransformer, we implement custom inverse_transform internally
            # So no need to check inverse_transform method here
            # For other transformers, only check when actually called
            if not (SKLEARN_AVAILABLE and isinstance(self.transform_method, ColumnTransformer)):
                # Non-ColumnTransformer needs to have inverse_transform
                if not hasattr(self.transform_method, 'inverse_transform'):
                    raise ValueError(
                        "transform_method must have an 'inverse_transform()' method. "
                        "ColumnTransformer is supported with custom inverse_transform logic."
                    )

        # Process factual data
        self.factual_df = self._process_input_data(
            factual_data,
            data_type='factual',
            column_names=column_names,
            reference_df=None
        )

        # Set numerical_features (only for recording feature type information)
        self.numerical_features = numerical_features if numerical_features is not None else []

        # If numerical_features is explicitly provided, treat remaining features as categorical
        # and convert their values to string type to avoid warnings or errors due to type
        # mismatches in subsequent interactions (e.g., DiCE generating counterfactuals).
        # Do not convert the label column.
        if self.numerical_features:
            try:
                categorical_cols = [
                    col for col in self.get_feature_columns() if col not in self.numerical_features
                ]
                for col in categorical_cols:
                    if col in self.factual_df.columns:
                        # Convert to string to avoid pandas warnings from int/str mixing
                        try:
                            self.factual_df[col] = self.factual_df[col].astype(str)
                        except Exception:
                            # If conversion fails (very rare), skip this column
                            pass
            except Exception:
                # Fault tolerance: any exception should not block COLAData construction
                pass

        # Process counterfactual data (if provided)
        self.counterfactual_df = None
        if counterfactual_data is not None:
            self.add_counterfactuals(counterfactual_data)

        # ========== Transformed Data Storage ==========
        # If transform_method is set, automatically compute and store transformed data
        self.transformed_factual_df = None
        self.transformed_counterfactual_df = None
        self.transformed_column_order = None  # Column order after transformation (ColumnTransformer changes column order)

        if self.transform_method is not None:
            # Transform factual data
            factual_features = self.get_factual_features()
            self.transformed_factual_df = self._transform(factual_features)

            # Record transformed column order
            self.transformed_column_order = self.transformed_factual_df.columns.tolist()

            # If counterfactual data already exists, transform it as well
            if self.counterfactual_df is not None:
                counterfactual_features = self.get_counterfactual_features()
                self.transformed_counterfactual_df = self._transform(counterfactual_features)
    
    def _process_input_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        data_type: str,
        column_names: Optional[List[str]] = None,
        reference_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Process input data and convert to DataFrame

        Parameters:
        -----------
        data : Union[pd.DataFrame, np.ndarray]
            Input data
        data_type : str
            'factual' or 'counterfactual'
        column_names : Optional[List[str]]
            Column names (required only for numpy)
        reference_df : Optional[pd.DataFrame]
            Reference DataFrame (for validating counterfactual)

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()

            # If factual, validate that label column exists
            if data_type == 'factual':
                if self.label_column not in df.columns:
                    raise ValueError(
                        f"Label column '{self.label_column}' not found in factual data. "
                        f"Available columns: {df.columns.tolist()}"
                    )

            # If counterfactual, validate column consistency
            elif data_type == 'counterfactual' and reference_df is not None:
                expected_cols = reference_df.columns.tolist()
                actual_cols = df.columns.tolist()
                if expected_cols != actual_cols:
                    raise ValueError(
                        f"Counterfactual columns must match factual columns.\n"
                        f"Expected: {expected_cols}\n"
                        f"Got: {actual_cols}"
                    )

            return df


        elif isinstance(data, np.ndarray):
            if column_names is None and reference_df is None:
                raise ValueError(
                    "When providing numpy array, you must either:\n"
                    "1. Provide column_names parameter (for factual)\n"
                    "2. Provide counterfactual using add_counterfactuals() (uses factual columns)"
                )

            # Use provided column names or reference DataFrame's column names
            if column_names is not None:
                if len(column_names) != data.shape[1]:
                    raise ValueError(
                        f"Number of column_names ({len(column_names)}) doesn't match "
                        f"data shape ({data.shape[1]} columns)"
                    )
                columns = column_names
            elif reference_df is not None:
                columns = reference_df.columns.tolist()
                if len(columns) != data.shape[1]:
                    raise ValueError(
                        f"Counterfactual shape ({data.shape[1]} columns) doesn't match "
                        f"factual shape ({len(columns)} columns)"
                    )
            else:
                raise ValueError("Must provide column_names")

            return pd.DataFrame(data, columns=columns)
        
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                f"Supported types: pd.DataFrame, np.ndarray"
            )
    
    def add_counterfactuals(
        self,
        counterfactual_data: Union[pd.DataFrame, np.ndarray],
        with_target_column: bool = True
    ):
        """
        Add or update counterfactual data

        Parameters:
        -----------
        counterfactual_data : Union[pd.DataFrame, np.ndarray]
            Counterfactual data
            If DataFrame: checks column consistency with factual (depends on with_target_column)
            If numpy: uses factual column names (depends on with_target_column)
        with_target_column : bool, default=False
            If True: counterfactual_data includes target column, same number of columns as factual
            If False: counterfactual_data does not include target column, only feature columns
                     In this case, will automatically reverse values from factual's target column (0->1, 1->0) and add

        Raises:
        -------
        ValueError
            If with_target_column=False and factual and counterfactual have different number of rows
        """
        if with_target_column:
            # Counterfactual includes target column, processing logic same as before
            self.counterfactual_df = self._process_input_data(
                counterfactual_data,
                data_type='counterfactual',
                reference_df=self.factual_df
            )
            # If numerical_features is specified, also convert counterfactual's categorical features to strings
            if self.numerical_features:
                try:
                    categorical_cols = [
                        col for col in self.get_feature_columns() if col not in self.numerical_features
                    ]
                    for col in categorical_cols:
                        if col in self.counterfactual_df.columns:
                            try:
                                self.counterfactual_df[col] = self.counterfactual_df[col].astype(str)
                            except Exception:
                                pass
                except Exception:
                    pass
        else:
            # Counterfactual does not include target column
            # First process feature data
            if isinstance(counterfactual_data, pd.DataFrame):
                cf_features_df = counterfactual_data.copy()
                # Check if target column is unexpectedly included
                if self.label_column in cf_features_df.columns:
                    raise ValueError(
                        f"Counterfactual data contains target column '{self.label_column}', "
                        f"but with_target_column=False. "
                        f"Either remove the target column from counterfactual data "
                        f"or set with_target_column=True."
                    )
            elif isinstance(counterfactual_data, np.ndarray):
                # numpy array, should be feature data
                feature_columns = self.get_feature_columns()
                if counterfactual_data.shape[1] != len(feature_columns):
                    # Check if target column is included
                    if counterfactual_data.shape[1] == len(self.get_all_columns()):
                        raise ValueError(
                            f"Counterfactual numpy array has {counterfactual_data.shape[1]} columns, "
                            f"which matches all columns (including target). "
                            f"Set with_target_column=True if counterfactual includes target column, "
                            f"or provide only {len(feature_columns)} feature columns."
                        )
                    else:
                        raise ValueError(
                            f"Counterfactual shape ({counterfactual_data.shape[1]} columns) doesn't match "
                            f"expected feature columns ({len(feature_columns)})."
                        )
                cf_features_df = pd.DataFrame(counterfactual_data, columns=feature_columns)
            else:
                raise TypeError(
                    f"Unsupported data type: {type(counterfactual_data)}. "
                    f"Supported types: pd.DataFrame, np.ndarray"
                )


            # Validate row count consistency
            if len(cf_features_df) != len(self.factual_df):
                raise ValueError(
                    f"Factual and counterfactual must have the same number of rows. "
                    f"Factual: {len(self.factual_df)} rows, "
                    f"Counterfactual: {len(cf_features_df)} rows."
                )

            # Get factual's target column values and reverse them (0->1, 1->0)
            factual_labels = self.get_factual_labels()
            reversed_labels = 1 - factual_labels  # Reverse: 0->1, 1->0

            # Create complete counterfactual DataFrame (including target column)
            self.counterfactual_df = cf_features_df.copy()
            self.counterfactual_df[self.label_column] = reversed_labels.values

            # Ensure column order matches factual
            self.counterfactual_df = self.counterfactual_df[self.get_all_columns()]

        # If transform_method is set, also transform counterfactual data
        if self.transform_method is not None:
            counterfactual_features = self.get_counterfactual_features()
            self.transformed_counterfactual_df = self._transform(counterfactual_features)

    # ========== Column Name Related Methods ==========
    
    def get_all_columns(self) -> List[str]:
        """
        Get all column names (including label column)

        Returns:
        --------
        List[str]
            List of all column names
        """
        return self.factual_df.columns.tolist()

    def get_feature_columns(self) -> List[str]:
        """
        Get feature column names (excluding label column)

        Returns:
        --------
        List[str]
            List of feature column names
        """
        return [col for col in self.factual_df.columns if col != self.label_column]

    def get_label_column(self) -> str:
        """
        Get label column name

        Returns:
        --------
        str
            Label column name
        """
        return self.label_column

    def get_numerical_features(self) -> List[str]:
        """
        Get numerical feature list

        Returns:
        --------
        List[str]
            List of numerical feature column names
        """
        return self.numerical_features.copy() if self.numerical_features else []

    def get_categorical_features(self) -> List[str]:
        """
        Get categorical feature list (all non-numerical features)

        Returns:
        --------
        List[str]
            List of categorical feature column names
        """
        feature_columns = self.get_feature_columns()
        if not self.numerical_features:
            # If no numerical_features specified, assume all features are numerical, return empty list
            return []
        return [col for col in feature_columns if col not in self.numerical_features]

    def get_transformed_feature_columns(self) -> Optional[List[str]]:
        """
        Get transformed feature column names

        For ColumnTransformer, column order becomes [numerical_features, categorical_features]
        For other transformers, column order remains unchanged

        Returns:
        --------
        Optional[List[str]]
            List of transformed feature column names, returns None if transform_method is not set
        """
        if self.transform_method is None:
            return None
        return self.transformed_column_order

    # ========== Factual Data Methods ==========
    
    def get_factual_all(self) -> pd.DataFrame:
        """
        Get complete factual DataFrame including label column

        Returns:
        --------
        pd.DataFrame
            Complete factual data (including label column)
        """
        return self.factual_df.copy()

    def get_factual_features(self) -> pd.DataFrame:
        """
        Get factual feature data excluding label column

        Returns:
        --------
        pd.DataFrame
            Factual feature data (excluding label column)
        """
        return self.factual_df.drop(columns=[self.label_column]).copy()

    def get_factual_labels(self) -> pd.Series:
        """
        Get factual label column

        Returns:
        --------
        pd.Series
            Factual label column
        """
        return self.factual_df[self.label_column].copy()

    def get_transformed_factual_features(self) -> Optional[pd.DataFrame]:
        """
        Get transformed factual feature data

        Returns:
        --------
        Optional[pd.DataFrame]
            Transformed factual feature data, returns None if transform_method is not set

        Example:
        --------
        >>> data = COLAData(df, label_column='Risk', transform_method=scaler)
        >>> transformed = data.get_transformed_factual_features()
        >>> # For calculating Shapley values or other computations based on transformed data
        """
        if self.transform_method is None:
            return None
        return self.transformed_factual_df.copy()

    def has_transformed_data(self) -> bool:
        """
        Check if transformed data exists

        Returns:
        --------
        bool
            Returns True if transform_method is set and transformed data exists
        """
        return self.transformed_factual_df is not None

    # ========== Counterfactual Data Methods ==========
    
    def get_counterfactual_all(self) -> pd.DataFrame:
        """
        Get complete counterfactual DataFrame including label column

        Returns:
        --------
        pd.DataFrame
            Complete counterfactual data (including label column)

        Raises:
        -------
        ValueError
            If counterfactual data has not been set
        """
        if self.counterfactual_df is None:
            raise ValueError("Counterfactual data has not been set. Use add_counterfactuals() first.")
        return self.counterfactual_df.copy()

    def get_counterfactual_features(self) -> pd.DataFrame:
        """
        Get counterfactual feature data excluding label column

        Returns:
        --------
        pd.DataFrame
            Counterfactual feature data (excluding label column)

        Raises:
        -------
        ValueError
            If counterfactual data has not been set
        """
        if self.counterfactual_df is None:
            raise ValueError("Counterfactual data has not been set. Use add_counterfactuals() first.")
        return self.counterfactual_df.drop(columns=[self.label_column]).copy()

    def get_counterfactual_labels(self) -> pd.Series:
        """
        Get counterfactual label column

        Returns:
        --------
        pd.Series
            Counterfactual label column

        Raises:
        -------
        ValueError
            If counterfactual data has not been set
        """
        if self.counterfactual_df is None:
            raise ValueError("Counterfactual data has not been set. Use add_counterfactuals() first.")
        return self.counterfactual_df[self.label_column].copy()

    def get_transformed_counterfactual_features(self) -> Optional[pd.DataFrame]:
        """
        Get transformed counterfactual feature data

        Returns:
        --------
        Optional[pd.DataFrame]
            Transformed counterfactual feature data, returns None if transform_method or counterfactual is not set

        Raises:
        -------
        ValueError
            If transform_method is set but counterfactual data has not been set

        Example:
        --------
        >>> data = COLAData(df, label_column='Risk', transform_method=scaler)
        >>> data.add_counterfactuals(cf_df)
        >>> transformed_cf = data.get_transformed_counterfactual_features()
        >>> # For calculating matching or Q in transformed space
        """
        if self.transform_method is None:
            return None
        if self.counterfactual_df is None:
            raise ValueError("Counterfactual data has not been set. Use add_counterfactuals() first.")
        return self.transformed_counterfactual_df.copy()

    # ========== Convenience Methods ==========
    
    def has_counterfactual(self) -> bool:
        """
        Check if counterfactual data has been set

        Returns:
        --------
        bool
            Returns True if counterfactual data exists
        """
        return self.counterfactual_df is not None

    def get_feature_count(self) -> int:
        """
        Get number of features (excluding label column)

        Returns:
        --------
        int
            Number of features
        """
        return len(self.get_feature_columns())

    def get_sample_count(self) -> int:
        """
        Get number of samples

        Returns:
        --------
        int
            Number of samples
        """
        return len(self.factual_df)

    # ========== NumPy Conversion Methods ==========
    
    def to_numpy_factual_features(self) -> np.ndarray:
        """
        Convert factual features to NumPy array

        Returns:
        --------
        np.ndarray
            Factual feature matrix
        """
        return self.get_factual_features().values

    def to_numpy_counterfactual_features(self) -> np.ndarray:
        """
        Convert counterfactual features to NumPy array

        Returns:
        --------
        np.ndarray
            Counterfactual feature matrix

        Raises:
        -------
        ValueError
            If counterfactual data has not been set
        """
        return self.get_counterfactual_features().values

    def to_numpy_labels(self) -> np.ndarray:
        """
        Convert labels to NumPy array

        Returns:
        --------
        np.ndarray
            Label array
        """
        return self.get_factual_labels().values

    def to_numpy_transformed_factual_features(self) -> Optional[np.ndarray]:
        """
        Convert transformed factual features to NumPy array

        Returns:
        --------
        Optional[np.ndarray]
            Transformed factual feature matrix, returns None if transform_method is not set

        Example:
        --------
        >>> data = COLAData(df, label_column='Risk', transform_method=scaler)
        >>> X_transformed = data.to_numpy_transformed_factual_features()
        >>> # Calculate Shapley values in transformed space
        """
        if self.transform_method is None:
            return None
        return self.transformed_factual_df.values

    def to_numpy_transformed_counterfactual_features(self) -> Optional[np.ndarray]:
        """
        Convert transformed counterfactual features to NumPy array

        Returns:
        --------
        Optional[np.ndarray]
            Transformed counterfactual feature matrix, returns None if transform_method or counterfactual is not set

        Raises:
        -------
        ValueError
            If transform_method is set but counterfactual data has not been set

        Example:
        --------
        >>> data = COLAData(df, label_column='Risk', transform_method=scaler)
        >>> data.add_counterfactuals(cf_df)
        >>> CF_transformed = data.to_numpy_transformed_counterfactual_features()
        >>> # Calculate matching distance in transformed space
        """
        if self.transform_method is None:
            return None
        if self.counterfactual_df is None:
            raise ValueError("Counterfactual data has not been set. Use add_counterfactuals() first.")
        return self.transformed_counterfactual_df.values

    # ========== Information Methods ==========
    
    def __repr__(self) -> str:
        """String representation"""
        cf_info = f", counterfactual: {len(self.counterfactual_df)} rows" if self.counterfactual_df is not None else ", no counterfactual"
        return (
            f"COLAData(factual: {len(self.factual_df)} rows, "
            f"features: {self.get_feature_count()}, "
            f"label: {self.label_column}{cf_info})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def summary(self) -> dict:
        """
        Get data summary information

        Returns:
        --------
        dict
            Dictionary containing data summary
        """
        info = {
            'factual_samples': len(self.factual_df),
            'feature_count': self.get_feature_count(),
            'label_column': self.label_column,
            'all_columns': self.get_all_columns(),
            'has_counterfactual': self.has_counterfactual(),
            'has_transform_method': self.transform_method is not None,
            'has_transformed_data': self.has_transformed_data()
        }

        if self.counterfactual_df is not None:
            info['counterfactual_samples'] = len(self.counterfactual_df)

        if self.has_transformed_data():
            info['transformed_feature_columns'] = self.get_transformed_feature_columns()
            info['has_transformed_counterfactual'] = self.transformed_counterfactual_df is not None

        return info

    # ========== Data Transformation Methods ==========

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data transformation (using transform_method)

        Parameters:
        -----------
        data : pd.DataFrame
            Data to transform (only feature columns, excluding label column)

        Returns:
        --------
        pd.DataFrame
            Transformed data

        Raises:
        -------
        ValueError
            If transform_method is not set

        Example:
        --------
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> scaler.fit(X_train)
        >>> data = COLAData(
        ...     factual_data=df,
        ...     label_column='Risk',
        ...     transform_method=scaler
        ... )
        >>> transformed = data._transform(data.get_factual_features())
        """
        if self.transform_method is None:
            raise ValueError("No transform_method is set. Cannot transform data.")

        # Apply transformation
        # For ColumnTransformer, pass DataFrame to select features based on column names
        if SKLEARN_AVAILABLE and isinstance(self.transform_method, ColumnTransformer):
            transformed_values = self.transform_method.transform(data)  # Pass DataFrame

            # ColumnTransformer output order is [numerical_features, categorical_features]
            # Not the original data column order
            transformed_column_order = self.numerical_features + self.get_categorical_features()

            transformed_df = pd.DataFrame(
                transformed_values,
                columns=transformed_column_order,  # Use transformed column order
                index=data.index
            )
        else:
            transformed_values = self.transform_method.transform(data.values)  # Pass numpy array

            # Create DataFrame, preserve original column names and index
            transformed_df = pd.DataFrame(
                transformed_values,
                columns=data.columns,
                index=data.index
            )

        return transformed_df

    def _inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inverse transformation (using transform_method's inverse_transform)

        Parameters:
        -----------
        data : pd.DataFrame
            Data to inverse transform (only feature columns, excluding label column)

        Returns:
        --------
        pd.DataFrame
            Inverse transformed data

        Raises:
        -------
        ValueError
            If transform_method is not set

        Example:
        --------
        >>> # Assuming data has been standardized
        >>> original = data._inverse_transform(transformed_data)

        Notes:
        ------
        For ColumnTransformer (including categorical encoders like OrdinalEncoder), this method intelligently handles:
        - Automatically detects ColumnTransformer and separates numerical and categorical features
        - Rounds categorical features before inverse transformation to avoid floating point errors
        - Correctly reorganizes feature order
        """
        if self.transform_method is None:
            raise ValueError("No transform_method is set. Cannot inverse transform data.")

        # Check if it's a ColumnTransformer
        if SKLEARN_AVAILABLE and isinstance(self.transform_method, ColumnTransformer):
            return self._inverse_transform_column_transformer(data)
        else:
            # Standard inverse transformation (for simple transformers like StandardScaler)
            inverse_transformed_values = self.transform_method.inverse_transform(data.values)

            # Create DataFrame, preserve column names and index
            inverse_transformed_df = pd.DataFrame(
                inverse_transformed_values,
                columns=data.columns,
                index=data.index
            )

            return inverse_transformed_df

    def _inverse_transform_column_transformer(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Specifically handles inverse transformation for ColumnTransformer

        This method solves the floating point issue when ColumnTransformer inverse transforms categorical features:
        - Separates numerical and categorical features
        - Rounds categorical features (if needed)
        - Inverse transforms separately then recombines

        Supports:
        - Simple transformers (StandardScaler, OrdinalEncoder)
        - Pipeline transformers (e.g., OrdinalEncoder + StandardScaler)

        Parameters:
        -----------
        data : pd.DataFrame
            Transformed data

        Returns:
        --------
        pd.DataFrame
            Inverse transformed data
        """
        transformer = self.transform_method

        # Get each sub-transformer
        num_transformer = transformer.named_transformers_.get('num')
        cat_transformer = transformer.named_transformers_.get('cat')

        n_num = len(self.numerical_features)
        n_cat = len(self.get_categorical_features())

        # Convert data to numpy array
        X_transformed = data.values

        # Separate numerical and categorical features
        results = []
        feature_names = []

        # Process numerical features
        if n_num > 0 and num_transformer is not None:
            X_num_scaled = X_transformed[:, :n_num]
            X_num_original = num_transformer.inverse_transform(X_num_scaled)
            results.append(X_num_original)
            feature_names.extend(self.numerical_features)

        # Process categorical features
        if n_cat > 0 and cat_transformer is not None:
            X_cat_encoded = X_transformed[:, n_num:n_num + n_cat]

            # Check if cat_transformer is a Pipeline
            is_pipeline = hasattr(cat_transformer, 'named_steps')

            if is_pipeline:
                # Pipeline case: typically OrdinalEncoder -> StandardScaler
                # We need to manually inverse transform, cannot use Pipeline.inverse_transform()
                # Because Pipeline.inverse_transform() automatically executes all steps,
                # directly returning strings, unable to insert round/clip logic
                #
                # Correct steps:
                # 1. StandardScaler.inverse_transform() → get floating point encoding
                # 2. round + clip → get valid integer encoding
                # 3. OrdinalEncoder.inverse_transform() → get original strings

                # Get each step in the Pipeline
                ordinal_encoder = None
                scaler = None

                for step_name, step_transformer in cat_transformer.named_steps.items():
                    if hasattr(step_transformer, 'categories_'):
                        ordinal_encoder = step_transformer
                    elif hasattr(step_transformer, 'mean_'):  # StandardScaler has mean_ attribute
                        scaler = step_transformer

                # Step 1: StandardScaler inverse transform (from standardized space → encoding space)
                if scaler is not None:
                    X_cat_after_scaler = scaler.inverse_transform(X_cat_encoded)
                else:
                    # If no scaler, use original data directly
                    X_cat_after_scaler = X_cat_encoded

                # Step 2: Round and limit to valid range
                X_cat_rounded = np.round(X_cat_after_scaler)

                if ordinal_encoder is not None:
                    # Limit encoding values to valid range
                    for i in range(X_cat_rounded.shape[1]):
                        n_categories = len(ordinal_encoder.categories_[i])
                        X_cat_rounded[:, i] = np.clip(X_cat_rounded[:, i], 0, n_categories - 1)

                    # Step 3: OrdinalEncoder inverse transform (from encoding → original strings)
                    X_cat_original = ordinal_encoder.inverse_transform(X_cat_rounded)
                else:
                    # If no OrdinalEncoder, directly use rounded result
                    X_cat_original = X_cat_rounded
            else:
                # Non-Pipeline case: directly OrdinalEncoder
                # Round categorical features to avoid floating point inverse transform errors
                X_cat_encoded_rounded = np.round(X_cat_encoded)

                # Limit encoding values to valid range (OrdinalEncoder encoding range is 0 to n_categories-1)
                if hasattr(cat_transformer, 'categories_'):
                    for i in range(X_cat_encoded_rounded.shape[1]):
                        n_categories = len(cat_transformer.categories_[i])
                        X_cat_encoded_rounded[:, i] = np.clip(X_cat_encoded_rounded[:, i], 0, n_categories - 1)

                # Inverse transform categorical features
                X_cat_original = cat_transformer.inverse_transform(X_cat_encoded_rounded)

            results.append(X_cat_original)
            feature_names.extend(self.get_categorical_features())

        # Merge results
        if len(results) == 0:
            raise ValueError("No features to inverse transform in ColumnTransformer")

        X_original = np.hstack(results) if len(results) > 1 else results[0]

        # Create DataFrame
        # Note: Since _transform() already ensures column order is [numerical_features, categorical_features]
        # feature_names order here is also [numerical_features, categorical_features]
        inverse_transformed_df = pd.DataFrame(
            X_original,
            columns=feature_names,
            index=data.index
        )

        # Reorder columns to match original data column order
        original_feature_columns = self.get_feature_columns()
        inverse_transformed_df = inverse_transformed_df[original_feature_columns]

        return inverse_transformed_df

