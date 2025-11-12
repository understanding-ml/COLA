===============
Data API
===============

.. currentmodule:: xai_cola.ce_sparsifier.data

Module contents
===============

class **COLAData** (factual_data, label_column, counterfactual_data=None, column_names=None, numerical_features=None, transform_method=None, preprocessor=None)

   **Bases:** ``object``

   COLA unified data interface - Data container for factual and counterfactual data

   Supports managing both factual and counterfactual data simultaneously with automatic
   validation of data consistency.

   **Parameters:**
      * **factual_data** (*Union[pd.DataFrame, np.ndarray]*) -- Factual data (original data), must include label column.
        If DataFrame: checks if label_column exists.
        If numpy: requires column_names (including label_column).
      * **label_column** (*str*) -- Label column name, should be the last column by default.
      * **counterfactual_data** (*Optional[Union[pd.DataFrame, np.ndarray]]**, **optional*) -- Counterfactual data (optional).
        If DataFrame: checks if columns match factual.
        If numpy: uses factual's column names.
      * **column_names** (*Optional[List[str]]**, **optional*) -- Required only when factual_data is numpy.
        Provide all column names (including label_column), order must match.
      * **numerical_features** (*Optional[List[str]]**, **optional*) -- List of numerical features. Used to record which features are continuous numerical.
        If None, defaults to all features being numerical.
        Other features are automatically inferred as categorical.
        Note: This parameter only records feature type information, does not perform data conversion. Default is None.
      * **transform_method** (*Optional[object]**, **optional*) -- Data preprocessor (e.g., sklearn's StandardScaler, ColumnTransformer, etc.).
        Must have transform() and inverse_transform() methods.
        Used to perform data transformation before and after generating counterfactuals.
        **Recommended parameter to use.** Default is None.
      * **preprocessor** (*Optional[object]**, **optional*) -- **Deprecated alias** for transform_method.
        Kept for backward compatibility. Use transform_method instead. Default is None.

   **Raises:**
      **ValueError** -- If both transform_method and preprocessor are specified, or if required parameters are missing.

   **Example:**

   .. code-block:: python

      import pandas as pd
      from xai_cola.ce_sparsifier.data import COLAData

      # Create DataFrame with label column
      df = pd.DataFrame({
          'Age': [25, 35, 45],
          'Income': [30000, 50000, 70000],
          'HasLoan': ['No', 'Yes', 'No'],
          'Risk': [1, 0, 1]  # Label column
      })

      # Initialize COLAData
      data = COLAData(
          factual_data=df,
          label_column='Risk',
          numerical_features=['Age', 'Income']
      )

      # Add counterfactuals
      data.add_counterfactuals(cf_df, with_target_column=True)

      # Access data
      factual_features = data.get_factual_features()
      counterfactual_features = data.get_counterfactual_features()

   **add_counterfactuals** (counterfactual_data, with_target_column=True)

      Add or update counterfactual data.

      **Parameters:**
         * **counterfactual_data** (*Union[pd.DataFrame, np.ndarray]*) -- Counterfactual data.
           If DataFrame: checks if columns match factual (depends on with_target_column).
           If numpy: uses factual's column names (depends on with_target_column).
         * **with_target_column** (*bool**, **default=True*) --
           If True: counterfactual_data includes target column, same number of columns as factual.
           If False: counterfactual_data does not include target column, only feature columns.
           In this case, automatically reverses factual's target column values (0->1, 1->0) and adds them.

      **Raises:**
         **ValueError** -- If with_target_column=False and factual and counterfactual have inconsistent row counts.

   **get_all_columns** ()

      Get all column names (including label column).

      **Returns:**
         List of all column names

      **Return type:**
         List[str]

   **get_feature_columns** ()

      Get feature column names (excluding label column).

      **Returns:**
         List of feature column names

      **Return type:**
         List[str]

   **get_label_column** ()

      Get label column name.

      **Returns:**
         Label column name

      **Return type:**
         str

   **get_numerical_features** ()

      Get list of numerical features.

      **Returns:**
         List of numerical feature names

      **Return type:**
         List[str]

   **get_categorical_features** ()

      Get list of categorical features (all non-numerical features).

      **Returns:**
         List of categorical feature names

      **Return type:**
         List[str]

   **get_transformed_feature_columns** ()

      Get transformed feature column names.

      For ColumnTransformer, column order changes to [numerical_features, categorical_features].
      For other transformers, column order remains unchanged.

      **Returns:**
         List of transformed feature column names, or None if transform_method is not set

      **Return type:**
         Optional[List[str]]

   **get_factual_all** ()

      Get complete factual DataFrame including label column.

      **Returns:**
         Complete factual data (including label column)

      **Return type:**
         pd.DataFrame

   **get_factual_features** ()

      Get factual feature data excluding label column.

      **Returns:**
         Factual feature data (excluding label column)

      **Return type:**
         pd.DataFrame

   **get_factual_labels** ()

      Get factual label column.

      **Returns:**
         Factual label column

      **Return type:**
         pd.Series

   **get_transformed_factual_features** ()

      Get transformed factual feature data.

      **Returns:**
         Transformed factual feature data, or None if transform_method is not set

      **Return type:**
         Optional[pd.DataFrame]

      **Example:**

      .. code-block:: python

         data = COLAData(df, label_column='Risk', transform_method=scaler)
         transformed = data.get_transformed_factual_features()
         # Used for calculating Shapley values or other computations based on transformed data

   **get_counterfactual_all** ()

      Get complete counterfactual DataFrame including label column.

      **Returns:**
         Complete counterfactual data (including label column)

      **Return type:**
         pd.DataFrame

      **Raises:**
         **ValueError** -- If counterfactual data has not been set

   **get_counterfactual_features** ()

      Get counterfactual feature data excluding label column.

      **Returns:**
         Counterfactual feature data (excluding label column)

      **Return type:**
         pd.DataFrame

      **Raises:**
         **ValueError** -- If counterfactual data has not been set

   **get_counterfactual_labels** ()

      Get counterfactual label column.

      **Returns:**
         Counterfactual label column

      **Return type:**
         pd.Series

      **Raises:**
         **ValueError** -- If counterfactual data has not been set

   **get_transformed_counterfactual_features** ()

      Get transformed counterfactual feature data.

      **Returns:**
         Transformed counterfactual feature data, or None if transform_method or counterfactual is not set

      **Return type:**
         Optional[pd.DataFrame]

      **Raises:**
         **ValueError** -- If transform_method is set but counterfactual data has not been set

      **Example:**

      .. code-block:: python

         data = COLAData(df, label_column='Risk', transform_method=scaler)
         data.add_counterfactuals(cf_df)
         transformed_cf = data.get_transformed_counterfactual_features()
         # Used for calculating matching or Q in transformed space

   **has_counterfactual** ()

      Check if counterfactual data has been set.

      **Returns:**
         True if counterfactual data exists

      **Return type:**
         bool

   **has_transformed_data** ()

      Check if transformed data exists.

      **Returns:**
         True if transform_method is set and transformed data exists

      **Return type:**
         bool

   **get_feature_count** ()

      Get number of features (excluding label column).

      **Returns:**
         Number of features

      **Return type:**
         int

   **get_sample_count** ()

      Get number of samples.

      **Returns:**
         Number of samples

      **Return type:**
         int

   **to_numpy_factual_features** ()

      Convert factual features to NumPy array.

      **Returns:**
         Factual feature matrix

      **Return type:**
         np.ndarray

   **to_numpy_counterfactual_features** ()

      Convert counterfactual features to NumPy array.

      **Returns:**
         Counterfactual feature matrix

      **Return type:**
         np.ndarray

      **Raises:**
         **ValueError** -- If counterfactual data has not been set

   **to_numpy_labels** ()

      Convert labels to NumPy array.

      **Returns:**
         Label array

      **Return type:**
         np.ndarray

   **to_numpy_transformed_factual_features** ()

      Convert transformed factual features to NumPy array.

      **Returns:**
         Transformed factual feature matrix, or None if transform_method is not set

      **Return type:**
         Optional[np.ndarray]

      **Example:**

      .. code-block:: python

         data = COLAData(df, label_column='Risk', transform_method=scaler)
         X_transformed = data.to_numpy_transformed_factual_features()
         # Calculate Shapley values in transformed space

   **to_numpy_transformed_counterfactual_features** ()

      Convert transformed counterfactual features to NumPy array.

      **Returns:**
         Transformed counterfactual feature matrix, or None if transform_method or counterfactual is not set

      **Return type:**
         Optional[np.ndarray]

      **Raises:**
         **ValueError** -- If transform_method is set but counterfactual data has not been set

      **Example:**

      .. code-block:: python

         data = COLAData(df, label_column='Risk', transform_method=scaler)
         data.add_counterfactuals(cf_df)
         CF_transformed = data.to_numpy_transformed_counterfactual_features()
         # Calculate matching distance in transformed space

   **summary** ()

      Get data summary information.

      **Returns:**
         Dictionary containing data summary

      **Return type:**
         dict

      **Example:**

      .. code-block:: python

         data = COLAData(df, label_column='Risk')
         info = data.summary()
         print(info)
         # Output:
         # {
         #     'factual_samples': 100,
         #     'feature_count': 10,
         #     'label_column': 'Risk',
         #     'all_columns': ['Age', 'Income', ..., 'Risk'],
         #     'has_counterfactual': True,
         #     'has_transform_method': True,
         #     'has_transformed_data': True,
         #     'counterfactual_samples': 100,
         #     'transformed_feature_columns': ['Age', 'Income', ...],
         #     'has_transformed_counterfactual': True
         # }

Examples
========

Basic Usage with DataFrame
---------------------------

.. code-block:: python

   import pandas as pd
   from xai_cola.ce_sparsifier.data import COLAData

   # Create DataFrame with label column
   df = pd.DataFrame({
       'Age': [25, 35, 45],
       'Income': [30000, 50000, 70000],
       'HasLoan': ['No', 'Yes', 'No'],
       'Risk': [1, 0, 1]  # Label column
   })

   # Initialize COLAData
   data = COLAData(
       factual_data=df,
       label_column='Risk',
       numerical_features=['Age', 'Income']
   )

   # Check data
   print(data.summary())

With NumPy Arrays
-----------------

.. code-block:: python

   import numpy as np

   # NumPy array (must include label)
   X = np.array([
       [25, 30000, 0, 1],
       [35, 50000, 1, 0],
       [45, 70000, 0, 1]
   ])

   # Must provide column names
   data = COLAData(
       factual_data=X,
       label_column='Risk',
       column_names=['Age', 'Income', 'HasLoan', 'Risk'],
       numerical_features=['Age', 'Income']
   )

With Preprocessor (StandardScaler)
-----------------------------------

.. code-block:: python

   from sklearn.preprocessing import StandardScaler

   # Create and fit preprocessor
   scaler = StandardScaler()
   scaler.fit(df[['Age', 'Income', 'HasLoan']])

   # Initialize with preprocessor
   data = COLAData(
       factual_data=df,
       label_column='Risk',
       numerical_features=['Age', 'Income'],
       transform_method=scaler
   )

   # Access transformed data
   transformed = data.get_transformed_factual_features()

With ColumnTransformer
----------------------

.. code-block:: python

   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import StandardScaler, OrdinalEncoder

   # Create ColumnTransformer
   transformer = ColumnTransformer([
       ('num', StandardScaler(), ['Age', 'Income']),
       ('cat', OrdinalEncoder(), ['HasLoan'])
   ])
   transformer.fit(df[['Age', 'Income', 'HasLoan']])

   # Initialize with ColumnTransformer (use transform_method)
   data = COLAData(
       factual_data=df,
       label_column='Risk',
       numerical_features=['Age', 'Income'],
       transform_method=transformer  # Recommended: use transform_method
   )

   # Transformed column order: [numerical_features, categorical_features]
   print(data.get_transformed_feature_columns())
   # Output: ['Age', 'Income', 'HasLoan']

   # Note: preprocessor=transformer also works (backward compatibility)
   # but transform_method is recommended

Adding Counterfactuals (with target column)
--------------------------------------------

.. code-block:: python

   # Generate counterfactuals using any explainer (e.g., DiCE)
   cf_df = explainer.generate_counterfactuals(...)

   # Add counterfactuals (includes target column)
   data.add_counterfactuals(cf_df, with_target_column=True)

   # Now can access both
   print(data.get_factual_all().shape)
   print(data.get_counterfactual_all().shape)

Adding Counterfactuals (without target column)
-----------------------------------------------

.. code-block:: python

   # If counterfactual data only contains features (no target column)
   cf_features = cf_df[['Age', 'Income', 'HasLoan']]

   # Automatically reverses factual's target values (0->1, 1->0)
   data.add_counterfactuals(cf_features, with_target_column=False)

   # Target column is automatically added with reversed values
   print(data.get_counterfactual_all())

Accessing Data
--------------

.. code-block:: python

   # Get factual data
   factual_all = data.get_factual_all()          # DataFrame with label
   factual_features = data.get_factual_features()  # DataFrame without label
   factual_labels = data.get_factual_labels()      # Series

   # Get counterfactual data
   if data.has_counterfactual():
       cf_all = data.get_counterfactual_all()
       cf_features = data.get_counterfactual_features()
       cf_labels = data.get_counterfactual_labels()

   # Get transformed data
   if data.has_transformed_data():
       transformed_factual = data.get_transformed_factual_features()
       transformed_cf = data.get_transformed_counterfactual_features()

   # Convert to NumPy
   X_factual = data.to_numpy_factual_features()
   X_cf = data.to_numpy_counterfactual_features()
   y = data.to_numpy_labels()

Feature Information
-------------------

.. code-block:: python

   # Get column information
   all_columns = data.get_all_columns()           # ['Age', 'Income', 'HasLoan', 'Risk']
   feature_columns = data.get_feature_columns()   # ['Age', 'Income', 'HasLoan']
   label_column = data.get_label_column()         # 'Risk'

   # Get feature type information
   num_features = data.get_numerical_features()   # ['Age', 'Income']
   cat_features = data.get_categorical_features() # ['HasLoan']

   # Get counts
   n_features = data.get_feature_count()          # 3
   n_samples = data.get_sample_count()            # 100

See Also
========

- :doc:`cola` - COLA main class
- :doc:`models` - Model interface documentation
