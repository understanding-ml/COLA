===============
Data API
===============

.. currentmodule:: xai_cola.ce_sparsifier.data

COLAData Class
==============

.. autoclass:: COLAData
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Data container for factual and counterfactual explanations.

   COLAData manages both factual (original) and counterfactual data, handles
   preprocessing transformations, and provides convenient access methods for
   features, labels, and metadata.

   **Key Responsibilities:**

   - Store and manage factual and counterfactual data
   - Track feature types (numerical vs categorical)
   - Handle data transformations and inverse transformations
   - Provide consistent data access interface
   - Support both DataFrame and NumPy array inputs

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~COLAData.__init__
      ~COLAData.add_counterfactuals
      ~COLAData.get_all_columns
      ~COLAData.get_feature_columns
      ~COLAData.get_label_column
      ~COLAData.get_numerical_features
      ~COLAData.get_categorical_features
      ~COLAData.get_factual_all
      ~COLAData.get_factual_features
      ~COLAData.get_factual_labels
      ~COLAData.get_counterfactual_all
      ~COLAData.get_counterfactual_features
      ~COLAData.get_counterfactual_labels
      ~COLAData.get_transformed_factual_features
      ~COLAData.get_transformed_counterfactual_features
      ~COLAData.to_numpy_factual_features
      ~COLAData.to_numpy_counterfactual_features
      ~COLAData.has_counterfactual
      ~COLAData.has_transformed_data
      ~COLAData.get_feature_count
      ~COLAData.get_sample_count
      ~COLAData.summary

Constructor
-----------

.. automethod:: COLAData.__init__

Adding Counterfactuals
-----------------------

.. automethod:: COLAData.add_counterfactuals

Column Information Methods
--------------------------

.. automethod:: COLAData.get_all_columns

   Get all column names including the label column.

.. automethod:: COLAData.get_feature_columns

   Get feature column names (excludes label).

.. automethod:: COLAData.get_label_column

   Get the label column name.

.. automethod:: COLAData.get_numerical_features

   Get list of numerical feature names.

.. automethod:: COLAData.get_categorical_features

   Get list of categorical feature names.

Factual Data Access Methods
----------------------------

.. automethod:: COLAData.get_factual_all

   Get complete factual DataFrame (features + label).

.. automethod:: COLAData.get_factual_features

   Get factual features only (excludes label).

.. automethod:: COLAData.get_factual_labels

   Get factual labels as pandas Series.

.. automethod:: COLAData.get_transformed_factual_features

   Get preprocessed/transformed factual features.
   Returns None if no preprocessor is set.

.. automethod:: COLAData.to_numpy_factual_features

   Get factual features as NumPy array.

Counterfactual Data Access Methods
-----------------------------------

.. automethod:: COLAData.get_counterfactual_all

   Get complete counterfactual DataFrame (features + label).

.. automethod:: COLAData.get_counterfactual_features

   Get counterfactual features only (excludes label).

.. automethod:: COLAData.get_counterfactual_labels

   Get counterfactual labels as pandas Series.

.. automethod:: COLAData.get_transformed_counterfactual_features

   Get preprocessed/transformed counterfactual features.
   Returns None if no preprocessor is set.

.. automethod:: COLAData.to_numpy_counterfactual_features

   Get counterfactual features as NumPy array.

Helper Methods
--------------

.. automethod:: COLAData.has_counterfactual

   Check if counterfactual data has been added.

.. automethod:: COLAData.has_transformed_data

   Check if preprocessor/transformer is available.

.. automethod:: COLAData.get_feature_count

   Get number of features (excludes label).

.. automethod:: COLAData.get_sample_count

   Get number of samples in factual data.

.. automethod:: COLAData.summary

   Print comprehensive data summary.

Attributes
==========

.. attribute:: COLAData.factual_df

   Pandas DataFrame containing factual (original) data with label column.

   :type: pandas.DataFrame

.. attribute:: COLAData.counterfactual_df

   Pandas DataFrame containing counterfactual data. ``None`` until ``add_counterfactuals()`` is called.

   :type: pandas.DataFrame or None

.. attribute:: COLAData.label_column

   Name of the target/label column.

   :type: str

.. attribute:: COLAData.feature_columns

   List of feature column names (excludes label column).

   :type: list of str

.. attribute:: COLAData.numerical_features

   List of numerical feature names.

   :type: list of str

.. attribute:: COLAData.categorical_features

   List of categorical feature names (automatically inferred from non-numerical features).

   :type: list of str

.. attribute:: COLAData.transform_method

   Optional preprocessor/transformer with ``transform()`` and ``inverse_transform()`` methods.

   :type: object or None

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
    data.summary()

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

With Preprocessor
-----------------

.. code-block:: python

    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    # Create preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Age', 'Income']),
        ('cat', 'passthrough', ['HasLoan'])
    ])
    preprocessor.fit(df[['Age', 'Income', 'HasLoan']])

    # Initialize with preprocessor
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Income'],
        preprocessor=preprocessor
    )

Adding Counterfactuals
-----------------------

.. code-block:: python

    # After generating counterfactuals
    data.add_counterfactuals(cf_df, with_target_column=True)

    # Now can access both
    print(data.factual_df.shape)
    print(data.counterfactual_df.shape)

Accessing Data
--------------

.. code-block:: python

    # Get factual data
    factual = data.factual_df

    # Get counterfactual data
    if data.counterfactual_df is not None:
        cf = data.counterfactual_df

    # Get feature names
    features = data.feature_columns
    num_features = data.numerical_features
    cat_features = data.categorical_features

See Also
========

- :doc:`../user_guide/data_interface` - Detailed usage guide
- :doc:`models` - Model interface
- :doc:`cola` - COLA main class
