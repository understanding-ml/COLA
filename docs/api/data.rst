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

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~COLAData.__init__
      ~COLAData.add_counterfactuals
      ~COLAData.summary
      ~COLAData.get_factual_data
      ~COLAData.get_counterfactual_data

Constructor
-----------

.. automethod:: COLAData.__init__

Adding Counterfactuals
-----------------------

.. automethod:: COLAData.add_counterfactuals

Data Access
-----------

.. automethod:: COLAData.get_factual_data

.. automethod:: COLAData.get_counterfactual_data

Utility Methods
---------------

.. automethod:: COLAData.summary

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
