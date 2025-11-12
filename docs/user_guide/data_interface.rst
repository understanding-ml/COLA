==================
Data Interface
==================

Overview
========

The ``COLAData`` class is the central data container in COLA. It manages both factual (original) and counterfactual (generated) data, ensuring consistency and providing convenient methods for data manipulation and inspection.

Key Features
------------

- **Unified Interface**: Works with both Pandas DataFrame and NumPy arrays
- **Automatic Validation**: Ensures data consistency between factual and counterfactual
- **Feature Type Tracking**: Distinguishes between numerical and categorical features
- **Preprocessor Integration**: Supports sklearn transformers for data preprocessing
- **Summary Information**: Quick overview of your data structure

When to Use COLAData
===================

Use ``COLAData`` when you need to:

- Organize factual instances and their counterfactual explanations together
- Automatically track which features are numerical vs categorical
- Integrate with preprocessing pipelines (StandardScaler, OneHotEncoder, etc.)
- Ensure data consistency before refinement

Basic Usage
===========

Scenario 1: Using Pandas DataFrame
----------------------------------

The most common and recommended approach:

.. code-block:: python

    from xai_cola.ce_sparsifier.data import COLAData
    import pandas as pd

    # Prepare your data with a target column
    df # dataframe of factual, including label column 'Risk'

    # Create COLAData instance
    data = COLAData(
        factual_data=df, # pandas dataframe
        label_column='Risk',
        numerical_features=['Age', 'Income']
    )

    # Check the data
    data.summary()

Output:

.. code-block:: text

    ========== COLAData Summary ==========
    {'factual_samples': 10,
    'feature_count': 9,
    'label_column': 'Risk',
    'all_columns': ['Age',
    'Sex',
    'Job',
    'Housing',
    'Saving accounts',
    'Checking account',
    'Credit amount',
    'Duration',
    'Purpose',
    'Risk'],
    'has_counterfactual': True,
    'has_transform_method': False,
    'has_transformed_data': False,
    'counterfactual_samples': None}
    ======================================


Scenario 2: Using NumPy Arrays
------------------------------

If you're working with NumPy arrays instead of DataFrames:

.. code-block:: python

    import numpy as np

    # Prepare numpy array (must include label column)
    X  # numpy array of factual, including label column 'Risk'
    all_columns = ['Age','Sex','Job','Housing','Saving accounts','Checking account','Credit amount','Duration','Purpose','Risk']
    # MUST provide column names when using numpy
    data = COLAData(
        factual_data=X, # numpy array
        label_column='Risk',
        column_names=all_columns,
        numerical_features=['Age', 'Income']
    )

.. warning::
    When using NumPy arrays, you **must** provide ``column_names`` that includes the label column.


Scenario 3: Adding Counterfactuals after initializing COLAData
----------------------------------

After generating counterfactuals using DiCE, DisCount, or another explainer:

.. code-block:: python

    # Generate counterfactuals (example with DiCE)
    from xai_cola.ce_generator import DiCE

    explainer = DiCE(ml_model=ml_model)
    factual, counterfactual = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2
    )

    # Add counterfactuals to COLAData
    data.add_counterfactuals(counterfactual, with_target_column=True)

    # Now summary shows both
    data.summary()

Output:

.. code-block:: text

    ========== COLAData Summary ==========
    {'factual_samples': 10,
    'feature_count': 9,
    'label_column': 'Risk',
    'all_columns': ['Age',
    'Sex',
    'Job',
    'Housing',
    'Saving accounts',
    'Checking account',
    'Credit amount',
    'Duration',
    'Purpose',
    'Risk'],
    'has_counterfactual': True,
    'has_transform_method': False,
    'has_transformed_data': False,
    'counterfactual_samples': 20}
    ...


Scenario 4: Adding Counterfactuals when initializing COLAData
----------------------------------

After generating counterfactuals using DiCE, DisCount, or another explainer:

.. code-block:: python

    from xai_cola.ce_sparsifier.data import COLAData
    import pandas as pd

    # Prepare your data with a target column
    df # dataframe of factual, including label column 'Risk'
    cf_df # dataframe of counterfactuals, including label column 'Risk'
    # Create COLAData instance
    data = COLAData(
        factual_data=df, # pandas dataframe
        label_column='Risk',
        counterfactual_data=cf_df, # pandas dataframe of counterfactuals
        numerical_features=['Age', 'Income']
    )

    # Check the data
    data.summary()

Output:

.. code-block:: text

    ========== COLAData Summary ==========
    {'factual_samples': 10,
    'feature_count': 9,
    'label_column': 'Risk',
    'all_columns': ['Age',
    'Sex',
    'Job',
    'Housing',
    'Saving accounts',
    'Checking account',
    'Credit amount',
    'Duration',
    'Purpose',
    'Risk'],
    'has_counterfactual': True,
    'has_transform_method': False,
    'has_transformed_data': False,
    'counterfactual_samples': 20}
    ...

Scenario 5: With Preprocessing
-------------------------------

Integrate sklearn preprocessors for automatic transformation:

.. code-block:: python

    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age', 'Income']),
            ('cat', OneHotEncoder(drop='first'), ['HasLoan'])
        ]
    )

    # Fit the preprocessor
    preprocessor.fit(df[['Age', 'Income', 'HasLoan']])

    # Create COLAData with preprocessor
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Income'],
        preprocessor=preprocessor
    )

.. note::
    ``transform_method`` and ``preprocessor`` are aliases - use either one.


Common Issues
=============

Issue 1: Missing Label Column
------------------------------

**Error:**

.. code-block:: text

    KeyError: "Label column 'Risk' not found in data"

**Solution:**

Make sure your DataFrame includes the label column:

.. code-block:: python

    # ❌ Wrong - label column missing
    df = pd.DataFrame({'Age': [25], 'Income': [30000]})
    data = COLAData(factual_data=df, label_column='Risk')

    # ✅ Correct - label column included
    df = pd.DataFrame({'Age': [25], 'Income': [30000], 'Risk': [1]})
    data = COLAData(factual_data=df, label_column='Risk')

Issue 2: Column Mismatch
-------------------------

**Error:**

.. code-block:: text

    ValueError: Counterfactual columns don't match factual columns

**Solution:**

Ensure counterfactuals have the same columns as factuals:

.. code-block:: python

    # Check column names
    print(data.factual_df.columns)
    print(counterfactual_df.columns)

    # Make sure they match (order doesn't matter)

Issue 3: NumPy Without Column Names
------------------------------------

**Error:**

.. code-block:: text

    ValueError: column_names must be provided when using numpy array

**Solution:**

Always provide column names for NumPy arrays:

.. code-block:: python

    # ❌ Wrong
    data = COLAData(factual_data=X, label_column='Risk')

    # ✅ Correct
    data = COLAData(
        factual_data=X,
        label_column='Risk',
        column_names=['Age', 'Income', 'Risk']
    )

Best Practices
==============

✅ **DO:**

1. **Always specify numerical_features explicitly**

   .. code-block:: python

       data = COLAData(
           factual_data=df,
           label_column='Risk',
           numerical_features=['Age', 'Income', 'Duration']
       )

2. **Use Pandas DataFrames when possible** - easier to debug

3. **Add counterfactuals before using COLA**

   .. code-block:: python

       data.add_counterfactuals(cf_df, with_target_column=True)
       sparsifier = COLA(data=data, ml_model=model)

4. **Check summary after creation**

   .. code-block:: python

       data.summary()  # Verify everything looks correct

❌ **DON'T:**

1. **Don't modify factual_df directly after adding counterfactuals**

   .. code-block:: python

       data.add_counterfactuals(cf_df)
       data.factual_df['NewColumn'] = 0  # ❌ Don't do this

2. **Don't forget the label column** in your data

3. **Don't assume feature types** - always specify numerical_features

4. **Don't mix preprocessed and raw data** - be consistent


API Reference
=============

For complete parameter details, see :class:`~xai_cola.ce_sparsifier.data.COLAData`.

Next Steps
==========

- Learn about :doc:`models` - Wrapping your ML model
- Explore :doc:`explainers` - Generating counterfactuals
- Continue to :doc:`matching_policies` - Configuring COLA
