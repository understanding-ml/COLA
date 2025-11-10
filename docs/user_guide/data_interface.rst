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
    df = pd.DataFrame({
        'Age': [25, 35, 45],
        'Income': [30000, 50000, 70000],
        'HasLoan': ['No', 'Yes', 'No'],
        'Risk': [1, 0, 1]  # Target column
    })

    # Create COLAData instance
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Income']
    )

    # Check the data
    data.summary()

Output:

.. code-block:: text

    ========== COLAData Summary ==========
    Factual data shape: (3, 4)
    Label column: Risk

    Feature columns (3):
      - Age
      - Income
      - HasLoan

    Numerical features (2): Age, Income
    Categorical features (1): HasLoan

    Counterfactual data: Not set
    ======================================

Scenario 2: Adding Counterfactuals
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
    Factual data shape: (3, 4)
    Counterfactual data shape: (6, 4)  # 3 instances × 2 CFs each
    Label column: Risk
    ...

Scenario 3: Using NumPy Arrays
------------------------------

If you're working with NumPy arrays instead of DataFrames:

.. code-block:: python

    import numpy as np

    # Prepare numpy array (must include label column)
    X = np.array([
        [25, 30000, 0, 1],  # Last column is label
        [35, 50000, 1, 0],
        [45, 70000, 0, 1]
    ])

    # MUST provide column names when using numpy
    data = COLAData(
        factual_data=X,
        label_column='Risk',
        column_names=['Age', 'Income', 'HasLoan', 'Risk'],
        numerical_features=['Age', 'Income']
    )

.. warning::
    When using NumPy arrays, you **must** provide ``column_names`` that includes the label column.

Scenario 4: With Preprocessing
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

Advanced Features
================

Accessing Data
--------------

Get the underlying DataFrames:

.. code-block:: python

    # Get factual data
    factual_df = data.factual_df

    # Get counterfactual data (if set)
    cf_df = data.counterfactual_df

    # Get feature columns (excluding label)
    feature_names = data.feature_columns

    # Get numerical/categorical features
    num_features = data.numerical_features
    cat_features = data.categorical_features

Feature Type Inference
----------------------

If you don't specify ``numerical_features``, COLA will try to infer them:

.. code-block:: python

    # Without specifying numerical_features
    data = COLAData(
        factual_data=df,
        label_column='Risk'
    )

    # COLA infers based on dtype
    # int/float → numerical
    # object/category → categorical

.. warning::
    Automatic inference may not always be correct. For best results, explicitly specify ``numerical_features``.

Data Validation
---------------

``COLAData`` automatically validates:

1. **Label column exists** in the data
2. **Column names match** between factual and counterfactual
3. **Feature types are consistent** across datasets
4. **Data shapes are compatible** for matching

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

Complete Example
================

Here's a complete workflow:

.. code-block:: python

    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_generator import DiCE
    from xai_cola import COLA
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    import pandas as pd

    # 1. Load your data
    df = pd.read_csv('data.csv')

    # 2. Define feature types
    numerical_features = ['Age', 'Income', 'Duration']
    categorical_features = ['Gender', 'HasLoan']

    # 3. Create preprocessor (optional)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', 'passthrough', categorical_features)
        ]
    )
    preprocessor.fit(df[numerical_features + categorical_features])

    # 4. Create COLAData
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=numerical_features,
        preprocessor=preprocessor
    )

    # 5. Check data
    data.summary()

    # 6. Generate counterfactuals
    explainer = DiCE(ml_model=your_model)
    _, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2
    )

    # 7. Add counterfactuals
    data.add_counterfactuals(cf, with_target_column=True)

    # 8. Use with COLA
    sparsifier = COLA(data=data, ml_model=your_model)
    sparsifier.set_policy(matcher='ot', attributor='pshap')

    # 9. Get refined counterfactuals
    refined = sparsifier.refine_counterfactuals(limited_actions=5)

API Reference
=============

For complete parameter details, see :class:`~xai_cola.ce_sparsifier.data.COLAData`.

Next Steps
==========

- Learn about :doc:`models` - Wrapping your ML model
- Explore :doc:`explainers` - Generating counterfactuals
- Continue to :doc:`matching_policies` - Configuring COLA
