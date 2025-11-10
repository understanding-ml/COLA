============================
Counterfactual Explainers
============================

Overview
========

COLA provides built-in counterfactual explanation generators and supports integration with external explainers. These explainers generate the initial counterfactuals that COLA then refines to minimize the number of required actions.

Built-in Explainers
===================

COLA includes two main explainers:

1. **DiCE** - Instance-wise counterfactual generation (Diverse Counterfactual Explanations)
2. **DisCount** - Distributional counterfactual generation (Distribution-aware Counterfactuals)

DiCE Explainer
==============

DiCE generates multiple diverse counterfactuals for individual instances.

**When to use DiCE:**

- You want diverse counterfactual options for each instance
- You need instance-level explanations
- You want to control which features can be changed
- You care about proximity and sparsity

Basic Usage
-----------

.. code-block:: python

    from xai_cola.ce_generator import DiCE
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # 1. Prepare data
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Income']
    )

    # 2. Wrap model
    ml_model = Model(model=trained_model, backend="sklearn")

    # 3. Create explainer
    explainer = DiCE(ml_model=ml_model)

    # 4. Generate counterfactuals
    factual, counterfactual = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,        # Generate CFs for instances with class 1
        total_cfs=2,            # Generate 2 CFs per instance
        continuous_features=['Age', 'Income']
    )

    # 5. Add to data
    data.add_counterfactuals(counterfactual, with_target_column=True)

Parameters Explained
--------------------

**factual_class** (*int or list*)
    Which class to generate counterfactuals for.

    .. code-block:: python

        # Generate CFs for class 1 only
        factual_class=1

        # Generate CFs for multiple classes
        factual_class=[0, 1]

**total_cfs** (*int*)
    Number of counterfactuals per instance.

    .. code-block:: python

        total_cfs=1   # One CF per instance (faster)
        total_cfs=5   # Five CFs per instance (more diverse)

**continuous_features** (*list*)
    Features that can take continuous values.

    .. code-block:: python

        continuous_features=['Age', 'Income', 'Duration']

**features_to_keep** (*list*)
    Features that should NOT be changed (immutable features).

    .. code-block:: python

        # Don't change Age or Gender
        features_to_keep=['Age', 'Gender']

**features_to_vary** (*list*)
    Only these features can be changed.

    .. code-block:: python

        # Only allow changing Income and Duration
        features_to_vary=['Income', 'Duration']

    .. note::
        Use either ``features_to_keep`` OR ``features_to_vary``, not both.

**permitted_range** (*dict*)
    Allowed ranges for numerical features.

    .. code-block:: python

        permitted_range={
            'Age': [18, 65],      # Age must be between 18-65
            'Income': [0, None]   # Income must be non-negative
        }

Advanced DiCE Usage
-------------------

Example 1: With Immutable Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Scenario: Age and Gender cannot be changed
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=3,
        features_to_keep=['Age', 'Gender'],
        continuous_features=['Income', 'Duration']
    )

Example 2: With Feature Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Scenario: Realistic constraints on features
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        continuous_features=['Age', 'Income', 'LoanAmount'],
        permitted_range={
            'Age': [18, 70],
            'Income': [10000, 200000],
            'LoanAmount': [1000, 50000]
        }
    )

Example 3: Selective Feature Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Scenario: Only allow financial features to change
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        features_to_vary=['Income', 'LoanAmount', 'Duration'],
        continuous_features=['Income', 'LoanAmount', 'Duration']
    )

DisCount Explainer
==================

DisCount generates distributional counterfactuals - it finds a counterfactual distribution that maintains similar structure to the factual distribution.

**When to use DisCount:**

- You have groups of instances to explain
- You care about distributional properties
- You want cost-efficient group-level changes
- You need to maintain data distribution shape

Basic Usage
-----------

.. code-block:: python

    from xai_cola.ce_generator import DisCount
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # 1. Prepare data
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Income']
    )

    # 2. Wrap model
    ml_model = Model(model=trained_model, backend="sklearn")

    # 3. Create explainer
    explainer = DisCount(ml_model=ml_model)

    # 4. Generate distributional counterfactuals
    factual, counterfactual = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        cost_type='L1',           # Cost metric
        continuous_features=['Age', 'Income']
    )

    # 5. Add to data
    data.add_counterfactuals(counterfactual, with_target_column=True)

Parameters Explained
--------------------

**cost_type** (*str*)
    Distance metric for optimization. Options: ``'L1'``, ``'L2'``

    .. code-block:: python

        cost_type='L1'  # Manhattan distance
        cost_type='L2'  # Euclidean distance

**features_to_keep** / **features_to_vary**
    Same as DiCE - control which features can change.

**continuous_features**
    Numerical features in the data.

External Explainers
===================

You can use COLA with any counterfactual explainer that produces DataFrames or arrays.

Using Alibi Explainers
-----------------------

.. code-block:: python

    from alibi.explainers import CounterfactualProto
    import numpy as np

    # 1. Create Alibi explainer
    cf_explainer = CounterfactualProto(
        predict_fn=model.predict_proba,
        shape=(1, n_features)
    )
    cf_explainer.fit(X_train)

    # 2. Generate counterfactuals
    explanation = cf_explainer.explain(X_test[:10])

    # 3. Extract counterfactual data
    cf_array = explanation.cf['X']  # Shape: (n_instances, n_features)
    cf_df = pd.DataFrame(cf_array, columns=feature_names)

    # 4. Use with COLA
    data = COLAData(factual_data=X_test[:10], label_column='target')
    data.add_counterfactuals(cf_df, with_target_column=False)

Using Custom Explainers
------------------------

Any explainer that outputs DataFrames or arrays works:

.. code-block:: python

    # Your custom explainer
    def my_explainer(X, model):
        # ... your counterfactual generation logic ...
        return counterfactuals_df

    # Generate CFs
    cf_df = my_explainer(X_test, model)

    # Use with COLA
    data = COLAData(factual_data=X_test, label_column='y')
    data.add_counterfactuals(cf_df, with_target_column=True)

    # Refine with COLA
    from xai_cola import COLA
    sparsifier = COLA(data=data, ml_model=ml_model)
    refined = sparsifier.refine_counterfactuals(limited_actions=5)

Complete Workflow Example
==========================

End-to-End with DiCE
--------------------

.. code-block:: python

    from xai_cola.datasets.german_credit import GermanCreditDataset
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model
    from xai_cola.ce_generator import DiCE
    from xai_cola import COLA

    # 1. Load data
    dataset = GermanCreditDataset()
    X_train, y_train, X_test, y_test = dataset.get_original_train_test_split()

    # 2. Train model
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])
    pipe.fit(X_train, y_train)

    # 3. Prepare COLA components
    numerical = ['Age', 'Credit amount', 'Duration']
    data = COLAData(
        factual_data=X_test,
        label_column='Risk',
        numerical_features=numerical
    )
    ml_model = Model(model=pipe, backend="sklearn")

    # 4. Generate counterfactuals with DiCE
    explainer = DiCE(ml_model=ml_model)
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        features_to_keep=['Age', 'Sex'],
        continuous_features=numerical
    )

    # 5. Add counterfactuals
    data.add_counterfactuals(cf, with_target_column=True)

    # 6. Refine with COLA
    sparsifier = COLA(data=data, ml_model=ml_model)
    sparsifier.set_policy(matcher='ot', attributor='pshap')

    min_actions = sparsifier.query_minimum_actions()
    print(f"Minimum actions needed: {min_actions}")

    refined_cf = sparsifier.refine_counterfactuals(limited_actions=min_actions)

    # 7. Visualize results
    sparsifier.heatmap_direction(save_path='./results')
    fig = sparsifier.stacked_bar_chart(save_path='./results')

Common Issues
=============

Issue 1: No Counterfactuals Found
----------------------------------

**Error:**

.. code-block:: text

    ValueError: No valid counterfactuals found

**Possible causes:**

1. **Too many immutable features** - relax ``features_to_keep``
2. **Too strict ranges** - widen ``permitted_range``
3. **Model is too confident** - increase ``total_cfs`` or adjust proximity weight

**Solutions:**

.. code-block:: python

    # ❌ Too restrictive
    explainer.generate_counterfactuals(
        data=data,
        features_to_keep=['Age', 'Gender', 'Income', 'Job'],  # Too many!
        permitted_range={'Duration': [1, 2]}  # Too narrow!
    )

    # ✅ More flexible
    explainer.generate_counterfactuals(
        data=data,
        features_to_keep=['Age', 'Gender'],  # Only truly immutable
        permitted_range={'Duration': [1, 60]}  # Reasonable range
    )

Issue 2: Counterfactuals Too Far
---------------------------------

**Problem:** Generated counterfactuals are unrealistic or too different.

**Solution:** Adjust proximity weight or use permitted ranges:

.. code-block:: python

    # Use tighter ranges
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=1,
        permitted_range={
            'Income': [factual['Income'].min() * 0.8,
                      factual['Income'].max() * 1.2]
        }
    )

Issue 3: Shape Mismatch
------------------------

**Error:**

.. code-block:: text

    ValueError: Factual and counterfactual have different number of columns

**Cause:** Counterfactual DataFrame doesn't match factual structure.

**Solution:** Ensure column consistency:

.. code-block:: python

    # Check columns
    print("Factual columns:", data.factual_df.columns.tolist())
    print("CF columns:", cf_df.columns.tolist())

    # Make sure they match (order doesn't matter, but names must)
    assert set(data.factual_df.columns) == set(cf_df.columns)

Best Practices
==============

✅ **DO:**

1. **Start with fewer CFs** for faster iteration

   .. code-block:: python

       # Start with 1 CF per instance
       total_cfs=1

       # Increase later if needed
       total_cfs=5

2. **Always specify continuous_features**

   .. code-block:: python

       continuous_features=['Age', 'Income', 'Duration']

3. **Use realistic feature constraints**

   .. code-block:: python

       features_to_keep=['Age', 'Gender']  # Truly immutable
       permitted_range={'Income': [0, 500000]}  # Realistic bounds

4. **Verify counterfactuals before refinement**

   .. code-block:: python

       # Check CF predictions
       cf_preds = ml_model.predict(cf_df.drop('Risk', axis=1))
       print("CF predictions:", cf_preds)
       print("Desired class:", desired_class)

❌ **DON'T:**

1. **Don't use too many immutable features**
2. **Don't forget to add counterfactuals to COLAData**

   .. code-block:: python

       # ❌ Forgot this step
       factual, cf = explainer.generate_counterfactuals(...)
       sparsifier = COLA(data=data, ml_model=ml_model)  # Error!

       # ✅ Remember to add CFs
       data.add_counterfactuals(cf, with_target_column=True)
       sparsifier = COLA(data=data, ml_model=ml_model)  # Works!

3. **Don't mix continuous and categorical in continuous_features**

API Reference
=============

For complete parameter details, see:

- :class:`~xai_cola.ce_generator.DiCE`
- :class:`~xai_cola.ce_generator.DisCount`

Next Steps
==========

- Learn about :doc:`matching_policies` - Configuring COLA refinement
- Explore :doc:`visualization` - Visualizing results
- See :doc:`data_interface` - Managing data
