===========
Quick Start
===========

Get started with COLA in 5 minutes! This guide shows you how to refine counterfactual explanations with minimal code.

Installation
============

.. code-block:: bash

    pip install xai-cola

Basic Workflow
==============

COLA follows a simple 5-step workflow:

.. code-block:: text

    1. Load Data → 2. Train Model → 3. Generate CFs → 4. Refine with COLA → 5. Visualize

Complete Example
================

Here's a complete working example using the built-in German Credit dataset:

.. code-block:: python

    # Step 1: Import libraries
    from xai_cola import COLA
    from xai_cola.datasets.german_credit import GermanCreditDataset
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model
    from xai_cola.ce_generator import DiCE

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # Step 2: Load and prepare data
    dataset = GermanCreditDataset()
    X_train, y_train, X_test, y_test = dataset.get_original_train_test_split()

    # Step 3: Train a model
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    print(f"Model accuracy: {pipe.score(X_test, y_test):.3f}")

    # Step 4: Prepare COLA data interface
    numerical_features = ['Age', 'Credit amount', 'Duration']
    data = COLAData(
        factual_data=X_test,
        label_column='Risk',
        numerical_features=numerical_features
    )

    # Step 5: Wrap model for COLA
    ml_model = Model(model=pipe, backend="sklearn")

    # Step 6: Generate counterfactuals with DiCE
    explainer = DiCE(ml_model=ml_model)
    factual, counterfactual = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,  # Generate CFs for high-risk instances
        total_cfs=2,      # 2 CFs per instance
        continuous_features=numerical_features
    )

    # Step 7: Add counterfactuals to data
    data.add_counterfactuals(counterfactual, with_target_column=True)

    # Step 8: Initialize COLA and set policy
    sparsifier = COLA(data=data, ml_model=ml_model)
    sparsifier.set_policy(
        matcher="ot",       # Optimal transport matching
        attributor="pshap", # PSHAP for feature attribution
        random_state=42     # For reproducibility
    )

    # Step 9: Query minimum actions needed
    min_actions = sparsifier.query_minimum_actions()
    print(f"Minimum actions needed: {min_actions}")

    # Step 10: Refine counterfactuals
    refined_cf = sparsifier.refine_counterfactuals(limited_actions=min_actions)
    print(f"✓ Refined {len(refined_cf)} counterfactuals!")

    # Step 11: Compare results
    factual_df, ce_df, ace_df = sparsifier.get_all_results(
        limited_actions=min_actions
    )
    original_changes = (factual_df != ce_df).sum().sum()
    refined_changes = (factual_df != ace_df).sum().sum()
    print(f"Original CF: {original_changes} feature changes")
    print(f"Refined ACE: {refined_changes} feature changes")
    print(f"Reduction: {original_changes - refined_changes} fewer changes!")

    # Step 12: Visualize results
    sparsifier.heatmap_direction(save_path='./results')
    sparsifier.stacked_bar_chart(save_path='./results')
    print("✓ Visualizations saved to ./results/")

Expected Output
===============

.. code-block:: text

    Model accuracy: 0.730
    Minimum actions needed: 3
    ✓ Refined 40 counterfactuals!
    Original CF: 120 feature changes
    Refined ACE: 60 feature changes
    Reduction: 60 fewer changes!
    ✓ Visualizations saved to ./results/

Breaking It Down
================

Step-by-Step Explanation
-------------------------

**Steps 1-3: Standard ML Workflow**

Train your model as usual. COLA works with any sklearn-compatible model.

**Step 4-5: Data and Model Interface**

.. code-block:: python

    # Wrap your data
    data = COLAData(
        factual_data=df,
        label_column='target',
        numerical_features=['Age', 'Income']
    )

    # Wrap your model
    ml_model = Model(model=your_model, backend="sklearn")

**Step 6-7: Generate Counterfactuals**

.. code-block:: python

    # Use any CF explainer (DiCE, DisCount, Alibi, etc.)
    explainer = DiCE(ml_model=ml_model)
    _, cf = explainer.generate_counterfactuals(...)

    # Add to data
    data.add_counterfactuals(cf, with_target_column=True)

**Step 8-10: Refine with COLA**

.. code-block:: python

    # Initialize COLA
    sparsifier = COLA(data=data, ml_model=ml_model)

    # Set policy
    sparsifier.set_policy(matcher="ot", attributor="pshap")

    # Refine
    refined = sparsifier.refine_counterfactuals(limited_actions=5)

**Step 11-12: Analyze and Visualize**

.. code-block:: python

    # Get all results
    factual, ce, ace = sparsifier.get_all_results(limited_actions=5)

    # Visualize
    sparsifier.heatmap_direction(save_path='./results')
    sparsifier.stacked_bar_chart(save_path='./results')

Minimal Example
===============

Absolute minimum code:

.. code-block:: python

    from xai_cola import COLA
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model
    from xai_cola.ce_generator import DiCE

    # Assuming you have: df (data), trained_model

    # 1. Prepare
    data = COLAData(factual_data=df, label_column='target')
    ml_model = Model(model=trained_model, backend="sklearn")

    # 2. Generate CFs
    explainer = DiCE(ml_model=ml_model)
    _, cf = explainer.generate_counterfactuals(data=data, factual_class=1)
    data.add_counterfactuals(cf, with_target_column=True)

    # 3. Refine
    sparsifier = COLA(data=data, ml_model=ml_model)
    sparsifier.set_policy(matcher="ot", attributor="pshap")
    refined = sparsifier.refine_counterfactuals(limited_actions=5)

    # Done!

Using Your Own Data
===================

Replace the German Credit dataset with your own:

.. code-block:: python

    import pandas as pd

    # Load your data
    df = pd.read_csv('your_data.csv')

    # Split into train/test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    X_train = train_df.drop('target_column', axis=1)
    y_train = train_df['target_column']
    X_test = test_df.drop('target_column', axis=1)
    y_test = test_df['target_column']

    # Define your numerical features
    numerical_features = ['feature1', 'feature2', 'feature3']

    # Continue with COLA workflow...
    data = COLAData(
        factual_data=X_test,
        label_column='target_column',
        numerical_features=numerical_features
    )

    # Rest of the code remains the same

Using Different Models
======================

Scikit-learn
------------

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    ml_model = Model(model=clf, backend="sklearn")

PyTorch
-------

.. code-block:: python

    import torch.nn as nn

    class MyModel(nn.Module):
        # ... your model definition ...

    model = MyModel()
    # ... train your model ...

    ml_model = Model(model=model, backend="pytorch")

TensorFlow/Keras
----------------

.. code-block:: python

    import tensorflow as tf

    model = tf.keras.Sequential([...])
    model.compile(...)
    model.fit(X_train, y_train)

    ml_model = Model(model=model, backend="TF2")

Common Variations
=================

Using ECT Matcher (Faster)
---------------------------

.. code-block:: python

    # For faster results, use ECT instead of OT
    sparsifier.set_policy(matcher="ect", attributor="pshap")

With Feature Restrictions
--------------------------

.. code-block:: python

    # Only allow certain features to change
    refined = sparsifier.refine_counterfactuals(
        limited_actions=5,
        features_to_vary=['Income', 'LoanAmount', 'Duration']
    )

Using DisCount Instead of DiCE
-------------------------------

.. code-block:: python

    from xai_cola.ce_generator import DisCount

    explainer = DisCount(ml_model=ml_model)
    _, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        cost_type='L1'
    )

Jupyter Notebook Tips
=====================

For better visualization in Jupyter:

.. code-block:: python

    from IPython.display import display

    # Display highlighted DataFrames
    factual_style, ce_style, ace_style = sparsifier.highlight_changes_final()
    display(ce_style)
    display(ace_style)

    # Display inline figures
    %matplotlib inline
    fig = sparsifier.heatmap_direction()

Troubleshooting
===============

**Error: "No counterfactuals found"**

Solution: Relax constraints or increase ``total_cfs``

.. code-block:: python

    # Increase CFs per instance
    explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=5  # More CFs = higher success rate
    )

**Error: "Must call set_policy before refining"**

Solution: Always call ``set_policy()`` before ``refine_counterfactuals()``

.. code-block:: python

    # Don't forget this!
    sparsifier.set_policy(matcher="ot", attributor="pshap")

**Error: "Counterfactual data not set"**

Solution: Add counterfactuals to COLAData before creating COLA

.. code-block:: python

    # Must do this before creating COLA
    data.add_counterfactuals(cf, with_target_column=True)
    sparsifier = COLA(data=data, ml_model=ml_model)

Next Steps
==========

Now that you've completed the quick start:

1. :doc:`tutorials/01_basic_tutorial` - Detailed tutorial with explanations
2. :doc:`user_guide/data_interface` - Learn about data management
3. :doc:`user_guide/explainers` - Explore different CF generators
4. :doc:`user_guide/matching_policies` - Understand matching strategies
5. :doc:`user_guide/visualization` - Master visualization tools

Resources
=========

- :doc:`installation` - Installation guide
- :doc:`faq` - Frequently asked questions
- :doc:`api/cola` - Complete API reference
- `GitHub Examples <https://github.com/understanding-ml/COLA/tree/main/examples>`_ - More examples

Getting Help
============

- :doc:`faq` - Check common questions
- `GitHub Issues <https://github.com/understanding-ml/COLA/issues>`_ - Report bugs
- Contact: leiyo@dtu.dk, s232291@dtu.dk
