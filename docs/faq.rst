==========================
Frequently Asked Questions
==========================

General Questions
=================

What is COLA?
-------------

COLA (COunterfactual explanations with Limited Actions) is a Python framework that refines counterfactual explanations by reducing the number of feature changes needed while maintaining the same outcome.

**Key benefits:**

- Reduces feature changes by 30-70%
- Works with any ML model
- Compatible with various CF explainers (DiCE, DisCount, Alibi, etc.)
- Provides rich visualizations

When should I use COLA?
------------------------

Use COLA when:

- You need **actionable** counterfactual explanations
- Users complain CFs require **too many changes**
- You want to provide **minimal action plans**
- You need to **compare** different CF methods
- You want **theoretically grounded** refinements

Installation & Setup
====================

How do I install COLA?
-----------------------

.. code-block:: bash

    pip install xai-cola

See :doc:`installation` for detailed instructions.

Which Python versions are supported?
-------------------------------------

Python 3.9 to 3.12 are supported.

Do I need PyTorch or TensorFlow?
---------------------------------

No, they're optional. COLA works with scikit-learn by default. Install PyTorch/TensorFlow only if you're using models from those frameworks.

Data & Models
=============

What data format does COLA accept?
-----------------------------------

COLA accepts:

- **Pandas DataFrame** (recommended)
- **NumPy arrays** (must provide column names)

.. code-block:: python

    # Option 1: DataFrame (easiest)
    data = COLAData(factual_data=df, label_column='target')

    # Option 2: NumPy array
    data = COLAData(
        factual_data=X,
        label_column='target',
        column_names=['feature1', 'feature2', 'target']
    )

Which ML frameworks are supported?
-----------------------------------

- ✅ **Scikit-learn** - All classifiers
- ✅ **PyTorch** - Neural networks
- ✅ **TensorFlow 1.x & 2.x** - Keras models
- ✅ **Any custom model** with ``predict()`` and ``predict_proba()`` methods

Can I use COLA with regression models?
---------------------------------------

Currently, COLA is designed for **classification** tasks only. Regression support is planned for future releases.

Does COLA work with multi-class classification?
------------------------------------------------

Yes! COLA supports both binary and multi-class classification.

.. code-block:: python

    # Works for any number of classes
    explainer.generate_counterfactuals(
        data=data,
        factual_class=2,  # Any class
        total_cfs=2
    )

Counterfactual Generation
==========================

Which CF explainers can I use?
-------------------------------

COLA includes:

- **DiCE** - Instance-wise CFs
- **DisCount** - Distributional CFs

External explainers also work:

- Alibi (CounterfactualProto, etc.)
- Any custom explainer that outputs DataFrames

.. code-block:: python

    # Use any explainer, then refine with COLA
    cf_df = your_explainer.generate(...)
    data.add_counterfactuals(cf_df)
    sparsifier = COLA(data=data, ml_model=ml_model)

How many counterfactuals should I generate per instance?
---------------------------------------------------------

**Recommendation:**

- Start with ``total_cfs=1`` for speed
- Use ``total_cfs=2-3`` for better quality
- Use ``total_cfs=5+`` for maximum diversity

.. code-block:: python

    # Balance between quality and speed
    explainer.generate_counterfactuals(
        data=data,
        total_cfs=2  # Good default
    )

More CFs give COLA more options, leading to potentially better refinements.

What if no counterfactuals are found?
--------------------------------------

**Common causes:**

1. Too many immutable features
2. Too strict feature ranges
3. Model is very confident

**Solutions:**

.. code-block:: python

    # Solution 1: Relax constraints
    explainer.generate_counterfactuals(
        features_to_keep=['Age'],  # Fewer immutable features
        permitted_range={'Income': [0, None]}  # Wider range
    )

    # Solution 2: Increase total_cfs
    explainer.generate_counterfactuals(total_cfs=5)

    # Solution 3: Use different desired_class
    explainer.generate_counterfactuals(factual_class=0)  # Try different class

COLA Refinement
===============

What does "limited_actions" mean?
----------------------------------

``limited_actions`` specifies the maximum number of features that can be changed in the refined counterfactual.

.. code-block:: python

    # Allow up to 5 feature changes
    refined = sparsifier.refine_counterfactuals(limited_actions=5)

Lower values = more sparse (fewer changes)
Higher values = less restrictive

How do I choose the right limited_actions value?
-------------------------------------------------

**Option 1: Use query_minimum_actions (recommended)**

.. code-block:: python

    min_actions = sparsifier.query_minimum_actions()
    refined = sparsifier.refine_counterfactuals(limited_actions=min_actions)

**Option 2: Try different values**

.. code-block:: python

    for k in [3, 5, 7]:
        refined = sparsifier.refine_counterfactuals(limited_actions=k)
        # Evaluate results...

**Option 3: Domain knowledge**

Based on what's realistic for your use case (e.g., "users can change at most 3 things").

Which matching policy should I use?
------------------------------------

+----------------+------------------+--------------------+
| Matcher        | When to Use      | Speed              |
+================+==================+====================+
| **ot**         | Best quality     | Slower (O(n³))     |
+----------------+------------------+--------------------+
| **ect**        | Fast, binary     | Fast (O(n))        |
+----------------+------------------+--------------------+
| **nn**         | Prototyping      | Very fast (O(n log n)) |
+----------------+------------------+--------------------+
| **softcem**    | Probabilistic    | Medium             |
+----------------+------------------+--------------------+

**Recommendation:** Start with **ect** for exploration, use **ot** for final results.

.. code-block:: python

    # Quick exploration
    sparsifier.set_policy(matcher="ect", attributor="pshap")

    # Best quality
    sparsifier.set_policy(matcher="ot", attributor="pshap")

Can I restrict which features can be modified?
-----------------------------------------------

Yes! Use ``features_to_vary``:

.. code-block:: python

    # Only these features can change
    refined = sparsifier.refine_counterfactuals(
        limited_actions=5,
        features_to_vary=['Income', 'Duration', 'LoanAmount']
    )

This is useful for:

- Enforcing immutable features (age, gender)
- Focusing on actionable features (income, education)
- Domain-specific constraints

Visualization
=============

How do I save visualizations?
------------------------------

.. code-block:: python

    import os
    os.makedirs('./results', exist_ok=True)

    # Save all visualizations
    sparsifier.heatmap_direction(save_path='./results')
    sparsifier.stacked_bar_chart(save_path='./results')

    # Highlighted DataFrames to HTML
    _, ce_style, ace_style = sparsifier.highlight_changes_final()
    ce_style.to_html('./results/original_cf.html')
    ace_style.to_html('./results/refined_ace.html')

What do the visualization colors mean?
---------------------------------------

**Direction Heatmap:**

- 🟦 Blue = Feature increased
- 🟧 Red/Orange = Feature decreased
- ⬜ White = No change

**Binary Heatmap:**

- ⬛ Black = Feature changed
- ⬜ White = No change

**Highlighted DataFrames:**

- 🟦 Blue background = Value increased
- 🟧 Orange background = Value decreased
- ⬜ White background = No change

How can I customize figure sizes?
----------------------------------

.. code-block:: python

    # For papers (high resolution)
    fig = sparsifier.heatmap_direction(
        figsize=(10, 6),
        dpi=300
    )

    # For presentations (larger)
    fig = sparsifier.stacked_bar_chart(
        figsize=(16, 10),
        dpi=150
    )

Performance
===========

How long does COLA take to run?
--------------------------------

**Typical timings:**

- DiCE generation: 1-30 seconds (depends on data size)
- COLA refinement with ECT: <1 second
- COLA refinement with OT: 1-10 seconds
- Visualization: <1 second

**For 100 instances:**

.. code-block:: python

    # Fast (ECT matcher)
    sparsifier.set_policy(matcher="ect", attributor="pshap")
    # ~2 seconds total

    # Best quality (OT matcher)
    sparsifier.set_policy(matcher="ot", attributor="pshap")
    # ~5 seconds total

How can I speed up COLA?
-------------------------

**Option 1: Use faster matcher**

.. code-block:: python

    # ECT is much faster than OT
    sparsifier.set_policy(matcher="ect", attributor="pshap")

**Option 2: Reduce data size**

.. code-block:: python

    # Process subset
    data = COLAData(factual_data=df.head(100), label_column='target')

**Option 3: Generate fewer CFs per instance**

.. code-block:: python

    # 1 CF per instance (fastest)
    explainer.generate_counterfactuals(total_cfs=1)

Can COLA handle large datasets?
--------------------------------

Yes, but with considerations:

- **<1000 instances**: No problem with any matcher
- **1000-10000 instances**: Use ECT or NN matcher
- **>10000 instances**: Batch processing recommended

.. code-block:: python

    # Batch processing for large datasets
    import pandas as pd

    batch_size = 1000
    all_refined = []

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        # ... process batch ...
        all_refined.append(refined)

    final_refined = pd.concat(all_refined)

Troubleshooting
===============

Error: "Counterfactual data not set"
-------------------------------------

**Cause:** Forgot to add counterfactuals to COLAData.

**Solution:**

.. code-block:: python

    # Must add CFs before creating COLA
    data.add_counterfactuals(cf_df, with_target_column=True)
    sparsifier = COLA(data=data, ml_model=ml_model)  # Now works!

Error: "Must call set_policy before refining"
----------------------------------------------

**Cause:** Trying to refine without setting matching policy.

**Solution:**

.. code-block:: python

    # Always set policy first
    sparsifier.set_policy(matcher="ot", attributor="pshap")
    refined = sparsifier.refine_counterfactuals(limited_actions=5)

Error: "Column mismatch between factual and counterfactual"
------------------------------------------------------------

**Cause:** Counterfactual DataFrame has different columns than factual.

**Solution:**

.. code-block:: python

    # Verify columns match
    print("Factual columns:", data.factual_df.columns.tolist())
    print("CF columns:", cf_df.columns.tolist())

    # Ensure they're the same (order doesn't matter)
    assert set(data.factual_df.columns) == set(cf_df.columns)

Visualizations are blank/all white
-----------------------------------

**Cause:** No features actually changed.

**Solution:**

.. code-block:: python

    # Check if any changes occurred
    factual, ce, ace = sparsifier.get_all_results(limited_actions=5)
    changes = (factual != ace).sum().sum()
    print(f"Total changes: {changes}")

    if changes == 0:
        # Increase limited_actions
        refined = sparsifier.refine_counterfactuals(limited_actions=10)

Best Practices
==============

What's the recommended workflow?
---------------------------------

.. code-block:: python

    # 1. Prepare data with numerical features specified
    data = COLAData(
        factual_data=df,
        label_column='target',
        numerical_features=['feature1', 'feature2']  # Explicit!
    )

    # 2. Use pipeline for model (recommended)
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('prep', preprocessor), ('clf', classifier)])
    pipe.fit(X_train, y_train)
    ml_model = Model(model=pipe, backend="sklearn")

    # 3. Generate multiple CFs per instance
    explainer.generate_counterfactuals(total_cfs=2)

    # 4. Always add CFs before COLA
    data.add_counterfactuals(cf, with_target_column=True)

    # 5. Use OT for best quality
    sparsifier.set_policy(matcher="ot", attributor="pshap", random_state=42)

    # 6. Query minimum actions
    min_actions = sparsifier.query_minimum_actions()

    # 7. Refine
    refined = sparsifier.refine_counterfactuals(limited_actions=min_actions)

    # 8. Visualize
    sparsifier.heatmap_direction(save_path='./results')
    sparsifier.stacked_bar_chart(save_path='./results')

How do I ensure reproducible results?
--------------------------------------

.. code-block:: python

    # Set all random seeds
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)

    # Set random_state in COLA
    sparsifier.set_policy(
        matcher="ot",
        attributor="pshap",
        random_state=42  # Reproducibility!
    )

Should I use a pipeline or separate preprocessing?
---------------------------------------------------

**Recommended: Use Pipeline**

.. code-block:: python

    # ✅ Best practice
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    pipe.fit(X_train, y_train)
    ml_model = Model(model=pipe, backend="sklearn")

**Alternative: PreprocessorWrapper**

.. code-block:: python

    # If you must separate them
    X_train_prep = preprocessor.fit_transform(X_train)
    classifier.fit(X_train_prep, y_train)

    ml_model = PreprocessorWrapper(
        model=classifier,
        backend="sklearn",
        preprocessor=preprocessor
    )

Advanced Topics
===============

Can I implement custom matchers?
---------------------------------

Yes! Inherit from ``BaseCounterfactualMatchingPolicy``:

.. code-block:: python

    from xai_cola.ce_sparsifier.policies.matching import (
        BaseCounterfactualMatchingPolicy
    )

    class MyCustomMatcher(BaseCounterfactualMatchingPolicy):
        def match(self, factual_df, cf_df):
            # Your matching logic
            return matching_dict

Can I use COLA with time-series data?
--------------------------------------

COLA is designed for tabular data. For time-series, you'd need to:

1. Extract tabular features from time-series
2. Use those features with COLA
3. Map refined CFs back to time-series

How does COLA compare to other CF methods?
-------------------------------------------

COLA is a **refinement** method, not a generation method. It works **on top of** existing CF generators to make them more actionable.

**Comparison:**

- **DiCE alone**: Generates diverse CFs (may require many changes)
- **COLA + DiCE**: Refines DiCE's CFs to require fewer changes
- **Result**: Same or similar outcome with 30-70% fewer actions

Getting Help
============

Where can I find more examples?
--------------------------------

- :doc:`tutorials/01_basic_tutorial` - Complete tutorial
- `GitHub examples/ directory <https://github.com/understanding-ml/COLA/tree/main/examples>`_
- :doc:`user_guide/data_interface` - Detailed guides

How do I report bugs?
----------------------

1. Check existing `GitHub Issues <https://github.com/understanding-ml/COLA/issues>`_
2. Open a new issue with:
   - Python version
   - COLA version (``print(xai_cola.__version__)``)
   - Minimal code to reproduce
   - Full error message

How can I contribute?
----------------------

See :doc:`contributing` for contribution guidelines.

Who do I contact for questions?
--------------------------------

- Lin Zhu: s232291@student.dtu.dk
- Lei You: leiyo@dtu.dk
- GitHub Issues: https://github.com/understanding-ml/COLA/issues

Citation
========

If you use COLA in your research, please cite:

.. code-block:: bibtex

    @article{you2024refining,
      title={Refining Counterfactual Explanations With Joint-Distribution-Informed Shapley Towards Actionable Minimality},
      author={You, Lei and Bian, Yijun and Cao, Lele},
      journal={arXiv preprint arXiv:2410.05419},
      year={2024}
    }

See Also
========

- :doc:`installation` - Installation guide
- :doc:`quickstart` - Quick start guide
- :doc:`api/cola` - API reference
- `Paper <https://arxiv.org/pdf/2410.05419>`_ - Research paper
