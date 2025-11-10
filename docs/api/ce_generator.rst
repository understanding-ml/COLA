==========================
Counterfactual Generators
==========================

.. currentmodule:: xai_cola.ce_generator

DiCE Explainer
==============

.. autoclass:: DiCE
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Diverse Counterfactual Explanations (DiCE) generator.

   Generates multiple diverse counterfactuals for each factual instance.
   Based on the DiCE paper: https://arxiv.org/abs/1905.07697

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~DiCE.__init__
      ~DiCE.generate_counterfactuals

Constructor
-----------

.. automethod:: DiCE.__init__

Generation Methods
------------------

.. automethod:: DiCE.generate_counterfactuals

DisCount Explainer
==================

.. autoclass:: DisCount
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Distribution-aware Counterfactual (DisCount) generator.

   Generates distributional counterfactuals that maintain similar structure
   to the factual distribution.

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~DisCount.__init__
      ~DisCount.generate_counterfactuals

Constructor
-----------

.. automethod:: DisCount.__init__

Generation Methods
------------------

.. automethod:: DisCount.generate_counterfactuals

Base Explainer
==============

.. autoclass:: BaseExplainer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Base class for counterfactual explainers.

   Use this as a template if you want to implement custom explainers
   compatible with COLA.

Examples
========

Using DiCE
----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_generator import DiCE
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # Setup
    data = COLAData(factual_data=df, label_column='Risk')
    ml_model = Model(model=trained_model, backend="sklearn")

    # Create DiCE explainer
    explainer = DiCE(ml_model=ml_model)

    # Generate counterfactuals
    factual, counterfactual = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        continuous_features=['Age', 'Income']
    )

    # Add to data
    data.add_counterfactuals(counterfactual, with_target_column=True)

With Immutable Features
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Don't change Age or Gender
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=3,
        features_to_keep=['Age', 'Gender'],
        continuous_features=['Income', 'Duration']
    )

With Feature Ranges
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Set realistic bounds
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        continuous_features=['Age', 'Income'],
        permitted_range={
            'Age': [18, 65],
            'Income': [10000, 200000]
        }
    )

Only Vary Specific Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Only financial features can change
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        features_to_vary=['Income', 'LoanAmount', 'Duration'],
        continuous_features=['Income', 'LoanAmount', 'Duration']
    )

Using DisCount
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_generator import DisCount

    # Create DisCount explainer
    explainer = DisCount(ml_model=ml_model)

    # Generate distributional counterfactuals
    factual, counterfactual = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        cost_type='L1',
        continuous_features=['Age', 'Income']
    )

    # Add to data
    data.add_counterfactuals(counterfactual, with_target_column=True)

With Different Cost Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # L1 cost (Manhattan distance)
    factual, cf1 = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        cost_type='L1',
        continuous_features=['Age', 'Income']
    )

    # L2 cost (Euclidean distance)
    factual, cf2 = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        cost_type='L2',
        continuous_features=['Age', 'Income']
    )

Integration with COLA
---------------------

Complete Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola import COLA
    from xai_cola.ce_generator import DiCE
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # 1. Setup
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Income']
    )
    ml_model = Model(model=trained_model, backend="sklearn")

    # 2. Generate CFs with DiCE
    explainer = DiCE(ml_model=ml_model)
    _, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        continuous_features=['Age', 'Income']
    )

    # 3. Add to data
    data.add_counterfactuals(cf, with_target_column=True)

    # 4. Refine with COLA
    sparsifier = COLA(data=data, ml_model=ml_model)
    sparsifier.set_policy(matcher='ot', attributor='pshap')
    refined = sparsifier.refine_counterfactuals(limited_actions=5)

    # 5. Visualize
    sparsifier.heatmap_direction(save_path='./results')

Using Custom Explainers
-----------------------

External Libraries
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Example: Using Alibi
    from alibi.explainers import CounterfactualProto
    import pandas as pd

    # Create Alibi explainer
    cf_explainer = CounterfactualProto(
        predict_fn=ml_model.predict_proba,
        shape=(1, n_features)
    )
    cf_explainer.fit(X_train)

    # Generate counterfactuals
    explanation = cf_explainer.explain(X_test)
    cf_array = explanation.cf['X']

    # Convert to DataFrame
    cf_df = pd.DataFrame(cf_array, columns=feature_names)

    # Use with COLA
    data = COLAData(factual_data=X_test, label_column='target')
    data.add_counterfactuals(cf_df, with_target_column=False)

    sparsifier = COLA(data=data, ml_model=ml_model)
    # ... continue with COLA refinement ...

Custom Implementation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_generator import BaseExplainer

    class MyCustomExplainer(BaseExplainer):
        """Custom counterfactual explainer."""

        def __init__(self, ml_model, **kwargs):
            super().__init__(ml_model)
            self.kwargs = kwargs

        def generate_counterfactuals(
            self,
            data,
            factual_class,
            **params
        ):
            """Generate counterfactuals using custom logic."""

            # Your counterfactual generation logic here
            factual_df = data.factual_df
            # ... generate cf_df ...

            return factual_df, cf_df

    # Use it
    explainer = MyCustomExplainer(ml_model=ml_model)
    factual, cf = explainer.generate_counterfactuals(data=data, factual_class=1)

Parameters Reference
====================

Common Parameters
-----------------

**data** : COLAData
    Data container with factual instances.

**factual_class** : int or list
    Class label(s) to generate counterfactuals for.

**continuous_features** : list of str
    Names of continuous/numerical features.

**features_to_keep** : list of str, optional
    Features that should not be changed (immutable).

**features_to_vary** : list of str, optional
    Only these features can be changed.

DiCE Parameters
---------------

**total_cfs** : int
    Number of counterfactuals to generate per instance.

**permitted_range** : dict, optional
    Allowed ranges for features. Format: ``{'feature': [min, max]}``

**diversity_weight** : float, optional
    Weight for diversity in generated CFs.

**proximity_weight** : float, optional
    Weight for proximity to factual instance.

DisCount Parameters
-------------------

**cost_type** : str
    Distance metric. Options: ``'L1'``, ``'L2'``

**max_iterations** : int, optional
    Maximum optimization iterations.

**tolerance** : float, optional
    Convergence tolerance.

See Also
========

- :doc:`../user_guide/explainers` - Detailed explainer guide
- :doc:`cola` - COLA refinement
- :doc:`data` - Data interface
- :doc:`models` - Model interface
