==============
Model Interface
==============

Overview
========

The ``Model`` class provides a unified interface for wrapping machine learning models from different frameworks. COLA supports scikit-learn and PyTorch through this abstraction layer.

Supported Frameworks
====================

COLA currently supports:

- ✅ **scikit-learn** - All sklearn classifiers
- ✅ **PyTorch** - Neural network models

Key Concepts
============

Pipeline vs Separate Components
--------------------------------

You can provide your model to COLA in two ways:

**Option 1: As a Pipeline** (Recommended)

Combine preprocessor and classifier into a single sklearn pipeline:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression


    # Step 1: Create the preprocessor
    # Linear models require feature scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),  # Scale numerical features
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Step 2: Create the Logistic Regression classifier
    lr_classifier = LogisticRegression(
        max_iter=1000,
        C=1.0,  # Inverse of regularization strength
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        solver='lbfgs'  # Suitable for small datasets
    )

    # Step 3: Create the Pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lr_classifier)
    ])

    # Train on raw data
    pipe.fit(X_train, y_train)

    # Use with COLA
    from xai_cola.ce_sparsifier.models import Model
    ml_model = Model(model=pipe, backend="sklearn")

**Option 2: Separate Components**

If your preprocessor and classifier are separate:

.. code-block:: python

    from xai_cola.ce_sparsifier.utils import PreprocessorWrapper

    # Train lr_classifier on preprocessed data
    X_train_scaled = preprocessor.fit_transform(X_train)
    lr_classifier.fit(X_train_scaled, y_train)

    # input classifier and preprocessor separately
    ml_model = Model(model=lr_classifier, backend="sklearn", preprocessor=preprocessor)

.. tip::
    Use **Option 1** (pipeline) whenever possible - it's cleaner and less error-prone.

Basic Usage
===========

Scenario 1: Scikit-learn Model
-------------------------------

The most common case:

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from xai_cola.ce_sparsifier.models import Model

    # Create and train pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100))
    ])
    pipe.fit(X_train, y_train)

    # Wrap for COLA
    ml_model = Model(model=pipe, backend="sklearn")

    # Test prediction
    predictions = ml_model.predict(X_test)
    probabilities = ml_model.predict_proba(X_test)

Scenario 2: PyTorch Model
--------------------------

For neural networks built with PyTorch:

.. code-block:: python

    import torch
    import torch.nn as nn
    from xai_cola.ce_sparsifier.models import Model

    # Define your PyTorch model
    class NeuralNet(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    # Train your model
    model = NeuralNet(input_size=10)
    # ... training code ...

    # Wrap for COLA
    ml_model = Model(model=model, backend="pytorch")

.. note::
    Your PyTorch model should output raw logits or probabilities. COLA will handle the conversion.


Scenario 3: Separate Preprocessor
----------------------------------

When preprocessor and classifier are not combined:

.. code-block:: python

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from xai_cola.ce_sparsifier.utils import PreprocessorWrapper

    # Fit preprocessor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train classifier on preprocessed data
    classifier = GradientBoostingClassifier()
    classifier.fit(X_train_scaled, y_train)

    # Wrap both together
    ml_model = PreprocessorWrapper(
        model=classifier,
        backend="sklearn",
        preprocessor=scaler
    )

    # COLA will automatically handle preprocessing
    # when generating counterfactuals

.. warning::
    Make sure your classifier was trained on **preprocessed** data when using this approach.

Advanced Usage
==============

Custom Backends
---------------

COLA uses backend specifications to handle different frameworks:

+----------------+-------------------------+---------------------------+
| Framework      | Backend String          | Requirements              |
+================+=========================+===========================+
| scikit-learn   | ``"sklearn"``           | sklearn installed         |
+----------------+-------------------------+---------------------------+
| PyTorch        | ``"pytorch"``           | torch installed           |
+----------------+-------------------------+---------------------------+


Model Requirements
------------------

Your model must support:

**For Classification:**

- ``predict(X)`` - returns class labels
- ``predict_proba(X)`` - returns probability distributions

**For PyTorch:**

.. code-block:: python

    class MyModel(nn.Module):
        def forward(self, x):
            # Should return logits or probabilities
            return output

Working with Different Data Types
----------------------------------

The Model interface handles data conversions automatically:

.. code-block:: python

    import pandas as pd
    import numpy as np

    # Works with pandas DataFrame
    pred1 = ml_model.predict(df)

    # Works with numpy array
    pred2 = ml_model.predict(np.array([[1, 2, 3]]))

    # Works with torch tensors (for PyTorch models)
    pred3 = ml_model.predict(torch.tensor([[1, 2, 3]]))

Complete Examples
=================

Example 1: End-to-End with Pipeline
------------------------------------

.. code-block:: python

    from xai_cola.datasets.german_credit import GermanCreditDataset
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from xai_cola.ce_sparsifier.models import Model

    # Load data
    dataset = GermanCreditDataset()
    X_train, y_train, X_test, y_test = dataset.get_original_train_test_split()

    # Define features
    numerical = ['Age', 'Credit amount', 'Duration']
    categorical = ['Sex', 'Job', 'Housing']

    # Create preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(drop='first'), categorical)
    ])

    # Create pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Wrap for COLA
    ml_model = Model(model=pipe, backend="sklearn")

    # Use with COLA
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola import COLA

    data = COLAData(factual_data=X_test, label_column='Risk')
    # ... generate counterfactuals ...
    sparsifier = COLA(data=data, ml_model=ml_model)

Example 2: PyTorch with Preprocessor
---------------------------------------------

.. code-block:: python

    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from xai_cola.ce_sparsifier import Model

    # Define PyTorch model
    class Classifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )

        def forward(self, x):
            return self.network(x)

    # Prepare data
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(drop='first'), categorical)
    ])
    X_train_scaled = preprocessor.fit_transform(X_train)

    # Train PyTorch model
    model = Classifier(input_dim=X_train_scaled.shape[1])
    # ... training loop ...

    # Wrap with preprocessor
    ml_model = Model(
        model=model,
        backend="pytorch",
        preprocessor=scaler
    )

    # Now COLA will:
    # 1. Scale input with scaler
    # 2. Convert to torch tensor
    # 3. Run through model
    # 4. Convert output back

Common Issues
=============

Issue 1: Preprocessing Mismatch
--------------------------------

**Error:**

.. code-block:: text

    ValueError: Input contains NaN, infinity or a value too large

**Cause:** Classifier was trained on preprocessed data but you're passing raw data.

**Solution:**

Use ``PreprocessorWrapper`` or a pipeline:

.. code-block:: python

    # ❌ Wrong - passing raw data to model trained on scaled data
    classifier.fit(preprocessor.fit_transform(X_train), y_train)
    ml_model = Model(model=classifier, backend="sklearn")
    ml_model.predict(X_test)  # Error! X_test is not scaled

    # ✅ Correct - input with preprocessor
    ml_model = Model(
        model=classifier,
        backend="sklearn",
        preprocessor=preprocessor
    )
    ml_model.predict(X_test)  # Automatically scales X_test

Issue 2: Wrong Backend
-----------------------

**Error:**

.. code-block:: text

    AttributeError: 'Sequential' object has no attribute 'predict_proba'

**Cause:** Using wrong backend for your model.

**Solution:**

Match backend to framework:

.. code-block:: python

    # ❌ Wrong - Pytorch model with sklearn backend
    keras_model = tf.keras.Sequential([...])
    ml_model = Model(model=keras_model, backend="sklearn")

    # ✅ Correct - use pytorch backend
    ml_model = Model(model=keras_model, backend="pytorch")

Issue 3: Model Not Trained
---------------------------

**Error:**

.. code-block:: text

    NotFittedError: This model is not fitted yet

**Cause:** Trying to use model before training.

**Solution:**

Always train before wrapping:

.. code-block:: python

    # ❌ Wrong - wrapping before training
    pipe = Pipeline([...])
    ml_model = Model(model=pipe, backend="sklearn")
    ml_model.predict(X_test)  # Error!

    # ✅ Correct - train first
    pipe.fit(X_train, y_train)
    ml_model = Model(model=pipe, backend="sklearn")
    ml_model.predict(X_test)  # Works!

Best Practices
==============

✅ **DO:**

1. **Use pipelines when possible**

   .. code-block:: python

       pipe = Pipeline([('prep', preprocessor), ('clf', classifier)])
       pipe.fit(X_train, y_train)
       ml_model = Model(model=pipe, backend="sklearn")

2. **Verify model works before wrapping**

   .. code-block:: python

       # Test directly first
       predictions = pipe.predict(X_test)
       print(f"Accuracy: {accuracy_score(y_test, predictions)}")

       # Then wrap
       ml_model = Model(model=pipe, backend="sklearn")

3. **Match backend to framework**

4. **Keep preprocessing consistent**

❌ **DON'T:**

1. **Don't mix preprocessed and raw data**

   .. code-block:: python

       # ❌ Bad
       clf.fit(preprocessor.transform(X_train), y_train)  # Trained on scaled
       ml_model = Model(model=clf, backend="sklearn")
       ml_model.predict(X_test)  # Passed raw data - mismatch!


API Reference
=============

For complete parameter details, see:

- :class:`~xai_cola.ce_sparsifier.models.Model`

Next Steps
==========

- Learn about :doc:`explainers` - Generating counterfactuals
- Explore :doc:`matching_policies` - Configuring COLA refinement
- See :doc:`data_interface` - Managing your data
