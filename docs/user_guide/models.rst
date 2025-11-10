==============
Model Interface
==============

Overview
========

The ``Model`` class provides a unified interface for wrapping machine learning models from different frameworks. COLA supports scikit-learn, PyTorch, TensorFlow, and other frameworks through this abstraction layer.

Supported Frameworks
====================

COLA currently supports:

- ✅ **scikit-learn** - All sklearn classifiers
- ✅ **PyTorch** - Neural network models
- ✅ **TensorFlow 1.x** - Legacy TensorFlow models
- ✅ **TensorFlow 2.x / Keras** - Modern TensorFlow/Keras models
- ✅ **Custom models** - Any model with predict methods

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

    # Create pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
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

    # Train classifier on preprocessed data
    X_train_scaled = scaler.fit_transform(X_train)
    classifier.fit(X_train_scaled, y_train)

    # Wrap with PreprocessorWrapper
    ml_model = PreprocessorWrapper(
        model=classifier,
        backend="sklearn",
        preprocessor=scaler
    )

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

Scenario 3: TensorFlow/Keras Model
-----------------------------------

For TensorFlow 2.x / Keras models:

.. code-block:: python

    import tensorflow as tf
    from xai_cola.ce_sparsifier.models import Model

    # Define Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # Compile and train
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(X_train, y_train, epochs=10)

    # Wrap for COLA
    ml_model = Model(model=model, backend="TF2")

Scenario 4: Separate Preprocessor
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
| TensorFlow 2.x | ``"TF2"``               | tensorflow >= 2.0         |
+----------------+-------------------------+---------------------------+
| TensorFlow 1.x | ``"TF1"``               | tensorflow < 2.0          |
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

**For TensorFlow:**

.. code-block:: python

    model = tf.keras.Sequential([...])
    # Must be compiled
    model.compile(...)

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

Example 2: PyTorch with Custom Preprocessing
---------------------------------------------

.. code-block:: python

    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from xai_cola.ce_sparsifier.utils import PreprocessorWrapper

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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train PyTorch model
    model = Classifier(input_dim=X_train_scaled.shape[1])
    # ... training loop ...

    # Wrap with preprocessor
    ml_model = PreprocessorWrapper(
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
    classifier.fit(scaler.fit_transform(X_train), y_train)
    ml_model = Model(model=classifier, backend="sklearn")
    ml_model.predict(X_test)  # Error! X_test is not scaled

    # ✅ Correct - wrap with preprocessor
    ml_model = PreprocessorWrapper(
        model=classifier,
        backend="sklearn",
        preprocessor=scaler
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

    # ❌ Wrong - TensorFlow model with sklearn backend
    keras_model = tf.keras.Sequential([...])
    ml_model = Model(model=keras_model, backend="sklearn")

    # ✅ Correct - use TF2 backend
    ml_model = Model(model=keras_model, backend="TF2")

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
       clf.fit(scaler.transform(X_train), y_train)  # Trained on scaled
       ml_model = Model(model=clf, backend="sklearn")
       ml_model.predict(X_test)  # Passed raw data - mismatch!

2. **Don't forget to compile TensorFlow models**

   .. code-block:: python

       # ❌ Bad
       model = tf.keras.Sequential([...])
       ml_model = Model(model=model, backend="TF2")  # Not compiled!

       # ✅ Good
       model.compile(optimizer='adam', loss='...')
       ml_model = Model(model=model, backend="TF2")

3. **Don't change model after wrapping**

API Reference
=============

For complete parameter details, see:

- :class:`~xai_cola.ce_sparsifier.models.Model`
- :class:`~xai_cola.ce_sparsifier.utils.PreprocessorWrapper`

Next Steps
==========

- Learn about :doc:`explainers` - Generating counterfactuals
- Explore :doc:`matching_policies` - Configuring COLA refinement
- See :doc:`data_interface` - Managing your data
