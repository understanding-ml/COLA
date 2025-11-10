==============
Models API
==============

.. currentmodule:: xai_cola.ce_sparsifier.models

Model Class
===========

.. autoclass:: Model
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~Model.__init__
      ~Model.predict
      ~Model.predict_proba

Constructor
-----------

.. automethod:: Model.__init__

Prediction Methods
------------------

.. automethod:: Model.predict

.. automethod:: Model.predict_proba

PreprocessorWrapper
===================

.. currentmodule:: xai_cola.ce_sparsifier.utils

.. autoclass:: PreprocessorWrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Wrapper for models that need separate preprocessing.

   This class combines a preprocessor and a model into a single interface.
   Useful when your classifier was trained on preprocessed data.

Constructor
-----------

.. automethod:: PreprocessorWrapper.__init__

Methods
-------

.. automethod:: PreprocessorWrapper.predict

.. automethod:: PreprocessorWrapper.predict_proba

Supported Backends
==================

COLA supports the following machine learning frameworks:

Scikit-learn
------------

**Backend string:** ``"sklearn"``

**Requirements:**

- scikit-learn installed
- Model must have ``predict()`` and ``predict_proba()`` methods

**Example:**

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xai_cola.ce_sparsifier.models import Model

    # Create pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier())
    ])
    pipe.fit(X_train, y_train)

    # Wrap for COLA
    ml_model = Model(model=pipe, backend="sklearn")

PyTorch
-------

**Backend string:** ``"pytorch"``

**Requirements:**

- PyTorch installed
- Model must be a ``torch.nn.Module``
- Model should output logits or probabilities

**Example:**

.. code-block:: python

    import torch
    import torch.nn as nn
    from xai_cola.ce_sparsifier.models import Model

    class NeuralNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )

        def forward(self, x):
            return self.layers(x)

    # Train model
    model = NeuralNet(input_dim=10)
    # ... training code ...

    # Wrap for COLA
    ml_model = Model(model=model, backend="pytorch")

TensorFlow 2.x / Keras
----------------------

**Backend string:** ``"TF2"``

**Requirements:**

- TensorFlow >= 2.0 installed
- Model must be compiled

**Example:**

.. code-block:: python

    import tensorflow as tf
    from xai_cola.ce_sparsifier.models import Model

    # Define Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy'
    )
    model.fit(X_train, y_train)

    # Wrap for COLA
    ml_model = Model(model=model, backend="TF2")

TensorFlow 1.x
--------------

**Backend string:** ``"TF1"``

**Requirements:**

- TensorFlow < 2.0 installed
- Session-based TensorFlow model

**Example:**

.. code-block:: python

    import tensorflow as tf
    from xai_cola.ce_sparsifier.models import Model

    # TensorFlow 1.x style
    # ... define graph ...

    # Wrap for COLA
    ml_model = Model(model=tf_model, backend="TF1")

Examples
========

Pipeline Model (Recommended)
-----------------------------

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from xai_cola.ce_sparsifier.models import Model

    # Create pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier())
    ])

    # Train on raw data
    pipe.fit(X_train, y_train)

    # Wrap
    ml_model = Model(model=pipe, backend="sklearn")

    # Automatically handles preprocessing
    predictions = ml_model.predict(X_test)

Separate Preprocessor and Classifier
-------------------------------------

.. code-block:: python

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from xai_cola.ce_sparsifier.utils import PreprocessorWrapper

    # Fit preprocessor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train classifier on scaled data
    clf = LogisticRegression()
    clf.fit(X_train_scaled, y_train)

    # Wrap both
    ml_model = PreprocessorWrapper(
        model=clf,
        backend="sklearn",
        preprocessor=scaler
    )

    # Automatically scales input before prediction
    predictions = ml_model.predict(X_test)  # X_test is raw

Custom Preprocessor
-------------------

.. code-block:: python

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    # Define custom preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Age', 'Income']),
        ('cat', OneHotEncoder(drop='first'), ['Gender', 'Job'])
    ])
    preprocessor.fit(X_train)

    # Train classifier
    X_train_processed = preprocessor.transform(X_train)
    clf.fit(X_train_processed, y_train)

    # Wrap
    ml_model = PreprocessorWrapper(
        model=clf,
        backend="sklearn",
        preprocessor=preprocessor
    )

Multiple Frameworks
-------------------

.. code-block:: python

    # Scikit-learn
    sklearn_model = Model(model=sklearn_pipe, backend="sklearn")

    # PyTorch
    pytorch_model = Model(model=torch_model, backend="pytorch")

    # TensorFlow
    tf_model = Model(model=keras_model, backend="TF2")

    # All work with COLA the same way!
    from xai_cola import COLA

    cola_sklearn = COLA(data=data, ml_model=sklearn_model)
    cola_pytorch = COLA(data=data, ml_model=pytorch_model)
    cola_tf = COLA(data=data, ml_model=tf_model)

See Also
========

- :doc:`../user_guide/models` - Detailed model usage guide
- :doc:`data` - Data interface
- :doc:`cola` - COLA main class
